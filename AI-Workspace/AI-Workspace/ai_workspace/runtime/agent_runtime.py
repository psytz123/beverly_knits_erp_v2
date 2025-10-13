"""
Agent Runtime - Process Isolation & Execution Engine.

Provides sandboxed execution environment for agents with resource limits,
process pooling, and fault isolation.

Features:
    - Process pooling for fast startup (<500ms target)
    - Cross-platform resource enforcement
    - Graceful shutdown with cleanup
    - Fault isolation (agent crashes don't affect orchestrator)
    - Comprehensive logging and monitoring

Example:
    >>> runtime = AgentRuntime()
    >>> await runtime.start()
    >>> result = runtime.execute(
    ...     agent_name="python-pro",
    ...     task=Task(action="write_function", spec="Fibonacci calculator")
    ... )
    >>> print(result.exit_code)
    0
    >>> runtime.stop()

Architecture:
    - AgentProcessPool: Pre-spawns workers to reduce latency
    - ResourceEnforcer: Platform-aware resource management
    - AgentRuntime: Main execution orchestrator

Process Lifecycle:
    1. Get worker from pool (or spawn new)
    2. Apply resource limits (memory, CPU, timeout)
    3. Execute agent task
    4. Monitor resource usage
    5. Collect results and cleanup
    6. Return worker to pool (if reusable)
"""

import asyncio
import atexit
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .interfaces import (
    AgentResult,
    ResourceLimits,
    Task,
    IAgentRuntime,
)
from .resource_limits import ResourceEnforcer
from .config import RuntimeConfig
from .worker_process import worker_main, WorkerTask, WorkerResult
from .ipc_protocol import IPCChannel, IPCMessage

logger = logging.getLogger(__name__)


class AgentNotFoundError(Exception):
    """Raised when requested agent doesn't exist."""
    pass


class ResourceLimitError(Exception):
    """Raised when agent exceeds resource limits."""
    pass


class ExecutionError(Exception):
    """Raised when agent execution fails."""
    pass


class ProcessResult:
    """Internal result from process execution."""

    def __init__(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        artifacts: Optional[List[str]] = None
    ):
        """
        Initialize process result.

        Args:
            exit_code: Process exit code
            stdout: Standard output
            stderr: Standard error
            memory_mb: Peak memory usage in MB
            cpu_percent: Average CPU usage percentage
            artifacts: List of created/modified files
        """
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.memory_mb = memory_mb
        self.cpu_percent = cpu_percent
        self.artifacts = artifacts or []


class PoolWorker:
    """Represents a worker process in the pool."""

    def __init__(
        self,
        worker_id: int,
        process: mp.Process,
        work_queue: mp.Queue,
        result_queue: mp.Queue
    ):
        """
        Initialize pool worker.

        Args:
            worker_id: Unique worker identifier
            process: Worker process instance
            work_queue: Queue for sending tasks
            result_queue: Queue for receiving results
        """
        self.worker_id = worker_id
        self.process = process
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.task_count = 0
        self.created_at = time.time()
        self.last_used = time.time()

    def is_alive(self) -> bool:
        """Check if worker process is still running."""
        return self.process.is_alive()

    def terminate(self) -> None:
        """Gracefully terminate worker process."""
        if self.process.is_alive():
            # Send shutdown command
            try:
                self.work_queue.put({"command": "SHUTDOWN"}, timeout=1)
            except:
                pass

            # Wait for graceful exit
            self.process.join(timeout=2)

            # Force kill if still alive
            if self.process.is_alive():
                logger.warning(f"Force terminating worker {self.worker_id}")
                self.process.terminate()
                self.process.join(timeout=1)

                if self.process.is_alive():
                    self.process.kill()


class AgentProcessPool:
    """
    Pre-spawn agent processes to reduce startup latency.

    Maintains a pool of warm worker processes ready to execute tasks.
    Target: Reduce startup from 5s → <500ms by having processes pre-initialized.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        workspace_path: Optional[Path] = None
    ):
        """
        Initialize pool with target size.

        Args:
            pool_size: Number of workers to maintain in pool
            workspace_path: Path to .ai-workspace directory
        """
        self.config = config
        self.pool_size = config.pool_size
        self.workspace_path = workspace_path
        self.workers: List[PoolWorker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue(maxsize=config.pool_size)
        self._maintain_pool_task: Optional[asyncio.Task] = None
        self._running = False
        self._next_worker_id = 0
        self._lock = asyncio.Lock()

        logger.info(f"AgentProcessPool initialized with target size: {config.pool_size}")

    async def start(self) -> None:
        """
        Start maintaining the pool.

        Spawns background task to keep pool filled with warm workers.
        """
        if self._running:
            logger.warning("Pool already running")
            return

        self._running = True

        # Spawn initial workers
        logger.info(f"Pre-spawning {self.pool_size} workers...")
        for i in range(self.pool_size):
            worker = await self._spawn_worker()
            if worker:
                self.workers.append(worker)
                await self.available_workers.put(worker)

        # Start maintenance task
        self._maintain_pool_task = asyncio.create_task(self._maintain_pool())

        logger.info(f"AgentProcessPool started with {len(self.workers)} workers")

    async def _maintain_pool(self) -> None:
        """
        Keep pool size constant by respawning workers.

        Runs continuously in background, spawning new workers when pool
        size drops below target.
        """
        while self._running:
            try:
                # Check for dead workers
                async with self._lock:
                    dead_workers = [w for w in self.workers if not w.is_alive()]

                    for worker in dead_workers:
                        logger.warning(f"Worker {worker.worker_id} died, removing from pool")
                        self.workers.remove(worker)

                        # Spawn replacement
                        new_worker = await self._spawn_worker()
                        if new_worker:
                            self.workers.append(new_worker)
                            await self.available_workers.put(new_worker)

                # Check if we need more workers
                alive_count = len([w for w in self.workers if w.is_alive()])
                if alive_count < self.pool_size:
                    logger.info(f"Pool under capacity ({alive_count}/{self.pool_size}), spawning workers...")
                    needed = self.pool_size - alive_count

                    for _ in range(needed):
                        worker = await self._spawn_worker()
                        if worker:
                            async with self._lock:
                                self.workers.append(worker)
                            await self.available_workers.put(worker)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error maintaining pool: {e}")
                await asyncio.sleep(5)

    async def _spawn_worker(self) -> Optional[PoolWorker]:
        """
        Spawn a new worker process.

        Returns:
            PoolWorker instance, or None if spawn failed
        """
        worker_id = self._next_worker_id
        self._next_worker_id += 1

        try:
            # Create communication queues
            work_queue = mp.Queue()
            result_queue = mp.Queue()

            # Spawn worker process
            workspace_str = str(self.workspace_path) if self.workspace_path else None
            process = mp.Process(
                target=worker_main,
                args=(work_queue, result_queue, workspace_str),
                name=f"Worker-{worker_id}",
                daemon=True
            )
            process.start()

            # Wait a bit to ensure worker started successfully
            await asyncio.sleep(0.1)

            if not process.is_alive():
                logger.error(f"Worker {worker_id} failed to start")
                return None

            worker = PoolWorker(worker_id, process, work_queue, result_queue)
            logger.info(f"Spawned worker {worker_id} (PID: {process.pid})")

            return worker

        except Exception as e:
            logger.error(f"Failed to spawn worker {worker_id}: {e}")
            return None

    async def get_worker(self, timeout: float = 5.0) -> Optional[PoolWorker]:
        """
        Get worker from pool (blocks if pool empty).

        Args:
            timeout: Maximum time to wait for worker (seconds)

        Returns:
            PoolWorker from pool, or None if timeout

        Raises:
            asyncio.TimeoutError: If no worker available within timeout
        """
        try:
            worker = await asyncio.wait_for(
                self.available_workers.get(),
                timeout=timeout
            )

            # Verify worker is still alive
            if not worker.is_alive():
                logger.warning(f"Retrieved dead worker {worker.worker_id}, getting another")
                return await self.get_worker(timeout)

            logger.debug(f"Retrieved worker {worker.worker_id} from pool")
            return worker

        except asyncio.TimeoutError:
            logger.debug("No worker available in pool, will spawn new process")
            return None

    async def return_worker(self, worker: PoolWorker) -> None:
        """
        Return worker to pool.

        Args:
            worker: Worker to return to pool
        """
        if worker.is_alive() and self.available_workers.qsize() < self.pool_size:
            worker.last_used = time.time()
            await self.available_workers.put(worker)
            logger.debug(f"Returned worker {worker.worker_id} to pool")
        else:
            # Worker dead or pool full - terminate it
            if worker.is_alive():
                worker.terminate()
            async with self._lock:
                if worker in self.workers:
                    self.workers.remove(worker)
            logger.debug(f"Worker {worker.worker_id} not returned to pool (dead or pool full)")

    async def execute_task(
        self,
        worker: PoolWorker,
        agent_path: Path,
        task: Task,
        timeout: float
    ) -> ProcessResult:
        """
        Execute task using a worker from the pool.

        Args:
            worker: Worker to use for execution
            agent_path: Path to agent markdown file
            task: Task to execute
            timeout: Execution timeout in seconds

        Returns:
            ProcessResult with execution details
        """
        start_time = time.time()

        # Create worker task
        worker_task = WorkerTask(
            task_id=f"task-{worker.worker_id}-{worker.task_count}",
            agent_name=agent_path.stem,
            action=task.action,
            spec=str(task.spec),
            timeout_seconds=int(timeout)
        )

        try:
            # Send task to worker using run_in_executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: worker.work_queue.put(worker_task.to_dict(), timeout=2)
            )
            worker.task_count += 1

            # Create IPC channel for result collection
            channel = IPCChannel(worker.work_queue, worker.result_queue)

            # Collect all results with timeout
            results = await channel.collect_all(timeout=timeout)

            # Parse results
            duration = time.time() - start_time

            # Extract stdout from logs
            stdout_lines = [log.get("message", "") for log in results.get("logs", [])]
            stdout = "\n".join(stdout_lines)

            # Extract stderr from errors
            stderr_lines = [err.get("message", "") for err in results.get("errors", [])]
            stderr = "\n".join(stderr_lines)

            # Extract metrics
            metrics = results.get("metrics", [])
            memory_mb = 0.0
            cpu_percent = 0.0

            for metric in metrics:
                if metric.get("metric") == "memory_peak_mb":
                    memory_mb = metric.get("value", 0.0)
                elif metric.get("metric") == "cpu_avg_percent":
                    cpu_percent = metric.get("value", 0.0)

            # Extract artifacts
            artifacts_data = results.get("artifacts", [])
            artifacts = [a.get("content", "") for a in artifacts_data]

            # Determine exit code from results
            exit_code = 0
            if results.get("results"):
                result_data = results["results"][0]
                if not result_data.get("success", True):
                    exit_code = 1

            return ProcessResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                artifacts=artifacts
            )

        except asyncio.TimeoutError:
            logger.error(f"Task execution timed out after {timeout}s")
            duration = time.time() - start_time

            return ProcessResult(
                exit_code=124,  # Timeout exit code
                stdout="",
                stderr=f"Task execution exceeded timeout of {timeout}s",
                memory_mb=0.0,
                cpu_percent=0.0,
                artifacts=[]
            )

        except Exception as e:
            logger.error(f"Error executing task on worker {worker.worker_id}: {e}")
            duration = time.time() - start_time

            return ProcessResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                memory_mb=0.0,
                cpu_percent=0.0,
                artifacts=[]
            )

    async def stop(self) -> None:
        """Stop pool and terminate all workers."""
        self._running = False

        if self._maintain_pool_task:
            self._maintain_pool_task.cancel()
            try:
                await self._maintain_pool_task
            except asyncio.CancelledError:
                pass

        # Terminate all workers
        logger.info(f"Terminating {len(self.workers)} workers...")
        async with self._lock:
            for worker in self.workers:
                worker.terminate()
            self.workers.clear()

        logger.info("AgentProcessPool stopped")


class AgentRuntime(IAgentRuntime):
    """
    Execute agents in isolated processes with resource management.

    Provides complete agent lifecycle management:
    - Discovery and validation
    - Process isolation
    - Resource enforcement
    - Result collection
    - Cleanup and monitoring

    Example:
        >>> runtime = AgentRuntime()
        >>> await runtime.start()
        >>>
        >>> task = Task(action="write_code", spec="Create Fibonacci function")
        >>> result = runtime.execute("python-pro", task)
        >>>
        >>> print(f"Success: {result.success}")
        >>> print(f"Output: {result.output}")
        >>>
        >>> runtime.stop()
    """

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None
    ):
        """
        Initialize agent runtime.

        Args:
            workspace_path: Path to .ai-workspace directory
                          (defaults to E:\\agents\\.ai-workspace)
        """
        if workspace_path is None:
            workspace_path = Path(__file__).parent.parent.parent / ".ai-workspace"

        if config is None:
            config = RuntimeConfig()
            logger.info("Using default RuntimeConfig")
        else:
            logger.info(f"Using provided RuntimeConfig: {config}")

        self.workspace = workspace_path
        self.agents_dir = workspace_path / "agents"
        self.config = config
        # Use regular set instead of WeakSet for thread safety
        self.active_processes: Set[mp.Process] = set()
        self._active_processes_lock = asyncio.Lock()
        self.process_pool = AgentProcessPool(config=config, workspace_path=workspace_path)
        self.resource_enforcer = ResourceEnforcer()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Register cleanup on exit
        atexit.register(self._kill_all_agents)

        logger.info(
            f"AgentRuntime initialized with workspace: {self.workspace}"
        )

    def start(self) -> None:
        """
        Start the runtime (initialize process pool).

        Must be called before executing agents.
        """
        # Get or create event loop
        try:
            self._event_loop = asyncio.get_running_loop()
            logger.debug("Using existing event loop")
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            logger.debug("Created new event loop")

        # Start process pool
        self._event_loop.run_until_complete(self.process_pool.start())
        logger.info("AgentRuntime started")

    def execute(
        self,
        agent_name: str,
        task: Task,
        limits: Optional[ResourceLimits] = None
    ) -> AgentResult:
        """
        Execute agent in isolated process.

        Args:
            agent_name: Name of agent to execute (e.g., "python-pro")
            task: Task specification
            limits: Resource constraints (optional, uses defaults if not provided)

        Returns:
            AgentResult with execution status and outputs

        Raises:
            AgentNotFoundError: If agent doesn't exist
            ResourceLimitError: If agent exceeds resource limits
            ExecutionError: If execution fails
        """
        if limits is None:
            limits = self._get_default_limits()

        # Ensure event loop exists
        if self._event_loop is None:
            self.start()

        # Run async execution in event loop
        return self._event_loop.run_until_complete(
            self._execute_async(agent_name, task, limits)
        )

    async def _execute_async(
        self,
        agent_name: str,
        task: Task,
        limits: ResourceLimits
    ) -> AgentResult:
        """
        Async execution implementation.

        Args:
            agent_name: Name of agent to execute
            task: Task to execute
            limits: Resource limits

        Returns:
            AgentResult with execution details
        """
        start_time = time.time()

        logger.info(
            f"Executing agent '{agent_name}' with task: {task.action} "
            f"(timeout: {limits.timeout_seconds}s, memory: {limits.memory_mb}MB)"
        )

        # Find agent file
        agent_path = self._find_agent(agent_name)
        if not agent_path:
            error_msg = f"Agent '{agent_name}' not found in {self.agents_dir}"
            logger.error(error_msg)
            return AgentResult(
                success=False,
                exit_code=1,
                output="",
                error=error_msg,
                artifacts=[],
                duration_seconds=0.0,
                resource_usage={}
            )

        # Try to get worker from pool
        worker = await self.process_pool.get_worker(timeout=1.0)

        # If no worker available, spawn new process
        if worker is None:
            logger.warning("No worker available in pool, spawning new worker")
            worker = await self.process_pool._spawn_worker()

        if worker is None:
            error_msg = f"Failed to spawn process for agent '{agent_name}'"
            logger.error(error_msg)
            return AgentResult(
                success=False,
                exit_code=1,
                output="",
                error=error_msg,
                artifacts=[],
                duration_seconds=0.0,
                resource_usage={}
            )

        # Add to active processes with lock
        async with self._active_processes_lock:
            self.active_processes.add(worker.process)

        try:
            # Start resource enforcement in background
            enforce_task = asyncio.create_task(
                self.resource_enforcer.enforce(worker.process, limits),
                name=f"enforce-{agent_name}-{worker.process.pid}"
            )

            # Execute task using worker
            result = await self.process_pool.execute_task(
                worker,
                agent_path,
                task,
                limits.timeout_seconds
            )

            # Cancel enforcement task
            enforce_task.cancel()
            try:
                await enforce_task
            except asyncio.CancelledError:
                pass

            duration = time.time() - start_time

            logger.info(
                f"Agent '{agent_name}' completed in {duration:.2f}s "
                f"(exit code: {result.exit_code})"
            )

            # Return worker to pool
            await self.process_pool.return_worker(worker)

            return AgentResult(
                success=result.exit_code == 0,
                exit_code=result.exit_code,
                output=result.stdout,
                error=result.stderr,
                artifacts=result.artifacts,
                duration_seconds=duration,
                resource_usage={
                    "memory_mb": result.memory_mb,
                    "cpu_percent": result.cpu_percent
                }
            )

        except asyncio.TimeoutError:
            error_msg = f"Agent '{agent_name}' exceeded timeout of {limits.timeout_seconds}s"
            logger.error(error_msg)

            # Terminate the worker
            worker.terminate()

            duration = time.time() - start_time
            return AgentResult(
                success=False,
                exit_code=124,  # Timeout exit code
                output="",
                error=error_msg,
                artifacts=[],
                duration_seconds=duration,
                resource_usage={}
            )

        except Exception as e:
            error_msg = f"Unexpected error executing agent '{agent_name}': {e}"
            logger.exception(error_msg)

            # Return worker to pool (it may still be usable)
            await self.process_pool.return_worker(worker)

            duration = time.time() - start_time
            return AgentResult(
                success=False,
                exit_code=1,
                output="",
                error=error_msg,
                artifacts=[],
                duration_seconds=duration,
                resource_usage={}
            )

        finally:
            # Remove from active processes with lock
            async with self._active_processes_lock:
                self.active_processes.discard(worker.process)

    def _find_agent(self, agent_name: str) -> Optional[Path]:
        """
        Find agent markdown file by name.

        Searches all category directories for matching agent file.

        Args:
            agent_name: Agent name (e.g., "python-pro")

        Returns:
            Path to agent markdown file, or None if not found
        """
        logger.debug(f"Searching for agent: {agent_name}")

        if not self.agents_dir.exists():
            logger.error(f"Agents directory not found: {self.agents_dir}")
            return None

        # Search in all category directories
        for category_dir in self.agents_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Try exact match first
            agent_file = category_dir / f"{agent_name}.md"
            if agent_file.exists():
                logger.debug(f"Found agent: {agent_file}")
                return agent_file

            # Search recursively
            for agent_file in category_dir.rglob(f"{agent_name}.md"):
                logger.debug(f"Found agent: {agent_file}")
                return agent_file

        logger.warning(f"Agent '{agent_name}' not found in {self.agents_dir}")
        return None

    def _get_default_limits(self) -> ResourceLimits:
        """
        Get default resource limits from config.

        Returns:
            ResourceLimits with values from RuntimeConfig
        """
        return ResourceLimits(
            memory_mb=self.config.default_memory_mb,
            cpu_percent=self.config.default_cpu_percent,
            timeout_seconds=self.config.default_timeout_seconds
        )

    def _kill_all_agents(self) -> None:
        """
        Emergency cleanup - kill all spawned agents.

        Called automatically on exit via atexit handler.
        Ensures no orphaned processes remain.
        """
        logger.info("Cleaning up agent processes...")

        killed_count = 0
        for proc in list(self.active_processes):
            try:
                if proc.is_alive():
                    # Try graceful termination
                    proc.terminate()
                    proc.join(timeout=2)

                    # Force kill if still alive
                    if proc.is_alive():
                        logger.warning(f"Force killing process {proc.pid}")
                        proc.kill()
                        proc.join(timeout=1)

                    killed_count += 1

            except Exception as e:
                logger.error(f"Error killing process: {e}")

        if killed_count > 0:
            logger.info(f"Terminated {killed_count} agent process(es)")
        else:
            logger.debug("No active processes to clean up")

    def stop(self) -> None:
        """
        Shutdown runtime gracefully.

        Terminates all active processes, stops the process pool,
        and cleans up resources.

        Should be called before program exit.
        """
        logger.info("Stopping AgentRuntime...")

        # Stop process pool
        if self._event_loop:
            self._event_loop.run_until_complete(self.process_pool.stop())

        # Kill all active processes
        self._kill_all_agents()

        logger.info("AgentRuntime stopped")
