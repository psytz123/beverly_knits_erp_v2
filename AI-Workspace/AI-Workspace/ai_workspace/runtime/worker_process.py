"""
Worker Process - Pre-warmed Process Pool for Fast Agent Startup.

This module implements long-running worker processes that execute agent tasks
from a work queue, enabling sub-500ms agent startup by eliminating process
spawn overhead (PERF-001).

Features:
    - Pre-loaded expensive imports (yaml, json, pathlib)
    - Work queue consumption with multiprocessing.Queue
    - Graceful shutdown and error handling
    - Keep-alive mechanism with 60s timeout
    - Task execution with result reporting
    - Memory-efficient design for pool reuse

Architecture:
    Main Process (AgentProcessPool)
        │
        ├─> Work Queue (multiprocessing.Queue)
        │   └─> Tasks sent here by orchestrator
        │
        ├─> Result Queue (multiprocessing.Queue)
        │   └─> Results collected here
        │
        └─> Worker Processes (4+ workers)
            ├─> Worker 1: worker_main() loop
            ├─> Worker 2: worker_main() loop
            ├─> Worker 3: worker_main() loop
            └─> Worker 4: worker_main() loop

Workflow:
    1. Worker starts and pre-loads imports
    2. Worker waits on work queue (60s timeout)
    3. Receives task → executes → sends result
    4. Loop continues until SHUTDOWN command
    5. Graceful cleanup on exit

Example:
    >>> import multiprocessing as mp
    >>> work_queue = mp.Queue()
    >>> result_queue = mp.Queue()
    >>>
    >>> # Start worker in background
    >>> worker = mp.Process(
    ...     target=worker_main,
    ...     args=(work_queue, result_queue)
    ... )
    >>> worker.start()
    >>>
    >>> # Send task
    >>> work_queue.put({
    ...     "task_id": "task-001",
    ...     "agent_name": "python-pro",
    ...     "action": "write_function",
    ...     "spec": "Create Fibonacci calculator"
    ... })
    >>>
    >>> # Get result
    >>> result = result_queue.get(timeout=10)
    >>> print(result["success"])
    True
    >>>
    >>> # Shutdown worker
    >>> work_queue.put({"command": "SHUTDOWN"})
    >>> worker.join(timeout=5)
"""

import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Pre-load expensive imports at module level
import yaml

# Setup logging for worker process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s/%(process)d] %(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Constants
WORKER_TIMEOUT_SECONDS = 60
SHUTDOWN_COMMAND = "SHUTDOWN"
KEEPALIVE_COMMAND = "KEEPALIVE"


class WorkerTask:
    """
    Task specification for worker execution.

    Attributes:
        task_id: Unique identifier for this task
        agent_name: Name of agent to execute (e.g., "python-pro")
        action: Action type (e.g., "write_function", "review_code")
        spec: Detailed task specification
        timeout_seconds: Maximum execution time
        parameters: Optional task-specific parameters
    """

    def __init__(
        self,
        task_id: str,
        agent_name: str,
        action: str,
        spec: str,
        timeout_seconds: int = 600,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize worker task.

        Args:
            task_id: Unique identifier for this task
            agent_name: Name of agent to execute
            action: Action type
            spec: Task specification
            timeout_seconds: Maximum execution time
            parameters: Optional parameters
        """
        self.task_id = task_id
        self.agent_name = agent_name
        self.action = action
        self.spec = spec
        self.timeout_seconds = timeout_seconds
        self.parameters = parameters or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerTask":
        """
        Create WorkerTask from dictionary.

        Args:
            data: Dictionary with task fields

        Returns:
            WorkerTask instance

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["task_id", "agent_name", "action", "spec"]
        missing = [f for f in required_fields if f not in data]

        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        return cls(
            task_id=data["task_id"],
            agent_name=data["agent_name"],
            action=data["action"],
            spec=data["spec"],
            timeout_seconds=data.get("timeout_seconds", 600),
            parameters=data.get("parameters", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary.

        Returns:
            Dictionary representation of task
        """
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "action": self.action,
            "spec": self.spec,
            "timeout_seconds": self.timeout_seconds,
            "parameters": self.parameters
        }


class WorkerResult:
    """
    Result from worker task execution.

    Attributes:
        task_id: ID of completed task
        success: Whether execution succeeded
        exit_code: Exit code (0 = success)
        output: Standard output
        error: Error message if failed
        duration_seconds: Actual execution time
        artifacts: List of created/modified files
    """

    def __init__(
        self,
        task_id: str,
        success: bool,
        exit_code: int,
        output: str,
        error: str,
        duration_seconds: float,
        artifacts: Optional[list] = None
    ):
        """
        Initialize worker result.

        Args:
            task_id: ID of completed task
            success: Whether execution succeeded
            exit_code: Exit code
            output: Standard output
            error: Error message
            duration_seconds: Execution time
            artifacts: Created/modified files
        """
        self.task_id = task_id
        self.success = success
        self.exit_code = exit_code
        self.output = output
        self.error = error
        self.duration_seconds = duration_seconds
        self.artifacts = artifacts or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation of result
        """
        return {
            "task_id": self.task_id,
            "success": self.success,
            "exit_code": self.exit_code,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "artifacts": self.artifacts
        }


def execute_task(task: WorkerTask, workspace_path: Path) -> WorkerResult:
    """
    Execute a single agent task.

    This function performs the actual agent execution:
    1. Loads agent from markdown file
    2. Parses agent metadata
    3. Executes agent logic
    4. Collects results

    Args:
        task: Task to execute
        workspace_path: Path to .ai-workspace directory

    Returns:
        WorkerResult with execution details

    Note:
        Currently returns a mock result for testing.
        Full implementation requires integration with AgentFactory.
    """
    start_time = time.time()
    logger.info(f"Executing task {task.task_id}: {task.agent_name}/{task.action}")

    try:
        # TODO: Integrate with AgentFactory for actual agent execution
        # from ai_workspace.runtime.agent_factory import AgentFactory
        # factory = AgentFactory(workspace_path)
        # agent = factory.create_agent(task.agent_name)
        # result = agent.execute(task.action, task.spec)

        # For now, return mock success result
        duration = time.time() - start_time

        # Simulate brief work
        time.sleep(0.1)

        return WorkerResult(
            task_id=task.task_id,
            success=True,
            exit_code=0,
            output=f"Task {task.task_id} completed successfully by {task.agent_name}",
            error="",
            duration_seconds=duration,
            artifacts=[]
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Task execution failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)

        return WorkerResult(
            task_id=task.task_id,
            success=False,
            exit_code=1,
            output="",
            error=error_msg,
            duration_seconds=duration,
            artifacts=[]
        )


def worker_main(work_queue: mp.Queue, result_queue: mp.Queue, workspace_path: Optional[str] = None) -> None:
    """
    Long-running worker process that executes agent tasks.

    This is the main entry point for worker processes. It:
    1. Pre-loads expensive imports (already done at module level)
    2. Initializes workspace path
    3. Enters main work loop
    4. Consumes tasks from work_queue
    5. Executes tasks and sends results to result_queue
    6. Handles shutdown gracefully

    Args:
        work_queue: Queue to receive tasks from orchestrator
        result_queue: Queue to send results back to orchestrator
        workspace_path: Path to .ai-workspace directory (optional)

    Workflow:
        1. Worker starts and logs initialization
        2. Enter main loop:
           a. Wait for task from queue (60s timeout)
           b. If SHUTDOWN: exit gracefully
           c. If KEEPALIVE: acknowledge and continue
           d. If task: execute and send result
           e. If timeout: loop continues (worker stays alive)
        3. On shutdown: cleanup and exit

    Keep-Alive Mechanism:
        - Worker waits 60s for tasks
        - Timeout is NOT an error (worker stays alive)
        - Orchestrator can send KEEPALIVE to check health
        - Orchestrator sends SHUTDOWN to terminate

    Error Handling:
        - Task execution errors are caught and returned as failed results
        - Worker continues running after task failures
        - Worker exits only on SHUTDOWN or fatal errors

    Example:
        >>> import multiprocessing as mp
        >>> work_queue = mp.Queue()
        >>> result_queue = mp.Queue()
        >>>
        >>> # Start worker
        >>> worker = mp.Process(
        ...     target=worker_main,
        ...     args=(work_queue, result_queue, "/path/to/.ai-workspace")
        ... )
        >>> worker.start()
        >>>
        >>> # Send task
        >>> work_queue.put({
        ...     "task_id": "test-001",
        ...     "agent_name": "python-pro",
        ...     "action": "write_code",
        ...     "spec": "Create hello world"
        ... })
        >>>
        >>> # Get result
        >>> result = result_queue.get(timeout=10)
        >>>
        >>> # Shutdown
        >>> work_queue.put({"command": "SHUTDOWN"})
        >>> worker.join(timeout=5)
    """
    # Determine workspace path
    if workspace_path is None:
        # Default to bundled .ai-workspace in package
        workspace_path = str(Path(__file__).parent.parent / ".ai-workspace")

    workspace = Path(workspace_path)

    logger.info(f"Worker process started (PID: {mp.current_process().pid})")
    logger.info(f"Workspace: {workspace}")
    logger.info("Pre-loaded modules: yaml, json, pathlib")
    logger.info(f"Ready to process tasks (timeout: {WORKER_TIMEOUT_SECONDS}s)")

    task_count = 0

    try:
        while True:
            try:
                # Wait for task with timeout (keep-alive mechanism)
                logger.debug(f"Waiting for task (timeout: {WORKER_TIMEOUT_SECONDS}s)...")
                message = work_queue.get(timeout=WORKER_TIMEOUT_SECONDS)

                # Check for control commands
                if isinstance(message, dict) and "command" in message:
                    command = message["command"]

                    if command == SHUTDOWN_COMMAND:
                        logger.info("Received SHUTDOWN command, exiting gracefully")
                        break

                    elif command == KEEPALIVE_COMMAND:
                        logger.debug("Received KEEPALIVE ping")
                        result_queue.put({
                            "command": "KEEPALIVE_ACK",
                            "pid": mp.current_process().pid,
                            "task_count": task_count
                        })
                        continue

                    else:
                        logger.warning(f"Unknown command: {command}")
                        continue

                # Parse task from message
                try:
                    task = WorkerTask.from_dict(message)
                except (ValueError, KeyError) as e:
                    logger.error(f"Invalid task format: {e}")
                    logger.error(f"Message: {message}")
                    continue

                # Execute task
                logger.info(f"Processing task {task.task_id}: {task.agent_name}/{task.action}")
                result = execute_task(task, workspace)
                task_count += 1

                # Send result back
                result_queue.put(result.to_dict())
                logger.info(
                    f"Task {task.task_id} completed in {result.duration_seconds:.3f}s "
                    f"(success: {result.success})"
                )

            except mp.queues.Empty:
                # Timeout is normal - worker stays alive
                logger.debug("No tasks received (timeout), continuing...")
                continue

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down")
                break

            except Exception as e:
                # Unexpected error in main loop
                error_msg = f"Unexpected error in worker loop: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)

                # Send error result if we have task context
                try:
                    error_result = {
                        "task_id": "unknown",
                        "success": False,
                        "exit_code": 1,
                        "output": "",
                        "error": error_msg,
                        "duration_seconds": 0.0,
                        "artifacts": []
                    }
                    result_queue.put(error_result)
                except Exception as queue_error:
                    logger.error(f"Failed to send error result: {queue_error}")

                # Continue running (don't crash on single error)
                continue

    except Exception as e:
        # Fatal error - worker must exit
        logger.critical(f"Fatal error in worker process: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

    finally:
        logger.info(f"Worker shutting down (processed {task_count} tasks)")
        logger.info("Cleanup complete, exiting")


if __name__ == "__main__":
    """
    Allow worker to be run standalone for testing.

    Example:
        >>> python worker_process.py
    """
    print("Worker Process Module - Standalone Test Mode")
    print("=" * 60)

    # Create test queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    print("Starting worker process...")
    worker = mp.Process(
        target=worker_main,
        args=(work_queue, result_queue),
        name="TestWorker"
    )
    worker.start()

    print(f"Worker started (PID: {worker.pid})")
    print("Sending test task...")

    # Send test task
    test_task = {
        "task_id": "test-001",
        "agent_name": "python-pro",
        "action": "test_action",
        "spec": "Test specification"
    }
    work_queue.put(test_task)

    # Wait for result
    print("Waiting for result...")
    try:
        result = result_queue.get(timeout=10)
        print("Result received:")
        print(json.dumps(result, indent=2))
    except mp.queues.Empty:
        print("ERROR: No result received within 10 seconds")

    # Shutdown worker
    print("\nShutting down worker...")
    work_queue.put({"command": SHUTDOWN_COMMAND})
    worker.join(timeout=5)

    if worker.is_alive():
        print("WARNING: Worker did not shutdown gracefully, terminating...")
        worker.terminate()
        worker.join(timeout=2)

    print("Test complete!")
