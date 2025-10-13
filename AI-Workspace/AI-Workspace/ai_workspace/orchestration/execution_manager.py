#!/usr/bin/env python3
"""
Execution Manager for autonomous multi-agent coordination.

Coordinates task graph execution with dependency management, parallel execution,
and fault tolerance. Integrates message broker and context store for state management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .agent_selector import AgentSelector
from .dependency_resolver import DependencyResolver
from .task_graph import Task, TaskGraph, TaskStatus
from ..runtime.context_store import ContextStore
from ..runtime.message_broker import InMemoryMessageBroker
from ..runtime.communication_protocol import AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLimits:
    """
    Limits for parallel task execution.

    Attributes:
        max_concurrent_tasks: Maximum tasks to execute in parallel
        max_retries: Maximum retry attempts for failed tasks
        task_timeout: Default timeout per task (seconds)
        execution_timeout: Total execution timeout (seconds)
    """

    max_concurrent_tasks: int = 10
    max_retries: int = 3
    task_timeout: int = 600  # 10 minutes per task
    execution_timeout: int = 3600  # 1 hour total

    def __post_init__(self) -> None:
        """Validate execution limits."""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.task_timeout <= 0:
            raise ValueError("task_timeout must be positive")
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")


@dataclass
class ExecutionResult:
    """
    Result of workflow execution.

    Attributes:
        success: Whether all tasks completed successfully
        execution_id: Unique execution identifier
        total_duration: Total execution time in seconds
        tasks_completed: Number of successfully completed tasks
        tasks_failed: Number of failed tasks
        artifacts: List of artifacts produced by tasks
        error: Error message if execution failed
    """

    success: bool
    execution_id: str
    total_duration: float  # seconds
    tasks_completed: int
    tasks_failed: int
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "execution_id": self.execution_id,
            "total_duration": self.total_duration,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "artifacts": self.artifacts,
            "error": self.error,
        }


class ExecutionManager:
    """
    Coordinate autonomous multi-agent task execution.

    Features:
    - Task graph validation and execution
    - Dependency resolution with parallel batching
    - Agent selection and task assignment
    - Message-based inter-agent communication
    - Context-based state management
    - Fault tolerance with retries
    - Progress tracking and monitoring

    Example:
        >>> broker = InMemoryMessageBroker()
        >>> context = ContextStore()
        >>> selector = AgentSelector(PerformanceTracker())
        >>> manager = ExecutionManager(broker, context, selector)
        >>> graph = TaskGraph(tasks=[...])
        >>> result = await manager.execute_graph(graph)
        >>> print(f"Success: {result.success}")
    """

    def __init__(
        self,
        broker: InMemoryMessageBroker,
        context: ContextStore,
        selector: AgentSelector,
        limits: Optional[ExecutionLimits] = None,
    ):
        """
        Initialize execution manager.

        Args:
            broker: Message broker for inter-agent communication
            context: Context store for shared state
            selector: Agent selector for task assignment
            limits: Execution limits (optional)
        """
        self.broker = broker
        self.context = context
        self.selector = selector
        self.limits = limits or ExecutionLimits()
        self.active_executions: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"ExecutionManager initialized with "
            f"max_concurrent={self.limits.max_concurrent_tasks}, "
            f"max_retries={self.limits.max_retries}"
        )

    async def execute_graph(self, graph: TaskGraph) -> ExecutionResult:
        """
        Execute complete task graph with dependency management.

        Steps:
        1. Validate graph (check for cycles)
        2. Resolve dependencies (get execution order)
        3. Find parallel batches (waves of concurrent tasks)
        4. Execute each wave with parallel execution
        5. Update task statuses in context store
        6. Collect results and return

        Args:
            graph: TaskGraph to execute

        Returns:
            ExecutionResult with completion status

        Raises:
            ValueError: If graph is invalid
            asyncio.TimeoutError: If execution exceeds timeout
        """
        execution_id = uuid4().hex
        start_time = time.time()

        logger.info(
            f"Starting graph execution {execution_id} with {len(graph.tasks)} tasks"
        )

        try:
            # 1. Validate graph
            resolver = DependencyResolver(graph)
            resolver.validate_graph()

            # 2. Get parallel batches (waves of tasks)
            batches = resolver.find_parallel_batches()
            logger.info(f"Resolved {len(batches)} execution waves")

            # 3. Create execution context
            self._create_execution_context(execution_id, graph)

            # 4. Execute each wave
            completed: List[Task] = []
            failed: List[Task] = []
            all_artifacts: List[Dict[str, Any]] = []

            for wave_num, wave in enumerate(batches, 1):
                logger.info(
                    f"Executing wave {wave_num}/{len(batches)} "
                    f"with {len(wave)} tasks"
                )

                # Execute wave with timeout
                try:
                    wave_results = await asyncio.wait_for(
                        self._execute_wave(wave, execution_id),
                        timeout=self.limits.execution_timeout,
                    )

                    # Categorize results
                    for task in wave_results:
                        if task.status == TaskStatus.COMPLETED:
                            completed.append(task)
                            if task.result:
                                all_artifacts.append(task.result)
                        elif task.status == TaskStatus.FAILED:
                            failed.append(task)

                    # Update execution status
                    self._update_execution_status(
                        execution_id,
                        status="in_progress",
                        tasks_completed=len(completed),
                        tasks_failed=len(failed),
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"Wave {wave_num} execution timeout exceeded "
                        f"{self.limits.execution_timeout}s"
                    )
                    # Mark remaining tasks as failed
                    for task in wave:
                        if task.status == TaskStatus.RUNNING:
                            task.mark_failed("Execution timeout exceeded")
                            failed.append(task)
                    break

            # 5. Build result
            duration = time.time() - start_time
            success = len(failed) == 0

            # Update final status
            self._update_execution_status(
                execution_id,
                status="completed" if success else "failed",
                tasks_completed=len(completed),
                tasks_failed=len(failed),
            )

            result = ExecutionResult(
                success=success,
                execution_id=execution_id,
                total_duration=duration,
                tasks_completed=len(completed),
                tasks_failed=len(failed),
                artifacts=all_artifacts,
                error=f"{len(failed)} tasks failed" if failed else None,
            )

            logger.info(
                f"Execution {execution_id} completed: "
                f"success={success}, duration={duration:.2f}s, "
                f"completed={len(completed)}, failed={len(failed)}"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Update execution status
            self._update_execution_status(
                execution_id, status="error", tasks_completed=0, tasks_failed=len(graph.tasks)
            )

            return ExecutionResult(
                success=False,
                execution_id=execution_id,
                total_duration=duration,
                tasks_completed=0,
                tasks_failed=len(graph.tasks),
                error=error_msg,
            )

    async def _execute_wave(
        self, wave: List[Task], execution_id: str
    ) -> List[Task]:
        """
        Execute batch of parallel tasks.

        Steps:
        1. Select best agent for each task
        2. Create execution tasks (coroutines)
        3. Execute with semaphore-limited concurrency
        4. Update context store with results
        5. Return completed tasks

        Args:
            wave: List of tasks to execute in parallel
            execution_id: Current execution ID

        Returns:
            List of completed tasks
        """
        if not wave:
            return []

        # 1. Select agents for all tasks in wave
        assignments = []
        for task in wave:
            # Convert orchestration Task to runtime-style task for selector
            runtime_task = self._convert_to_runtime_task(task)
            assignment = self.selector.select_agent(runtime_task)
            assignments.append((task, assignment.agent_id))

            logger.debug(
                f"Assigned task '{task.name}' to agent '{assignment.agent_id}' "
                f"(confidence: {assignment.confidence:.2f})"
            )

        # 2. Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.limits.max_concurrent_tasks)

        # 3. Execute tasks with concurrency limit
        async def execute_with_semaphore(
            task: Task, agent_id: str
        ) -> Task:
            """Execute single task with semaphore."""
            async with semaphore:
                return await self._execute_task(task, agent_id, execution_id)

        # Create all execution tasks
        execution_tasks = [
            execute_with_semaphore(task, agent_id)
            for task, agent_id in assignments
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # Handle exceptions
        completed_tasks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = assignments[i][0]
                task.mark_failed(f"Execution error: {str(result)}")
                logger.error(
                    f"Task '{task.name}' failed with exception: {result}",
                    exc_info=result,
                )
            else:
                completed_tasks.append(result)

        # 4. Update context store
        for task in completed_tasks:
            self.context.set_task_state(
                task.id,
                status=task.status.value,
                progress=1.0 if task.status == TaskStatus.COMPLETED else 0.0,
                metadata={
                    "result": task.result,
                    "error": task.error,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                },
            )

        return completed_tasks

    async def _execute_task(
        self, task: Task, agent_id: str, execution_id: str
    ) -> Task:
        """
        Execute single task via message broker (MVP: mock execution).

        For MVP, this simulates task execution. In production, this would:
        1. Send REQUEST message to agent via broker
        2. Wait for RESPONSE message
        3. Update task status based on response

        Args:
            task: Task to execute
            agent_id: Agent to use
            execution_id: Execution context

        Returns:
            Completed task
        """
        # Mark task as started
        task.mark_started()

        logger.debug(
            f"Executing task '{task.name}' with agent '{agent_id}' "
            f"(execution: {execution_id})"
        )

        # Update agent status in context
        self.context.update_agent_status(
            agent_id, status="busy", current_task=task.id, workload=1
        )

        try:
            # MVP: Simulate task execution with delay
            # In production: Use message broker to communicate with agent
            execution_time = min(task.estimated_effort / 60.0, 1.0)  # Convert minutes to seconds (cap at 1s for testing)
            await asyncio.sleep(execution_time)

            # Simulate success/failure based on complexity
            # Higher complexity = higher chance of failure (for testing)
            import random

            success_probability = 1.0 - (task.complexity * 0.1)  # 90% success for complexity 1, 50% for complexity 5
            success = random.random() < success_probability

            if success:
                # Mark as completed with result
                result = {
                    "agent": agent_id,
                    "execution_id": execution_id,
                    "output": f"Task '{task.name}' completed successfully",
                    "artifacts": [],
                    "timestamp": datetime.now().isoformat(),
                }
                task.mark_completed(result)
                logger.info(f"Task '{task.name}' completed successfully")
            else:
                # Mark as failed
                error = f"Task complexity {task.complexity} caused failure (simulated)"
                task.mark_failed(error)
                logger.warning(f"Task '{task.name}' failed: {error}")

        except Exception as e:
            # Handle execution error
            error_msg = f"Task execution error: {str(e)}"
            task.mark_failed(error_msg)
            logger.error(f"Task '{task.name}' failed: {error_msg}", exc_info=True)

        finally:
            # Update agent status back to idle
            self.context.update_agent_status(
                agent_id, status="idle", current_task=None, workload=0
            )

        return task

    def _convert_to_runtime_task(self, task: Task) -> Any:
        """
        Convert orchestration Task to runtime Task format.

        Args:
            task: Orchestration task

        Returns:
            Runtime-style task object
        """
        # Import runtime Task to avoid circular imports
        from ..runtime.interfaces import Task as RuntimeTask

        # Extract parameters from task metadata
        parameters = task.metadata.copy() if task.metadata else {}

        # Add task-specific info to parameters
        parameters.update(
            {
                "complexity": task.complexity,
                "priority": task.priority,
                "estimated_effort": task.estimated_effort,
            }
        )

        return RuntimeTask(
            action=task.name,
            spec=task.description or f"Execute task: {task.name}",
            timeout_seconds=task.estimated_effort * 60,  # Convert minutes to seconds
            parameters=parameters,
        )

    def _create_execution_context(
        self, execution_id: str, graph: TaskGraph
    ) -> None:
        """
        Create execution context in store.

        Stores:
        - Execution ID
        - Task graph
        - Start time
        - Initial status

        Args:
            execution_id: Execution identifier
            graph: Task graph to execute
        """
        context_data = {
            "execution_id": execution_id,
            "status": "running",
            "graph": graph.to_dict(),
            "start_time": datetime.now().isoformat(),
            "tasks_total": len(graph.tasks),
            "tasks_completed": 0,
            "tasks_failed": 0,
        }

        self.context.set_project_config(f"execution_{execution_id}", context_data)

        self.active_executions[execution_id] = context_data

        logger.debug(f"Created execution context: {execution_id}")

    def _update_execution_status(
        self,
        execution_id: str,
        status: str,
        tasks_completed: int = 0,
        tasks_failed: int = 0,
    ) -> None:
        """
        Update execution status in context store.

        Args:
            execution_id: Execution identifier
            status: New status (running, completed, failed, error)
            tasks_completed: Number of completed tasks
            tasks_failed: Number of failed tasks
        """
        # Get existing context
        context_data = self.context.get_project_config(f"execution_{execution_id}")

        if context_data:
            # Update status
            context_data["status"] = status
            context_data["tasks_completed"] = tasks_completed
            context_data["tasks_failed"] = tasks_failed
            context_data["updated_at"] = datetime.now().isoformat()

            # Save back to context store
            self.context.set_project_config(f"execution_{execution_id}", context_data)

            # Update active executions
            self.active_executions[execution_id] = context_data

            logger.debug(
                f"Updated execution {execution_id}: status={status}, "
                f"completed={tasks_completed}, failed={tasks_failed}"
            )

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get current execution status.

        Args:
            execution_id: Execution to query

        Returns:
            Dict with status, progress, active tasks, etc.
        """
        # Try active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]

        # Try context store
        context_data = self.context.get_project_config(f"execution_{execution_id}")

        if context_data:
            return context_data

        # Not found
        logger.warning(f"Execution {execution_id} not found")
        return {
            "execution_id": execution_id,
            "status": "not_found",
            "error": "Execution not found",
        }

    def get_all_executions(self) -> List[Dict[str, Any]]:
        """
        Get all execution statuses.

        Returns:
            List of execution status dictionaries
        """
        return list(self.active_executions.values())

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ExecutionManager(active_executions={len(self.active_executions)}, "
            f"max_concurrent={self.limits.max_concurrent_tasks})"
        )
