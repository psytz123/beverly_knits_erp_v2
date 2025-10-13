#!/usr/bin/env python3
"""
Task graph data structures for workflow orchestration.

Provides Task, TaskGraph, and TaskStatus for representing
multi-agent work breakdown with dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    Represents a single unit of work in a workflow.

    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        description: Detailed task description
        agent_id: Assigned agent (e.g., "python-pro", "backend-developer")
        dependencies: List of task IDs that must complete first
        estimated_effort: Estimated time in minutes
        complexity: Complexity score (1-5 scale)
        priority: Priority level (0-10, higher = more urgent)
        metadata: Additional task-specific data
        status: Current execution status
        result: Task execution result (populated after completion)
        error: Error message if failed
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
    """

    name: str
    agent_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: int = 30  # minutes
    complexity: int = 2  # 1-5 scale
    priority: int = 5  # 0-10 scale
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate task fields."""
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not self.agent_id:
            raise ValueError("Task must have an assigned agent")
        if self.complexity not in range(1, 6):
            raise ValueError(f"Complexity must be 1-5, got {self.complexity}")
        if self.priority not in range(0, 11):
            raise ValueError(f"Priority must be 0-10, got {self.priority}")
        if self.estimated_effort <= 0:
            raise ValueError(f"Estimated effort must be positive, got {self.estimated_effort}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "dependencies": self.dependencies,
            "estimated_effort": self.estimated_effort,
            "complexity": self.complexity,
            "priority": self.priority,
            "metadata": self.metadata,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize task from dictionary."""
        data = data.copy()
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])
        if "started_at" in data and isinstance(data["started_at"], str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if "completed_at" in data and isinstance(data["completed_at"], str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)

    def is_ready(self, completed_tasks: List[str]) -> bool:
        """
        Check if task is ready to execute.

        Args:
            completed_tasks: List of completed task IDs

        Returns:
            True if all dependencies are satisfied
        """
        return all(dep_id in completed_tasks for dep_id in self.dependencies)

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark task as completed.

        Args:
            result: Task execution result
        """
        self.status = TaskStatus.COMPLETED
        self.result = result or {}
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """
        Mark task as failed.

        Args:
            error: Error message
        """
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Task(id={self.id[:8]}, name='{self.name}', "
            f"agent={self.agent_id}, status={self.status.value})"
        )


@dataclass
class TaskGraph:
    """
    Directed Acyclic Graph (DAG) of tasks with dependencies.

    Attributes:
        tasks: List of tasks in the graph
        metadata: Additional graph-level metadata (e.g., project name)
    """

    tasks: List[Task] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate graph structure."""
        task_ids = [task.id for task in self.tasks]

        # Check for duplicate task IDs
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found in graph")

        # Validate dependencies reference existing tasks
        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"Task {task.id} depends on non-existent task {dep_id}"
                    )

    def add_task(self, task: Task) -> None:
        """
        Add task to graph.

        Args:
            task: Task to add

        Raises:
            ValueError: If task ID already exists
        """
        if any(t.id == task.id for t in self.tasks):
            raise ValueError(f"Task {task.id} already exists in graph")

        # Validate dependencies
        task_ids = {t.id for t in self.tasks}
        for dep_id in task.dependencies:
            if dep_id not in task_ids:
                raise ValueError(f"Dependency {dep_id} not found in graph")

        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute.

        Returns:
            List of tasks with all dependencies satisfied
        """
        completed_ids = [t.id for t in self.tasks if t.status == TaskStatus.COMPLETED]

        ready_tasks = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and task.is_ready(completed_ids):
                ready_tasks.append(task)

        return ready_tasks

    def is_complete(self) -> bool:
        """
        Check if all tasks are complete.

        Returns:
            True if all tasks completed successfully
        """
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def has_failed_tasks(self) -> bool:
        """
        Check if any tasks have failed.

        Returns:
            True if one or more tasks failed
        """
        return any(task.status == TaskStatus.FAILED for task in self.tasks)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """Deserialize graph from dictionary."""
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        metadata = data.get("metadata", {})
        return cls(tasks=tasks, metadata=metadata)

    def __repr__(self) -> str:
        """String representation for debugging."""
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return f"TaskGraph(tasks={len(self.tasks)}, completed={completed})"
