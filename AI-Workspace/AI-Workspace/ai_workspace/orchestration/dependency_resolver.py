#!/usr/bin/env python3
"""
Dependency Resolver - Graph Analysis for Task Orchestration.

This module provides algorithms for analyzing and resolving task dependencies
in a directed acyclic graph (DAG). It includes topological sorting, critical path
analysis, parallel batch identification, and cycle detection.

Classes:
    CycleDetectedError: Raised when circular dependency detected
    InvalidGraphError: Raised when graph validation fails
    DependencyResolver: Analyze and resolve task dependencies

Example:
    >>> from task_graph import Task, TaskGraph
    >>> graph = TaskGraph(tasks=[
    ...     Task(id="t1", name="Setup", agent_id="python-pro", dependencies=[], estimated_effort=10),
    ...     Task(id="t2", name="Build", agent_id="backend-developer", dependencies=["t1"], estimated_effort=30),
    ... ])
    >>> resolver = DependencyResolver(graph)
    >>> ordered = resolver.topological_sort()
    >>> print([t.id for t in ordered])
    ['t1', 't2']
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)


class CycleDetectedError(Exception):
    """
    Raised when circular dependency detected in task graph.

    This exception is raised when:
    - A task depends on itself (directly or indirectly)
    - A dependency cycle exists between multiple tasks
    - Graph validation detects a cycle via DFS
    """

    pass


class InvalidGraphError(Exception):
    """
    Raised when graph validation fails.

    This exception is raised when:
    - Task dependencies reference non-existent tasks
    - Task has self-dependency
    - Graph structure is invalid for DAG operations
    """

    pass


class DependencyResolver:
    """
    Analyze and resolve task dependencies for optimal execution.

    This class provides graph algorithms for:
    - Topological sorting (execution order)
    - Critical path analysis (bottleneck identification)
    - Parallel batch computation (concurrent execution groups)
    - Cycle detection (dependency validation)
    - Graph validation (DAG structure verification)

    Attributes:
        graph: TaskGraph instance containing tasks and dependencies

    Example:
        >>> graph = TaskGraph(tasks=[
        ...     Task(id="t1", name="Setup", agent_id="python-pro", dependencies=[], estimated_effort=10),
        ...     Task(id="t2", name="Build", agent_id="backend-developer", dependencies=["t1"], estimated_effort=30),
        ...     Task(id="t3", name="Test", agent_id="test-automator", dependencies=["t2"], estimated_effort=20),
        ... ])
        >>> resolver = DependencyResolver(graph)
        >>> ordered = resolver.topological_sort()  # [t1, t2, t3]
        >>> critical = resolver.identify_critical_path()  # [t1, t2, t3] (60min total)
        >>> batches = resolver.find_parallel_batches()  # [[t1], [t2], [t3]]
    """

    def __init__(self, graph: TaskGraph):
        """
        Initialize dependency resolver with task graph.

        Args:
            graph: TaskGraph instance to analyze

        Raises:
            ValueError: If graph is None or invalid
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

        self.graph = graph
        logger.info(f"DependencyResolver initialized with {len(graph.tasks)} tasks")

    def topological_sort(self) -> List[Task]:
        """
        Order tasks for sequential execution respecting dependencies.

        Uses Kahn's algorithm to compute a valid topological ordering
        of tasks. The resulting order ensures that all dependencies of
        a task are executed before the task itself.

        Returns:
            List of tasks in valid execution order

        Raises:
            CycleDetectedError: If circular dependency detected
            InvalidGraphError: If graph structure is invalid

        Example:
            >>> # Graph: t1 -> t2 -> t3
            >>> ordered = resolver.topological_sort()
            >>> print([t.id for t in ordered])
            ['t1', 't2', 't3']
        """
        # Validate graph first
        self.validate_graph()

        # Handle empty graph
        if not self.graph.tasks:
            logger.warning("Topological sort called on empty graph")
            return []

        # Build adjacency list and in-degree map
        in_degree: Dict[str, int] = {task.id: 0 for task in self.graph.tasks}
        adjacency: Dict[str, List[str]] = defaultdict(list)
        task_map: Dict[str, Task] = {task.id: task for task in self.graph.tasks}

        for task in self.graph.tasks:
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Initialize queue with tasks that have no dependencies
        queue: deque[str] = deque()
        for task_id, degree in in_degree.items():
            if degree == 0:
                queue.append(task_id)

        # Kahn's algorithm
        sorted_tasks: List[Task] = []

        while queue:
            current_id = queue.popleft()
            sorted_tasks.append(task_map[current_id])

            # Reduce in-degree for dependent tasks
            for dependent_id in adjacency[current_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Check if all tasks were processed (no cycle)
        if len(sorted_tasks) != len(self.graph.tasks):
            cycle = self.detect_cycles()
            raise CycleDetectedError(
                f"Circular dependency detected: {' -> '.join(cycle) if cycle else 'unknown'}"
            )

        logger.info(f"Topological sort completed: {len(sorted_tasks)} tasks ordered")
        return sorted_tasks

    def identify_critical_path(self) -> List[Task]:
        """
        Find longest path through graph (bottleneck for completion).

        Uses estimated_effort to calculate path lengths. The critical path
        represents the minimum time needed to complete all tasks, assuming
        unlimited parallelism for non-dependent tasks.

        Returns:
            List of tasks on critical path (longest weighted path)

        Raises:
            CycleDetectedError: If circular dependency detected

        Example:
            >>> # Graph: t1(10min) -> t2(30min) -> t3(20min)
            >>> critical = resolver.identify_critical_path()
            >>> total_time = sum(t.estimated_effort for t in critical)
            >>> print(f"Critical path: {total_time} minutes")
            Critical path: 60 minutes
        """
        # Get topologically sorted tasks
        sorted_tasks = self.topological_sort()

        if not sorted_tasks:
            logger.warning("Critical path called on empty graph")
            return []

        # Build task map and adjacency list
        task_map: Dict[str, Task] = {task.id: task for task in self.graph.tasks}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for task in self.graph.tasks:
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)

        # Calculate longest path to each task (dynamic programming)
        longest_path: Dict[str, int] = {}
        predecessor: Dict[str, Optional[str]] = {}

        for task in sorted_tasks:
            task_id = task.id

            # Base case: tasks with no dependencies
            if not task.dependencies:
                longest_path[task_id] = task.estimated_effort
                predecessor[task_id] = None
            else:
                # Find maximum path from dependencies
                max_path = 0
                best_pred = None

                for dep_id in task.dependencies:
                    if longest_path[dep_id] > max_path:
                        max_path = longest_path[dep_id]
                        best_pred = dep_id

                longest_path[task_id] = max_path + task.estimated_effort
                predecessor[task_id] = best_pred

        # Find task with longest path (critical path end)
        critical_end_id = max(longest_path.keys(), key=lambda k: longest_path[k])

        # Reconstruct critical path by following predecessors
        critical_path: List[Task] = []
        current_id: Optional[str] = critical_end_id

        while current_id is not None:
            critical_path.append(task_map[current_id])
            current_id = predecessor[current_id]

        # Reverse to get correct order (start -> end)
        critical_path.reverse()

        total_time = sum(task.estimated_effort for task in critical_path)
        logger.info(
            f"Critical path identified: {len(critical_path)} tasks, "
            f"{total_time} minutes total"
        )

        return critical_path

    def find_parallel_batches(self) -> List[List[Task]]:
        """
        Group tasks that can run concurrently (same dependency level).

        Returns list of batches where each batch contains tasks that:
        1. Have all dependencies satisfied by previous batches
        2. Can execute in parallel with other tasks in same batch

        Returns:
            List of batches, each containing tasks executable in parallel

        Raises:
            CycleDetectedError: If circular dependency detected

        Example:
            >>> # Graph: t1 -> [t2, t3] -> t4
            >>> batches = resolver.find_parallel_batches()
            >>> print([[t.id for t in batch] for batch in batches])
            [['t1'], ['t2', 't3'], ['t4']]
        """
        # Validate graph
        self.validate_graph()

        if not self.graph.tasks:
            logger.warning("Parallel batches called on empty graph")
            return []

        # Build task map and in-degree
        task_map: Dict[str, Task] = {task.id: task for task in self.graph.tasks}
        in_degree: Dict[str, int] = {task.id: 0 for task in self.graph.tasks}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for task in self.graph.tasks:
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Initialize first batch with tasks that have no dependencies
        batches: List[List[Task]] = []
        current_batch_ids: Set[str] = {
            task_id for task_id, degree in in_degree.items() if degree == 0
        }

        # Process batches level by level
        processed_count = 0

        while current_batch_ids:
            # Convert IDs to Task objects and add to batches
            batch_tasks = [task_map[task_id] for task_id in current_batch_ids]
            batches.append(batch_tasks)
            processed_count += len(batch_tasks)

            # Find next batch: tasks whose dependencies are now satisfied
            next_batch_ids: Set[str] = set()

            for task_id in current_batch_ids:
                # Reduce in-degree for dependent tasks
                for dependent_id in adjacency[task_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_batch_ids.add(dependent_id)

            current_batch_ids = next_batch_ids

        # Verify all tasks were processed (no cycle)
        if processed_count != len(self.graph.tasks):
            cycle = self.detect_cycles()
            raise CycleDetectedError(
                f"Circular dependency detected: {' -> '.join(cycle) if cycle else 'unknown'}"
            )

        logger.info(
            f"Parallel batches computed: {len(batches)} batches, "
            f"max parallelism: {max(len(batch) for batch in batches) if batches else 0}"
        )

        return batches

    def detect_cycles(self) -> Optional[List[str]]:
        """
        Detect circular dependencies using DFS.

        Uses depth-first search with visited and recursion stack tracking
        to identify cycles in the dependency graph.

        Returns:
            List of task IDs forming cycle (in order), or None if no cycle

        Example:
            >>> # Graph with cycle: t1 -> t2 -> t3 -> t1
            >>> cycle = resolver.detect_cycles()
            >>> print(cycle)
            ['t1', 't2', 't3', 't1']
        """
        if not self.graph.tasks:
            return None

        # Build adjacency list
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for task in self.graph.tasks:
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)

        # Track visited and recursion stack
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        parent: Dict[str, Optional[str]] = {}

        def dfs(task_id: str) -> Optional[List[str]]:
            """
            DFS helper for cycle detection.

            Args:
                task_id: Current task being visited

            Returns:
                List of task IDs in cycle if found, None otherwise
            """
            visited.add(task_id)
            rec_stack.add(task_id)

            # Explore dependencies
            for dependent_id in adjacency[task_id]:
                if dependent_id not in visited:
                    parent[dependent_id] = task_id
                    cycle = dfs(dependent_id)
                    if cycle:
                        return cycle
                elif dependent_id in rec_stack:
                    # Cycle detected - reconstruct path
                    cycle_path: List[str] = [dependent_id]
                    current = task_id

                    while current != dependent_id:
                        cycle_path.append(current)
                        current = parent.get(current)
                        if current is None:
                            break

                    cycle_path.append(dependent_id)  # Complete cycle
                    cycle_path.reverse()
                    return cycle_path

            rec_stack.remove(task_id)
            return None

        # Run DFS from all unvisited nodes
        for task in self.graph.tasks:
            if task.id not in visited:
                cycle = dfs(task.id)
                if cycle:
                    logger.warning(f"Cycle detected: {' -> '.join(cycle)}")
                    return cycle

        logger.debug("No cycles detected in graph")
        return None

    def validate_graph(self) -> bool:
        """
        Validate graph is a valid DAG.

        Performs comprehensive validation:
        1. All dependencies exist in graph
        2. No self-dependencies
        3. No circular dependencies (is acyclic)
        4. Graph structure is valid

        Returns:
            True if graph is valid DAG

        Raises:
            InvalidGraphError: If validation fails with descriptive error

        Example:
            >>> try:
            ...     resolver.validate_graph()
            ...     print("Graph is valid DAG")
            ... except InvalidGraphError as e:
            ...     print(f"Invalid graph: {e}")
        """
        if not self.graph.tasks:
            logger.debug("Empty graph validated successfully")
            return True

        # Build task ID set for quick lookup
        task_ids = {task.id for task in self.graph.tasks}

        # Check 1: All dependencies exist
        for task in self.graph.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise InvalidGraphError(
                        f"Task '{task.id}' depends on non-existent task '{dep_id}'. "
                        f"Ensure all dependencies reference valid task IDs."
                    )

        # Check 2: No self-dependencies
        for task in self.graph.tasks:
            if task.id in task.dependencies:
                raise InvalidGraphError(
                    f"Task '{task.id}' has self-dependency. "
                    f"Tasks cannot depend on themselves."
                )

        # Check 3: No cycles (acyclic graph)
        cycle = self.detect_cycles()
        if cycle:
            raise InvalidGraphError(
                f"Circular dependency detected: {' -> '.join(cycle)}. "
                f"Task graphs must be acyclic (DAG)."
            )

        logger.info(f"Graph validated successfully: {len(self.graph.tasks)} tasks")
        return True

    def get_dependency_depth(self, task_id: str) -> int:
        """
        Calculate dependency depth for a task.

        Depth is the length of the longest path from a root task (no dependencies)
        to the given task.

        Args:
            task_id: Task identifier

        Returns:
            Dependency depth (0 for root tasks)

        Raises:
            ValueError: If task_id doesn't exist

        Example:
            >>> # Graph: t1 -> t2 -> t3
            >>> depth_t3 = resolver.get_dependency_depth("t3")
            >>> print(depth_t3)  # 2 (t1->t2->t3)
            2
        """
        task = self.graph.get_task(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found in graph")

        # Topological sort ensures correct order
        sorted_tasks = self.topological_sort()
        depth_map: Dict[str, int] = {}

        for task in sorted_tasks:
            if not task.dependencies:
                depth_map[task.id] = 0
            else:
                # Depth is max depth of dependencies + 1
                max_dep_depth = max(depth_map[dep_id] for dep_id in task.dependencies)
                depth_map[task.id] = max_dep_depth + 1

        return depth_map[task_id]

    def get_task_level_map(self) -> Dict[int, List[Task]]:
        """
        Group tasks by dependency level.

        Returns:
            Dictionary mapping level (depth) to list of tasks at that level

        Example:
            >>> # Graph: t1 -> [t2, t3] -> t4
            >>> levels = resolver.get_task_level_map()
            >>> print({k: [t.id for t in v] for k, v in levels.items()})
            {0: ['t1'], 1: ['t2', 't3'], 2: ['t4']}
        """
        sorted_tasks = self.topological_sort()
        levels: Dict[int, List[Task]] = defaultdict(list)

        for task in sorted_tasks:
            depth = self.get_dependency_depth(task.id)
            levels[depth].append(task)

        return dict(levels)

    def __repr__(self) -> str:
        """String representation of dependency resolver."""
        return f"DependencyResolver(tasks={len(self.graph.tasks)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        cycle_status = "acyclic" if not self.detect_cycles() else "contains cycles"
        return f"DependencyResolver: {len(self.graph.tasks)} tasks ({cycle_status})"
