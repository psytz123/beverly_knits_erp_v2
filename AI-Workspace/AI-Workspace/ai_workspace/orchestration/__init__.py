#!/usr/bin/env python3
"""
Orchestration package for multi-agent coordination.

Provides task decomposition, dependency resolution, agent selection,
execution management, and high-performance message batching for
autonomous multi-agent workflows.
"""

from .agent_selector import AgentAssignment, AgentSelector
from .dependency_resolver import (
    CycleDetectedError,
    DependencyResolver,
    InvalidGraphError,
)
# Commented out due to import issues - import directly if needed
# from .execution_manager import ExecutionManager, ExecutionResult
from .message_broker import (
    BatchConfig,
    BatchMetrics,
    DeliveryStrategy,
    Message,
    MessageBroker,
)
from .performance_tracker import AgentMetrics, PerformanceTracker
from .task_graph import Task, TaskGraph, TaskStatus

__all__ = [
    # Task Graph
    "Task",
    "TaskGraph",
    "TaskStatus",
    # Dependency Resolution
    "DependencyResolver",
    "CycleDetectedError",
    "InvalidGraphError",
    # Performance Tracking
    "PerformanceTracker",
    "AgentMetrics",
    # Agent Selection
    "AgentSelector",
    "AgentAssignment",
    # Message Batching
    "MessageBroker",
    "Message",
    "BatchConfig",
    "BatchMetrics",
    "DeliveryStrategy",
    # Execution Management - commented out
    # "ExecutionManager",
    # "ExecutionResult",
]
