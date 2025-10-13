"""
AI Workspace Runtime System.

This module provides the runtime execution environment for AI agents,
including dynamic agent loading, execution orchestration, resource management,
and process isolation.

Modules:
    interfaces: Protocol definitions and contracts for runtime components
    agent_factory: Dynamic agent instantiation from markdown specifications
    agent_runtime: Agent execution engine with process isolation and resource limits
    resource_limits: Cross-platform resource enforcement
    worker_process: Pre-warmed worker pool for fast agent startup
    ipc_protocol: Structured inter-process communication

Components:
    - AgentRuntime: Main execution orchestrator with process pooling
    - AgentProcessPool: Pre-spawned worker pool
    - ResourceEnforcer: Platform-aware resource limit enforcement
    - IPCChannel: Bi-directional structured messaging
    - Task, AgentResult, ResourceLimits: Core data structures
    - IAgentRuntime, IAgentFactory: Protocol contracts

Example:
    >>> from ai_workspace.runtime import AgentRuntime, Task, ResourceLimits
    >>>
    >>> # Initialize runtime
    >>> runtime = AgentRuntime()
    >>> runtime.start()
    >>>
    >>> # Execute agent with custom limits
    >>> task = Task(
    ...     action="write_function",
    ...     spec="Create Fibonacci calculator with memoization"
    ... )
    >>> limits = ResourceLimits(memory_mb=1024, timeout_seconds=300)
    >>> result = runtime.execute("python-pro", task, limits)
    >>>
    >>> # Check results
    >>> if result.success:
    ...     print(f"Output: {result.output}")
    ...     print(f"Duration: {result.duration_seconds:.2f}s")
    ...     print(f"Memory: {result.resource_usage['memory_mb']:.1f}MB")
    >>>
    >>> # Cleanup
    >>> runtime.stop()
"""

from .interfaces import (
    AgentMetadata,
    IAgentFactory,
    IAgentRuntime,
    Task,
    AgentResult,
    ResourceLimits,
)

from .agent_runtime import (
    AgentRuntime,
    AgentProcessPool,
    AgentNotFoundError,
    ResourceLimitError,
    ExecutionError,
)

from .resource_limits import (
    ResourceEnforcer,
    BaseEnforcer,
    PollingEnforcer,
    RlimitEnforcer,
    CgroupEnforcer,
)
from .config import RuntimeConfig

from .worker_process import (
    worker_main,
    WorkerTask,
    WorkerResult,
    SHUTDOWN_COMMAND,
    KEEPALIVE_COMMAND,
)

from .ipc_protocol import (
    IPCMessage,
    IPCChannel,
    MSG_TYPE_COMMAND,
    MSG_TYPE_RESULT,
    MSG_TYPE_LOG,
    MSG_TYPE_METRIC,
    MSG_TYPE_ARTIFACT,
    MSG_TYPE_ERROR,
)

# Import AgentFactory if available (may not exist yet - Task 3)
try:
    from .agent_factory import AgentFactory, ExecutableAgent, AgentParsingError

    __all__ = [
        # Interfaces and protocols
        "AgentMetadata",
        "IAgentFactory",
        "IAgentRuntime",
        "Task",
        "AgentResult",
        "ResourceLimits",
        "RuntimeConfig",

        # Runtime components
        "AgentRuntime",
        "AgentProcessPool",
        "AgentNotFoundError",
        "ResourceLimitError",
        "ExecutionError",

        # Resource enforcement
        "ResourceEnforcer",
        "BaseEnforcer",
        "PollingEnforcer",
        "RlimitEnforcer",
        "CgroupEnforcer",

        # Worker process pool
        "worker_main",
        "WorkerTask",
        "WorkerResult",
        "SHUTDOWN_COMMAND",
        "KEEPALIVE_COMMAND",

        # IPC protocol
        "IPCMessage",
        "IPCChannel",
        "MSG_TYPE_COMMAND",
        "MSG_TYPE_RESULT",
        "MSG_TYPE_LOG",
        "MSG_TYPE_METRIC",
        "MSG_TYPE_ARTIFACT",
        "MSG_TYPE_ERROR",

        # Agent factory (if available)
        "AgentFactory",
        "ExecutableAgent",
        "AgentParsingError",
    ]

except ImportError:
    # AgentFactory not yet implemented - that's Task 3
    __all__ = [
        # Interfaces and protocols
        "AgentMetadata",
        "IAgentFactory",
        "IAgentRuntime",
        "Task",
        "AgentResult",
        "ResourceLimits",
        "RuntimeConfig",

        # Runtime components
        "AgentRuntime",
        "AgentProcessPool",
        "AgentNotFoundError",
        "ResourceLimitError",
        "ExecutionError",

        # Resource enforcement
        "ResourceEnforcer",
        "BaseEnforcer",
        "PollingEnforcer",
        "RlimitEnforcer",
        "CgroupEnforcer",

        # Worker process pool
        "worker_main",
        "WorkerTask",
        "WorkerResult",
        "SHUTDOWN_COMMAND",
        "KEEPALIVE_COMMAND",

        # IPC protocol
        "IPCMessage",
        "IPCChannel",
        "MSG_TYPE_COMMAND",
        "MSG_TYPE_RESULT",
        "MSG_TYPE_LOG",
        "MSG_TYPE_METRIC",
        "MSG_TYPE_ARTIFACT",
        "MSG_TYPE_ERROR",
    ]
