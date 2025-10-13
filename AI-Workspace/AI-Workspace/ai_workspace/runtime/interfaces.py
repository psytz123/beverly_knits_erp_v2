"""
Runtime System Interfaces.

This module defines Protocol-based contracts for all runtime components,
ensuring type safety and clear API boundaries.

Protocols:
    IAgentFactory: Agent instantiation contract
    IAgentRuntime: Agent execution contract with process isolation

Data Classes:
    AgentMetadata: Agent metadata extracted from markdown frontmatter
    Task: Task specification for agent execution
    AgentResult: Result from agent execution
    ResourceLimits: Resource constraints for agent execution
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime


@dataclass
class AgentMetadata:
    """
    Metadata extracted from agent markdown frontmatter.

    This data class represents the configuration and capabilities of an AI agent
    as defined in its markdown specification file.

    Attributes:
        name: Unique identifier for the agent (e.g., "python-pro")
        description: Human-readable description of agent's expertise
        tools: List of tool names the agent has permission to use
        model: AI model to use for this agent (default: "claude-sonnet-4")
        category: Agent category derived from file path (e.g., "Languages / scripting")
        file_path: Absolute path to the agent's markdown file

    Example:
        >>> metadata = AgentMetadata(
        ...     name="python-pro",
        ...     description="Expert Python developer...",
        ...     tools=["Read", "Write", "Bash"],
        ...     model="claude-sonnet-4",
        ...     category="Languages / scripting",
        ...     file_path="/path/to/python-pro.md"
        ... )
    """
    name: str
    description: str
    tools: List[str]
    model: str = "claude-sonnet-4"
    category: str = ""
    file_path: str = ""

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if not self.description:
            raise ValueError("Agent description cannot be empty")
        if not self.tools:
            raise ValueError("Agent must have at least one tool")


@dataclass
class Task:
    """
    Task specification for agent execution.

    Attributes:
        action: Action type (e.g., "write_function", "review_code", "debug")
        spec: Detailed task specification/description
        timeout_seconds: Maximum execution time (default: 600 seconds / 10 minutes)
        parameters: Optional task-specific parameters

    Example:
        >>> task = Task(
        ...     action="write_function",
        ...     spec="Create a Fibonacci calculator with memoization",
        ...     timeout_seconds=300,
        ...     parameters={"language": "python", "style": "functional"}
        ... )
    """
    action: str
    spec: str
    timeout_seconds: int = 600
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.action:
            raise ValueError("Task action cannot be empty")
        if not self.spec:
            raise ValueError("Task spec cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.parameters is None:
            self.parameters = {}


@dataclass
class AgentResult:
    """
    Result from agent execution.

    Attributes:
        success: Whether execution completed successfully
        exit_code: Process exit code (0 = success)
        output: Standard output from agent
        error: Standard error from agent
        artifacts: List of file paths created/modified by agent
        duration_seconds: Actual execution time
        resource_usage: Resource consumption metrics

    Example:
        >>> result = AgentResult(
        ...     success=True,
        ...     exit_code=0,
        ...     output="Function created successfully",
        ...     error="",
        ...     artifacts=["src/fibonacci.py"],
        ...     duration_seconds=2.5,
        ...     resource_usage={"memory_mb": 45.2, "cpu_percent": 12.5}
        ... )
    """
    success: bool
    exit_code: int
    output: str
    error: str
    artifacts: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    """
    Resource constraints for agent execution.

    Attributes:
        memory_mb: Maximum memory usage in megabytes (default: 2048 MB / 2 GB)
        cpu_percent: Maximum CPU usage percentage (default: 80%)
        timeout_seconds: Maximum execution time (default: 600 seconds / 10 minutes)

    Example:
        >>> limits = ResourceLimits(
        ...     memory_mb=1024,  # 1 GB
        ...     cpu_percent=50,   # 50% of one core
        ...     timeout_seconds=300  # 5 minutes
        ... )
    """
    memory_mb: int = 2048
    cpu_percent: int = 80
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        """Validate resource limits after initialization."""
        if self.memory_mb <= 0:
            raise ValueError("Memory limit must be positive")
        if self.cpu_percent <= 0 or self.cpu_percent > 100:
            raise ValueError("CPU percent must be between 1 and 100")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@runtime_checkable
class IAgentFactory(Protocol):
    """
    Contract for agent factory implementations.

    Defines the interface for creating executable agents from markdown
    specifications. Implementations must provide agent parsing and
    instantiation capabilities.

    Methods:
        create_agent: Create executable agent from markdown file
        parse_agent_metadata: Extract metadata from agent markdown file
    """

    def create_agent(self, agent_name: str) -> Any:
        """
        Create executable agent from markdown file.

        Args:
            agent_name: Name of agent to create (e.g., "python-pro")

        Returns:
            ExecutableAgent instance ready for execution

        Raises:
            AgentNotFoundError: If agent file doesn't exist
            AgentParsingError: If agent markdown is malformed
        """
        ...

    def parse_agent_metadata(self, md_path: Path) -> AgentMetadata:
        """
        Extract metadata from agent markdown file.

        Args:
            md_path: Path to agent markdown file

        Returns:
            AgentMetadata with parsed frontmatter

        Raises:
            AgentParsingError: If markdown is malformed or missing required fields
        """
        ...


@runtime_checkable
class IAgentRuntime(Protocol):
    """
    Contract for agent runtime implementations with process isolation.

    Defines the interface for executing agents in isolated processes with
    resource limits, process pooling, and fault isolation.

    Methods:
        execute: Execute agent with resource limits in isolated process
        stop: Shutdown runtime gracefully
    """

    def execute(
        self,
        agent_name: str,
        task: Task,
        limits: Optional[ResourceLimits] = None
    ) -> AgentResult:
        """
        Execute agent with resource limits in isolated process.

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
        ...

    def stop(self) -> None:
        """
        Shutdown runtime gracefully.

        Terminates all active processes, cleans up resources, and stops
        the process pool. Should be called before program exit.
        """
        ...
