"""
Runtime Configuration System.

Provides comprehensive configuration management for the agent runtime,
including process pool settings, resource limits, IPC parameters, and
performance monitoring options.

Features:
    - Type-safe configuration with validation
    - Multiple loading sources (YAML, dict, environment)
    - Sensible defaults optimized for performance
    - Runtime validation and bounds checking

Example:
    >>> # Load from YAML
    >>> config = RuntimeConfig.from_yaml(Path("config.yml"))
    >>>
    >>> # Load from environment
    >>> config = RuntimeConfig.from_env()
    >>>
    >>> # Use in runtime
    >>> runtime = AgentRuntime(config=config)

Configuration Priority:
    1. Explicitly provided config object
    2. Environment variables (AI_RUNTIME_*)
    3. YAML file (if specified)
    4. Built-in defaults
"""

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """
    Complete runtime configuration with all tunable parameters.

    Configuration is organized into logical sections:
    - Process Pool: Worker management and lifecycle
    - Resource Limits: Default constraints for agents
    - IPC Settings: Inter-process communication parameters
    - Performance: Monitoring and metrics collection
    - Context Sync: Incremental context synchronization (PERF-011)

    All values have sensible defaults optimized for production use,
    but can be tuned based on workload characteristics.
    """

    # ==================== Process Pool Configuration ====================

    pool_size: int = 4
    """Number of pre-spawned worker processes in the pool.

    Recommended: Set to number of CPU cores for CPU-bound tasks,
                or higher (2-4x cores) for I/O-bound tasks.
    Performance Impact: Higher = lower latency, higher memory usage.
    """

    pool_warmup_enabled: bool = True
    """Whether to pre-spawn workers during startup.

    When enabled, workers are spawned proactively to reduce first-task latency.
    When disabled, workers are spawned on-demand (lazy initialization).
    """

    worker_recycle_after_tasks: int = 100
    """Recycle worker process after executing this many tasks.

    Prevents memory leaks and ensures fresh environment periodically.
    Set to 0 to disable recycling (workers live until max_age).
    """

    worker_max_age_seconds: int = 3600
    """Maximum lifetime for a worker process (seconds).

    Workers exceeding this age are gracefully terminated and replaced.
    Default: 1 hour (3600s). Set to 0 to disable age-based recycling.
    """

    worker_timeout_seconds: int = 60
    """Keep-alive timeout for idle workers (seconds).

    Workers idle longer than this are eligible for termination
    if pool size exceeds minimum. Default: 60s.
    """

    # ==================== Resource Limits Configuration ====================

    default_memory_mb: int = 2048
    """Default memory limit per agent (MB).

    Applied when no explicit limit is provided.
    Recommended: 1024-4096 MB for typical agents.
    """

    default_cpu_percent: int = 80
    """Default CPU limit as percentage of one core.

    100 = 1 full core, 200 = 2 cores, etc.
    Applied when no explicit limit is provided.
    """

    default_timeout_seconds: int = 600
    """Default execution timeout per agent (seconds).

    Applied when no explicit timeout is provided.
    Default: 10 minutes (600s).
    """

    enable_resource_monitoring: bool = True
    """Enable real-time resource usage monitoring.

    When enabled, tracks memory/CPU usage during execution.
    Disable for maximum performance (skip monitoring overhead).
    """

    monitoring_interval_seconds: float = 1.0
    """Interval between resource usage checks (seconds).

    Lower = more accurate, higher overhead.
    Higher = less overhead, less accurate.
    Recommended: 0.5 - 2.0 seconds.
    """

    # ==================== IPC Configuration ====================

    ipc_timeout_seconds: float = 5.0
    """Timeout for IPC message send operations (seconds).

    How long to wait when sending messages to workers.
    Increase if experiencing frequent timeouts under load.
    """

    ipc_collection_timeout_seconds: float = 10.0
    """Timeout for collecting results from workers (seconds).

    Maximum time to wait for worker response after task completion.
    Should be >= ipc_timeout_seconds.
    """

    max_message_size_kb: int = 1024
    """Maximum size for IPC messages (KB).

    Prevents excessive memory usage from large payloads.
    Default: 1 MB (1024 KB). Increase if handling large artifacts.
    """

    # ==================== Performance Configuration ====================

    enable_metrics_collection: bool = True
    """Enable performance metrics tracking.

    Collects task duration, resource usage, and throughput stats.
    Disable for maximum performance in production.
    """

    metrics_retention_count: int = 1000
    """Number of task metrics to retain in memory.

    Used for computing averages and percentiles.
    Higher = more accurate stats, higher memory usage.
    """

    enable_health_monitoring: bool = True
    """Enable health monitoring for workers.

    Periodically checks worker health and recycles unhealthy workers.
    Recommended to keep enabled in production.
    """

    # ==================== Message Batching Configuration ====================

    enable_message_batching: bool = True
    """Enable message batching in orchestration layer.

    When enabled, IPC messages (logs, metrics) are batched to reduce overhead.
    Provides 20x performance improvement for bulk message operations.
    """

    message_batch_size: int = 100
    """Maximum messages per batch before forcing flush.

    Higher = better throughput, higher latency.
    Lower = lower latency, more overhead.
    Recommended: 50-200 for most workloads.
    """

    message_batch_wait_ms: float = 50.0
    """Maximum wait time before flushing batch (milliseconds).

    Ensures messages don't wait too long even if batch not full.
    Recommended: 10-100ms depending on latency requirements.
    """

    # ==================== Context Sync Configuration (PERF-011) ====================

    context_sync_strategy: str = "incremental"
    """Context synchronization strategy.

    Options:
        - "incremental": Delta-based sync (fastest, <100ms for 5% changes)
        - "full": Always sync complete context (slowest, most reliable)
        - "snapshot": Periodic snapshots + incremental deltas

    Recommended: "incremental" for best performance (8x faster than full).
    """

    context_conflict_resolution: str = "last_write_wins"
    """Conflict resolution strategy for concurrent context updates.

    Options:
        - "last_write_wins": Most recent write takes precedence (default)
        - "merge": Attempt to merge both changes (dict merge, array concat)
        - "custom": Use custom resolver function

    Recommended: "last_write_wins" for simplicity, "merge" for collaborative workflows.
    """

    context_snapshot_interval: int = 10
    """Create full snapshot every N versions (for snapshot strategy).

    Used with "snapshot" strategy to balance delta size and rollback capability.
    Lower = more snapshots (faster rollback, higher memory).
    Higher = fewer snapshots (slower rollback, lower memory).
    """

    context_max_history: int = 10
    """Maximum number of context versions to retain in memory.

    Used for rollback and audit trail. Older versions are pruned automatically.
    Memory impact: ~5MB per 1000-key context * max_history.
    """

    
    # ==================== Agent Loading Configuration (PERF-015) ====================

    enable_agent_caching: bool = True
    """Enable persistent caching of agent metadata.

    When enabled, agent metadata is cached to disk for fast warm starts.
    Reduces loading time from 2800ms to <400ms (7x improvement).
    Disable to always load fresh metadata from disk.
    """

    agent_cache_ttl_hours: int = 24
    """Time-to-live for agent cache in hours.

    Cache is invalidated after this duration to ensure freshness.
    Recommended: 24 hours for daily updates, 168 hours (1 week) for stable systems.
    Set to 0 to disable TTL (cache never expires based on age).
    """

    max_concurrent_agent_loads: int = 20
    """Maximum concurrent agent file loads during initialization.

    Controls parallelism of async agent loading to prevent resource exhaustion.
    Higher = faster loading, higher memory/CPU usage.
    Lower = slower loading, lower resource usage.
    Recommended: 10-50 depending on disk I/O capabilities.
    """

    enable_lazy_agent_loading: bool = True
    """Enable lazy loading of agents on-demand.

    When enabled, agents are loaded only when first accessed.
    When disabled, all agents are loaded upfront during initialization.
    Recommended: Keep enabled for faster startup with large agent sets.
    """

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            ValueError: If configuration values are invalid
        """
        errors: list[str] = []

        # Process Pool Validation
        if self.pool_size < 1:
            errors.append("pool_size must be >= 1")

        if self.pool_size > 32:
            logger.warning(
                f"pool_size={self.pool_size} is very high, "
                f"may cause excessive memory usage"
            )

        if self.worker_recycle_after_tasks < 0:
            errors.append("worker_recycle_after_tasks must be >= 0")

        if self.worker_max_age_seconds < 0:
            errors.append("worker_max_age_seconds must be >= 0")

        if self.worker_timeout_seconds < 1:
            errors.append("worker_timeout_seconds must be >= 1")

        # Resource Limits Validation
        if self.default_memory_mb < 128:
            errors.append("default_memory_mb must be >= 128")

        if self.default_memory_mb > 16384:
            logger.warning(
                f"default_memory_mb={self.default_memory_mb} is very high, "
                f"ensure system has enough RAM"
            )

        if self.default_cpu_percent < 1:
            errors.append("default_cpu_percent must be >= 1")

        if self.default_cpu_percent > 800:
            logger.warning(
                f"default_cpu_percent={self.default_cpu_percent} exceeds 8 cores, "
                f"ensure system has enough CPUs"
            )

        if self.default_timeout_seconds < 1:
            errors.append("default_timeout_seconds must be >= 1")

        if self.monitoring_interval_seconds < 0.1:
            errors.append("monitoring_interval_seconds must be >= 0.1")

        if self.monitoring_interval_seconds > 60:
            logger.warning(
                f"monitoring_interval_seconds={self.monitoring_interval_seconds} "
                f"is very high, monitoring will be imprecise"
            )

        # IPC Validation
        if self.ipc_timeout_seconds < 0.1:
            errors.append("ipc_timeout_seconds must be >= 0.1")

        if self.ipc_collection_timeout_seconds < self.ipc_timeout_seconds:
            errors.append(
                "ipc_collection_timeout_seconds must be >= ipc_timeout_seconds"
            )

        if self.max_message_size_kb < 1:
            errors.append("max_message_size_kb must be >= 1")

        if self.max_message_size_kb > 10240:
            logger.warning(
                f"max_message_size_kb={self.max_message_size_kb} (>10MB) "
                f"may cause performance issues"
            )

        # Performance Validation
        if self.metrics_retention_count < 10:
            errors.append("metrics_retention_count must be >= 10")

        if self.metrics_retention_count > 100000:
            logger.warning(
                f"metrics_retention_count={self.metrics_retention_count} "
                f"may cause excessive memory usage"
            )

        # Context Sync Validation
        valid_strategies = {"incremental", "full", "snapshot"}
        if self.context_sync_strategy not in valid_strategies:
            errors.append(
                f"context_sync_strategy must be one of {valid_strategies}, "
                f"got '{self.context_sync_strategy}'"
            )

        valid_resolutions = {"last_write_wins", "merge", "custom"}
        if self.context_conflict_resolution not in valid_resolutions:
            errors.append(
                f"context_conflict_resolution must be one of {valid_resolutions}, "
                f"got '{self.context_conflict_resolution}'"
            )

        if self.context_snapshot_interval < 1:
            errors.append("context_snapshot_interval must be >= 1")

        if self.context_max_history < 1:
            errors.append("context_max_history must be >= 1")

        if self.context_max_history > 100:
            logger.warning(
                f"context_max_history={self.context_max_history} is very high, "
                f"may cause excessive memory usage"
            )


        # Agent Loading Validation (PERF-015)
        if self.agent_cache_ttl_hours < 0:
            errors.append("agent_cache_ttl_hours must be >= 0")

        if self.agent_cache_ttl_hours > 8760:  # 1 year
            logger.warning(
                f"agent_cache_ttl_hours={self.agent_cache_ttl_hours} is very high "
                f"(>1 year), cache may become stale"
            )

        if self.max_concurrent_agent_loads < 1:
            errors.append("max_concurrent_agent_loads must be >= 1")

        if self.max_concurrent_agent_loads > 100:
            logger.warning(
                f"max_concurrent_agent_loads={self.max_concurrent_agent_loads} "
                f"is very high, may cause resource exhaustion"
            )

                # Raise all validation errors
        if errors:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug("Runtime configuration validated successfully")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RuntimeConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RuntimeConfig instance with loaded values

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or contains bad values

        Example:
            >>> config = RuntimeConfig.from_yaml("config.yml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info(f"Loading configuration from {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                data = {}

            if not isinstance(data, dict):
                raise ValueError(f"Expected YAML dict, got {type(data)}")

            return cls.from_dict(data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        """
        Load configuration from dictionary.

        Args:
            data: Dictionary with configuration values
                 (keys match field names, unknown keys ignored)

        Returns:
            RuntimeConfig instance with loaded values

        Raises:
            ValueError: If values are invalid

        Example:
            >>> config = RuntimeConfig.from_dict({
            ...     "pool_size": 8,
            ...     "default_memory_mb": 4096
            ... })
        """
        # Extract only known fields
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}

        # Log ignored keys
        ignored = set(data.keys()) - field_names
        if ignored:
            logger.debug(f"Ignoring unknown config keys: {ignored}")

        logger.debug(f"Creating config from dict with {len(filtered)} fields")

        try:
            return cls(**filtered)
        except TypeError as e:
            raise ValueError(f"Invalid configuration values: {e}")

    @classmethod
    def from_env(cls, prefix: str = "AI_RUNTIME_") -> "RuntimeConfig":
        """
        Load configuration from environment variables.

        Environment variable names are constructed as:
        {prefix}{FIELD_NAME_UPPER}

        Examples:
            AI_RUNTIME_POOL_SIZE=8
            AI_RUNTIME_DEFAULT_MEMORY_MB=4096
            AI_RUNTIME_ENABLE_METRICS_COLLECTION=false

        Args:
            prefix: Prefix for environment variable names
                   (default: "AI_RUNTIME_")

        Returns:
            RuntimeConfig instance with values from environment
            (falls back to defaults for missing variables)

        Example:
            >>> # Set environment variables
            >>> os.environ["AI_RUNTIME_POOL_SIZE"] = "8"
            >>> config = RuntimeConfig.from_env()
        """
        logger.info(f"Loading configuration from environment (prefix: {prefix})")

        data: Dict[str, Any] = {}
        loaded_count = 0

        for field_info in fields(cls):
            env_key = f"{prefix}{field_info.name.upper()}"

            if env_key in os.environ:
                raw_value = os.environ[env_key]

                # Type conversion
                try:
                    if field_info.type == bool:
                        # Parse bool from string
                        data[field_info.name] = raw_value.lower() in (
                            "true", "1", "yes", "on"
                        )
                    elif field_info.type == int:
                        data[field_info.name] = int(raw_value)
                    elif field_info.type == float:
                        data[field_info.name] = float(raw_value)
                    else:
                        data[field_info.name] = raw_value

                    loaded_count += 1
                    logger.debug(f"Loaded {env_key}={raw_value}")

                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse {env_key}={raw_value} as "
                        f"{field_info.type}: {e}"
                    )

        logger.info(f"Loaded {loaded_count} config values from environment")

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration values

        Example:
            >>> config = RuntimeConfig()
            >>> config_dict = config.to_dict()
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output file path

        Raises:
            ImportError: If PyYAML not installed

        Example:
            >>> config = RuntimeConfig(pool_size=8)
            >>> config.to_yaml("config.yml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )

        path = Path(path)
        logger.info(f"Saving configuration to {path}")

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False
            )

        logger.debug(f"Configuration saved to {path}")

    def __repr__(self) -> str:
        """String representation showing key configuration values."""
        return (
            f"RuntimeConfig("
            f"pool_size={self.pool_size}, "
            f"memory_mb={self.default_memory_mb}, "
            f"timeout_s={self.default_timeout_seconds}, "
            f"monitoring={self.enable_resource_monitoring}, "
            f"context_sync={self.context_sync_strategy}"
            f")"
        )
