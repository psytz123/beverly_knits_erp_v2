"""
Cross-Platform Resource Limit Enforcement.

This module provides platform-aware resource enforcement for agent processes,
ensuring agents stay within memory, CPU, and time limits.

Supports:
    - Linux: cgroups v2 (best - kernel-level enforcement)
    - macOS: setrlimit (good - OS-level limits)
    - Windows: psutil polling (adequate - monitoring with termination)

Classes:
    ResourceEnforcer: Main enforcer with platform detection
    CgroupEnforcer: Linux cgroups implementation
    RlimitEnforcer: macOS/BSD setrlimit implementation
    PollingEnforcer: Windows fallback implementation

Example:
    >>> enforcer = ResourceEnforcer()
    >>> limits = ResourceLimits(memory_mb=512, cpu_percent=50)
    >>> await enforcer.enforce(process, limits)
"""

import asyncio
import logging
import platform
import sys
from abc import ABC, abstractmethod
from multiprocessing import Process
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

from .interfaces import ResourceLimits

logger = logging.getLogger(__name__)


class BaseEnforcer(ABC):
    """Abstract base class for resource enforcers."""

    @abstractmethod
    async def enforce(self, process: Process, limits: ResourceLimits) -> None:
        """
        Enforce resource limits on process.

        Args:
            process: Process to monitor and limit
            limits: Resource constraints to enforce

        Raises:
            ResourceLimitError: If enforcement fails
        """
        pass


class PollingEnforcer(BaseEnforcer):
    """
    Windows fallback - poll resource usage and terminate if exceeded.

    Uses psutil to monitor process memory and CPU usage. When limits are
    exceeded, sends SIGTERM (graceful) then SIGKILL (forceful) if needed.

    This is less efficient than kernel-level enforcement but works reliably
    on Windows where cgroups and setrlimit are not available.
    """

    def __init__(self) -> None:
        """Initialize polling enforcer."""
        if psutil is None:
            raise ImportError("psutil is required for PollingEnforcer. Install with: pip install psutil")
        self.poll_interval_seconds = 1.0  # Poll every second

    async def enforce(self, process: Process, limits: ResourceLimits) -> None:
        """
        Poll resource usage and terminate if limits exceeded.

        Args:
            process: Process to monitor
            limits: Resource limits to enforce

        Note:
            This runs concurrently with the process until it exits or
            is terminated for exceeding limits.
        """
        logger.info(
            f"Starting resource monitoring for PID {process.pid} "
            f"(max memory: {limits.memory_mb}MB, poll interval: {self.poll_interval_seconds}s)"
        )

        while process.is_alive():
            try:
                proc = psutil.Process(process.pid)
                memory_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_percent = proc.cpu_percent(interval=0.1)

                # Check memory limit
                if memory_mb > limits.memory_mb:
                    logger.warning(
                        f"Process {process.pid} exceeds memory limit: "
                        f"{memory_mb:.1f}MB > {limits.memory_mb}MB - terminating"
                    )
                    await self._graceful_terminate(process)
                    break

                # Check CPU limit (optional enforcement - log warning only)
                if cpu_percent > limits.cpu_percent:
                    logger.debug(
                        f"Process {process.pid} CPU usage high: "
                        f"{cpu_percent:.1f}% > {limits.cpu_percent}%"
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process died or we lost access
                logger.debug(f"Process {process.pid} no longer accessible")
                break
            except Exception as e:
                logger.error(f"Error monitoring process {process.pid}: {e}")
                break

            await asyncio.sleep(self.poll_interval_seconds)

        logger.debug(f"Resource monitoring stopped for PID {process.pid}")

    async def _graceful_terminate(self, process: Process) -> None:
        """
        Gracefully terminate process (SIGTERM then SIGKILL).

        Args:
            process: Process to terminate
        """
        try:
            # Try graceful shutdown first
            process.terminate()
            logger.debug(f"Sent SIGTERM to process {process.pid}")

            # Wait up to 5 seconds for graceful exit
            for _ in range(50):
                if not process.is_alive():
                    logger.info(f"Process {process.pid} terminated gracefully")
                    return
                await asyncio.sleep(0.1)

            # Force kill if still alive
            if process.is_alive():
                logger.warning(f"Process {process.pid} did not terminate gracefully, sending SIGKILL")
                process.kill()
                logger.info(f"Process {process.pid} killed forcefully")

        except Exception as e:
            logger.error(f"Error terminating process {process.pid}: {e}")


class RlimitEnforcer(BaseEnforcer):
    """
    macOS/BSD implementation using setrlimit.

    Uses resource.setrlimit() to set hard limits on memory and CPU time.
    This is enforced by the OS kernel and is more efficient than polling.

    Note:
        CPU limits are enforced via RLIMIT_CPU (total CPU seconds), not percentage.
    """

    def __init__(self) -> None:
        """Initialize rlimit enforcer."""
        try:
            import resource
            self.resource = resource
        except ImportError:
            raise ImportError("resource module not available on this platform")

    async def enforce(self, process: Process, limits: ResourceLimits) -> None:
        """
        Enforce limits using setrlimit (runs in child process).

        Args:
            process: Process to limit
            limits: Resource limits

        Note:
            This must be called BEFORE the process starts, as it sets
            limits that the child process inherits.
        """
        # Convert MB to bytes for memory limit
        memory_bytes = limits.memory_mb * 1024 * 1024

        # Convert timeout to CPU seconds (conservative estimate)
        cpu_seconds = limits.timeout_seconds

        logger.info(
            f"Setting resource limits for process: "
            f"memory={limits.memory_mb}MB, cpu_time={cpu_seconds}s"
        )

        # These limits are applied in the child process via preexec_fn
        # We create a closure to set limits after fork but before exec
        def set_limits() -> None:
            """Set resource limits in child process."""
            try:
                # Set memory limit (address space)
                self.resource.setrlimit(
                    self.resource.RLIMIT_AS,
                    (memory_bytes, memory_bytes)
                )

                # Set CPU time limit
                self.resource.setrlimit(
                    self.resource.RLIMIT_CPU,
                    (cpu_seconds, cpu_seconds)
                )

                logger.debug(f"Resource limits set: AS={memory_bytes}, CPU={cpu_seconds}s")

            except Exception as e:
                logger.error(f"Failed to set resource limits: {e}")
                raise

        # Store the limit setter for use during process spawn
        if not hasattr(process, '_limit_setter'):
            process._limit_setter = set_limits  # type: ignore

        # Note: Actual enforcement happens when process.start() is called with preexec_fn


class CgroupEnforcer(BaseEnforcer):
    """
    Linux implementation using cgroups v2.

    Provides kernel-level resource enforcement using Linux control groups.
    This is the most robust approach, preventing processes from exceeding
    limits at the kernel level.

    Requires:
        - Linux kernel with cgroups v2 enabled
        - Write access to /sys/fs/cgroup
        - Root or CAP_SYS_ADMIN capability (optional, falls back to user cgroups)
    """

    def __init__(self) -> None:
        """Initialize cgroup enforcer."""
        self.cgroup_root = Path("/sys/fs/cgroup")
        self.user_cgroup_base = Path.home() / ".ai-workspace" / "cgroups"

        # Check if cgroups v2 is available
        if not self.cgroup_root.exists():
            raise RuntimeError("cgroups not available on this system")

        logger.debug("CgroupEnforcer initialized")

    async def enforce(self, process: Process, limits: ResourceLimits) -> None:
        """
        Enforce limits using cgroups v2.

        Args:
            process: Process to limit
            limits: Resource limits

        Creates a dedicated cgroup for the process and sets memory and CPU limits.
        """
        cgroup_name = f"ai-agent-{process.pid}"
        cgroup_path = self._get_cgroup_path(cgroup_name)

        try:
            # Create cgroup
            await self._create_cgroup(cgroup_path)

            # Set memory limit
            await self._set_memory_limit(cgroup_path, limits.memory_mb)

            # Set CPU limit (optional - may not be supported everywhere)
            try:
                await self._set_cpu_limit(cgroup_path, limits.cpu_percent)
            except Exception as e:
                logger.warning(f"Could not set CPU limit: {e}")

            # Add process to cgroup
            await self._add_process(cgroup_path, process.pid)

            logger.info(
                f"Process {process.pid} added to cgroup '{cgroup_name}' "
                f"with limits: memory={limits.memory_mb}MB, cpu={limits.cpu_percent}%"
            )

            # Monitor for OOM kills (runs until process exits)
            await self._monitor_cgroup(cgroup_path, process)

        finally:
            # Cleanup cgroup when done
            await self._cleanup_cgroup(cgroup_path)

    def _get_cgroup_path(self, name: str) -> Path:
        """
        Get path to cgroup directory.

        Args:
            name: Cgroup name

        Returns:
            Path to cgroup directory
        """
        # Try system cgroup first (requires root)
        system_path = self.cgroup_root / "ai-workspace" / name
        if self._can_write(self.cgroup_root):
            return system_path

        # Fall back to user cgroup
        return self.user_cgroup_base / name

    def _can_write(self, path: Path) -> bool:
        """Check if we can write to path."""
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except (PermissionError, OSError):
            return False

    async def _create_cgroup(self, cgroup_path: Path) -> None:
        """Create cgroup directory."""
        cgroup_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created cgroup: {cgroup_path}")

    async def _set_memory_limit(self, cgroup_path: Path, memory_mb: int) -> None:
        """Set memory limit in cgroup."""
        memory_bytes = memory_mb * 1024 * 1024
        memory_max_file = cgroup_path / "memory.max"

        memory_max_file.write_text(str(memory_bytes))
        logger.debug(f"Set memory limit: {memory_mb}MB")

    async def _set_cpu_limit(self, cgroup_path: Path, cpu_percent: int) -> None:
        """Set CPU limit in cgroup."""
        # CPU quota is specified in microseconds per 100ms period
        # e.g., 50% = 50000 (50ms per 100ms period)
        period_us = 100000  # 100ms
        quota_us = int(period_us * (cpu_percent / 100.0))

        cpu_max_file = cgroup_path / "cpu.max"
        cpu_max_file.write_text(f"{quota_us} {period_us}")
        logger.debug(f"Set CPU limit: {cpu_percent}%")

    async def _add_process(self, cgroup_path: Path, pid: int) -> None:
        """Add process to cgroup."""
        procs_file = cgroup_path / "cgroup.procs"
        procs_file.write_text(str(pid))
        logger.debug(f"Added PID {pid} to cgroup")

    async def _monitor_cgroup(self, cgroup_path: Path, process: Process) -> None:
        """
        Monitor cgroup for OOM events.

        Args:
            cgroup_path: Path to cgroup
            process: Process being monitored
        """
        events_file = cgroup_path / "memory.events"

        while process.is_alive():
            try:
                if events_file.exists():
                    events = events_file.read_text()
                    if "oom_kill" in events:
                        logger.error(f"Process {process.pid} killed by OOM killer")
                        break
            except Exception as e:
                logger.debug(f"Error reading memory.events: {e}")

            await asyncio.sleep(1.0)

    async def _cleanup_cgroup(self, cgroup_path: Path) -> None:
        """Remove cgroup directory."""
        try:
            if cgroup_path.exists():
                cgroup_path.rmdir()
                logger.debug(f"Removed cgroup: {cgroup_path}")
        except Exception as e:
            logger.warning(f"Could not remove cgroup {cgroup_path}: {e}")


class ResourceEnforcer:
    """
    Platform-aware resource limit enforcement.

    Automatically selects the best enforcement strategy based on the OS:
    - Linux: CgroupEnforcer (kernel-level enforcement)
    - macOS: RlimitEnforcer (OS-level limits)
    - Windows: PollingEnforcer (monitoring with termination)

    Example:
        >>> enforcer = ResourceEnforcer()
        >>> limits = ResourceLimits(memory_mb=512, cpu_percent=50)
        >>> await enforcer.enforce(process, limits)
    """

    def __init__(self) -> None:
        """Initialize resource enforcer with platform detection."""
        self.platform = platform.system()
        self.enforcer = self._create_enforcer()

        logger.info(
            f"ResourceEnforcer initialized for {self.platform} "
            f"using {self.enforcer.__class__.__name__}"
        )

    def _create_enforcer(self) -> BaseEnforcer:
        """
        Create platform-specific enforcer.

        Returns:
            BaseEnforcer instance appropriate for current platform

        Raises:
            RuntimeError: If no suitable enforcer is available
        """
        if self.platform == "Linux":
            try:
                return CgroupEnforcer()
            except Exception as e:
                logger.warning(f"Could not initialize CgroupEnforcer: {e}, falling back to polling")
                return PollingEnforcer()

        elif self.platform == "Darwin":  # macOS
            try:
                return RlimitEnforcer()
            except Exception as e:
                logger.warning(f"Could not initialize RlimitEnforcer: {e}, falling back to polling")
                return PollingEnforcer()

        else:  # Windows and others
            return PollingEnforcer()

    async def enforce(self, process: Process, limits: ResourceLimits) -> None:
        """
        Apply resource limits to process.

        Args:
            process: Process to limit
            limits: Resource constraints to enforce

        Raises:
            ResourceLimitError: If enforcement fails
        """
        await self.enforcer.enforce(process, limits)
