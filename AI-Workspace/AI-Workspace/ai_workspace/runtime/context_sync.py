#!/usr/bin/env python3
"""
Context synchronization for concurrent agent access.

Provides optimistic locking, conflict detection, and resolution
for concurrent updates to shared context.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .context_store import ContextStore

logger = logging.getLogger(__name__)


@dataclass
class Lock:
    """Represents an optimistic lock on a resource."""

    resource_key: str
    agent_id: str
    version: int
    acquired_at: datetime
    lock_id: str


@dataclass
class Conflict:
    """Represents a conflict between concurrent updates."""

    resource_key: str
    agent1: str
    agent2: str
    version1: int
    version2: int
    value1: Any
    value2: Any


@dataclass
class Resolution:
    """Result of conflict resolution."""

    winner: str
    final_value: Any
    strategy: str  # "last-write-wins", "merge", "manual"


class ContextSynchronizer:
    """
    Manage concurrent context updates across agents.

    Features:
    - Optimistic locking (non-blocking)
    - Conflict detection
    - Automatic resolution strategies
    - Version tracking
    - Checkpoint/rollback support

    Example:
        >>> sync = ContextSynchronizer(store)
        >>> if sync.acquire_lock("config", "agent-1"):
        ...     # Update context
        ...     store.set_project_config("key", "value")
        ...     sync.release_lock("config", "agent-1")
    """

    def __init__(self, store: ContextStore):
        """
        Initialize synchronizer.

        Args:
            store: ContextStore instance to synchronize
        """
        self.store = store
        self.locks: Dict[str, Lock] = {}
        self.versions: Dict[str, int] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()  # Protect internal state

        logger.info("ContextSynchronizer initialized")

    def acquire_lock(
        self,
        resource_key: str,
        agent_id: str,
        timeout: int = 10
    ) -> bool:
        """
        Acquire optimistic lock on resource.

        Args:
            resource_key: Resource identifier
            agent_id: Agent requesting lock
            timeout: Lock timeout in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        # Check if already locked
        if resource_key in self.locks:
            existing_lock = self.locks[resource_key]

            # Check if lock expired
            age = (datetime.now() - existing_lock.acquired_at).total_seconds()
            if age < timeout:
                logger.warning(
                    f"Resource {resource_key} already locked by "
                    f"{existing_lock.agent_id}"
                )
                return False

            # Lock expired, remove it
            logger.info(f"Lock on {resource_key} expired, removing")
            del self.locks[resource_key]

        # Get current version
        current_version = self.versions.get(resource_key, 0)

        # Create new lock
        lock = Lock(
            resource_key=resource_key,
            agent_id=agent_id,
            version=current_version,
            acquired_at=datetime.now(),
            lock_id=uuid4().hex
        )

        self.locks[resource_key] = lock

        logger.debug(
            f"Lock acquired: {resource_key} by {agent_id} "
            f"(version={current_version})"
        )

        return True

    def release_lock(self, resource_key: str, agent_id: str) -> bool:
        """
        Release lock on resource.

        Args:
            resource_key: Resource identifier
            agent_id: Agent releasing lock

        Returns:
            True if lock released, False if agent didn't own lock
        """
        if resource_key not in self.locks:
            logger.warning(f"No lock on {resource_key} to release")
            return False

        lock = self.locks[resource_key]

        if lock.agent_id != agent_id:
            logger.error(
                f"Agent {agent_id} cannot release lock owned by {lock.agent_id}"
            )
            return False

        # Increment version on release
        self.versions[resource_key] = lock.version + 1

        del self.locks[resource_key]

        logger.debug(f"Lock released: {resource_key} by {agent_id}")

        return True

    def check_version(self, resource_key: str, expected_version: int) -> bool:
        """
        Check if resource version matches expected.

        Used for detecting concurrent modifications.

        Args:
            resource_key: Resource identifier
            expected_version: Expected version number

        Returns:
            True if versions match, False if conflict detected
        """
        current_version = self.versions.get(resource_key, 0)

        if current_version != expected_version:
            logger.warning(
                f"Version conflict on {resource_key}: "
                f"expected={expected_version}, actual={current_version}"
            )
            return False

        return True

    def detect_conflict(
        self,
        resource_key: str,
        agent_id: str,
        new_value: Any
    ) -> Optional[Conflict]:
        """
        Detect if update would cause a conflict.

        Args:
            resource_key: Resource identifier
            agent_id: Agent attempting update
            new_value: New value to write

        Returns:
            Conflict object if conflict detected, None otherwise
        """
        if resource_key not in self.locks:
            return None  # No conflict if not locked

        lock = self.locks[resource_key]

        # Check if version changed since lock acquired
        current_version = self.versions.get(resource_key, 0)

        if current_version != lock.version:
            # Concurrent modification detected
            # Get the conflicting value (this is simplified)
            # In production, would fetch actual value from store

            return Conflict(
                resource_key=resource_key,
                agent1=lock.agent_id,
                agent2=agent_id,
                version1=lock.version,
                version2=current_version,
                value1=None,  # Would fetch from store
                value2=new_value
            )

        return None

    def resolve_conflict(self, conflict: Conflict) -> Resolution:
        """
        Resolve conflict using strategy.

        Current strategy: Last-write-wins

        Args:
            conflict: Conflict to resolve

        Returns:
            Resolution with winner and final value
        """
        # Simple last-write-wins strategy
        # Future: Could implement CRDT, vector clocks, manual resolution

        logger.info(
            f"Resolving conflict on {conflict.resource_key} "
            f"between {conflict.agent1} and {conflict.agent2}"
        )

        # Choose the one with higher version (later write)
        if conflict.version2 > conflict.version1:
            winner = conflict.agent2
            final_value = conflict.value2
        else:
            winner = conflict.agent1
            final_value = conflict.value1

        return Resolution(
            winner=winner,
            final_value=final_value,
            strategy="last-write-wins"
        )

    def create_checkpoint(self, checkpoint_id: str) -> None:
        """
        Create checkpoint of current state.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        # Snapshot current project config
        project_config = self.store.get_all_project_config()

        self.checkpoints[checkpoint_id] = {
            "project_config": project_config,
            "versions": self.versions.copy(),
            "created_at": datetime.now().isoformat()
        }

        logger.info(f"Checkpoint created: {checkpoint_id}")

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to previous checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore

        Returns:
            True if rollback successful, False otherwise
        """
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        checkpoint = self.checkpoints[checkpoint_id]

        # Restore project config
        project_config = checkpoint["project_config"]
        for key, value in project_config.items():
            self.store.set_project_config(key, value)

        # Restore versions
        self.versions = checkpoint["versions"].copy()

        logger.info(f"Rolled back to checkpoint: {checkpoint_id}")

        return True

    def get_lock_status(self, resource_key: str) -> Optional[Dict[str, Any]]:
        """
        Get current lock status for resource.

        Args:
            resource_key: Resource identifier

        Returns:
            Lock info dict or None if not locked
        """
        if resource_key not in self.locks:
            return None

        lock = self.locks[resource_key]

        return {
            "resource_key": lock.resource_key,
            "agent_id": lock.agent_id,
            "version": lock.version,
            "acquired_at": lock.acquired_at.isoformat(),
            "lock_id": lock.lock_id,
            "age_seconds": (datetime.now() - lock.acquired_at).total_seconds()
        }

    def get_all_locks(self) -> List[Dict[str, Any]]:
        """
        Get all active locks.

        Returns:
            List of lock info dicts
        """
        return [
            self.get_lock_status(key)
            for key in self.locks.keys()
        ]

    def force_release_lock(self, resource_key: str) -> bool:
        """
        Force release a lock (emergency use).

        Args:
            resource_key: Resource identifier

        Returns:
            True if lock released
        """
        if resource_key in self.locks:
            agent_id = self.locks[resource_key].agent_id
            del self.locks[resource_key]
            logger.warning(
                f"Force released lock on {resource_key} "
                f"(was held by {agent_id})"
            )
            return True

        return False

    def cleanup_expired_locks(self, timeout: int = 600) -> int:
        """
        Remove locks older than timeout.

        Args:
            timeout: Lock age threshold in seconds

        Returns:
            Number of locks removed
        """
        now = datetime.now()
        expired = []

        for key, lock in self.locks.items():
            age = (now - lock.acquired_at).total_seconds()
            if age > timeout:
                expired.append(key)

        for key in expired:
            agent_id = self.locks[key].agent_id
            del self.locks[key]
            logger.warning(
                f"Removed expired lock on {key} "
                f"(was held by {agent_id})"
            )

        return len(expired)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ContextSynchronizer(locks={len(self.locks)}, "
            f"checkpoints={len(self.checkpoints)})"
        )
