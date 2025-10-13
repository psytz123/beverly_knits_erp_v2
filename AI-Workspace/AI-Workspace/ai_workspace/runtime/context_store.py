#!/usr/bin/env python3
"""
Multi-tier context store for agent state management.

Provides centralized state storage accessible to all agents with
caching, versioning, and concurrent access support.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class ContextStore:
    """
    Centralized state management for multi-agent workflows.

    Provides 4 tiers of context:
    1. Project context (global config, long-lived)
    2. Task context (current goals, medium-lived)
    3. Agent context (agent status, short-lived)
    4. Artifact registry (code, docs, permanent)

    Features:
    - SQL ite-based persistence
    - TTL caching for performance
    - Versioning for rollback
    - Concurrent access support

    Example:
        >>> store = ContextStore()
        >>> store.set_project_config("language", "Python")
        >>> lang = store.get_project_config("language")
        >>> print(lang)  # "Python"
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize context store.

        Args:
            db_path: Path to SQLite database
                    (defaults to .ai-workspace/context.db)
        """
        if db_path is None:
            workspace = Path(__file__).parent.parent.parent / ".ai-workspace"
            db_path = str(workspace / "context.db")

        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL

        self._create_tables()
        logger.info(f"ContextStore initialized with database: {db_path}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.db.cursor()

        # Project context (global config)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_context (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Task state (current tasks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_state (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Agent status (agent availability)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_status (
                agent_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                current_task TEXT,
                workload INT DEFAULT 0,
                last_heartbeat TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Artifact registry (code, docs, results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                path TEXT,
                content TEXT,
                metadata TEXT,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Version history (for rollback)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_history (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_status
            ON task_state(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_status
            ON agent_status(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_type
            ON artifacts(type)
        """)

        self.db.commit()
        logger.debug("Database tables created/verified")

    # ========== Project Context ==========

    def set_project_config(self, key: str, value: Any) -> None:
        """
        Set project-level configuration.

        Args:
            key: Configuration key
            value: Configuration value (will be JSON-encoded)
        """
        value_json = json.dumps(value)
        cache_key = f"project:{key}"

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO project_context (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
        """, (key, value_json))

        self.db.commit()
        self.cache[cache_key] = value

        logger.debug(f"Set project config: {key} = {value}")

    def get_project_config(self, key: str) -> Optional[Any]:
        """
        Get project-level configuration.

        Args:
            key: Configuration key

        Returns:
            Configuration value, or None if not found
        """
        cache_key = f"project:{key}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Query database
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT value FROM project_context WHERE key = ?",
            (key,)
        )

        row = cursor.fetchone()
        if row:
            value = json.loads(row[0])
            self.cache[cache_key] = value
            return value

        return None

    def get_all_project_config(self) -> Dict[str, Any]:
        """
        Get all project configuration.

        Returns:
            Dictionary of all project config
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT key, value FROM project_context")

        result = {}
        for key, value_json in cursor.fetchall():
            result[key] = json.loads(value_json)

        return result

    # ========== Task Context ==========

    def set_task_state(
        self,
        task_id: str,
        status: str,
        progress: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update task state.

        Args:
            task_id: Task identifier
            status: Task status (pending, in_progress, completed, failed)
            progress: Progress percentage (0.0 to 1.0)
            metadata: Additional task metadata
        """
        metadata_json = json.dumps(metadata or {})

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO task_state (task_id, status, progress, metadata, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(task_id) DO UPDATE SET
                status = excluded.status,
                progress = excluded.progress,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
        """, (task_id, status, progress, metadata_json))

        self.db.commit()

        logger.debug(f"Updated task {task_id}: status={status}, progress={progress}")

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task state.

        Args:
            task_id: Task identifier

        Returns:
            Task state dict or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT status, progress, metadata, created_at, updated_at
            FROM task_state WHERE task_id = ?
        """, (task_id,))

        row = cursor.fetchone()
        if row:
            return {
                "task_id": task_id,
                "status": row[0],
                "progress": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "created_at": row[3],
                "updated_at": row[4]
            }

        return None

    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all tasks with specific status.

        Args:
            status: Task status to filter by

        Returns:
            List of task state dicts
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT task_id, status, progress, metadata, created_at, updated_at
            FROM task_state WHERE status = ?
        """, (status,))

        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "task_id": row[0],
                "status": row[1],
                "progress": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
                "updated_at": row[5]
            })

        return tasks

    # ========== Agent Context ==========

    def update_agent_status(
        self,
        agent_id: str,
        status: str,
        current_task: Optional[str] = None,
        workload: int = 0
    ) -> None:
        """
        Update agent status.

        Args:
            agent_id: Agent identifier
            status: Agent status (idle, busy, error, offline)
            current_task: Current task ID (if busy)
            workload: Number of queued tasks
        """
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO agent_status (
                agent_id, status, current_task, workload,
                last_heartbeat, updated_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(agent_id) DO UPDATE SET
                status = excluded.status,
                current_task = excluded.current_task,
                workload = excluded.workload,
                last_heartbeat = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        """, (agent_id, status, current_task, workload))

        self.db.commit()

        logger.debug(f"Updated agent {agent_id}: status={status}, workload={workload}")

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent status.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent status dict or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT status, current_task, workload, last_heartbeat, updated_at
            FROM agent_status WHERE agent_id = ?
        """, (agent_id,))

        row = cursor.fetchone()
        if row:
            return {
                "agent_id": agent_id,
                "status": row[0],
                "current_task": row[1],
                "workload": row[2],
                "last_heartbeat": row[3],
                "updated_at": row[4]
            }

        return None

    def get_available_agents(self) -> List[str]:
        """
        Get list of idle agents.

        Returns:
            List of agent IDs with status='idle'
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT agent_id FROM agent_status
            WHERE status = 'idle'
            ORDER BY workload ASC
        """)

        return [row[0] for row in cursor.fetchall()]

    # ========== Artifact Registry ==========

    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        path: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> None:
        """
        Register an artifact (code, docs, test results, etc.).

        Args:
            artifact_id: Unique artifact identifier
            artifact_type: Type (code, docs, test_results, etc.)
            path: File path (optional)
            content: Artifact content (optional)
            metadata: Additional metadata
            created_by: Agent that created it
        """
        metadata_json = json.dumps(metadata or {})

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO artifacts (
                artifact_id, type, path, content, metadata, created_by, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(artifact_id) DO UPDATE SET
                type = excluded.type,
                path = excluded.path,
                content = excluded.content,
                metadata = excluded.metadata
        """, (artifact_id, artifact_type, path, content, metadata_json, created_by))

        self.db.commit()

        logger.debug(f"Registered artifact: {artifact_id} (type={artifact_type})")

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Get artifact by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Artifact dict or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT type, path, content, metadata, created_by, created_at
            FROM artifacts WHERE artifact_id = ?
        """, (artifact_id,))

        row = cursor.fetchone()
        if row:
            return {
                "artifact_id": artifact_id,
                "type": row[0],
                "path": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_by": row[4],
                "created_at": row[5]
            }

        return None

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts of specific type.

        Args:
            artifact_type: Artifact type to filter by

        Returns:
            List of artifact dicts
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT artifact_id, type, path, content, metadata, created_by, created_at
            FROM artifacts WHERE type = ?
            ORDER BY created_at DESC
        """, (artifact_type,))

        artifacts = []
        for row in cursor.fetchall():
            artifacts.append({
                "artifact_id": row[0],
                "type": row[1],
                "path": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_by": row[5],
                "created_at": row[6]
            })

        return artifacts

    # ========== Utilities ==========

    def clear_all(self) -> None:
        """Clear all data from the store (for testing)."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM project_context")
        cursor.execute("DELETE FROM task_state")
        cursor.execute("DELETE FROM agent_status")
        cursor.execute("DELETE FROM artifacts")
        cursor.execute("DELETE FROM version_history")
        self.db.commit()
        self.cache.clear()

        logger.warning("Cleared all context data")

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
        logger.info("ContextStore closed")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ContextStore(db_path={self.db_path})"
