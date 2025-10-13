#!/usr/bin/env python3
"""
Performance tracker for agent execution metrics.

Tracks agent performance over time to enable intelligent agent selection
based on historical success rates, execution speed, and quality metrics.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Aggregated agent performance metrics."""

    agent_id: str
    task_type: Optional[str]
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float  # 0.0-1.0
    avg_duration_seconds: float
    avg_quality_score: float  # 0.0-1.0
    avg_cpu_percent: float
    avg_memory_mb: float


class PerformanceTracker:
    """
    Track and analyze agent performance over time.

    Provides SQLite-based persistence for agent execution metrics,
    enabling data-driven agent selection and performance optimization.

    Features:
    - Execution tracking with resource metrics
    - Quality scoring and user feedback
    - Historical performance analysis
    - Best agent recommendations
    - Automatic data cleanup

    Example:
        >>> tracker = PerformanceTracker()
        >>> tracker.record_execution(
        ...     agent_id="backend-developer",
        ...     task_id="task-123",
        ...     task_type="api_development",
        ...     task_name="Build REST API",
        ...     start_time=datetime.now() - timedelta(minutes=5),
        ...     end_time=datetime.now(),
        ...     success=True,
        ...     output_quality_score=0.85
        ... )
        >>> metrics = tracker.get_agent_metrics("backend-developer")
        >>> print(f"Success rate: {metrics.success_rate:.1%}")
    """

    def __init__(self, db_path: str = ".ai-workspace/metrics.db"):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database
                    (defaults to .ai-workspace/metrics.db)
        """
        # Handle absolute and relative paths
        if not Path(db_path).is_absolute():
            workspace = Path(__file__).parent.parent.parent / ".ai-workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            db_path = str(workspace / "metrics.db")

        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)

        self._create_tables()
        logger.info(f"PerformanceTracker initialized with database: {db_path}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.db.cursor()

        # Agent executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_name TEXT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds INTEGER,
                success BOOLEAN NOT NULL,
                exit_code INTEGER,
                cpu_percent REAL,
                memory_mb REAL,
                output_quality_score REAL,
                user_corrections INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_task
            ON agent_executions(agent_id, task_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON agent_executions(start_time)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success
            ON agent_executions(success)
        """)

        self.db.commit()
        logger.debug("Database tables created/verified")

    def record_execution(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        task_name: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        exit_code: int = 0,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        output_quality_score: float = 0.0,
        user_corrections: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Record single agent execution.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            task_type: Type of task (e.g., "api_development", "database_migration")
            task_name: Human-readable task name
            start_time: Execution start timestamp
            end_time: Execution end timestamp
            success: Whether execution succeeded
            exit_code: Process exit code (0 = success)
            cpu_percent: Average CPU usage during execution
            memory_mb: Peak memory usage in MB
            output_quality_score: Quality score 0.0-1.0
            user_corrections: Number of corrections needed
            error_message: Error message if failed
        """
        # Calculate duration
        duration_seconds = int((end_time - start_time).total_seconds())

        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT INTO agent_executions (
                agent_id, task_id, task_type, task_name,
                start_time, end_time, duration_seconds,
                success, exit_code, cpu_percent, memory_mb,
                output_quality_score, user_corrections, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                task_id,
                task_type,
                task_name,
                start_time.isoformat(),
                end_time.isoformat(),
                duration_seconds,
                success,
                exit_code,
                cpu_percent,
                memory_mb,
                output_quality_score,
                user_corrections,
                error_message,
            ),
        )

        self.db.commit()

        logger.debug(
            f"Recorded execution: {agent_id} / {task_type} / "
            f"success={success} / duration={duration_seconds}s"
        )

    def get_agent_metrics(
        self, agent_id: str, task_type: Optional[str] = None, days: int = 30
    ) -> AgentMetrics:
        """
        Get aggregated metrics for agent.

        Args:
            agent_id: Agent identifier
            task_type: Optional task type filter
            days: Number of days to look back

        Returns:
            AgentMetrics with aggregated performance data
        """
        cursor = self.db.cursor()

        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Build query with optional task_type filter
        if task_type:
            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(duration_seconds) as avg_duration,
                    AVG(COALESCE(output_quality_score, 0)) as avg_quality,
                    AVG(COALESCE(cpu_percent, 0)) as avg_cpu,
                    AVG(COALESCE(memory_mb, 0)) as avg_memory
                FROM agent_executions
                WHERE agent_id = ? AND task_type = ? AND start_time >= ?
            """
            cursor.execute(query, (agent_id, task_type, cutoff_date))
        else:
            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(duration_seconds) as avg_duration,
                    AVG(COALESCE(output_quality_score, 0)) as avg_quality,
                    AVG(COALESCE(cpu_percent, 0)) as avg_cpu,
                    AVG(COALESCE(memory_mb, 0)) as avg_memory
                FROM agent_executions
                WHERE agent_id = ? AND start_time >= ?
            """
            cursor.execute(query, (agent_id, cutoff_date))

        row = cursor.fetchone()

        if row and row[0] > 0:
            total = row[0]
            successful = row[1] or 0
            failed = row[2] or 0
            success_rate = successful / total if total > 0 else 0.0

            return AgentMetrics(
                agent_id=agent_id,
                task_type=task_type,
                total_executions=total,
                successful_executions=successful,
                failed_executions=failed,
                success_rate=success_rate,
                avg_duration_seconds=row[3] or 0.0,
                avg_quality_score=row[4] or 0.0,
                avg_cpu_percent=row[5] or 0.0,
                avg_memory_mb=row[6] or 0.0,
            )
        else:
            # No data found
            return AgentMetrics(
                agent_id=agent_id,
                task_type=task_type,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                success_rate=0.0,
                avg_duration_seconds=0.0,
                avg_quality_score=0.0,
                avg_cpu_percent=0.0,
                avg_memory_mb=0.0,
            )

    def get_best_agent_for_task(self, task_type: str, top_n: int = 3) -> List[str]:
        """
        Get top N agents for task type by success rate and speed.

        Ranking algorithm:
        - Primary: Success rate (higher is better)
        - Secondary: Average duration (lower is better)
        - Tertiary: Average quality score (higher is better)

        Args:
            task_type: Type of task to find best agents for
            top_n: Number of top agents to return

        Returns:
            List of agent IDs ranked by performance
        """
        cursor = self.db.cursor()

        # Look back 30 days, require minimum 3 executions
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

        query = """
            SELECT
                agent_id,
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(duration_seconds) as avg_duration,
                AVG(COALESCE(output_quality_score, 0)) as avg_quality
            FROM agent_executions
            WHERE task_type = ? AND start_time >= ?
            GROUP BY agent_id
            HAVING total >= 3
            ORDER BY
                (CAST(successful AS REAL) / total) DESC,
                avg_duration ASC,
                avg_quality DESC
            LIMIT ?
        """

        cursor.execute(query, (task_type, cutoff_date, top_n))

        return [row[0] for row in cursor.fetchall()]

    def get_recent_failures(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent failed executions for analysis.

        Args:
            days: Number of days to look back

        Returns:
            List of failed execution dicts with details
        """
        cursor = self.db.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT
                agent_id, task_id, task_type, task_name,
                start_time, duration_seconds, exit_code, error_message
            FROM agent_executions
            WHERE success = 0 AND start_time >= ?
            ORDER BY start_time DESC
        """

        cursor.execute(query, (cutoff_date,))

        failures = []
        for row in cursor.fetchall():
            failures.append(
                {
                    "agent_id": row[0],
                    "task_id": row[1],
                    "task_type": row[2],
                    "task_name": row[3],
                    "start_time": row[4],
                    "duration_seconds": row[5],
                    "exit_code": row[6],
                    "error_message": row[7],
                }
            )

        return failures

    def clear_old_data(self, days: int = 90) -> int:
        """
        Delete executions older than N days.

        Args:
            days: Number of days to retain

        Returns:
            Count of deleted records
        """
        cursor = self.db.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            "DELETE FROM agent_executions WHERE start_time < ?", (cutoff_date,)
        )

        deleted_count = cursor.rowcount
        self.db.commit()

        logger.info(f"Deleted {deleted_count} executions older than {days} days")

        return deleted_count

    def get_all_agent_stats(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get performance statistics for all agents.

        Args:
            days: Number of days to look back

        Returns:
            List of dicts with agent statistics
        """
        cursor = self.db.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT
                agent_id,
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(duration_seconds) as avg_duration,
                AVG(COALESCE(output_quality_score, 0)) as avg_quality,
                COUNT(DISTINCT task_type) as task_types_handled
            FROM agent_executions
            WHERE start_time >= ?
            GROUP BY agent_id
            ORDER BY successful DESC, avg_duration ASC
        """

        cursor.execute(query, (cutoff_date,))

        stats = []
        for row in cursor.fetchall():
            total = row[1]
            successful = row[2] or 0
            success_rate = successful / total if total > 0 else 0.0

            stats.append(
                {
                    "agent_id": row[0],
                    "total_executions": total,
                    "successful_executions": successful,
                    "success_rate": success_rate,
                    "avg_duration_seconds": row[3] or 0.0,
                    "avg_quality_score": row[4] or 0.0,
                    "task_types_handled": row[5],
                }
            )

        return stats

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
        logger.info("PerformanceTracker closed")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PerformanceTracker(db_path={self.db_path})"
