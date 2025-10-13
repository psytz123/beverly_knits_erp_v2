"""Metrics API endpoints (Async Version).

Read-only endpoints for accessing workspace metrics and statistics
with async I/O for non-blocking operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from quart import Blueprint, jsonify

from ..services.analytics import MetricsAnalyzer
from ..services.async_data_loader import AsyncDataLoader

metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")

# Thread pool for CPU-bound analytics operations
_executor = ThreadPoolExecutor(max_workers=4)


async def _get_analyzer() -> MetricsAnalyzer:
    """Get metrics analyzer instance asynchronously.

    Returns:
        MetricsAnalyzer instance
    """
    loader = AsyncDataLoader(Path.cwd())

    # Create analyzer with async-loaded data (run in thread pool as it's sync)
    loop = asyncio.get_event_loop()
    analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, loader)

    return analyzer


@metrics_bp.route("/summary", methods=["GET"])
async def get_summary() -> Dict[str, Any]:
    """Get overall metrics summary (async).

    Returns:
        JSON response with summary metrics
    """
    try:
        # Load data asynchronously
        loader = AsyncDataLoader(Path.cwd())

        # Load all data in parallel
        project_info_task = loader.get_project_info_async()
        workspace_stats_task = loader.get_workspace_stats_async()

        project_info, workspace_stats = await asyncio.gather(
            project_info_task,
            workspace_stats_task
        )

        # Run analytics in thread pool (CPU-bound operations)
        loop = asyncio.get_event_loop()

        # Create analyzer with the loader
        from ..services.data_loader import WorkspaceDataLoader
        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        # Run analytics calculations in parallel in thread pool
        reuse_summary_task = loop.run_in_executor(_executor, analyzer.calculate_reuse_summary)
        gate_progress_task = loop.run_in_executor(_executor, analyzer.calculate_gate_progress)
        quality_metrics_task = loop.run_in_executor(_executor, analyzer.calculate_quality_metrics)

        reuse_summary, gate_progress, quality_metrics = await asyncio.gather(
            reuse_summary_task,
            gate_progress_task,
            quality_metrics_task
        )

        return jsonify(
            {
                "project": project_info,
                "reuse": {
                    "avg_percentage": reuse_summary["avg_reuse_percentage"],
                    "total_checks": reuse_summary["total_checks"],
                },
                "gates": {
                    "completed": gate_progress["completed_gates"],
                    "total": gate_progress["total_gates"],
                    "current_phase": gate_progress["current_phase"],
                },
                "quality": quality_metrics,
                "workspace": workspace_stats,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/reuse", methods=["GET"])
async def get_reuse_metrics() -> Dict[str, Any]:
    """Get detailed reuse metrics (async).

    Returns:
        JSON response with reuse analysis data
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        from ..services.data_loader import WorkspaceDataLoader
        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        # Run all analytics in parallel
        summary_task = loop.run_in_executor(_executor, analyzer.calculate_reuse_summary)
        distribution_task = loop.run_in_executor(_executor, analyzer.calculate_reuse_distribution)
        violations_task = loop.run_in_executor(_executor, analyzer.find_reuse_violations)

        summary, distribution, violations = await asyncio.gather(
            summary_task,
            distribution_task,
            violations_task
        )

        return jsonify(
            {
                "summary": summary,
                "distribution": distribution,
                "violations": violations,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/quality", methods=["GET"])
async def get_quality_metrics() -> Dict[str, Any]:
    """Get quality compliance metrics (async).

    Returns:
        JSON response with quality indicators
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        from ..services.data_loader import WorkspaceDataLoader
        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        quality_metrics = await loop.run_in_executor(_executor, analyzer.calculate_quality_metrics)

        return jsonify(quality_metrics)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/trends", methods=["GET"])
async def get_trends() -> Dict[str, Any]:
    """Get metrics trends over time (async).

    Returns:
        JSON response with trend data
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        from ..services.data_loader import WorkspaceDataLoader
        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        # Run analytics in parallel
        activity_task = loop.run_in_executor(_executor, analyzer.get_recent_activity, 30)
        language_task = loop.run_in_executor(_executor, analyzer.get_language_stats)

        recent_activity, language_stats = await asyncio.gather(
            activity_task,
            language_task
        )

        return jsonify(
            {
                "recent_activity": recent_activity,
                "languages": language_stats,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
