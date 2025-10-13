"""Reuse Analysis API endpoints (Async Version).

Read-only endpoints for accessing reuse check data with async I/O.
Replaces synchronous reuse.py with non-blocking operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quart import Blueprint, jsonify, request

from ..services.analytics import MetricsAnalyzer
from ..services.async_data_loader import AsyncDataLoader, ReuseCheck
from ..services.data_loader import WorkspaceDataLoader

reuse_bp = Blueprint("reuse", __name__, url_prefix="/api/reuse")

# Thread pool for CPU-bound filtering operations
_executor = ThreadPoolExecutor(max_workers=4)


def _parse_iso_date(date_str: str) -> Optional[datetime]:
    """Parse ISO date string to datetime object.

    Args:
        date_str: ISO format date string (YYYY-MM-DD or ISO 8601)

    Returns:
        datetime object (timezone-aware UTC) or None if parsing fails
    """
    if not date_str:
        return None

    try:
        # Try ISO 8601 format first (with time)
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        try:
            # Try simple date format
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Make timezone-aware (assume UTC)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _filter_reuse_checks(
    checks: List[ReuseCheck],
    query: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: str = "all",
    min_reuse: Optional[float] = None,
) -> List[ReuseCheck]:
    """Filter reuse checks based on query parameters (runs in thread pool).

    Args:
        checks: List of reuse checks to filter
        query: Search query for task name or file paths
        start_date: ISO date string for start of date range
        end_date: ISO date string for end of date range
        status: Filter by approval status ("approved", "rejected", or "all")
        min_reuse: Minimum reuse percentage (0-100)

    Returns:
        Filtered list of reuse checks
    """
    filtered = checks

    # Filter by search query (case-insensitive)
    if query:
        query_lower = query.lower()
        filtered = [
            c
            for c in filtered
            if query_lower in c.task_name.lower()
            or any(query_lower in f.lower() for f in c.files_checked)
        ]

    # Filter by date range
    if start_date:
        start_dt = _parse_iso_date(start_date)
        if start_dt:
            filtered = [
                c
                for c in filtered
                if _parse_iso_date(c.timestamp)
                and _parse_iso_date(c.timestamp) >= start_dt
            ]

    if end_date:
        end_dt = _parse_iso_date(end_date)
        if end_dt:
            from datetime import timedelta

            end_dt_inclusive = end_dt + timedelta(days=1)
            filtered = [
                c
                for c in filtered
                if _parse_iso_date(c.timestamp)
                and _parse_iso_date(c.timestamp) < end_dt_inclusive
            ]

    # Filter by approval status
    if status == "approved":
        filtered = [c for c in filtered if c.approved_to_create]
    elif status == "rejected":
        filtered = [c for c in filtered if not c.approved_to_create]

    # Filter by minimum reuse percentage
    if min_reuse is not None:
        filtered = [c for c in filtered if c.reuse_percentage >= min_reuse]

    return filtered


@reuse_bp.route("/search", methods=["GET"])
async def search_checks() -> tuple[Dict[str, Any], int]:
    """Search and filter reuse checks (async).

    Query parameters:
        q: Search query for task name or file paths (optional)
        start_date: ISO date string for start of range (optional)
        end_date: ISO date string for end of range (optional)
        status: Filter by status - "approved", "rejected", or "all" (default: "all")
        min_reuse: Minimum reuse percentage 0-100 (optional)

    Returns:
        JSON response with filtered results and metadata
    """
    try:
        # Parse query parameters
        query = request.args.get("q", default=None, type=str)
        start_date = request.args.get("start_date", default=None, type=str)
        end_date = request.args.get("end_date", default=None, type=str)
        status = request.args.get("status", default="all", type=str)
        min_reuse_str = request.args.get("min_reuse", default=None, type=str)

        # Validate status parameter
        if status not in ["approved", "rejected", "all"]:
            return (
                jsonify(
                    {
                        "error": "Invalid status parameter. Must be 'approved', 'rejected', or 'all'",
                        "status": status,
                    }
                ),
                400,
            )

        # Validate and parse min_reuse
        min_reuse = None
        if min_reuse_str is not None:
            try:
                min_reuse = float(min_reuse_str)
                if min_reuse < 0 or min_reuse > 100:
                    return (
                        jsonify(
                            {
                                "error": "min_reuse must be between 0 and 100",
                                "value": min_reuse,
                            }
                        ),
                        400,
                    )
            except ValueError:
                return (
                    jsonify(
                        {
                            "error": "min_reuse must be a valid number",
                            "value": min_reuse_str,
                        }
                    ),
                    400,
                )

        # Validate date formats
        if start_date and not _parse_iso_date(start_date):
            return (
                jsonify(
                    {
                        "error": "Invalid start_date format. Use ISO format (YYYY-MM-DD)",
                        "value": start_date,
                    }
                ),
                400,
            )

        if end_date and not _parse_iso_date(end_date):
            return (
                jsonify(
                    {
                        "error": "Invalid end_date format. Use ISO format (YYYY-MM-DD)",
                        "value": end_date,
                    }
                ),
                400,
            )

        # Load all checks asynchronously
        loader = AsyncDataLoader(Path.cwd())
        all_checks = await loader.load_reuse_checks_async()
        total_count = len(all_checks)

        # Apply filters in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        filtered_checks = await loop.run_in_executor(
            _executor,
            _filter_reuse_checks,
            all_checks,
            query,
            start_date,
            end_date,
            status,
            min_reuse,
        )

        # Convert to dict for JSON serialization
        results = [
            {
                "task_name": c.task_name,
                "search_completed": c.search_completed,
                "reuse_analyzed": c.reuse_analyzed,
                "reuse_percentage": c.reuse_percentage,
                "timestamp": c.timestamp,
                "files_checked": c.files_checked,
                "approved": c.approved_to_create,
            }
            for c in filtered_checks
        ]

        # Build response with query metadata
        response = {
            "results": results,
            "total": total_count,
            "filtered": len(results),
            "query": {
                "q": query,
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "min_reuse": min_reuse,
            },
        }

        return jsonify(response), 200

    except Exception as e:
        return (
            jsonify(
                {"error": "Internal server error", "message": str(e), "query": {}}
            ),
            500,
        )


@reuse_bp.route("/checks", methods=["GET"])
async def get_checks() -> Dict[str, Any]:
    """Get all reuse checks (async).

    Query parameters:
        limit: Maximum number of checks to return (default: 50)

    Returns:
        JSON response with reuse checks
    """
    try:
        loader = AsyncDataLoader(Path.cwd())
        limit = request.args.get("limit", default=50, type=int)

        # Load checks asynchronously
        checks = await loader.load_reuse_checks_async()
        checks = checks[:limit]

        # Convert to dict for JSON serialization
        checks_data = [
            {
                "task_name": c.task_name,
                "search_completed": c.search_completed,
                "reuse_analyzed": c.reuse_analyzed,
                "reuse_percentage": c.reuse_percentage,
                "timestamp": c.timestamp,
                "files_checked": c.files_checked,
                "approved": c.approved_to_create,
            }
            for c in checks
        ]

        return jsonify({"checks": checks_data, "total": len(checks), "limit": limit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reuse_bp.route("/history", methods=["GET"])
async def get_history() -> Dict[str, Any]:
    """Get reuse check history with statistics (async).

    Returns:
        JSON response with reuse history and trends
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        # Run analytics in parallel
        summary_task = loop.run_in_executor(_executor, analyzer.calculate_reuse_summary)
        distribution_task = loop.run_in_executor(_executor, analyzer.calculate_reuse_distribution)

        summary, distribution = await asyncio.gather(summary_task, distribution_task)

        return jsonify(
            {
                "summary": summary,
                "distribution": distribution,
                "recent_checks": summary["recent_checks"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reuse_bp.route("/violations", methods=["GET"])
async def get_violations() -> Dict[str, Any]:
    """Get reuse violations (blocked creations) (async).

    Returns:
        JSON response with violations
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        violations = await loop.run_in_executor(_executor, analyzer.find_reuse_violations)

        return jsonify({"violations": violations, "count": len(violations)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@reuse_bp.route("/stats", methods=["GET"])
async def get_stats() -> Dict[str, Any]:
    """Get reuse statistics (async).

    Returns:
        JSON response with reuse statistics
    """
    try:
        # Run analytics in thread pool
        loop = asyncio.get_event_loop()

        sync_loader = WorkspaceDataLoader(Path.cwd())
        analyzer = await loop.run_in_executor(_executor, MetricsAnalyzer, sync_loader)

        summary = await loop.run_in_executor(_executor, analyzer.calculate_reuse_summary)

        return jsonify(
            {
                "total_checks": summary["total_checks"],
                "avg_reuse_percentage": summary["avg_reuse_percentage"],
                "high_reuse_count": summary["high_reuse_count"],
                "medium_reuse_count": summary["medium_reuse_count"],
                "low_reuse_count": summary["low_reuse_count"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
