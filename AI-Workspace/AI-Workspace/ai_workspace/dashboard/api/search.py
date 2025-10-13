"""Search API endpoints (Async Version).

Provides full-text search across reuse checks, phase gates, and agents
with relevance ranking and multi-field queries using async I/O.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from quart import Blueprint, jsonify, request

from ..services.async_search_engine import AsyncSearchEngine

logger = logging.getLogger(__name__)

search_bp = Blueprint("search", __name__, url_prefix="/api/search")

# Global async search engine instance (initialized lazily)
_search_engine: Optional[AsyncSearchEngine] = None


async def _get_search_engine() -> AsyncSearchEngine:
    """Get or initialize async search engine instance.

    Returns:
        AsyncSearchEngine instance with indexes built
    """
    global _search_engine

    if _search_engine is None:
        _search_engine = AsyncSearchEngine(Path.cwd())
        # Build indexes on first request (async)
        await _search_engine.rebuild_indexes_async()

    return _search_engine


@search_bp.route("/query", methods=["POST"])
async def search_query() -> tuple[Dict[str, Any], int]:
    """Execute full-text search across workspace data (async).

    Request body (JSON):
        {
            "query": "search terms",
            "filters": {
                "types": ["reuse_check", "phase_gate", "agent"],  // optional
                "category": "backend",  // optional
                "date_range": {...}  // optional, future enhancement
            },
            "limit": 20  // optional, default 20
        }

    Returns:
        JSON response with search results and metadata

    Response format:
        {
            "results": [
                {
                    "type": "reuse_check|phase_gate|agent",
                    "title": "Result title",
                    "description": "Brief description",
                    "content": "Matched content snippet",
                    "score": 0.85,
                    "matched_fields": ["field1", "field2"],
                    "metadata": {...}
                }
            ],
            "total": 100,
            "returned": 20,
            "query_time_ms": 45.2,
            "query": {...}
        }

    Error responses:
        400: Invalid request (missing query, invalid filters)
        500: Internal server error
    """
    try:
        # Parse request body
        data = await request.get_json()

        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # Validate query parameter
        query = data.get("query")
        if not query or not isinstance(query, str):
            return jsonify({"error": "Missing or invalid 'query' parameter"}), 400

        # Parse optional parameters
        filters = data.get("filters", {})
        limit = data.get("limit", 20)

        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return (
                jsonify(
                    {
                        "error": "Invalid 'limit' parameter. Must be integer between 1 and 100",
                        "value": limit,
                    }
                ),
                400,
            )

        # Validate filters structure
        if not isinstance(filters, dict):
            return jsonify({"error": "'filters' must be a dictionary"}), 400

        # Validate filter types if provided
        if "types" in filters:
            valid_types = {"reuse_check", "phase_gate", "agent"}
            filter_types = filters["types"]

            if not isinstance(filter_types, list):
                return jsonify({"error": "'filters.types' must be a list"}), 400

            invalid_types = set(filter_types) - valid_types
            if invalid_types:
                return (
                    jsonify(
                        {
                            "error": f"Invalid types in filter: {invalid_types}",
                            "valid_types": list(valid_types),
                        }
                    ),
                    400,
                )

        # Execute async search
        engine = await _get_search_engine()
        results, total_count, query_time_ms = await engine.search_async(
            query=query, filters=filters, limit=limit
        )

        # Convert results to JSON-serializable format
        results_data = [
            {
                "type": r.result_type,
                "title": r.title,
                "description": r.description,
                "content": r.content,
                "score": round(r.relevance_score, 3),
                "matched_fields": r.matched_fields,
                "metadata": r.metadata,
            }
            for r in results
        ]

        # Build response
        response = {
            "results": results_data,
            "total": total_count,
            "returned": len(results_data),
            "query_time_ms": round(query_time_ms, 2),
            "query": {"query": query, "filters": filters, "limit": limit},
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in search_query: {e}", exc_info=True)
        return (
            jsonify(
                {
                    "error": "Internal server error",
                    "message": str(e),
                    "query": data.get("query", "") if data else "",
                }
            ),
            500,
        )


@search_bp.route("/rebuild-index", methods=["POST"])
async def rebuild_index() -> tuple[Dict[str, Any], int]:
    """Rebuild search indexes from current workspace data (async).

    Useful when workspace data has been updated and search results
    need to reflect the latest changes.

    Returns:
        JSON response with rebuild status

    Response format:
        {
            "status": "success",
            "indexes_rebuilt": {
                "reuse_checks": 50,
                "phase_gates": 20,
                "agents": 156
            },
            "rebuild_time_ms": 123.4
        }

    Error responses:
        500: Internal server error
    """
    try:
        import time

        start_time = time.time()

        # Rebuild indexes asynchronously
        engine = await _get_search_engine()
        await engine.rebuild_indexes_async()

        rebuild_time_ms = (time.time() - start_time) * 1000

        # Get index counts
        stats = await engine.get_stats_async()

        response = {
            "status": "success",
            "indexes_rebuilt": {
                "reuse_checks": stats["indexes"].get("reuse_checks", {}).get("total_documents", 0),
                "phase_gates": stats["indexes"].get("phase_gates", {}).get("total_documents", 0),
                "agents": stats["indexes"].get("agents", {}).get("total_documents", 0),
            },
            "rebuild_time_ms": round(rebuild_time_ms, 2),
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in rebuild_index: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@search_bp.route("/stats", methods=["GET"])
async def get_search_stats() -> tuple[Dict[str, Any], int]:
    """Get search index statistics (async).

    Returns:
        JSON response with index statistics

    Response format:
        {
            "indexes": {
                "reuse_checks": {
                    "total_documents": 50,
                    "total_terms": 1234,
                    "fields_indexed": ["task_name", "files_checked"]
                },
                "phase_gates": {...},
                "agents": {...}
            },
            "total_documents": 226,
            "total_unique_terms": 5678
        }

    Error responses:
        500: Internal server error
    """
    try:
        engine = await _get_search_engine()
        stats = await engine.get_stats_async()

        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error in get_search_stats: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@search_bp.route("/suggestions", methods=["GET"])
async def get_suggestions() -> tuple[Dict[str, Any], int]:
    """Get search query suggestions based on common terms (async).

    Query parameters:
        prefix: Partial query string (optional)
        limit: Maximum number of suggestions (default: 10)

    Returns:
        JSON response with suggested search terms

    Response format:
        {
            "suggestions": [
                {"term": "dashboard", "frequency": 25},
                {"term": "api", "frequency": 18},
                ...
            ],
            "total": 10
        }

    Error responses:
        500: Internal server error
    """
    try:
        engine = await _get_search_engine()

        prefix = request.args.get("prefix", "").lower()
        limit = request.args.get("limit", default=10, type=int)

        # Get suggestions asynchronously
        suggestions = await engine.get_suggestions_async(prefix, limit)

        response = {"suggestions": suggestions, "total": len(suggestions)}

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in get_suggestions: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )
