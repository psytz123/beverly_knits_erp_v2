"""Cache management API endpoints.

Provides cache statistics, health checks, and manual cache invalidation
for the multi-layer caching system.
"""

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

cache_bp = Blueprint("cache", __name__, url_prefix="/api/cache")

# Global cache manager instance (set by search engine)
_cache_manager = None


def set_cache_manager(manager: Any) -> None:
    """Set the global cache manager instance.

    Args:
        manager: CacheManager instance
    """
    global _cache_manager
    _cache_manager = manager
    logger.info("Cache manager registered with API")


def get_cache_manager() -> Any:
    """Get the global cache manager instance.

    Returns:
        CacheManager instance or None if not set
    """
    return _cache_manager


@cache_bp.route("/stats", methods=["GET"])
def get_cache_stats() -> tuple[Dict[str, Any], int]:
    """Get cache performance statistics.

    Returns:
        JSON response with cache metrics

    Response format:
        {
            "l1_hits": 150,
            "l1_misses": 25,
            "l1_hit_rate": 85.7,
            "l1_size": 45,
            "l1_max_size": 100,
            "l1_utilization": 45.0,
            "l2_hits": 10,
            "l2_misses": 15,
            "l2_hit_rate": 40.0,
            "l2_available": true,
            "l2_size": 50,
            "l3_hits": 5,
            "l3_misses": 20,
            "l3_hit_rate": 20.0,
            "l3_dir": "/path/to/cache",
            "l3_size": 100,
            "overall_hit_rate": 68.8,
            "total_requests": 240,
            "total_hits": 165,
            "avg_query_time_ms": 5.2
        }

    Error responses:
        503: Cache manager not initialized
        500: Internal server error
    """
    try:
        cache = get_cache_manager()

        if not cache:
            return (
                jsonify({
                    "error": "Cache manager not initialized",
                    "message": "Search engine not yet started"
                }),
                503,
            )

        stats = cache.get_metrics()
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error in get_cache_stats: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@cache_bp.route("/health", methods=["GET"])
def get_cache_health() -> tuple[Dict[str, Any], int]:
    """Check health of all cache layers.

    Returns:
        JSON response with health status for each layer

    Response format:
        {
            "l1": {
                "status": "ok",
                "size": 45
            },
            "l2": {
                "status": "ok",
                "size": 50
            },
            "l3": {
                "status": "ok",
                "size": 100
            }
        }

    Error responses:
        503: Cache manager not initialized
        500: Internal server error
    """
    try:
        cache = get_cache_manager()

        if not cache:
            return (
                jsonify({
                    "error": "Cache manager not initialized",
                    "message": "Search engine not yet started"
                }),
                503,
            )

        health = cache.health_check()
        return jsonify(health), 200

    except Exception as e:
        logger.error(f"Error in get_cache_health: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@cache_bp.route("/invalidate", methods=["POST"])
def invalidate_cache() -> tuple[Dict[str, Any], int]:
    """Manually invalidate cache entries.

    Request body (JSON):
        {
            "pattern": "*",  // optional, default "*" (all entries)
        }

    Supported patterns:
        - "*": Clear all cache entries
        - "search:*": Clear all search-related entries
        - "search:abc123": Clear specific entry

    Returns:
        JSON response with invalidation results

    Response format:
        {
            "status": "success",
            "pattern": "*",
            "entries_cleared": 245,
            "message": "Cache invalidated successfully"
        }

    Error responses:
        400: Invalid request (invalid pattern)
        503: Cache manager not initialized
        500: Internal server error
    """
    try:
        cache = get_cache_manager()

        if not cache:
            return (
                jsonify({
                    "error": "Cache manager not initialized",
                    "message": "Search engine not yet started"
                }),
                503,
            )

        # Parse request body
        data = request.get_json() or {}
        pattern = data.get("pattern", "*")

        # Validate pattern
        if not isinstance(pattern, str):
            return (
                jsonify({
                    "error": "Invalid pattern",
                    "message": "Pattern must be a string"
                }),
                400,
            )

        # Invalidate cache
        entries_cleared = cache.invalidate(pattern)

        response = {
            "status": "success",
            "pattern": pattern,
            "entries_cleared": entries_cleared,
            "message": f"Cache invalidated successfully. Cleared {entries_cleared} entries.",
        }

        logger.info(f"Cache invalidated: pattern='{pattern}', cleared={entries_cleared}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in invalidate_cache: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@cache_bp.route("/reset-metrics", methods=["POST"])
def reset_cache_metrics() -> tuple[Dict[str, Any], int]:
    """Reset cache performance metrics to zero.

    Useful for benchmarking or testing cache performance from a clean state.
    Does not clear cached data, only resets hit/miss counters.

    Returns:
        JSON response confirming reset

    Response format:
        {
            "status": "success",
            "message": "Cache metrics reset successfully"
        }

    Error responses:
        503: Cache manager not initialized
        500: Internal server error
    """
    try:
        cache = get_cache_manager()

        if not cache:
            return (
                jsonify({
                    "error": "Cache manager not initialized",
                    "message": "Search engine not yet started"
                }),
                503,
            )

        cache.reset_metrics()

        response = {
            "status": "success",
            "message": "Cache metrics reset successfully",
        }

        logger.info("Cache metrics reset")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in reset_cache_metrics: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )


@cache_bp.route("/config", methods=["GET"])
def get_cache_config() -> tuple[Dict[str, Any], int]:
    """Get cache configuration parameters.

    Returns:
        JSON response with cache configuration

    Response format:
        {
            "l1_max_size": 100,
            "l2_ttl": 300,
            "l2_available": true,
            "l3_dir": "/path/to/cache"
        }

    Error responses:
        503: Cache manager not initialized
        500: Internal server error
    """
    try:
        cache = get_cache_manager()

        if not cache:
            return (
                jsonify({
                    "error": "Cache manager not initialized",
                    "message": "Search engine not yet started"
                }),
                503,
            )

        config = {
            "l1_max_size": cache.l1_max_size,
            "l2_ttl": cache.l2_ttl,
            "l2_available": cache.redis is not None,
            "l3_dir": str(cache.l3_dir),
        }

        return jsonify(config), 200

    except Exception as e:
        logger.error(f"Error in get_cache_config: {e}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "message": str(e)}),
            500,
        )
