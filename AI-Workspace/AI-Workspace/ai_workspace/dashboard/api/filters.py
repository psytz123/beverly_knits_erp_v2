"""Filter API endpoints.

Provides REST API endpoints for multi-dimensional filtering of dashboard data
including filter dimension discovery, filter application, and preset management.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ..services.filter_engine import FilterEngine

logger = logging.getLogger(__name__)

filters_bp = Blueprint('filters', __name__, url_prefix='/api/filters')

# Global filter engine instance (initialized lazily)
_filter_engine: FilterEngine | None = None


def _get_filter_engine() -> FilterEngine:
    """Get or initialize filter engine instance.

    Returns:
        FilterEngine instance
    """
    global _filter_engine

    if _filter_engine is None:
        _filter_engine = FilterEngine(Path.cwd())

    return _filter_engine


@filters_bp.route('/available', methods=['GET'])
def get_available_filters() -> tuple[Dict[str, Any], int]:
    """Get all available filter dimensions.

    Returns:
        JSON response with filter dimensions and HTTP status code
    """
    try:
        engine = _get_filter_engine()
        dimensions = engine.get_available_dimensions()

        return jsonify({
            'dimensions': dimensions,
            'count': len(dimensions)
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving filter dimensions: {e}")
        return jsonify({
            'error': 'Failed to retrieve filter dimensions',
            'message': str(e)
        }), 500


@filters_bp.route('/apply', methods=['POST'])
def apply_filters() -> tuple[Dict[str, Any], int]:
    """Apply filters to dashboard data.

    Request Body:
        {
            "filters": {
                "status": ["failed", "blocked"],
                "date_range": "last_day",
                "type": ["reuse_check"]
            },
            "operator": "AND",
            "data_sources": ["reuse_checks", "phase_gates"]
        }

    Returns:
        JSON response with filtered results and HTTP status code
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400

        filters = data.get('filters', {})
        operator = data.get('operator', 'AND')
        data_sources = data.get('data_sources')

        # Validate operator
        if operator not in ['AND', 'OR']:
            return jsonify({
                'error': 'Invalid operator',
                'message': 'Operator must be "AND" or "OR"'
            }), 400

        # Apply filters
        engine = _get_filter_engine()
        results = engine.apply_filters(
            filters=filters,
            operator=operator,
            data_sources=data_sources
        )

        # Calculate counts
        counts = {source: len(items) for source, items in results.items()}
        total = sum(counts.values())

        return jsonify({
            'results': results,
            'counts': counts,
            'total': total,
            'applied_filters': filters,
            'operator': operator
        }), 200

    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        return jsonify({
            'error': 'Failed to apply filters',
            'message': str(e)
        }), 500


@filters_bp.route('/presets', methods=['GET'])
def get_presets() -> tuple[Dict[str, Any], int]:
    """Get all predefined filter presets.

    Returns:
        JSON response with preset configurations and HTTP status code
    """
    try:
        engine = _get_filter_engine()
        predefined = engine.get_presets()

        return jsonify({
            'presets': {
                'predefined': predefined,
                'custom': []
            },
            'count': {
                'predefined': len(predefined),
                'custom': 0
            }
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving presets: {e}")
        return jsonify({
            'error': 'Failed to retrieve presets',
            'message': str(e)
        }), 500


@filters_bp.route('/presets/<preset_id>', methods=['GET'])
def get_preset_by_id(preset_id: str) -> tuple[Dict[str, Any], int]:
    """Get a specific preset by ID.

    Args:
        preset_id: Preset identifier

    Returns:
        JSON response with preset configuration and HTTP status code
    """
    try:
        engine = _get_filter_engine()
        predefined = engine.get_presets()
        preset = next((p for p in predefined if p['id'] == preset_id), None)

        if preset:
            return jsonify({'preset': preset}), 200

        return jsonify({
            'error': 'Preset not found',
            'message': f'No preset found with ID: {preset_id}'
        }), 404

    except Exception as e:
        logger.error(f"Error retrieving preset: {e}")
        return jsonify({
            'error': 'Failed to retrieve preset',
            'message': str(e)
        }), 500


@filters_bp.route('/stats', methods=['POST'])
def get_filter_stats() -> tuple[Dict[str, Any], int]:
    """Get statistics about data matching filters.

    Request Body:
        {
            "filters": {...},
            "operator": "AND"
        }

    Returns:
        JSON response with statistics and HTTP status code
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400

        filters = data.get('filters', {})
        operator = data.get('operator', 'AND')

        # Apply filters
        engine = _get_filter_engine()
        results = engine.apply_filters(filters=filters, operator=operator)

        # Calculate statistics
        total_matches = sum(len(items) for items in results.values())
        by_source = {source: len(items) for source, items in results.items()}

        # Aggregate by status
        by_status: Dict[str, int] = {}
        for items in results.values():
            for item in items:
                status = item.get('status', 'unknown')
                by_status[status] = by_status.get(status, 0) + 1

        return jsonify({
            'stats': {
                'total_matches': total_matches,
                'by_source': by_source,
                'by_status': by_status
            }
        }), 200

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return jsonify({
            'error': 'Failed to calculate statistics',
            'message': str(e)
        }), 500
