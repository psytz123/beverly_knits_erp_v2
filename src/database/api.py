#!/usr/bin/env python3
"""
Database API Endpoints
Purpose: RESTful API for database operations and sync control
Usage: Import and register with Flask app
"""

from __future__ import annotations
from typing import Dict, Any
from flask import Blueprint, jsonify, request
from datetime import datetime
import logging

from database.config import get_session
from database.scheduler import get_scheduler, init_scheduler
from database.models import (
    CFVersion, YarnRequirement, ProductionOrder,
    YarnInventory, KnitOrder, APISync
)
from database.efab_api_sync import EFabAPISync

logger = logging.getLogger(__name__)

# Create Blueprint
db_api = Blueprint('database_api', __name__, url_prefix='/api/db')


@db_api.route('/status', methods=['GET'])
def get_database_status() -> tuple:
    """
    Get database connection status and statistics.

    Returns:
        JSON response with database status
    """
    try:
        with get_session() as session:
            # Get record counts
            stats = {
                'cf_versions': session.query(CFVersion).count(),
                'yarn_requirements': session.query(YarnRequirement).count(),
                'production_orders': session.query(ProductionOrder).count(),
                'yarn_inventory': session.query(YarnInventory).count(),
                'knit_orders': session.query(KnitOrder).count(),
                'api_syncs': session.query(APISync).count()
            }

            # Get last sync info
            last_sync = session.query(APISync).order_by(
                APISync.started_at.desc()
            ).first()

            return jsonify({
                'status': 'connected',
                'statistics': stats,
                'last_sync': {
                    'started_at': last_sync.started_at.isoformat() if last_sync else None,
                    'completed_at': last_sync.completed_at.isoformat() if last_sync and last_sync.completed_at else None,
                    'status': last_sync.status if last_sync else None,
                    'records_processed': last_sync.records_processed if last_sync else 0
                }
            }), 200

    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@db_api.route('/sync/manual', methods=['POST'])
def trigger_manual_sync() -> tuple:
    """
    Trigger manual data sync from eFab API.

    Returns:
        JSON response with sync results
    """
    try:
        # Get session cookie from request or environment
        session_cookie = request.json.get('session_cookie') if request.json else None
        if not session_cookie:
            import os
            session_cookie = os.environ.get('EFAB_SESSION')

        if not session_cookie:
            return jsonify({
                'error': 'Session cookie required'
            }), 400

        # Get database URL
        from database.config import get_database_url
        db_url = get_database_url()

        # Run sync
        sync_service = EFabAPISync(db_url, session_cookie)
        results = sync_service.sync_all()

        return jsonify({
            'success': results['success'],
            'duration': results['duration'],
            'cf_versions': results['cf_versions'],
            'production_orders': results['production_orders'],
            'errors': results.get('errors', [])
        }), 200 if results['success'] else 500

    except Exception as e:
        logger.exception(f"Manual sync failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/sync/scheduler/start', methods=['POST'])
def start_scheduler() -> tuple:
    """
    Start the data sync scheduler.

    Returns:
        JSON response with scheduler status
    """
    try:
        # Get configuration
        data = request.json or {}
        session_cookie = data.get('session_cookie')
        interval_hours = data.get('interval_hours', 2)

        if not session_cookie:
            import os
            session_cookie = os.environ.get('EFAB_SESSION')

        if not session_cookie:
            return jsonify({
                'error': 'Session cookie required'
            }), 400

        # Initialize scheduler
        from database.config import get_database_url
        db_url = get_database_url()

        scheduler = init_scheduler(
            db_url,
            session_cookie,
            sync_interval_hours=interval_hours,
            auto_start=True
        )

        return jsonify({
            'status': 'started',
            'scheduler': scheduler.get_status()
        }), 200

    except Exception as e:
        logger.exception(f"Scheduler start failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/sync/scheduler/stop', methods=['POST'])
def stop_scheduler() -> tuple:
    """
    Stop the data sync scheduler.

    Returns:
        JSON response with confirmation
    """
    try:
        scheduler = get_scheduler()
        if scheduler:
            scheduler.stop()
            return jsonify({
                'status': 'stopped'
            }), 200
        else:
            return jsonify({
                'status': 'not_running'
            }), 200

    except Exception as e:
        logger.exception(f"Scheduler stop failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/sync/scheduler/status', methods=['GET'])
def get_scheduler_status() -> tuple:
    """
    Get scheduler status.

    Returns:
        JSON response with scheduler status
    """
    try:
        scheduler = get_scheduler()
        if scheduler:
            return jsonify(scheduler.get_status()), 200
        else:
            return jsonify({
                'running': False,
                'message': 'Scheduler not initialized'
            }), 200

    except Exception as e:
        logger.exception(f"Status check failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/cf-versions', methods=['GET'])
def get_cf_versions() -> tuple:
    """
    Get CF versions from database.

    Returns:
        JSON response with CF versions
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        style = request.args.get('style')
        customer = request.args.get('customer')

        with get_session() as session:
            query = session.query(CFVersion)

            # Apply filters
            if style:
                query = query.filter(CFVersion.style_number.contains(style))
            if customer:
                query = query.filter(CFVersion.customer_code == customer)

            # Get total count
            total = query.count()

            # Apply pagination
            versions = query.offset(offset).limit(limit).all()

            # Format response
            data = []
            for v in versions:
                data.append({
                    'id': v.id,
                    'version_id': v.version_id,
                    'style_number': v.style_number,
                    'description': v.description,
                    'customer_code': v.customer_code,
                    'fabric_type': v.fabric_type,
                    'construction': v.construction,
                    'width': v.width,
                    'weight': v.weight,
                    'status': v.status,
                    'created_at': v.created_at.isoformat(),
                    'yarn_count': len(v.yarn_requirements)
                })

            return jsonify({
                'total': total,
                'limit': limit,
                'offset': offset,
                'data': data
            }), 200

    except Exception as e:
        logger.exception(f"CF versions query failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/production-orders', methods=['GET'])
def get_production_orders() -> tuple:
    """
    Get production orders from database.

    Returns:
        JSON response with production orders
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        machine = request.args.get('machine')

        with get_session() as session:
            query = session.query(ProductionOrder)

            # Apply filters
            if status:
                query = query.filter(ProductionOrder.status == status)
            if machine:
                query = query.filter(ProductionOrder.machine_id == machine)

            # Get total count
            total = query.count()

            # Apply pagination and order by due date
            orders = query.order_by(
                ProductionOrder.due_date
            ).offset(offset).limit(limit).all()

            # Format response
            data = []
            for o in orders:
                data.append({
                    'id': o.id,
                    'order_number': o.order_number,
                    'customer_po': o.customer_po,
                    'quantity_ordered': float(o.quantity_ordered),
                    'quantity_produced': float(o.quantity_produced),
                    'unit_of_measure': o.unit_of_measure,
                    'due_date': o.due_date.isoformat() if o.due_date else None,
                    'start_date': o.start_date.isoformat() if o.start_date else None,
                    'status': o.status,
                    'priority': o.priority,
                    'work_center': o.work_center,
                    'machine_id': o.machine_id,
                    'style_number': o.cf_version.style_number if o.cf_version else None
                })

            return jsonify({
                'total': total,
                'limit': limit,
                'offset': offset,
                'data': data
            }), 200

    except Exception as e:
        logger.exception(f"Production orders query failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@db_api.route('/yarn-inventory', methods=['GET'])
def get_yarn_inventory() -> tuple:
    """
    Get yarn inventory from database.

    Returns:
        JSON response with yarn inventory
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        yarn_code = request.args.get('yarn_code')
        low_stock = request.args.get('low_stock', 'false').lower() == 'true'

        with get_session() as session:
            query = session.query(YarnInventory)

            # Apply filters
            if yarn_code:
                query = query.filter(YarnInventory.yarn_code.contains(yarn_code))
            if low_stock:
                query = query.filter(
                    YarnInventory.quantity_available < YarnInventory.reorder_point
                )

            # Get total count
            total = query.count()

            # Apply pagination
            inventory = query.offset(offset).limit(limit).all()

            # Format response
            data = []
            for i in inventory:
                data.append({
                    'id': i.id,
                    'yarn_code': i.yarn_code,
                    'location': i.location,
                    'quantity_on_hand': float(i.quantity_on_hand),
                    'quantity_allocated': float(i.quantity_allocated),
                    'quantity_available': float(i.quantity_available),
                    'reorder_point': float(i.reorder_point),
                    'reorder_quantity': float(i.reorder_quantity),
                    'last_received_date': i.last_received_date.isoformat() if i.last_received_date else None,
                    'last_counted_date': i.last_counted_date.isoformat() if i.last_counted_date else None
                })

            return jsonify({
                'total': total,
                'limit': limit,
                'offset': offset,
                'data': data
            }), 200

    except Exception as e:
        logger.exception(f"Yarn inventory query failed: {e}")
        return jsonify({
            'error': str(e)
        }), 500


def register_database_api(app):
    """
    Register database API blueprint with Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(db_api)
    logger.info("Database API registered at /api/db")


if __name__ == "__main__":
    # Test API endpoints
    from flask import Flask

    app = Flask(__name__)
    register_database_api(app)

    # Run test server
    app.run(debug=True, port=5007)