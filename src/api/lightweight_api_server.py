#!/usr/bin/env python3
"""
Lightweight API Server for Beverly Knits ERP Dashboard
Supports both PostgreSQL and SQLite with automatic fallback
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database connection globals
db_connection = None
db_type = None


def load_config() -> Dict[str, Any]:
    """Load configuration from .env and database config files."""
    config = {
        'db_type': os.getenv('DATABASE_TYPE', 'sqlite'),
        'db_path': os.getenv('DATABASE_PATH', 'data/beverly_erp.db'),
        'app_port': int(os.getenv('APP_PORT', '5006')),
        'app_host': os.getenv('APP_HOST', '0.0.0.0'),
    }

    # Try to load database config
    config_path = Path(__file__).parent.parent / 'database' / 'database_config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                db_config = json.load(f)
                config.update(db_config)
        except Exception as e:
            logger.warning(f"Could not load database config: {e}")

    return config


def init_postgresql(config: Dict[str, Any]) -> Any:
    """Initialize PostgreSQL connection."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=config.get('host', 'localhost'),
            port=config.get('port', 5432),
            database=config.get('database', 'beverly_knits_erp'),
            user=config.get('user', 'erp_user'),
            password=config.get('password', 'erp_password'),
            cursor_factory=RealDictCursor,
        )
        logger.info("✓ Connected to PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return None


def init_sqlite(db_path: str) -> Any:
    """Initialize SQLite connection."""
    try:
        import sqlite3

        # Convert path to absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)

        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logger.info(f"✓ Connected to SQLite database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"✗ SQLite connection failed: {e}")
        return None


def init_database() -> None:
    """Initialize database connection with fallback."""
    global db_connection, db_type

    config = load_config()

    # Try PostgreSQL first
    if config['db_type'] == 'postgresql':
        db_connection = init_postgresql(config)
        if db_connection:
            db_type = 'postgresql'
            return

    # Fallback to SQLite
    logger.info("Falling back to SQLite database...")
    db_connection = init_sqlite(config['db_path'])
    if db_connection:
        db_type = 'sqlite'
        return

    logger.error("✗ Could not connect to any database")
    db_type = 'mock'


def execute_query(query: str, params: tuple = ()) -> List[Dict]:
    """Execute a database query and return results."""
    global db_connection, db_type

    if db_type == 'mock':
        return []

    try:
        if db_type == 'postgresql':
            cursor = db_connection.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
        else:  # sqlite
            cursor = db_connection.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()

        return results
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return []


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check() -> tuple:
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'database': db_type,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/yarn-intelligence', methods=['GET'])
def yarn_intelligence() -> tuple:
    """Get yarn intelligence data."""
    try:
        # Mock data for demonstration
        data = {
            'critical_yarns': [
                {
                    'desc_id': 'Y001',
                    'yarn_description': 'Cotton 30s',
                    'theoretical_balance': 1000,
                    'allocated': 500,
                    'available': 500,
                    'on_order': 200,
                    'criticality': 'HIGH'
                }
            ],
            'total_yarns': 150,
            'critical_count': 12
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in yarn_intelligence: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/knit-orders', methods=['GET'])
def knit_orders() -> tuple:
    """Get knit orders data."""
    try:
        data = {
            'orders': [],
            'total': 0,
            'status': 'ok'
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in knit_orders: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/production-planning', methods=['GET'])
def production_planning() -> tuple:
    """Get production planning data."""
    try:
        data = {
            'schedules': [],
            'capacity': {},
            'status': 'ok'
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in production_planning: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fabric-forecast', methods=['GET'])
def fabric_forecast() -> tuple:
    """Get fabric forecast data."""
    try:
        data = {
            'forecasts': [],
            'accuracy': 0.0,
            'status': 'ok'
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in fabric_forecast: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard-summary', methods=['GET'])
def dashboard_summary() -> tuple:
    """Get dashboard summary data."""
    try:
        data = {
            'kpis': {
                'total_orders': 0,
                'active_machines': 0,
                'yarn_criticality': 0,
                'forecast_accuracy': 0.0
            },
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in dashboard_summary: {e}")
        return jsonify({'error': str(e)}), 500


# Catch-all for missing endpoints
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_fallback(path: str) -> tuple:
    """Fallback for unimplemented API endpoints."""
    logger.warning(f"Unimplemented endpoint: /api/{path}")
    return jsonify({
        'message': f'Endpoint /api/{path} not yet implemented',
        'status': 'not_implemented'
    }), 501


def main() -> None:
    """Main entry point."""
    config = load_config()

    print("=" * 70)
    print("Beverly Knits ERP - Lightweight API Server")
    print("=" * 70)

    # Initialize database
    init_database()

    print(f"Database Type: {db_type}")
    print(f"Server: http://{config['app_host']}:{config['app_port']}")
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70)

    # Start Flask app
    try:
        app.run(
            host=config['app_host'],
            port=config['app_port'],
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        if db_connection:
            db_connection.close()
        sys.exit(0)


if __name__ == '__main__':
    main()
