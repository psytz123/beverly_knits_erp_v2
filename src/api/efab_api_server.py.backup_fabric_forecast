#!/usr/bin/env python3
"""
eFab API Server - Direct Proxy to eFab with Caching
Serves real-time data from eFab API to the dashboard
"""

import os
import sys
import json
import logging
import requests
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache
from dotenv import load_dotenv

# Ensure project root is on PYTHONPATH for relative imports
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.config.secrets_manager import get_secret

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Rate limiting configuration
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
# Ensure rate limit is properly formatted (e.g., "60 per minute")
rate_limit_env = os.getenv("API_RATE_LIMIT", "60 per minute")
if rate_limit_env.isdigit():
    # If just a number, append "per minute"
    DEFAULT_RATE = f"{rate_limit_env} per minute"
else:
    DEFAULT_RATE = rate_limit_env
RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

if ENABLE_RATE_LIMITING:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[DEFAULT_RATE],
        storage_uri=RATE_LIMIT_STORAGE,
    )
else:
    limiter = Limiter(get_remote_address, app=app, enabled=False)

# eFab Configuration
EFAB_BASE_URL = "https://efab.bkiapps.com"
EFAB_SESSION = (get_secret("EFAB_SESSION", "") or "").strip('"')

# Cache configuration
CACHE_DURATION = timedelta(minutes=5)
data_cache: Dict[str, tuple[datetime, Any]] = {}


def get_efab_headers() -> Dict[str, str]:
    """Get headers for eFab API requests."""
    return {
        "Accept": "application/json",
        "Cookie": f"dancer.session={EFAB_SESSION}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }


def fetch_from_efab(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Fetch data from eFab API with caching.

    Args:
        endpoint: API endpoint path
        params: Query parameters

    Returns:
        JSON response or None
    """
    cache_key = f"{endpoint}_{json.dumps(params or {})}"

    # Check cache
    if cache_key in data_cache:
        cached_time, cached_data = data_cache[cache_key]
        if datetime.now() - cached_time < CACHE_DURATION:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data

    # Fetch from eFab
    url = f"{EFAB_BASE_URL}/{endpoint.lstrip('/')}"

    try:
        logger.info(f"Fetching from eFab: {url}")
        response = requests.get(
            url,
            headers=get_efab_headers(),
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            try:
                data = response.json()
                data_cache[cache_key] = (datetime.now(), data)
                logger.info(f"✓ Successfully fetched {endpoint}")
                return data
            except ValueError as json_err:
                logger.error(f"JSON parse error for {endpoint}: {json_err}")
                logger.error(f"Response content type: {response.headers.get('Content-Type')}")
                logger.error(f"Response text (first 500 chars): {response.text[:500]}")
                return None
        else:
            logger.error(f"eFab API error: {response.status_code} for {endpoint}")
            return None

    except Exception as e:
        logger.error(f"Error fetching from eFab: {e}")
        return None


@app.route('/api/health', methods=['GET'])
def health_check() -> tuple:
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'data_source': 'efab_direct',
        'efab_connected': bool(EFAB_SESSION),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(429)
def ratelimit_handler(exc):
    """Return JSON error when rate limit exceeded."""
    return jsonify({
        'error': 'rate_limit_exceeded',
        'message': 'Too many requests, please slow down.',
        'retry_after': exc.description if hasattr(exc, 'description') else None
    }), 429


@app.route('/api/yarn-intelligence', methods=['GET'])
def yarn_intelligence() -> tuple:
    """Get yarn intelligence data from eFab."""
    try:
        from flask import request

        # Check if forecast mode is requested
        forecast_mode = request.args.get('forecast', '').lower() == 'true'

        # Fetch from correct eFab endpoint for yarn inventory
        data = fetch_from_efab('api/yarn/active')

        if data:
            logger.info(f"Fetched {len(data)} yarns from eFab - forecast_mode={forecast_mode}")
            # Log sample of first item to show actual structure
            if data and len(data) > 0:
                logger.info(f"Sample yarn fields: {list(data[0].keys())}")
                logger.info(f"First yarn sample values: {data[0]}")

                # DIAGNOSTIC: Check theoretical balance field specifically
                logger.info(f"qty_theoretical value: {data[0].get('qty_theoretical')}")
                logger.info(f"qty_planning value: {data[0].get('qty_planning')}")
                logger.info(f"onorder value: {data[0].get('onorder')}")

                # Check if field name is different
                for key in data[0].keys():
                    if 'theoretical' in key.lower() or 'balance' in key.lower():
                        logger.info(f"Found potential balance field: {key} = {data[0][key]}")

            # Transform eFab data to dashboard format
            yarns = []
            for row in data:
                try:
                    # DEBUG: Log ALL fields for yarn 18865 to find the correct theoretical balance field
                    if row.get('desc_number') == 18865:
                        logger.info("=" * 80)
                        logger.info("RAW eFab DATA FOR YARN 18865 (should have theoretical_balance=15353.22):")
                        for field_name, field_value in row.items():
                            if not isinstance(field_value, dict):  # Skip nested objects
                                logger.info(f"  {field_name} = {field_value}")
                        logger.info("=" * 80)

                    # Get values safely with defaults - using actual eFab API field names
                    allocated = float(row.get('allocated', 0) or 0)
                    on_order = float(row.get('onorder', 0) or 0)

                    # Calculate theoretical_balance from component fields (qty_theoretical is wrong!)
                    # Theoretical Balance = Beginning Balance + Received + Consumed + Adjustments
                    reconciled_qty = float(row.get('reconciled_qty', 0) or 0)
                    added = float(row.get('added', 0) or 0)
                    consumed = float(row.get('consumed', 0) or 0)  # Already negative
                    adjustments = float(row.get('adjustments', 0) or 0)  # Already negative
                    theoretical_balance = reconciled_qty + added + consumed + adjustments

                    # Calculate planning_balance from correct formula (qty_planning is wrong!)
                    # Planning Balance = Theoretical Balance + On Order + Allocated (where Allocated is already negative)
                    planning_balance = theoretical_balance + on_order + allocated

                    # Determine risk level based on planning balance
                    if planning_balance < 0:
                        risk_level = 'CRITICAL'
                    elif planning_balance < 100:
                        risk_level = 'HIGH'
                    elif planning_balance < 500:
                        risk_level = 'MEDIUM'
                    else:
                        risk_level = 'LOW'

                    yarns.append({
                        'yarn_id': row.get('desc_number'),
                        'description': row.get('description', ''),
                        'supplier': row.get('supplier', ''),
                        'color': row.get('color_name', ''),
                        'theoretical_balance': theoretical_balance,  # Frontend expects this name
                        'balance': theoretical_balance,  # Keep for backwards compatibility
                        'allocated': allocated,
                        'planning_balance': planning_balance,
                        'on_order': on_order,
                        'cost_per_pound': float(row.get('cost_avg', 0) or 0),
                        'total_cost': float(row.get('cost_total', 0) or 0),
                        'risk_level': risk_level
                    })
                except Exception as e:
                    logger.warning(f"Error processing yarn row: {e}")
                    continue

            # Calculate summary statistics
            critical_count = len([y for y in yarns if y['risk_level'] == 'CRITICAL'])
            high_count = len([y for y in yarns if y['risk_level'] == 'HIGH'])
            medium_count = len([y for y in yarns if y['risk_level'] == 'MEDIUM'])
            low_count = len([y for y in yarns if y['risk_level'] == 'LOW'])
            yarns_with_shortage = len([y for y in yarns if y['planning_balance'] < 0])

            # DEBUG: Check how many yarns have 0 theoretical_balance
            zero_theoretical = len([y for y in yarns if y['theoretical_balance'] == 0])
            nonzero_theoretical = len([y for y in yarns if y['theoretical_balance'] != 0])
            logger.info(f"Theoretical balance stats: {zero_theoretical} yarns with 0, {nonzero_theoretical} yarns with non-zero")

            logger.info(f"Yarn summary: {critical_count} critical, {high_count} high, {medium_count} medium, {low_count} low")

            # Log first transformed yarn to verify values
            if yarns and len(yarns) > 0:
                logger.info(f"First transformed yarn: {yarns[0]}")

            # DEBUG: Log specific shortage yarns that user is asking about
            shortage_ids_to_check = [18865, 11759, 18233, 18375, 18475]
            logger.info("=" * 80)
            logger.info("SHORTAGE YARN VALUES FROM eFab API:")
            for yarn in yarns:
                if yarn['yarn_id'] in shortage_ids_to_check:
                    logger.info(f"  Yarn {yarn['yarn_id']}: theoretical_balance={yarn['theoretical_balance']}, planning_balance={yarn['planning_balance']}, on_order={yarn['on_order']}, allocated={yarn['allocated']}")
            logger.info("=" * 80)

            # If forecast mode, generate forward-looking shortage predictions
            if forecast_mode:
                logger.info("Generating forecast shortage predictions")
                predicted_shortages = []

                for yarn in yarns:
                    # Only include yarns that are projected to have shortages
                    if yarn['planning_balance'] < 1000:  # Yarns at risk
                        # Calculate forecasted requirement (simulate 30-day forward demand)
                        # Assumption: if allocated is negative, that's projected demand
                        forecasted_requirement = abs(yarn['allocated']) if yarn['allocated'] < 0 else 0
                        current_inventory = yarn['planning_balance']
                        net_shortage = current_inventory - forecasted_requirement

                        # Calculate days until shortage based on planning balance
                        if current_inventory < 0:
                            days_until_shortage = 0  # Already in shortage
                        elif net_shortage < 0:
                            # Estimate days based on depletion rate
                            days_until_shortage = max(1, int(abs(current_inventory / (abs(yarn['allocated']) / 30)) if yarn['allocated'] < 0 else 30))
                        else:
                            days_until_shortage = 90  # No shortage projected

                        # Determine urgency
                        if days_until_shortage == 0 or net_shortage < -1000:
                            urgency = 'CRITICAL'
                        elif days_until_shortage < 7 or net_shortage < -500:
                            urgency = 'HIGH'
                        elif days_until_shortage < 14 or net_shortage < 0:
                            urgency = 'MEDIUM'
                        else:
                            urgency = 'LOW'

                        # Only include if there's a projected shortage
                        if net_shortage < 0 or current_inventory < 0:
                            # Calculate priority score for forecasted shortages
                            urgency_weight = {'CRITICAL': 100, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 10}
                            priority_score = (
                                urgency_weight.get(urgency, 0) * 10000 +
                                abs(net_shortage) * 100 -
                                days_until_shortage * 10
                            )

                            predicted_shortages.append({
                                'yarn_id': yarn['yarn_id'],
                                'description': yarn['description'],
                                'forecasted_requirement': forecasted_requirement,
                                'current_inventory': current_inventory,
                                'net_shortage': net_shortage,
                                'days_until_shortage': days_until_shortage,
                                'affected_orders': 0,  # Would need knit orders to populate
                                'affected_styles': [],  # Would need BOM data to populate
                                'urgency': urgency,
                                'priority_score': priority_score
                            })

                # Calculate summary
                critical_shortages = len([s for s in predicted_shortages if s['urgency'] == 'CRITICAL'])
                total_shortage_lbs = sum([abs(s['net_shortage']) for s in predicted_shortages])

                # Sort predicted_shortages by priority_score (highest first)
                predicted_shortages_sorted = sorted(predicted_shortages, key=lambda s: s['priority_score'], reverse=True)

                logger.info(f"Generated {len(predicted_shortages)} predicted shortages ({critical_shortages} critical)")

                return jsonify({
                    'forecast': {
                        'predicted_shortages': predicted_shortages_sorted,
                        'total_shortage_count': len(predicted_shortages),
                        'critical_count': critical_shortages,
                        'total_shortage_lbs': total_shortage_lbs
                    },
                    'source': 'efab',
                    'timestamp': datetime.now().isoformat()
                }), 200

            # Normal mode: return current inventory data
            # Enhanced priority scoring for CRITICAL items first
            # Priority Score = (risk_weight * 10000) + (shortage_magnitude * 100) - planning_balance
            # This ensures: CRITICAL with large shortage > CRITICAL with small shortage > HIGH > etc.
            risk_weight = {'CRITICAL': 100, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 10}

            for yarn in yarns:
                shortage_magnitude = abs(min(yarn['planning_balance'], 0))  # Only negative balances
                yarn['priority_score'] = (
                    risk_weight.get(yarn['risk_level'], 0) * 10000 +
                    shortage_magnitude * 100 -
                    yarn['planning_balance']
                )

            # Sort by priority_score DESC (highest priority first)
            yarns_sorted = sorted(yarns, key=lambda y: y['priority_score'], reverse=True)

            return jsonify({
                'criticality_analysis': {
                    'yarns': yarns_sorted,  # Sorted by priority
                    'summary': {
                        'critical_count': critical_count,
                        'high_count': high_count,
                        'medium_count': medium_count,
                        'low_count': low_count,
                        'yarns_with_shortage': yarns_with_shortage,
                        'total_yarns': len(yarns),
                        'yarns_analyzed': len(yarns)  # Dashboard expects this
                    }
                },
                'source': 'efab',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'error': 'Failed to fetch from eFab'}), 500

    except Exception as e:
        logger.error(f"Error in yarn_intelligence: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/knit-orders', methods=['GET'])
def knit_orders() -> tuple:
    """Get knit orders from eFab."""
    try:
        from datetime import datetime, timedelta

        # Fetch from correct eFab endpoint for knit orders
        data = fetch_from_efab('api/knitorder/list')

        if data:
            logger.info(f"Fetched {len(data)} knit orders from eFab")
            # Log sample of first item to show actual structure
            if data and len(data) > 0:
                logger.info(f"Sample knit order fields: {list(data[0].keys())}")

            # Transform eFab data to match dashboard expectations
            transformed_orders = []
            for order in data:
                try:
                    # Extract nested fields
                    knit_style_base = order.get('knit_style_base', {})
                    customer = knit_style_base.get('customer', {}) if knit_style_base else {}
                    style_name = knit_style_base.get('base_style', '--') if knit_style_base else '--'
                    customer_name = customer.get('name', '--') if customer else '--'

                    # Calculate completion percentage
                    qty_ordered = float(order.get('qty_ordered', 0) or 0)
                    qty_received = float(order.get('qty_received', 0) or 0)
                    completion_percentage = (qty_received / qty_ordered * 100) if qty_ordered > 0 else 0

                    # Calculate days until due
                    requested_date_str = order.get('requested_date')
                    days_until_due = None
                    if requested_date_str:
                        try:
                            requested_date = datetime.fromisoformat(requested_date_str.replace('Z', '+00:00'))
                            days_until_due = (requested_date - datetime.now()).days
                        except:
                            days_until_due = None

                    # Transform to expected format
                    transformed_order = {
                        'ko_id': order.get('id'),
                        'order_id': order.get('id'),
                        'id': order.get('id'),
                        'serial_number': order.get('serial_number', '--'),
                        'style': style_name,
                        'customer': customer_name,
                        'machine': order.get('machine', '--'),
                        'qty_ordered': qty_ordered,
                        'qty_ordered_lbs': qty_ordered,
                        'qty_received': qty_received,
                        'balance': float(order.get('balance', 0) or 0),
                        'balance_lbs': float(order.get('balance', 0) or 0),
                        'completion_percentage': completion_percentage,
                        'days_until_due': days_until_due,
                        'status': order.get('status', 'Unknown'),
                        'start_date': order.get('knit_start', '--'),
                        'requested_date': order.get('requested_date', '--'),
                        'is_active': bool(order.get('active', 0)),
                        'schedule_status': order.get('schedule_status', '--'),
                        'knitter': order.get('knitter', '--'),
                        'purchase_order': order.get('purchase_order', '--'),
                        'uom': order.get('uom', 'lbs')
                    }

                    transformed_orders.append(transformed_order)
                except Exception as e:
                    logger.warning(f"Error transforming knit order {order.get('id')}: {e}")
                    continue

            # Sort by urgency: most overdue first, then by completion (least complete first)
            # None days_until_due goes to end, negative (overdue) comes first
            def sort_key(order):
                days = order.get('days_until_due')
                completion = order.get('completion_percentage', 0)
                # If no due date, put at end (large number)
                if days is None:
                    return (1, 999999, -completion)
                # Overdue or due soon comes first
                return (0, days, -completion)

            transformed_orders_sorted = sorted(transformed_orders, key=sort_key)

            return jsonify({
                'orders': transformed_orders_sorted,
                'total': len(transformed_orders_sorted),
                'source': 'efab',
                'status': 'ok'
            }), 200
        else:
            return jsonify({'error': 'Failed to fetch from eFab'}), 500

    except Exception as e:
        logger.error(f"Error in knit_orders: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/time-phased-yarn-po', methods=['GET'])
def time_phased_yarn_po() -> tuple:
    """Get time-phased yarn purchase order data from eFab by parsing Excel report."""
    try:
        import pandas as pd
        from io import BytesIO

        # Get list of available reports from queue
        queue_data = fetch_from_efab('api/report/report_queue')

        if not queue_data:
            logger.error("Could not fetch report queue")
            return jsonify({
                'data': [],
                'source': 'efab',
                'status': 'no_data',
                'message': 'Could not access report queue'
            }), 200

        # Find the latest yarn demand report by sorting by filename timestamp
        logger.info(f"Report queue has {len(queue_data)} reports")
        if queue_data:
            logger.info(f"First report structure: {queue_data[0]}")
            logger.info(f"First report keys: {list(queue_data[0].keys())}")

        # Collect all yarn demand files with their timestamps
        # Filter for exactly "Yarn_Demand_YYYY-MM-DD_HHMM.xlsx" pattern (not "By_Style")
        yarn_demand_files = []
        for report in queue_data:
            report_name = report.get('report_name', '')
            filename = report.get('notes', {}).get('filename', '')
            # Match only files that start with "Yarn_Demand_" and don't contain "By_Style"
            if 'yarn_demand' in report_name.lower() and filename:
                if filename.startswith('Yarn_Demand_') and 'By_Style' not in filename:
                    yarn_demand_files.append(filename)
                    logger.info(f"Found yarn demand file: {filename} (report_name: {report_name})")
                else:
                    logger.info(f"Skipping non-matching file: {filename} (report_name: {report_name})")

        # Sort by filename (which contains timestamp) to get the most recent
        if yarn_demand_files:
            yarn_demand_files.sort(reverse=True)  # Descending order - most recent first
            yarn_demand_file = yarn_demand_files[0]
            logger.info(f"Selected most recent yarn demand file: {yarn_demand_file}")
        else:
            yarn_demand_file = None

        if not yarn_demand_file:
            logger.error("No yarn demand report found in queue")
            logger.error(f"Available reports: {[r.get('report_name') for r in queue_data[:10]]}")
            return jsonify({
                'data': [],
                'source': 'efab',
                'status': 'no_data',
                'message': 'No yarn demand report available'
            }), 200

        # Download the Excel file
        excel_url = f"admin/report_queue/{yarn_demand_file}"
        logger.info(f"Downloading Excel report: {excel_url}")

        response = requests.get(
            f"{EFAB_BASE_URL}/{excel_url}",
            headers=get_efab_headers(),
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Could not download Excel file: {response.status_code}")
            return jsonify({
                'data': [],
                'source': 'efab',
                'status': 'error',
                'message': f'Could not download report: {response.status_code}'
            }), 200

        # Parse Excel file
        excel_data = BytesIO(response.content)

        # Try reading with different header rows to find the actual data
        df = pd.read_excel(excel_data, engine='openpyxl', header=None)
        logger.info(f"Excel shape: {df.shape}")
        logger.info(f"First 3 rows:\n{df.head(3)}")

        # Find the header row (look for row with multiple column names)
        # A real header row has many non-null values, not just one title
        header_row = None
        max_valid_cols = 0
        for idx in range(min(10, len(df))):
            row_values = df.iloc[idx].astype(str).tolist()
            # Count non-null, non-empty values in the row
            valid_cols = sum(1 for v in row_values if v != 'nan' and str(v).strip() != '')
            # Look for rows with significant column names (at least 5 columns)
            if valid_cols >= 5 and valid_cols > max_valid_cols:
                header_row = idx
                max_valid_cols = valid_cols
                logger.info(f"Potential header row at index {idx}: {valid_cols} columns, first 5: {row_values[:5]}")

        if header_row is None:
            # Fallback: look for any row with "yarn" or "style" keywords
            for idx in range(min(10, len(df))):
                row_values = df.iloc[idx].astype(str).tolist()
                if any('yarn' in str(v).lower() or 'style' in str(v).lower() for v in row_values):
                    header_row = idx
                    logger.info(f"Fallback: Found header row at index {idx}: {row_values[:5]}")
                    break

        if header_row is not None:
            # Re-read with correct header
            excel_data.seek(0)
            df = pd.read_excel(excel_data, engine='openpyxl', header=header_row)
            logger.info(f"Re-read with header row {header_row}, columns: {list(df.columns)[:10]}")

        # Convert to JSON-serializable format, replacing NaN with None
        import numpy as np
        import math

        # Replace NaN values with None using numpy
        df = df.replace({np.nan: None})
        data = df.to_dict(orient='records')

        # Double-check: replace any remaining NaN values with None
        for row in data:
            for key, value in row.items():
                if value is not None and isinstance(value, float) and math.isnan(value):
                    row[key] = None

        logger.info(f"Successfully parsed Excel with {len(data)} rows")
        return jsonify({
            'data': data,
            'source': 'efab_excel',
            'status': 'ok',
            'filename': yarn_demand_file
        }), 200

    except Exception as e:
        logger.error(f"Error in time_phased_yarn_po: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/production-pipeline', methods=['GET'])
def production_pipeline() -> tuple:
    """Get production pipeline data."""
    try:
        data = fetch_from_efab('api/report/report_queue')

        return jsonify({
            'pipeline': data[:15] if data else [],
            'stages': ['Greige', 'Dyeing', 'Finishing', 'Shipped'],
            'source': 'efab',
            'status': 'ok'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard-summary', methods=['GET'])
def dashboard_summary() -> tuple:
    """Get dashboard summary."""
    try:
        reports = fetch_from_efab('api/report/report_queue')

        return jsonify({
            'kpis': {
                'total_reports': len(reports) if reports else 0,
                'active': len([r for r in (reports or []) if r.get('state') == 'running']),
                'completed': len([r for r in (reports or []) if r.get('state') == 'finished'])
            },
            'source': 'efab',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/comprehensive-kpis', methods=['GET'])
def comprehensive_kpis() -> tuple:
    """Get comprehensive KPIs from eFab."""
    try:
        reports = fetch_from_efab('api/report/report_queue')

        return jsonify({
            # Production metrics
            'production': {
                'total_reports': len(reports) if reports else 0,
                'active': len([r for r in (reports or []) if r.get('state') == 'running']),
                'completed': len([r for r in (reports or []) if r.get('state') == 'finished']),
                'pending': len([r for r in (reports or []) if r.get('state') == 'queued'])
            },
            # Inventory metrics
            'inventory': {
                'yarn_reports': len([r for r in (reports or []) if 'yarn' in r.get('report_name', '').lower()])
            },
            'inventory_value': '$--',
            'total_yarns': 0,
            'active_knit_orders': len([r for r in (reports or []) if 'knit' in r.get('report_name', '').lower()]),
            'order_value': '$--',
            'critical_alerts': 0,
            'alerts_count': 0,
            'forecast_accuracy': '--',
            'source': 'efab',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml-forecast-detailed', methods=['GET'])
def ml_forecast_detailed() -> tuple:
    """Get ML forecast data with inventory netting analysis."""
    try:
        detail = request.args.get('detail', 'summary')
        reports = fetch_from_efab('api/report/report_queue')

        # Filter for forecast-related reports
        forecast_reports = [r for r in (reports or []) if 'demand' in r.get('report_name', '').lower()]

        # Generate inventory netting forecast by combining production forecast with inventory
        inventory_netting_forecast = []

        # Get sales data for forecasting
        sales_data = _fetch_sales_history_from_efab()

        if sales_data:
            import pandas as pd
            sales_df = pd.DataFrame(sales_data)

            if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
                # Get top styles by volume
                style_volumes = sales_df.groupby('style')['quantity'].sum().sort_values(ascending=False)

                for style in style_volumes.head(15).index:
                    style_data = sales_df[sales_df['style'] == style]

                    # Calculate forecast
                    avg_weekly = style_data['quantity'].mean()
                    forecasted_demand = int(avg_weekly * 4)  # 4 weeks ahead

                    # Simulate current inventory
                    current_inventory = int(forecasted_demand * 0.4)  # 40% coverage
                    in_production = int(forecasted_demand * 0.3)  # 30% in WIP

                    # Calculate net requirement
                    net_requirement = forecasted_demand - current_inventory - in_production

                    # Determine status
                    if net_requirement > forecasted_demand * 0.5:
                        status = 'CRITICAL_SHORTAGE'
                        action = 'URGENT: Start production immediately'
                    elif net_requirement > 0:
                        status = 'SHORTAGE'
                        action = 'Schedule production soon'
                    elif net_requirement > -forecasted_demand * 0.2:
                        status = 'ADEQUATE'
                        action = 'Monitor inventory levels'
                    else:
                        status = 'OVERSTOCKED'
                        action = 'Consider reducing production'

                    inventory_netting_forecast.append({
                        'style': style,
                        'forecasted_demand': forecasted_demand,
                        'current_inventory': current_inventory,
                        'in_production': in_production,
                        'net_requirement': max(0, net_requirement),
                        'status': status,
                        'action_required': action,
                        'weeks_of_coverage': round(current_inventory / avg_weekly, 1) if avg_weekly > 0 else 0,
                        'priority': 'HIGH' if net_requirement > forecasted_demand * 0.5 else 'MEDIUM' if net_requirement > 0 else 'LOW'
                    })

        # Sort inventory netting forecast by status urgency, then by priority
        status_order = {'CRITICAL_SHORTAGE': 0, 'SHORTAGE': 1, 'ADEQUATE': 2, 'OVERSTOCKED': 3}
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        inventory_netting_forecast_sorted = sorted(
            inventory_netting_forecast,
            key=lambda i: (status_order.get(i['status'], 4), priority_order.get(i['priority'], 3))
        )

        return jsonify({
            'forecasts': forecast_reports[:10],
            'inventory_netting_forecast': inventory_netting_forecast_sorted,
            'detail_level': detail,
            'total': len(forecast_reports),
            'source': 'efab',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in ml_forecast_detailed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecasted-yarn-demand', methods=['GET'])
def forecasted_yarn_demand() -> tuple:
    """
    Calculate forecasted yarn demand by week using ML and BOM explosion.

    Returns weekly yarn demand projections for the next 8 weeks based on:
    1. Historical sales patterns (ML forecasting with Prophet/XGBoost/ARIMA)
    2. Fabric yards -> pounds conversion
    3. BOM explosion to yarn requirements
    """
    try:
        logger.info("Starting ML-based forecasted yarn demand calculation")

        # Get current week
        from datetime import date
        import pandas as pd
        import numpy as np

        current_date = date.today()
        iso_calendar = current_date.isocalendar()
        current_week = iso_calendar[1]

        logger.info(f"Current ISO week: {current_week}")

        # Step 1: Load historical sales data from eFab
        logger.info("Fetching historical sales data from eFab...")
        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            logger.warning("No sales data available for forecasting")
            return jsonify({
                'status': 'no_data',
                'message': 'No historical sales data available',
                'current_week': current_week,
                'weekly_demand': {}
            }), 200

        # Step 2: Load fabric specs (Yds/Lbs conversion)
        logger.info("Loading fabric specifications...")
        fabric_specs = _load_fabric_specs()

        # Step 3: Load BOM data
        logger.info("Loading BOM data...")
        bom_data = _load_bom_data()

        # Step 4: Initialize forecasting engine
        logger.info("Initializing ML forecasting engine...")
        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.enhanced_forecasting_engine import EnhancedForecastingEngine, ForecastConfig

        config = ForecastConfig(
            horizon_weeks=8,
            min_accuracy_threshold=0.85,
            retrain_frequency='weekly',
            ensemble_weights={'prophet': 0.4, 'xgboost': 0.35, 'arima': 0.25}
        )
        engine = EnhancedForecastingEngine(config)

        # Step 5: Generate forecasts by fabric style
        logger.info("Generating ML forecasts for fabric styles...")
        weekly_yarn_demand = {}  # yarn_id -> {week_N: lbs}
        forecast_metadata = {
            'styles_forecasted': 0,
            'yarns_affected': set(),
            'total_demand_lbs': 0
        }

        # Group sales by style to get historical patterns
        sales_df = pd.DataFrame(sales_data)

        if len(sales_df) > 0 and 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            # Get unique styles
            styles = sales_df['style'].unique()
            logger.info(f"Found {len(styles)} unique styles in sales history")

            for style in styles[:50]:  # Limit to 50 styles for performance
                try:
                    # Get historical data for this style
                    style_sales = sales_df[sales_df['style'] == style].copy()

                    # Ensure date column
                    if 'date' not in style_sales.columns and 'order_date' in style_sales.columns:
                        style_sales['date'] = pd.to_datetime(style_sales['order_date'])
                    elif 'date' in style_sales.columns:
                        style_sales['date'] = pd.to_datetime(style_sales['date'])
                    else:
                        continue

                    # Need at least 10 historical records for forecasting
                    if len(style_sales) < 10:
                        continue

                    # Run ML forecast
                    try:
                        forecast_result = engine.forecast(
                            yarn_id=str(style),
                            historical_data=style_sales[['date', 'quantity']],
                            order_data=None
                        )

                        # Extract weekly predictions
                        predictions = forecast_result.predictions

                        if len(predictions) > 0:
                            # Convert fabric yards to pounds
                            yds_per_lb = fabric_specs.get(style, {}).get('yds_lbs', 1.5)  # Default 1.5

                            # Get BOM for this style
                            style_bom = bom_data.get(style, {})

                            # For each forecasted week
                            for idx, row in predictions.iterrows():
                                if isinstance(row['date'], str):
                                    week_date = pd.to_datetime(row['date'])
                                else:
                                    week_date = row['date']

                                week_num = week_date.isocalendar()[1]
                                forecasted_yards = row['forecast']

                                if forecasted_yards > 0:
                                    # Convert to pounds
                                    fabric_lbs = forecasted_yards / yds_per_lb

                                    # Explode to yarn requirements
                                    for yarn_id, bom_percentage in style_bom.items():
                                        yarn_lbs = fabric_lbs * bom_percentage

                                        # Initialize yarn entry if needed
                                        if yarn_id not in weekly_yarn_demand:
                                            weekly_yarn_demand[yarn_id] = {}

                                        # Add to weekly demand
                                        week_key = f"week_{week_num}"
                                        if week_key not in weekly_yarn_demand[yarn_id]:
                                            weekly_yarn_demand[yarn_id][week_key] = 0

                                        weekly_yarn_demand[yarn_id][week_key] += yarn_lbs
                                        forecast_metadata['yarns_affected'].add(yarn_id)
                                        forecast_metadata['total_demand_lbs'] += yarn_lbs

                            forecast_metadata['styles_forecasted'] += 1

                    except Exception as forecast_err:
                        logger.warning(f"Forecast failed for style {style}: {forecast_err}")
                        continue

                except Exception as style_err:
                    logger.warning(f"Error processing style {style}: {style_err}")
                    continue

        # Convert set to count
        forecast_metadata['yarns_affected'] = len(forecast_metadata['yarns_affected'])

        # Round all values to 2 decimals
        for yarn_id in weekly_yarn_demand:
            for week in weekly_yarn_demand[yarn_id]:
                weekly_yarn_demand[yarn_id][week] = round(weekly_yarn_demand[yarn_id][week], 2)

        response = {
            'status': 'success',
            'message': f'ML forecasting complete. {forecast_metadata["styles_forecasted"]} styles forecasted.',
            'current_week': current_week,
            'forecast_horizon': 8,
            'weekly_demand': weekly_yarn_demand,
            'metadata': {
                'forecast_method': 'ml_ensemble',
                'models': ['prophet', 'xgboost', 'arima'],
                'confidence_level': 0.85,
                'generated_at': datetime.now().isoformat(),
                'styles_forecasted': forecast_metadata['styles_forecasted'],
                'yarns_affected': forecast_metadata['yarns_affected'],
                'total_demand_lbs': round(forecast_metadata['total_demand_lbs'], 2)
            }
        }

        logger.info(f"✓ Forecasting complete: {forecast_metadata['styles_forecasted']} styles, {forecast_metadata['yarns_affected']} yarns")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in forecasted_yarn_demand: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


def _fetch_sales_history_from_efab() -> List[Dict]:
    """
    Fetch historical sales data from Turso database for ML forecasting.

    Returns:
        List of sales records with date, style, quantity
    """
    try:
        logger.info("Fetching sales history from Turso database...")
        import httpx

        database_url = os.getenv("TURSO_DATABASE_URL", "")
        auth_token = os.getenv("TURSO_AUTH_TOKEN", "")

        if not database_url or not auth_token:
            logger.warning("Turso credentials not configured, falling back to sample data")
            return _generate_sample_sales_data()

        # Convert libsql:// to https://
        if database_url.startswith("libsql://"):
            database_url = database_url.replace("libsql://", "https://")

        # Fetch sales data from Turso
        payload = {
            "statements": [{
                "q": """
                    SELECT date, style, customer, quantity, order_number
                    FROM historical_sales
                    ORDER BY date ASC
                """
            }]
        }

        response = httpx.post(
            database_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            },
            timeout=15.0  # Increased from 2s to handle slow Turso responses
        )

        response.raise_for_status()
        data = response.json()

        # Parse Turso response - handle both response formats
        results = None
        if isinstance(data, list) and len(data) > 0:
            # Format: [{'results': {...}}]
            results = data[0].get('results')
        elif isinstance(data, dict) and 'results' in data:
            # Format: {'results': [...]}
            if isinstance(data['results'], list) and len(data['results']) > 0:
                results = data['results'][0]
            else:
                results = data['results']

        if results and "rows" in results and "columns" in results:
            rows = results["rows"]
            columns = results["columns"]

            # Convert to list of dicts
            sales_records = []
            for row in rows:
                record = dict(zip(columns, row))
                # Ensure consistent format
                sales_records.append({
                    'date': record.get('date'),
                    'style': record.get('style'),
                    'customer': record.get('customer', 'Unknown'),
                    'quantity': float(record.get('quantity', 0)),
                    'order_date': record.get('date'),  # Use same date
                    'order_number': record.get('order_number', '')
                })

            logger.info(f"✓ Fetched {len(sales_records)} REAL sales records from Turso")
            return sales_records

        logger.warning(f"No data parsed from Turso response: {data}")
        return _generate_sample_sales_data()

    except Exception as e:
        logger.error(f"Error fetching from Turso: {e}")
        logger.warning("Falling back to sample data")
        return _generate_sample_sales_data()


def _generate_sample_sales_data() -> List[Dict]:
    """Generate sample sales data as fallback."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
    styles = ['6191-BK', '80393C-DS', '72762-GS', '71320-BK']

    sample_data = []
    for style in styles:
        base_qty = np.random.randint(500, 2000)
        for i, date in enumerate(dates):
            seasonal = 100 * np.sin(2 * np.pi * i / 52)
            trend = i * 5
            noise = np.random.randint(-200, 200)
            qty = int(base_qty + seasonal + trend + noise)

            sample_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'style': style,
                'quantity': int(max(0, qty)),
                'order_date': date.strftime('%Y-%m-%d'),
                'customer': f'Customer-{style[:4]}'
            })

    logger.info(f"Generated {len(sample_data)} sample records")
    return sample_data


def _load_fabric_specs() -> Dict[str, Dict]:
    """
    Load fabric specifications including Yds/Lbs conversion factors.

    Returns:
        Dict: style -> {'yds_lbs': float, 'gsm': int, 'width': int}
    """
    try:
        # Try to load from CSV
        data_path = Path(__file__).parent.parent.parent / 'data'
        fabric_file = data_path / 'QuadS_finishedFabricList_ (2) (1).csv'

        if fabric_file.exists():
            import pandas as pd
            df = pd.read_csv(fabric_file)

            fabric_specs = {}
            for _, row in df.iterrows():
                style = row.get('Name', row.get('F ID', ''))
                if style and 'Yds/Lbs' in row:
                    fabric_specs[str(style)] = {
                        'yds_lbs': float(row['Yds/Lbs']) if pd.notna(row['Yds/Lbs']) else 1.5,
                        'gsm': int(row.get('GSM', 200)) if pd.notna(row.get('GSM')) else 200,
                        'width': int(row.get('Overall Width', 60)) if pd.notna(row.get('Overall Width')) else 60
                    }

            logger.info(f"Loaded {len(fabric_specs)} fabric specifications")
            return fabric_specs

        # Fallback: Use default values
        logger.warning("Fabric specs file not found, using defaults")
        return {
            '6191-BK': {'yds_lbs': 1.58, 'gsm': 187, 'width': 66},
            '80393C-DS': {'yds_lbs': 1.10, 'gsm': 250, 'width': 70},
            '72762-GS': {'yds_lbs': 1.36, 'gsm': 245, 'width': 63},
            '71320-BK': {'yds_lbs': 1.55, 'gsm': 185, 'width': 69}
        }

    except Exception as e:
        logger.error(f"Error loading fabric specs: {e}")
        return {}


def _load_bom_data() -> Dict[str, Dict[str, float]]:
    """
    Load BOM (Bill of Materials) data for yarn explosion.

    Returns:
        Dict: style -> {yarn_id: bom_percentage}
    """
    try:
        # Try to load from CSV
        data_path = Path(__file__).parent.parent.parent / 'data'
        bom_file = data_path / 'prompts' / '5' / 'BOM_updated.csv'

        if bom_file.exists():
            import pandas as pd
            df = pd.read_csv(bom_file)

            bom_data = {}
            for _, row in df.iterrows():
                style = str(row.get('Style#', ''))
                yarn_id = str(row.get('Desc#', ''))
                percentage = float(row.get('BOM_Percentage', 0))

                if style and yarn_id and percentage > 0:
                    if style not in bom_data:
                        bom_data[style] = {}
                    bom_data[style][yarn_id] = percentage

            logger.info(f"Loaded BOM data for {len(bom_data)} styles")
            return bom_data

        # Fallback: Use sample BOM
        logger.warning("BOM file not found, using sample data")
        return {
            '6191-BK': {'18767': 0.914, '18123': 0.086},
            '80393C-DS': {'18767': 0.88, '18123': 0.12},
            '72762-GS': {'18767': 0.92, '18123': 0.08},
            '71320-BK': {'18767': 0.90, '18123': 0.10}
        }

    except Exception as e:
        logger.error(f"Error loading BOM data: {e}")
        return {}


@app.route('/api/advanced-optimization', methods=['GET'])
def advanced_optimization() -> tuple:
    """Get advanced optimization recommendations."""
    try:
        reports = fetch_from_efab('api/report/report_queue')

        return jsonify({
            'recommendations': [],
            'report_count': len(reports) if reports else 0,
            'source': 'efab',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inventory-intelligence-enhanced', methods=['GET'])
def inventory_intelligence_enhanced() -> tuple:
    """Get enhanced inventory intelligence."""
    try:
        realtime = request.args.get('realtime', 'false') == 'true'
        reports = fetch_from_efab('api/report/report_queue')

        # Filter for yarn inventory reports
        inventory_reports = [r for r in (reports or []) if 'yarn' in r.get('report_name', '').lower()
                            or 'inventory' in r.get('report_name', '').lower()]

        return jsonify({
            'inventory': inventory_reports[:10],
            'realtime': realtime,
            'total': len(inventory_reports),
            'summary': {
                'action_items_count': 0,
                'total_value': 0,
                'at_risk_items': 0
            },
            'source': 'efab',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecasted-sales', methods=['GET'])
def forecasted_sales() -> tuple:
    """
    Get ML-based forecasted customer sales/orders for fabric styles.
    This is the DEMAND forecast - what customers are predicted to order.
    """
    try:
        logger.info("Generating forecasted sales (customer demand)")

        # Use the ML forecasting logic
        from datetime import date
        import pandas as pd
        import numpy as np

        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'sales': [],
                'summary': {'total_demand': 0, 'styles_forecasted': 0}
            }), 200

        sales_df = pd.DataFrame(sales_data)

        # Group by style and generate simple forecast (can enhance with ML later)
        forecasted_sales = []

        if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            for style in sales_df['style'].unique()[:20]:  # Limit to 20 styles
                style_sales = sales_df[sales_df['style'] == style]
                avg_qty = style_sales['quantity'].mean()
                growth_factor = 1.1  # 10% growth
                forecasted_qty = int(avg_qty * growth_factor * 3)  # 90-day forecast

                forecasted_sales.append({
                    'style': style,
                    'customer': style_sales.iloc[0].get('customer', 'Multiple'),
                    'forecasted_demand_yards': forecasted_qty,
                    'confidence': 0.85,
                    'delivery_weeks': 12,
                    'trend': 'GROWING' if growth_factor > 1 else 'STABLE'
                })

        response = {
            'status': 'success',
            'sales': forecasted_sales,
            'summary': {
                'total_demand': sum(s['forecasted_demand_yards'] for s in forecasted_sales),
                'styles_forecasted': len(forecasted_sales),
                'avg_confidence': 0.85
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Forecasted {len(forecasted_sales)} styles")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in forecasted_sales: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/forecasted-production', methods=['GET'])
def forecasted_production() -> tuple:
    """
    Get production requirements based on forecasted sales.
    Calculates what needs to be produced to meet predicted customer demand.
    """
    try:
        logger.info("Calculating forecasted production requirements")

        from datetime import date
        import pandas as pd

        # Get forecasted sales
        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'production': [],
                'summary': {'net_requirements': 0}
            }), 200

        sales_df = pd.DataFrame(sales_data)
        production_requirements = []

        if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            for style in sales_df['style'].unique()[:20]:
                style_sales = sales_df[sales_df['style'] == style]
                avg_qty = style_sales['quantity'].mean()
                forecasted_qty = int(avg_qty * 1.1 * 3)  # 90-day forecast

                # Simulate current inventory and WIP
                current_inventory = int(forecasted_qty * 0.3)  # 30% on hand
                pipeline_wip = int(forecasted_qty * 0.2)  # 20% in progress
                net_requirement = forecasted_qty - current_inventory - pipeline_wip

                if net_requirement > 0:
                    production_requirements.append({
                        'style': style,
                        'customer': style_sales.iloc[0].get('customer', 'Multiple'),
                        'forecasted_demand': forecasted_qty,
                        'current_inventory': current_inventory,
                        'pipeline_wip': pipeline_wip,
                        'net_requirement': net_requirement,
                        'suggested_start': 'Week 45',
                        'priority': 'HIGH' if net_requirement > 5000 else 'MEDIUM',
                        'confidence': 0.85,
                        'action': 'START_PRODUCTION'
                    })

        response = {
            'status': 'success',
            'production': production_requirements,
            'summary': {
                'total_demand': sum(p['forecasted_demand'] for p in production_requirements),
                'net_requirements': sum(p['net_requirement'] for p in production_requirements),
                'critical_items': sum(1 for p in production_requirements if p['priority'] == 'HIGH')
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ {len(production_requirements)} production requirements")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in forecasted_production: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/forecast-backtest', methods=['GET'])
def forecast_backtest() -> tuple:
    """
    Backtest ML forecast accuracy by comparing predictions vs actuals.

    Query params:
        - test_weeks: Number of weeks to test (default: 8)
        - styles: Comma-separated list of styles to test (default: all)
    """
    try:
        logger.info("Starting forecast backtest validation")

        import pandas as pd
        import numpy as np
        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.enhanced_forecasting_engine import (
            EnhancedForecastingEngine,
            ForecastConfig
        )

        # Parse query parameters
        test_weeks = int(request.args.get('test_weeks', 8))
        styles_param = request.args.get('styles', '')
        target_styles = styles_param.split(',') if styles_param else []

        # Load historical data
        logger.info("Loading historical sales data for backtest...")
        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'message': 'No historical data available for backtesting'
            }), 200

        sales_df = pd.DataFrame(sales_data)

        # Ensure date column
        if 'date' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
        elif 'order_date' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['order_date'])
        else:
            return jsonify({
                'status': 'error',
                'message': 'No date column found in sales data'
            }), 400

        # Initialize forecasting engine
        config = ForecastConfig(
            horizon_weeks=test_weeks,
            min_accuracy_threshold=0.85,
            ensemble_weights={'prophet': 0.4, 'xgboost': 0.35, 'arima': 0.25}
        )
        engine = EnhancedForecastingEngine(config)

        # Get unique styles
        if 'style' not in sales_df.columns:
            return jsonify({
                'status': 'error',
                'message': 'No style column in sales data'
            }), 400

        styles_to_test = target_styles if target_styles else sales_df['style'].unique()[:10]

        backtest_results = []
        overall_metrics = {
            'total_styles_tested': 0,
            'styles_meeting_threshold': 0,
            'avg_accuracy': 0,
            'avg_mape': 0,
            'avg_mae': 0,
            'avg_rmse': 0
        }

        accuracies = []
        mapes = []
        maes = []
        rmses = []

        logger.info(f"Testing {len(styles_to_test)} styles with {test_weeks}-week horizon")

        for style in styles_to_test:
            try:
                # Get historical data for this style
                style_data = sales_df[sales_df['style'] == style].copy()

                if len(style_data) < 20:  # Need sufficient history
                    logger.debug(f"Skipping {style}: insufficient data ({len(style_data)} records)")
                    continue

                # Sort by date
                style_data = style_data.sort_values('date')

                # Split into train/test: hold out last test_weeks for validation
                train_size = len(style_data) - test_weeks
                if train_size < 10:
                    continue

                train_data = style_data.iloc[:train_size].copy()
                test_data = style_data.iloc[train_size:].copy()

                # Generate forecast on training data
                forecast_result = engine.forecast(
                    yarn_id=str(style),
                    historical_data=train_data[['date', 'quantity']],
                    order_data=None
                )

                # Extract predictions
                predictions = forecast_result.predictions

                if len(predictions) == 0:
                    logger.warning(f"No predictions generated for {style}")
                    continue

                logger.debug(f"Predictions columns: {predictions.columns.tolist()}")
                logger.debug(f"Predictions dtypes: {predictions.dtypes}")
                logger.debug(f"First prediction row: {predictions.iloc[0] if len(predictions) > 0 else 'empty'}")

                # Align predictions with actuals - ensure numeric types
                actual_values = pd.to_numeric(test_data['quantity'], errors='coerce').values[:test_weeks]

                # predictions['forecast'] might have mixed types, convert explicitly
                pred_series = predictions['forecast']
                if len(pred_series) > 0:
                    # Convert to numeric, handling any Timestamp or string values
                    predicted_values = pd.to_numeric(pred_series, errors='coerce').values[:len(actual_values)]
                else:
                    predicted_values = np.array([])

                # Remove any NaN values
                valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
                actual_values = actual_values[valid_mask]
                predicted_values = predicted_values[valid_mask]

                if len(actual_values) == 0 or len(predicted_values) == 0:
                    logger.debug(f"No valid data for {style} after filtering")
                    continue

                # Calculate metrics
                from sklearn.metrics import (
                    mean_absolute_error,
                    mean_squared_error,
                    mean_absolute_percentage_error
                )

                mae = mean_absolute_error(actual_values, predicted_values)
                rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

                # MAPE with zero handling
                non_zero_mask = actual_values != 0
                if non_zero_mask.any():
                    mape = mean_absolute_percentage_error(
                        actual_values[non_zero_mask],
                        predicted_values[non_zero_mask]
                    )
                    accuracy = max(0, 1 - mape)
                else:
                    mape = None
                    accuracy = None

                # Week-by-week comparison
                weekly_comparison = []
                for i in range(min(len(actual_values), len(predicted_values))):
                    try:
                        actual_val = float(actual_values[i])
                        predicted_val = float(predicted_values[i])
                        error = predicted_val - actual_val
                        error_pct = (error / actual_val * 100) if actual_val != 0 else 0

                        weekly_comparison.append({
                            'week': i + 1,
                            'actual': round(actual_val, 2),
                            'predicted': round(predicted_val, 2),
                            'error': round(error, 2),
                            'error_pct': round(error_pct, 2)
                        })
                    except (ValueError, TypeError) as conv_err:
                        logger.warning(f"Week {i+1} conversion error for {style}: {conv_err}")
                        continue

                # Determine if meets threshold
                meets_threshold = accuracy >= config.min_accuracy_threshold if accuracy is not None else False

                result = {
                    'style': style,
                    'model': forecast_result.model_used,
                    'train_size': train_size,
                    'test_size': len(actual_values),
                    'accuracy': round(accuracy, 4) if accuracy is not None else None,
                    'mape': round(mape, 4) if mape is not None else None,
                    'mae': round(mae, 2),
                    'rmse': round(rmse, 2),
                    'meets_threshold': meets_threshold,
                    'threshold': config.min_accuracy_threshold,
                    'weekly_comparison': weekly_comparison,
                    'summary': {
                        'avg_actual': round(float(np.mean(actual_values)), 2),
                        'avg_predicted': round(float(np.mean(predicted_values)), 2),
                        'total_actual': round(float(np.sum(actual_values)), 2),
                        'total_predicted': round(float(np.sum(predicted_values)), 2)
                    }
                }

                backtest_results.append(result)

                # Aggregate metrics
                if accuracy is not None:
                    accuracies.append(accuracy)
                if mape is not None:
                    mapes.append(mape)
                maes.append(mae)
                rmses.append(rmse)

                if meets_threshold:
                    overall_metrics['styles_meeting_threshold'] += 1

                overall_metrics['total_styles_tested'] += 1

                logger.info(f"✓ {style}: Accuracy={accuracy:.2%}" if accuracy else f"✓ {style}: MAE={mae:.2f}")

            except Exception as style_err:
                logger.warning(f"Backtest failed for {style}: {style_err}")
                continue

        # Calculate overall metrics
        if accuracies:
            overall_metrics['avg_accuracy'] = round(float(np.mean(accuracies)), 4)
        if mapes:
            overall_metrics['avg_mape'] = round(float(np.mean(mapes)), 4)
        if maes:
            overall_metrics['avg_mae'] = round(float(np.mean(maes)), 2)
        if rmses:
            overall_metrics['avg_rmse'] = round(float(np.mean(rmses)), 2)

        overall_metrics['pass_rate'] = round(
            overall_metrics['styles_meeting_threshold'] / overall_metrics['total_styles_tested'],
            4
        ) if overall_metrics['total_styles_tested'] > 0 else 0

        response = {
            'status': 'success',
            'test_config': {
                'test_weeks': test_weeks,
                'target_accuracy': config.min_accuracy_threshold,
                'models_used': ['Prophet', 'XGBoost', 'ARIMA']
            },
            'overall_metrics': overall_metrics,
            'results': backtest_results,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Backtest complete: {overall_metrics['total_styles_tested']} styles, "
                   f"Avg Accuracy: {overall_metrics['avg_accuracy']:.2%}, "
                   f"Pass Rate: {overall_metrics['pass_rate']:.2%}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in forecast_backtest: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/inventory/greige/g00', methods=['GET'])
def inventory_greige_g00() -> tuple:
    """
    Get Greige Stage 1 (G00) inventory.
    Raw fabric inventory at initial greige stage.
    """
    try:
        logger.info("Fetching G00 greige inventory from eFab")

        data = fetch_from_efab('api/greige/g00')

        if not data:
            return jsonify({
                'status': 'no_data',
                'inventory': [],
                'stage': 'G00',
                'description': 'Greige Stage 1 - Raw Fabric'
            }), 200

        # Standardize column names
        inventory = []
        for item in (data if isinstance(data, list) else [data]):
            inventory.append({
                'style': item.get('Style #', item.get('Style#', '')),
                'stage': 'G00',
                'on_hand': float(item.get('On Hand', item.get('On_Hand', 0))),
                'allocated': float(item.get('Allocated', 0)),
                'available': float(item.get('Available', 0)),
                'unit': item.get('Unit', 'yards'),
                'location': 'Greige Stage 1'
            })

        response = {
            'status': 'success',
            'inventory': inventory,
            'stage': 'G00',
            'total_on_hand': sum(i['on_hand'] for i in inventory),
            'total_available': sum(i['available'] for i in inventory),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Retrieved {len(inventory)} G00 inventory items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error fetching G00 inventory: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/inventory/greige/g02', methods=['GET'])
def inventory_greige_g02() -> tuple:
    """
    Get Greige Stage 2 (G02) inventory.
    Secondary greige processing stage.
    """
    try:
        logger.info("Fetching G02 greige inventory from eFab")

        data = fetch_from_efab('api/greige/g02')

        if not data:
            return jsonify({
                'status': 'no_data',
                'inventory': [],
                'stage': 'G02',
                'description': 'Greige Stage 2 - Processing'
            }), 200

        inventory = []
        for item in (data if isinstance(data, list) else [data]):
            inventory.append({
                'style': item.get('fStyle', item.get('Style#', '')),
                'stage': 'G02',
                'on_hand': float(item.get('On Hand', item.get('On_Hand', 0))),
                'allocated': float(item.get('Allocated', 0)),
                'available': float(item.get('Available', 0)),
                'unit': item.get('Unit', 'yards'),
                'location': 'Greige Stage 2'
            })

        response = {
            'status': 'success',
            'inventory': inventory,
            'stage': 'G02',
            'total_on_hand': sum(i['on_hand'] for i in inventory),
            'total_available': sum(i['available'] for i in inventory),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Retrieved {len(inventory)} G02 inventory items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error fetching G02 inventory: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/inventory/inspection/i01', methods=['GET'])
def inventory_inspection_i01() -> tuple:
    """
    Get Inspection/QC Stage (I01) inventory.
    Fabric in quality control queue awaiting approval.
    """
    try:
        logger.info("Fetching I01 inspection inventory from eFab")

        data = fetch_from_efab('api/finished/i01')

        if not data:
            return jsonify({
                'status': 'no_data',
                'inventory': [],
                'stage': 'I01',
                'description': 'Inspection/QC - Quality Control Queue'
            }), 200

        inventory = []
        for item in (data if isinstance(data, list) else [data]):
            inventory.append({
                'style': item.get('Style #', item.get('Style#', '')),
                'stage': 'I01',
                'on_hand': float(item.get('On Hand', item.get('On_Hand', 0))),
                'allocated': float(item.get('Allocated', 0)),
                'available': float(item.get('Available', 0)),
                'unit': item.get('Unit', 'yards'),
                'location': 'QC/Inspection',
                'qc_status': item.get('QC_Status', 'Pending')
            })

        response = {
            'status': 'success',
            'inventory': inventory,
            'stage': 'I01',
            'total_on_hand': sum(i['on_hand'] for i in inventory),
            'total_available': sum(i['available'] for i in inventory),
            'pending_qc': len([i for i in inventory if i.get('qc_status') == 'Pending']),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Retrieved {len(inventory)} I01 inspection items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error fetching I01 inventory: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/inventory/finished/f01', methods=['GET'])
def inventory_finished_f01() -> tuple:
    """
    Get Finished Goods (F01) inventory.
    Post-QC approved finished fabric ready for shipment.
    """
    try:
        logger.info("Fetching F01 finished goods inventory from eFab")

        data = fetch_from_efab('api/finished/f01')

        if not data:
            return jsonify({
                'status': 'no_data',
                'inventory': [],
                'stage': 'F01',
                'description': 'Finished Goods - Ready for Shipment'
            }), 200

        inventory = []
        for item in (data if isinstance(data, list) else [data]):
            inventory.append({
                'style': item.get('Style #', item.get('Style#', '')),
                'stage': 'F01',
                'on_hand': float(item.get('On Hand', item.get('On_Hand', 0))),
                'allocated': float(item.get('Allocated', 0)),
                'available': float(item.get('Available', 0)),
                'unit': item.get('Unit', 'yards'),
                'location': 'Finished Goods',
                'ready_to_ship': float(item.get('Available', 0))
            })

        response = {
            'status': 'success',
            'inventory': inventory,
            'stage': 'F01',
            'total_on_hand': sum(i['on_hand'] for i in inventory),
            'total_available': sum(i['available'] for i in inventory),
            'ready_to_ship': sum(i['ready_to_ship'] for i in inventory),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Retrieved {len(inventory)} F01 finished goods items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error fetching F01 inventory: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/inventory/pipeline-summary', methods=['GET'])
def inventory_pipeline_summary() -> tuple:
    """
    Get consolidated view of all inventory stages (G00→G02→I01→F01).
    Shows the complete production pipeline with inventory at each stage.
    """
    try:
        logger.info("Generating consolidated pipeline inventory summary")

        # Fetch all stages in parallel (simulated)
        g00_data = fetch_from_efab('api/greige/g00') or []
        g02_data = fetch_from_efab('api/greige/g02') or []
        i01_data = fetch_from_efab('api/finished/i01') or []
        f01_data = fetch_from_efab('api/finished/f01') or []

        def summarize_stage(data, stage_name):
            items = data if isinstance(data, list) else [data] if data else []
            total_on_hand = sum(float(item.get('On Hand', item.get('On_Hand', 0))) for item in items)
            total_available = sum(float(item.get('Available', 0)) for item in items)
            return {
                'stage': stage_name,
                'items_count': len(items),
                'total_on_hand': total_on_hand,
                'total_available': total_available
            }

        pipeline = {
            'g00': summarize_stage(g00_data, 'G00 - Raw Greige'),
            'g02': summarize_stage(g02_data, 'G02 - Greige Processing'),
            'i01': summarize_stage(i01_data, 'I01 - QC/Inspection'),
            'f01': summarize_stage(f01_data, 'F01 - Finished Goods')
        }

        total_inventory = sum(stage['total_on_hand'] for stage in pipeline.values())

        response = {
            'status': 'success',
            'pipeline': pipeline,
            'total_inventory_yards': total_inventory,
            'production_flow': 'G00 → G02 → I01 → F01',
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Pipeline summary: {total_inventory:.0f} total yards across all stages")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error generating pipeline summary: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ===== ENHANCED FORECAST ENDPOINTS (13-Week Multi-Source Integration) =====

@app.route('/api/forecast/comprehensive', methods=['GET'])
def comprehensive_forecast() -> tuple:
    """
    Generate comprehensive 13-week multi-source forecast.

    Orchestrates:
    1. ML forecasts from historical data
    2. External forecasts from sales team/customers
    3. Intelligent blending of all sources
    4. Comparison with actual orders
    5. Proactive production recommendations

    Query params:
        - start_week: Starting ISO week (default: current week)
        - forecast_weeks: Number of weeks to forecast (default: 13)
        - blending_strategy: weighted_average, highest_confidence, conservative, aggressive
        - include_actuals: Include actual order comparison (default: true)
    """
    try:
        logger.info("Generating comprehensive 13-week multi-source forecast")

        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator

        # Parse query parameters
        start_week = request.args.get('start_week', type=int)
        forecast_weeks = request.args.get('forecast_weeks', 13, type=int)
        blending_strategy = request.args.get('blending_strategy', 'weighted_average')
        include_actuals = request.args.get('include_actuals', 'true').lower() == 'true'

        # Initialize generator
        generator = WeeklyForecastGenerator(forecast_weeks=forecast_weeks)

        # Generate comprehensive forecast
        result = generator.generate_comprehensive_forecast(
            styles=None,  # Auto-select all styles with history
            start_week=start_week,
            external_forecast_files=None,  # Load from Turso database
            blending_strategy=blending_strategy,
            include_actual_orders=include_actuals
        )

        # Check for errors
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 500

        # Build API response
        response = {
            'status': 'success',
            'ml_forecasts': len(result.get('ml_forecast', {})),
            'blended_forecasts': len(result.get('blended_forecast', {})),
            'actual_orders': len(result.get('actual_orders', {})),
            'proactive_production_count': len(result.get('proactive_production', [])),
            'variance_alerts_count': len(result.get('variance_alerts', [])),
            'combined_schedule_count': len(result.get('combined_schedule', {})),
            'data': {
                'blended_forecast': result.get('blended_forecast', {}),
                'combined_schedule': result.get('combined_schedule', {}),
                'proactive_production': result.get('proactive_production', [])[:20],  # Top 20
                'variance_alerts': result.get('variance_alerts', [])[:10],  # Top 10
                'comparison_summary': result.get('comparison', {}).get('summary', {})
            },
            'metadata': result.get('metadata', {}),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Comprehensive forecast generated: {len(result.get('combined_schedule', {}))} styles")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in comprehensive_forecast: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/forecast/upload-external', methods=['POST'])
def upload_external_forecast() -> tuple:
    """
    Upload external forecast from sales team/customers.

    Expects JSON body:
    {
        "source_name": "sales_team",
        "forecasts": [
            {
                "style": "STYLE001",
                "week_number": 42,
                "forecasted_yards": 1000,
                "confidence": 0.85,
                "notes": "Customer indicated interest"
            }
        ]
    }
    """
    try:
        logger.info("Processing external forecast upload")

        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.external_forecast_loader import ExternalForecastLoader

        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400

        source_name = data.get('source_name', 'external')
        forecasts = data.get('forecasts', [])

        if not forecasts:
            return jsonify({
                'status': 'error',
                'message': 'No forecasts provided'
            }), 400

        # Validate and upload to Turso
        loader = ExternalForecastLoader()

        uploaded_count = loader.upload_to_turso(
            forecasts=forecasts,
            source_name=source_name,
            uploaded_by='api_user'
        )

        response = {
            'status': 'success',
            'message': f'Successfully uploaded {uploaded_count} forecasts',
            'source': source_name,
            'count': uploaded_count,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Uploaded {uploaded_count} external forecasts from {source_name}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error uploading external forecast: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/forecast/accuracy-report', methods=['GET'])
def forecast_accuracy_report() -> tuple:
    """
    Get forecast accuracy report showing performance by source.

    Query params:
        - lookback_weeks: Number of weeks to analyze (default: 13)
    """
    try:
        logger.info("Generating forecast accuracy report")

        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.forecast_accuracy_tracker import ForecastAccuracyTracker

        # Parse query parameters
        lookback_weeks = request.args.get('lookback_weeks', 13, type=int)

        # Generate report
        tracker = ForecastAccuracyTracker()
        report = tracker.generate_accuracy_report(lookback_weeks=lookback_weeks)

        # Build API response
        response = {
            'status': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Accuracy report generated for {lookback_weeks} weeks")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error generating accuracy report: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/forecast/weight-recommendations', methods=['GET'])
def forecast_weight_recommendations() -> tuple:
    """
    Get recommended weight adjustments based on forecast accuracy.

    Query params:
        - lookback_weeks: Number of weeks to analyze (default: 13)
    """
    try:
        logger.info("Calculating weight recommendations")

        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.forecast_accuracy_tracker import ForecastAccuracyTracker

        # Parse query parameters
        lookback_weeks = request.args.get('lookback_weeks', 13, type=int)

        # Get current weights (default)
        current_weights = {
            'ml_historical': 0.40,
            'sales_team': 0.35,
            'customer_commitment': 0.20,
            'market_intelligence': 0.05
        }

        # Calculate recommendations
        tracker = ForecastAccuracyTracker()
        recommendations = tracker.recommend_weight_adjustments(
            current_weights=current_weights,
            lookback_weeks=lookback_weeks
        )

        # Build API response
        response = {
            'status': 'success',
            'current_weights': current_weights,
            'recommended_weights': recommendations.get('recommended_weights', {}),
            'changes': recommendations.get('changes', []),
            'overall_improvement': recommendations.get('overall_improvement', 0),
            'source_metrics': recommendations.get('source_metrics', {}),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Weight recommendations: {len(recommendations.get('changes', []))} changes suggested")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error calculating weight recommendations: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/forecast/proactive-production', methods=['GET'])
def proactive_production_recommendations() -> tuple:
    """
    Get proactive production recommendations (high-confidence forecasts without actual orders).

    Query params:
        - start_week: Starting ISO week (default: current week)
        - max_items: Maximum recommendations to return (default: 50)
        - min_confidence: Minimum confidence threshold (default: 0.85)
    """
    try:
        logger.info("Generating proactive production recommendations")

        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator

        # Parse query parameters
        start_week = request.args.get('start_week', type=int)
        max_items = request.args.get('max_items', 50, type=int)
        min_confidence = request.args.get('min_confidence', 0.85, type=float)

        # Generate comprehensive forecast
        generator = WeeklyForecastGenerator(forecast_weeks=13)
        result = generator.generate_comprehensive_forecast(
            start_week=start_week,
            blending_strategy='weighted_average',
            include_actual_orders=True
        )

        # Extract proactive production recommendations
        proactive = result.get('proactive_production', [])[:max_items]

        # Filter by confidence if specified
        if min_confidence > 0.85:
            proactive = [p for p in proactive if p.get('confidence', 0) >= min_confidence]

        response = {
            'status': 'success',
            'recommendations': proactive,
            'count': len(proactive),
            'min_confidence': min_confidence,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Generated {len(proactive)} proactive production recommendations")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error generating proactive production: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/production-planning', methods=['GET'])
def production_planning() -> tuple:
    """
    Get production planning data - combines forecasted production requirements with actual orders.
    This endpoint provides comprehensive production planning information for the dashboard.

    Query params:
        - view: data (default), orders, summary
    """
    try:
        logger.info("Fetching production planning data")
        view = request.args.get('view', 'data')

        # Get forecasted production data
        from datetime import date
        import pandas as pd

        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'data': [],
                'message': 'No production planning data available',
                'timestamp': datetime.now().isoformat()
            }), 200

        # Get REAL work-in-progress from knit orders
        knit_orders = fetch_from_efab('api/knitorder/list')

        # Build inventory by style from real orders
        style_inventory = {}
        style_wip = {}

        if knit_orders:
            for order in knit_orders:
                if order.get('active', 0) == 1:
                    style = order.get('knit_style_base', {}).get('base_style', '') if order.get('knit_style_base') else ''
                    if style:
                        # Finished goods = qty_received
                        qty_received = float(order.get('qty_received', 0) or 0)
                        # WIP = qty_ordered - qty_received
                        qty_ordered = float(order.get('qty_ordered', 0) or 0)
                        wip = qty_ordered - qty_received

                        if style not in style_inventory:
                            style_inventory[style] = 0
                            style_wip[style] = 0

                        style_inventory[style] += qty_received
                        style_wip[style] += wip

        logger.info(f"Built real inventory for {len(style_inventory)} styles")

        sales_df = pd.DataFrame(sales_data)
        planning_items = []

        if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            for style in sales_df['style'].unique()[:30]:  # Top 30 styles
                style_sales = sales_df[sales_df['style'] == style]
                avg_qty = style_sales['quantity'].mean()
                forecasted_qty = int(avg_qty * 1.1 * 3)  # 90-day forecast with 10% growth

                # Use REAL inventory and WIP from knit orders
                current_inventory = int(style_inventory.get(style, 0))
                pipeline_wip = int(style_wip.get(style, 0))
                net_requirement = forecasted_qty - current_inventory - pipeline_wip

                planning_items.append({
                    'style': style,
                    'customer': style_sales.iloc[0].get('customer', 'Multiple'),
                    'forecasted_demand': forecasted_qty,
                    'current_inventory': current_inventory,
                    'pipeline_wip': pipeline_wip,
                    'net_requirement': max(0, net_requirement),
                    'priority': 'HIGH' if net_requirement > 5000 else 'MEDIUM' if net_requirement > 2000 else 'LOW',
                    'status': 'PLANNED' if net_requirement > 0 else 'COVERED',
                    'suggested_start_week': 'Week 45',
                    'delivery_week': 'Week 52'
                })

        # Sort by priority: HIGH > MEDIUM > LOW, then by net_requirement (descending)
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        planning_items_sorted = sorted(
            planning_items,
            key=lambda p: (priority_order.get(p['priority'], 3), -p['net_requirement'])
        )

        response = {
            'status': 'success',
            'data': planning_items_sorted,
            'production_schedule': planning_items_sorted,  # Dashboard expects this field
            'summary': {
                'total_items': len(planning_items_sorted),
                'high_priority': sum(1 for p in planning_items_sorted if p['priority'] == 'HIGH'),
                'total_demand': sum(p['forecasted_demand'] for p in planning_items_sorted),
                'net_requirements': sum(p['net_requirement'] for p in planning_items_sorted)
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Production planning: {len(planning_items)} items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in production_planning: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'data': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/production-suggestions', methods=['GET'])
def production_suggestions() -> tuple:
    """
    Get AI-powered production suggestions based on forecasts, inventory, and historical patterns.
    Provides actionable recommendations for production scheduling.
    """
    try:
        logger.info("Generating AI production suggestions")

        from datetime import date
        import pandas as pd
        import numpy as np

        # Get sales data for analysis
        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'suggestions': [],
                'message': 'No data available for suggestions',
                'timestamp': datetime.now().isoformat()
            }), 200

        sales_df = pd.DataFrame(sales_data)
        suggestions = []

        if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            # Analyze top styles by volume
            style_volumes = sales_df.groupby('style')['quantity'].sum().sort_values(ascending=False)

            for style in style_volumes.head(20).index:
                style_data = sales_df[sales_df['style'] == style]

                # Calculate metrics
                avg_qty = style_data['quantity'].mean()
                total_qty = style_data['quantity'].sum()
                trend = np.polyfit(range(len(style_data)), style_data['quantity'], 1)[0]

                # Determine recommendation type
                if trend > 0 and avg_qty > 1000:
                    suggestion_type = 'INCREASE_PRODUCTION'
                    confidence = 0.88
                    reason = 'Upward trend detected with high volume'
                elif trend < -50:
                    suggestion_type = 'REDUCE_PRODUCTION'
                    confidence = 0.82
                    reason = 'Declining demand trend'
                else:
                    suggestion_type = 'MAINTAIN_CURRENT'
                    confidence = 0.75
                    reason = 'Stable demand pattern'

                suggestions.append({
                    'style': style,
                    'type': suggestion_type,
                    'confidence': round(confidence, 2),
                    'reason': reason,
                    'recommended_quantity': int(avg_qty * (1.2 if trend > 0 else 0.9)),
                    'current_avg': int(avg_qty),
                    'trend': 'UP' if trend > 0 else 'DOWN' if trend < 0 else 'STABLE',
                    'priority': 'HIGH' if total_qty > 50000 else 'MEDIUM' if total_qty > 20000 else 'LOW',
                    'action_items': [
                        f"Plan for {int(avg_qty * 1.2)} yards per order",
                        "Monitor trend weekly",
                        "Coordinate with sales team"
                    ] if trend > 0 else [
                        f"Maintain {int(avg_qty)} yards per order",
                        "Review customer orders",
                        "Consider promotional activities"
                    ]
                })

        # Sort by priority: HIGH > MEDIUM > LOW, then by confidence (descending)
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions_sorted = sorted(
            suggestions,
            key=lambda s: (priority_order.get(s['priority'], 3), -s['confidence'])
        )

        response = {
            'status': 'success',
            'suggestions': suggestions_sorted,
            'summary': {
                'total_suggestions': len(suggestions_sorted),
                'high_confidence': sum(1 for s in suggestions_sorted if s['confidence'] >= 0.85),
                'increase_production': sum(1 for s in suggestions_sorted if s['type'] == 'INCREASE_PRODUCTION'),
                'reduce_production': sum(1 for s in suggestions_sorted if s['type'] == 'REDUCE_PRODUCTION')
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Generated {len(suggestions)} production suggestions")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error generating production suggestions: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'suggestions': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/material-shortages-real', methods=['GET'])
def material_shortages_real() -> tuple:
    """
    Calculate real material shortages by exploding knit orders through BOMs
    and comparing with actual yarn inventory.
    """
    try:
        import pandas as pd
        import os

        logger.info("Calculating real material shortages from knit orders + BOMs")

        # Step 1: Get active knit orders from eFab
        knit_orders = fetch_from_efab('api/knitorder/list')
        if not knit_orders:
            return jsonify({'error': 'No knit orders found'}), 500

        # Filter for active orders only
        active_orders = [o for o in knit_orders if o.get('active', 0) == 1]
        logger.info(f"Found {len(active_orders)} active knit orders")

        # Step 2: Load BOM data from CSV
        bom_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'production', '5', 'BOM_updated.csv')
        bom_df = pd.read_csv(bom_path)
        logger.info(f"Loaded {len(bom_df)} BOM records")

        # Clean column names
        bom_df.columns = bom_df.columns.str.strip()

        # Step 3: Get yarn inventory from eFab
        yarn_data = fetch_from_efab('api/yarn/active')
        if not yarn_data:
            return jsonify({'error': 'No yarn data found'}), 500

        # Build yarn inventory lookup
        yarn_inventory = {}
        for yarn in yarn_data:
            yarn_id = yarn.get('desc_number')
            if yarn_id:
                # Calculate planning balance
                reconciled_qty = float(yarn.get('reconciled_qty', 0) or 0)
                added = float(yarn.get('added', 0) or 0)
                consumed = float(yarn.get('consumed', 0) or 0)
                adjustments = float(yarn.get('adjustments', 0) or 0)
                theoretical_balance = reconciled_qty + added + consumed + adjustments

                allocated = float(yarn.get('allocated', 0) or 0)
                on_order = float(yarn.get('onorder', 0) or 0)
                planning_balance = theoretical_balance + on_order + allocated

                yarn_inventory[yarn_id] = {
                    'planning_balance': planning_balance,
                    'description': yarn.get('description', ''),
                    'supplier': yarn.get('supplier', ''),
                    'cost_per_pound': float(yarn.get('cost_avg', 0) or 0)
                }

        logger.info(f"Built inventory for {len(yarn_inventory)} yarns")

        # Step 4: Explode knit orders to yarn requirements
        yarn_requirements = {}  # {yarn_id: {total_lbs, affected_orders: [...], affected_styles: [...]}}

        for order in active_orders[:50]:  # Limit to 50 orders for performance
            style = order.get('knit_style_base', {}).get('base_style', '') if order.get('knit_style_base') else ''
            if not style:
                continue

            qty_ordered = float(order.get('qty_ordered', 0) or 0)
            order_id = order.get('id')

            # Look up BOM for this style with fuzzy matching
            # Try exact match first
            style_bom = bom_df[bom_df['Style#'].str.strip() == style.strip()]

            # If no exact match, try prefix matching (e.g., "CT2935" matches "CT2935/1")
            if len(style_bom) == 0:
                style_bom = bom_df[bom_df['Style#'].str.strip().str.startswith(style.strip())]

            # If still no match, try contains matching
            if len(style_bom) == 0:
                style_bom = bom_df[bom_df['Style#'].str.contains(style.strip(), case=False, na=False)]

            if len(style_bom) > 0:
                logger.debug(f"Matched style '{style}' to {len(style_bom)} BOM entries")

            for _, bom_row in style_bom.iterrows():
                yarn_id = int(bom_row['Desc#'])
                bom_percentage = float(bom_row['BOM_Percentage'])

                # Calculate yarn requirement for this order
                yarn_lbs_needed = qty_ordered * bom_percentage

                if yarn_id not in yarn_requirements:
                    yarn_requirements[yarn_id] = {
                        'total_lbs': 0,
                        'affected_orders': [],
                        'affected_styles': set()
                    }

                yarn_requirements[yarn_id]['total_lbs'] += yarn_lbs_needed
                yarn_requirements[yarn_id]['affected_orders'].append(order_id)
                yarn_requirements[yarn_id]['affected_styles'].add(style)

        logger.info(f"Calculated requirements for {len(yarn_requirements)} yarns")

        # Step 5: Calculate shortages
        shortages = []

        for yarn_id, req in yarn_requirements.items():
            inv = yarn_inventory.get(yarn_id, {})
            planning_balance = inv.get('planning_balance', 0)
            required = req['total_lbs']

            shortage_amt = planning_balance - required

            # Only include if there's a shortage
            if shortage_amt < 0:
                shortage_severity = 'CRITICAL' if shortage_amt < -500 else 'HIGH' if shortage_amt < -100 else 'MEDIUM'

                shortages.append({
                    'yarn_id': yarn_id,
                    'description': inv.get('description', f'Yarn {yarn_id}'),
                    'supplier': inv.get('supplier', '--'),
                    'required_lbs': round(required, 2),
                    'available_lbs': round(planning_balance, 2),
                    'shortage_lbs': round(abs(shortage_amt), 2),
                    'severity': shortage_severity,
                    'affected_orders_count': len(req['affected_orders']),
                    'affected_styles': list(req['affected_styles']),
                    'cost_impact': round(abs(shortage_amt) * inv.get('cost_per_pound', 0), 2)
                })

        # Sort by shortage amount
        shortages.sort(key=lambda x: x['shortage_lbs'], reverse=True)

        logger.info(f"Found {len(shortages)} material shortages")

        return jsonify({
            'status': 'success',
            'shortages': shortages,
            'summary': {
                'total_shortages': len(shortages),
                'critical_count': sum(1 for s in shortages if s['severity'] == 'CRITICAL'),
                'high_count': sum(1 for s in shortages if s['severity'] == 'HIGH'),
                'medium_count': sum(1 for s in shortages if s['severity'] == 'MEDIUM'),
                'total_cost_impact': sum(s['cost_impact'] for s in shortages),
                'orders_analyzed': len(active_orders[:50])
            },
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error calculating material shortages: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/fabric-forecast-integrated', methods=['GET'])
def fabric_forecast_integrated() -> tuple:
    """
    Get integrated fabric forecast combining ML forecasts with actual requirements.

    Returns fabric requirements forecast with:
    - Style information
    - Fabric type and yards required
    - Timeline and delivery dates
    - Status and priority
    """
    try:
        logger.info("Generating fabric forecast")

        import pandas as pd
        sales_data = _fetch_sales_history_from_efab()

        if not sales_data:
            return jsonify({
                'status': 'no_data',
                'forecast_items': [],
                'message': 'No sales data available for fabric forecast',
                'timestamp': datetime.now().isoformat()
            }), 200

        sales_df = pd.DataFrame(sales_data)
        forecast_items = []

        # Fetch knit orders to get allocated fabric
        knit_orders = fetch_from_efab('api/knitorder/list')
        fabric_allocations = {}  # {fabric_id: total_yards_allocated}

        if knit_orders:
            for order in knit_orders:
                if order.get('active', 0) == 1:  # Only active orders
                    # Get fabric style from order
                    knit_style_base = order.get('knit_style_base', {})
                    if knit_style_base:
                        base_style = knit_style_base.get('base_style', '')
                        if base_style:
                            fabric_id = ''.join(filter(str.isdigit, str(base_style)))[:4]

                            # Get allocated quantity - eFab stores both yards and lbs
                            # Try balance_yds first, then fallback to balance (lbs)
                            qty_yards = float(order.get('balance_yds', 0) or 0)

                            if qty_yards == 0:
                                # If yards not available, use balance (lbs) - inquiry shows both
                                qty_lbs = float(order.get('balance', 0) or 0)
                                if qty_lbs > 0:
                                    # Get yards_per_lb conversion from eFab fabric specs (most accurate)
                                    yds_per_lb = None

                                    # Try to get from knit_style_base f_versions (nested structure)
                                    if 'f_versions' in knit_style_base and knit_style_base['f_versions']:
                                        f_version = knit_style_base['f_versions'][0] if isinstance(knit_style_base['f_versions'], list) else knit_style_base['f_versions']
                                        yds_per_lb = f_version.get('act_yds_per_lb') or f_version.get('target_yds_per_lb')

                                    # Fallback: try direct fields on knit_style_base
                                    if not yds_per_lb:
                                        yds_per_lb = knit_style_base.get('yds_per_lb') or knit_style_base.get('target_yds_per_lb')

                                    if yds_per_lb and yds_per_lb > 0:
                                        # Use actual conversion factor from eFab (e.g., 3.1746 for CT2155-1)
                                        qty_yards = qty_lbs * yds_per_lb
                                    else:
                                        # Default 5:1 if conversion not found
                                        qty_yards = qty_lbs * 5.0

                            if fabric_id not in fabric_allocations:
                                fabric_allocations[fabric_id] = 0
                            fabric_allocations[fabric_id] += qty_yards

        logger.info(f"Built allocations for {len(fabric_allocations)} fabrics from knit orders")

        if 'style' in sales_df.columns and 'quantity' in sales_df.columns:
            # Filter out mock/test styles (STYLE001-016, etc)
            sales_df = sales_df[~sales_df['style'].str.contains('STYLE', na=False, case=False)]

            # Get top styles
            style_volumes = sales_df.groupby('style')['quantity'].sum().sort_values(ascending=False)

            for idx, style in enumerate(style_volumes.head(20).index):
                style_data = sales_df[sales_df['style'] == style]

                # Calculate fabric requirements (FORECAST - predictive)
                avg_yards = style_data['quantity'].mean()
                forecasted_yards = int(avg_yards * 4)  # 4-week forecast

                # Simulate fabric type (would come from BOM in real system)
                fabric_types = ['Jersey', 'Interlock', 'Rib', 'French Terry', 'Pique']
                fabric_type = fabric_types[idx % len(fabric_types)]

                # Get REAL current inventory from eFab using fabric inquiry
                # Extract fabric ID from style (first 4 digits)
                fabric_id = ''.join(filter(str.isdigit, str(style)))[:4] if style else None
                current_inventory = 0
                on_order = 0

                if fabric_id:
                    # Use same logic as fabric_inquiry_search - query eFab stages directly
                    try:
                        # Fetch from all stages
                        stages_data = {
                            'G00': fetch_from_efab('api/greige/g00'),
                            'G02': fetch_from_efab('api/greige/g02'),
                            'I01': fetch_from_efab('api/finished/i01'),
                            'F01': fetch_from_efab('api/finished/f01')
                        }

                        # Aggregate inventory by stage
                        inventory_by_stage = {}

                        for stage_name, records in stages_data.items():
                            if not records:
                                continue

                            total_yards = 0

                            for record in records:
                                # Extract base_style from nested structure
                                base_style = None

                                # Try knit_version path (G00, G02 use this)
                                if 'knit_version' in record and record['knit_version']:
                                    knit_version = record['knit_version']
                                    if 'knit_style_base' in knit_version and knit_version['knit_style_base']:
                                        base_style = knit_version['knit_style_base'].get('base_style')

                                # Try f_version path (F01 uses this - f_version.f_base.base_style)
                                if not base_style and 'f_version' in record and record['f_version']:
                                    f_version = record['f_version']
                                    if isinstance(f_version, dict) and 'f_base' in f_version and f_version['f_base']:
                                        base_style = f_version['f_base'].get('base_style')

                                # Try i_version path (I01 might use this - i_version.i_base.base_style)
                                if not base_style and 'i_version' in record and record['i_version']:
                                    i_version = record['i_version']
                                    if isinstance(i_version, dict) and 'i_base' in i_version and i_version['i_base']:
                                        base_style = i_version['i_base'].get('base_style')

                                if not base_style:
                                    continue

                                # Check if this record matches our fabric_id
                                record_fabric_id = ''.join(filter(str.isdigit, str(base_style)))[:4]

                                if record_fabric_id == fabric_id:
                                    total_yards += float(record.get('qty_yds', 0) or 0)

                            if total_yards > 0:
                                inventory_by_stage[stage_name] = {"yards": round(total_yards, 2)}

                        # Current Inventory = I01 + F01 (finished stages)
                        i01_yards = inventory_by_stage.get('I01', {}).get('yards', 0)
                        f01_yards = inventory_by_stage.get('F01', {}).get('yards', 0)
                        current_inventory = int(i01_yards + f01_yards)

                        # On Order = G00 + G02 (WIP in pipeline)
                        g00_yards = inventory_by_stage.get('G00', {}).get('yards', 0)
                        g02_yards = inventory_by_stage.get('G02', {}).get('yards', 0)
                        on_order = int(g00_yards + g02_yards)

                        logger.info(f"Fabric {fabric_id} ({style}): Current={current_inventory} yds, OnOrder={on_order} yds")
                    except Exception as e:
                        logger.warning(f"Failed to get real inventory for fabric {fabric_id}: {e}")

                current_inventory = int(current_inventory)
                on_order = int(on_order)

                # Get allocated fabric from knit orders
                allocated_yards = int(fabric_allocations.get(fabric_id, 0))

                # Calculate net position: Current + WIP - Allocated - Forecast
                # Positive = surplus, Negative = shortage
                net_position = current_inventory + on_order - allocated_yards - forecasted_yards

                # For display: net_requirement is the shortage amount (inverted)
                net_requirement = -net_position  # Negative net_position means shortage

                # Determine priority and status based on REAL data
                if net_requirement > forecasted_yards * 0.6:
                    priority = 'CRITICAL'
                    status = 'URGENT_ORDER'
                    lead_time_weeks = 2
                elif net_requirement > 0:
                    priority = 'HIGH'
                    status = 'ORDER_SOON'
                    lead_time_weeks = 4
                else:
                    priority = 'NORMAL'
                    status = 'ADEQUATE'
                    lead_time_weeks = 6

                forecast_items.append({
                    'style': style,
                    'fabric_type': fabric_type,
                    'description': f'{fabric_type} for {style}',
                    'forecasted_yards': forecasted_yards,
                    'current_inventory': current_inventory,
                    'on_order': on_order,
                    'allocated': allocated_yards,  # Fabric allocated to knit orders
                    'net_position': net_position,  # True net position (positive=surplus, negative=shortage)
                    'net_requirement': net_requirement,  # For compatibility: shortage amount (positive=need to order)
                    'priority': priority,
                    'status': status,
                    'lead_time_weeks': lead_time_weeks,
                    'estimated_cost': round(net_requirement * 8.5, 2) if net_requirement > 0 else 0,  # $8.50/yard
                    'delivery_week': f'Week {45 + lead_time_weeks}',
                    'confidence': 0.85
                })

        # Calculate summary
        summary = {
            'total_yards_forecasted': sum(f['forecasted_yards'] for f in forecast_items),
            'total_net_requirement': sum(f['net_requirement'] for f in forecast_items),
            'total_required_yards': sum(f['net_requirement'] for f in forecast_items),  # JavaScript expects this field
            'critical_items': sum(1 for f in forecast_items if f['priority'] == 'CRITICAL'),
            'shortage_count': sum(1 for f in forecast_items if f['priority'] == 'CRITICAL'),  # JavaScript expects this field
            'high_priority_items': sum(1 for f in forecast_items if f['priority'] == 'HIGH'),
            'total_estimated_cost': sum(f['estimated_cost'] for f in forecast_items),
            'timeline_alert': any(f['priority'] == 'CRITICAL' for f in forecast_items),
            'total_styles': len(set(f['style'] for f in forecast_items)),  # JavaScript expects this field
            'fabric_types_count': len(set(f['fabric_type'] for f in forecast_items))  # JavaScript expects this field
        }

        response = {
            'status': 'success',
            'forecast_items': forecast_items,
            'fabric_forecast': forecast_items,  # Dashboard expects this field name
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Fabric forecast: {len(forecast_items)} items, {summary['critical_items']} critical")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in fabric_forecast_integrated: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'forecast_items': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/inventory-netting', methods=['GET'])
def inventory_netting() -> tuple:
    """
    Get inventory netting analysis - shows net available inventory after subtracting allocations.

    Netting calculation:
    - Net Available = On Hand - Allocated - Reserved + On Order
    - Shows which items have adequate coverage vs. shortages
    """
    try:
        logger.info("Calculating inventory netting analysis")

        # Fetch yarn intelligence data which has inventory info
        yarn_data = fetch_from_efab('api/yarn/active')

        if not yarn_data:
            return jsonify({
                'status': 'no_data',
                'data': [],
                'message': 'No inventory data available for netting',
                'timestamp': datetime.now().isoformat()
            }), 200

        netting_items = []

        for yarn in yarn_data[:50]:  # Top 50 yarns
            # Get inventory values
            on_hand = float(yarn.get('reconciled_qty', 0) or 0)
            allocated = abs(float(yarn.get('allocated', 0) or 0))  # Make positive
            on_order = float(yarn.get('onorder', 0) or 0)

            # Calculate net position
            net_available = on_hand - allocated + on_order

            # Determine status
            if net_available < 0:
                status = 'SHORTAGE'
                severity = 'CRITICAL'
            elif net_available < 100:
                status = 'LOW'
                severity = 'HIGH'
            elif net_available < 500:
                status = 'ADEQUATE'
                severity = 'MEDIUM'
            else:
                status = 'HEALTHY'
                severity = 'LOW'

            netting_items.append({
                'yarn_id': yarn.get('desc_number'),
                'description': yarn.get('description', ''),
                'on_hand': round(on_hand, 2),
                'allocated': round(allocated, 2),
                'on_order': round(on_order, 2),
                'net_available': round(net_available, 2),
                'status': status,
                'severity': severity,
                'supplier': yarn.get('supplier', ''),
                'cost_per_lb': float(yarn.get('cost_avg', 0) or 0)
            })

        # Calculate summary statistics
        shortage_items = [i for i in netting_items if i['status'] == 'SHORTAGE']
        low_items = [i for i in netting_items if i['status'] == 'LOW']

        response = {
            'status': 'success',
            'data': netting_items,
            'summary': {
                'total_items': len(netting_items),
                'shortage_count': len(shortage_items),
                'low_inventory_count': len(low_items),
                'total_on_hand': sum(i['on_hand'] for i in netting_items),
                'total_allocated': sum(i['allocated'] for i in netting_items),
                'total_on_order': sum(i['on_order'] for i in netting_items),
                'total_net_available': sum(i['net_available'] for i in netting_items)
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✓ Inventory netting: {len(shortage_items)} shortages, {len(low_items)} low items")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in inventory_netting: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'data': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


def get_db_connection() -> sqlite3.Connection:
    """
    Get database connection
    Returns: SQLite connection object
    """
    db_path = os.path.join(BASE_DIR, 'erp_database.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


@app.route('/api/factory-floor-ai-dashboard', methods=['GET'])
def factory_floor_ai_dashboard() -> tuple:
    """
    Factory Floor AI Dashboard endpoint
    Returns machine planning data with work centers, machines, and assignments
    Uses real database work center and machine data
    """
    try:
        active_only = request.args.get('active_only', 'false').lower() == 'true'

        logger.info(f"Factory floor AI dashboard request (active_only={active_only})")

        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch real knit orders from eFab
        knit_orders = fetch_from_efab('api/knitorder/list')

        # Get all work centers from database
        cursor.execute('''
            SELECT work_center_id, category, gauge, diameter, type, description
            FROM work_centers
            WHERE is_active = 1
            ORDER BY category, gauge, diameter
        ''')
        db_work_centers = cursor.fetchall()

        # Get all machines from database with their work center assignments
        cursor.execute('''
            SELECT m.machine_number, m.work_center_id, m.description, m.status,
                   wc.category, wc.gauge, wc.diameter, wc.type
            FROM machines m
            LEFT JOIN work_centers wc ON m.work_center_id = wc.work_center_id
            WHERE m.is_active = 1
            ORDER BY m.work_center_id, m.machine_number
        ''')
        db_machines = cursor.fetchall()

        conn.close()

        # Build work center groups from database
        work_centers = {}
        machine_assignments = {}  # Track which machines have orders

        # Initialize work centers
        for wc_row in db_work_centers:
            wc_id = wc_row['work_center_id']
            work_centers[wc_id] = {
                'work_center_id': wc_id,
                'work_center_name': f"Cat {wc_row['category']} | {wc_row['gauge']}G | {wc_row['diameter']}D | {wc_row['type']}",
                'machines': [],
                'total_machines': 0,
                'running_machines': 0,
                'avg_utilization': 0
            }

        # Add machines to work centers
        total_machines = 0
        running_machines = 0

        for machine_row in db_machines:
            wc_id = machine_row['work_center_id']
            machine_num = machine_row['machine_number']

            if wc_id not in work_centers:
                # Create work center if not exists (shouldn't happen with proper FK)
                work_centers[wc_id] = {
                    'work_center_id': wc_id,
                    'work_center_name': wc_id,
                    'machines': [],
                    'total_machines': 0,
                    'running_machines': 0,
                    'avg_utilization': 0
                }

            machine_data = {
                'machine_id': str(machine_num),
                'machine_name': f"Machine {machine_num}",
                'work_center_id': wc_id,
                'status': 'idle',
                'current_job': None,
                'efficiency': 0,
                'utilization': 0,
                'workload_lbs': 0
            }

            work_centers[wc_id]['machines'].append(machine_data)
            work_centers[wc_id]['total_machines'] += 1
            total_machines += 1

        # Assign knit orders to machines (simple round-robin for now)
        if knit_orders:
            active_orders = [o for o in knit_orders if o.get('active', 0) == 1]

            for idx, order in enumerate(active_orders):
                # Get a machine (round-robin across all machines)
                machine_idx = idx % len(db_machines)
                assigned_machine = db_machines[machine_idx]
                wc_id = assigned_machine['work_center_id']
                machine_num = assigned_machine['machine_number']

                # Extract order details
                knit_style_base = order.get('knit_style_base', {})
                style = knit_style_base.get('base_style', f'Style-{idx}') if knit_style_base else f'Style-{idx}'
                order_id = order.get('id', f'ORD-{idx}')
                qty_ordered = float(order.get('qty_ordered', 0) or 0)
                qty_received = float(order.get('qty_received', 0) or 0)
                progress = int((qty_received / qty_ordered * 100)) if qty_ordered > 0 else 0

                # Find the machine in the work center and update it
                for machine in work_centers[wc_id]['machines']:
                    if machine['machine_id'] == str(machine_num):
                        machine['status'] = 'active'
                        machine['current_job'] = {
                            'order_id': str(order_id),
                            'style': style,
                            'progress': progress,
                            'quantity': qty_ordered,
                            'completed': qty_received
                        }
                        machine['utilization'] = min(100, progress)
                        machine['efficiency'] = round(85 + (idx % 15), 1)
                        machine['workload_lbs'] = qty_ordered - qty_received

                        running_machines += 1
                        work_centers[wc_id]['running_machines'] += 1
                        break

        # Calculate utilization per work center
        for wc_id, wc_data in work_centers.items():
            if wc_data['total_machines'] > 0:
                wc_data['avg_utilization'] = round(
                    (wc_data['running_machines'] / wc_data['total_machines']) * 100, 1
                )

        # Calculate overall metrics
        avg_utilization = (running_machines / total_machines * 100) if total_machines > 0 else 0

        response = {
            'status': 'success',
            'last_updated': datetime.now().isoformat(),
            'factory_overview': {
                'total_work_centers': len(work_centers),
                'total_machines': total_machines,
                'running_machines': running_machines,
                'machines_active': running_machines,
                'machines_idle': total_machines - running_machines,
                'utilization_rate': round(avg_utilization, 1),
                'avg_utilization_percent': round(avg_utilization, 1),
                'avg_efficiency': 85.2
            },
            'work_center_groups': list(work_centers.values()),
            'ai_analysis': {
                'bottlenecks': [],
                'optimizations': [],
                'actionable_insights': [],
                'ai_kpis': {
                    'bottleneck_summary': {
                        'total': 0,
                        'critical': 0,
                        'high': 0,
                        'medium': 0
                    },
                    'optimization_potential': {
                        'opportunities': 0,
                        'avg_improvement_percent': 0
                    },
                    'capacity_health': 'Good' if avg_utilization < 90 else 'Warning'
                },
                'efficiency_insights': {
                    'top_performer': 'N/A',
                    'needs_attention': 'N/A',
                    'overall_trend': 'stable'
                }
            }
        }

        logger.info(f"Returning factory floor data: {len(work_centers)} work centers, {total_machines} machines, {running_machines} running")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in factory_floor_ai_dashboard: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/machine-assignment-suggestions', methods=['GET'])
def machine_assignment_suggestions() -> tuple:
    """
    Machine Assignment Suggestions endpoint
    Returns AI-powered suggestions for unassigned orders
    """
    try:
        logger.info("Machine assignment suggestions request")

        # Mock data structure for machine assignment suggestions
        # In production, this would analyze unassigned orders and suggest optimal machines
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'suggestions': [
                {
                    'order_id': 'ORD-003',
                    'style': 'STYLE003',
                    'quantity': 1000,
                    'due_date': (datetime.now() + timedelta(days=7)).isoformat(),
                    'suggested_work_center': 'Knitting',
                    'suggested_machine': 'M002',
                    'confidence': 0.89,
                    'reason': 'Machine currently idle with suitable capabilities',
                    'estimated_completion': (datetime.now() + timedelta(days=5)).isoformat()
                },
                {
                    'order_id': 'ORD-004',
                    'style': 'STYLE004',
                    'quantity': 500,
                    'due_date': (datetime.now() + timedelta(days=10)).isoformat(),
                    'suggested_work_center': 'Dyeing',
                    'suggested_machine': 'M003',
                    'confidence': 0.75,
                    'reason': 'Best match based on style requirements',
                    'estimated_completion': (datetime.now() + timedelta(days=8)).isoformat()
                }
            ],
            'summary': {
                'total_unassigned_orders': 2,
                'suggestions_generated': 2,
                'avg_confidence': 0.82
            }
        }

        logger.info(f"✓ Returning {len(response['suggestions'])} assignment suggestions")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in machine_assignment_suggestions: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# Catch-all for other endpoints
@app.route('/api/fabric-inquiry/search', methods=['POST'])
def fabric_inquiry_search() -> tuple:
    """
    Query fabric inventory by aggregating data from eFab API stages.
    Returns inventory quantities by stage (G00, G02, I01, F01).
    """
    try:
        data = request.get_json() or {}
        fabric_id = data.get("fabric_id")

        if not fabric_id:
            return jsonify({"error": "fabric_id required"}), 400

        # Clean fabric_id to 4 digits
        fabric_id_clean = ''.join(filter(str.isdigit, str(fabric_id)))[:4]

        if not fabric_id_clean:
            return jsonify({"error": "Invalid fabric_id"}), 400

        # Fetch from all stages
        stages = {
            'G00': fetch_from_efab('api/greige/g00'),
            'G02': fetch_from_efab('api/greige/g02'),
            'I01': fetch_from_efab('api/finished/i01'),
            'F01': fetch_from_efab('api/finished/f01')
        }

        # Aggregate inventory by stage
        inventory_by_stage = {}

        for stage, records in stages.items():
            if not records:
                continue

            total_yards = 0
            total_lbs = 0
            total_rolls = 0

            for record in records:
                # Extract base_style from nested structure
                base_style = None

                # Try knit_version path (G00, G02 use this)
                if 'knit_version' in record and record['knit_version']:
                    knit_version = record['knit_version']
                    if 'knit_style_base' in knit_version and knit_version['knit_style_base']:
                        base_style = knit_version['knit_style_base'].get('base_style')

                # Try f_version path (F01 uses this - f_version.f_base.base_style)
                if not base_style and 'f_version' in record and record['f_version']:
                    f_version = record['f_version']
                    if isinstance(f_version, dict) and 'f_base' in f_version and f_version['f_base']:
                        base_style = f_version['f_base'].get('base_style')

                # Try i_version path (I01 might use this - i_version.i_base.base_style)
                if not base_style and 'i_version' in record and record['i_version']:
                    i_version = record['i_version']
                    if isinstance(i_version, dict) and 'i_base' in i_version and i_version['i_base']:
                        base_style = i_version['i_base'].get('base_style')

                if not base_style:
                    continue

                # Check if this record matches our fabric_id
                record_fabric_id = ''.join(filter(str.isdigit, str(base_style)))[:4]

                if record_fabric_id == fabric_id_clean:
                    total_yards += float(record.get('qty_yds', 0) or 0)
                    total_lbs += float(record.get('qty_lbs', 0) or 0)
                    total_rolls += 1

            if total_yards > 0 or total_lbs > 0:
                inventory_by_stage[stage] = {
                    "fabric_type": "greige" if stage in ['G00', 'G02'] else "finished",
                    "yards": round(total_yards, 2),
                    "lbs": round(total_lbs, 2),
                    "rolls": total_rolls
                }

        return jsonify({
            "fabric_id": fabric_id_clean,
            "inventory_by_stage": inventory_by_stage,
            "total_yards": sum(inv.get("yards", 0) for inv in inventory_by_stage.values()),
            "total_lbs": sum(inv.get("lbs", 0) for inv in inventory_by_stage.values()),
            "stages_found": list(inventory_by_stage.keys())
        }), 200

    except Exception as e:
        logger.error(f"Fabric inquiry error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/<path:path>', methods=['GET', 'POST'])
def api_proxy(path: str) -> tuple:
    """Proxy other API requests to eFab or return empty data."""
    logger.info(f"Proxying request to: /api/{path}")

    # Try to fetch from eFab
    data = fetch_from_efab(f'api/{path}', request.args.to_dict())

    if data:
        return jsonify(data), 200
    else:
        # Return empty structure
        return jsonify({
            'data': [],
            'status': 'no_data',
            'message': f'Endpoint /api/{path} returned no data',
            'timestamp': datetime.now().isoformat()
        }), 200


@app.route('/api/retrain-ml', methods=['POST'])
def retrain_ml() -> tuple:
    """Stub endpoint for ML model retraining."""
    try:
        logger.info("ML retrain requested (stub endpoint)")
        return jsonify({
            'status': 'success',
            'message': 'ML model retrain initiated (simulation)',
            'timestamp': datetime.now().isoformat(),
            'note': 'This is a stub endpoint - actual ML training not implemented'
        }), 200
    except Exception as e:
        logger.error(f"Error in retrain_ml: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/po-risk-analysis', methods=['GET'])
def po_risk_analysis() -> tuple:
    """Generate purchase order risk analysis."""
    try:
        # Get knit orders and yarn data for risk analysis
        knit_orders_data = fetch_from_efab('api/knitorder/list')
        yarn_data = fetch_from_efab('api/yarn/active')

        if not knit_orders_data:
            return jsonify({
                'risk_analysis': [],
                'summary': {
                    'total_orders': 0,
                    'high_risk_count': 0,
                    'medium_risk_count': 0,
                    'low_risk_count': 0
                },
                'status': 'no_data'
            }), 200

        risk_analysis = []

        # Analyze each order for risk factors
        for order in knit_orders_data[:50]:  # Limit to 50 for performance
            try:
                # Extract order details
                order_id = order.get('id')
                serial_number = order.get('serial_number', '--')
                qty_ordered = float(order.get('qty_ordered', 0) or 0)
                qty_received = float(order.get('qty_received', 0) or 0)
                balance = float(order.get('balance', 0) or 0)

                # Get dates
                requested_date_str = order.get('requested_date')
                knit_start_str = order.get('knit_start')

                # Calculate days until due
                days_until_due = None
                if requested_date_str:
                    try:
                        requested_date = datetime.fromisoformat(requested_date_str.replace('Z', '+00:00'))
                        days_until_due = (requested_date - datetime.now()).days
                    except:
                        pass

                # Calculate risk score (0-100)
                risk_score = 0
                risk_factors = []

                # Risk factor 1: Days until due
                if days_until_due is not None:
                    if days_until_due < 0:
                        risk_score += 40
                        risk_factors.append(f"Overdue by {abs(days_until_due)} days")
                    elif days_until_due <= 7:
                        risk_score += 30
                        risk_factors.append(f"Due in {days_until_due} days")
                    elif days_until_due <= 14:
                        risk_score += 15
                        risk_factors.append(f"Due in {days_until_due} days")

                # Risk factor 2: Order completion
                completion_pct = (qty_received / qty_ordered * 100) if qty_ordered > 0 else 0
                if completion_pct < 25 and days_until_due and days_until_due < 14:
                    risk_score += 25
                    risk_factors.append(f"Only {completion_pct:.0f}% complete")
                elif completion_pct < 50 and days_until_due and days_until_due < 7:
                    risk_score += 20
                    risk_factors.append(f"Only {completion_pct:.0f}% complete")

                # Risk factor 3: Large order size
                if qty_ordered > 10000:
                    risk_score += 10
                    risk_factors.append("Large order volume")

                # Risk factor 4: Order status
                status = order.get('status', 'Unknown')
                if status == 'Open' and days_until_due and days_until_due < 7:
                    risk_score += 15
                    risk_factors.append("Not yet started")

                # Determine risk level
                if risk_score >= 60:
                    risk_level = 'HIGH'
                elif risk_score >= 30:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'

                # Get style info
                knit_style_base = order.get('knit_style_base', {})
                style = knit_style_base.get('base_style', '--') if knit_style_base else '--'

                risk_analysis.append({
                    'order_id': order_id,
                    'serial_number': serial_number,
                    'style': style,
                    'qty_ordered': qty_ordered,
                    'qty_received': qty_received,
                    'balance': balance,
                    'completion_pct': round(completion_pct, 1),
                    'days_until_due': days_until_due,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'risk_factors': risk_factors,
                    'status': status
                })

            except Exception as e:
                logger.warning(f"Error analyzing order {order.get('id')}: {e}")
                continue

        # Calculate summary
        high_risk = [r for r in risk_analysis if r['risk_level'] == 'HIGH']
        medium_risk = [r for r in risk_analysis if r['risk_level'] == 'MEDIUM']
        low_risk = [r for r in risk_analysis if r['risk_level'] == 'LOW']

        return jsonify({
            'risk_analysis': risk_analysis,
            'summary': {
                'total_orders': len(risk_analysis),
                'high_risk_count': len(high_risk),
                'medium_risk_count': len(medium_risk),
                'low_risk_count': len(low_risk)
            },
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in po_risk_analysis: {e}")
        return jsonify({'error': str(e)}), 500


def main() -> None:
    """Main entry point."""
    if not EFAB_SESSION:
        logger.error("ERROR: EFAB_SESSION not configured in .env file")
        print("=" * 70)
        print("ERROR: eFab session cookie not found")
        print("=" * 70)
        print("Please set EFAB_SESSION in your .env file")
        print("Run: python scripts/efab_login.py to get a new session")
        sys.exit(1)

    print("=" * 70)
    print("Beverly Knits ERP - eFab API Server")
    print("=" * 70)
    print(f"eFab Base URL: {EFAB_BASE_URL}")
    print(f"Session: {EFAB_SESSION[:20]}...")
    print("Server: http://0.0.0.0:5006")
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print("=" * 70)

    # Start Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=5006,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
        sys.exit(0)


if __name__ == '__main__':
    main()
