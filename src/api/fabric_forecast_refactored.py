#!/usr/bin/env python3
"""
Refactored Fabric Forecast Function - API-First Architecture
Calls localhost:5006 endpoints instead of reading CSV files or external APIs
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from flask import jsonify

logger = logging.getLogger(__name__)


# ===== BUSINESS CONSTANTS =====
# These constants define business rules and conversion factors.
# Modify these to change behavior without touching business logic.

# Unit conversions
LBS_TO_YARDS_RATIO: float = 5.0
"""Conversion factor from pounds to yards for knit fabric."""

COST_PER_YARD_USD: float = 8.5
"""Estimated cost per yard in USD for forecasting purposes."""

# Priority thresholds
CRITICAL_SHORTAGE_THRESHOLD: float = 0.6
"""
Net requirement must exceed this percentage of forecasted yards to be CRITICAL.
Example: If forecasted 1000 yards and net requirement is 700 yards (70% > 60%),
it's CRITICAL priority.
"""

# Confidence scores
DEFAULT_FORECAST_CONFIDENCE: float = 0.85
"""Default confidence score (0.0-1.0) for forecast items."""

# Lead times (weeks from order to delivery)
LEAD_TIME_CRITICAL_WEEKS: int = 2
"""Lead time in weeks for CRITICAL priority orders."""

LEAD_TIME_HIGH_WEEKS: int = 4
"""Lead time in weeks for HIGH priority orders."""

LEAD_TIME_NORMAL_WEEKS: int = 6
"""Lead time in weeks for NORMAL priority orders."""

# Calendar settings - REMOVED: Now calculated dynamically from current date

# Processing limits
MAX_FORECAST_ITEMS: int = 20
"""Maximum number of forecast items to return (top N by priority)."""

# ===== END CONSTANTS =====


def fetch_local_api_data(endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch data from local API server (localhost:5006).

    Args:
        endpoint: API endpoint path (e.g., '/api/knit-orders')
        timeout: Request timeout in seconds

    Returns:
        API response as dict, or None if request fails
    """
    try:
        url = f"http://localhost:5006{endpoint}"
        logger.info(f"Fetching from local API: {url}")

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Successfully fetched data from {endpoint}")
        return data

    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling {endpoint} after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error - eFab API server (localhost:5006) unreachable for {endpoint}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error calling {endpoint}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling {endpoint}: {str(e)}")
        return None


def fabric_forecast_integrated_refactored() -> Tuple[Dict[str, Any], int]:
    """
    Get integrated fabric forecast using live API calls to localhost:5006.

    NO CSV FALLBACK - Returns error if API is unavailable.

    Returns fabric requirements forecast with:
    - Style information
    - Fabric type and yards required
    - Timeline and delivery dates
    - Status and priority

    Data Sources (all from localhost:5006):
    - /api/knit-orders: Production orders with fabric requirements
    - /api/inventory/pipeline-summary: Inventory across all stages (G00, G02, I01, F01)
    - /api/yarn-intelligence: Yarn availability for netting calculations

    Returns:
        Tuple of (response_dict, status_code)
    """
    try:
        logger.info("Generating fabric forecast from local API endpoints")

        # STEP 1: Fetch knit orders from local API
        knit_orders_response = fetch_local_api_data('/api/knit-orders')
        if not knit_orders_response:
            return {
                "status": "error",
                "message": "eFab API unavailable - Cannot load production orders from /api/knit-orders",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_summary(),
                "timestamp": datetime.now().isoformat()
            }, 500

        knit_orders = knit_orders_response.get('orders', [])
        if not knit_orders:
            return {
                "status": "no_data",
                "message": "No knit orders available for fabric forecast",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_summary(),
                "timestamp": datetime.now().isoformat()
            }, 200

        logger.info(f"Loaded {len(knit_orders)} knit orders from API")

        # STEP 2: Fetch inventory pipeline from local API
        inventory_response = fetch_local_api_data('/api/inventory/pipeline-summary')
        if not inventory_response:
            return {
                "status": "error",
                "message": "eFab API unavailable - Cannot load inventory data from /api/inventory/pipeline-summary",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_summary(),
                "timestamp": datetime.now().isoformat()
            }, 500

        pipeline = inventory_response.get('pipeline', {})
        logger.info(f"Loaded inventory pipeline with {len(pipeline)} stages")

        # STEP 3: Fetch yarn intelligence (optional - for enhanced forecasting)
        yarn_response = fetch_local_api_data('/api/yarn-intelligence')
        yarn_data = yarn_response.get('yarn', []) if yarn_response else []
        logger.info(f"Loaded {len(yarn_data)} yarn intelligence records")

        # STEP 4: Process knit orders to build fabric allocations
        fabric_allocations = _build_fabric_allocations(knit_orders)
        logger.info(f"Built allocations for {len(fabric_allocations)} fabrics from knit orders")

        # STEP 5: Process inventory to build stage-wise availability
        inventory_by_fabric = _process_inventory_pipeline(pipeline)
        logger.info(f"Processed inventory for {len(inventory_by_fabric)} fabric types")

        # STEP 6: Generate forecast items by combining orders + inventory
        forecast_items = _generate_forecast_items(
            knit_orders=knit_orders,
            fabric_allocations=fabric_allocations,
            inventory_by_fabric=inventory_by_fabric
        )

        # STEP 7: Calculate summary metrics
        summary = _calculate_summary(forecast_items)

        response = {
            'status': 'success',
            'forecast_items': forecast_items,
            'fabric_forecast': forecast_items,  # Dashboard expects this field name
            'summary': summary,
            'data_sources': {
                'knit_orders_count': len(knit_orders),
                'inventory_stages': list(pipeline.keys()),
                'yarn_records': len(yarn_data)
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Fabric forecast generated: {len(forecast_items)} items, {summary['critical_items']} critical")
        return response, 200

    except Exception as e:
        logger.error(f"Error in fabric_forecast_integrated_refactored: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f"Internal error generating fabric forecast: {str(e)}",
            'forecast_items': [],
            'fabric_forecast': [],
            'summary': _empty_summary(),
            'timestamp': datetime.now().isoformat()
        }, 500


def _calculate_target_date(lead_time_weeks: int) -> Tuple[str, str]:
    """
    Calculate target delivery date and week number based on lead time.

    Args:
        lead_time_weeks: Number of weeks from now to delivery

    Returns:
        Tuple of (week_string, iso_date_string)
        Example: ("Week 43", "2025-10-26")
    """
    today = datetime.now()
    target_date = today + timedelta(weeks=lead_time_weeks)

    # Calculate ISO week number (1-53)
    iso_year, iso_week, iso_weekday = target_date.isocalendar()

    # Format as "Week XX" and ISO date
    week_string = f"Week {iso_week}"
    date_string = target_date.strftime("%Y-%m-%d")

    return week_string, date_string


def _build_fabric_allocations(knit_orders: List[Dict]) -> Dict[str, float]:
    """
    Build fabric allocations from knit orders.

    Returns dict mapping fabric_id -> total_yards_allocated
    """
    allocations = {}

    for order in knit_orders:
        if not order.get('is_active', True):
            continue

        # Extract style to determine fabric ID
        style = order.get('style', '')
        if not style:
            continue

        # Extract fabric ID (first 4 digits of style)
        fabric_id = ''.join(filter(str.isdigit, str(style)))[:4]
        if not fabric_id:
            continue

        # Get quantity in yards (convert from lbs if needed)
        qty_yards = float(order.get('balance_lbs', 0))

        # Apply conversion factor (simplified - should come from BOM)
        if qty_yards > 0:
            qty_yards = qty_yards * LBS_TO_YARDS_RATIO

        if fabric_id not in allocations:
            allocations[fabric_id] = 0
        allocations[fabric_id] += qty_yards

    return allocations


def _process_inventory_pipeline(pipeline: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    TEMPORARY: Process inventory pipeline with aggregate data.

    WARNING: This is a simplified implementation that returns empty dict
    because the API does not provide fabric-level breakdown. All forecasts
    will show 0 current inventory until API is enhanced to return fabric-level data.

    Returns:
        Dict mapping fabric_id -> {stage_name: yards}
        Currently returns empty dict for all fabrics.
    """
    # Get total yards by stage for logging purposes
    stage_totals = {}
    for stage_key, stage_data in pipeline.items():
        stage_name = stage_key.upper()
        total_yards = float(stage_data.get('total_on_hand', 0))
        stage_totals[stage_name] = total_yards

    # Log warning about missing fabric-level data
    logger.warning(
        f"Inventory pipeline provides only aggregate data (stages: {list(stage_totals.keys())}). "
        "Cannot allocate inventory to specific fabrics. "
        "All forecasts will show 0 current inventory. "
        "TODO: Update API to return fabric-level breakdown."
    )

    # Return empty dict = all fabrics get 0 inventory
    # This is better than incorrect allocation
    return {}


def _generate_forecast_items(
    knit_orders: List[Dict],
    fabric_allocations: Dict[str, float],
    inventory_by_fabric: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Generate forecast items by combining order data with inventory.

    Returns list of forecast item dicts.
    """
    forecast_items = []

    # Group orders by fabric type
    fabric_types_map = ['Jersey', 'Interlock', 'Rib', 'French Terry', 'Pique']

    # Sort orders by balance_lbs descending to prioritize largest orders
    sorted_orders = sorted(
        knit_orders,
        key=lambda x: float(x.get('balance_lbs', 0)),
        reverse=True
    )[:MAX_FORECAST_ITEMS]

    for idx, order in enumerate(sorted_orders):
        style = order.get('style', 'Unknown')
        fabric_id = ''.join(filter(str.isdigit, str(style)))[:4] if style else None

        # Determine fabric type (simplified - should come from BOM)
        fabric_type = fabric_types_map[idx % len(fabric_types_map)]

        # Get order quantity
        qty_ordered_lbs = float(order.get('qty_ordered_lbs', 0))
        balance_lbs = float(order.get('balance_lbs', 0))

        # Convert to yards using constant
        forecasted_yards = int(balance_lbs * LBS_TO_YARDS_RATIO)

        # CRITICAL FIX #1: Get fabric-specific inventory (not aggregate)
        fabric_inv = inventory_by_fabric.get(fabric_id, {})

        # Current Inventory = I01 + F01 (finished stages) for THIS SPECIFIC fabric
        current_inventory = int(fabric_inv.get('I01', 0) + fabric_inv.get('F01', 0))

        # On Order = G00 + G02 (WIP in pipeline) for THIS SPECIFIC fabric
        on_order = int(fabric_inv.get('G00', 0) + fabric_inv.get('G02', 0))

        # Get allocated fabric from knit orders
        allocated_yards = int(fabric_allocations.get(fabric_id, 0)) if fabric_id else 0

        # Calculate net position: Current + WIP - Allocated - Forecast
        net_position = current_inventory + on_order - allocated_yards - forecasted_yards

        # Net requirement is shortage amount (positive = need to order)
        net_requirement = -net_position

        # Determine priority and status using constant
        if net_requirement > forecasted_yards * CRITICAL_SHORTAGE_THRESHOLD:
            priority = 'CRITICAL'
            status = 'URGENT_ORDER'
            lead_time_weeks = LEAD_TIME_CRITICAL_WEEKS
        elif net_requirement > 0:
            priority = 'HIGH'
            status = 'ORDER_SOON'
            lead_time_weeks = LEAD_TIME_HIGH_WEEKS
        else:
            priority = 'NORMAL'
            status = 'ADEQUATE'
            lead_time_weeks = LEAD_TIME_NORMAL_WEEKS

        # Calculate target delivery date based on lead time
        delivery_week, target_date = _calculate_target_date(lead_time_weeks)

        forecast_items.append({
            'style': style,
            'fabric_type': fabric_type,
            'description': f'{fabric_type} for {style}',
            'forecasted_yards': forecasted_yards,
            'forecasted_qty': forecasted_yards,  # Alias for frontend compatibility
            'knit_orders_yards': allocated_yards,  # Yards allocated to knit orders
            'current_inventory': current_inventory,
            'on_order': on_order,
            'allocated': allocated_yards,
            'net_position': net_position,
            'net_requirement': max(0, net_requirement),  # Only show positive shortage (0 if surplus)
            'priority': priority,
            'status': status,
            'lead_time_weeks': lead_time_weeks,
            'lead_time': f'{lead_time_weeks * 7} days',  # Convert weeks to days for display
            'estimated_cost': round(net_requirement * COST_PER_YARD_USD, 2) if net_requirement > 0 else 0,
            'delivery_week': delivery_week,
            'target_date': target_date,  # Add ISO date for frontend display
            'confidence': DEFAULT_FORECAST_CONFIDENCE,
            'order_id': order.get('order_id', 'N/A'),
            'customer': order.get('customer', 'N/A')
        })

    return forecast_items


def _calculate_summary(forecast_items: List[Dict]) -> Dict[str, Any]:
    """Calculate summary metrics from forecast items."""
    return {
        'total_yards_forecasted': sum(f['forecasted_yards'] for f in forecast_items),
        'total_net_requirement': sum(f['net_requirement'] for f in forecast_items),
        'total_required_yards': sum(f['net_requirement'] for f in forecast_items),
        'critical_items': sum(1 for f in forecast_items if f['priority'] == 'CRITICAL'),
        'shortage_count': sum(1 for f in forecast_items if f['priority'] == 'CRITICAL'),
        'high_priority_items': sum(1 for f in forecast_items if f['priority'] == 'HIGH'),
        'total_estimated_cost': sum(f['estimated_cost'] for f in forecast_items),
        'timeline_alert': any(f['priority'] == 'CRITICAL' for f in forecast_items),
        'total_styles': len(set(f['style'] for f in forecast_items)),
        'fabric_types_count': len(set(f['fabric_type'] for f in forecast_items))
    }


def _empty_summary() -> Dict[str, Any]:
    """Return empty summary structure."""
    return {
        'total_yards_forecasted': 0,
        'total_net_requirement': 0,
        'total_required_yards': 0,
        'critical_items': 0,
        'shortage_count': 0,
        'high_priority_items': 0,
        'total_estimated_cost': 0,
        'timeline_alert': False,
        'total_styles': 0,
        'fabric_types_count': 0
    }
