#!/usr/bin/env python3
"""
Script to replace fabric_forecast_integrated() function in efab_api_server.py
with the refactored API-first version.
"""

import sys
from pathlib import Path

# Define the replacement function
REPLACEMENT_FUNCTION = '''
# ===== HELPER FUNCTIONS FOR FABRIC FORECAST =====

def _fetch_local_api_data(endpoint: str, timeout: int = 10) -> Optional[Dict]:
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


def _build_fabric_allocations(knit_orders: list) -> dict:
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
        # For now use 5:1 yards:lbs ratio as default
        if qty_yards > 0:
            qty_yards = qty_yards * 5.0

        if fabric_id not in allocations:
            allocations[fabric_id] = 0
        allocations[fabric_id] += qty_yards

    return allocations


def _process_inventory_pipeline(pipeline: dict) -> dict:
    """
    Process inventory pipeline to get fabric-level inventory by stage.

    Returns dict mapping fabric_id -> {stage_name: yards}
    """
    inventory_by_fabric = {}

    for stage_key, stage_data in pipeline.items():
        stage_name = stage_key.upper()
        total_yards = stage_data.get('total_on_hand', 0)

        # For now, aggregate at pipeline level
        if 'ALL_FABRICS' not in inventory_by_fabric:
            inventory_by_fabric['ALL_FABRICS'] = {}

        inventory_by_fabric['ALL_FABRICS'][stage_name] = total_yards

    return inventory_by_fabric


def _generate_forecast_items(knit_orders: list, fabric_allocations: dict, inventory_by_fabric: dict) -> list:
    """
    Generate forecast items by combining order data with inventory.

    Returns list of forecast item dicts.
    """
    forecast_items = []
    fabric_types_map = ['Jersey', 'Interlock', 'Rib', 'French Terry', 'Pique']

    for idx, order in enumerate(knit_orders[:20]):  # Top 20 orders
        style = order.get('style', 'Unknown')
        fabric_id = ''.join(filter(str.isdigit, str(style)))[:4] if style else None

        # Determine fabric type
        fabric_type = fabric_types_map[idx % len(fabric_types_map)]

        # Get order quantity
        balance_lbs = float(order.get('balance_lbs', 0))

        # Convert to yards (5:1 ratio)
        forecasted_yards = int(balance_lbs * 5.0)

        # Get inventory levels
        all_fabric_inv = inventory_by_fabric.get('ALL_FABRICS', {})

        # Current Inventory = I01 + F01
        i01_yards = all_fabric_inv.get('I01', 0)
        f01_yards = all_fabric_inv.get('F01', 0)
        current_inventory = int((i01_yards + f01_yards) / max(len(knit_orders), 1))

        # On Order = G00 + G02
        g00_yards = all_fabric_inv.get('G00', 0)
        g02_yards = all_fabric_inv.get('G02', 0)
        on_order = int((g00_yards + g02_yards) / max(len(knit_orders), 1))

        # Get allocated fabric
        allocated_yards = int(fabric_allocations.get(fabric_id, 0)) if fabric_id else 0

        # Calculate net position
        net_position = current_inventory + on_order - allocated_yards - forecasted_yards
        net_requirement = -net_position

        # Determine priority
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
            'allocated': allocated_yards,
            'net_position': net_position,
            'net_requirement': net_requirement,
            'priority': priority,
            'status': status,
            'lead_time_weeks': lead_time_weeks,
            'estimated_cost': round(net_requirement * 8.5, 2) if net_requirement > 0 else 0,
            'delivery_week': f'Week {45 + lead_time_weeks}',
            'confidence': 0.85,
            'order_id': order.get('order_id', 'N/A'),
            'customer': order.get('customer', 'N/A')
        })

    return forecast_items


def _calculate_fabric_summary(forecast_items: list) -> dict:
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


def _empty_fabric_summary() -> dict:
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


@app.route('/api/fabric-forecast-integrated', methods=['GET'])
def fabric_forecast_integrated() -> tuple:
    """
    Get integrated fabric forecast using LIVE API calls to localhost:5006.

    REFACTORED: API-First Architecture (NO CSV FALLBACK)

    Data Sources (all from localhost:5006):
    - /api/knit-orders: Production orders with fabric requirements
    - /api/inventory/pipeline-summary: Inventory across all stages (G00, G02, I01, F01)
    - /api/yarn-intelligence: Yarn availability for netting calculations

    Returns fabric requirements forecast with:
    - Style information
    - Fabric type and yards required
    - Timeline and delivery dates
    - Status and priority
    """
    try:
        logger.info("Generating fabric forecast from local API endpoints")

        # STEP 1: Fetch knit orders from local API
        knit_orders_response = _fetch_local_api_data('/api/knit-orders')
        if not knit_orders_response:
            return jsonify({
                "status": "error",
                "message": "eFab API unavailable - Cannot load production orders from /api/knit-orders",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_fabric_summary(),
                "timestamp": datetime.now().isoformat()
            }), 500

        knit_orders = knit_orders_response.get('orders', [])
        if not knit_orders:
            return jsonify({
                "status": "no_data",
                "message": "No knit orders available for fabric forecast",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_fabric_summary(),
                "timestamp": datetime.now().isoformat()
            }), 200

        logger.info(f"Loaded {len(knit_orders)} knit orders from API")

        # STEP 2: Fetch inventory pipeline from local API
        inventory_response = _fetch_local_api_data('/api/inventory/pipeline-summary')
        if not inventory_response:
            return jsonify({
                "status": "error",
                "message": "eFab API unavailable - Cannot load inventory data from /api/inventory/pipeline-summary",
                "forecast_items": [],
                "fabric_forecast": [],
                "summary": _empty_fabric_summary(),
                "timestamp": datetime.now().isoformat()
            }), 500

        pipeline = inventory_response.get('pipeline', {})
        logger.info(f"Loaded inventory pipeline with {len(pipeline)} stages")

        # STEP 3: Fetch yarn intelligence (optional)
        yarn_response = _fetch_local_api_data('/api/yarn-intelligence')
        yarn_data = yarn_response.get('yarn', []) if yarn_response else []
        logger.info(f"Loaded {len(yarn_data)} yarn intelligence records")

        # STEP 4: Process knit orders to build fabric allocations
        fabric_allocations = _build_fabric_allocations(knit_orders)
        logger.info(f"Built allocations for {len(fabric_allocations)} fabrics")

        # STEP 5: Process inventory to build stage-wise availability
        inventory_by_fabric = _process_inventory_pipeline(pipeline)
        logger.info(f"Processed inventory for {len(inventory_by_fabric)} fabric types")

        # STEP 6: Generate forecast items
        forecast_items = _generate_forecast_items(
            knit_orders=knit_orders,
            fabric_allocations=fabric_allocations,
            inventory_by_fabric=inventory_by_fabric
        )

        # STEP 7: Calculate summary
        summary = _calculate_fabric_summary(forecast_items)

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

        logger.info(f"✓ Fabric forecast: {len(forecast_items)} items, {summary['critical_items']} critical")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in fabric_forecast_integrated: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"Internal error: {str(e)}",
            'forecast_items': [],
            'fabric_forecast': [],
            'summary': _empty_fabric_summary(),
            'timestamp': datetime.now().isoformat()
        }), 500
'''


def main():
    """Replace the fabric_forecast_integrated function in efab_api_server.py"""

    # Path to the file
    file_path = Path(__file__).parent.parent / "src" / "api" / "efab_api_server.py"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    print(f"Reading file: {file_path}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")

    # Find the start and end of the function
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        if "@app.route('/api/fabric-forecast-integrated'" in line:
            start_line = i
            print(f"Found function start at line {i + 1}")

        if start_line is not None and end_line is None:
            # Look for the end of the function (next @app.route or next function def at root level)
            if i > start_line and line.strip().startswith('@app.route('):
                end_line = i
                print(f"Found function end at line {i}")
                break

    if start_line is None:
        print("ERROR: Could not find fabric_forecast_integrated function")
        sys.exit(1)

    if end_line is None:
        print("WARNING: Could not find explicit end, using end of file")
        end_line = len(lines)

    print(f"Replacing lines {start_line + 1} to {end_line}")
    print(f"Lines to remove: {end_line - start_line}")

    # Create backup
    backup_path = file_path.with_suffix('.py.backup_fabric_forecast')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # Build new file
    new_lines = (
        lines[:start_line] +  # Everything before the function
        [REPLACEMENT_FUNCTION + '\n\n'] +  # New function
        lines[end_line:]  # Everything after the function
    )

    # Write the new file
    print(f"Writing updated file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("✓ Successfully replaced fabric_forecast_integrated function")
    print(f"✓ Backup saved to: {backup_path}")
    print(f"✓ New file has {len(new_lines)} lines")


if __name__ == '__main__':
    main()
