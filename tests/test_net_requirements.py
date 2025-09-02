import requests
import json

# Test the APIs that the dashboard uses
base_url = "http://localhost:5006"

# Get production planning data
planning = requests.get(f"{base_url}/api/production-planning").json()
print("Production Planning API:")
print(f"  Schedule items: {len(planning.get('production_schedule', []))}")
if planning.get('production_schedule'):
    first = planning['production_schedule'][0]
    print(f"  First item: {first['style']} - Qty: {first['quantity_lbs']} lbs")

# Get ML forecast data
forecast = requests.get(f"{base_url}/api/ml-forecast-detailed").json()
print("\nML Forecast API:")
if forecast.get('forecast_details'):
    print(f"  Forecast items: {len(forecast['forecast_details'])}")
    first_forecast = forecast['forecast_details'][0]
    print(f"  First forecast: {first_forecast['style']} - 30 day forecast: {first_forecast['forecast_30_days']}")

# Get inventory data
inventory = requests.get(f"{base_url}/api/inventory-intelligence-enhanced").json()
print("\nInventory API:")
print(f"  Total inventory value: ${inventory.get('kpis', {}).get('total_inventory_value', 0):,.2f}")

# Get netting data
netting = requests.get(f"{base_url}/api/inventory-netting").json()
print("\nNetting API:")
if netting.get('style_netting'):
    print(f"  Style netting items: {len(netting['style_netting'])}")

print("\n=== Net Requirements Calculation Example ===")
if planning.get('production_schedule') and len(planning['production_schedule']) > 0:
    order = planning['production_schedule'][0]
    print(f"Style: {order['style']}")
    print(f"Forecasted Demand (from order qty): {order['quantity_lbs']} lbs")
    
    # In the dashboard, it would look up inventory for this style
    # For now, use example values
    current_inventory = 500  # Example
    pipeline_inventory = 200  # Example
    
    net_requirement = max(0, order['quantity_lbs'] - current_inventory - pipeline_inventory)
    print(f"Net Requirement = max(0, {order['quantity_lbs']} - {current_inventory} - {pipeline_inventory}) = {net_requirement}")
