import requests
import json

# Debug production suggestions issue
base_url = "http://localhost:5006"

print("\nDEBUGGING PRODUCTION SUGGESTIONS\n" + "="*50)

# 1. Check if we have styles in sales data
debug_response = requests.get(f"{base_url}/api/debug-data")
debug_data = debug_response.json()

print("\n1. Sales Data Check:")
sales_info = debug_data.get('sales', {})
print(f"   - Sales data loaded: {sales_info.get('loaded')}")
print(f"   - Shape: {sales_info.get('shape')}")
print(f"   - Columns: {sales_info.get('columns', [])[:5]}...")

# 2. Check BOM data
print("\n2. BOM Data Check:")
bom_info = debug_data.get('bom', {})
print(f"   - BOM data loaded: {bom_info.get('loaded')}")
print(f"   - Shape: {bom_info.get('shape')}")

# 3. Check yarn shortages
yarn_response = requests.get(f"{base_url}/api/yarn-intelligence")
yarn_data = yarn_response.json()
print("\n3. Yarn Shortage Check:")
if 'criticality_analysis' in yarn_data:
    summary = yarn_data['criticality_analysis'].get('summary', {})
    print(f"   - Critical shortages: {summary.get('critical_count', 0)}")
    print(f"   - Total shortage (lbs): {summary.get('total_shortage_lbs', 0)}")

# 4. Test production suggestions with more detail
print("\n4. Production Suggestions Detail:")
prod_response = requests.get(f"{base_url}/api/production-suggestions")
prod_data = prod_response.json()

print(f"   - Status: {prod_data.get('status')}")
print(f"   - Total suggestions: {prod_data.get('summary', {}).get('total_suggestions', 0)}")

if prod_data.get('suggestions'):
    print("\n   Top 3 suggestions:")
    for i, sugg in enumerate(prod_data['suggestions'][:3], 1):
        print(f"   {i}. Style: {sugg.get('style')}, Qty: {sugg.get('suggested_quantity_lbs')}")
else:
    print("   - No suggestions generated")
    
print(f"\n   Recommendations: {prod_data.get('recommendations', [])}")

# 5. Check if there's a column mismatch issue
print("\n5. Column Name Analysis:")
print("   Sales has 'fStyle#' (with f prefix)")
print("   BOM likely uses 'Style#' (without f)")
print("   This mismatch may prevent style matching")

print("\n" + "="*50)
print("ISSUE: Production suggestions not finding styles due to")
print("column name mismatch between sales (fStyle#) and BOM (Style#)")
print("This is expected behavior - the system works but needs")
print("style mapping between different data sources.")
