import requests
import json
import sys

def test_endpoint(name, url, expected_fields=None):
    """Test an endpoint and validate expected fields"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return f"❌ {name}: HTTP {response.status_code}"
        
        data = response.json()
        
        # Check expected fields if provided
        issues = []
        if expected_fields:
            for field, check in expected_fields.items():
                if field not in data:
                    issues.append(f"Missing field: {field}")
                elif check == "not_zero" and (data[field] == 0 or data[field] == "0" or data[field] == "$0"):
                    issues.append(f"{field} is zero/empty: {data[field]}")
                elif check == "not_empty" and (not data[field] or data[field] == []):
                    issues.append(f"{field} is empty")
        
        if issues:
            return f"⚠️  {name}: {', '.join(issues)}"
        
        # Return success with key metrics
        return f"✅ {name}: OK - {json.dumps(data)[:100]}..."
        
    except Exception as e:
        return f"❌ {name}: {str(e)}"

# Run comprehensive tests
base_url = "http://localhost:5006"

tests = [
    # Test 1: Sales Revenue (was $0)
    ("Sales Revenue in KPIs", f"{base_url}/api/comprehensive-kpis", {
        "sales_revenue": "not_zero",
        "active_knit_orders": "not_zero"
    }),
    
    # Test 2: Forecast Accuracy (was 0%)
    ("Forecast Accuracy", f"{base_url}/api/comprehensive-kpis", {
        "forecast_accuracy": "not_zero"
    }),
    
    # Test 3: Order Fill Rate (was 0%)
    ("Order Fill Rate", f"{base_url}/api/comprehensive-kpis", {
        "order_fill_rate": "not_zero"
    }),
    
    # Test 4: Process Efficiency (was 0%)
    ("Process Efficiency", f"{base_url}/api/comprehensive-kpis", {
        "process_efficiency": "not_zero"
    }),
    
    # Test 5: Inventory Netting with allocation
    ("Inventory Netting", f"{base_url}/api/inventory-netting", {
        "yarn_netting": "not_empty",
        "status": "not_empty"
    }),
    
    # Test 6: ML Confidence (was hardcoded)
    ("ML Forecast Confidence", f"{base_url}/api/ml-forecast-detailed?detail=summary", {
        "summary": "not_empty"
    }),
    
    # Test 7: Production Planning
    ("Production Planning", f"{base_url}/api/production-planning", None),
    
    # Test 8: Yarn Intelligence
    ("Yarn Intelligence", f"{base_url}/api/yarn-intelligence", None),
    
    # Test 9: Consolidated Endpoints
    ("Inventory Intelligence", f"{base_url}/api/inventory-intelligence-enhanced", None),
    
    # Test 10: Supply Chain Analysis
    ("Planning Phases", f"{base_url}/api/planning-phases", None)
]

print("\n" + "="*60)
print("COMPREHENSIVE FIX VERIFICATION REPORT")
print("="*60 + "\n")

all_passed = True
for test_name, url, expected in tests:
    result = test_endpoint(test_name, url, expected)
    print(result)
    if "❌" in result or "⚠️" in result:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✅ ALL TESTS PASSED - All fixes are functional!")
else:
    print("⚠️ SOME ISSUES DETECTED - Review warnings above")
print("="*60)
