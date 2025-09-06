#!/usr/bin/env python3
"""
Test script to verify ML forecasting functionality
Tests the /api/ml-forecast-detailed endpoint with various parameters
"""

import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:5006"

def test_ml_forecast_endpoints():
    """Test various ML forecast endpoint configurations"""
    
    print("=" * 60)
    print("Testing ML Forecast Endpoints")
    print("=" * 60)
    
    # Test cases with different parameters
    test_cases = [
        {
            "name": "Default ML Forecast",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {}
        },
        {
            "name": "Summary Detail Level",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {"detail": "summary"}
        },
        {
            "name": "Report Format",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {"format": "report"}
        },
        {
            "name": "Chart Format",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {"format": "chart"}
        },
        {
            "name": "30-Day Horizon",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {"horizon": "30"}
        },
        {
            "name": "With Stock Comparison",
            "endpoint": "/api/ml-forecast-detailed",
            "params": {"compare": "stock"}
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print(f"Endpoint: {test['endpoint']}")
        print(f"Parameters: {test['params']}")
        
        try:
            # Make the API request
            response = requests.get(
                f"{BASE_URL}{test['endpoint']}", 
                params=test['params'],
                timeout=10
            )
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                
                # Check for required fields
                has_models = "models" in data and isinstance(data.get("models"), list)
                has_status = data.get("status") == "success"
                has_summary = "summary" in data
                
                print(f"  ✓ Status Code: {response.status_code}")
                print(f"  {'✓' if has_status else '✗'} Status: {data.get('status', 'missing')}")
                print(f"  {'✓' if has_models else '✗'} Models Array: {'present' if has_models else 'missing'}")
                
                if has_models:
                    print(f"    - Models count: {len(data['models'])}")
                    for model in data['models'][:3]:  # Show first 3 models
                        print(f"      • {model.get('model', 'Unknown')}: {model.get('accuracy', 0)}% ({model.get('status', 'unknown')})")
                
                if has_summary:
                    summary = data['summary']
                    print(f"  ✓ Summary: {summary.get('total_styles', 0)} styles, {summary.get('average_confidence', 0)}% confidence")
                
                # Check format-specific fields
                if test['params'].get('format') == 'report' and 'report' in data:
                    print(f"  ✓ Report format detected")
                elif test['params'].get('format') == 'chart' and 'chart_type' in data:
                    print(f"  ✓ Chart format detected")
                
                results.append({
                    "test": test['name'],
                    "success": True,
                    "has_models": has_models
                })
                
            else:
                print(f"  ✗ Status Code: {response.status_code}")
                print(f"  ✗ Error: {response.text[:200]}")
                results.append({
                    "test": test['name'],
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Connection Error: Is the server running on port 5006?")
            results.append({
                "test": test['name'],
                "success": False,
                "error": "Connection failed"
            })
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                "test": test['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r.get("success"))
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    
    # Check critical requirement: models array
    models_present = sum(1 for r in results if r.get("has_models"))
    print(f"\nModels Array Present: {models_present}/{total} tests")
    
    if successful == total:
        print("\n✓ All tests passed! ML forecasting is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nFailed tests:")
        for r in results:
            if not r.get("success"):
                print(f"  - {r['test']}: {r.get('error', 'Unknown error')}")
    
    return successful == total

if __name__ == "__main__":
    # Test ML validation endpoint too
    print("\nTesting ML Validation Summary Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/ml-validation-summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "models" in data and "risk_assessment" in data:
                print("✓ ML Validation endpoint working")
                print(f"  - Models validated: {len(data.get('models', {}))}")
                print(f"  - Overall confidence: {data.get('risk_assessment', {}).get('overall_confidence', 0)}%")
            else:
                print("✗ ML Validation endpoint missing required fields")
        else:
            print(f"✗ ML Validation endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"✗ ML Validation endpoint error: {e}")
    
    print("\n" + "=" * 60)
    
    # Run the main tests
    success = test_ml_forecast_endpoints()
    
    if success:
        print("\n🎉 ML Forecasting functionality is fully operational!")
    else:
        print("\n⚠️  ML Forecasting has issues that need attention.")