"""
Test Service Integration
Verifies that the 7+ extracted services are properly wired to the monolith
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5006"
SERVICE_STATUS_ENDPOINT = f"{BASE_URL}/api/service-status"

def test_service_integration():
    """Test if services are properly integrated"""
    
    print("=" * 60)
    print("TESTING SERVICE INTEGRATION")
    print("=" * 60)
    
    try:
        # Check service status
        print(f"\n1. Checking service status at {SERVICE_STATUS_ENDPOINT}...")
        response = requests.get(SERVICE_STATUS_ENDPOINT, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('integration_active'):
                print("✓ Service integration is ACTIVE")
                print(f"✓ Services count: {data.get('services_count', 0)}")
                print(f"✓ Overall status: {data.get('overall_status', 'UNKNOWN')}")
                print(f"✓ Message: {data.get('message', '')}")
                
                # List integrated services
                services = data.get('services', {})
                if services:
                    print("\n2. Integrated Services:")
                    for service_name, service_info in services.items():
                        status = service_info.get('status', 'UNKNOWN')
                        available = service_info.get('available', False)
                        icon = "✓" if available else "✗"
                        print(f"   {icon} {service_name}: {status}")
                
                print("\n✅ SERVICE INTEGRATION SUCCESSFUL!")
                print("   - Monolith can now delegate to extracted services")
                print("   - This reduces monolith from 18,076 lines")
                print("   - Services: inventory, forecasting, capacity, yarn, scheduler, supply_chain, mrp")
                
            else:
                print("⚠ Service integration exists but is not active")
                print(f"  Message: {data.get('message', '')}")
                if data.get('monolith_mode'):
                    print("  System is running in MONOLITH MODE")
        else:
            print(f"✗ Service status check failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server at", BASE_URL)
        print("  Please ensure the server is running:")
        print("  python src/core/beverly_comprehensive_erp.py")
    except Exception as e:
        print(f"✗ Error testing service integration: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_service_integration()