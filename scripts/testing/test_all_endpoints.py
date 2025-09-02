import requests
import json
import time

base_url = "http://localhost:5006"

endpoints = [
    "/api/comprehensive-kpis",
    "/api/inventory-intelligence-enhanced",
    "/api/production-planning",
    "/api/ml-forecast-detailed",
    "/api/inventory-netting",
    "/api/yarn-intelligence",
    "/api/production-suggestions",
    "/api/po-risk-analysis",
    "/api/production-pipeline",
    "/api/yarn-substitution-intelligent",
    "/api/production-recommendations-ml",
    "/api/planning-phases",
    "/api/debug-data",
    "/api/consolidation-metrics"
]

print("\n" + "="*60)
print("TESTING ALL API ENDPOINTS")
print("="*60 + "\n")

failed = []
slow = []
errors = []

for endpoint in endpoints:
    try:
        start = time.time()
        response = requests.get(f"{base_url}{endpoint}", timeout=10)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            if elapsed > 2:
                print(f"⚠️  {endpoint}: SLOW ({elapsed:.2f}s)")
                slow.append(endpoint)
            else:
                print(f"✅ {endpoint}: OK ({elapsed:.2f}s)")
        else:
            print(f"❌ {endpoint}: HTTP {response.status_code}")
            failed.append(endpoint)
            
    except Exception as e:
        print(f"❌ {endpoint}: ERROR - {str(e)}")
        errors.append((endpoint, str(e)))

print("\n" + "="*60)
print("SUMMARY:")
print(f"✅ Successful: {len(endpoints) - len(failed) - len(errors)}/{len(endpoints)}")
if failed:
    print(f"❌ Failed: {failed}")
if errors:
    print(f"❌ Errors: {[e[0] for e in errors]}")
if slow:
    print(f"⚠️  Slow (>2s): {slow}")

if not failed and not errors:
    print("\n🎉 ALL ENDPOINTS WORKING!")
else:
    print("\n⚠️ Some endpoints have issues")
print("="*60)
