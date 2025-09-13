# Enterprise API Architecture Documentation
## Beverly Knits ERP v2 - Complete Technical Reference

### Version: 2.0.0 | Date: September 13, 2025
### Classification: Technical Documentation - Enterprise Grade

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Backend API Flow (Detailed)](#backend-api-flow-detailed)
4. [Frontend API Flow (Detailed)](#frontend-api-flow-detailed)
5. [API Consolidation Strategy](#api-consolidation-strategy)
6. [Current Issues & Solutions](#current-issues--solutions)
7. [Enterprise Fix Implementation](#enterprise-fix-implementation)
8. [Testing & Validation](#testing--validation)
9. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Executive Summary

The Beverly Knits ERP v2 system implements a complex API architecture with 100+ endpoints serving a comprehensive textile manufacturing ERP system. As of September 2025, the system processes:

- **1,200+ yarn items** via real-time API integration
- **28,653+ BOM entries** for style-to-yarn mappings
- **195 production orders** with real-time tracking
- **557,671+ lbs** total production workload

### Critical Issue Identified
A circular redirect in the yarn API (`/api/yarn/unified` → `/api/yarn/unified`) is preventing data display on the dashboard despite successful data retrieval.

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Browser (consolidated_dashboard.html)                 │   │
│  │     - API Compatibility Layer (Lines 3674-3710)          │   │
│  │     - Fetch Interceptor (Lines 3690-3800)                │   │
│  │     - Redirect Mapping (deprecatedEndpoints)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP/HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                      MIDDLEWARE LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Flask Before Request Handler                          │   │
│  │     (beverly_comprehensive_erp.py:772-850)               │   │
│  │     - intercept_deprecated_endpoints()                   │   │
│  │     - Redirect map with 45+ mappings                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Internal Routing
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND API LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Flask Application (beverly_comprehensive_erp.py)      │   │
│  │     - 100+ API endpoints                                  │   │
│  │     - 19 yarn-specific endpoints                          │   │
│  │     - 4 unified endpoints (production, forecast, etc.)    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Data Access
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     ERP Wrapper Service (Port 8000)                       │   │
│  │     - FastAPI microservice                                │   │
│  │     - eFab API integration                                │   │
│  │     - Session management                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Backend API Flow (Detailed)

### 1. Request Reception
**File**: `src/core/beverly_comprehensive_erp.py`

#### Entry Point (Line 772)
```python
@app.before_request
def intercept_deprecated_endpoints():
    """
    Intercepts ALL incoming requests before routing
    Checks against redirect_map for deprecated endpoints
    """
```

#### Request Processing Flow
```
1. Client Request → Flask Server (Port 5006)
   ↓
2. before_request interceptor (Line 772)
   ↓
3. Check FEATURE_FLAGS (Line 778)
   - api_consolidation_enabled
   - redirect_deprecated_apis
   ↓
4. If deprecated endpoint found in redirect_map (Lines 782-830):
   - Extract new endpoint and parameters
   - Build new URL with query parameters
   - Return redirect(new_url, code=307)
   ↓
5. If not deprecated:
   - Continue to actual route handler
```

### 2. API Endpoint Structure

#### Yarn Domain Endpoints (19 total)
```python
# Line 10617 - Basic yarn data
@app.route("/api/yarn-data")
def get_yarn_data():
    # Returns raw yarn inventory data

# Line 10751 - Shortage analysis
@app.route("/api/yarn-shortage-analysis")
def yarn_shortage_analysis():
    # Analyzes yarn shortages with criticality

# Line 12764 - Main intelligence endpoint
@app.route("/api/yarn-intelligence")
def get_yarn_intelligence():
    """
    CONSOLIDATED ENDPOINT - Should be the unified target
    Parameters:
    - view: full, data, summary
    - analysis: standard, shortage, requirements
    - forecast: true/false
    - yarn_id: specific yarn
    - ai: true/false (AI enhancement)
    """

# Line 13744 - Enhanced intelligence
@app.route("/api/yarn-intelligence-enhanced")
def get_yarn_intelligence_enhanced():
    # Extended version with more analytics

# Line 16391 - Substitution intelligence
@app.route("/api/yarn-substitution-intelligent")
def yarn_substitution_intelligent():
    # ML-powered yarn substitution recommendations
```

#### Unified Endpoints (Existing)
```python
# THESE EXIST AND WORK:
@app.route("/api/production/unified")  # Line not shown - needs verification
@app.route("/api/forecast/unified")    # Line not shown - needs verification
@app.route("/api/inventory/unified")   # Line not shown - needs verification

# THIS DOES NOT EXIST - THE PROBLEM:
@app.route("/api/yarn/unified")  # MISSING!
```

### 3. Data Processing Pipeline

#### Cache Layer (Lines 12782-12791)
```python
if CACHE_MANAGER_AVAILABLE:
    cache_key = "yarn_intelligence"
    cached_result = cache_manager.get(cache_key, namespace="api")
    if cached_result and cached_result.get('criticality_analysis'):
        cached_result['_cache_hit'] = True
        return jsonify(clean_response_for_json(cached_result))
```

#### Data Aggregation (Lines 12810-12900)
```python
# Process yarn shortage data
shortage_data = []
df = analyzer.raw_materials_data.copy()

# Apply filters and calculations
df['Planning_Balance'] = pd.to_numeric(df['Planning_Balance'], errors='coerce')
df['Allocated'] = pd.to_numeric(df['Allocated'], errors='coerce')
df['shortage'] = -df['Planning_Balance'].clip(upper=0)

# Aggregate by criticality
critical_yarns = df[df['shortage'] > 0]
```

### 4. Response Formatting

#### JSON Sanitization (clean_response_for_json function)
```python
def clean_response_for_json(obj):
    """
    Ensures all response data is JSON-serializable
    - Converts NaN to None
    - Handles datetime objects
    - Removes numpy types
    """
```

---

## Frontend API Flow (Detailed)

### 1. API Compatibility Layer
**File**: `web/consolidated_dashboard.html`

#### Initialization (Lines 3674-3710)
```javascript
// Deprecated endpoint mapping
const deprecatedEndpoints = {
    // Problem entry:
    '/api/yarn/unified': '/api/yarn/unified',  // CIRCULAR!

    // Working entries:
    '/api/production-planning': '/api/production/unified',
    '/api/ml-forecast-detailed': '/api/forecast/unified',

    // Should be:
    '/api/yarn/unified': '/api/yarn-intelligence',
};
```

#### Fetch Interceptor (Lines 3690-3800)
```javascript
// Store original fetch
const originalFetch = window.fetch;

// Override fetch
window.fetch = function(url, options = {}) {
    // Parse URL
    const urlStr = typeof url === 'string' ? url : url.toString();

    // Check if deprecated
    for (const [deprecated, replacement] of Object.entries(deprecatedEndpoints)) {
        if (urlStr.includes(deprecated)) {
            console.warn(`[API Compatibility] Redirecting: ${deprecated} → ${replacement}`);

            // Build new URL
            const newUrl = urlStr.replace(deprecated, replacement);

            // Call original fetch with new URL
            return originalFetch(newUrl, options);
        }
    }

    // Not deprecated, proceed normally
    return originalFetch(url, options);
};
```

### 2. API Call Patterns

#### Standard API Call (Lines 4040-4045)
```javascript
async function loadYarnIntelligence() {
    try {
        // This calls /api/yarn/unified which doesn't exist
        const yarnData = await fetchAPI('/api/yarn/unified');
        console.log('Yarn intelligence loaded:', yarnData);

        // Process data
        if (yarnData && yarnData.criticality_analysis) {
            updateYarnDisplay(yarnData);
        }
    } catch (error) {
        console.error('Failed to load yarn data:', error);
    }
}
```

#### FetchAPI Wrapper (Lines 3520-3540)
```javascript
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error for ${endpoint}:`, error);
        throw error;
    }
}
```

### 3. Data Display Pipeline

#### Yarn Data Processing (Lines 13507-13560)
```javascript
fetchAPI('/api/yarn/unified')
    .then(data => {
        if (data.criticality_analysis && data.criticality_analysis.yarns) {
            const yarnsWithShortage = data.criticality_analysis.yarns
                .filter(y => y.shortage > 0)
                .sort((a, b) => b.shortage - a.shortage);

            // Update UI elements
            document.getElementById('yarnShortageCount').textContent = yarnsWithShortage.length;
            document.getElementById('totalShortageAmount').textContent =
                yarnsWithShortage.reduce((sum, y) => sum + y.shortage, 0).toFixed(0);

            // Update table
            updateYarnTable(yarnsWithShortage);
        }
    })
    .catch(error => {
        console.error('Failed to load yarn intelligence:', error);
        // UI shows loading spinner indefinitely - THE VISIBLE PROBLEM
    });
```

---

## API Consolidation Strategy

### Current State (September 2025)

#### Consolidated Domains
| Domain | Unified Endpoint | Individual Endpoints | Status |
|--------|-----------------|---------------------|---------|
| Production | `/api/production/unified` | 12 | ✅ Working |
| Forecast | `/api/forecast/unified` | 8 | ✅ Working |
| Inventory | `/api/inventory/unified` | 10 | ✅ Working |
| System | `/api/system/unified` | 5 | ✅ Working |

#### Unconsolidated Domains
| Domain | Unified Endpoint | Individual Endpoints | Status |
|--------|-----------------|---------------------|---------|
| Yarn | `/api/yarn/unified` | 19 | ❌ Missing |
| Analytics | None | 7 | ❌ Not started |
| Planning | None | 6 | ❌ Not started |

### Consolidation Principles

1. **Parameter-Based Routing**
   ```python
   @app.route("/api/domain/unified")
   def get_domain_unified():
       view = request.args.get('view', 'full')
       analysis = request.args.get('analysis', 'standard')

       # Route based on parameters
       if view == 'summary':
           return get_domain_summary()
       elif analysis == 'advanced':
           return get_domain_advanced_analysis()
       else:
           return get_domain_full()
   ```

2. **Backward Compatibility**
   - Deprecated endpoints redirect to unified with appropriate parameters
   - Parameters preserve original functionality
   - Response structure remains consistent

3. **Performance Optimization**
   - Single cache key per domain
   - Aggregated data retrieval
   - Reduced round trips

---

## Current Issues & Solutions

### Issue #1: Circular Redirect for Yarn API

#### The Problem
```javascript
// In frontend (Line 3676):
'/api/yarn/unified': '/api/yarn/unified'  // Redirects to itself!
```

#### Root Cause
- Backend endpoint `/api/yarn/unified` was never created
- Frontend assumes it exists and maps it to itself
- Results in infinite redirect loop

#### The Solution

**Option A: Quick Fix (5 minutes)**
```javascript
// Change Line 3676 in consolidated_dashboard.html:
'/api/yarn/unified': '/api/yarn-intelligence'
```

**Option B: Proper Fix (30 minutes)**
```python
# Add to beverly_comprehensive_erp.py after Line 12764:

@app.route("/api/yarn/unified")
def get_yarn_unified():
    """
    Unified yarn endpoint - Consolidates all yarn functionality
    Combines data from multiple yarn endpoints
    """
    # Get parameters
    view = request.args.get('view', 'full')
    analysis = request.args.get('analysis', 'standard')
    substitution = request.args.get('substitution', 'false').lower() == 'true'

    # Route to appropriate handler
    if substitution:
        # Use substitution endpoint
        from flask import current_app
        with current_app.test_request_context(
            '/api/yarn-substitution-intelligent',
            query_string=request.query_string
        ):
            return yarn_substitution_intelligent()
    else:
        # Use main intelligence endpoint
        with current_app.test_request_context(
            '/api/yarn-intelligence',
            query_string=request.query_string
        ):
            return get_yarn_intelligence()
```

### Issue #2: Inconsistent API Naming

#### The Problem
- 19 yarn endpoints with different naming conventions
- No clear hierarchy or organization
- Difficult to maintain and extend

#### The Solution
Implement domain-based naming convention:
```
/api/{domain}/unified           - Unified endpoint
/api/{domain}/{function}         - Specific function
/api/{domain}/{function}/{id}    - Resource-specific
```

### Issue #3: Missing Error Handling

#### The Problem
When endpoints fail, the UI shows infinite loading

#### The Solution
```javascript
// Add timeout and retry logic:
async function fetchAPIWithRetry(endpoint, options = {}, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10000);

            const response = await fetch(endpoint, {
                ...options,
                signal: controller.signal
            });

            clearTimeout(timeout);

            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
}
```

---

## Enterprise Fix Implementation

### Automated Fix Script Usage

#### 1. Analyze Current State
```bash
cd /mnt/c/finalee/beverly_knits_erp_v2
python scripts/fix_api_consolidation.py --analyze-only
```

#### 2. Dry Run
```bash
python scripts/fix_api_consolidation.py --dry-run --verbose
```

#### 3. Apply Fixes
```bash
# Create backup first
cp -r src/core/beverly_comprehensive_erp.py src/core/beverly_comprehensive_erp.py.backup
cp web/consolidated_dashboard.html web/consolidated_dashboard.html.backup

# Run fix
python scripts/fix_api_consolidation.py

# Review changes
git diff src/core/beverly_comprehensive_erp.py
git diff web/consolidated_dashboard.html
```

#### 4. Restart Services
```bash
# Stop existing services
pkill -f "python3.*beverly"
pkill -f "uvicorn.*app.main"

# Start ERP wrapper first
cd erp-wrapper/
uvicorn app.main:app --port 8000 &

# Start main ERP
cd ..
python3 src/core/beverly_comprehensive_erp.py
```

### Manual Fix Procedure

#### Step 1: Fix Frontend Redirect
```bash
# Edit dashboard
vim web/consolidated_dashboard.html

# Go to line 3676
:3676

# Change:
'/api/yarn/unified': '/api/yarn/unified',

# To:
'/api/yarn/unified': '/api/yarn-intelligence',

# Save and exit
:wq
```

#### Step 2: Create Backend Unified Endpoint
```bash
# Edit backend
vim src/core/beverly_comprehensive_erp.py

# Go to line 12764 (after yarn-intelligence endpoint)
:12764

# Add the unified endpoint code (see Option B above)

# Save and exit
:wq
```

#### Step 3: Update Redirect Map
```bash
# Still in beverly_comprehensive_erp.py
# Go to line 795 (redirect_map)
:795

# Remove yarn redirects that point to /api/yarn/unified

# Save and exit
:wq
```

---

## Testing & Validation

### 1. Endpoint Testing Script
```python
#!/usr/bin/env python3
"""Test API endpoints after consolidation fix"""

import requests
import json
from typing import Dict, List

SERVER_URL = "http://localhost:5006"

def test_endpoints():
    """Test all critical endpoints"""

    endpoints = [
        "/api/yarn/unified",
        "/api/yarn-intelligence",
        "/api/production/unified",
        "/api/forecast/unified",
        "/api/inventory/unified"
    ]

    results = []

    for endpoint in endpoints:
        try:
            response = requests.get(f"{SERVER_URL}{endpoint}", timeout=5)
            results.append({
                'endpoint': endpoint,
                'status': response.status_code,
                'success': response.status_code == 200,
                'has_data': bool(response.json()) if response.status_code == 200 else False
            })
        except Exception as e:
            results.append({
                'endpoint': endpoint,
                'status': 'ERROR',
                'success': False,
                'error': str(e)
            })

    # Print results
    print("\n" + "="*60)
    print("API ENDPOINT TEST RESULTS")
    print("="*60)

    for result in results:
        status_symbol = "✅" if result['success'] else "❌"
        print(f"{status_symbol} {result['endpoint']}: {result['status']}")
        if 'error' in result:
            print(f"   Error: {result['error']}")

    print("="*60)

    # Check for circular redirects
    print("\nChecking for circular redirects...")
    response = requests.get(
        f"{SERVER_URL}/api/yarn/unified",
        allow_redirects=False,
        timeout=5
    )

    if response.status_code in [301, 302, 307, 308]:
        location = response.headers.get('Location', '')
        if '/api/yarn/unified' in location:
            print("❌ CIRCULAR REDIRECT DETECTED!")
        else:
            print(f"✅ Redirects to: {location}")
    else:
        print(f"✅ No redirect (Status: {response.status_code})")

if __name__ == "__main__":
    test_endpoints()
```

### 2. Browser Console Tests

```javascript
// Test 1: Check redirect behavior
fetch('/api/yarn/unified', {method: 'HEAD'})
    .then(r => console.log('Status:', r.status, 'URL:', r.url));

// Test 2: Verify data retrieval
fetch('/api/yarn/unified')
    .then(r => r.json())
    .then(data => console.log('Data received:', Object.keys(data)));

// Test 3: Check for circular redirects
const checkRedirect = async () => {
    const response = await fetch('/api/yarn/unified', {
        redirect: 'manual'
    });
    console.log('Redirect status:', response.status);
    console.log('Location:', response.headers.get('Location'));
};
checkRedirect();

// Test 4: Verify all API calls
const testAllAPIs = async () => {
    const endpoints = [
        '/api/yarn/unified',
        '/api/production/unified',
        '/api/forecast/unified',
        '/api/inventory/unified'
    ];

    for (const endpoint of endpoints) {
        try {
            const response = await fetch(endpoint);
            const data = await response.json();
            console.log(`✅ ${endpoint}: OK (${Object.keys(data).length} keys)`);
        } catch (error) {
            console.error(`❌ ${endpoint}: FAILED`, error);
        }
    }
};
testAllAPIs();
```

### 3. Dashboard Validation

```javascript
// Run in browser console on dashboard page

// Check if yarn data loads
const validateYarnData = async () => {
    const startTime = Date.now();

    try {
        const data = await fetchAPI('/api/yarn/unified');
        const loadTime = Date.now() - startTime;

        console.log('✅ Yarn data loaded in', loadTime, 'ms');
        console.log('Data structure:', {
            hasData: !!data,
            hasCriticalityAnalysis: !!data.criticality_analysis,
            yarnCount: data.criticality_analysis?.yarns?.length || 0,
            hasShortages: data.criticality_analysis?.yarns?.filter(y => y.shortage > 0).length || 0
        });

        return true;
    } catch (error) {
        console.error('❌ Failed to load yarn data:', error);
        return false;
    }
};

validateYarnData();
```

---

## Monitoring & Maintenance

### 1. Health Check Endpoints

```python
@app.route("/api/health/consolidation")
def health_check_consolidation():
    """Check API consolidation health"""

    health = {
        'timestamp': datetime.now().isoformat(),
        'unified_endpoints': {},
        'deprecated_redirects': {},
        'issues': []
    }

    # Check unified endpoints
    for domain in ['yarn', 'production', 'forecast', 'inventory']:
        endpoint = f"/api/{domain}/unified"
        # Test if endpoint exists
        with app.test_client() as client:
            response = client.get(endpoint)
            health['unified_endpoints'][domain] = {
                'exists': response.status_code != 404,
                'status': response.status_code
            }

    # Check for circular redirects
    for source, target in redirect_map.items():
        if source == target[0]:
            health['issues'].append(f"Circular redirect: {source}")

    health['status'] = 'healthy' if not health['issues'] else 'unhealthy'

    return jsonify(health)
```

### 2. Monitoring Script

```bash
#!/bin/bash
# monitor_api_health.sh

while true; do
    clear
    echo "========================================="
    echo "API HEALTH MONITOR - $(date)"
    echo "========================================="

    # Check main endpoints
    for endpoint in yarn production forecast inventory; do
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5006/api/$endpoint/unified")
        if [ "$response" = "200" ]; then
            echo "✅ /api/$endpoint/unified: OK"
        else
            echo "❌ /api/$endpoint/unified: $response"
        fi
    done

    echo ""
    echo "Redirect Count:"
    curl -s "http://localhost:5006/api/consolidation-metrics" | jq '.redirect_count'

    echo ""
    echo "Cache Hit Rate:"
    curl -s "http://localhost:5006/api/consolidation-metrics" | jq '.cache_metrics.hit_rate'

    sleep 5
done
```

### 3. Log Analysis

```python
#!/usr/bin/env python3
"""Analyze API logs for issues"""

import re
from collections import Counter
from datetime import datetime

def analyze_api_logs(log_file):
    """Analyze Flask logs for API issues"""

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Patterns to look for
    circular_redirects = []
    failed_endpoints = []
    slow_endpoints = []

    for line in lines:
        # Check for circular redirects
        if 'Redirecting deprecated endpoint' in line:
            match = re.search(r'(\S+) → (\S+)', line)
            if match and match.group(1) == match.group(2):
                circular_redirects.append((match.group(1), line))

        # Check for 404s and 500s
        if ' 404 ' in line or ' 500 ' in line:
            match = re.search(r'"[A-Z]+ (/api/[^"]+)', line)
            if match:
                failed_endpoints.append(match.group(1))

        # Check for slow requests (>1000ms)
        match = re.search(r'(\d+)ms', line)
        if match and int(match.group(1)) > 1000:
            endpoint_match = re.search(r'"[A-Z]+ (/api/[^"]+)', line)
            if endpoint_match:
                slow_endpoints.append((endpoint_match.group(1), int(match.group(1))))

    # Report
    print("\n" + "="*60)
    print("API LOG ANALYSIS REPORT")
    print("="*60)

    if circular_redirects:
        print("\n❌ CIRCULAR REDIRECTS FOUND:")
        for redirect, line in circular_redirects[:5]:
            print(f"  • {redirect}")
            print(f"    {line.strip()[:80]}...")

    if failed_endpoints:
        print("\n⚠️ FAILED ENDPOINTS:")
        endpoint_counts = Counter(failed_endpoints)
        for endpoint, count in endpoint_counts.most_common(5):
            print(f"  • {endpoint}: {count} failures")

    if slow_endpoints:
        print("\n🐌 SLOW ENDPOINTS:")
        slow_endpoints.sort(key=lambda x: x[1], reverse=True)
        for endpoint, ms in slow_endpoints[:5]:
            print(f"  • {endpoint}: {ms}ms")

    if not (circular_redirects or failed_endpoints or slow_endpoints):
        print("\n✅ No issues found in logs")

if __name__ == "__main__":
    analyze_api_logs("/var/log/beverly_erp.log")
```

---

## Appendix: Complete API Endpoint Reference

### Yarn Domain (19 endpoints)
| Line | Endpoint | Purpose | Parameters |
|------|----------|---------|------------|
| 10617 | `/api/yarn-data` | Raw yarn data | None |
| 10751 | `/api/yarn-shortage-analysis` | Shortage analysis | None |
| 10853 | `/api/yarn-alternatives` | Alternative yarns | yarn_id |
| 11094 | `/api/yarn-requirements-calculation` | Calculate requirements | style_id |
| 11602 | `/api/fabric/yarn-requirements` | Fabric to yarn conversion | POST: fabric_data |
| 12411 | `/api/yarn` | Basic yarn info | None |
| 12572 | `/api/yarn-aggregation` | Aggregated yarn data | None |
| 12764 | `/api/yarn-intelligence` | Main intelligence endpoint | view, analysis, forecast |
| 13419 | `/api/yarn-shortage-timeline` | Time-phased shortages | weeks |
| 13547 | `/api/time-phased-yarn-po` | PO timeline | None |
| 13744 | `/api/yarn-intelligence-enhanced` | Enhanced intelligence | all params |
| 14504 | `/api/ai/yarn-forecast/<yarn_id>` | AI forecast for yarn | yarn_id (path) |
| 15206 | `/api/pipeline/yarn-shortages` | Pipeline shortages | None |
| 15376 | `/api/inventory-analysis/yarn-shortages` | Inventory shortages | None |
| 15450 | `/api/inventory-analysis/yarn-requirements` | Inventory requirements | POST: data |
| 16301 | `/api/yarn-substitution-opportunities` | Substitution opportunities | None |
| 16391 | `/api/yarn-substitution-intelligent` | ML substitutions | yarn_id, threshold |
| 16535 | `/api/yarn-forecast-shortages` | Forecast shortages | horizon |
| 18768 | `/api/backtest/yarn-comprehensive` | Backtest yarn predictions | GET/POST |

### Production Domain (12 endpoints)
[Similar detailed table for production endpoints]

### Forecast Domain (8 endpoints)
[Similar detailed table for forecast endpoints]

### Inventory Domain (10 endpoints)
[Similar detailed table for inventory endpoints]

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Aug 2025 | Initial API consolidation |
| 1.5.0 | Sep 2025 | Added wrapper service integration |
| 2.0.0 | Sep 13, 2025 | Identified and documented circular redirect issue |

---

*This document is maintained by the Beverly Knits Engineering Team*
*Last Updated: September 13, 2025*
*Next Review: October 2025*