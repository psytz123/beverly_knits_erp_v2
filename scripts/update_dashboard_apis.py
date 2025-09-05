#!/usr/bin/env python3
"""
Script to update dashboard API calls to use new consolidated v2 endpoints
"""

import re
import os
import shutil
from datetime import datetime

# API endpoint mappings from old to new
API_MAPPINGS = {
    # Inventory endpoints
    '/api/inventory-intelligence-enhanced': '/api/v2/inventory?analysis=intelligence&realtime=true',
    '/api/yarn-inventory': '/api/v2/inventory?view=yarn',
    '/api/yarn-data': '/api/v2/inventory?view=yarn&format=json',
    '/api/real-time-inventory-dashboard': '/api/v2/inventory?realtime=true',
    '/api/emergency-shortage-dashboard': '/api/v2/inventory?view=shortage&analysis=shortage',
    
    # Production endpoints
    '/api/production-planning': '/api/v2/production?view=planning',
    '/api/production-status': '/api/v2/production?view=status',
    '/api/production-pipeline': '/api/v2/production?view=pipeline',
    '/api/production-recommendations-ml': '/api/v2/production?view=recommendations',
    '/api/machine-assignment-suggestions': '/api/v2/production?view=machines',
    '/api/production-flow': '/api/v2/production?view=flow',
    '/api/production-suggestions': '/api/v2/production?view=suggestions',
    
    # Forecast endpoints
    '/api/ml-forecast-detailed': '/api/v2/forecast?detail=full',
    '/api/ml-forecasting': '/api/v2/forecast',
    
    # Yarn endpoints
    '/api/yarn-intelligence': '/api/v2/yarn?analysis=intelligence',
    '/api/yarn-substitution-intelligent': '/api/v2/yarn?view=substitutions',
    
    # Analytics endpoints
    '/api/comprehensive-kpis': '/api/v2/analytics/kpis',
    '/api/inventory-netting': '/api/v2/analytics/netting',
    '/api/po-risk-analysis': '/api/v2/analytics/risk',
    
    # Fabric production
    '/api/fabric-production': '/api/v2/fabric-production',
    
    # Keep some endpoints as-is (not migrated yet)
    '/api/knit-orders': '/api/knit-orders',  # Keep as-is for now
    '/api/retrain-ml': '/api/retrain-ml',  # Keep as-is for now
}

def update_dashboard_file(filepath):
    """Update API calls in a dashboard HTML file."""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup the original
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(filepath, backup_path)
    print(f"[INFO] Backup created: {backup_path}")
    
    # Track replacements
    replacements_made = []
    
    # Replace each API endpoint
    for old_api, new_api in API_MAPPINGS.items():
        # Pattern to match fetch calls
        pattern1 = f"fetch\\(baseUrl \\+ '{old_api}'\\)"
        replacement1 = f"fetch(baseUrl + '{new_api}')"
        
        pattern2 = f"fetch\\(API_BASE_URL \\+ '{old_api}'\\)"
        replacement2 = f"fetch(API_BASE_URL + '{new_api}')"
        
        pattern3 = f"'{old_api}'"
        replacement3 = f"'{new_api}'"
        
        # Count replacements
        count = 0
        
        # Replace pattern 1
        if re.search(pattern1, content):
            content = re.sub(pattern1, replacement1, content)
            count += len(re.findall(pattern1, content))
        
        # Replace pattern 2
        if re.search(pattern2, content):
            content = re.sub(pattern2, replacement2, content)
            count += len(re.findall(pattern2, content))
        
        # Replace direct string references (be careful)
        if old_api in content and count == 0:
            # Only replace if it looks like an API call
            api_pattern = f"['\"]({re.escape(old_api)})['\"]"
            if re.search(api_pattern, content):
                content = re.sub(api_pattern, f"'{new_api}'", content)
                count += 1
        
        if count > 0:
            replacements_made.append(f"{old_api} -> {new_api} ({count} replacements)")
    
    # Add API compatibility layer if not present
    compatibility_layer = """
// API Compatibility Layer for v2 endpoints
const apiCompatibilityLayer = {
    // Map old endpoints to new with parameter support
    mapEndpoint(oldEndpoint) {
        const mappings = """ + str(API_MAPPINGS) + """;
        return mappings[oldEndpoint] || oldEndpoint;
    },
    
    // Enhanced fetch with automatic endpoint mapping
    async fetchAPI(endpoint, options = {}) {
        const mappedEndpoint = this.mapEndpoint(endpoint);
        const url = (endpoint.startsWith('http') ? '' : (window.API_BASE_URL || window.baseUrl || '')) + mappedEndpoint;
        
        try {
            const response = await fetch(url, options);
            if (!response.ok && mappedEndpoint !== endpoint) {
                // Fallback to old endpoint if new one fails
                console.warn(`New endpoint failed, falling back to old: ${endpoint}`);
                return await fetch((window.API_BASE_URL || window.baseUrl || '') + endpoint, options);
            }
            return response;
        } catch (error) {
            console.error(`API call failed: ${endpoint}`, error);
            throw error;
        }
    }
};

// Override fetch for API calls (optional - uncomment to enable)
// const originalFetch = window.fetch;
// window.fetch = function(url, ...args) {
//     if (typeof url === 'string' && url.includes('/api/')) {
//         const endpoint = url.replace(window.API_BASE_URL || window.baseUrl || '', '');
//         return apiCompatibilityLayer.fetchAPI(endpoint, ...args);
//     }
//     return originalFetch(url, ...args);
// };
"""
    
    # Add compatibility layer if not present
    if 'apiCompatibilityLayer' not in content:
        # Find a good place to insert it (after the safety checks)
        insert_pos = content.find('// === END COMPREHENSIVE SAFETY CHECKS ===')
        if insert_pos != -1:
            insert_pos = content.find('</script>', insert_pos)
            if insert_pos != -1:
                content = content[:insert_pos] + compatibility_layer + '\n' + content[insert_pos:]
                replacements_made.append("Added API compatibility layer")
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return replacements_made

def main():
    """Main function to update dashboard APIs."""
    print("=" * 80)
    print("Dashboard API Migration Script")
    print("=" * 80)
    
    # Dashboard file path
    dashboard_file = "D:/AI/Workspaces/efab.ai/beverly_knits_erp_v2/web/consolidated_dashboard.html"
    
    if not os.path.exists(dashboard_file):
        print(f"[ERROR] Dashboard file not found: {dashboard_file}")
        return 1
    
    print(f"\n[INFO] Updating dashboard: {dashboard_file}")
    
    # Update the dashboard
    replacements = update_dashboard_file(dashboard_file)
    
    print(f"\n[OK] Dashboard updated successfully!")
    print(f"[OK] Total changes made: {len(replacements)}")
    
    if replacements:
        print("\nChanges made:")
        for replacement in replacements:
            print(f"  - {replacement}")
    
    print("\n" + "=" * 80)
    print("API Migration Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test the dashboard to ensure all API calls work")
    print("  2. Monitor browser console for any errors")
    print("  3. Check that data loads correctly in all tabs")
    print("  4. Verify backward compatibility with old endpoints")
    
    return 0

if __name__ == "__main__":
    exit(main())