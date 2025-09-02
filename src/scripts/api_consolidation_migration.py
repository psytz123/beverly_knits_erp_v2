#!/usr/bin/env python3
"""
API Consolidation Migration Script
Helps migrate from old endpoints to new consolidated ones
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse
import time

BASE_URL = "http://localhost:5006"

class APIConsolidationMigrator:
    """Handles the migration from old to new consolidated endpoints"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.migration_report = {
            'started_at': datetime.now().isoformat(),
            'phases': [],
            'errors': [],
            'warnings': []
        }
    
    def check_server_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_current_feature_flags(self) -> Dict:
        """Get current feature flag settings"""
        try:
            response = requests.get(f"{self.base_url}/api/admin/feature-flags", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.migration_report['errors'].append(f"Failed to get feature flags: {e}")
            return {}
    
    def update_feature_flag(self, flag_name: str, value: Any) -> bool:
        """Update a single feature flag"""
        try:
            response = requests.post(
                f"{self.base_url}/api/admin/feature-flags",
                json={flag_name: value},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.migration_report['errors'].append(f"Failed to update {flag_name}: {e}")
            return False
    
    def test_endpoint(self, url: str, params: Dict = None) -> bool:
        """Test if an endpoint is working"""
        try:
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def phase1_enable_redirects(self) -> bool:
        """Phase 1: Enable redirect middleware"""
        print("\n=== Phase 1: Enable Redirect Middleware ===")
        phase_report = {
            'phase': 'enable_redirects',
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        # Enable redirects
        if self.update_feature_flag('enable_redirects', True):
            print("✓ Redirects enabled")
            
            # Enable logging
            if self.update_feature_flag('log_deprecated_usage', True):
                print("✓ Deprecated usage logging enabled")
                phase_report['success'] = True
            else:
                print("✗ Failed to enable logging")
        else:
            print("✗ Failed to enable redirects")
        
        phase_report['completed_at'] = datetime.now().isoformat()
        self.migration_report['phases'].append(phase_report)
        return phase_report['success']
    
    def phase2_test_consolidated_endpoints(self) -> bool:
        """Phase 2: Test all consolidated endpoints"""
        print("\n=== Phase 2: Test Consolidated Endpoints ===")
        phase_report = {
            'phase': 'test_consolidated',
            'started_at': datetime.now().isoformat(),
            'endpoints_tested': [],
            'success': True
        }
        
        endpoints = [
            {'name': 'inventory', 'url': f'{self.base_url}/api/inventory/unified'},
            {'name': 'forecast', 'url': f'{self.base_url}/api/forecast/unified'},
            {'name': 'production', 'url': f'{self.base_url}/api/production/unified'},
            {'name': 'yarn', 'url': f'{self.base_url}/api/yarn/unified'},
            {'name': 'planning', 'url': f'{self.base_url}/api/planning/unified'},
            {'name': 'system', 'url': f'{self.base_url}/api/system/unified'}
        ]
        
        for endpoint in endpoints:
            success = self.test_endpoint(endpoint['url'])
            phase_report['endpoints_tested'].append({
                'name': endpoint['name'],
                'success': success
            })
            
            if success:
                print(f"✓ {endpoint['name']} endpoint working")
            else:
                print(f"✗ {endpoint['name']} endpoint failed")
                phase_report['success'] = False
        
        phase_report['completed_at'] = datetime.now().isoformat()
        self.migration_report['phases'].append(phase_report)
        return phase_report['success']
    
    def phase3_gradual_consolidation(self, category: str) -> bool:
        """Phase 3: Enable consolidation for a specific category"""
        print(f"\n=== Phase 3: Enable {category} Consolidation ===")
        phase_report = {
            'phase': f'enable_{category}_consolidation',
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        flag_name = f"{category}_consolidated"
        
        # Enable consolidation for this category
        if self.update_feature_flag(flag_name, True):
            print(f"✓ {category} consolidation enabled")
            
            # Test the consolidated endpoint
            endpoint_url = f"{self.base_url}/api/{category}/unified"
            if self.test_endpoint(endpoint_url):
                print(f"✓ {category} consolidated endpoint working")
                phase_report['success'] = True
            else:
                print(f"✗ {category} consolidated endpoint failed")
                # Rollback
                self.update_feature_flag(flag_name, False)
                print(f"↺ Rolled back {category} consolidation")
        else:
            print(f"✗ Failed to enable {category} consolidation")
        
        phase_report['completed_at'] = datetime.now().isoformat()
        self.migration_report['phases'].append(phase_report)
        return phase_report['success']
    
    def phase4_monitor_usage(self, duration_seconds: int = 60) -> Dict:
        """Phase 4: Monitor deprecated endpoint usage"""
        print(f"\n=== Phase 4: Monitor Usage ({duration_seconds}s) ===")
        phase_report = {
            'phase': 'monitor_usage',
            'started_at': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'deprecation_report': None
        }
        
        print(f"Monitoring for {duration_seconds} seconds...")
        time.sleep(duration_seconds)
        
        # Get deprecation report
        try:
            response = requests.get(f"{self.base_url}/api/admin/deprecation-report", timeout=5)
            if response.status_code == 200:
                phase_report['deprecation_report'] = response.json()
                report = response.json()
                print(f"✓ Total deprecated calls: {report.get('total_deprecated_calls', 0)}")
                print(f"✓ Unique deprecated endpoints: {report.get('unique_deprecated_endpoints', 0)}")
                
                # Show top deprecated endpoints
                if report.get('endpoints'):
                    print("\nTop deprecated endpoints:")
                    for ep in report['endpoints'][:5]:
                        print(f"  - {ep['old_endpoint']}: {ep['usage_count']} calls")
        except Exception as e:
            self.migration_report['errors'].append(f"Failed to get deprecation report: {e}")
            print(f"✗ Failed to get deprecation report: {e}")
        
        phase_report['completed_at'] = datetime.now().isoformat()
        self.migration_report['phases'].append(phase_report)
        return phase_report
    
    def phase5_complete_migration(self) -> bool:
        """Phase 5: Complete the migration"""
        print("\n=== Phase 5: Complete Migration ===")
        phase_report = {
            'phase': 'complete_migration',
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        # Enable all consolidations
        categories = ['inventory', 'forecasting', 'production', 'yarn', 'planning', 'cache']
        all_success = True
        
        for category in categories:
            if self.update_feature_flag(f"{category}_consolidated", True):
                print(f"✓ {category} consolidation enabled")
            else:
                print(f"✗ Failed to enable {category} consolidation")
                all_success = False
        
        if all_success:
            print("\n✓ All consolidations enabled successfully")
            
            # Set block date for deprecated endpoints (30 days from now)
            block_date = (datetime.now() + timedelta(days=30)).isoformat()
            if self.update_feature_flag('block_deprecated_after', block_date):
                print(f"✓ Deprecated endpoints will be blocked after {block_date}")
                phase_report['success'] = True
        
        phase_report['completed_at'] = datetime.now().isoformat()
        self.migration_report['phases'].append(phase_report)
        return phase_report['success']
    
    def rollback_all(self) -> bool:
        """Rollback all consolidation changes"""
        print("\n=== ROLLBACK: Disabling All Consolidations ===")
        
        flags_to_disable = [
            'inventory_consolidated',
            'forecasting_consolidated',
            'production_consolidated',
            'yarn_consolidated',
            'planning_consolidated',
            'cache_consolidated',
            'enable_redirects'
        ]
        
        success = True
        for flag in flags_to_disable:
            if self.update_feature_flag(flag, False):
                print(f"✓ Disabled {flag}")
            else:
                print(f"✗ Failed to disable {flag}")
                success = False
        
        # Clear block date
        self.update_feature_flag('block_deprecated_after', None)
        
        return success
    
    def save_report(self, filename: str = "migration_report.json"):
        """Save migration report to file"""
        self.migration_report['completed_at'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        print(f"\n✓ Migration report saved to {filename}")
    
    def run_migration(self, mode: str = 'gradual'):
        """Run the complete migration process"""
        print("=" * 60)
        print("API Consolidation Migration")
        print(f"Mode: {mode}")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # Check server health
        if not self.check_server_health():
            print("\n✗ ERROR: Server is not running or unhealthy")
            return False
        
        print("\n✓ Server is healthy")
        
        # Get current state
        current_flags = self.get_current_feature_flags()
        print(f"\nCurrent feature flags:")
        for flag, value in current_flags.items():
            print(f"  - {flag}: {value}")
        
        if mode == 'gradual':
            # Gradual migration
            if self.phase1_enable_redirects():
                if self.phase2_test_consolidated_endpoints():
                    # Enable consolidation for each category one by one
                    categories = ['inventory', 'forecasting', 'production', 'yarn', 'planning', 'cache']
                    for category in categories:
                        if not self.phase3_gradual_consolidation(category):
                            print(f"\n✗ Failed to consolidate {category}, stopping migration")
                            break
                        time.sleep(2)  # Brief pause between categories
                    
                    # Monitor usage
                    self.phase4_monitor_usage(30)
                    
                    # Complete migration
                    self.phase5_complete_migration()
        
        elif mode == 'immediate':
            # Immediate migration - enable everything at once
            if self.phase1_enable_redirects():
                if self.phase2_test_consolidated_endpoints():
                    self.phase5_complete_migration()
        
        elif mode == 'test':
            # Test mode - just test endpoints without enabling
            self.phase2_test_consolidated_endpoints()
        
        elif mode == 'rollback':
            # Rollback mode
            self.rollback_all()
        
        # Save report
        self.save_report()
        
        print("\n" + "=" * 60)
        print("Migration Complete")
        print("=" * 60)
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='API Consolidation Migration Tool')
    parser.add_argument(
        '--mode',
        choices=['gradual', 'immediate', 'test', 'rollback'],
        default='gradual',
        help='Migration mode (default: gradual)'
    )
    parser.add_argument(
        '--url',
        default=BASE_URL,
        help=f'Base URL of the server (default: {BASE_URL})'
    )
    
    args = parser.parse_args()
    
    migrator = APIConsolidationMigrator(args.url)
    success = migrator.run_migration(args.mode)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())