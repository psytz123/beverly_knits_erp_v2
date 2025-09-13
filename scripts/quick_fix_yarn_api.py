#!/usr/bin/env python3
"""
Quick Fix Script for Yarn API Circular Redirect Issue
======================================================
Purpose: Immediately fix the yarn API issue with minimal changes
Version: 1.0.0
Date: September 13, 2025

This script performs the MINIMAL changes needed to fix the issue:
1. Updates frontend redirect map to point to existing endpoint
2. Optionally creates the unified endpoint in backend
3. Validates the fix
"""

import os
import sys
import re
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import requests
import time

# Configuration
PROJECT_ROOT = Path("/mnt/c/finalee/beverly_knits_erp_v2")
BACKEND_FILE = PROJECT_ROOT / "src/core/beverly_comprehensive_erp.py"
FRONTEND_FILE = PROJECT_ROOT / "web/consolidated_dashboard.html"
BACKUP_DIR = PROJECT_ROOT / "backups" / f"quick_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

class YarnAPIQuickFix:
    """Quick fix for yarn API circular redirect"""

    def __init__(self, create_unified=False, skip_backup=False):
        self.create_unified = create_unified
        self.skip_backup = skip_backup
        self.server_was_running = False
        self.changes_made = []

    def run(self):
        """Execute the quick fix"""
        print_header("YARN API QUICK FIX")

        # Step 1: Check current status
        print_info("Step 1: Checking current status...")
        if not self.check_current_status():
            print_error("Pre-check failed. Please review the errors above.")
            return False

        # Step 2: Create backup
        if not self.skip_backup:
            print_info("Step 2: Creating backups...")
            self.create_backups()

        # Step 3: Stop server if running
        print_info("Step 3: Checking server status...")
        self.stop_server_if_running()

        # Step 4: Apply frontend fix
        print_info("Step 4: Applying frontend fix...")
        if not self.fix_frontend_redirect():
            print_error("Frontend fix failed!")
            self.rollback()
            return False

        # Step 5: Optionally create unified endpoint
        if self.create_unified:
            print_info("Step 5: Creating unified endpoint...")
            if not self.create_unified_endpoint():
                print_error("Backend fix failed!")
                self.rollback()
                return False
        else:
            print_info("Step 5: Skipping unified endpoint creation (using redirect)")

        # Step 6: Restart server if it was running
        if self.server_was_running:
            print_info("Step 6: Restarting server...")
            self.restart_server()
            time.sleep(5)  # Wait for server to start

        # Step 7: Validate fix
        print_info("Step 7: Validating fix...")
        if self.validate_fix():
            print_success("Fix successfully applied!")
            self.print_summary()
            return True
        else:
            print_error("Validation failed!")
            self.rollback()
            return False

    def check_current_status(self):
        """Check if the issue exists"""
        issues_found = []

        # Check if frontend file exists
        if not FRONTEND_FILE.exists():
            print_error(f"Frontend file not found: {FRONTEND_FILE}")
            return False

        # Check if backend file exists
        if not BACKEND_FILE.exists():
            print_error(f"Backend file not found: {BACKEND_FILE}")
            return False

        # Check for circular redirect in frontend
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        if "'/api/yarn-intelligence': '/api/yarn-intelligence'" in content:
            print_warning("Found circular redirect: /api/yarn-intelligence → /api/yarn-intelligence")
            issues_found.append("circular_redirect")
        else:
            print_info("No circular redirect found in frontend")

        # Check if unified endpoint exists in backend
        with open(BACKEND_FILE, 'r', encoding='utf-8') as f:
            backend_content = f.read()

        if '@app.route("/api/yarn-intelligence")' not in backend_content:
            print_warning("Missing backend endpoint: /api/yarn-intelligence")
            issues_found.append("missing_endpoint")
        else:
            print_info("Backend endpoint /api/yarn-intelligence exists")

        if not issues_found:
            print_success("No issues found! The API appears to be working correctly.")
            return False

        print_info(f"Found {len(issues_found)} issue(s) to fix")
        return True

    def create_backups(self):
        """Create backup copies of files"""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        for file_path in [FRONTEND_FILE, BACKEND_FILE]:
            if file_path.exists():
                backup_path = BACKUP_DIR / file_path.name
                shutil.copy2(file_path, backup_path)
                print_success(f"Backed up {file_path.name}")

        # Save backup info
        backup_info = {
            'timestamp': datetime.now().isoformat(),
            'files': [str(FRONTEND_FILE), str(BACKEND_FILE)],
            'backup_dir': str(BACKUP_DIR)
        }

        with open(BACKUP_DIR / 'backup_info.json', 'w') as f:
            json.dump(backup_info, f, indent=2)

    def stop_server_if_running(self):
        """Check if server is running and stop it"""
        try:
            response = requests.get("http://localhost:5006/api/health", timeout=2)
            if response.status_code == 200:
                self.server_was_running = True
                print_warning("Server is running. Stopping it...")

                # Stop the server
                os.system("pkill -f 'python3.*beverly'")
                time.sleep(2)
                print_success("Server stopped")
        except:
            print_info("Server is not running")

    def fix_frontend_redirect(self):
        """Fix the circular redirect in frontend"""
        try:
            with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find and fix the circular redirect
            original_line = "'/api/yarn-intelligence': '/api/yarn-intelligence'"
            fixed_line = "'/api/yarn-intelligence': '/api/yarn-intelligence'"

            if original_line in content:
                content = content.replace(original_line, fixed_line)

                with open(FRONTEND_FILE, 'w', encoding='utf-8') as f:
                    f.write(content)

                print_success("Fixed circular redirect in frontend")
                print_info(f"  Changed: {original_line}")
                print_info(f"  To: {fixed_line}")
                self.changes_made.append("frontend_redirect")
                return True
            else:
                print_warning("Circular redirect not found in frontend (may already be fixed)")
                return True

        except Exception as e:
            print_error(f"Error fixing frontend: {e}")
            return False

    def create_unified_endpoint(self):
        """Create the unified endpoint in backend"""
        try:
            with open(BACKEND_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Check if endpoint already exists
            for line in lines:
                if '@app.route("/api/yarn-intelligence")' in line:
                    print_warning("Unified endpoint already exists in backend")
                    return True

            # Find where to insert (after /api/yarn-intelligence endpoint)
            insert_line = None
            for i, line in enumerate(lines):
                if '@app.route("/api/yarn-intelligence")' in line:
                    # Find the end of this function
                    for j in range(i + 1, len(lines)):
                        # Look for next route or end of file
                        if j < len(lines) - 1:
                            if '@app.route' in lines[j] or (j > i + 10 and lines[j].strip() and not lines[j].startswith(' ')):
                                insert_line = j
                                break

            if insert_line is None:
                print_error("Could not find insertion point for unified endpoint")
                return False

            # Create the unified endpoint code
            unified_code = '''
@app.route("/api/yarn-intelligence")
def get_yarn_unified():
    """
    Unified yarn endpoint - Consolidates all yarn functionality
    Created by Quick Fix Script

    This endpoint routes to the appropriate yarn handler based on parameters.
    """
    global new_api_count
    new_api_count += 1

    # Get request parameters
    view = request.args.get('view', 'full')
    analysis = request.args.get('analysis', 'standard')
    substitution = request.args.get('substitution', 'false').lower() == 'true'

    try:
        # Log the request
        logger.info(f"Yarn unified endpoint called with view={view}, analysis={analysis}, substitution={substitution}")

        # Route to appropriate handler
        if substitution:
            # Delegate to substitution endpoint
            return yarn_substitution_intelligent()
        else:
            # Delegate to main intelligence endpoint
            return get_yarn_intelligence()

    except Exception as e:
        logger.error(f"Error in yarn unified endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

'''

            # Insert the code
            lines.insert(insert_line, unified_code)

            # Write back
            with open(BACKEND_FILE, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print_success("Created unified endpoint in backend")
            self.changes_made.append("backend_endpoint")
            return True

        except Exception as e:
            print_error(f"Error creating backend endpoint: {e}")
            return False

    def restart_server(self):
        """Restart the server"""
        try:
            print_info("Starting server...")

            # Start the server in background
            subprocess.Popen(
                ["python3", "src/core/beverly_comprehensive_erp.py"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Wait for server to be ready
            for i in range(10):
                try:
                    response = requests.get("http://localhost:5006/api/health", timeout=2)
                    if response.status_code == 200:
                        print_success("Server started successfully")
                        return True
                except:
                    time.sleep(1)

            print_warning("Server started but may not be fully ready")
            return True

        except Exception as e:
            print_error(f"Error starting server: {e}")
            return False

    def validate_fix(self):
        """Validate that the fix works"""
        validation_passed = True

        # Check 1: No circular redirect in frontend
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        if "'/api/yarn-intelligence': '/api/yarn-intelligence'" in content:
            print_error("Validation failed: Circular redirect still exists")
            validation_passed = False
        else:
            print_success("Frontend redirect is correct")

        # Check 2: Test API if server is running
        try:
            # Test the endpoint
            response = requests.get("http://localhost:5006/api/yarn-intelligence", timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data and 'criticality_analysis' in data:
                    print_success("API endpoint returns valid data")
                else:
                    print_warning("API endpoint returns incomplete data")
            else:
                print_error(f"API endpoint returned status {response.status_code}")
                validation_passed = False

        except requests.exceptions.ConnectionError:
            print_info("Server not running - skipping API test")
        except Exception as e:
            print_warning(f"Could not test API: {e}")

        return validation_passed

    def rollback(self):
        """Rollback changes if something went wrong"""
        print_warning("Rolling back changes...")

        if not self.skip_backup and BACKUP_DIR.exists():
            # Restore frontend
            frontend_backup = BACKUP_DIR / FRONTEND_FILE.name
            if frontend_backup.exists():
                shutil.copy2(frontend_backup, FRONTEND_FILE)
                print_success("Restored frontend file")

            # Restore backend if we modified it
            if "backend_endpoint" in self.changes_made:
                backend_backup = BACKUP_DIR / BACKEND_FILE.name
                if backend_backup.exists():
                    shutil.copy2(backend_backup, BACKEND_FILE)
                    print_success("Restored backend file")

    def print_summary(self):
        """Print summary of changes"""
        print_header("FIX SUMMARY")

        print(f"{Colors.BOLD}Changes Applied:{Colors.ENDC}")
        if "frontend_redirect" in self.changes_made:
            print("  ✅ Fixed circular redirect in frontend")
            print("     /api/yarn-intelligence now redirects to /api/yarn-intelligence")

        if "backend_endpoint" in self.changes_made:
            print("  ✅ Created /api/yarn-intelligence endpoint in backend")
            print("     New endpoint delegates to existing yarn handlers")

        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Clear browser cache (Ctrl+Shift+R)")
        print("  2. Reload the dashboard")
        print("  3. Check if yarn data displays correctly")

        if not self.server_was_running:
            print(f"\n{Colors.WARNING}Note: Server was not running. Start it with:{Colors.ENDC}")
            print("  cd /mnt/c/finalee/beverly_knits_erp_v2")
            print("  python3 src/core/beverly_comprehensive_erp.py")

        if not self.skip_backup:
            print(f"\n{Colors.BOLD}Backup Location:{Colors.ENDC}")
            print(f"  {BACKUP_DIR}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quick fix for yarn API circular redirect issue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script provides a QUICK FIX for the yarn API issue.

It will:
1. Fix the circular redirect in the frontend
2. Optionally create the unified endpoint in backend
3. Validate the changes

Examples:
  # Quick fix (frontend only)
  python quick_fix_yarn_api.py

  # Complete fix (frontend + backend)
  python quick_fix_yarn_api.py --create-unified

  # Fix without backup (risky but faster)
  python quick_fix_yarn_api.py --skip-backup
        """
    )

    parser.add_argument(
        '--create-unified',
        action='store_true',
        help='Also create the unified endpoint in backend (recommended)'
    )

    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip creating backups (faster but risky)'
    )

    args = parser.parse_args()

    # Run the fix
    fixer = YarnAPIQuickFix(
        create_unified=args.create_unified,
        skip_backup=args.skip_backup
    )

    try:
        success = fixer.run()

        if success:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ FIX COMPLETED SUCCESSFULLY!{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}❌ FIX FAILED OR NOT NEEDED{Colors.ENDC}")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Process interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()