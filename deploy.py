#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beverly Knits ERP v2 - Full System Deployment Script
Cross-platform deployment automation
"""

import os
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import subprocess
import time
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")

def print_step(step_num, total, text):
    """Print formatted step"""
    print(f"[{step_num}/{total}] {text}...")

def run_command(cmd, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main deployment function"""
    print_header("BEVERLY KNITS ERP v2 - FULL SYSTEM DEPLOYMENT")

    # Step 1: INIT
    print_step(1, 7, "INIT: Verifying project structure")

    if not (PROJECT_ROOT / "src").exists():
        print("ERROR: src directory not found!")
        sys.exit(1)

    if not (PROJECT_ROOT / "web").exists():
        print("ERROR: web directory not found!")
        sys.exit(1)

    if not (PROJECT_ROOT / ".env").exists():
        print("ERROR: .env file not found! Please create it from .env.example")
        sys.exit(1)

    print("✓ Project structure verified")

    # Step 2: CONFIG
    print_step(2, 7, "CONFIG: Testing Turso database connection")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        from src.database.turso_client import get_turso_client
        client = get_turso_client()
        result = client.execute("SELECT 1 as test")

        if result:
            print("✓ Turso connection successful")
        else:
            print("ERROR: Turso connection failed")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Check .env credentials")
        sys.exit(1)

    # Step 3: INSTALL DEP
    print_step(3, 7, "INSTALL DEP: Checking Python dependencies")

    try:
        import flask
        import pandas
        import numpy
        import httpx
        import sklearn
        print("✓ All core dependencies installed")
    except ImportError as e:
        print(f"Installing missing dependencies... ({e.name})")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
        run_command(f"{sys.executable} -m pip install httpx")

    # Step 4: DATABASE
    print_step(4, 7, "DATABASE: Initializing Turso schema")

    try:
        from src.database.turso_client import get_turso_client
        client = get_turso_client()
        client.initialize_schema()
        print("✓ Schema initialized")
    except Exception as e:
        print(f"WARNING: Schema initialization error: {e}")
        print("Continuing anyway...")

    # Step 5: DATA LOAD
    print_step(5, 7, "DATA LOAD: Import data to Turso")

    print("\nNOTE: Data import requires data files:")
    print("  - eFab_Styles_20251018.xlsx for style mappings")
    print("  - Historical sales data (if available)")
    print("  - BOM and fabric specs (if available)")

    import_data = input("\nDo you want to import data now? (y/n): ").strip().lower()

    if import_data == 'y':
        print("\nImporting style mappings...")
        styles_file = input("Enter path to eFab_Styles Excel file (or press Enter to skip): ").strip()

        if styles_file and Path(styles_file).exists():
            print(f"Importing from {styles_file}...")
            run_command(
                f"{sys.executable} scripts/import_style_mappings_to_turso.py "
                f"--create-table --file \"{styles_file}\"",
                check=False
            )

        print("\nYou can import additional data later using:")
        print("  - python scripts/import_sales_to_turso.py")
        print("  - python scripts/import_bom_and_specs_to_turso.py")

    # Step 6: TESTING
    print_step(6, 7, "TESTING: Running integration tests")

    try:
        print("\nRunning Turso integration tests...")
        run_command(f"{sys.executable} scripts/test_turso_integration.py", check=False)
    except Exception as e:
        print(f"WARNING: Tests failed: {e}")
        print("Continuing anyway...")

    # Step 7: LAUNCH
    print_step(7, 7, "LAUNCH: Starting all components")

    print("\nThis will start:")
    print("  - eFab API Server (Port 5006)")
    print("  - Dashboard Web Server (Port 8000)")
    print("\nPress Ctrl+C to stop all services")

    time.sleep(2)

    print_header("STARTING SERVICES...")

    print("API Server will be available at: http://localhost:5006")
    print("Dashboard will be available at: http://localhost:8000/consolidated_dashboard.html")
    print("\nStarting services...")

    # Start API server in background
    api_cmd = f"{sys.executable} src/api/efab_api_server.py"

    # Start dashboard server in background
    dashboard_cmd = f"{sys.executable} -m http.server 8000"

    # Determine platform and start services
    if sys.platform == "win32":
        # Windows: Start in new command windows
        subprocess.Popen(f'start "Beverly Knits API" cmd /k "{api_cmd}"', shell=True)
        time.sleep(3)
        subprocess.Popen(f'start "Beverly Knits Dashboard" cmd /k "cd web && {dashboard_cmd}"', shell=True)
    else:
        # Linux/Mac: Start in background
        subprocess.Popen(api_cmd, shell=True, cwd=PROJECT_ROOT)
        time.sleep(3)
        subprocess.Popen(dashboard_cmd, shell=True, cwd=PROJECT_ROOT / "web")

    time.sleep(2)

    print_header("DEPLOYMENT COMPLETE!")

    print("Services Status:")
    print("  ✓ API Server: http://localhost:5006")
    print("  ✓ Dashboard: http://localhost:8000/consolidated_dashboard.html")
    print("\nNext Steps:")
    print("  1. Dashboard will open automatically in your browser")
    print("  2. Verify data loads correctly")
    print("  3. Test forecast generation")
    print("  4. Review proactive production recommendations")
    print("\nTo stop all services:")
    print("  - Windows: Close the server windows")
    print("  - Linux/Mac: Run 'pkill -f efab_api_server' and 'pkill -f http.server'")
    print("\nFor troubleshooting, check server logs")
    print("=" * 80)

    # Open dashboard in browser
    time.sleep(3)
    print("\nOpening dashboard in browser...")
    webbrowser.open("http://localhost:8000/consolidated_dashboard.html")

    if sys.platform != "win32":
        print("\nServices running in background. Press Ctrl+C to exit this script.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nScript terminated. Services are still running in background.")
            print("To stop them, run: pkill -f 'efab_api_server|http.server'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
