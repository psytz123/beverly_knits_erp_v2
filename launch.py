#!/usr/bin/env python3
"""
Quick launch script for Beverly Knits ERP Dashboard
Starts both backend API and frontend web servers
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# Fix Windows console encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_python() -> bool:
    """Check if Python is available."""
    try:
        result = subprocess.run(
            [sys.executable, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"[OK] Python: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"[ERROR] Python check failed: {e}")
        return False


def start_api_server(script_dir: Path) -> subprocess.Popen:
    """Start the backend API server."""
    api_script = script_dir / 'src' / 'api' / 'lightweight_api_server.py'

    print("\n" + "=" * 70)
    print("Starting Backend API Server...")
    print("=" * 70)

    if os.name == 'nt':  # Windows
        process = subprocess.Popen(
            [sys.executable, str(api_script)],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Linux/Mac
        process = subprocess.Popen(
            [sys.executable, str(api_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    print(f"[OK] API Server started (PID: {process.pid})")
    return process


def start_web_server(script_dir: Path) -> subprocess.Popen:
    """Start the frontend web server."""
    web_script = script_dir / 'web' / 'server.py'

    print("\n" + "=" * 70)
    print("Starting Frontend Dashboard Server...")
    print("=" * 70)

    if os.name == 'nt':  # Windows
        process = subprocess.Popen(
            [sys.executable, str(web_script), '8080'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Linux/Mac
        process = subprocess.Popen(
            [sys.executable, str(web_script), '8080'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    print(f"[OK] Web Server started (PID: {process.pid})")
    return process


def open_dashboard() -> None:
    """Open the dashboard in the default browser."""
    dashboard_url = 'http://localhost:8080/consolidated_dashboard_visual_preserved.html'

    print("\n" + "=" * 70)
    print("Opening Dashboard...")
    print("=" * 70)

    time.sleep(2)  # Wait for servers to fully start

    try:
        webbrowser.open(dashboard_url)
        print(f"[OK] Dashboard opened: {dashboard_url}")
    except Exception as e:
        print(f"[ERROR] Could not open browser: {e}")
        print(f"  Please manually navigate to: {dashboard_url}")


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print("Beverly Knits ERP Dashboard - Quick Launch")
    print("=" * 70)

    # Get script directory
    script_dir = Path(__file__).parent

    # Check Python
    if not check_python():
        print("\nERROR: Python is not available")
        sys.exit(1)

    try:
        # Start servers
        api_process = start_api_server(script_dir)
        time.sleep(3)  # Wait for API server to initialize

        web_process = start_web_server(script_dir)
        time.sleep(2)  # Wait for web server to initialize

        # Open dashboard
        open_dashboard()

        # Display info
        print("\n" + "=" * 70)
        print("Services Running")
        print("=" * 70)
        print(f"Backend API:  http://localhost:5006")
        print(f"Dashboard:    http://localhost:8080/consolidated_dashboard_visual_preserved.html")
        print("\nProcess IDs:")
        print(f"  API Server: {api_process.pid}")
        print(f"  Web Server: {web_process.pid}")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 70)

        # Keep script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping services...")
            api_process.terminate()
            web_process.terminate()
            print("[OK] Services stopped")

    except Exception as e:
        print(f"\n[ERROR] Error during launch: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
