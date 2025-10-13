#!/usr/bin/env python3
"""Test dashboard loading in a headless browser to see actual requests."""
import subprocess
import time

# Use PowerShell to open the page and wait
print("Opening dashboard in browser...")
subprocess.Popen([
    'powershell', '-Command',
    'Start-Process "http://localhost:8081/consolidated_dashboard_visual_preserved.html"'
])

print("Waiting 10 seconds for page to load...")
time.sleep(10)

print("Check the eFab server logs for incoming requests.")
print("If you see NO new requests in the logs, the browser is NOT connecting.")
