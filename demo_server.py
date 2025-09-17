#!/usr/bin/env python3
"""
Simple demo server for inventory improvements dashboard
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8888
DIRECTORY = Path(__file__).parent / "web"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Add CORS headers for demo
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        # Custom logging
        print(f"[DEMO] {self.address_string()} - {format % args}")

def main():
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Beverly Knits - Inventory Improvements Demo Server      ║
    ╚══════════════════════════════════════════════════════════════╝

    Starting demo server on port {PORT}...

    Access the demo dashboards at:

    🔹 Live Data Demo (NEW):
       http://localhost:{PORT}/inventory_improvements_demo_v3.html

    🔹 Time-Phased View Demo:
       http://localhost:{PORT}/inventory_improvements_demo_v2.html

    🔹 Improvements Demo:
       http://localhost:{PORT}/inventory_improvements_demo.html

    🔹 Original Dashboard:
       http://localhost:{PORT}/consolidated_dashboard.html

    Press Ctrl+C to stop the server.
    """)

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()