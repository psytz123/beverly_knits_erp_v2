"""
Simple HTTP server to serve the Beverly Knits ERP Dashboard.

Serves static files from the web/ directory on port 8000.
"""
import http.server
import socketserver
import os
from pathlib import Path

# Change to web directory
web_dir = Path(__file__).parent / "web"
os.chdir(web_dir)

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

print("=" * 80)
print("Beverly Knits ERP - Dashboard Server")
print("=" * 80)
print(f"Serving from: {web_dir}")
print(f"Dashboard URL: http://localhost:{PORT}/consolidated_dashboard.html")
print(f"Server: http://0.0.0.0:{PORT}")
print("=" * 80)
print("Press Ctrl+C to stop")
print("=" * 80)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"\nDashboard server running on port {PORT}...")
    httpd.serve_forever()
