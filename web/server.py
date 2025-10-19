#!/usr/bin/env python3
"""
Simple HTTP server for serving the consolidated dashboard HTML file.
This server handles CORS and serves static files from the web directory.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys
from typing import Any


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support."""

    def end_headers(self) -> None:
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Log HTTP requests with custom formatting."""
        print(f"[HTTP] {format % args}")


def run_server(port: int = 8000, directory: str = ".") -> None:
    """
    Start the HTTP server.

    Args:
        port: Port number to listen on
        directory: Directory to serve files from
    """
    # Change to the web directory
    os.chdir(directory)

    server_address = ("", port)
    httpd = HTTPServer(server_address, CORSRequestHandler)

    print(f"=" * 70)
    print(f"Beverly Knits ERP Dashboard Server")
    print(f"=" * 70)
    print(f"Server running at: http://localhost:{port}")
    print(f"Serving files from: {os.getcwd()}")
    print(f"Dashboard URL: http://localhost:{port}/consolidated_dashboard_visual_preserved.html")
    print(f"=" * 70)
    print(f"Note: Backend API is running on port 5000")
    print(f"Press Ctrl+C to stop the server")
    print(f"=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        httpd.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    # Get port from command line argument or use default
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)

    # Get the web directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_server(port=port, directory=script_dir)
