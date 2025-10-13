#!/bin/bash

################################################################################
# Beverly Knits ERP Dashboard Launcher (Linux/Mac)
################################################################################
# This script starts both the backend API server and the frontend dashboard
################################################################################

echo ""
echo "============================================================================"
echo "Beverly Knits ERP Dashboard - Starting Services"
echo "============================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Starting Backend API Server on port 5006..."
echo ""

# Start the backend API server in the background (lightweight version)
python3 src/api/lightweight_api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "API Server PID: $API_PID"

# Wait for the API server to start
sleep 5

echo ""
echo "Starting Frontend Dashboard Server on port 8080..."
echo ""

# Start the dashboard web server in the background
python3 web/server.py 8080 > logs/web_server.log 2>&1 &
WEB_PID=$!
echo "Web Server PID: $WEB_PID"

# Wait a moment
sleep 3

echo ""
echo "============================================================================"
echo "Services Started Successfully!"
echo "============================================================================"
echo ""
echo "Backend API:  http://localhost:5006"
echo "Dashboard:    http://localhost:8080/consolidated_dashboard_visual_preserved.html"
echo ""
echo "Process IDs:"
echo "  API Server: $API_PID"
echo "  Web Server: $WEB_PID"
echo ""
echo "To stop the servers, run:"
echo "  kill $API_PID $WEB_PID"
echo ""
echo "Or use: ./stop_dashboard.sh"
echo "============================================================================"
echo ""

# Save PIDs to a file for easy stopping
mkdir -p logs
echo "$API_PID" > logs/api_server.pid
echo "$WEB_PID" > logs/web_server.pid

# Try to open the dashboard in the default browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8080/consolidated_dashboard_visual_preserved.html 2>/dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:8080/consolidated_dashboard_visual_preserved.html 2>/dev/null &
fi

echo "Dashboard should open in your browser automatically."
echo "If not, navigate to: http://localhost:8080/consolidated_dashboard_visual_preserved.html"
echo ""
