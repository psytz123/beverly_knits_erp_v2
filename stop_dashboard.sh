#!/bin/bash

################################################################################
# Beverly Knits ERP Dashboard Stopper (Linux/Mac)
################################################################################
# This script stops the backend API server and frontend dashboard
################################################################################

echo ""
echo "============================================================================"
echo "Beverly Knits ERP Dashboard - Stopping Services"
echo "============================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if PID files exist
if [ -f logs/api_server.pid ]; then
    API_PID=$(cat logs/api_server.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "Stopping API Server (PID: $API_PID)..."
        kill $API_PID
        rm logs/api_server.pid
    else
        echo "API Server is not running"
        rm logs/api_server.pid
    fi
else
    echo "No API Server PID file found"
fi

if [ -f logs/web_server.pid ]; then
    WEB_PID=$(cat logs/web_server.pid)
    if ps -p $WEB_PID > /dev/null 2>&1; then
        echo "Stopping Web Server (PID: $WEB_PID)..."
        kill $WEB_PID
        rm logs/web_server.pid
    else
        echo "Web Server is not running"
        rm logs/web_server.pid
    fi
else
    echo "No Web Server PID file found"
fi

echo ""
echo "============================================================================"
echo "Services Stopped"
echo "============================================================================"
echo ""
