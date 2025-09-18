#!/bin/bash
# Beverly Knits ERP Server Startup Script

# Set environment variables
export EFAB_SESSION="aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B"
export ENABLE_YARN_SCHEDULER=true
export FILTER_NONPRODUCTION_YARNS=true

# Kill any existing processes on port 5006
echo "Checking for existing processes on port 5006..."
lsof -i :5006 2>/dev/null | grep LISTEN | awk '{print $2}' | xargs -r kill -9 2>/dev/null
pkill -f "python.*beverly" 2>/dev/null
sleep 1

# Start the server
echo "Starting Beverly Knits ERP Server..."
echo "Server will be available at: http://localhost:5006"
echo "Dashboard: http://localhost:5006/consolidated"
echo ""

# Use python3 from WSL
/usr/bin/python3 src/core/beverly_comprehensive_erp.py