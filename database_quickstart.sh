#!/bin/bash

# Database Quick Start Script
# Purpose: Initialize and start the eFab database integration

echo "========================================="
echo "eFab Database Integration - Quick Start"
echo "========================================="

# Check for required environment variables
if [ -z "$EFAB_SESSION" ]; then
    echo "ERROR: EFAB_SESSION environment variable not set!"
    echo ""
    echo "To get your session cookie:"
    echo "1. Log into eFab in your browser"
    echo "2. Open Developer Tools (F12)"
    echo "3. Go to Application/Storage > Cookies"
    echo "4. Copy the 'dancer.session' cookie value"
    echo "5. Run: export EFAB_SESSION='your_cookie_value'"
    exit 1
fi

# Set default database URL if not provided
if [ -z "$DATABASE_URL" ]; then
    echo "Using default PostgreSQL database..."
    export DATABASE_URL="postgresql://postgres:password@localhost/efab_erp"
fi

echo ""
echo "Configuration:"
echo "  Database: $DATABASE_URL"
echo "  Session: ${EFAB_SESSION:0:20}..."
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
pip show sqlalchemy > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing database dependencies..."
    pip install sqlalchemy psycopg2-binary schedule
fi

# Initialize database
echo ""
echo "Initializing database..."
python3 src/database/setup.py --init

# Run initial sync
echo ""
echo "Running initial data sync from eFab API..."
python3 src/database/setup.py --sync

# Show status
echo ""
python3 src/database/setup.py --status

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start the scheduler (syncs every 2 hours):"
echo "  python3 src/database/setup.py --schedule"
echo ""
echo "To integrate with your Flask app, add:"
echo "  from database.api import register_database_api"
echo "  register_database_api(app)"
echo ""
echo "API endpoints available at:"
echo "  GET  /api/db/status"
echo "  POST /api/db/sync/manual"
echo "  GET  /api/db/cf-versions"
echo "  GET  /api/db/production-orders"
echo "  GET  /api/db/yarn-inventory"
echo ""