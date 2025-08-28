#!/usr/bin/env python3
"""
Simple startup script for Beverly Knits ERP
Ensures it runs on port 5006
"""

import os
import sys

# Set environment variable for port
os.environ['APP_PORT'] = '5006'
os.environ['APP_HOST'] = '0.0.0.0'
os.environ['APP_DEBUG'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting Beverly Knits ERP on port 5006...")
print("-" * 60)

# Import and run the app
from src.core.beverly_comprehensive_erp import app

print("Server starting on http://0.0.0.0:5006")
print("Access the application at:")
print("  - http://localhost:5006")
print("  - http://127.0.0.1:5006")
print("-" * 60)

# Run the Flask app
app.run(host='0.0.0.0', port=5006, debug=False)