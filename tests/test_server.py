#!/usr/bin/env python3
"""Test server to verify Flask app is working"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.beverly_comprehensive_erp import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING FLASK APP ROUTES")
    print("="*60)
    
    # List all registered routes
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        print(f"  {rule.rule:50} [{methods}]")
    
    print("\n" + "="*60)
    print("Starting test server on http://localhost:5007")
    print("Try accessing: http://localhost:5007/hello")
    print("="*60 + "\n")
    
    # Run with debug to see any errors
    app.run(debug=True, port=5007, host='127.0.0.1')