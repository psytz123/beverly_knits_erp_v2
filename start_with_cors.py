#!/usr/bin/env python3
"""
Start Beverly Knits ERP with CORS enabled for ngrok/external access
"""

import os
import sys

# Add paths
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src')
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src/core')

# Change to project directory
os.chdir('/mnt/c/finalee/beverly_knits_erp_v2')

print("="*60)
print("🚀 STARTING BEVERLY KNITS ERP WITH CORS ENABLED")
print("="*60)

# Import the app
from src.core.beverly_comprehensive_erp import app

# Enable CORS for all origins (for testing with ngrok)
try:
    from flask_cors import CORS
    CORS(app, origins="*", supports_credentials=True)
    print("✅ CORS enabled for all origins")
except ImportError:
    print("⚠️ flask-cors not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-cors"])
    from flask_cors import CORS
    CORS(app, origins="*", supports_credentials=True)
    print("✅ CORS installed and enabled")

# Add additional headers for ngrok
@app.after_request
def after_request(response):
    # Allow all origins for testing
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    # Security headers that work with ngrok
    response.headers['X-Frame-Options'] = 'ALLOWALL'  # Allow embedding in ngrok preview
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response

print("\n📊 Dashboard will be available at:")
print("  - Local: http://localhost:5006/consolidated")
print("  - Ngrok: https://[your-url].ngrok.app/consolidated")
print("\n🌐 To share via ngrok:")
print("  1. Open new terminal")
print("  2. Run: ngrok http 5006")
print("  3. Share the HTTPS URL")
print("\n✅ APIs configured for external access")
print("="*60 + "\n")

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=False)