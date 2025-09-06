#!/usr/bin/env python3
"""
Mock eFab API Server for Testing
Simulates eFab API responses for local development
"""

from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

app = Flask(__name__)

# Mock session cookie
VALID_SESSION = "aLfHTrRrtWWy4FPgLnxdEPC7ohA37dlR"

def check_auth():
    """Check if request has valid session cookie"""
    session_cookie = request.cookies.get('dancer.session')
    if session_cookie != VALID_SESSION:
        return False
    return True

@app.route('/api/sales-order/plan/list', methods=['GET'])
def get_sales_orders():
    """Mock sales order plan list endpoint"""
    if not check_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Generate mock sales order data
    num_orders = 50
    data = []
    
    for i in range(num_orders):
        order = {
            'order_id': f'SO-2025-{1000 + i}',
            'customer': f'Customer {random.choice(["ABC", "XYZ", "DEF", "GHI"])}',
            'style': f'STYLE-{random.randint(100, 999)}',
            'quantity': random.randint(100, 5000),
            'unit': 'lbs',
            'delivery_date': (datetime.now() + timedelta(days=random.randint(7, 90))).isoformat(),
            'status': random.choice(['planned', 'in_progress', 'completed', 'on_hold']),
            'priority': random.choice(['high', 'medium', 'low']),
            'fabric_type': random.choice(['Jersey', 'Interlock', 'Rib', 'French Terry']),
            'color': random.choice(['White', 'Black', 'Navy', 'Grey', 'Custom']),
            'price_per_lb': round(random.uniform(2.5, 8.5), 2)
        }
        data.append(order)
    
    return jsonify(data)

@app.route('/api/knit-orders', methods=['GET'])
@app.route('/api/production/knit-orders', methods=['GET'])
def get_knit_orders():
    """Mock knit orders endpoint"""
    if not check_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Generate mock knit order data
    num_orders = 194  # Match the number from the real data
    data = []
    
    machines = list(range(100, 300, 5))  # Machine IDs
    work_centers = ['9.38.20.F', '8.32.18.M', '7.30.16.C', '9.36.22.V']
    
    for i in range(num_orders):
        # 154 assigned, 40 unassigned (as per real data)
        is_assigned = i < 154
        
        order = {
            'order_id': f'KO-2025-{2000 + i}',
            'style': f'STYLE-{random.randint(100, 999)}',
            'quantity': random.randint(500, 10000),
            'unit': 'lbs',
            'machine': random.choice(machines) if is_assigned else None,
            'work_center': random.choice(work_centers) if is_assigned else None,
            'start_date': datetime.now().isoformat() if is_assigned else None,
            'end_date': (datetime.now() + timedelta(days=random.randint(1, 14))).isoformat() if is_assigned else None,
            'status': 'assigned' if is_assigned else 'pending',
            'completion_percentage': random.randint(0, 100) if is_assigned else 0,
            'operator': f'Operator-{random.randint(1, 20)}' if is_assigned else None
        }
        data.append(order)
    
    return jsonify({'data': data})

@app.route('/api/inventory/<warehouse>', methods=['GET'])
def get_inventory(warehouse):
    """Mock inventory endpoint for different warehouses"""
    if not check_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    valid_warehouses = ['F01', 'G00', 'G02', 'I01']
    if warehouse not in valid_warehouses:
        return jsonify({'error': f'Invalid warehouse. Must be one of {valid_warehouses}'}), 400
    
    # Generate mock inventory data
    num_items = random.randint(100, 300)
    data = []
    
    yarn_types = ['Cotton', 'Polyester', 'Blend', 'Wool', 'Nylon', 'Spandex']
    
    for i in range(num_items):
        item = {
            'item_code': f'YARN-{random.randint(1000, 9999)}',
            'description': f'{random.choice(yarn_types)} Yarn {random.randint(10, 50)}/1',
            'warehouse': warehouse,
            'quantity': random.randint(0, 50000),
            'unit': 'lbs',
            'location': f'{warehouse}-{random.choice(["A", "B", "C"])}{random.randint(1, 99)}',
            'last_updated': datetime.now().isoformat(),
            'min_stock': random.randint(100, 1000),
            'max_stock': random.randint(5000, 20000),
            'reorder_point': random.randint(500, 2000)
        }
        data.append(item)
    
    return jsonify(data)

@app.route('/login', methods=['POST'])
def login():
    """Mock login endpoint"""
    username = request.form.get('username') or request.json.get('username')
    password = request.form.get('password') or request.json.get('password')
    
    # Check credentials
    if username == 'psytz' and password == 'big$cat':
        response = make_response(jsonify({'success': True, 'message': 'Login successful'}))
        response.set_cookie('dancer.session', VALID_SESSION, 
                          domain='localhost',
                          path='/',
                          max_age=86400)  # 24 hours
        return response
    else:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'endpoints': [
            '/api/sales-order/plan/list',
            '/api/knit-orders',
            '/api/inventory/{warehouse}',
            '/login',
            '/api/health'
        ]
    })

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════╗
    ║   Mock eFab API Server                ║
    ║   Running on http://localhost:5007    ║
    ╚════════════════════════════════════════╝
    
    Test credentials:
    - Username: psytz
    - Password: big$cat
    - Session: aLfHTrRrtWWy4FPgLnxdEPC7ohA37dlR
    
    Available endpoints:
    - GET  /api/health
    - POST /login
    - GET  /api/sales-order/plan/list
    - GET  /api/knit-orders
    - GET  /api/inventory/{F01|G00|G02|I01}
    """)
    
    app.run(host='0.0.0.0', port=5007, debug=True)