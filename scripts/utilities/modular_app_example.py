#!/usr/bin/env python3
"""
Modular Flask Application Example
Shows how to use existing modules to replace the monolithic beverly_comprehensive_erp.py
This is a working example that can be run alongside the existing application
"""
import sys
import os
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src')

from flask import Flask, jsonify, request
from services.service_manager import ServiceManager
from data_loaders.unified_data_loader import ConsolidatedDataLoader
from api.consolidated_endpoints import register_consolidated_endpoints
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5'
PORT = 5007  # Different port to run alongside existing app

# Create Flask app
app = Flask(__name__)

# Initialize global components
service_manager = None
data_loader = None


def initialize_components():
    """Initialize all modular components"""
    global service_manager, data_loader
    
    logger.info("Initializing modular components...")
    
    # Initialize service manager with all services
    service_manager = ServiceManager({
        'data_path': DATA_PATH,
        'safety_stock_multiplier': 1.5,
        'lead_time_days': 30,
        'forecast_horizon': 90,
        'target_accuracy': 85.0
    })
    logger.info(f"✓ ServiceManager initialized with {len(service_manager.services)} services")
    
    # Initialize data loader
    data_loader = ConsolidatedDataLoader(
        data_path=DATA_PATH,
        max_workers=5
    )
    logger.info("✓ ConsolidatedDataLoader initialized")
    
    # Note: register_consolidated_endpoints expects an analyzer instance
    # We'll create individual endpoints instead
    logger.info("✓ Components initialized successfully")


# ============= USING EXISTING SERVICES =============

@app.route('/api/modular/inventory-analysis')
def modular_inventory_analysis():
    """Inventory analysis using existing service modules"""
    try:
        # Get the inventory analyzer service
        analyzer = service_manager.get_service('inventory')
        
        # Load data using consolidated loader
        inventory_data = data_loader.load_yarn_inventory()
        
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Perform analysis using the service
        analysis = analyzer.analyze_inventory(inventory_data)
        
        return jsonify({
            'source': 'modular_service',
            'service': 'InventoryAnalyzerService',
            'data': analysis
        })
    
    except Exception as e:
        logger.error(f"Error in modular inventory analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/sales-forecast')
def modular_sales_forecast():
    """Sales forecasting using existing service modules"""
    try:
        # Get the forecasting service
        forecasting = service_manager.get_service('forecasting')
        
        # Load sales data
        sales_data = data_loader.load_sales_orders()
        
        if sales_data is None:
            return jsonify({'error': 'No sales data available'}), 404
        
        # Generate forecast using the service
        forecast = forecasting.generate_forecast(sales_data)
        
        return jsonify({
            'source': 'modular_service',
            'service': 'SalesForecastingService',
            'data': forecast
        })
    
    except Exception as e:
        logger.error(f"Error in modular sales forecast: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/capacity-planning')
def modular_capacity_planning():
    """Capacity planning using existing service modules"""
    try:
        # Get the capacity planning service
        capacity = service_manager.get_service('capacity')
        
        # Load production data
        knit_orders = data_loader.load_knit_orders()
        
        if knit_orders is None:
            return jsonify({'error': 'No production data available'}), 404
        
        # Calculate capacity using the service
        capacity_analysis = capacity.calculate_capacity_utilization(knit_orders)
        
        return jsonify({
            'source': 'modular_service',
            'service': 'CapacityPlanningService',
            'data': capacity_analysis
        })
    
    except Exception as e:
        logger.error(f"Error in modular capacity planning: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/pipeline-analysis')
def modular_pipeline_analysis():
    """Complete pipeline analysis using existing service modules"""
    try:
        # Get the pipeline service
        pipeline = service_manager.get_service('pipeline')
        
        # Load all required data
        sales_data = data_loader.load_sales_orders()
        inventory_data = data_loader.load_yarn_inventory()
        
        # Run complete pipeline analysis
        results = pipeline.run_complete_analysis(
            sales_data=sales_data,
            inventory_data=inventory_data,
            yarn_data=inventory_data
        )
        
        return jsonify({
            'source': 'modular_service',
            'service': 'InventoryManagementPipelineService',
            'data': results
        })
    
    except Exception as e:
        logger.error(f"Error in modular pipeline analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/data-status')
def modular_data_status():
    """Check data loading status using consolidated loader"""
    try:
        # Load all data to check availability
        all_data = data_loader.load_all_data()
        
        status = {
            'loader': 'ConsolidatedDataLoader',
            'data_path': DATA_PATH,
            'parallel_workers': data_loader.max_workers,
            'available_data': {}
        }
        
        for key, df in all_data.items():
            if df is not None:
                status['available_data'][key] = {
                    'records': len(df),
                    'columns': list(df.columns) if hasattr(df, 'columns') else []
                }
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error checking data status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/services-info')
def modular_services_info():
    """Get information about available services"""
    try:
        services_info = {
            'total_services': len(service_manager.services),
            'available_services': list(service_manager.services.keys()),
            'service_details': {}
        }
        
        for name, service in service_manager.services.items():
            services_info['service_details'][name] = {
                'class': type(service).__name__,
                'methods': [m for m in dir(service) if not m.startswith('_') and callable(getattr(service, m))]
            }
        
        return jsonify(services_info)
    
    except Exception as e:
        logger.error(f"Error getting services info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modular/test')
def modular_test():
    """Test endpoint to verify modular app is working"""
    return jsonify({
        'status': 'ok',
        'message': 'Modular application is running!',
        'port': PORT,
        'services_available': len(service_manager.services) if service_manager else 0,
        'data_loader': 'ConsolidatedDataLoader' if data_loader else None
    })


@app.route('/')
def index():
    """Home page showing available endpoints"""
    endpoints = [
        '/api/modular/test - Test endpoint',
        '/api/modular/services-info - List available services',
        '/api/modular/data-status - Check data availability',
        '/api/modular/inventory-analysis - Inventory analysis',
        '/api/modular/sales-forecast - Sales forecasting',
        '/api/modular/capacity-planning - Capacity planning',
        '/api/modular/pipeline-analysis - Complete pipeline analysis'
    ]
    
    html = """
    <html>
    <head>
        <title>Modular ERP Application</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            .endpoint { 
                background: #ecf0f1; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px;
                font-family: 'Courier New', monospace;
            }
            .status { 
                background: #27ae60; 
                color: white; 
                padding: 5px 10px; 
                border-radius: 3px; 
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>Beverly Knits ERP - Modular Version</h1>
        <p>This is a modular implementation using existing service modules.</p>
        <p class="status">Running on port """ + str(PORT) + """</p>
        
        <h2>Available Endpoints:</h2>
    """
    
    for endpoint in endpoints:
        html += f'<div class="endpoint">{endpoint}</div>'
    
    html += """
        <h2>Architecture:</h2>
        <ul>
            <li>ServiceManager - Orchestrates all business services</li>
            <li>ConsolidatedDataLoader - Handles all data loading with caching</li>
            <li>Modular Services - Inventory, Forecasting, Capacity, Pipeline</li>
        </ul>
        
        <p><em>This demonstrates how the monolithic app can be modularized using existing components.</em></p>
    </body>
    </html>
    """
    
    return html


def main():
    """Main entry point"""
    print("="*70)
    print(" MODULAR ERP APPLICATION ")
    print(" Using Existing Service Modules ")
    print("="*70)
    
    # Initialize components
    initialize_components()
    
    # Start Flask app
    print(f"\nStarting modular application on port {PORT}...")
    print(f"Visit: http://localhost:{PORT}")
    print("\nThis runs alongside the existing app on port 5006")
    print("All endpoints are prefixed with /api/modular/ to avoid conflicts")
    print("\nPress Ctrl+C to stop")
    print("="*70)
    
    app.run(host='0.0.0.0', port=PORT, debug=False)


if __name__ == '__main__':
    main()