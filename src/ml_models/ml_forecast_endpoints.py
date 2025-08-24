#!/usr/bin/env python3
"""
ML Forecast Endpoints for Beverly Knits ERP
Add these endpoints to beverly_comprehensive_erp.py
"""

# Add these imports at the top of beverly_comprehensive_erp.py:
"""
# ML Forecast Integration
try:
    from ml_forecast_integration import (
        MLForecastIntegration,
        get_ml_forecast,
        get_demand_forecast_30day,
        get_forecast_by_date
    )
    ML_FORECAST_AVAILABLE = True
    print("ML Forecast Integration loaded successfully")
except ImportError as e:
    ML_FORECAST_AVAILABLE = False
    print(f"ML Forecast Integration not available: {e}")
"""

# Initialize ML Forecast Integration (add after other initializations):
"""
# Initialize ML Forecast Integration
ml_forecast = None
if ML_FORECAST_AVAILABLE:
    try:
        ml_forecast = MLForecastIntegration(DATA_PATH / "prompts" / "5")
        print("ML Forecast Integration initialized")
    except Exception as e:
        print(f"Could not initialize ML Forecast Integration: {e}")
        ML_FORECAST_AVAILABLE = False
"""

# Add these endpoints before if __name__ == "__main__":

ML_FORECAST_ENDPOINTS = '''
# ========== ML FORECAST ENDPOINTS ==========

@app.route("/api/ml-forecast")
def get_ml_forecast_endpoint():
    """Get ML forecast using integrated forecasting system"""
    if not ML_FORECAST_AVAILABLE or not ml_forecast:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Get forecast (uses cache if recent)
        forecast = ml_forecast.train_forecast_model()
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/30day")
def get_ml_forecast_30day():
    """Get 30-day demand forecast"""
    if not ML_FORECAST_AVAILABLE:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Use simple interface
        forecast = get_demand_forecast_30day()
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'forecast_30day': forecast['forecast_30day'],
            'daily_average': forecast['daily_average'],
            'accuracy': forecast['accuracy'],
            'daily_forecasts': forecast['daily_forecasts'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/date/<date_str>")
def get_ml_forecast_by_date(date_str):
    """Get forecast for specific date"""
    if not ML_FORECAST_AVAILABLE:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Parse date
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get forecast
        forecast = get_forecast_by_date(target_date)
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'date': date_str,
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/range", methods=['POST'])
def get_ml_forecast_range():
    """Get forecast for date range"""
    if not ML_FORECAST_AVAILABLE or not ml_forecast:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        data = request.get_json()
        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
        
        # Get forecast for range
        forecast = ml_forecast.get_forecast_for_date_range(start_date, end_date)
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/product/<product_id>")
def get_ml_forecast_by_product(product_id):
    """Get demand forecast for specific product"""
    if not ML_FORECAST_AVAILABLE or not ml_forecast:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Get product forecast
        forecast = ml_forecast.get_product_demand_forecast(product_id)
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'product_id': product_id,
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/train", methods=['POST'])
def trigger_ml_forecast_training():
    """Manually trigger ML model retraining"""
    if not ML_FORECAST_AVAILABLE or not ml_forecast:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Force retrain
        forecast = ml_forecast.train_forecast_model(force_retrain=True)
        
        if 'error' in forecast:
            return jsonify(forecast), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'best_model': forecast.get('best_model'),
            'accuracy': f"{forecast.get('best_accuracy', 0):.1f}%",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/recommendations")
def get_ml_inventory_recommendations():
    """Get inventory recommendations based on ML forecast"""
    if not ML_FORECAST_AVAILABLE or not ml_forecast:
        return jsonify({"error": "ML forecast integration not available"}), 503
    
    try:
        # Get current inventory (from analyzer if available)
        if hasattr(analyzer, 'yarn_inventory_data') and analyzer.yarn_inventory_data is not None:
            current_inventory = analyzer.yarn_inventory_data
        else:
            # Use empty DataFrame if no inventory data
            current_inventory = pd.DataFrame()
        
        # Get recommendations
        recommendations = ml_forecast.get_inventory_recommendations(current_inventory)
        
        if 'error' in recommendations:
            return jsonify(recommendations), 400
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ml-forecast/status")
def get_ml_forecast_status():
    """Get ML forecast system status"""
    status = {
        'ml_forecast_available': ML_FORECAST_AVAILABLE,
        'ml_engines': {
            'full_engine': ML_AVAILABLE,
            'minimal_engine': True,  # Always available with basic sklearn
            'prophet': ML_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
            'statsmodels': STATSMODELS_AVAILABLE
        },
        'last_training': None,
        'cached_forecast': False,
        'data_source': str(DATA_PATH / "prompts" / "5") if ML_FORECAST_AVAILABLE else None
    }
    
    if ML_FORECAST_AVAILABLE and ml_forecast:
        status['last_training'] = ml_forecast.last_training_date.isoformat() if ml_forecast.last_training_date else None
        status['cached_forecast'] = ml_forecast.cached_forecast is not None
        status['engine_type'] = ml_forecast.engine_type
    
    return jsonify(status)

# ========== END ML FORECAST ENDPOINTS ==========
'''

print("ML Forecast Endpoints code generated.")
print("\nTo integrate, add the following to beverly_comprehensive_erp.py:")
print("1. Import statements (at top)")
print("2. Initialize ml_forecast object (after other initializations)")
print("3. Add the endpoints (before if __name__ == '__main__')")
print("\nEndpoints provided:")
print("- GET /api/ml-forecast - Get full 90-day forecast")
print("- GET /api/ml-forecast/30day - Get 30-day forecast summary")
print("- GET /api/ml-forecast/date/<date> - Get forecast for specific date")
print("- POST /api/ml-forecast/range - Get forecast for date range")
print("- GET /api/ml-forecast/product/<id> - Get product-specific forecast")
print("- POST /api/ml-forecast/train - Trigger model retraining")
print("- GET /api/ml-forecast/recommendations - Get inventory recommendations")
print("- GET /api/ml-forecast/status - Check system status")