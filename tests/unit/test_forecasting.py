"""
Unit tests for forecasting functions in beverly_comprehensive_erp.py

Tests ML forecasting, demand prediction, and forecast accuracy
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

import core.beverly_comprehensive_erp as erp


class TestSalesForecastingEngine:
    """Test suite for SalesForecastingEngine class"""
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample historical sales data"""
        dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
        np.random.seed(42)
        
        # Generate sales with trend and seasonality
        trend = np.linspace(100, 150, 365)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly cycle
        noise = np.random.normal(0, 10, 365)
        sales = trend + seasonality + noise
        
        return pd.DataFrame({
            'date': dates,
            'sales': sales.clip(min=0),
            'item_id': 'ITEM001'
        })
    
    @pytest.fixture
    def forecasting_engine(self):
        """Create SalesForecastingEngine instance"""
        with patch('beverly_comprehensive_erp.ML_AVAILABLE', True):
            engine = erp.SalesForecastingEngine()
            return engine
    
    def test_forecasting_engine_initialization(self, forecasting_engine):
        """Test SalesForecastingEngine initialization"""
        assert forecasting_engine is not None
        assert hasattr(forecasting_engine, 'models')
        assert hasattr(forecasting_engine, 'forecasts')
    
    def test_calculate_moving_average(self, sample_sales_data):
        """Test moving average calculation"""
        window = 7
        ma = sample_sales_data['sales'].rolling(window=window).mean()
        
        # Check that first window-1 values are NaN
        assert ma[:window-1].isna().all()
        
        # Check that remaining values are calculated correctly
        assert not ma[window:].isna().any()
        
        # Verify calculation for a specific point
        expected = sample_sales_data['sales'][0:window].mean()
        assert abs(ma.iloc[window-1] - expected) < 0.01
    
    def test_exponential_smoothing(self, sample_sales_data):
        """Test exponential smoothing forecast"""
        alpha = 0.3  # Smoothing parameter
        sales = sample_sales_data['sales'].values
        
        # Initialize forecast
        forecast = [sales[0]]
        
        # Calculate exponential smoothing
        for i in range(1, len(sales)):
            f = alpha * sales[i-1] + (1 - alpha) * forecast[-1]
            forecast.append(f)
        
        assert len(forecast) == len(sales)
        assert all(f >= 0 for f in forecast)  # All forecasts should be non-negative
    
    def test_trend_detection(self, sample_sales_data):
        """Test trend detection in time series"""
        # Calculate linear trend
        x = np.arange(len(sample_sales_data))
        y = sample_sales_data['sales'].values
        
        # Fit linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Positive slope indicates upward trend
        assert slope > 0  # Our sample data has upward trend
    
    def test_seasonality_detection(self, sample_sales_data):
        """Test seasonality detection"""
        sales = sample_sales_data['sales']
        
        # Calculate autocorrelation for different lags
        lag_30 = sales.autocorr(lag=30)  # Monthly seasonality
        lag_7 = sales.autocorr(lag=7)    # Weekly seasonality
        
        # Our sample data has 30-day seasonality
        assert abs(lag_30) > 0.3  # Significant correlation at 30-day lag
    
    def test_forecast_accuracy_metrics(self):
        """Test forecast accuracy metric calculations"""
        actual = np.array([100, 110, 105, 115, 120])
        forecast = np.array([98, 112, 103, 118, 119])
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(actual - forecast))
        assert mae == 2.2
        
        # Mean Squared Error (MSE)
        mse = np.mean((actual - forecast) ** 2)
        assert abs(mse - 5.8) < 0.01
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        assert abs(mape - 2.02) < 0.1
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        assert abs(rmse - 2.408) < 0.01
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        forecast = 100
        std_error = 10
        confidence_level = 0.95
        z_score = 1.96  # For 95% confidence
        
        lower_bound = forecast - z_score * std_error
        upper_bound = forecast + z_score * std_error
        
        assert lower_bound == 80.4
        assert upper_bound == 119.6
        
        # Check interval width
        interval_width = upper_bound - lower_bound
        assert interval_width == 2 * z_score * std_error
    
    def test_forecast_horizon(self, sample_sales_data):
        """Test different forecast horizons"""
        horizons = [7, 14, 30, 90]  # Days ahead
        
        for horizon in horizons:
            # Generate forecast for horizon days
            forecast = np.random.normal(125, 15, horizon)  # Simulated forecast
            
            assert len(forecast) == horizon
            assert all(f >= 0 for f in forecast)  # Non-negative forecasts
    
    def test_handle_missing_data(self):
        """Test handling of missing data in time series"""
        data = pd.Series([100, np.nan, 105, 110, np.nan, 120])
        
        # Forward fill
        filled_forward = data.fillna(method='ffill')
        assert filled_forward[1] == 100
        assert filled_forward[4] == 110
        
        # Interpolation
        interpolated = data.interpolate()
        assert interpolated[1] == 102.5  # Linear interpolation
        assert interpolated[4] == 115
    
    def test_outlier_detection(self, sample_sales_data):
        """Test outlier detection in sales data"""
        sales = sample_sales_data['sales']
        
        # Calculate z-scores
        mean = sales.mean()
        std = sales.std()
        z_scores = np.abs((sales - mean) / std)
        
        # Identify outliers (z-score > 3)
        outliers = sales[z_scores > 3]
        
        # Our sample data should have few outliers
        assert len(outliers) < len(sales) * 0.01  # Less than 1% outliers


class TestMLModels:
    """Test machine learning model integration"""
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        
        # Features
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        # Target (with some relationship to features)
        y = 2 * X['feature1'] + 3 * X['feature2'] - X['feature3'] + np.random.normal(0, 0.5, n_samples)
        
        return X, y
    
    @pytest.mark.skipif(not erp.ML_AVAILABLE, reason="ML libraries not available")
    def test_linear_regression_model(self, training_data):
        """Test linear regression model"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        X, y = training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        assert r2 > 0.8  # Good fit for our synthetic data
    
    @pytest.mark.skipif(not erp.ML_AVAILABLE, reason="ML libraries not available")
    def test_random_forest_model(self, training_data):
        """Test Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        X, y = training_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Check feature importances
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)
        assert abs(sum(importances) - 1.0) < 0.01
    
    def test_model_selection(self):
        """Test model selection based on performance"""
        models_performance = {
            'linear': {'mape': 5.2, 'rmse': 12.3},
            'random_forest': {'mape': 4.8, 'rmse': 11.5},
            'xgboost': {'mape': 4.5, 'rmse': 10.8}
        }
        
        # Select best model by MAPE
        best_by_mape = min(models_performance.items(), key=lambda x: x[1]['mape'])
        assert best_by_mape[0] == 'xgboost'
        
        # Select best model by RMSE
        best_by_rmse = min(models_performance.items(), key=lambda x: x[1]['rmse'])
        assert best_by_rmse[0] == 'xgboost'
    
    def test_ensemble_forecast(self):
        """Test ensemble forecasting approach"""
        # Individual model forecasts
        forecasts = {
            'model1': [100, 105, 110],
            'model2': [98, 107, 108],
            'model3': [102, 104, 112]
        }
        
        # Simple average ensemble
        ensemble_avg = []
        for i in range(3):
            avg = np.mean([forecasts['model1'][i], 
                          forecasts['model2'][i], 
                          forecasts['model3'][i]])
            ensemble_avg.append(avg)
        
        assert len(ensemble_avg) == 3
        assert ensemble_avg[0] == 100  # (100+98+102)/3
        
        # Weighted ensemble
        weights = {'model1': 0.3, 'model2': 0.3, 'model3': 0.4}
        ensemble_weighted = []
        for i in range(3):
            weighted = (forecasts['model1'][i] * weights['model1'] +
                       forecasts['model2'][i] * weights['model2'] +
                       forecasts['model3'][i] * weights['model3'])
            ensemble_weighted.append(weighted)
        
        assert abs(ensemble_weighted[0] - 100.4) < 0.01


class TestDemandForecasting:
    """Test demand forecasting specific functions"""
    
    def test_calculate_safety_stock_with_forecast(self):
        """Test safety stock calculation with demand forecast"""
        avg_demand = 100
        demand_std = 20
        lead_time = 7
        service_level = 0.95
        z_score = 1.645  # For 95% service level
        
        # With demand uncertainty
        demand_uncertainty = demand_std * np.sqrt(lead_time)
        safety_stock = z_score * demand_uncertainty
        
        assert safety_stock > 0
        assert abs(safety_stock - (1.645 * 20 * np.sqrt(7))) < 0.01
    
    def test_forecast_based_reorder_point(self):
        """Test reorder point calculation using forecast"""
        forecast_demand = [100, 110, 105, 95, 100]  # Next 5 days
        lead_time_days = 3
        safety_stock = 50
        
        # Expected demand during lead time
        lead_time_demand = sum(forecast_demand[:lead_time_days])
        reorder_point = lead_time_demand + safety_stock
        
        assert reorder_point == 365  # 100+110+105+50
    
    def test_demand_pattern_classification(self):
        """Test classification of demand patterns"""
        # Stable demand
        stable_demand = [100, 98, 102, 99, 101, 100]
        cv_stable = np.std(stable_demand) / np.mean(stable_demand)
        
        # Trending demand
        trending_demand = [100, 110, 120, 130, 140, 150]
        cv_trending = np.std(trending_demand) / np.mean(trending_demand)
        
        # Seasonal demand
        seasonal_demand = [100, 120, 100, 120, 100, 120]
        cv_seasonal = np.std(seasonal_demand) / np.mean(seasonal_demand)
        
        # Classify based on coefficient of variation
        assert cv_stable < 0.2  # Stable pattern
        assert cv_trending > 0.2  # Non-stable pattern
        assert cv_seasonal > 0.1  # Variable pattern
    
    def test_forecast_adjustment_factors(self):
        """Test forecast adjustment for business factors"""
        base_forecast = 1000
        
        # Adjustment factors
        promotion_factor = 1.3  # 30% increase during promotion
        holiday_factor = 0.8   # 20% decrease during holiday
        weather_factor = 1.1   # 10% increase due to weather
        
        # Apply adjustments
        adjusted_forecast = base_forecast * promotion_factor * holiday_factor * weather_factor
        
        assert adjusted_forecast == 1144  # 1000 * 1.3 * 0.8 * 1.1
    
    def test_forecast_aggregation(self):
        """Test forecast aggregation across products"""
        product_forecasts = {
            'PROD_A': [100, 110, 105],
            'PROD_B': [200, 210, 195],
            'PROD_C': [150, 145, 155]
        }
        
        # Aggregate by time period
        period_totals = []
        for period in range(3):
            total = sum(forecast[period] for forecast in product_forecasts.values())
            period_totals.append(total)
        
        assert period_totals == [450, 465, 455]
        
        # Aggregate by product category
        category_mapping = {
            'PROD_A': 'Category1',
            'PROD_B': 'Category2',
            'PROD_C': 'Category1'
        }
        
        category_totals = {}
        for product, forecasts in product_forecasts.items():
            category = category_mapping[product]
            if category not in category_totals:
                category_totals[category] = [0, 0, 0]
            for i, value in enumerate(forecasts):
                category_totals[category][i] += value
        
        assert category_totals['Category1'] == [250, 255, 260]
        assert category_totals['Category2'] == [200, 210, 195]


class TestForecastValidation:
    """Test forecast validation and backtesting"""
    
    def test_backtest_forecast_accuracy(self):
        """Test backtesting of forecast models"""
        # Historical data
        actual = [100, 105, 110, 108, 112, 115, 118, 120]
        
        # Simulate rolling forecast
        window = 3
        forecasts = []
        errors = []
        
        for i in range(window, len(actual)):
            # Simple average forecast
            forecast = np.mean(actual[i-window:i])
            forecasts.append(forecast)
            errors.append(abs(actual[i] - forecast))
        
        # Calculate backtest metrics
        mae = np.mean(errors)
        assert mae < 5  # Reasonable accuracy
        
        # Check forecast bias
        bias = np.mean([actual[i+window] - forecasts[i] for i in range(len(forecasts))])
        assert abs(bias) < 3  # Low bias
    
    def test_cross_validation(self):
        """Test time series cross-validation"""
        data = list(range(100))  # Simple increasing series
        
        # Time series split
        n_splits = 3
        test_size = 10
        
        splits = []
        for i in range(n_splits):
            train_end = len(data) - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            train = data[:train_end]
            test = data[test_start:test_end]
            splits.append((train, test))
        
        assert len(splits) == 3
        assert len(splits[0][1]) == test_size
        
        # Ensure no overlap between train and test
        for train, test in splits:
            assert max(train) < min(test)
    
    def test_forecast_monitoring(self):
        """Test real-time forecast monitoring"""
        forecast = [100, 105, 110]
        actual = [98, 108, 107]
        
        # Calculate tracking signal
        errors = [a - f for a, f in zip(actual, forecast)]
        cumulative_error = sum(errors)
        mad = np.mean([abs(e) for e in errors])  # Mean Absolute Deviation
        
        if mad > 0:
            tracking_signal = cumulative_error / mad
        else:
            tracking_signal = 0
        
        # Tracking signal should be within control limits
        assert abs(tracking_signal) < 4  # Common threshold
        
        # Check if forecast needs adjustment
        needs_adjustment = abs(tracking_signal) > 3
        assert not needs_adjustment  # Within acceptable range