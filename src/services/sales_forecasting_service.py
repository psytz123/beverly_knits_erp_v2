#!/usr/bin/env python3
"""
Beverly Knits ERP - Sales Forecasting Service
Extracted from beverly_comprehensive_erp.py (lines 587-1792)
Advanced ML forecasting with multiple models and ensemble predictions
Target: >85% forecast accuracy with 90-day horizon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for sales forecasting"""
    forecast_horizon: int = 90  # 90-day forecast
    target_accuracy: float = 0.85  # 85% accuracy target
    min_history_points: int = 3  # Minimum data points for forecasting
    consistency_threshold_high: float = 0.7  # High consistency threshold
    consistency_threshold_low: float = 0.3  # Low consistency threshold
    ensemble_enabled: bool = True  # Enable ensemble predictions


class SalesForecastingService:
    """
    Advanced Sales Forecasting Service with Multi-Model Approach
    Preserves all original business logic from monolith
    
    Original location: beverly_comprehensive_erp.py lines 587-1792
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize sales forecasting service
        
        Args:
            config: Optional configuration override
        """
        self.config = config or ForecastConfig()
        
        # Initialize model containers
        self.models = {}
        self.feature_extractors = {}
        self.validation_metrics = {}
        self.ensemble_weights = {}
        
        # Configuration from original
        self.forecast_horizon = self.config.forecast_horizon
        self.target_accuracy = self.config.target_accuracy
        self.ML_AVAILABLE = False
        self.ml_engines = {}
        
        # Initialize ML engines
        self.initialize_ml_engines()
        
        logger.info(f"SalesForecastingService initialized")
        logger.info(f"  Forecast horizon: {self.forecast_horizon} days")
        logger.info(f"  Target accuracy: {self.target_accuracy * 100}%")
        logger.info(f"  ML Available: {self.ML_AVAILABLE}")
        logger.info(f"  Available engines: {list(self.ml_engines.keys())}")
    
    def initialize_ml_engines(self):
        """Initialize available ML engines with proper error handling"""
        self.ml_engines = {}
        
        # Try to import RandomForest
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.ml_engines['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ML_AVAILABLE = True
            logger.info("  ✓ RandomForest available")
        except ImportError:
            logger.warning("  ✗ RandomForest not available - sklearn not installed")
        
        # Try to import Prophet
        try:
            from prophet import Prophet
            self.ml_engines['prophet'] = Prophet
            self.ML_AVAILABLE = True
            logger.info("  ✓ Prophet available")
        except ImportError:
            logger.warning("  ✗ Prophet not available")
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.ml_engines['xgboost'] = xgb.XGBRegressor
            self.ML_AVAILABLE = True
            logger.info("  ✓ XGBoost available")
        except ImportError:
            logger.warning("  ✗ XGBoost not available")
        
        # Try to import ARIMA
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.ml_engines['arima'] = ARIMA
            self.ML_AVAILABLE = True
            logger.info("  ✓ ARIMA available")
        except ImportError:
            logger.warning("  ✗ ARIMA not available")
        
        # Try basic sklearn models as fallback
        if not self.ml_engines:
            try:
                from sklearn.linear_model import LinearRegression
                self.ml_engines['linear'] = LinearRegression()
                self.ML_AVAILABLE = True
                logger.info("  ✓ Linear Regression (fallback) available")
            except ImportError:
                logger.error("  ✗ No ML engines available - using fallback forecasting")
                self.ML_AVAILABLE = False
    
    def fallback_forecast(self, historical_data):
        """
        Simple moving average fallback when no ML engines are available
        Preserves original logic from monolith
        """
        if isinstance(historical_data, pd.DataFrame):
            if 'quantity' in historical_data.columns:
                data = historical_data['quantity']
            elif 'sales' in historical_data.columns:
                data = historical_data['sales']
            else:
                data = historical_data.iloc[:, 0]
        else:
            data = historical_data
        
        # Simple moving average (original logic)
        if len(data) >= 3:
            return float(data[-3:].mean())
        elif len(data) > 0:
            return float(data.mean())
        else:
            return 0.0
    
    def calculate_consistency_score(self, style_history):
        """
        Calculate consistency score for a style's historical sales
        Uses Coefficient of Variation (CV) to measure consistency
        
        PRESERVED LOGIC: Original CV calculation and thresholds
        
        Args:
            style_history: DataFrame or Series with historical sales data
            
        Returns:
            dict with consistency_score (0-1), cv value, and recommendation
        """
        # Extract quantity data (original column mapping)
        if isinstance(style_history, pd.DataFrame):
            if 'quantity' in style_history.columns:
                data = style_history['quantity']
            elif 'Yds_ordered' in style_history.columns:
                data = style_history['Yds_ordered']
            elif 'sales' in style_history.columns:
                data = style_history['sales']
            else:
                # Try to find any numeric column
                numeric_cols = style_history.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = style_history[numeric_cols[0]]
                else:
                    return {'consistency_score': 0, 'cv': 1.0, 'recommendation': 'insufficient_data'}
        else:
            data = style_history
        
        # Remove zeros and NaN values
        data = pd.Series(data).dropna()
        data = data[data > 0]
        
        # Need minimum history for consistency calculation
        if len(data) < self.config.min_history_points:
            return {
                'consistency_score': 0,
                'cv': 1.0,
                'recommendation': 'insufficient_history',
                'data_points': len(data)
            }
        
        # Calculate mean and standard deviation
        mean_value = data.mean()
        std_value = data.std()
        
        # Calculate Coefficient of Variation (CV)
        # Lower CV = more consistent (original formula)
        if mean_value > 0:
            cv = std_value / mean_value
        else:
            cv = 1.0
        
        # Convert CV to consistency score (0-1, where 1 is most consistent)
        # Original conversion formula preserved
        consistency_score = max(0, 1 - cv)
        
        # Determine recommendation based on consistency score
        # Original thresholds preserved
        if consistency_score >= self.config.consistency_threshold_high:
            recommendation = 'use_ml_forecast'
        elif consistency_score >= self.config.consistency_threshold_low:
            recommendation = 'use_weighted_forecast'
        else:
            recommendation = 'react_to_orders_only'
        
        return {
            'consistency_score': consistency_score,
            'cv': cv,
            'mean': mean_value,
            'std': std_value,
            'recommendation': recommendation,
            'data_points': len(data)
        }
    
    def forecast_with_consistency(self, style_data, horizon_days=None):
        """
        Generate forecast based on consistency score
        High consistency → Use ML forecast
        Medium consistency → Use weighted average
        Low consistency → React to orders only
        
        PRESERVED LOGIC: Original consistency-based forecasting strategy
        
        Args:
            style_data: Historical data for the style
            horizon_days: Forecast horizon in days
            
        Returns:
            dict with forecast, confidence, and method used
        """
        if horizon_days is None:
            horizon_days = self.forecast_horizon
        
        # Calculate consistency score
        consistency_result = self.calculate_consistency_score(style_data)
        consistency_score = consistency_result['consistency_score']
        
        # Initialize result
        result = {
            'consistency_score': consistency_score,
            'cv': consistency_result['cv'],
            'method': '',
            'forecast': 0,
            'confidence': 0,
            'horizon_days': horizon_days
        }
        
        # High consistency (CV < 0.3, score > 0.7): Use ML forecast
        if consistency_score >= self.config.consistency_threshold_high and self.ML_AVAILABLE:
            try:
                # Use ML model for forecasting
                forecast_value = self.generate_ml_forecast(style_data, horizon_days)
                result['forecast'] = forecast_value
                result['confidence'] = consistency_score * 0.9  # Original confidence formula
                result['method'] = 'ml_forecast'
                
            except Exception as e:
                logger.warning(f"ML forecast failed: {e}, using fallback")
                result['forecast'] = self.fallback_forecast(style_data)
                result['confidence'] = consistency_score * 0.5
                result['method'] = 'fallback_after_ml_failure'
        
        # Medium consistency: Use weighted average
        elif consistency_score >= self.config.consistency_threshold_low:
            result['forecast'] = self.calculate_weighted_average(style_data)
            result['confidence'] = consistency_score * 0.7  # Original confidence formula
            result['method'] = 'weighted_average'
        
        # Low consistency: React to orders only
        else:
            result['forecast'] = self.calculate_reactive_forecast(style_data)
            result['confidence'] = consistency_score * 0.5  # Original confidence formula
            result['method'] = 'react_to_orders'
        
        return result
    
    def generate_ml_forecast(self, style_data, horizon_days):
        """
        Generate ML-based forecast using available engines
        
        Args:
            style_data: Historical data
            horizon_days: Forecast horizon
            
        Returns:
            Forecasted value
        """
        forecasts = []
        
        # Try each available ML engine
        if 'random_forest' in self.ml_engines:
            try:
                forecast = self.forecast_with_random_forest(style_data, horizon_days)
                forecasts.append(forecast)
            except Exception as e:
                logger.debug(f"RandomForest forecast failed: {e}")
        
        if 'prophet' in self.ml_engines:
            try:
                forecast = self.forecast_with_prophet(style_data, horizon_days)
                forecasts.append(forecast)
            except Exception as e:
                logger.debug(f"Prophet forecast failed: {e}")
        
        if 'xgboost' in self.ml_engines:
            try:
                forecast = self.forecast_with_xgboost(style_data, horizon_days)
                forecasts.append(forecast)
            except Exception as e:
                logger.debug(f"XGBoost forecast failed: {e}")
        
        # Return ensemble average if multiple forecasts available
        if len(forecasts) > 1 and self.config.ensemble_enabled:
            return np.mean(forecasts)
        elif len(forecasts) == 1:
            return forecasts[0]
        else:
            # Fallback if all ML methods fail
            return self.fallback_forecast(style_data)
    
    def forecast_with_random_forest(self, style_data, horizon_days):
        """Random Forest forecasting"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare data
        if isinstance(style_data, pd.DataFrame):
            y = style_data.iloc[:, 0].values
        else:
            y = style_data.values if hasattr(style_data, 'values') else style_data
        
        # Create features (time index)
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict future
        future_X = np.array([[len(y) + horizon_days/30]])  # Approximate monthly steps
        forecast = model.predict(future_X)[0]
        
        return forecast
    
    def forecast_with_prophet(self, style_data, horizon_days):
        """Prophet forecasting"""
        from prophet import Prophet
        
        # Prepare data in Prophet format
        if isinstance(style_data, pd.DataFrame):
            df = pd.DataFrame({
                'ds': pd.date_range(end=datetime.now(), periods=len(style_data), freq='D'),
                'y': style_data.iloc[:, 0].values
            })
        else:
            df = pd.DataFrame({
                'ds': pd.date_range(end=datetime.now(), periods=len(style_data), freq='D'),
                'y': style_data
            })
        
        # Train Prophet model
        model = Prophet(daily_seasonality=False, weekly_seasonality=True)
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        
        # Return average of future predictions
        future_forecast = forecast.iloc[-horizon_days:]['yhat'].mean()
        
        return future_forecast
    
    def forecast_with_xgboost(self, style_data, horizon_days):
        """XGBoost forecasting"""
        import xgboost as xgb
        
        # Prepare data
        if isinstance(style_data, pd.DataFrame):
            y = style_data.iloc[:, 0].values
        else:
            y = style_data.values if hasattr(style_data, 'values') else style_data
        
        # Create features
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Train model
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict future
        future_X = np.array([[len(y) + horizon_days/30]])
        forecast = model.predict(future_X)[0]
        
        return forecast
    
    def calculate_weighted_average(self, style_data):
        """
        Calculate weighted average for medium consistency styles
        Recent data gets higher weight
        """
        if isinstance(style_data, pd.DataFrame):
            data = style_data.iloc[:, 0].values
        else:
            data = style_data.values if hasattr(style_data, 'values') else style_data
        
        if len(data) == 0:
            return 0
        
        # Create weights (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(data))
        weights = weights / weights.sum()
        
        # Calculate weighted average
        return np.average(data, weights=weights)
    
    def calculate_reactive_forecast(self, style_data):
        """
        Reactive forecast for low consistency styles
        Based only on most recent orders
        """
        if isinstance(style_data, pd.DataFrame):
            data = style_data.iloc[:, 0].values
        else:
            data = style_data.values if hasattr(style_data, 'values') else style_data
        
        # Use only last few data points
        if len(data) >= 2:
            return float(data[-2:].mean())
        elif len(data) > 0:
            return float(data[-1])
        else:
            return 0
    
    def calculate_forecast_accuracy(self, actual, predicted):
        """
        Calculate forecast accuracy metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {
                'mape': None,
                'rmse': None,
                'mae': None,
                'accuracy': 0
            }
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if actual.any() else 100
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(actual - predicted))
        
        # Accuracy (1 - MAPE/100)
        accuracy = max(0, 1 - mape/100)
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy
        }
    
    def get_forecast_summary(self, style_data_dict):
        """
        Generate forecast summary for multiple styles
        
        Args:
            style_data_dict: Dictionary of style_id -> historical data
            
        Returns:
            Summary report with forecasts and metrics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_styles': len(style_data_dict),
            'forecasts': {},
            'method_distribution': {
                'ml_forecast': 0,
                'weighted_average': 0,
                'react_to_orders': 0,
                'fallback': 0
            },
            'average_confidence': 0,
            'high_consistency_styles': [],
            'low_consistency_styles': []
        }
        
        total_confidence = 0
        
        for style_id, style_data in style_data_dict.items():
            # Generate forecast
            forecast_result = self.forecast_with_consistency(style_data)
            
            # Store forecast
            summary['forecasts'][style_id] = forecast_result
            
            # Update method distribution
            method = forecast_result.get('method', 'fallback')
            if 'fallback' in method:
                summary['method_distribution']['fallback'] += 1
            else:
                summary['method_distribution'][method] += 1
            
            # Track confidence
            total_confidence += forecast_result.get('confidence', 0)
            
            # Categorize by consistency
            if forecast_result['consistency_score'] >= self.config.consistency_threshold_high:
                summary['high_consistency_styles'].append(style_id)
            elif forecast_result['consistency_score'] < self.config.consistency_threshold_low:
                summary['low_consistency_styles'].append(style_id)
        
        # Calculate average confidence
        if len(style_data_dict) > 0:
            summary['average_confidence'] = total_confidence / len(style_data_dict)
        
        return summary


# Singleton instance for backward compatibility
_instance = None

def get_sales_forecasting_engine() -> SalesForecastingService:
    """
    Get singleton instance of SalesForecastingService
    Maintains backward compatibility with monolith
    """
    global _instance
    if _instance is None:
        _instance = SalesForecastingService()
    return _instance


# For backward compatibility with original class name
SalesForecastingEngine = SalesForecastingService


def test_service():
    """Test the extracted service"""
    print("=" * 80)
    print("Testing SalesForecastingService")
    print("=" * 80)
    
    # Create service instance
    forecaster = SalesForecastingService()
    
    # Test data - simulate style sales history
    test_styles = {
        'STYLE001': pd.DataFrame({
            'quantity': [100, 105, 98, 102, 101, 99, 103, 100, 104, 101]  # Consistent
        }),
        'STYLE002': pd.DataFrame({
            'quantity': [50, 150, 30, 200, 45, 180, 25, 190, 60, 170]  # Inconsistent
        }),
        'STYLE003': pd.DataFrame({
            'quantity': [80, 90, 85, 95, 88, 92, 87, 91, 89, 93]  # Medium consistency
        })
    }
    
    print("\nTesting individual style forecasts:")
    for style_id, style_data in test_styles.items():
        # Calculate consistency
        consistency = forecaster.calculate_consistency_score(style_data)
        print(f"\n{style_id}:")
        print(f"  Consistency Score: {consistency['consistency_score']:.3f}")
        print(f"  CV: {consistency['cv']:.3f}")
        print(f"  Recommendation: {consistency['recommendation']}")
        
        # Generate forecast
        forecast = forecaster.forecast_with_consistency(style_data)
        print(f"  Forecast: {forecast['forecast']:.2f}")
        print(f"  Confidence: {forecast['confidence']:.3f}")
        print(f"  Method: {forecast['method']}")
    
    # Generate summary report
    print("\nGenerating forecast summary...")
    summary = forecaster.get_forecast_summary(test_styles)
    
    print(f"\nForecast Summary:")
    print(f"  Total Styles: {summary['total_styles']}")
    print(f"  Average Confidence: {summary['average_confidence']:.3f}")
    print(f"  Method Distribution: {summary['method_distribution']}")
    print(f"  High Consistency Styles: {summary['high_consistency_styles']}")
    print(f"  Low Consistency Styles: {summary['low_consistency_styles']}")
    
    print("\n" + "=" * 80)
    print("✅ Service test complete")


if __name__ == "__main__":
    test_service()