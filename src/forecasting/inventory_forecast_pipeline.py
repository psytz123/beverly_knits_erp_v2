#!/usr/bin/env python3
"""
Inventory Forecast Pipeline Module
End-to-end pipeline for inventory forecasting with ML integration
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available")


@dataclass
class PipelineConfig:
    """Configuration for forecast pipeline"""
    forecast_horizons: List[int] = None  # [7, 14, 30, 60, 90] days
    model_types: List[str] = None  # ['prophet', 'rf', 'gbm', 'ensemble']
    confidence_level: float = 0.95
    min_history_days: int = 30
    seasonality_mode: str = 'multiplicative'
    use_external_features: bool = True
    retrain_frequency_days: int = 7
    model_save_path: str = './models/forecast'
    
    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [7, 14, 30, 60, 90]
        if self.model_types is None:
            self.model_types = ['prophet', 'rf', 'ensemble']


class InventoryForecastPipeline:
    """
    Complete pipeline for inventory forecasting
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.models = {}
        self.scalers = {}
        self.forecast_cache = {}
        self.model_performance = {}
        self._initialize_models()
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("InventoryForecastPipeline initialized")
    
    def _initialize_models(self):
        """Initialize ML models based on configuration"""
        if ML_AVAILABLE:
            if 'rf' in self.config.model_types:
                self.models['rf'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.scalers['rf'] = StandardScaler()
            
            if 'gbm' in self.config.model_types:
                self.models['gbm'] = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                self.scalers['gbm'] = StandardScaler()
        
        if PROPHET_AVAILABLE and 'prophet' in self.config.model_types:
            # Prophet initialized per product
            self.models['prophet'] = 'per_product'
    
    def run_forecast_pipeline(self,
                            historical_data: pd.DataFrame,
                            product_ids: Optional[List[str]] = None,
                            external_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run complete forecasting pipeline
        
        Args:
            historical_data: Historical sales/inventory data
            product_ids: Optional list of products to forecast
            external_features: Optional external features (promotions, holidays, etc.)
        
        Returns:
            Dictionary with forecasts and metadata
        """
        try:
            pipeline_results = {
                'timestamp': datetime.now().isoformat(),
                'forecasts': {},
                'model_performance': {},
                'summary': {},
                'errors': []
            }
            
            # Validate data
            if not self._validate_data(historical_data):
                raise ValueError("Invalid historical data format")
            
            # Get products to forecast
            if product_ids is None:
                product_ids = historical_data['product_id'].unique() if 'product_id' in historical_data else []
            
            # Forecast each product
            successful_forecasts = 0
            for product_id in product_ids:
                try:
                    product_forecast = self._forecast_product(
                        product_id=product_id,
                        historical_data=historical_data,
                        external_features=external_features
                    )
                    
                    if product_forecast:
                        pipeline_results['forecasts'][product_id] = product_forecast
                        successful_forecasts += 1
                
                except Exception as e:
                    logger.error(f"Error forecasting product {product_id}: {str(e)}")
                    pipeline_results['errors'].append({
                        'product_id': product_id,
                        'error': str(e)
                    })
            
            # Aggregate model performance
            pipeline_results['model_performance'] = self._aggregate_model_performance()
            
            # Summary statistics
            pipeline_results['summary'] = {
                'total_products': len(product_ids),
                'successful_forecasts': successful_forecasts,
                'failed_forecasts': len(product_ids) - successful_forecasts,
                'forecast_horizons': self.config.forecast_horizons,
                'models_used': list(self.models.keys())
            }
            
            # Cache results
            self.forecast_cache['latest'] = pipeline_results
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = ['date', 'product_id', 'quantity']
        
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check data types
        try:
            data['date'] = pd.to_datetime(data['date'])
            data['quantity'] = pd.to_numeric(data['quantity'])
            return True
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _forecast_product(self,
                         product_id: str,
                         historical_data: pd.DataFrame,
                         external_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Forecast individual product using configured models
        """
        # Filter product data
        product_data = historical_data[historical_data['product_id'] == product_id].copy()
        
        if len(product_data) < self.config.min_history_days:
            logger.warning(f"Insufficient history for product {product_id}")
            return None
        
        # Prepare data
        product_data = product_data.sort_values('date')
        product_data.set_index('date', inplace=True)
        
        forecasts = {}
        
        # Prophet forecast
        if 'prophet' in self.models and PROPHET_AVAILABLE:
            prophet_forecast = self._prophet_forecast(product_data, product_id)
            if prophet_forecast is not None:
                forecasts['prophet'] = prophet_forecast
        
        # ML model forecasts
        if ML_AVAILABLE:
            if 'rf' in self.models:
                rf_forecast = self._ml_forecast(product_data, 'rf', product_id)
                if rf_forecast is not None:
                    forecasts['rf'] = rf_forecast
            
            if 'gbm' in self.models:
                gbm_forecast = self._ml_forecast(product_data, 'gbm', product_id)
                if gbm_forecast is not None:
                    forecasts['gbm'] = gbm_forecast
        
        # Ensemble forecast
        if len(forecasts) > 1 and 'ensemble' in self.config.model_types:
            ensemble_forecast = self._ensemble_forecast(forecasts)
            forecasts['ensemble'] = ensemble_forecast
        
        # Select best forecast
        best_forecast = self._select_best_forecast(forecasts)
        
        return {
            'product_id': product_id,
            'forecast_date': datetime.now().isoformat(),
            'historical_points': len(product_data),
            'forecasts': forecasts,
            'best_forecast': best_forecast,
            'confidence_intervals': self._calculate_confidence_intervals(best_forecast)
        }
    
    def _prophet_forecast(self, data: pd.DataFrame, product_id: str) -> Optional[Dict[str, float]]:
        """Generate forecast using Prophet"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['quantity']
            })
            
            # Initialize and fit Prophet
            model = Prophet(
                seasonality_mode=self.config.seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            # Suppress Prophet output
            with open('nul', 'w') as devnull:
                import sys
                old_stdout = sys.stdout
                sys.stdout = devnull
                model.fit(prophet_data)
                sys.stdout = old_stdout
            
            # Make predictions
            future = model.make_future_dataframe(periods=max(self.config.forecast_horizons))
            forecast = model.predict(future)
            
            # Extract forecasts for configured horizons
            results = {}
            last_date = data.index.max()
            
            for horizon in self.config.forecast_horizons:
                forecast_date = last_date + timedelta(days=horizon)
                if forecast_date in forecast['ds'].values:
                    idx = forecast[forecast['ds'] == forecast_date].index[0]
                    results[f'day_{horizon}'] = max(0, forecast.loc[idx, 'yhat'])
            
            return results
            
        except Exception as e:
            logger.error(f"Prophet forecast error for {product_id}: {str(e)}")
            return None
    
    def _ml_forecast(self, data: pd.DataFrame, model_type: str, product_id: str) -> Optional[Dict[str, float]]:
        """Generate forecast using ML models"""
        try:
            # Feature engineering
            features = self._create_features(data)
            
            if features is None or len(features) < self.config.min_history_days:
                return None
            
            # Prepare training data
            X = features[['day_of_week', 'month', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']]
            y = features['quantity']
            
            # Train model
            X_scaled = self.scalers[model_type].fit_transform(X)
            self.models[model_type].fit(X_scaled, y)
            
            # Generate forecasts
            results = {}
            last_features = X.iloc[-1:].copy()
            
            for horizon in self.config.forecast_horizons:
                # Simple feature projection (would be more sophisticated in production)
                future_features = last_features.copy()
                future_features['day_of_week'] = (last_features['day_of_week'].iloc[0] + horizon) % 7
                
                # Predict
                future_scaled = self.scalers[model_type].transform(future_features)
                prediction = self.models[model_type].predict(future_scaled)[0]
                results[f'day_{horizon}'] = max(0, prediction)
            
            return results
            
        except Exception as e:
            logger.error(f"ML forecast error for {product_id} with {model_type}: {str(e)}")
            return None
    
    def _create_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create features for ML models"""
        try:
            df = data.copy()
            
            # Time features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['day_of_month'] = df.index.day
            
            # Lag features
            df['lag_7'] = df['quantity'].shift(7)
            df['lag_14'] = df['quantity'].shift(14)
            df['lag_30'] = df['quantity'].shift(30)
            
            # Rolling statistics
            df['rolling_mean_7'] = df['quantity'].rolling(window=7, min_periods=1).mean()
            df['rolling_mean_30'] = df['quantity'].rolling(window=30, min_periods=1).mean()
            df['rolling_std_7'] = df['quantity'].rolling(window=7, min_periods=1).std()
            
            # Drop NaN values
            df = df.dropna()
            
            return df if len(df) > 0 else None
            
        except Exception as e:
            logger.error(f"Feature creation error: {str(e)}")
            return None
    
    def _ensemble_forecast(self, forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Create ensemble forecast from multiple models"""
        ensemble = {}
        
        # Define weights (could be learned from validation performance)
        weights = {
            'prophet': 0.4,
            'rf': 0.3,
            'gbm': 0.3
        }
        
        # Get all forecast horizons
        all_horizons = set()
        for model_forecasts in forecasts.values():
            all_horizons.update(model_forecasts.keys())
        
        # Calculate weighted average for each horizon
        for horizon in all_horizons:
            weighted_sum = 0
            total_weight = 0
            
            for model, model_forecast in forecasts.items():
                if horizon in model_forecast:
                    weight = weights.get(model, 1.0 / len(forecasts))
                    weighted_sum += model_forecast[horizon] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble[horizon] = weighted_sum / total_weight
        
        return ensemble
    
    def _select_best_forecast(self, forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Select best forecast based on historical performance"""
        if not forecasts:
            return {}
        
        # If we have ensemble, prefer it
        if 'ensemble' in forecasts:
            return forecasts['ensemble']
        
        # Otherwise, prefer Prophet if available
        if 'prophet' in forecasts:
            return forecasts['prophet']
        
        # Return first available forecast
        return next(iter(forecasts.values()))
    
    def _calculate_confidence_intervals(self, forecast: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for forecasts"""
        intervals = {}
        
        for horizon, value in forecast.items():
            # Simple confidence interval (would use model uncertainty in production)
            std_error = value * 0.1  # 10% standard error assumption
            z_score = 1.96  # 95% confidence
            
            lower = max(0, value - z_score * std_error)
            upper = value + z_score * std_error
            
            intervals[horizon] = (lower, upper)
        
        return intervals
    
    def _aggregate_model_performance(self) -> Dict[str, Any]:
        """Aggregate model performance metrics"""
        return {
            'models_trained': len(self.models),
            'last_training': datetime.now().isoformat(),
            'performance_metrics': self.model_performance
        }
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all configured models
        """
        training_results = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': [],
            'errors': []
        }
        
        # Train each model type
        for model_type in self.config.model_types:
            try:
                if model_type == 'prophet':
                    # Prophet trained per product during forecasting
                    training_results['models_trained'].append({
                        'model': model_type,
                        'status': 'ready'
                    })
                elif model_type in ['rf', 'gbm'] and ML_AVAILABLE:
                    # Train ML models on aggregate data
                    self._train_ml_model(model_type, training_data)
                    training_results['models_trained'].append({
                        'model': model_type,
                        'status': 'trained'
                    })
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                training_results['errors'].append({
                    'model': model_type,
                    'error': str(e)
                })
        
        return training_results
    
    def _train_ml_model(self, model_type: str, data: pd.DataFrame):
        """Train individual ML model"""
        # Aggregate training across all products
        all_features = []
        all_targets = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].copy()
            product_data.set_index('date', inplace=True)
            
            features = self._create_features(product_data)
            if features is not None and len(features) > 10:
                X = features[['day_of_week', 'month', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']]
                y = features['quantity']
                
                all_features.append(X)
                all_targets.append(y)
        
        if all_features:
            X_combined = pd.concat(all_features)
            y_combined = pd.concat(all_targets)
            
            # Train model
            X_scaled = self.scalers[model_type].fit_transform(X_combined)
            self.models[model_type].fit(X_scaled, y_combined)
            
            # Save model
            model_path = Path(self.config.model_save_path) / f"{model_type}_model.pkl"
            joblib.dump(self.models[model_type], model_path)
            
            scaler_path = Path(self.config.model_save_path) / f"{model_type}_scaler.pkl"
            joblib.dump(self.scalers[model_type], scaler_path)
    
    def load_models(self) -> bool:
        """Load saved models"""
        try:
            for model_type in ['rf', 'gbm']:
                model_path = Path(self.config.model_save_path) / f"{model_type}_model.pkl"
                scaler_path = Path(self.config.model_save_path) / f"{model_type}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
                    logger.info(f"Loaded {model_type} model")
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_forecast_summary(self, forecast_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert forecast results to summary DataFrame
        """
        summary_data = []
        
        for product_id, forecast in forecast_results.get('forecasts', {}).items():
            row = {
                'product_id': product_id,
                'forecast_date': forecast.get('forecast_date', ''),
                'historical_points': forecast.get('historical_points', 0)
            }
            
            # Add best forecast values
            best_forecast = forecast.get('best_forecast', {})
            for horizon in self.config.forecast_horizons:
                key = f'day_{horizon}'
                row[f'forecast_{horizon}d'] = best_forecast.get(key, 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


# Module exports
__all__ = [
    'InventoryForecastPipeline',
    'PipelineConfig'
]