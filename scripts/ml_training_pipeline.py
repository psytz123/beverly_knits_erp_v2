#!/usr/bin/env python3
"""
ML Training Pipeline for Beverly Knits ERP
Automated training, validation, and deployment of ML models
Created: 2025-09-02
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import joblib
import warnings
import logging
from typing import Dict, List, Any, Optional, Tuple
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import schedule
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import ML configuration
try:
    from config.ml_config import (
        ML_GLOBAL_CONFIG,
        MODEL_REGISTRY,
        TRAINING_SCHEDULE,
        ml_config_manager,
        MODEL_PATH,
        TRAINING_PATH
    )
except ImportError:
    print("Warning: ML configuration not available")
    MODEL_PATH = Path(__file__).parent.parent / "models"
    TRAINING_PATH = Path(__file__).parent.parent / "training_results"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class MLTrainingPipeline:
    """Automated ML training pipeline"""
    
    def __init__(self, data_path: str = None):
        """Initialize training pipeline"""
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).parent.parent / "data" / "production" / "5"
        
        self.models = {}
        self.scalers = {}
        self.training_history = []
        self.model_path = MODEL_PATH
        self.training_path = TRAINING_PATH
        
        # Ensure directories exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.training_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ML Training Pipeline initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Model path: {self.model_path}")
    
    def load_and_prepare_data(self, data_type: str = 'sales') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and prepare data for training"""
        try:
            data_info = {'type': data_type, 'loaded_at': datetime.now()}
            
            if data_type == 'sales':
                # Load sales data
                sales_file = self.data_path / "ERP Data" / "Sales Activity Report.csv"
                if not sales_file.exists():
                    sales_file = self.data_path / "Sales Activity Report.csv"
                
                if sales_file.exists():
                    df = pd.read_csv(sales_file)
                    
                    # Process dates
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                        df = df.dropna(subset=[date_cols[0]])
                    
                    # Clean price columns - remove $ signs and spaces, convert to float
                    price_cols = [col for col in df.columns if 'price' in col.lower()]
                    for col in price_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Clean other potentially problematic numeric columns
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Check if it's a numeric column with formatting issues
                            sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                            if '$' in sample_val or ',' in sample_val:
                                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    data_info['records'] = len(df)
                    data_info['columns'] = df.columns.tolist()
                    
                    logger.info(f"Loaded {data_type} data: {len(df)} records")
                    return df, data_info
            
            elif data_type == 'yarn':
                # Load yarn inventory
                yarn_file = self.data_path / "ERP Data" / "yarn_inventory.xlsx"
                if not yarn_file.exists():
                    yarn_file = self.data_path / "yarn_inventory.xlsx"
                
                if yarn_file.exists():
                    df = pd.read_excel(yarn_file)
                    data_info['records'] = len(df)
                    data_info['columns'] = df.columns.tolist()
                    
                    logger.info(f"Loaded {data_type} data: {len(df)} records")
                    return df, data_info
            
            elif data_type == 'orders':
                # Load knit orders
                orders_file = self.data_path / "ERP Data" / "eFab_Knit_Orders.xlsx"
                if not orders_file.exists():
                    orders_file = self.data_path / "eFab_Knit_Orders.xlsx"
                
                if orders_file.exists():
                    df = pd.read_excel(orders_file)
                    data_info['records'] = len(df)
                    data_info['columns'] = df.columns.tolist()
                    
                    logger.info(f"Loaded {data_type} data: {len(df)} records")
                    return df, data_info
            
            return pd.DataFrame(), data_info
            
        except Exception as e:
            logger.error(f"Error loading {data_type} data: {e}")
            return pd.DataFrame(), {'error': str(e)}
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Engineer features for ML models"""
        try:
            # Create copy
            df_features = df.copy()
            
            # Time-based features
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            if date_cols:
                date_col = date_cols[0]
                df_features['month'] = df_features[date_col].dt.month
                df_features['quarter'] = df_features[date_col].dt.quarter
                df_features['dayofweek'] = df_features[date_col].dt.dayofweek
                df_features['day'] = df_features[date_col].dt.day
                df_features['weekofyear'] = df_features[date_col].dt.isocalendar().week
            
            # Lag features for numeric columns
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            
            if target_col and target_col in numeric_cols:
                for lag in [1, 7, 14, 30]:
                    df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
                
                # Rolling statistics
                for window in [7, 14, 30]:
                    df_features[f'{target_col}_rolling_mean_{window}'] = \
                        df_features[target_col].rolling(window).mean()
                    df_features[f'{target_col}_rolling_std_{window}'] = \
                        df_features[target_col].rolling(window).std()
            
            # Drop rows with NaN from feature engineering
            df_features = df_features.dropna()
            
            logger.info(f"Engineered features: {len(df_features.columns)} total columns")
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return df
    
    def train_arima_model(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            result = {
                'model_type': 'arima',
                'status': 'failed',
                'trained_at': datetime.now()
            }
            
            # Prepare time series
            ts_data = data[target_col].values
            
            # Check stationarity
            adf_test = adfuller(ts_data)
            result['adf_statistic'] = adf_test[0]
            result['adf_pvalue'] = adf_test[1]
            
            # Get configuration
            config = MODEL_REGISTRY.get('arima')
            if config:
                p = config.hyperparameters.get('p', 2)
                d = config.hyperparameters.get('d', 1)
                q = config.hyperparameters.get('q', 2)
            else:
                p, d, q = 2, 1, 2
            
            # Split data
            train_size = int(len(ts_data) * 0.8)
            train, test = ts_data[:train_size], ts_data[train_size:]
            
            # Train model
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            
            # Validate
            predictions = model_fit.forecast(steps=len(test))
            mape = np.mean(np.abs((test - predictions) / test)) * 100
            rmse = np.sqrt(mean_squared_error(test, predictions))
            
            # Save model
            model_name = f"arima_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_path / f"{model_name}.pkl"
            joblib.dump(model_fit, model_path)
            
            result['status'] = 'success'
            result['model_path'] = str(model_path)
            result['metrics'] = {
                'mape': mape,
                'rmse': rmse,
                'train_size': train_size,
                'test_size': len(test)
            }
            result['parameters'] = {'p': p, 'd': d, 'q': q}
            
            logger.info(f"ARIMA model trained: MAPE={mape:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {'model_type': 'arima', 'status': 'failed', 'error': str(e)}
    
    def train_xgboost_model(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:
            import xgboost as xgb
            
            result = {
                'model_type': 'xgboost',
                'status': 'failed',
                'trained_at': datetime.now()
            }
            
            # Engineer features
            df_features = self.engineer_features(data, target_col)
            
            # Prepare features and target
            feature_cols = [col for col in df_features.columns 
                          if col != target_col and df_features[col].dtype in ['int64', 'float64']]
            
            X = df_features[feature_cols].values
            y = df_features[target_col].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get configuration
            config = MODEL_REGISTRY.get('xgboost')
            if config:
                params = config.hyperparameters.copy()
            else:
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)
            
            # Validate
            predictions = model.predict(X_test_scaled)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Save model and scaler
            model_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_path / f"{model_name}.pkl"
            scaler_path = self.model_path / f"{model_name}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            result['status'] = 'success'
            result['model_path'] = str(model_path)
            result['scaler_path'] = str(scaler_path)
            result['metrics'] = {
                'mape': mape,
                'rmse': rmse,
                'r2': r2,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            result['top_features'] = top_features
            result['feature_columns'] = feature_cols
            
            logger.info(f"XGBoost model trained: MAPE={mape:.2f}%, R²={r2:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {'model_type': 'xgboost', 'status': 'failed', 'error': str(e)}
    
    def train_prophet_model(self, data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Train Prophet model"""
        try:
            from prophet import Prophet
            
            result = {
                'model_type': 'prophet',
                'status': 'failed',
                'trained_at': datetime.now()
            }
            
            # Prepare data for Prophet
            prophet_data = data[[date_col, target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            prophet_data = prophet_data.dropna()
            
            # Split data
            train_size = int(len(prophet_data) * 0.8)
            train = prophet_data[:train_size]
            test = prophet_data[train_size:]
            
            # Get configuration
            config = MODEL_REGISTRY.get('prophet')
            if config:
                params = config.hyperparameters.copy()
                # Remove non-Prophet parameters
                params.pop('interval_width', None)
                params.pop('n_changepoints', None)
            else:
                params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10,
                    'seasonality_mode': 'multiplicative'
                }
            
            # Train model
            model = Prophet(**params)
            model.fit(train)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test))
            forecast = model.predict(future)
            predictions = forecast.iloc[-len(test):]['yhat'].values
            
            # Calculate metrics
            test_values = test['y'].values
            mape = np.mean(np.abs((test_values - predictions) / test_values)) * 100
            rmse = np.sqrt(mean_squared_error(test_values, predictions))
            
            # Save model
            model_name = f"prophet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_path / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            result['status'] = 'success'
            result['model_path'] = str(model_path)
            result['metrics'] = {
                'mape': mape,
                'rmse': rmse,
                'train_size': len(train),
                'test_size': len(test)
            }
            
            logger.info(f"Prophet model trained: MAPE={mape:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {'model_type': 'prophet', 'status': 'failed', 'error': str(e)}
    
    def train_model(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        """Train a specific model"""
        logger.info(f"Starting training for {model_name}")
        
        # Check if model needs training
        if not force and not self.should_train_model(model_name):
            logger.info(f"Model {model_name} doesn't need training yet")
            return {'model': model_name, 'status': 'skipped', 'reason': 'Not scheduled'}
        
        # Load data
        data, data_info = self.load_and_prepare_data('sales')
        
        if data.empty:
            logger.error(f"No data available for training {model_name}")
            return {'model': model_name, 'status': 'failed', 'error': 'No data'}
        
        # Determine target column
        qty_cols = [col for col in data.columns if 'qty' in col.lower() or 'quantity' in col.lower()]
        value_cols = [col for col in data.columns if 'price' in col.lower() or 'value' in col.lower()]
        target_col = qty_cols[0] if qty_cols else (value_cols[0] if value_cols else 'Yds_ordered')
        
        # Train based on model type
        if model_name == 'arima':
            result = self.train_arima_model(data, target_col)
        elif model_name == 'xgboost':
            result = self.train_xgboost_model(data, target_col)
        elif model_name == 'prophet':
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                result = self.train_prophet_model(data, date_cols[0], target_col)
            else:
                result = {'model_type': 'prophet', 'status': 'failed', 'error': 'No date column found'}
        else:
            result = {'model_type': model_name, 'status': 'failed', 'error': 'Model not implemented'}
        
        # Save training history
        self.save_training_history(result)
        
        return result
    
    def should_train_model(self, model_name: str) -> bool:
        """Check if model should be trained based on schedule"""
        config = MODEL_REGISTRY.get(model_name)
        
        if not config or not config.enabled:
            return False
        
        # Check last training time
        last_training = self.get_last_training_time(model_name)
        
        if last_training:
            days_since = (datetime.now() - last_training).days
            if days_since < config.retrain_frequency_days:
                return False
        
        return True
    
    def get_last_training_time(self, model_name: str) -> Optional[datetime]:
        """Get last training time for a model"""
        history_file = self.training_path / "training_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                
                for entry in reversed(history):
                    if entry.get('model_type') == model_name and entry.get('status') == 'success':
                        return datetime.fromisoformat(entry.get('trained_at', ''))
        
        return None
    
    def save_training_history(self, result: Dict[str, Any]) -> None:
        """Save training history"""
        history_file = self.training_path / "training_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Convert datetime objects to strings
        result_copy = result.copy()
        if isinstance(result_copy.get('trained_at'), datetime):
            result_copy['trained_at'] = result_copy['trained_at'].isoformat()
        
        history.append(result_copy)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Training history saved")
    
    def train_all_scheduled_models(self) -> Dict[str, Any]:
        """Train all models scheduled for today"""
        models_to_train = ml_config_manager.get_models_for_training()
        
        logger.info(f"Models scheduled for training: {models_to_train}")
        
        results = {}
        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")
            result = self.train_model(model_name)
            results[model_name] = result
        
        # Summary
        summary = {
            'trained_at': datetime.now().isoformat(),
            'models_attempted': len(models_to_train),
            'models_succeeded': sum(1 for r in results.values() if r.get('status') == 'success'),
            'models_failed': sum(1 for r in results.values() if r.get('status') == 'failed'),
            'results': results
        }
        
        # Save summary
        summary_file = self.training_path / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training complete: {summary['models_succeeded']}/{summary['models_attempted']} succeeded")
        
        return summary
    
    def evaluate_model(self, model_path: str, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a trained model"""
        try:
            model = joblib.load(model_path)
            
            # Model-specific evaluation logic would go here
            # This is a placeholder
            
            return {'status': 'success', 'metrics': {}}
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def deploy_model(self, model_name: str, model_path: str) -> bool:
        """Deploy model to production"""
        try:
            # Create production model directory
            prod_model_path = self.model_path / 'production'
            prod_model_path.mkdir(exist_ok=True)
            
            # Copy model to production
            import shutil
            dest_path = prod_model_path / f"{model_name}_production.pkl"
            shutil.copy2(model_path, dest_path)
            
            # Update model registry
            registry_file = prod_model_path / "model_registry.json"
            
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {}
            
            registry[model_name] = {
                'path': str(dest_path),
                'deployed_at': datetime.now().isoformat(),
                'source_path': str(model_path)
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Model {model_name} deployed to production")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ML Training Pipeline')
    parser.add_argument('--model', type=str, help='Specific model to train')
    parser.add_argument('--force', action='store_true', help='Force training even if not scheduled')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    parser.add_argument('--deploy', type=str, help='Deploy model to production')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline(data_path=args.data_path)
    
    if args.schedule:
        # Run on schedule
        logger.info("Starting scheduled training pipeline")
        
        # Schedule daily training
        schedule.every().day.at("02:00").do(pipeline.train_all_scheduled_models)
        
        # Schedule weekly training (Monday)
        schedule.every().monday.at("03:00").do(
            lambda: pipeline.train_model('prophet', force=True)
        )
        
        logger.info("Training pipeline scheduled. Press Ctrl+C to stop.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    elif args.deploy:
        # Deploy specific model
        model_files = list(pipeline.model_path.glob(f"{args.deploy}_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            success = pipeline.deploy_model(args.deploy, latest_model)
            if success:
                print(f"Model {args.deploy} deployed successfully")
            else:
                print(f"Failed to deploy model {args.deploy}")
        else:
            print(f"No trained model found for {args.deploy}")
    
    elif args.model:
        # Train specific model
        logger.info(f"Training {args.model} model")
        result = pipeline.train_model(args.model, force=args.force)
        
        print("\n" + "="*60)
        print(f"Training Results for {args.model}")
        print("="*60)
        
        if result.get('status') == 'success':
            print(f"Status: SUCCESS")
            print(f"Model saved to: {result.get('model_path')}")
            
            if 'metrics' in result:
                print("\nMetrics:")
                for metric, value in result['metrics'].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.3f}")
                    else:
                        print(f"  {metric}: {value}")
            
            if 'top_features' in result:
                print("\nTop Features:")
                for feature, importance in result['top_features'][:5]:
                    print(f"  {feature}: {importance:.3f}")
        else:
            print(f"Status: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    else:
        # Train all scheduled models
        logger.info("Training all scheduled models")
        summary = pipeline.train_all_scheduled_models()
        
        print("\n" + "="*60)
        print("Training Pipeline Summary")
        print("="*60)
        print(f"Models attempted: {summary['models_attempted']}")
        print(f"Models succeeded: {summary['models_succeeded']}")
        print(f"Models failed: {summary['models_failed']}")
        
        print("\nResults by model:")
        for model_name, result in summary['results'].items():
            status = result.get('status', 'unknown')
            print(f"  {model_name}: {status}")
            
            if status == 'success' and 'metrics' in result:
                mape = result['metrics'].get('mape', 'N/A')
                if isinstance(mape, float):
                    print(f"    MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()