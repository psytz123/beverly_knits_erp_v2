#!/usr/bin/env python3
"""
ML Model Backtesting Script for Beverly Knits ERP
Comprehensive backtesting for all ML models with performance metrics
Created: 2025-09-02
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import logging
from typing import Dict, List, Tuple, Any, Optional
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import ML configuration
try:
    from config.ml_config import (
        ML_GLOBAL_CONFIG, 
        MODEL_REGISTRY,
        PERFORMANCE_BENCHMARKS,
        ml_config_manager
    )
except ImportError:
    print("Warning: ML configuration not available, using defaults")
    ML_GLOBAL_CONFIG = {'enable_ml': True}
    MODEL_REGISTRY = {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MLBacktester:
    """Comprehensive ML model backtesting system"""
    
    def __init__(self, data_path: str = None):
        """Initialize backtester with data path"""
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Use project root for consistent path resolution
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "production" / "5"
        
        self.results = {}
        self.models = {}
        self.data = {}
        
        logger.info(f"ML Backtester initialized with data path: {self.data_path}")
    
    def load_data(self) -> bool:
        """Load all required data for backtesting"""
        try:
            # Load sales data
            sales_file = self.data_path / "ERP Data" / "Sales Activity Report.csv"
            if not sales_file.exists():
                sales_file = self.data_path / "Sales Activity Report.csv"
            
            if sales_file.exists():
                self.data['sales'] = pd.read_csv(sales_file)
                logger.info(f"Loaded sales data: {len(self.data['sales'])} records")
            
            # Load yarn inventory
            yarn_file = self.data_path / "ERP Data" / "yarn_inventory.xlsx"
            if not yarn_file.exists():
                yarn_file = self.data_path / "yarn_inventory.xlsx"
            
            if yarn_file.exists():
                self.data['yarn'] = pd.read_excel(yarn_file)
                logger.info(f"Loaded yarn data: {len(self.data['yarn'])} items")
            
            # Load knit orders
            orders_file = self.data_path / "ERP Data" / "eFab_Knit_Orders.xlsx"
            if not orders_file.exists():
                orders_file = self.data_path / "eFab_Knit_Orders.xlsx"
            
            if orders_file.exists():
                self.data['orders'] = pd.read_excel(orders_file)
                logger.info(f"Loaded orders data: {len(self.data['orders'])} orders")
            
            return len(self.data) > 0
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def prepare_timeseries_data(self, df: pd.DataFrame, 
                               date_col: str = 'Date',
                               value_col: str = 'Quantity') -> pd.DataFrame:
        """Prepare data for time series modeling"""
        try:
            # Ensure date column is datetime
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                
                # Aggregate by date
                ts_data = df.groupby(date_col)[value_col].sum().reset_index()
                ts_data = ts_data.sort_values(date_col)
                
                # Fill missing dates
                date_range = pd.date_range(
                    start=ts_data[date_col].min(),
                    end=ts_data[date_col].max(),
                    freq='D'
                )
                ts_data = ts_data.set_index(date_col).reindex(date_range).fillna(0)
                ts_data.index.name = date_col
                ts_data = ts_data.reset_index()
                
                return ts_data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error preparing timeseries data: {e}")
            return pd.DataFrame()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return metrics
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.inf
        
        # Directional accuracy
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = (direction_true == direction_pred).mean()
        
        # Bias
        metrics['bias'] = np.mean(y_pred - y_true)
        
        return metrics
    
    def backtest_arima(self, data: pd.DataFrame, test_size: int = 30) -> Dict[str, Any]:
        """Backtest ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            results = {
                'model': 'arima',
                'status': 'failed',
                'metrics': {},
                'predictions': []
            }
            
            if len(data) < 100:
                logger.warning("Insufficient data for ARIMA backtesting")
                return results
            
            # Split data
            train_size = len(data) - test_size
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            # Get value column (assume it's the second column)
            value_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
            
            # Train ARIMA model
            model_config = MODEL_REGISTRY.get('arima', {})
            order = (2, 1, 2)  # Default ARIMA order
            
            if hasattr(model_config, 'hyperparameters'):
                p = model_config.hyperparameters.get('p', 2)
                d = model_config.hyperparameters.get('d', 1)
                q = model_config.hyperparameters.get('q', 2)
                order = (p, d, q)
            
            model = ARIMA(train_data[value_col], order=order)
            model_fit = model.fit()
            
            # Make predictions
            predictions = model_fit.forecast(steps=test_size)
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                test_data[value_col].values,
                predictions.values
            )
            
            results['status'] = 'success'
            results['metrics'] = metrics
            results['predictions'] = predictions.tolist()
            results['test_actual'] = test_data[value_col].tolist()
            results['train_size'] = train_size
            results['test_size'] = test_size
            
            logger.info(f"ARIMA backtest complete: MAPE={metrics.get('mape', 'N/A'):.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA backtest failed: {e}")
            return {
                'model': 'arima',
                'status': 'failed',
                'error': str(e)
            }
    
    def backtest_prophet(self, data: pd.DataFrame, test_size: int = 30) -> Dict[str, Any]:
        """Backtest Prophet model"""
        try:
            from prophet import Prophet
            
            results = {
                'model': 'prophet',
                'status': 'failed',
                'metrics': {},
                'predictions': []
            }
            
            if len(data) < 100:
                logger.warning("Insufficient data for Prophet backtesting")
                return results
            
            # Prepare data for Prophet
            date_col = data.columns[0]
            value_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
            
            prophet_data = data.rename(columns={date_col: 'ds', value_col: 'y'})
            
            # Split data
            train_size = len(prophet_data) - test_size
            train_data = prophet_data.iloc[:train_size]
            test_data = prophet_data.iloc[train_size:]
            
            # Train Prophet model
            model_config = MODEL_REGISTRY.get('prophet', {})
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            if hasattr(model_config, 'hyperparameters'):
                for param, value in model_config.hyperparameters.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
            
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=test_size)
            forecast = model.predict(future)
            predictions = forecast.iloc[-test_size:]['yhat']
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                test_data['y'].values,
                predictions.values
            )
            
            results['status'] = 'success'
            results['metrics'] = metrics
            results['predictions'] = predictions.tolist()
            results['test_actual'] = test_data['y'].tolist()
            results['train_size'] = train_size
            results['test_size'] = test_size
            
            logger.info(f"Prophet backtest complete: MAPE={metrics.get('mape', 'N/A'):.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Prophet backtest failed: {e}")
            return {
                'model': 'prophet',
                'status': 'failed',
                'error': str(e)
            }
    
    def backtest_xgboost(self, data: pd.DataFrame, test_size: int = 30) -> Dict[str, Any]:
        """Backtest XGBoost model with feature engineering"""
        try:
            import xgboost as xgb
            
            results = {
                'model': 'xgboost',
                'status': 'failed',
                'metrics': {},
                'predictions': []
            }
            
            if len(data) < 100:
                logger.warning("Insufficient data for XGBoost backtesting")
                return results
            
            # Prepare features
            value_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
            
            # Create lag features
            for lag in [1, 7, 14, 30]:
                data[f'lag_{lag}'] = data[value_col].shift(lag)
            
            # Create rolling features
            for window in [7, 14, 30]:
                data[f'rolling_mean_{window}'] = data[value_col].rolling(window).mean()
                data[f'rolling_std_{window}'] = data[value_col].rolling(window).std()
            
            # Create time features
            if data.columns[0].dtype == 'datetime64[ns]':
                data['month'] = pd.to_datetime(data[data.columns[0]]).dt.month
                data['dayofweek'] = pd.to_datetime(data[data.columns[0]]).dt.dayofweek
                data['quarter'] = pd.to_datetime(data[data.columns[0]]).dt.quarter
            
            # Drop NaN values
            data = data.dropna()
            
            if len(data) < test_size + 50:
                logger.warning("Insufficient data after feature engineering")
                return results
            
            # Prepare features and target
            feature_cols = [col for col in data.columns if col not in [data.columns[0], value_col]]
            X = data[feature_cols].values
            y = data[value_col].values
            
            # Split data
            split_point = len(data) - test_size
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Train XGBoost model
            model_config = MODEL_REGISTRY.get('xgboost', {})
            
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            if hasattr(model_config, 'hyperparameters'):
                params.update(model_config.hyperparameters)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, predictions)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            results['status'] = 'success'
            results['metrics'] = metrics
            results['predictions'] = predictions.tolist()
            results['test_actual'] = y_test.tolist()
            results['train_size'] = len(X_train)
            results['test_size'] = len(X_test)
            results['feature_importance'] = feature_importance
            
            logger.info(f"XGBoost backtest complete: MAPE={metrics.get('mape', 'N/A'):.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"XGBoost backtest failed: {e}")
            return {
                'model': 'xgboost',
                'status': 'failed',
                'error': str(e)
            }
    
    def backtest_ensemble(self, data: pd.DataFrame, test_size: int = 30) -> Dict[str, Any]:
        """Backtest ensemble of models"""
        results = {
            'model': 'ensemble',
            'status': 'failed',
            'metrics': {},
            'predictions': [],
            'individual_results': {}
        }
        
        try:
            # Run individual models
            models_results = {}
            models_results['arima'] = self.backtest_arima(data, test_size)
            models_results['prophet'] = self.backtest_prophet(data, test_size)
            models_results['xgboost'] = self.backtest_xgboost(data, test_size)
            
            # Filter successful models
            successful_models = {
                name: result for name, result in models_results.items()
                if result.get('status') == 'success'
            }
            
            if len(successful_models) == 0:
                logger.warning("No models succeeded for ensemble")
                return results
            
            # Get ensemble configuration
            ensemble_config = MODEL_REGISTRY.get('ensemble', {})
            weights = {'arima': 0.25, 'prophet': 0.35, 'xgboost': 0.4}
            
            if hasattr(ensemble_config, 'hyperparameters'):
                model_weights = ensemble_config.hyperparameters.get('weights', [])
                model_names = ensemble_config.hyperparameters.get('models', [])
                if model_weights and model_names:
                    weights = dict(zip(model_names, model_weights))
            
            # Normalize weights for available models
            available_weights = {k: v for k, v in weights.items() if k in successful_models}
            weight_sum = sum(available_weights.values())
            normalized_weights = {k: v/weight_sum for k, v in available_weights.items()}
            
            # Combine predictions
            ensemble_predictions = np.zeros(test_size)
            actual_values = None
            
            for model_name, weight in normalized_weights.items():
                model_preds = np.array(successful_models[model_name]['predictions'])
                ensemble_predictions += weight * model_preds
                
                if actual_values is None:
                    actual_values = np.array(successful_models[model_name]['test_actual'])
            
            # Calculate ensemble metrics
            metrics = self.calculate_metrics(actual_values, ensemble_predictions)
            
            results['status'] = 'success'
            results['metrics'] = metrics
            results['predictions'] = ensemble_predictions.tolist()
            results['test_actual'] = actual_values.tolist()
            results['individual_results'] = successful_models
            results['weights'] = normalized_weights
            results['models_used'] = list(successful_models.keys())
            
            logger.info(f"Ensemble backtest complete: MAPE={metrics.get('mape', 'N/A'):.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble backtest failed: {e}")
            results['error'] = str(e)
            return results
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtest for all models"""
        logger.info("Starting comprehensive backtest")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data")
            return {'status': 'failed', 'error': 'Data loading failed'}
        
        # Prepare time series data
        if 'sales' in self.data:
            # Find date and quantity columns
            date_cols = [col for col in self.data['sales'].columns if 'date' in col.lower()]
            qty_cols = [col for col in self.data['sales'].columns if 'qty' in col.lower() or 'quantity' in col.lower()]
            
            if date_cols and qty_cols:
                ts_data = self.prepare_timeseries_data(
                    self.data['sales'],
                    date_col=date_cols[0],
                    value_col=qty_cols[0]
                )
            else:
                # Use default columns
                ts_data = self.prepare_timeseries_data(self.data['sales'])
        else:
            logger.error("No sales data available for backtesting")
            return {'status': 'failed', 'error': 'No sales data'}
        
        if ts_data.empty:
            logger.error("Failed to prepare time series data")
            return {'status': 'failed', 'error': 'Data preparation failed'}
        
        # Run backtests
        backtest_results = {}
        
        # Individual models
        logger.info("Running ARIMA backtest...")
        backtest_results['arima'] = self.backtest_arima(ts_data)
        
        logger.info("Running Prophet backtest...")
        backtest_results['prophet'] = self.backtest_prophet(ts_data)
        
        logger.info("Running XGBoost backtest...")
        backtest_results['xgboost'] = self.backtest_xgboost(ts_data)
        
        # Ensemble
        logger.info("Running Ensemble backtest...")
        backtest_results['ensemble'] = self.backtest_ensemble(ts_data)
        
        # Summary
        summary = self.generate_summary(backtest_results)
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'data_points': len(ts_data),
            'models_tested': list(backtest_results.keys()),
            'results': backtest_results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of backtest results"""
        summary = {
            'best_model': None,
            'best_mape': float('inf'),
            'model_comparison': {},
            'recommendations': []
        }
        
        for model_name, result in results.items():
            if result.get('status') == 'success':
                metrics = result.get('metrics', {})
                mape = metrics.get('mape', float('inf'))
                
                summary['model_comparison'][model_name] = {
                    'mape': mape,
                    'rmse': metrics.get('rmse', 'N/A'),
                    'r2': metrics.get('r2', 'N/A'),
                    'status': 'success'
                }
                
                if mape < summary['best_mape']:
                    summary['best_mape'] = mape
                    summary['best_model'] = model_name
            else:
                summary['model_comparison'][model_name] = {
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                }
        
        # Generate recommendations
        if summary['best_model']:
            summary['recommendations'].append(
                f"Use {summary['best_model']} for production (MAPE: {summary['best_mape']:.2f}%)"
            )
            
            # Check against benchmarks
            if summary['best_model'] in PERFORMANCE_BENCHMARKS:
                benchmark = PERFORMANCE_BENCHMARKS[summary['best_model']]
                if summary['best_mape'] > benchmark.get('mape', 0) * 100:
                    summary['recommendations'].append(
                        f"Consider retraining {summary['best_model']} - performance below benchmark"
                    )
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save backtest results to file"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'training_results'
            output_path.mkdir(exist_ok=True)
            output_path = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ML Model Backtest Results', fontsize=16)
            
            # Plot 1: Model comparison (MAPE)
            ax1 = axes[0, 0]
            model_names = []
            mape_values = []
            
            for model, result in results.get('results', {}).items():
                if result.get('status') == 'success':
                    model_names.append(model)
                    mape_values.append(result['metrics'].get('mape', 0))
            
            ax1.bar(model_names, mape_values)
            ax1.set_title('Model Comparison (MAPE)')
            ax1.set_ylabel('MAPE (%)')
            ax1.set_xlabel('Model')
            
            # Plot 2: Best model predictions vs actual
            ax2 = axes[0, 1]
            best_model = results.get('summary', {}).get('best_model')
            
            if best_model and best_model in results.get('results', {}):
                best_result = results['results'][best_model]
                if 'predictions' in best_result and 'test_actual' in best_result:
                    ax2.plot(best_result['test_actual'], label='Actual', alpha=0.7)
                    ax2.plot(best_result['predictions'], label='Predicted', alpha=0.7)
                    ax2.set_title(f'Best Model ({best_model}) - Predictions vs Actual')
                    ax2.set_xlabel('Time Period')
                    ax2.set_ylabel('Value')
                    ax2.legend()
            
            # Plot 3: Metrics comparison
            ax3 = axes[1, 0]
            metrics_data = []
            
            for model, result in results.get('results', {}).items():
                if result.get('status') == 'success':
                    metrics = result.get('metrics', {})
                    metrics_data.append({
                        'Model': model,
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'R2': metrics.get('r2', 0)
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.set_index('Model').plot(kind='bar', ax=ax3)
                ax3.set_title('Metrics Comparison')
                ax3.set_ylabel('Value')
                ax3.legend()
            
            # Plot 4: Feature importance (if XGBoost was run)
            ax4 = axes[1, 1]
            if 'xgboost' in results.get('results', {}) and results['results']['xgboost'].get('status') == 'success':
                feature_importance = results['results']['xgboost'].get('feature_importance', {})
                if feature_importance:
                    features = list(feature_importance.keys())[:10]  # Top 10
                    importances = [feature_importance[f] for f in features]
                    ax4.barh(features, importances)
                    ax4.set_title('XGBoost Feature Importance (Top 10)')
                    ax4.set_xlabel('Importance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to: {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ML Model Backtesting')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    parser.add_argument('--test-size', type=int, default=30, help='Test set size in days')
    parser.add_argument('--model', type=str, help='Specific model to test')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize backtester
    backtester = MLBacktester(data_path=args.data_path)
    
    # Run backtest
    if args.model:
        # Test specific model
        logger.info(f"Running backtest for {args.model}")
        
        # Load and prepare data
        if not backtester.load_data():
            logger.error("Failed to load data")
            return
        
        # Prepare time series data
        if 'sales' in backtester.data:
            ts_data = backtester.prepare_timeseries_data(backtester.data['sales'])
        else:
            logger.error("No sales data available")
            return
        
        # Run specific model backtest
        if args.model == 'arima':
            results = backtester.backtest_arima(ts_data, args.test_size)
        elif args.model == 'prophet':
            results = backtester.backtest_prophet(ts_data, args.test_size)
        elif args.model == 'xgboost':
            results = backtester.backtest_xgboost(ts_data, args.test_size)
        elif args.model == 'ensemble':
            results = backtester.backtest_ensemble(ts_data, args.test_size)
        else:
            logger.error(f"Unknown model: {args.model}")
            return
        
        # Wrap in comprehensive format
        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'results': {args.model: results},
            'summary': backtester.generate_summary({args.model: results})
        }
    else:
        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest()
    
    # Display results
    print("\n" + "="*60)
    print("ML BACKTEST RESULTS")
    print("="*60)
    
    if results.get('status') == 'success':
        summary = results.get('summary', {})
        
        print(f"\nBest Model: {summary.get('best_model')}")
        print(f"Best MAPE: {summary.get('best_mape', 'N/A'):.2f}%")
        
        print("\nModel Comparison:")
        for model, metrics in summary.get('model_comparison', {}).items():
            if metrics.get('status') == 'success':
                print(f"  {model}:")
                print(f"    - MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                print(f"    - RMSE: {metrics.get('rmse', 'N/A'):.2f}")
                print(f"    - R²: {metrics.get('r2', 'N/A'):.3f}")
            else:
                print(f"  {model}: FAILED - {metrics.get('error', 'Unknown error')}")
        
        print("\nRecommendations:")
        for rec in summary.get('recommendations', []):
            print(f"  - {rec}")
    else:
        print(f"Backtest failed: {results.get('error', 'Unknown error')}")
    
    # Save results if requested
    if args.save_results and results.get('status') == 'success':
        output_path = backtester.save_results(results)
        print(f"\nResults saved to: {output_path}")
    
    # Generate plots if requested
    if args.plot and results.get('status') == 'success':
        plot_path = None
        if args.save_results:
            plot_path = str(Path(output_path).with_suffix('.png'))
        backtester.plot_results(results, plot_path)
        if plot_path:
            print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()