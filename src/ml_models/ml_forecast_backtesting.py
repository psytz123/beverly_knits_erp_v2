#!/usr/bin/env python3
"""
Comprehensive ML Forecast Backtesting System
Tests all forecasting models against historical data to validate accuracy
Provides detailed performance metrics and model comparison
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings('ignore')

# Import forecasting modules
try:
    from ai_inventory_optimization import AIInventoryOptimizer
    from inventory_forecast_pipeline import InventoryForecastPipeline
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI modules not available for backtesting")

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from xgboost import XGBRegressor
    from prophet import Prophet
    import tensorflow as tf
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Some ML libraries not available")


class MLForecastBacktester:
    """
    Comprehensive backtesting system for ML forecasts
    Tests multiple models against historical data
    """
    
    def __init__(self, data_path: str = None):
        """Initialize backtester with data path"""
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Try to get from environment variable or use relative path
            import os
            default_path = os.environ.get('DATA_PATH', 
                                         os.path.join(os.path.dirname(__file__), 
                                                     '../../data/production'))
            self.data_path = Path(default_path)
        
        self.models = {}
        self.backtest_results = {}
        self.model_performance = defaultdict(dict)
        self.best_models = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all available ML models for testing"""
        if ML_AVAILABLE:
            self.models['linear_regression'] = LinearRegression()
            self.models['ridge'] = Ridge(alpha=1.0)
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.models['gradient_boost'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            
            try:
                self.models['xgboost'] = XGBRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
            except:
                pass
            
            # Prophet requires special handling - will be initialized per time series
            self.models['prophet'] = 'prophet'  # Will be created during fit
            
            # LSTM model creation (if TensorFlow available)
            try:
                import tensorflow as tf
                self.models['lstm'] = 'lstm'  # Will be created during fit with proper input shape
            except ImportError:
                print("TensorFlow not available, LSTM model skipped")
    
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load all historical data for backtesting"""
        print("="*60)
        print("LOADING HISTORICAL DATA FOR BACKTESTING")
        print("="*60)
        
        data = {}
        
        # Load Sales History
        sales_files = [
            "Sales_Activity_Report.xlsx",
            "Sales Activity Report (4).xlsx",
            "eFab_SO_List_202508101032.xlsx"
        ]
        
        for file in sales_files:
            file_path = self.data_path / file
            if file_path.exists():
                data['sales'] = pd.read_excel(file_path)
                print(f"✅ Loaded sales: {file} ({len(data['sales'])} records)")
                break
        
        # Load Inventory History
        inventory_files = [
            "eFab_Inventory_F01_20250810.xlsx",
            "eFab_Inventory_P01_20250810.xlsx"
        ]
        
        for file in inventory_files:
            file_path = self.data_path / file
            if file_path.exists():
                data['inventory'] = pd.read_excel(file_path)
                print(f"✅ Loaded inventory: {file} ({len(data['inventory'])} items)")
                break
        
        # Load Yarn Inventory History
        yarn_file = self.data_path / "yarn_inventory (2).xlsx"
        if yarn_file.exists():
            data['yarn_inventory'] = pd.read_excel(yarn_file)
            print(f"✅ Loaded yarn inventory: {len(data['yarn_inventory'])} yarns")
        
        # Load Yarn Demand History
        demand_file = self.data_path / "Yarn_Demand_2025-08-09_0442.xlsx"
        if demand_file.exists():
            data['yarn_demand'] = pd.read_excel(demand_file)
            print(f"✅ Loaded yarn demand: {len(data['yarn_demand'])} entries")
        
        print()
        return data
    
    def prepare_time_series(self, data: pd.DataFrame, 
                           date_col: str, 
                           value_col: str,
                           group_col: str = None) -> Dict[str, pd.DataFrame]:
        """Prepare time series data for backtesting"""
        time_series = {}
        
        # Convert date column
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data = data.dropna(subset=[date_col])
        
        if group_col and group_col in data.columns:
            # Group by product/yarn
            for group_id in data[group_col].unique():
                group_data = data[data[group_col] == group_id].copy()
                group_data = group_data.sort_values(date_col)
                
                # Aggregate by day
                daily = group_data.groupby(group_data[date_col].dt.date)[value_col].sum().reset_index()
                daily.columns = ['date', 'value']
                
                if len(daily) >= 30:  # Need minimum history
                    time_series[str(group_id)] = daily
        else:
            # Single time series
            data = data.sort_values(date_col)
            daily = data.groupby(data[date_col].dt.date)[value_col].sum().reset_index()
            daily.columns = ['date', 'value']
            time_series['total'] = daily
        
        return time_series
    
    def backtest_model(self, 
                      time_series: pd.DataFrame,
                      model_name: str,
                      test_size: float = 0.2,
                      forecast_horizon: int = 30) -> Dict:
        """
        Backtest a single model on time series data
        Returns performance metrics
        """
        results = {
            'model': model_name,
            'metrics': {},
            'predictions': [],
            'actuals': []
        }
        
        # Split data
        split_point = int(len(time_series) * (1 - test_size))
        train_data = time_series[:split_point].copy()
        test_data = time_series[split_point:].copy()
        
        if len(test_data) < forecast_horizon:
            forecast_horizon = len(test_data)
        
        try:
            if model_name == 'prophet':
                # Prophet model
                predictions = self._prophet_forecast(train_data, forecast_horizon)
            elif model_name == 'lstm':
                # LSTM model
                predictions = self._lstm_forecast(train_data, test_data, forecast_horizon)
            else:
                # Sklearn models
                predictions = self._sklearn_forecast(
                    train_data, test_data, self.models[model_name], forecast_horizon
                )
            
            # Get actual values
            actuals = test_data['value'].iloc[:forecast_horizon].values
            
            # Calculate metrics
            if len(predictions) == len(actuals) and len(predictions) > 0:
                results['predictions'] = predictions.tolist()
                results['actuals'] = actuals.tolist()
                
                # Performance metrics
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                
                # MAPE - handle zero values
                non_zero_mask = actuals != 0
                if non_zero_mask.any():
                    mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / 
                                         actuals[non_zero_mask])) * 100
                else:
                    mape = np.inf
                
                # R-squared
                if len(actuals) > 1:
                    r2 = r2_score(actuals, predictions)
                else:
                    r2 = 0
                
                results['metrics'] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape) if not np.isinf(mape) else 100.0,
                    'r2': float(r2),
                    'forecast_accuracy': float(100 - min(mape, 100)) if not np.isinf(mape) else 0
                }
                
        except Exception as e:
            print(f"   ⚠️ {model_name} failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _sklearn_forecast(self, train_data: pd.DataFrame, 
                         test_data: pd.DataFrame,
                         model: Any,
                         horizon: int) -> np.ndarray:
        """Generate forecast using sklearn model"""
        # Create features
        train_X = np.arange(len(train_data)).reshape(-1, 1)
        train_y = train_data['value'].values
        
        # Train model
        model.fit(train_X, train_y)
        
        # Predict
        test_X = np.arange(len(train_data), len(train_data) + horizon).reshape(-1, 1)
        predictions = model.predict(test_X)
        
        return predictions
    
    def _prophet_forecast(self, train_data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate forecast using Prophet"""
        if not ML_AVAILABLE:
            return np.array([])
        
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(train_data['date']),
                'y': train_data['value']
            })
            
            # Initialize and fit
            model = Prophet(daily_seasonality=False, weekly_seasonality=True)
            model.fit(prophet_data)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            
            # Return predictions for test period
            predictions = forecast['yhat'].iloc[-horizon:].values
            return predictions
            
        except:
            return np.array([])
    
    def _lstm_forecast(self, train_data: pd.DataFrame, 
                      test_data: pd.DataFrame,
                      horizon: int) -> np.ndarray:
        """Generate forecast using LSTM"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            # Prepare data
            values = train_data['value'].values.reshape(-1, 1)
            
            # Scale data
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)
            
            # Create sequences
            look_back = min(30, len(scaled) // 3)
            X, y = [], []
            for i in range(look_back, len(scaled)):
                X.append(scaled[i-look_back:i, 0])
                y.append(scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(look_back, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Predict
            predictions = []
            last_sequence = scaled[-look_back:]
            
            for _ in range(horizon):
                next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred)
            
            # Inverse transform
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return predictions.flatten()
            
        except:
            return np.array([])
    
    def run_complete_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run complete backtesting on all available data
        """
        print("="*60)
        print("RUNNING COMPLETE ML FORECAST BACKTESTING")
        print("="*60)
        print(f"Models to test: {list(self.models.keys())}")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'sales_forecast': {},
            'inventory_forecast': {},
            'yarn_demand_forecast': {},
            'model_rankings': {},
            'best_models': {}
        }
        
        # 1. Backtest Sales Forecasts
        if 'sales' in data:
            print("\n📊 BACKTESTING SALES FORECASTS")
            print("-"*40)
            
            # Identify columns
            date_col = None
            for col in ['Invoice Date', 'Date', 'Order Date']:
                if col in data['sales'].columns:
                    date_col = col
                    break
            
            qty_col = None
            for col in ['Qty Shipped', 'Quantity', 'Qty']:
                if col in data['sales'].columns:
                    qty_col = col
                    break
            
            style_col = None
            for col in ['fStyle#', 'Style', 'Style#']:
                if col in data['sales'].columns:
                    style_col = col
                    break
            
            if date_col and qty_col:
                # Prepare time series
                time_series = self.prepare_time_series(
                    data['sales'], date_col, qty_col, style_col
                )
                
                print(f"Found {len(time_series)} time series to test")
                
                # Test each time series
                for series_id, ts_data in list(time_series.items())[:5]:  # Test top 5
                    print(f"\nTesting: {series_id}")
                    series_results = {}
                    
                    for model_name in self.models.keys():
                        if model_name in ['prophet', 'lstm'] and len(ts_data) < 100:
                            continue  # Skip complex models for short series
                        
                        result = self.backtest_model(ts_data, model_name)
                        series_results[model_name] = result
                        
                        if 'metrics' in result and result['metrics']:
                            print(f"   {model_name}: MAPE={result['metrics']['mape']:.2f}%, "
                                  f"Accuracy={result['metrics']['forecast_accuracy']:.1f}%")
                    
                    all_results['sales_forecast'][series_id] = series_results
                    
                    # Find best model for this series
                    best_model = min(
                        series_results.items(),
                        key=lambda x: x[1]['metrics'].get('mape', float('inf')) 
                        if 'metrics' in x[1] else float('inf')
                    )
                    all_results['best_models'][series_id] = best_model[0]
        
        # 2. Backtest Yarn Demand Forecasts
        if 'yarn_demand' in data:
            print("\n🧶 BACKTESTING YARN DEMAND FORECASTS")
            print("-"*40)
            
            # Process yarn demand data
            yarn_cols = [col for col in data['yarn_demand'].columns if 'Week' in col]
            
            if yarn_cols:
                for idx, row in data['yarn_demand'].head(5).iterrows():
                    yarn_id = row.get('Desc#', row.get('yarn_id', f'yarn_{idx}'))
                    
                    # Create time series from weekly data
                    weekly_values = []
                    for week_col in yarn_cols:
                        val = row.get(week_col, 0)
                        if pd.notna(val):
                            weekly_values.append(float(val))
                    
                    if len(weekly_values) >= 4:
                        # Create dates
                        dates = pd.date_range(
                            end=datetime.now(), 
                            periods=len(weekly_values), 
                            freq='W'
                        )
                        
                        ts_data = pd.DataFrame({
                            'date': dates,
                            'value': weekly_values
                        })
                        
                        print(f"\nTesting Yarn: {yarn_id}")
                        yarn_results = {}
                        
                        for model_name in ['linear_regression', 'ridge', 'random_forest']:
                            result = self.backtest_model(
                                ts_data, model_name, 
                                test_size=0.25, 
                                forecast_horizon=4
                            )
                            yarn_results[model_name] = result
                            
                            if 'metrics' in result and result['metrics']:
                                print(f"   {model_name}: MAPE={result['metrics']['mape']:.2f}%")
                        
                        all_results['yarn_demand_forecast'][str(yarn_id)] = yarn_results
        
        # 3. Calculate Model Rankings
        print("\n📈 MODEL PERFORMANCE RANKINGS")
        print("-"*40)
        
        model_scores = defaultdict(list)
        
        # Aggregate scores across all forecasts
        for forecast_type in ['sales_forecast', 'yarn_demand_forecast']:
            for series_id, series_results in all_results[forecast_type].items():
                for model_name, result in series_results.items():
                    if 'metrics' in result and result['metrics']:
                        mape = result['metrics'].get('mape', 100)
                        if mape < 100:  # Exclude failed forecasts
                            model_scores[model_name].append(mape)
        
        # Calculate average performance
        model_rankings = {}
        for model_name, scores in model_scores.items():
            if scores:
                avg_mape = np.mean(scores)
                std_mape = np.std(scores)
                model_rankings[model_name] = {
                    'avg_mape': float(avg_mape),
                    'std_mape': float(std_mape),
                    'count': len(scores),
                    'accuracy': float(100 - avg_mape)
                }
        
        # Sort by performance
        sorted_rankings = sorted(
            model_rankings.items(), 
            key=lambda x: x[1]['avg_mape']
        )
        
        print("\n🏆 Model Rankings (by Average MAPE):")
        for rank, (model, stats) in enumerate(sorted_rankings, 1):
            print(f"{rank}. {model}: {stats['avg_mape']:.2f}% "
                  f"(±{stats['std_mape']:.2f}%, n={stats['count']})")
        
        all_results['model_rankings'] = dict(sorted_rankings)
        
        # 4. Generate Summary Report
        print("\n="*60)
        print("BACKTESTING SUMMARY")
        print("="*60)
        
        total_tests = sum([
            len(all_results['sales_forecast']),
            len(all_results['yarn_demand_forecast'])
        ])
        
        if sorted_rankings:
            best_overall = sorted_rankings[0]
            print(f"✅ Total Forecasts Tested: {total_tests}")
            print(f"🏆 Best Overall Model: {best_overall[0]}")
            print(f"📊 Best Average MAPE: {best_overall[1]['avg_mape']:.2f}%")
            print(f"🎯 Best Average Accuracy: {best_overall[1]['accuracy']:.1f}%")
        
        return all_results
    
    def generate_backtest_report(self, results: Dict) -> str:
        """Generate detailed backtesting report"""
        report = []
        report.append("="*60)
        report.append("ML FORECAST BACKTESTING REPORT")
        report.append("="*60)
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        # Model Rankings
        report.append("MODEL PERFORMANCE RANKINGS")
        report.append("-"*40)
        for model, stats in results['model_rankings'].items():
            report.append(f"{model}:")
            report.append(f"  Average MAPE: {stats['avg_mape']:.2f}%")
            report.append(f"  Std Dev: {stats['std_mape']:.2f}%")
            report.append(f"  Accuracy: {stats['accuracy']:.1f}%")
            report.append(f"  Tests Run: {stats['count']}")
            report.append("")
        
        # Best Models by Product
        if results['best_models']:
            report.append("BEST MODELS BY PRODUCT/SERIES")
            report.append("-"*40)
            for series_id, model in results['best_models'].items():
                report.append(f"{series_id}: {model}")
        
        # Detailed Results
        report.append("")
        report.append("DETAILED FORECAST RESULTS")
        report.append("-"*40)
        
        for forecast_type in ['sales_forecast', 'yarn_demand_forecast']:
            if results[forecast_type]:
                report.append(f"\n{forecast_type.upper()}:")
                for series_id, series_results in list(results[forecast_type].items())[:3]:
                    report.append(f"\n  {series_id}:")
                    for model_name, result in series_results.items():
                        if 'metrics' in result and result['metrics']:
                            metrics = result['metrics']
                            report.append(f"    {model_name}:")
                            report.append(f"      MAPE: {metrics['mape']:.2f}%")
                            report.append(f"      RMSE: {metrics['rmse']:.2f}")
                            report.append(f"      MAE: {metrics['mae']:.2f}")
                            report.append(f"      R²: {metrics['r2']:.3f}")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save backtesting results to file"""
        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_path / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        clean_results = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {filepath}")
        
        # Also save text report
        report = self.generate_backtest_report(results)
        report_file = filepath.with_suffix('.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_file}")
    
    def plot_forecast_comparison(self, results: Dict, series_id: str):
        """Create visualization comparing forecast models"""
        if not series_id in results['sales_forecast']:
            print(f"Series {series_id} not found in results")
            return
        
        series_results = results['sales_forecast'][series_id]
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Forecast Model Comparison: {series_id}', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(series_results.items()):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            if 'predictions' in result and 'actuals' in result:
                predictions = result['predictions']
                actuals = result['actuals']
                
                x = range(len(actuals))
                ax.plot(x, actuals, 'b-', label='Actual', linewidth=2)
                ax.plot(x, predictions, 'r--', label='Predicted', linewidth=2)
                
                # Add metrics
                if 'metrics' in result:
                    metrics = result['metrics']
                    ax.text(0.02, 0.98, 
                           f"MAPE: {metrics['mape']:.1f}%\n"
                           f"RMSE: {metrics['rmse']:.1f}\n"
                           f"R²: {metrics['r2']:.3f}",
                           transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_title(model_name)
                ax.set_xlabel('Time Period')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.data_path / f"backtest_plot_{series_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved to: {plot_file}")


def create_backtest_endpoints(app):
    """Add backtesting endpoints to Flask app"""
    
    @app.route("/api/backtest/run", methods=['POST'])
    def run_backtest():
        """Run complete ML forecast backtesting"""
        try:
            backtester = MLForecastBacktester()
            data = backtester.load_historical_data()
            results = backtester.run_complete_backtest(data)
            backtester.save_results(results)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route("/api/backtest/models", methods=['GET'])
    def get_model_performance():
        """Get model performance rankings from last backtest"""
        try:
            # Load most recent backtest results
            backtester = MLForecastBacktester()
            result_files = list(backtester.data_path.glob("backtest_results_*.json"))
            
            if not result_files:
                return jsonify({'error': 'No backtest results found'}), 404
            
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            return jsonify({
                'timestamp': results.get('timestamp'),
                'model_rankings': results.get('model_rankings', {}),
                'best_models': results.get('best_models', {})
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route("/api/backtest/forecast/<product_id>", methods=['GET'])
    def get_product_backtest(product_id):
        """Get backtest results for specific product"""
        try:
            backtester = MLForecastBacktester()
            result_files = list(backtester.data_path.glob("backtest_results_*.json"))
            
            if not result_files:
                return jsonify({'error': 'No backtest results found'}), 404
            
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            # Look for product in results
            product_results = None
            for forecast_type in ['sales_forecast', 'yarn_demand_forecast']:
                if product_id in results.get(forecast_type, {}):
                    product_results = results[forecast_type][product_id]
                    break
            
            if not product_results:
                return jsonify({'error': f'No results for {product_id}'}), 404
            
            return jsonify({
                'product_id': product_id,
                'results': product_results,
                'best_model': results.get('best_models', {}).get(product_id)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    print("✅ Backtest endpoints added to Flask app")


if __name__ == "__main__":
    # Run standalone backtest
    print("\n" + "="*60)
    print("STARTING ML FORECAST BACKTESTING")
    print("="*60)
    
    backtester = MLForecastBacktester()
    data = backtester.load_historical_data()
    
    if data:
        results = backtester.run_complete_backtest(data)
        backtester.save_results(results)
        
        # Generate report
        report = backtester.generate_backtest_report(results)
        print("\n" + report)
        
        print("\n✅ Backtesting completed successfully!")
    else:
        print("❌ No data available for backtesting")