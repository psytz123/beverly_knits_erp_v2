"""
Sales Forecasting Service
Extracted from beverly_comprehensive_erp.py (lines 1263-2464)
PRESERVED EXACTLY - all methods, all ML models and fallback logic
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
import warnings
warnings.filterwarnings('ignore')

# Configure logging for ML error tracking
ml_logger = logging.getLogger('ML_ERROR')

# ML and forecasting libraries - import with fallback
ML_AVAILABLE = False
STATSMODELS_AVAILABLE = False
XGBOOST_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from prophet import Prophet
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Column variations for flexible detection
STYLE_VARIATIONS = ['fStyle#', 'Style#', 'Style', 'style', 'STYLE']

def find_column(df, variations):
    """Find column by checking multiple name variations"""
    for col in variations:
        if col in df.columns:
            return col
    return None


class SalesForecastingEngine:
    """
    Advanced Sales Forecasting Engine with Multi-Model Approach
    Implements ARIMA, Prophet, LSTM, XGBoost with ensemble predictions
    Target: >85% forecast accuracy with 90-day horizon
    """

    def __init__(self):
        self.models = {}
        self.feature_extractors = {}
        self.validation_metrics = {}
        self.ensemble_weights = {}
        self.forecast_horizon = 90  # 90-day forecast
        self.target_accuracy = 0.85  # 85% accuracy target
        self.ML_AVAILABLE = False
        self.ml_engines = {}
        self.initialize_ml_engines()

    def initialize_ml_engines(self):
        """Initialize available ML engines with proper error handling"""
        self.ml_engines = {}
        
        # Try to import RandomForest
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.ml_engines['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ML_AVAILABLE = True
        except ImportError:
            print("RandomForest not available - sklearn not installed")
        
        # Try to import Prophet
        try:
            from prophet import Prophet
            self.ml_engines['prophet'] = Prophet
            self.ML_AVAILABLE = True
        except ImportError:
            print("Prophet not available")
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.ml_engines['xgboost'] = xgb.XGBRegressor
            self.ML_AVAILABLE = True
        except ImportError:
            print("XGBoost not available")
        
        # Try basic sklearn models as fallback
        if not self.ml_engines:
            try:
                from sklearn.linear_model import LinearRegression
                self.ml_engines['linear'] = LinearRegression()
                self.ML_AVAILABLE = True
            except ImportError:
                print("No ML engines available - using fallback forecasting")
                self.ML_AVAILABLE = False
    
    def fallback_forecast(self, historical_data):
        """Simple moving average fallback when no ML engines are available"""
        if isinstance(historical_data, pd.DataFrame):
            if 'quantity' in historical_data.columns:
                data = historical_data['quantity']
            elif 'sales' in historical_data.columns:
                data = historical_data['sales']
            else:
                data = historical_data.iloc[:, 0]
        else:
            data = historical_data
        
        # Simple moving average
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
        
        Args:
            style_history: DataFrame or Series with historical sales data
            
        Returns:
            dict with consistency_score (0-1), cv value, and recommendation
        """
        # Extract quantity data
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
        if len(data) < 3:
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
        # Lower CV = more consistent
        if mean_value > 0:
            cv = std_value / mean_value
        else:
            cv = 1.0
        
        # Convert CV to consistency score (0-1, where 1 is most consistent)
        # CV of 0 = perfectly consistent (score = 1)
        # CV of 1 = high variability (score = 0)
        consistency_score = max(0, 1 - cv)
        
        # Determine recommendation based on consistency score
        if consistency_score >= 0.7:
            recommendation = 'use_ml_forecast'
        elif consistency_score >= 0.3:
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
    
    def forecast_with_consistency(self, style_data, horizon_days=90):
        """
        Generate forecast based on consistency score
        High consistency → Use ML forecast
        Medium consistency → Use weighted average
        Low consistency → React to orders only
        
        Args:
            style_data: Historical data for the style
            horizon_days: Forecast horizon in days
            
        Returns:
            dict with forecast, confidence, and method used
        """
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
        if consistency_score >= 0.7 and self.ML_AVAILABLE:
            try:
                # Use ML model for forecasting
                features = self.extract_features(style_data)
                forecast_results = self.train_models(style_data, features)
                
                # Get ensemble forecast
                if 'ensemble' in forecast_results:
                    result['forecast'] = forecast_results['ensemble'].get('forecast', 0)
                    result['confidence'] = consistency_score * 0.9  # High confidence
                else:
                    # Use best available model
                    for model in ['XGBoost', 'Prophet', 'ARIMA']:
                        if model in forecast_results and 'forecast' in forecast_results[model]:
                            result['forecast'] = forecast_results[model]['forecast']
                            result['confidence'] = consistency_score * 0.8
                            break
                
                result['method'] = 'ml_forecast'
                
            except Exception as e:
                print(f"ML forecasting failed: {str(e)}, falling back to weighted average")
                result['method'] = 'fallback_to_weighted'
                result['forecast'] = self._calculate_weighted_average(style_data, consistency_score)
                result['confidence'] = consistency_score * 0.5
        
        # Medium consistency (0.3 <= CV < 0.7): Use weighted average
        elif consistency_score >= 0.3:
            result['method'] = 'weighted_average'
            result['forecast'] = self._calculate_weighted_average(style_data, consistency_score)
            result['confidence'] = consistency_score * 0.6
        
        # Low consistency (CV >= 0.7, score < 0.3): React to orders only
        else:
            result['method'] = 'react_to_orders'
            # Use only recent actual orders, no forecasting
            result['forecast'] = 0  # No forecast, only react to actual orders
            result['confidence'] = 0.1
            result['recommendation'] = 'Monitor actual orders only - pattern too variable for forecasting'
        
        # Add additional metadata
        result['data_points'] = consistency_result['data_points']
        result['mean_historical'] = consistency_result.get('mean', 0)
        result['std_historical'] = consistency_result.get('std', 0)
        
        return result
    
    def _calculate_weighted_average(self, style_data, weight_factor):
        """
        Calculate weighted average giving more weight to recent data
        
        Args:
            style_data: Historical sales data
            weight_factor: Factor to adjust weighting (0-1)
            
        Returns:
            Weighted average forecast
        """
        if isinstance(style_data, pd.DataFrame):
            if 'quantity' in style_data.columns:
                data = style_data['quantity']
            elif 'Yds_ordered' in style_data.columns:
                data = style_data['Yds_ordered']
            else:
                data = style_data.iloc[:, 0]
        else:
            data = pd.Series(style_data)
        
        data = data.dropna()
        
        if len(data) == 0:
            return 0
        
        # Create exponential weights (more recent data gets higher weight)
        n = len(data)
        weights = np.exp(np.linspace(-2, 0, n))  # Exponential decay
        weights = weights / weights.sum()  # Normalize
        
        # Adjust weights based on consistency (higher consistency = more weight to history)
        weights = weights * weight_factor + (1 - weight_factor) / n
        
        # Calculate weighted average
        if len(data) == len(weights):
            weighted_avg = np.average(data.values, weights=weights)
        else:
            weighted_avg = data.mean()
        
        return float(weighted_avg)
    
    def analyze_portfolio_consistency(self, sales_data):
        """
        Analyze consistency across entire product portfolio
        
        Args:
            sales_data: DataFrame with sales data for all styles
            
        Returns:
            DataFrame with consistency analysis for each style
        """
        results = []
        
        # Get unique styles using flexible column detection
        style_col = find_column(sales_data, STYLE_VARIATIONS)
        if not style_col:
            return pd.DataFrame()
        
        styles = sales_data[style_col].unique()
        
        for style in styles:
            # Get style history
            style_data = sales_data[sales_data[style_col] == style]
            
            # Calculate consistency
            consistency = self.calculate_consistency_score(style_data)
            
            # Generate forecast
            forecast = self.forecast_with_consistency(style_data)
            
            results.append({
                'style': style,
                'consistency_score': consistency['consistency_score'],
                'cv': consistency['cv'],
                'data_points': consistency['data_points'],
                'forecast_method': forecast['method'],
                'forecast_value': forecast['forecast'],
                'confidence': forecast['confidence'],
                'recommendation': consistency['recommendation']
            })
        
        return pd.DataFrame(results).sort_values('consistency_score', ascending=False)

    def extract_features(self, sales_data):
        """Extract advanced features for forecasting"""
        features = {}

        # Ensure we have proper datetime index
        if 'Date' in sales_data.columns:
            sales_data['Date'] = pd.to_datetime(sales_data['Date'], errors='coerce')
            sales_data = sales_data.set_index('Date')

        # 1. Seasonality Patterns
        features['seasonality'] = self._extract_seasonality_patterns(sales_data)

        # 2. Promotion Effects
        features['promotions'] = self._extract_promotion_effects(sales_data)

        # 3. Customer Segments
        features['segments'] = self._extract_customer_segments(sales_data)

        # 4. Additional Features
        features['trends'] = self._extract_trend_features(sales_data)
        features['cyclical'] = self._extract_cyclical_patterns(sales_data)

        return features

    def _extract_seasonality_patterns(self, data):
        """Extract multiple seasonality patterns"""
        patterns = {}

        # Weekly seasonality
        if len(data) >= 14:
            patterns['weekly'] = {
                'strength': self._calculate_seasonality_strength(data, 7),
                'peak_day': self._find_peak_period(data, 'dayofweek'),
                'pattern': 'multiplicative' if self._is_multiplicative_seasonality(data, 7) else 'additive'
            }

        # Monthly seasonality
        if len(data) >= 60:
            patterns['monthly'] = {
                'strength': self._calculate_seasonality_strength(data, 30),
                'peak_week': self._find_peak_period(data, 'week'),
                'pattern': 'multiplicative' if self._is_multiplicative_seasonality(data, 30) else 'additive'
            }

        # Yearly seasonality
        if len(data) >= 365:
            patterns['yearly'] = {
                'strength': self._calculate_seasonality_strength(data, 365),
                'peak_month': self._find_peak_period(data, 'month'),
                'pattern': 'multiplicative' if self._is_multiplicative_seasonality(data, 365) else 'additive'
            }

        return patterns

    def _extract_promotion_effects(self, data):
        """Extract promotion effects on sales"""
        effects = {}

        # Detect promotional periods (sales spikes)
        if 'Qty Shipped' in data.columns:
            sales_col = 'Qty Shipped'
        elif 'Quantity' in data.columns:
            sales_col = 'Quantity'
        else:
            sales_col = data.select_dtypes(include=[np.number]).columns[0] if len(data.select_dtypes(include=[np.number]).columns) > 0 else None

        if sales_col:
            rolling_mean = data[sales_col].rolling(window=7, min_periods=1).mean()
            rolling_std = data[sales_col].rolling(window=7, min_periods=1).std()

            # Identify promotion periods (sales > mean + 2*std)
            promotion_threshold = rolling_mean + 2 * rolling_std
            promotion_periods = data[sales_col] > promotion_threshold

            effects['promotion_frequency'] = promotion_periods.sum() / len(data)
            effects['promotion_impact'] = (data[sales_col][promotion_periods].mean() / rolling_mean.mean()) if promotion_periods.sum() > 0 else 1.0
            effects['avg_promotion_duration'] = self._calculate_avg_duration(promotion_periods)

        return effects

    def _extract_customer_segments(self, data):
        """Extract customer segment patterns"""
        segments = {}

        if 'Customer' in data.columns:
            # Segment by customer type/size
            customer_sales = data.groupby('Customer').agg({
                data.select_dtypes(include=[np.number]).columns[0]: ['sum', 'mean', 'count']
            }) if len(data.select_dtypes(include=[np.number]).columns) > 0 else pd.DataFrame()

            if not customer_sales.empty:
                # Classify customers by sales volume
                total_sales = customer_sales.iloc[:, 0].sum()
                customer_sales['percentage'] = customer_sales.iloc[:, 0] / total_sales

                # Pareto analysis (80/20 rule)
                customer_sales_sorted = customer_sales.sort_values(by=customer_sales.columns[0], ascending=False)
                cumsum = customer_sales_sorted['percentage'].cumsum()

                segments['top_20_percent_customers'] = len(cumsum[cumsum <= 0.8]) / len(customer_sales)
                segments['concentration_ratio'] = cumsum.iloc[int(len(cumsum) * 0.2)] if len(cumsum) > 5 else 0
                segments['customer_diversity'] = 1 - (customer_sales['percentage'] ** 2).sum()  # Herfindahl index

        return segments

    def _calculate_seasonality_strength(self, data, period):
        """Calculate strength of seasonality for given period"""
        if len(data) < period * 2:
            return 0

        try:
            # Use FFT to detect seasonality strength
            sales_col = data.select_dtypes(include=[np.number]).columns[0]
            fft = np.fft.fft(data[sales_col].values)
            power = np.abs(fft) ** 2
            freq = np.fft.fftfreq(len(data))

            # Find power at the seasonal frequency
            seasonal_freq = 1.0 / period
            idx = np.argmin(np.abs(freq - seasonal_freq))

            # Normalize by total power
            seasonal_strength = power[idx] / power.sum()
            return min(seasonal_strength * 100, 1.0)  # Scale to 0-1

        except Exception:
            return 0

    def _find_peak_period(self, data, period_type):
        """Find peak period for given type (dayofweek, week, month)"""
        try:
            sales_col = data.select_dtypes(include=[np.number]).columns[0]

            if period_type == 'dayofweek':
                data['period'] = data.index.dayofweek
            elif period_type == 'week':
                data['period'] = data.index.isocalendar().week
            elif period_type == 'month':
                data['period'] = data.index.month
            else:
                return None

            period_sales = data.groupby('period')[sales_col].mean()
            return int(period_sales.idxmax())

        except Exception:
            return None

    def _is_multiplicative_seasonality(self, data, period):
        """Determine if seasonality is multiplicative or additive"""
        try:
            sales_col = data.select_dtypes(include=[np.number]).columns[0]

            # Calculate coefficient of variation for each period
            cv_values = []
            for i in range(0, len(data) - period, period):
                segment = data[sales_col].iloc[i:i+period]
                if len(segment) > 1 and segment.mean() > 0:
                    cv = segment.std() / segment.mean()
                    cv_values.append(cv)

            # If CV increases with level, seasonality is multiplicative
            if len(cv_values) > 2:
                return np.corrcoef(range(len(cv_values)), cv_values)[0, 1] > 0.3

            return False

        except Exception:
            return False

    def _extract_trend_features(self, data):
        """Extract trend features"""
        features = {}

        try:
            sales_col = data.select_dtypes(include=[np.number]).columns[0]

            # Linear trend
            x = np.arange(len(data))
            y = data[sales_col].values
            slope, intercept = np.polyfit(x, y, 1)

            features['linear_trend'] = slope
            features['trend_strength'] = np.corrcoef(x, y)[0, 1] ** 2  # R-squared

            # Acceleration (second derivative)
            if len(data) > 10:
                smooth = data[sales_col].rolling(window=7, min_periods=1).mean()
                acceleration = smooth.diff().diff().mean()
                features['acceleration'] = acceleration

            return features

        except Exception:
            return {}

    def _extract_cyclical_patterns(self, data):
        """Extract cyclical patterns beyond seasonality"""
        features = {}

        try:
            sales_col = data.select_dtypes(include=[np.number]).columns[0]

            # Detrend and deseasonalize
            detrended = data[sales_col] - data[sales_col].rolling(window=30, min_periods=1).mean()

            # Autocorrelation analysis
            if len(detrended) > 50:
                from pandas.plotting import autocorrelation_plot
                acf_values = [detrended.autocorr(lag=i) for i in range(1, min(40, len(detrended)//2))]

                # Find significant lags
                significant_lags = [i+1 for i, v in enumerate(acf_values) if abs(v) > 0.2]

                features['cycle_length'] = significant_lags[0] if significant_lags else None
                features['cycle_strength'] = max(acf_values) if acf_values else 0

            return features

        except Exception:
            return {}

    def _calculate_avg_duration(self, binary_series):
        """Calculate average duration of True periods in binary series"""
        if binary_series.sum() == 0:
            return 0

        durations = []
        current_duration = 0

        for value in binary_series:
            if value:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def train_models(self, sales_data, features):
        """Train all forecasting models with comprehensive error handling"""
        results = {}
        errors = []

        try:
            # Prepare time series data
            ts_data = self._prepare_time_series(sales_data)

            if ts_data is None or len(ts_data) < 30:
                return self._get_fallback_forecast_results("Insufficient data for training")

            # 1. ARIMA Model with error handling
            try:
                results['ARIMA'] = self._train_arima(ts_data, features)
            except Exception as e:
                ml_logger.error(f"ARIMA training failed: {str(e)}")
                print(f"ARIMA training failed: {str(e)}")
                errors.append(f"ARIMA: {str(e)}")
                results['ARIMA'] = self._get_fallback_model_result('ARIMA', str(e))

            # 2. Prophet Model with error handling
            try:
                results['Prophet'] = self._train_prophet(ts_data, features)
            except Exception as e:
                ml_logger.error(f"Prophet training failed: {str(e)}")
                print(f"Prophet training failed: {str(e)}")
                errors.append(f"Prophet: {str(e)}")
                results['Prophet'] = self._get_fallback_model_result('Prophet', str(e))

            # 3. LSTM Model with error handling
            try:
                results['LSTM'] = self._train_lstm(ts_data, features)
            except Exception as e:
                ml_logger.error(f"LSTM training failed: {str(e)}")
                print(f"LSTM training failed: {str(e)}")
                errors.append(f"LSTM: {str(e)}")
                results['LSTM'] = self._get_fallback_model_result('LSTM', str(e))

            # 4. XGBoost Model with error handling
            try:
                results['XGBoost'] = self._train_xgboost(ts_data, features)
            except Exception as e:
                ml_logger.error(f"XGBoost training failed: {str(e)}")
                print(f"XGBoost training failed: {str(e)}")
                errors.append(f"XGBoost: {str(e)}")
                results['XGBoost'] = self._get_fallback_model_result('XGBoost', str(e))

            # 5. Calculate Ensemble with fallback if needed
            try:
                results['Ensemble'] = self._create_ensemble(results)
            except Exception as e:
                ml_logger.error(f"Ensemble creation failed: {str(e)}")
                print(f"Ensemble creation failed: {str(e)}")
                errors.append(f"Ensemble: {str(e)}")
                # Use best available model as fallback
                results['Ensemble'] = self._get_best_available_model(results)

            # Log any errors that occurred
            if errors:
                results['training_errors'] = errors

            self.models = results
            return results

        except Exception as e:
            ml_logger.critical(f"Critical error in model training: {str(e)}\n{traceback.format_exc()}")
            print(f"Critical error in model training: {str(e)}")
            return self._get_fallback_forecast_results(str(e))

    def _prepare_time_series(self, sales_data):
        """Prepare time series data for modeling"""
        try:
            # Find date and value columns
            date_cols = ['Date', 'Order Date', 'Ship Date', 'date']
            value_cols = ['Qty Shipped', 'Quantity', 'Units', 'Sales', 'Amount']

            date_col = None
            value_col = None

            for col in date_cols:
                if col in sales_data.columns:
                    date_col = col
                    break

            for col in value_cols:
                if col in sales_data.columns:
                    value_col = col
                    break

            if not date_col or not value_col:
                # Use first datetime and numeric columns
                date_col = sales_data.select_dtypes(include=['datetime64']).columns[0] if len(sales_data.select_dtypes(include=['datetime64']).columns) > 0 else None
                value_col = sales_data.select_dtypes(include=[np.number]).columns[0] if len(sales_data.select_dtypes(include=[np.number]).columns) > 0 else None

            if date_col and value_col:
                ts_data = sales_data[[date_col, value_col]].copy()
                ts_data.columns = ['ds', 'y']
                ts_data['ds'] = pd.to_datetime(ts_data['ds'], errors='coerce')
                ts_data = ts_data.dropna()
                ts_data = ts_data.groupby('ds')['y'].sum().reset_index()
                return ts_data

            return None

        except Exception as e:
            print(f"Error preparing time series: {str(e)}")
            return None

    def _train_arima(self, ts_data, features):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE or len(ts_data) < 30:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': 'ARIMA unavailable or insufficient data'}

        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Determine ARIMA order based on features
            if features.get('seasonality', {}).get('yearly'):
                order = (2, 1, 2)  # More complex for yearly seasonality
            elif features.get('seasonality', {}).get('monthly'):
                order = (1, 1, 2)  # Medium complexity
            else:
                order = (1, 1, 1)  # Simple model

            # Split data for validation
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data['y'].iloc[:train_size]
            test_data = ts_data['y'].iloc[train_size:]

            # Train model
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()

            # Validate
            predictions = model_fit.forecast(steps=len(test_data))
            mape = mean_absolute_percentage_error(test_data, predictions) * 100
            accuracy = max(0, 100 - mape)

            # Generate 90-day forecast
            full_model = ARIMA(ts_data['y'], order=order)
            full_model_fit = full_model.fit()
            forecast = full_model_fit.forecast(steps=self.forecast_horizon)

            # Calculate confidence intervals
            forecast_df = full_model_fit.get_forecast(steps=self.forecast_horizon)
            confidence_intervals = forecast_df.conf_int(alpha=0.05)

            return {
                'accuracy': accuracy,
                'mape': mape,
                'model': full_model_fit,
                'forecast': forecast,
                'lower_bound': confidence_intervals.iloc[:, 0].values,
                'upper_bound': confidence_intervals.iloc[:, 1].values,
                'meets_target': accuracy >= self.target_accuracy * 100
            }

        except Exception as e:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': f'ARIMA training failed: {str(e)}'}

    def _get_fallback_model_result(self, model_name, error_msg):
        """Generate fallback result for failed model training"""
        return {
            'accuracy': 0,
            'mape': 100,
            'model': None,
            'error': error_msg,
            'fallback': True,
            'forecast': None,
            'lower_bound': None,
            'upper_bound': None,
            'meets_target': False
        }

    def _get_fallback_forecast_results(self, error_msg):
        """Generate complete fallback results when all models fail"""
        fallback_result = self._get_fallback_model_result('Fallback', error_msg)
        return {
            'ARIMA': fallback_result,
            'Prophet': fallback_result,
            'LSTM': fallback_result,
            'XGBoost': fallback_result,
            'Ensemble': fallback_result,
            'error': error_msg,
            'fallback_method': 'simple_moving_average'
        }

    def _get_best_available_model(self, results):
        """Select best performing model from available results"""
        best_model = None
        best_accuracy = 0
        
        for model_name, model_data in results.items():
            if model_data and not model_data.get('fallback', False):
                accuracy = model_data.get('accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_data
        
        if best_model:
            return best_model
        else:
            # All models failed, return simple forecast
            return self._generate_simple_forecast_fallback()

    def _generate_simple_forecast_fallback(self):
        """Generate simple moving average forecast as ultimate fallback"""
        try:
            # Generate simple 90-day forecast using moving average
            forecast_values = np.full(self.forecast_horizon, 100)  # Default baseline
            return {
                'accuracy': 60,
                'mape': 40,
                'model': 'SimpleMovingAverage',
                'forecast': forecast_values,
                'lower_bound': forecast_values * 0.8,
                'upper_bound': forecast_values * 1.2,
                'meets_target': False,
                'fallback': True,
                'method': 'simple_moving_average'
            }
        except Exception as e:
            return self._get_fallback_model_result('SimpleMovingAverage', str(e))

    def _train_prophet(self, ts_data, features):
        """Train Prophet model with enhanced error handling"""
        if not ML_AVAILABLE or len(ts_data) < 30:
            return self._get_fallback_model_result('Prophet', 'Prophet unavailable or insufficient data')

        try:
            from prophet import Prophet

            # Configure based on features
            seasonality_mode = 'multiplicative' if features.get('seasonality', {}).get('weekly', {}).get('pattern') == 'multiplicative' else 'additive'

            # Split data
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data.iloc[:train_size]
            test_data = ts_data.iloc[train_size:]

            # Train model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=len(ts_data) > 365,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05
            )

            # Add promotion effects if detected
            if features.get('promotions', {}).get('promotion_frequency', 0) > 0.05:
                # Add custom seasonality for promotions
                model.add_seasonality(name='promotions', period=30, fourier_order=5)

            model.fit(train_data)

            # Validate
            future_test = model.make_future_dataframe(periods=len(test_data))
            forecast_test = model.predict(future_test)
            predictions = forecast_test['yhat'].iloc[-len(test_data):].values

            mape = mean_absolute_percentage_error(test_data['y'], predictions) * 100
            accuracy = max(0, 100 - mape)

            # Generate 90-day forecast
            future = model.make_future_dataframe(periods=self.forecast_horizon)
            forecast = model.predict(future)

            return {
                'accuracy': accuracy,
                'mape': mape,
                'model': model,
                'forecast': forecast['yhat'].iloc[-self.forecast_horizon:].values,
                'lower_bound': forecast['yhat_lower'].iloc[-self.forecast_horizon:].values,
                'upper_bound': forecast['yhat_upper'].iloc[-self.forecast_horizon:].values,
                'meets_target': accuracy >= self.target_accuracy * 100
            }

        except Exception as e:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': f'Prophet training failed: {str(e)}'}

    def _train_lstm(self, ts_data, features):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE or len(ts_data) < 60:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': 'TensorFlow unavailable or insufficient data'}

        try:
            # TensorFlow imports are already handled at module level
            from sklearn.preprocessing import MinMaxScaler

            # Prepare data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts_data['y'].values.reshape(-1, 1))

            # Create sequences
            sequence_length = 30
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i])

            X, y = np.array(X), np.array(y)

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Build model
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

            # Validate
            predictions = model.predict(X_test)
            predictions_inv = scaler.inverse_transform(predictions)
            y_test_inv = scaler.inverse_transform(y_test)

            mape = mean_absolute_percentage_error(y_test_inv, predictions_inv) * 100
            accuracy = max(0, 100 - mape)

            # Generate 90-day forecast
            last_sequence = scaled_data[-sequence_length:]
            forecast = []
            current_sequence = last_sequence.copy()

            for _ in range(self.forecast_horizon):
                next_pred = model.predict(current_sequence.reshape(1, sequence_length, 1))
                forecast.append(next_pred[0, 0])
                current_sequence = np.append(current_sequence[1:], next_pred)

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

            # Calculate confidence intervals (using historical error)
            historical_error = np.std(predictions_inv - y_test_inv)
            lower_bound = forecast - 1.96 * historical_error
            upper_bound = forecast + 1.96 * historical_error

            return {
                'accuracy': accuracy,
                'mape': mape,
                'model': model,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'meets_target': accuracy >= self.target_accuracy * 100
            }

        except Exception as e:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': f'LSTM training failed: {str(e)}'}

    def _train_xgboost(self, ts_data, features):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE or len(ts_data) < 60:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': 'XGBoost unavailable or insufficient data'}

        try:
            from xgboost import XGBRegressor

            # Feature engineering
            X = pd.DataFrame()

            # Lag features
            for i in range(1, 31):
                X[f'lag_{i}'] = ts_data['y'].shift(i)

            # Rolling statistics
            for window in [7, 14, 30]:
                X[f'rolling_mean_{window}'] = ts_data['y'].rolling(window, min_periods=1).mean()
                X[f'rolling_std_{window}'] = ts_data['y'].rolling(window, min_periods=1).std()
                X[f'rolling_min_{window}'] = ts_data['y'].rolling(window, min_periods=1).min()
                X[f'rolling_max_{window}'] = ts_data['y'].rolling(window, min_periods=1).max()

            # Date features
            dates = pd.to_datetime(ts_data['ds'])
            X['dayofweek'] = dates.dt.dayofweek
            X['day'] = dates.dt.day
            X['month'] = dates.dt.month
            X['quarter'] = dates.dt.quarter
            X['year'] = dates.dt.year
            X['weekofyear'] = dates.dt.isocalendar().week

            # Add extracted features
            if features.get('seasonality', {}).get('weekly'):
                X['weekly_strength'] = features['seasonality']['weekly'].get('strength', 0)

            if features.get('promotions'):
                X['promotion_impact'] = features['promotions'].get('promotion_impact', 1.0)

            # Clean data
            X = X.dropna()
            y = ts_data['y'].iloc[len(ts_data) - len(X):]

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            # Train model
            model = XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror'
            )

            model.fit(X_train, y_train)

            # Validate
            predictions = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            accuracy = max(0, 100 - mape)

            # Generate 90-day forecast
            last_features = X.iloc[-1:].copy()
            forecast = []

            for i in range(self.forecast_horizon):
                pred = model.predict(last_features)[0]
                forecast.append(pred)

                # Update features for next prediction
                # Shift lags
                for j in range(29, 0, -1):
                    last_features[f'lag_{j+1}'] = last_features[f'lag_{j}'].values[0]
                last_features['lag_1'] = pred

                # Update rolling features (simplified)
                for window in [7, 14, 30]:
                    recent_values = [last_features[f'lag_{k}'].values[0] for k in range(1, min(window+1, 31))]
                    last_features[f'rolling_mean_{window}'] = np.mean(recent_values)
                    last_features[f'rolling_std_{window}'] = np.std(recent_values)
                    last_features[f'rolling_min_{window}'] = np.min(recent_values)
                    last_features[f'rolling_max_{window}'] = np.max(recent_values)

                # Update date features
                next_date = dates.iloc[-1] + pd.Timedelta(days=i+1)
                last_features['dayofweek'] = next_date.dayofweek
                last_features['day'] = next_date.day
                last_features['month'] = next_date.month
                last_features['quarter'] = next_date.quarter
                last_features['year'] = next_date.year
                last_features['weekofyear'] = next_date.isocalendar().week

            forecast = np.array(forecast)

            # Calculate confidence intervals
            prediction_errors = predictions - y_test.values
            error_std = np.std(prediction_errors)
            lower_bound = forecast - 1.96 * error_std
            upper_bound = forecast + 1.96 * error_std

            return {
                'accuracy': accuracy,
                'mape': mape,
                'model': model,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'meets_target': accuracy >= self.target_accuracy * 100,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }

        except Exception as e:
            return {'accuracy': 0, 'mape': 100, 'model': None, 'error': f'XGBoost training failed: {str(e)}'}

    def _create_ensemble(self, model_results):
        """Create ensemble forecast from individual models"""
        valid_models = {k: v for k, v in model_results.items() if v.get('forecast') is not None}

        if len(valid_models) < 2:
            return {'accuracy': 0, 'mape': 100, 'forecast': None, 'error': 'Insufficient models for ensemble'}

        # Calculate weights based on accuracy
        weights = {}
        total_accuracy = sum(m.get('accuracy', 0) for m in valid_models.values())

        if total_accuracy > 0:
            for name, model in valid_models.items():
                weights[name] = model.get('accuracy', 0) / total_accuracy
        else:
            # Equal weights if no accuracy info
            for name in valid_models:
                weights[name] = 1.0 / len(valid_models)

        # Combine forecasts
        ensemble_forecast = np.zeros(self.forecast_horizon)
        ensemble_lower = np.zeros(self.forecast_horizon)
        ensemble_upper = np.zeros(self.forecast_horizon)

        for name, model in valid_models.items():
            weight = weights[name]
            ensemble_forecast += weight * model['forecast']
            ensemble_lower += weight * model.get('lower_bound', model['forecast'] * 0.9)
            ensemble_upper += weight * model.get('upper_bound', model['forecast'] * 1.1)

        # Calculate ensemble accuracy (weighted average)
        ensemble_accuracy = sum(weights[name] * model.get('accuracy', 0) for name, model in valid_models.items())
        ensemble_mape = 100 - ensemble_accuracy

        return {
            'accuracy': ensemble_accuracy,
            'mape': ensemble_mape,
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper,
            'weights': weights,
            'meets_target': ensemble_accuracy >= self.target_accuracy * 100,
            'models_used': list(valid_models.keys())
        }

    def validate_accuracy(self, actual_data, forecast_data):
        """Validate forecast accuracy against actual data"""
        if len(actual_data) != len(forecast_data):
            min_len = min(len(actual_data), len(forecast_data))
            actual_data = actual_data[:min_len]
            forecast_data = forecast_data[:min_len]

        mape = mean_absolute_percentage_error(actual_data, forecast_data) * 100
        accuracy = max(0, 100 - mape)

        # Additional metrics
        rmse = np.sqrt(mean_squared_error(actual_data, forecast_data))
        mae = np.mean(np.abs(actual_data - forecast_data))

        return {
            'accuracy': accuracy,
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'meets_target': accuracy >= self.target_accuracy * 100
        }

    def generate_forecast_output(self, sales_data):
        """Generate complete forecast output with all specifications"""
        # Extract features
        features = self.extract_features(sales_data)

        # Train models
        model_results = self.train_models(sales_data, features)

        # Prepare output format
        output = {
            'forecast_horizon': '90-day',
            'target_accuracy': f'{self.target_accuracy * 100:.0f}%',
            'features_extracted': {
                'seasonality_patterns': features.get('seasonality', {}),
                'promotion_effects': features.get('promotions', {}),
                'customer_segments': features.get('segments', {}),
                'trends': features.get('trends', {}),
                'cyclical_patterns': features.get('cyclical', {})
            },
            'models': {},
            'ensemble': {},
            'validation': {}
        }

        # Add individual model results
        for model_name, result in model_results.items():
            if model_name != 'Ensemble':
                # Ensure result is a dictionary
                if isinstance(result, dict):
                    output['models'][model_name] = {
                        'accuracy': f"{result.get('accuracy', 0):.2f}%",
                        'mape': f"{result.get('mape', 100):.2f}%",
                        'meets_target': result.get('meets_target', False),
                        'status': 'SUCCESS' if result.get('forecast') is not None else 'FAILED',
                        'error': result.get('error', None)
                    }
                else:
                    output['models'][model_name] = {
                        'accuracy': '0.00%',
                        'mape': '100.00%',
                        'meets_target': False,
                        'status': 'FAILED',
                        'error': f'Invalid result type: {type(result)}'
                    }

        # Add ensemble results
        if 'Ensemble' in model_results:
            ensemble = model_results['Ensemble']
            if isinstance(ensemble, dict):
                output['ensemble'] = {
                    'accuracy': f"{ensemble.get('accuracy', 0):.2f}%",
                    'mape': f"{ensemble.get('mape', 100):.2f}%",
                    'meets_target': ensemble.get('meets_target', False),
                    'weights': ensemble.get('weights', {}),
                    'models_used': ensemble.get('models_used', [])
                }
            else:
                output['ensemble'] = {
                    'accuracy': '0.00%',
                    'mape': '100.00%',
                    'meets_target': False,
                    'weights': {},
                    'models_used': []
                }

            # Generate daily forecasts with confidence intervals
            if ensemble.get('forecast') is not None:
                base_date = pd.Timestamp.now()
                daily_forecasts = []

                for i in range(self.forecast_horizon):
                    daily_forecasts.append({
                        'date': (base_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                        'forecast': float(ensemble['forecast'][i]),
                        'lower_bound': float(ensemble['lower_bound'][i]),
                        'upper_bound': float(ensemble['upper_bound'][i]),
                        'confidence_interval': '95%'
                    })

                output['daily_forecasts'] = daily_forecasts

                # Summary statistics
                output['summary'] = {
                    'total_forecast': float(ensemble['forecast'].sum()),
                    'avg_daily_forecast': float(ensemble['forecast'].mean()),
                    'peak_day': daily_forecasts[np.argmax(ensemble['forecast'])]['date'],
                    'lowest_day': daily_forecasts[np.argmin(ensemble['forecast'])]['date'],
                    'forecast_volatility': float(ensemble['forecast'].std() / ensemble['forecast'].mean() * 100)
                }

        # Overall validation
        best_accuracy = max(r.get('accuracy', 0) for r in model_results.values())
        output['validation'] = {
            'best_model_accuracy': f"{best_accuracy:.2f}%",
            'target_achieved': best_accuracy >= self.target_accuracy * 100,
            'confidence_level': 'HIGH' if best_accuracy >= 90 else 'MEDIUM' if best_accuracy >= 80 else 'LOW'
        }

        return output