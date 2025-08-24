#!/usr/bin/env python3
"""
ML Model Validation and Comparison System for Beverly Knits ERP
Comprehensive testing and validation of all ML models used in forecasting,
inventory optimization, and production planning with cross-validation and ensemble methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import sqlite3

warnings.filterwarnings('ignore')

# ML libraries with fallback
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelValidationResult:
    """Results from ML model validation"""
    model_name: str
    model_type: str
    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    error_analysis: Optional[Dict[str, Any]] = None


class BaseMLModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create the ML model instance"""
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseMLModel':
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None


class SklearnModel(BaseMLModel):
    """Wrapper for sklearn models"""
    
    def __init__(self, name: str, model_class, model_params: Dict = None, use_scaler: bool = True):
        super().__init__(name, "sklearn")
        self.model_class = model_class
        self.model_params = model_params or {}
        self.use_scaler = use_scaler
        
    def create_model(self, **kwargs):
        params = {**self.model_params, **kwargs}
        return self.model_class(**params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
            
        self.model = self.create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.use_scaler and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(range(len(self.model.feature_importances_)), 
                           self.model.feature_importances_))
        return None


class XGBoostModel(BaseMLModel):
    """Wrapper for XGBoost models"""
    
    def __init__(self, name: str, model_params: Dict = None):
        super().__init__(name, "xgboost")
        self.model_params = model_params or {}
        
    def create_model(self, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        params = {**self.model_params, **kwargs}
        return XGBRegressor(**params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(range(len(self.model.feature_importances_)), 
                           self.model.feature_importances_))
        return None


class ProphetModel(BaseMLModel):
    """Wrapper for Prophet time series model"""
    
    def __init__(self, name: str, model_params: Dict = None):
        super().__init__(name, "prophet")
        self.model_params = model_params or {}
        
    def create_model(self, **kwargs):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        params = {**self.model_params, **kwargs}
        return Prophet(**params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Prophet expects specific column names
        df = pd.DataFrame({
            'ds': X.index if isinstance(X.index, pd.DatetimeIndex) else pd.date_range(start='2025-01-01', periods=len(X)),
            'y': y.values
        })
        
        self.model = self.create_model()
        self.model.fit(df)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = pd.DataFrame({
            'ds': X.index if isinstance(X.index, pd.DatetimeIndex) else pd.date_range(start='2025-01-01', periods=len(X))
        })
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values


class LSTMModel(BaseMLModel):
    """Wrapper for LSTM neural network"""
    
    def __init__(self, name: str, sequence_length: int = 30, model_params: Dict = None):
        super().__init__(name, "lstm")
        self.sequence_length = sequence_length
        self.model_params = model_params or {}
        
    def create_model(self, input_shape, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _prepare_sequences(self, X: pd.DataFrame, y: pd.Series = None):
        """Prepare data sequences for LSTM"""
        sequences_X = []
        sequences_y = []
        
        data = X.values
        
        for i in range(self.sequence_length, len(data)):
            sequences_X.append(data[i-self.sequence_length:i])
            if y is not None:
                sequences_y.append(y.iloc[i])
        
        if y is not None:
            return np.array(sequences_X), np.array(sequences_y)
        return np.array(sequences_X)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length}")
        
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        self.scaler = MinMaxScaler()
        X_seq_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        
        self.model = self.create_model((self.sequence_length, X.shape[1]))
        
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.model.fit(X_seq_scaled, y_seq, epochs=100, batch_size=32, 
                      callbacks=[early_stopping], verbose=0)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(X) < self.sequence_length:
            # Pad with last available values
            padding_needed = self.sequence_length - len(X)
            last_row = X.iloc[-1:].copy()
            padding = pd.concat([last_row] * padding_needed, ignore_index=True)
            X_padded = pd.concat([padding, X], ignore_index=True)
        else:
            X_padded = X
        
        X_seq = self._prepare_sequences(X_padded)
        X_seq_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        
        predictions = self.model.predict(X_seq_scaled, verbose=0)
        return predictions.flatten()


class MLValidationSystem:
    """Comprehensive ML model validation system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = []
        self.results = {}
        self.ensemble_results = {}
        
        # Initialize database
        self.db_path = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp/ml_validation_results.db"
        self._init_database()
        
        # Initialize available models
        self._initialize_models()
        
    def _default_config(self) -> Dict:
        """Default configuration for ML validation"""
        return {
            "validation": {
                "cv_folds": 5,
                "test_size": 0.2,
                "random_state": 42,
                "scoring": "neg_mean_absolute_error"
            },
            "models": {
                "linear_regression": True,
                "random_forest": SKLEARN_AVAILABLE,
                "gradient_boosting": SKLEARN_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "prophet": PROPHET_AVAILABLE,
                "lstm": TENSORFLOW_AVAILABLE
            },
            "ensemble": {
                "enabled": True,
                "methods": ["simple_average", "weighted_average", "stacking"]
            }
        }
    
    def _init_database(self):
        """Initialize database for results storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    model_type TEXT,
                    dataset_name TEXT,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    mape REAL,
                    r2 REAL,
                    cv_mean REAL,
                    cv_std REAL,
                    training_time REAL,
                    prediction_time REAL,
                    hyperparameters TEXT,
                    feature_importance TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("ML validation database initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _initialize_models(self):
        """Initialize all available ML models"""
        self.models = []
        
        if SKLEARN_AVAILABLE:
            # Linear models
            if self.config["models"]["linear_regression"]:
                self.models.extend([
                    SklearnModel("Linear Regression", LinearRegression),
                    SklearnModel("Ridge Regression", Ridge, {"alpha": 1.0}),
                    SklearnModel("Lasso Regression", Lasso, {"alpha": 1.0}),
                    SklearnModel("Elastic Net", ElasticNet, {"alpha": 1.0, "l1_ratio": 0.5})
                ])
            
            # Tree-based models
            if self.config["models"]["random_forest"]:
                self.models.extend([
                    SklearnModel("Random Forest", RandomForestRegressor, 
                               {"n_estimators": 100, "random_state": 42}, use_scaler=False),
                    SklearnModel("Extra Trees", ExtraTreesRegressor, 
                               {"n_estimators": 100, "random_state": 42}, use_scaler=False)
                ])
            
            if self.config["models"]["gradient_boosting"]:
                self.models.append(
                    SklearnModel("Gradient Boosting", GradientBoostingRegressor, 
                               {"n_estimators": 100, "random_state": 42}, use_scaler=False)
                )
        
        # XGBoost
        if XGBOOST_AVAILABLE and self.config["models"]["xgboost"]:
            self.models.extend([
                XGBoostModel("XGBoost", {"n_estimators": 100, "random_state": 42}),
                XGBoostModel("XGBoost Tuned", {
                    "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, 
                    "subsample": 0.8, "random_state": 42
                })
            ])
        
        # Prophet
        if PROPHET_AVAILABLE and self.config["models"]["prophet"]:
            self.models.extend([
                ProphetModel("Prophet", {"yearly_seasonality": True, "weekly_seasonality": True}),
                ProphetModel("Prophet Simple", {"yearly_seasonality": False, "weekly_seasonality": False})
            ])
        
        # LSTM
        if TENSORFLOW_AVAILABLE and self.config["models"]["lstm"]:
            self.models.extend([
                LSTMModel("LSTM", sequence_length=30),
                LSTMModel("LSTM Short", sequence_length=10)
            ])
        
        logger.info(f"Initialized {len(self.models)} ML models for validation")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from data"""
        # Find target column
        target_cols = ['Ordered', 'Qty Shipped', 'Quantity', 'Balance', 'Demand', 'Sales']
        target_col = next((col for col in target_cols if col in data.columns), None)
        
        if not target_col:
            raise ValueError("No target column found in data")
        
        # Create features
        features_df = pd.DataFrame()
        
        # Date-based features if date column exists
        date_cols = ['Date', 'Order Date', 'Ship Date']
        date_col = next((col for col in date_cols if col in data.columns), None)
        
        if date_col:
            data[date_col] = pd.to_datetime(data[date_col])
            features_df['day_of_year'] = data[date_col].dt.dayofyear
            features_df['month'] = data[date_col].dt.month
            features_df['day_of_week'] = data[date_col].dt.dayofweek
            features_df['week_of_year'] = data[date_col].dt.isocalendar().week
        
        # Lag features
        if target_col in data.columns:
            target_series = data[target_col].fillna(0)
            for lag in [1, 2, 3, 7, 14, 30]:
                if len(target_series) > lag:
                    features_df[f'{target_col}_lag_{lag}'] = target_series.shift(lag)
        
        # Rolling statistics
        if target_col in data.columns and len(data) > 7:
            features_df[f'{target_col}_rolling_7_mean'] = data[target_col].rolling(7).mean()
            features_df[f'{target_col}_rolling_7_std'] = data[target_col].rolling(7).std()
            features_df[f'{target_col}_rolling_30_mean'] = data[target_col].rolling(30).mean()
        
        # Numeric columns as features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col and col not in features_df.columns:
                features_df[col] = data[col].fillna(0)
        
        # Add trend feature
        if len(features_df) > 0:
            features_df['trend'] = np.arange(len(features_df))
        
        # Remove rows with NaN values
        target = data[target_col].fillna(0)
        
        # Align features and target
        common_index = features_df.dropna().index.intersection(target.index)
        if len(common_index) == 0:
            # If no common index, create simple features
            features_df = pd.DataFrame({'trend': np.arange(len(target))}, index=target.index)
            common_index = features_df.index
        
        final_features = features_df.loc[common_index]
        final_target = target.loc[common_index]
        
        if len(final_features) < 10:
            raise ValueError(f"Insufficient data after preprocessing: {len(final_features)} samples")
        
        return final_features, final_target
    
    def validate_model(self, model: BaseMLModel, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> ModelValidationResult:
        """Validate a single model"""
        try:
            start_time = datetime.now()
            
            # Split data
            split_point = int(len(X) * (1 - self.config["validation"]["test_size"]))
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Train model
            train_start = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - train_start).total_seconds()
            
            # Make predictions
            pred_start = datetime.now()
            y_pred = model.predict(X_test)
            prediction_time = (datetime.now() - pred_start).total_seconds()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation (if enough data and not LSTM/Prophet)
            cv_scores = []
            cv_mean = 0
            cv_std = 0
            
            if len(X) > 50 and model.model_type not in ["lstm", "prophet"]:
                try:
                    tscv = TimeSeriesSplit(n_splits=min(self.config["validation"]["cv_folds"], len(X) // 20))
                    
                    for train_idx, val_idx in tscv.split(X):
                        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Create fresh model instance
                        cv_model = model.__class__(model.name, **getattr(model, 'model_params', {}))
                        cv_model.fit(X_cv_train, y_cv_train)
                        cv_pred = cv_model.predict(X_cv_val)
                        cv_score = mean_absolute_error(y_cv_val, cv_pred)
                        cv_scores.append(cv_score)
                    
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {model.name}: {e}")
            
            # Feature importance
            feature_importance = model.get_feature_importance()
            
            # Error analysis
            errors = y_test - y_pred
            error_analysis = {
                "mean_error": float(np.mean(errors)),
                "error_std": float(np.std(errors)),
                "max_error": float(np.max(np.abs(errors))),
                "error_skewness": float(errors.skew()) if hasattr(errors, 'skew') else 0
            }
            
            result = ModelValidationResult(
                model_name=model.name,
                model_type=model.model_type,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                r2=r2,
                cv_scores=cv_scores,
                cv_mean=cv_mean,
                cv_std=cv_std,
                training_time=training_time,
                prediction_time=prediction_time,
                feature_importance=feature_importance,
                hyperparameters=getattr(model, 'model_params', {}),
                error_analysis=error_analysis
            )
            
            # Save to database
            self._save_result_to_db(result, dataset_name)
            
            logger.info(f"Validated {model.name} - MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error validating {model.name}: {e}")
            return ModelValidationResult(
                model_name=model.name,
                model_type=model.model_type,
                mae=float('inf'), mse=float('inf'), rmse=float('inf'),
                mape=float('inf'), r2=-float('inf'),
                cv_scores=[], cv_mean=0, cv_std=0,
                training_time=0, prediction_time=0
            )
    
    def _save_result_to_db(self, result: ModelValidationResult, dataset_name: str):
        """Save validation result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_validation_results 
                (model_name, model_type, dataset_name, mae, mse, rmse, mape, r2,
                 cv_mean, cv_std, training_time, prediction_time, hyperparameters, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.model_name, result.model_type, dataset_name,
                result.mae, result.mse, result.rmse, result.mape, result.r2,
                result.cv_mean, result.cv_std, result.training_time, result.prediction_time,
                json.dumps(result.hyperparameters), json.dumps(result.feature_importance)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def validate_all_models(self, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, ModelValidationResult]]:
        """Validate all models on all datasets"""
        logger.info(f"Starting validation of {len(self.models)} models on {len(data_sources)} datasets")
        
        results = {}
        
        for dataset_name, data in data_sources.items():
            logger.info(f"Validating on dataset: {dataset_name}")
            dataset_results = {}
            
            try:
                # Prepare features
                X, y = self.prepare_features(data)
                
                for model in self.models:
                    result = self.validate_model(model, X, y, dataset_name)
                    dataset_results[model.name] = result
                    
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
            
            results[dataset_name] = dataset_results
        
        self.results = results
        return results
    
    def create_ensemble_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create ensemble predictions using multiple methods"""
        if not self.config["ensemble"]["enabled"]:
            return {}
        
        ensemble_results = {}
        
        try:
            # Train all models
            trained_models = []
            predictions = []
            
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            for model in self.models[:5]:  # Use top 5 models
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    predictions.append(pred)
                    trained_models.append(model.name)
                except Exception as e:
                    logger.warning(f"Ensemble training failed for {model.name}: {e}")
                    continue
            
            if len(predictions) < 2:
                return {"error": "Not enough models for ensemble"}
            
            predictions = np.array(predictions)
            
            # Simple average
            if "simple_average" in self.config["ensemble"]["methods"]:
                simple_avg = np.mean(predictions, axis=0)
                mae_simple = mean_absolute_error(y_test, simple_avg)
                mape_simple = mean_absolute_percentage_error(y_test, simple_avg) * 100
                
                ensemble_results["simple_average"] = {
                    "mae": mae_simple,
                    "mape": mape_simple,
                    "models_used": trained_models
                }
            
            # Weighted average (based on inverse MAPE)
            if "weighted_average" in self.config["ensemble"]["methods"]:
                weights = []
                for i, model in enumerate(trained_models):
                    # Find model's performance (use simple validation)
                    mae = mean_absolute_error(y_test, predictions[i])
                    weight = 1 / (mae + 1e-8)
                    weights.append(weight)
                
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                weighted_avg = np.average(predictions, axis=0, weights=weights)
                mae_weighted = mean_absolute_error(y_test, weighted_avg)
                mape_weighted = mean_absolute_percentage_error(y_test, weighted_avg) * 100
                
                ensemble_results["weighted_average"] = {
                    "mae": mae_weighted,
                    "mape": mape_weighted,
                    "weights": weights.tolist(),
                    "models_used": trained_models
                }
            
            logger.info("Ensemble predictions created successfully")
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            ensemble_results["error"] = str(e)
        
        return ensemble_results
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        if not self.results:
            return {"error": "No validation results available"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models": len(self.models),
                "total_datasets": len(self.results),
                "best_model": None,
                "worst_model": None,
                "avg_mae": 0,
                "avg_mape": 0
            },
            "model_rankings": [],
            "dataset_analysis": {},
            "ensemble_results": self.ensemble_results,
            "recommendations": []
        }
        
        # Calculate summary statistics
        all_maes = []
        all_mapes = []
        model_performances = {}
        
        for dataset_name, dataset_results in self.results.items():
            dataset_maes = []
            dataset_mapes = []
            
            for model_name, result in dataset_results.items():
                if result.mae != float('inf'):
                    all_maes.append(result.mae)
                    dataset_maes.append(result.mae)
                    
                if result.mape != float('inf'):
                    all_mapes.append(result.mape)
                    dataset_mapes.append(result.mape)
                    
                if model_name not in model_performances:
                    model_performances[model_name] = []
                model_performances[model_name].append(result.mape)
            
            report["dataset_analysis"][dataset_name] = {
                "avg_mae": float(np.mean(dataset_maes)) if dataset_maes else 0,
                "avg_mape": float(np.mean(dataset_mapes)) if dataset_mapes else 0,
                "best_model": min(dataset_results.items(), key=lambda x: x[1].mape)[0] if dataset_results else None
            }
        
        if all_maes and all_mapes:
            report["summary"]["avg_mae"] = float(np.mean(all_maes))
            report["summary"]["avg_mape"] = float(np.mean(all_mapes))
            
            # Model rankings
            model_avg_mapes = {name: np.mean(perfs) for name, perfs in model_performances.items() 
                              if perfs and all(p != float('inf') for p in perfs)}
            
            if model_avg_mapes:
                sorted_models = sorted(model_avg_mapes.items(), key=lambda x: x[1])
                report["model_rankings"] = [
                    {"model": name, "avg_mape": mape} 
                    for name, mape in sorted_models
                ]
                
                report["summary"]["best_model"] = sorted_models[0][0]
                report["summary"]["worst_model"] = sorted_models[-1][0]
        
        # Generate recommendations
        report["recommendations"] = self._generate_ml_recommendations(report)
        
        # Save report
        report_path = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp/ml_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"ML validation report saved to: {report_path}")
        return report
    
    def _generate_ml_recommendations(self, report: Dict) -> List[str]:
        """Generate ML recommendations based on validation results"""
        recommendations = []
        
        if "best_model" in report["summary"] and report["summary"]["best_model"]:
            best_model = report["summary"]["best_model"]
            recommendations.append(f"Deploy {best_model} as primary forecasting model")
        
        if "model_rankings" in report and len(report["model_rankings"]) > 1:
            top_3 = [m["model"] for m in report["model_rankings"][:3]]
            recommendations.append(f"Consider ensemble of top 3 models: {', '.join(top_3)}")
        
        # Model-specific recommendations
        if "model_rankings" in report:
            for model_info in report["model_rankings"]:
                if model_info["avg_mape"] > 50:
                    recommendations.append(f"{model_info['model']} shows poor performance - consider parameter tuning or removal")
                elif model_info["avg_mape"] < 10:
                    recommendations.append(f"{model_info['model']} shows excellent performance - prioritize for production use")
        
        recommendations.extend([
            "Retrain models monthly with new data to maintain accuracy",
            "Monitor model drift and performance degradation over time",
            "Implement automated model selection based on recent performance",
            "Use ensemble methods for critical forecasting decisions",
            "Set up alerts when model accuracy drops below acceptable thresholds"
        ])
        
        return recommendations


def main():
    """Main function to run ML validation"""
    print("🧠 ML Model Validation System for Beverly Knits")
    print("=" * 60)
    
    # Initialize system
    validation_system = MLValidationSystem()
    
    print(f"Initialized {len(validation_system.models)} ML models for validation:")
    for model in validation_system.models:
        print(f"  • {model.name} ({model.model_type})")
    
    # Load data sources
    data_sources = {}
    
    # Load from various sources
    data_paths = [
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp/data"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/4"),
        Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5")
    ]
    
    for data_path in data_paths:
        if data_path.exists():
            for file_path in data_path.glob("*.csv"):
                try:
                    data_sources[f"{data_path.name}_{file_path.stem}"] = pd.read_csv(file_path)
                    print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            for file_path in data_path.glob("*.xlsx"):
                try:
                    data_sources[f"{data_path.name}_{file_path.stem}"] = pd.read_excel(file_path)
                    print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    # Create synthetic data if no real data available
    if not data_sources:
        print("No data found. Creating synthetic data for demonstration...")
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        synthetic_data = pd.DataFrame({
            'Date': dates,
            'Quantity': np.random.poisson(100, 100) + 10 * np.sin(np.arange(100) * 2 * np.pi / 30),
            'Temperature': np.random.normal(20, 5, 100),
            'Seasonality': np.sin(np.arange(100) * 2 * np.pi / 365) * 50
        })
        data_sources["synthetic"] = synthetic_data
    
    print(f"\nRunning validation on {len(data_sources)} datasets...")
    
    # Run validation
    results = validation_system.validate_all_models(data_sources)
    
    # Create ensemble predictions
    for dataset_name, data in data_sources.items():
        try:
            X, y = validation_system.prepare_features(data)
            ensemble_results = validation_system.create_ensemble_predictions(X, y)
            validation_system.ensemble_results[dataset_name] = ensemble_results
        except Exception as e:
            print(f"Ensemble creation failed for {dataset_name}: {e}")
    
    # Generate report
    report = validation_system.generate_validation_report()
    
    # Print summary
    print(f"\n{'='*60}")
    print("ML MODEL VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if "summary" in report:
        summary = report["summary"]
        print(f"Models Tested: {summary['total_models']}")
        print(f"Datasets: {summary['total_datasets']}")
        print(f"Average MAE: {summary['avg_mae']:.2f}")
        print(f"Average MAPE: {summary['avg_mape']:.2f}%")
        
        if summary.get("best_model"):
            print(f"Best Model: {summary['best_model']}")
    
    if "model_rankings" in report and report["model_rankings"]:
        print(f"\nModel Rankings (by MAPE):")
        for i, model in enumerate(report["model_rankings"][:5], 1):
            print(f"{i}. {model['model']}: {model['avg_mape']:.2f}%")
    
    print(f"\n✅ ML validation completed!")
    print(f"📁 Detailed report saved to: ml_validation_report.json")
    print(f"💾 Results stored in database: ml_validation_results.db")


if __name__ == "__main__":
    main()