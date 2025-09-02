#!/usr/bin/env python3
"""
ML Model Configuration for Beverly Knits ERP
Centralized configuration for all ML models and parameters
Created: 2025-09-02
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "production" / "5"
MODEL_PATH = BASE_DIR / "models"
TRAINING_PATH = BASE_DIR / "training_results"

# Ensure directories exist
MODEL_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_PATH.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for individual ML models"""
    name: str
    type: str  # 'timeseries', 'regression', 'classification', 'clustering'
    algorithm: str  # 'arima', 'prophet', 'lstm', 'xgboost', 'random_forest', etc.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    test_split: float = 0.1
    features: List[str] = field(default_factory=list)
    target: str = ""
    enabled: bool = True
    retrain_frequency_days: int = 7
    min_accuracy_threshold: float = 0.7
    max_training_time_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'type': self.type,
            'algorithm': self.algorithm,
            'hyperparameters': self.hyperparameters,
            'training_params': self.training_params,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'features': self.features,
            'target': self.target,
            'enabled': self.enabled,
            'retrain_frequency_days': self.retrain_frequency_days,
            'min_accuracy_threshold': self.min_accuracy_threshold,
            'max_training_time_seconds': self.max_training_time_seconds
        }

# ARIMA Configuration
ARIMA_CONFIG = ModelConfig(
    name="demand_forecast_arima",
    type="timeseries",
    algorithm="arima",
    hyperparameters={
        'p': 2,  # Autoregressive order
        'd': 1,  # Differencing order
        'q': 2,  # Moving average order
        'seasonal': True,
        'seasonal_order': (1, 1, 1, 12),  # (P, D, Q, S)
        'trend': 'c',  # Constant trend
        'enforce_stationarity': False,
        'enforce_invertibility': False
    },
    training_params={
        'method': 'lbfgs',
        'maxiter': 100,
        'disp': False
    },
    target="demand",
    validation_split=0.2,
    min_accuracy_threshold=0.75,
    retrain_frequency_days=7
)

# Prophet Configuration
PROPHET_CONFIG = ModelConfig(
    name="demand_forecast_prophet",
    type="timeseries",
    algorithm="prophet",
    hyperparameters={
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10,
        'seasonality_mode': 'multiplicative',
        'changepoint_range': 0.8,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'growth': 'linear',
        'interval_width': 0.95,
        'n_changepoints': 25
    },
    training_params={
        'include_history': True,
        'uncertainty_samples': 1000
    },
    target="demand",
    validation_split=0.2,
    min_accuracy_threshold=0.80,
    retrain_frequency_days=7
)

# LSTM Configuration
LSTM_CONFIG = ModelConfig(
    name="demand_forecast_lstm",
    type="timeseries",
    algorithm="lstm",
    hyperparameters={
        'units': [128, 64, 32],  # Units per layer
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'use_bias': True,
        'return_sequences': [True, True, False],  # Per layer
        'stateful': False,
        'time_steps': 30,  # Look-back period
        'forecast_horizon': 90  # Forecast period
    },
    training_params={
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae', 'mape'],
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'validation_freq': 1,
        'shuffle': True
    },
    features=['demand', 'price', 'inventory_level', 'season', 'trend'],
    target="demand",
    validation_split=0.2,
    test_split=0.1,
    min_accuracy_threshold=0.85,
    retrain_frequency_days=14,
    max_training_time_seconds=600
)

# XGBoost Configuration
XGBOOST_CONFIG = ModelConfig(
    name="demand_forecast_xgboost",
    type="regression",
    algorithm="xgboost",
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'n_jobs': -1,
        'gamma': 0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'random_state': 42
    },
    training_params={
        'eval_metric': ['rmse', 'mae'],
        'early_stopping_rounds': 10,
        'verbose': False
    },
    features=[
        'lag_1', 'lag_7', 'lag_30',  # Lagged features
        'rolling_mean_7', 'rolling_mean_30',  # Rolling statistics
        'month', 'quarter', 'dayofweek',  # Time features
        'price', 'inventory_level',  # Business features
        'is_holiday', 'is_weekend'  # Calendar features
    ],
    target="demand",
    validation_split=0.2,
    test_split=0.1,
    min_accuracy_threshold=0.82,
    retrain_frequency_days=7
)

# Random Forest Configuration for Yarn Substitution
YARN_SUBSTITUTION_CONFIG = ModelConfig(
    name="yarn_substitution_classifier",
    type="classification",
    algorithm="random_forest",
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    training_params={
        'cv_folds': 5,
        'scoring': 'f1_weighted'
    },
    features=[
        'color_similarity',
        'weight_similarity',
        'material_match',
        'price_ratio',
        'availability_score',
        'quality_score',
        'supplier_reliability'
    ],
    target="is_substitutable",
    validation_split=0.2,
    min_accuracy_threshold=0.90,
    retrain_frequency_days=30
)

# Inventory Optimization Configuration
INVENTORY_OPTIMIZATION_CONFIG = ModelConfig(
    name="inventory_optimizer",
    type="regression",
    algorithm="gradient_boosting",
    hyperparameters={
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 4,
        'min_samples_leaf': 3,
        'subsample': 0.8,
        'max_features': 0.8,
        'loss': 'huber',
        'alpha': 0.9,  # For huber loss
        'random_state': 42
    },
    training_params={
        'warm_start': False,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'tol': 0.0001
    },
    features=[
        'current_inventory',
        'lead_time',
        'demand_variability',
        'service_level',
        'holding_cost',
        'ordering_cost',
        'stockout_cost',
        'supplier_reliability',
        'seasonality_index'
    ],
    target="optimal_order_quantity",
    validation_split=0.2,
    min_accuracy_threshold=0.78,
    retrain_frequency_days=14
)

# Ensemble Configuration
ENSEMBLE_CONFIG = ModelConfig(
    name="demand_forecast_ensemble",
    type="ensemble",
    algorithm="voting",
    hyperparameters={
        'models': ['arima', 'prophet', 'lstm', 'xgboost'],
        'weights': [0.2, 0.25, 0.35, 0.2],  # Model weights
        'voting': 'soft',  # 'hard' or 'soft'
        'aggregation': 'weighted_mean',  # 'mean', 'weighted_mean', 'median'
        'confidence_threshold': 0.7
    },
    training_params={
        'retrain_base_models': True,
        'optimize_weights': True,
        'weight_optimization_method': 'grid_search'
    },
    target="demand",
    validation_split=0.2,
    min_accuracy_threshold=0.88,
    retrain_frequency_days=7
)

# Global ML Configuration
ML_GLOBAL_CONFIG = {
    # General settings
    'enable_ml': True,
    'auto_retrain': True,
    'parallel_training': True,
    'max_parallel_jobs': 4,
    'cache_predictions': True,
    'cache_ttl_hours': 24,
    
    # Data settings
    'min_training_samples': 100,
    'max_training_samples': 100000,
    'handle_missing_data': 'interpolate',  # 'drop', 'interpolate', 'forward_fill'
    'outlier_detection': True,
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
    'outlier_threshold': 3,
    
    # Feature engineering
    'auto_feature_engineering': True,
    'create_lag_features': True,
    'lag_periods': [1, 7, 14, 30],
    'create_rolling_features': True,
    'rolling_windows': [7, 14, 30],
    'create_time_features': True,
    'polynomial_features': False,
    'interaction_features': False,
    
    # Model selection
    'auto_model_selection': True,
    'model_selection_metric': 'mape',  # 'rmse', 'mae', 'mape', 'r2'
    'cross_validation_folds': 5,
    'hyperparameter_tuning': True,
    'tuning_method': 'bayesian',  # 'grid', 'random', 'bayesian'
    'tuning_iterations': 50,
    
    # Performance monitoring
    'track_performance': True,
    'performance_window_days': 30,
    'alert_on_degradation': True,
    'degradation_threshold': 0.1,  # 10% performance drop
    'log_predictions': True,
    'log_level': 'INFO',
    
    # Forecasting settings
    'default_forecast_horizon': 90,
    'confidence_intervals': True,
    'confidence_level': 0.95,
    'seasonal_decomposition': True,
    'trend_analysis': True,
    
    # Production settings
    'model_versioning': True,
    'rollback_on_failure': True,
    'a_b_testing': False,
    'canary_deployment': False,
    'model_registry': str(MODEL_PATH),
    'training_registry': str(TRAINING_PATH)
}

# Model Registry
MODEL_REGISTRY = {
    'arima': ARIMA_CONFIG,
    'prophet': PROPHET_CONFIG,
    'lstm': LSTM_CONFIG,
    'xgboost': XGBOOST_CONFIG,
    'yarn_substitution': YARN_SUBSTITUTION_CONFIG,
    'inventory_optimization': INVENTORY_OPTIMIZATION_CONFIG,
    'ensemble': ENSEMBLE_CONFIG
}

# Training Schedule
TRAINING_SCHEDULE = {
    'daily': ['arima', 'xgboost'],
    'weekly': ['prophet', 'lstm', 'ensemble'],
    'biweekly': ['inventory_optimization'],
    'monthly': ['yarn_substitution']
}

# Performance Benchmarks
PERFORMANCE_BENCHMARKS = {
    'arima': {'mape': 0.15, 'rmse': 100, 'training_time': 30},
    'prophet': {'mape': 0.12, 'rmse': 90, 'training_time': 45},
    'lstm': {'mape': 0.10, 'rmse': 80, 'training_time': 300},
    'xgboost': {'mape': 0.11, 'rmse': 85, 'training_time': 60},
    'ensemble': {'mape': 0.09, 'rmse': 75, 'training_time': 120}
}

class MLConfigManager:
    """Manager for ML configurations"""
    
    def __init__(self):
        self.config_file = MODEL_PATH / "ml_config.json"
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file if exists"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                saved_config = json.load(f)
                # Update global config with saved values
                ML_GLOBAL_CONFIG.update(saved_config.get('global', {}))
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        config = {
            'global': ML_GLOBAL_CONFIG,
            'models': {name: model.to_dict() for name, model in MODEL_REGISTRY.items()},
            'schedule': TRAINING_SCHEDULE,
            'benchmarks': PERFORMANCE_BENCHMARKS,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return MODEL_REGISTRY.get(model_name)
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> bool:
        """Update model configuration"""
        if model_name in MODEL_REGISTRY:
            model_config = MODEL_REGISTRY[model_name]
            for key, value in updates.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            self.save_config()
            return True
        return False
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models"""
        return [name for name, config in MODEL_REGISTRY.items() if config.enabled]
    
    def get_models_for_training(self) -> List[str]:
        """Get models that need training based on schedule"""
        models_to_train = []
        today = datetime.now().date()
        
        for schedule, models in TRAINING_SCHEDULE.items():
            should_train = False
            
            if schedule == 'daily':
                should_train = True
            elif schedule == 'weekly' and today.weekday() == 0:  # Monday
                should_train = True
            elif schedule == 'biweekly' and today.day in [1, 15]:
                should_train = True
            elif schedule == 'monthly' and today.day == 1:
                should_train = True
            
            if should_train:
                models_to_train.extend(models)
        
        # Filter for enabled models only
        return [m for m in models_to_train if MODEL_REGISTRY[m].enabled]
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate all configurations"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Check model configs
        for name, config in MODEL_REGISTRY.items():
            if config.validation_split + config.test_split >= 1.0:
                issues['errors'].append(f"{name}: validation + test split >= 1.0")
            
            if config.min_accuracy_threshold > 0.95:
                issues['warnings'].append(f"{name}: very high accuracy threshold ({config.min_accuracy_threshold})")
            
            if config.retrain_frequency_days > 30:
                issues['warnings'].append(f"{name}: infrequent retraining ({config.retrain_frequency_days} days)")
        
        # Check global config
        if ML_GLOBAL_CONFIG['max_parallel_jobs'] > os.cpu_count():
            issues['warnings'].append(f"max_parallel_jobs ({ML_GLOBAL_CONFIG['max_parallel_jobs']}) > CPU count ({os.cpu_count()})")
        
        return issues
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {}
        
        for model_name in self.get_enabled_models():
            benchmark = PERFORMANCE_BENCHMARKS.get(model_name, {})
            config = MODEL_REGISTRY[model_name]
            
            summary[model_name] = {
                'enabled': config.enabled,
                'accuracy_threshold': config.min_accuracy_threshold,
                'benchmark_mape': benchmark.get('mape', 'N/A'),
                'benchmark_rmse': benchmark.get('rmse', 'N/A'),
                'expected_training_time': benchmark.get('training_time', 'N/A'),
                'retrain_frequency': config.retrain_frequency_days
            }
        
        return summary

# Initialize config manager
ml_config_manager = MLConfigManager()

# Export key components
__all__ = [
    'ML_GLOBAL_CONFIG',
    'MODEL_REGISTRY',
    'TRAINING_SCHEDULE',
    'PERFORMANCE_BENCHMARKS',
    'MLConfigManager',
    'ml_config_manager',
    'ModelConfig',
    'ARIMA_CONFIG',
    'PROPHET_CONFIG',
    'LSTM_CONFIG',
    'XGBOOST_CONFIG',
    'YARN_SUBSTITUTION_CONFIG',
    'INVENTORY_OPTIMIZATION_CONFIG',
    'ENSEMBLE_CONFIG'
]

if __name__ == "__main__":
    # Test configuration
    print("ML Configuration System")
    print("=" * 60)
    
    # Validate configuration
    issues = ml_config_manager.validate_config()
    if issues['errors']:
        print("Configuration Errors:")
        for error in issues['errors']:
            print(f"  - {error}")
    
    if issues['warnings']:
        print("\nConfiguration Warnings:")
        for warning in issues['warnings']:
            print(f"  - {warning}")
    
    # Show enabled models
    print("\nEnabled Models:")
    for model in ml_config_manager.get_enabled_models():
        config = MODEL_REGISTRY[model]
        print(f"  - {model}: {config.algorithm} ({config.type})")
    
    # Show training schedule
    print("\nToday's Training Schedule:")
    for model in ml_config_manager.get_models_for_training():
        print(f"  - {model}")
    
    # Show performance summary
    print("\nPerformance Benchmarks:")
    summary = ml_config_manager.get_performance_summary()
    for model, metrics in summary.items():
        print(f"  {model}:")
        print(f"    - MAPE: {metrics['benchmark_mape']}")
        print(f"    - Training time: {metrics['expected_training_time']}s")
    
    # Save configuration
    ml_config_manager.save_config()
    print(f"\nConfiguration saved to: {ml_config_manager.config_file}")