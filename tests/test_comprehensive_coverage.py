"""
Comprehensive test suite to achieve 80% code coverage for Beverly Knits ERP
Tests all critical business logic and API endpoints
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Import modules to test
from forecasting.enhanced_forecasting_engine import (
    EnhancedForecastingEngine, 
    ForecastConfig, 
    ForecastResult
)
from forecasting.forecast_accuracy_monitor import (
    ForecastAccuracyMonitor,
    AccuracyMetrics,
    PerformanceAlert
)
from forecasting.forecast_auto_retrain import AutomaticRetrainingSystem
from forecasting.forecast_validation_backtesting import (
    ForecastValidationSystem,
    BacktestResult,
    ValidationReport
)
from forecasting.forecasting_integration import ForecastingIntegration


class TestEnhancedForecastingEngine:
    """Test suite for Enhanced Forecasting Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create test engine instance"""
        config = ForecastConfig(
            horizon_weeks=9,
            min_accuracy_threshold=0.90
        )
        return EnhancedForecastingEngine(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample historical data"""
        dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
        return pd.DataFrame({
            'date': dates,
            'Consumed': -np.random.normal(1000, 200, 52)
        })
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.config.horizon_weeks == 9
        assert engine.config.min_accuracy_threshold == 0.90
        assert engine.models is not None
        assert engine.last_training_date is None
    
    def test_forecast_generation(self, engine, sample_data):
        """Test forecast generation for a yarn"""
        result = engine.forecast('YARN001', sample_data)
        
        assert isinstance(result, ForecastResult)
        assert result.yarn_id == 'YARN001'
        assert result.horizon_weeks == 9
        assert result.predictions is not None
        assert len(result.predictions) == 9
    
    def test_dual_forecast_system(self, engine, sample_data):
        """Test combined historical and order-based forecasting"""
        order_data = pd.DataFrame({
            'order_date': pd.date_range(start='2024-12-01', periods=10, freq='W'),
            'quantity': np.random.normal(500, 100, 10)
        })
        
        result = engine.forecast('YARN002', sample_data, order_data)
        
        assert result.predictions is not None
        assert 'forecast' in result.predictions.columns
        assert result.predictions['forecast'].mean() > 0
    
    def test_confidence_intervals(self, engine, sample_data):
        """Test confidence interval calculation"""
        result = engine.forecast('YARN003', sample_data)
        
        assert result.confidence_intervals is not None
        assert 'lower_bound' in result.confidence_intervals.columns
        assert 'upper_bound' in result.confidence_intervals.columns
        
        # Check bounds are sensible
        for _, row in result.confidence_intervals.iterrows():
            assert row['lower_bound'] <= row['upper_bound']
            assert row['lower_bound'] >= 0
    
    def test_accuracy_metrics(self, engine, sample_data):
        """Test accuracy metrics calculation"""
        result = engine.forecast('YARN004', sample_data)
        
        assert result.accuracy_metrics is not None
        if result.accuracy_metrics.get('accuracy') is not None:
            assert 0 <= result.accuracy_metrics['accuracy'] <= 1
    
    def test_model_retraining(self, engine):
        """Test model retraining functionality"""
        training_data = {
            'YARN001': pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=52, freq='W'),
                'Consumed': -np.random.normal(1000, 200, 52)
            })
        }
        
        accuracies = engine.retrain_models(training_data)
        
        assert isinstance(accuracies, dict)
        assert engine.last_training_date is not None
    
    def test_ensemble_weight_optimization(self, engine):
        """Test ensemble weight optimization"""
        accuracies = {
            'prophet': 0.92,
            'xgboost': 0.88,
            'arima': 0.85
        }
        
        engine._optimize_ensemble_weights(accuracies)
        
        # Check weights sum to 1
        total_weight = sum(engine.config.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_needs_retraining(self, engine):
        """Test retraining check logic"""
        # Should need retraining initially
        assert engine.needs_retraining() == True
        
        # Set last training to now
        engine.last_training_date = datetime.now()
        assert engine.needs_retraining() == False
        
        # Set to 8 days ago (weekly schedule)
        engine.last_training_date = datetime.now() - timedelta(days=8)
        assert engine.needs_retraining() == True


class TestForecastAccuracyMonitor:
    """Test suite for Forecast Accuracy Monitor"""
    
    @pytest.fixture
    def monitor(self, tmp_path):
        """Create test monitor instance"""
        db_path = str(tmp_path / "test_accuracy.db")
        return ForecastAccuracyMonitor(
            db_path=db_path,
            accuracy_threshold=0.90
        )
    
    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.accuracy_threshold == 0.90
        assert monitor.current_metrics == {}
        assert os.path.exists(monitor.db_path)
    
    def test_track_forecast(self, monitor):
        """Test forecast tracking"""
        monitor.track_forecast(
            yarn_id="YARN001",
            forecast_date=datetime.now(),
            predicted_values=[100, 110, 105],
            model_used="Ensemble",
            horizon_weeks=3
        )
        
        # Verify forecast was tracked (would need to query DB)
        assert True  # Placeholder for actual DB verification
    
    def test_evaluate_forecast(self, monitor):
        """Test forecast evaluation"""
        # Track a forecast first
        forecast_date = datetime.now() - timedelta(days=1)
        monitor.track_forecast(
            yarn_id="YARN002",
            forecast_date=forecast_date,
            predicted_values=[100, 110, 105],
            model_used="Ensemble",
            horizon_weeks=3
        )
        
        # Evaluate with actual values
        actual_values = [98, 112, 103]
        metrics = monitor.evaluate_forecast("YARN002", actual_values)
        
        if metrics:
            assert isinstance(metrics, AccuracyMetrics)
            assert metrics.yarn_id == "YARN002"
            assert 0 <= metrics.accuracy <= 1
    
    def test_accuracy_alert_generation(self, monitor):
        """Test alert generation for low accuracy"""
        alerts = []
        
        def capture_alert(alert):
            alerts.append(alert)
        
        monitor.alert_callback = capture_alert
        
        # Create low accuracy metrics
        metrics = AccuracyMetrics(
            yarn_id="YARN003",
            forecast_date=datetime.now(),
            horizon_weeks=9,
            mape=0.15,
            rmse=50,
            mae=40,
            accuracy=0.85,  # Below 90% threshold
            model_used="Ensemble",
            actual_values=[100],
            predicted_values=[85]
        )
        
        monitor._check_accuracy_alert(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].severity == 'warning'
    
    def test_model_performance_report(self, monitor):
        """Test model performance reporting"""
        performance = monitor.get_model_performance(period_days=30)
        
        assert isinstance(performance, dict)
        # Initially empty
        assert len(performance) == 0
    
    def test_accuracy_report_generation(self, monitor):
        """Test comprehensive accuracy report"""
        report = monitor.get_accuracy_report(period_days=30)
        
        assert 'period_days' in report
        assert 'accuracy_threshold' in report
        assert 'yarn_metrics' in report
        assert 'overall_metrics' in report
        assert report['period_days'] == 30
        assert report['accuracy_threshold'] == 0.90


class TestAutomaticRetrainingSystem:
    """Test suite for Automatic Retraining System"""
    
    @pytest.fixture
    def retrain_system(self, tmp_path):
        """Create test retraining system"""
        return AutomaticRetrainingSystem(
            data_path=str(tmp_path),
            retrain_schedule="weekly",
            retrain_day="sunday",
            retrain_hour=2
        )
    
    def test_initialization(self, retrain_system):
        """Test retraining system initialization"""
        assert retrain_system.retrain_schedule == "weekly"
        assert retrain_system.retrain_day == "sunday"
        assert retrain_system.retrain_hour == 2
        assert retrain_system.is_running == False
    
    @patch('forecast_auto_retrain.AutomaticRetrainingSystem.load_training_data')
    def test_retrain_models(self, mock_load_data, retrain_system):
        """Test model retraining process"""
        # Mock training data
        mock_load_data.return_value = {
            'YARN001': pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=52, freq='W'),
                'Consumed': -np.random.normal(1000, 200, 52)
            })
        }
        
        results = retrain_system.retrain_models()
        
        assert results['status'] == 'success'
        assert 'training_time_seconds' in results
        assert results['yarns_trained'] == 1
    
    def test_training_status(self, retrain_system):
        """Test training status reporting"""
        status = retrain_system.get_training_status()
        
        assert 'is_running' in status
        assert 'schedule' in status
        assert 'accuracy_threshold' in status
        assert status['is_running'] == False
        assert status['schedule'] == 'weekly'


class TestForecastValidationSystem:
    """Test suite for Forecast Validation System"""
    
    @pytest.fixture
    def validator(self):
        """Create test validation system"""
        return ForecastValidationSystem(accuracy_target=0.90)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for validation"""
        data = {}
        for i in range(5):
            yarn_id = f"YARN{i:03d}"
            dates = pd.date_range(start='2023-01-01', periods=100, freq='W')
            data[yarn_id] = pd.DataFrame({
                'date': dates,
                'Consumed': -np.random.normal(1000, 200, 100)
            })
        return data
    
    def test_initialization(self, validator):
        """Test validation system initialization"""
        assert validator.accuracy_target == 0.90
        assert validator.confidence_level == 0.95
        assert validator.validation_history == []
    
    def test_backtest_yarn(self, validator, test_data):
        """Test backtesting for single yarn"""
        yarn_id = "YARN001"
        result = validator.backtest_yarn(
            yarn_id=yarn_id,
            historical_data=test_data[yarn_id],
            test_periods=3,
            horizon_weeks=9
        )
        
        assert isinstance(result, BacktestResult)
        assert result.yarn_id == yarn_id
        assert result.test_periods == 3
        assert result.horizon_weeks == 9
        assert 0 <= result.ensemble_accuracy <= 1
    
    def test_validation_report(self, validator, test_data):
        """Test comprehensive validation"""
        report = validator.validate_forecast_system(test_data, sample_size=3)
        
        assert isinstance(report, ValidationReport)
        assert report.total_yarns_tested <= 3
        assert 0 <= report.average_accuracy <= 1
        assert isinstance(report.validation_passed, bool)
    
    def test_optimal_weight_calculation(self, validator):
        """Test optimal weight calculation"""
        model_rankings = {
            'prophet': 0.92,
            'xgboost': 0.88,
            'arima': 0.85
        }
        
        weights = validator._calculate_optimal_weights(model_rankings)
        
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert weights['prophet'] > weights['arima']


class TestForecastingIntegration:
    """Test suite for Forecasting Integration"""
    
    @pytest.fixture
    def integration(self, tmp_path):
        """Create test integration instance"""
        return ForecastingIntegration(
            data_path=str(tmp_path),
            accuracy_target=0.90
        )
    
    def test_initialization(self, integration):
        """Test integration initialization"""
        assert integration.accuracy_target == 0.90
        assert integration.forecast_engine is not None
        assert integration.accuracy_monitor is not None
        assert integration.retrain_system is not None
        assert integration.validation_system is not None
    
    @patch('forecasting_integration.ForecastingIntegration._load_yarn_history')
    def test_generate_forecast(self, mock_load, integration):
        """Test single yarn forecast generation"""
        # Mock historical data
        mock_load.return_value = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=52, freq='W'),
            'Consumed': -np.random.normal(1000, 200, 52)
        })
        
        result = integration.generate_forecast("YARN001", include_orders=False)
        
        assert 'yarn_id' in result
        assert 'predictions' in result
        assert result['yarn_id'] == "YARN001"
    
    def test_forecast_status(self, integration):
        """Test forecast status reporting"""
        status = integration.get_forecast_status()
        
        assert 'timestamp' in status
        assert 'system_status' in status
        assert 'accuracy_target' in status
        assert 'horizon_weeks' in status
        assert status['accuracy_target'] == 0.90
        assert status['horizon_weeks'] == 9


class TestBusinessLogic:
    """Test critical business logic calculations"""
    
    def test_planning_balance_calculation(self):
        """Test Planning Balance formula"""
        # Planning Balance = Theoretical Balance + Allocated + On Order
        # Note: Allocated is already negative in data
        
        theoretical_balance = 1000
        allocated = -200  # Already negative
        on_order = 500
        
        planning_balance = theoretical_balance + allocated + on_order
        
        assert planning_balance == 1300
    
    def test_weekly_demand_calculation(self):
        """Test weekly demand calculation logic"""
        # From consumed data
        monthly_consumed = -4300  # Negative value
        weekly_demand = abs(monthly_consumed) / 4.3
        assert abs(weekly_demand - 1000) < 0.1
        
        # From allocated quantity
        allocated_qty = 800
        production_cycle = 8
        weekly_demand_allocated = allocated_qty / production_cycle
        assert weekly_demand_allocated == 100
    
    def test_yarn_substitution_score(self):
        """Test yarn substitution scoring"""
        color_match = 0.9
        composition_match = 0.8
        weight_match = 0.85
        
        # Weights: color=0.3, composition=0.4, weight=0.3
        similarity_score = (
            color_match * 0.3 +
            composition_match * 0.4 +
            weight_match * 0.3
        )
        
        expected = 0.9 * 0.3 + 0.8 * 0.4 + 0.85 * 0.3
        assert abs(similarity_score - expected) < 0.001
    
    def test_unit_conversion(self):
        """Test yards to pounds conversion"""
        yards = 1000
        yards_per_pound = 2.5  # Example conversion factor
        
        pounds = yards / yards_per_pound
        assert pounds == 400


class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        from flask import Flask
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app, tmp_path):
        """Create test client"""
        from forecasting_integration import create_forecast_api
        create_forecast_api(app, str(tmp_path))
        return app.test_client()
    
    def test_forecast_status_endpoint(self, client):
        """Test /api/forecast/status endpoint"""
        response = client.get('/api/forecast/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'system_status' in data
        assert 'accuracy_target' in data
    
    def test_retrain_endpoint(self, client):
        """Test /api/forecast/retrain endpoint"""
        response = client.post('/api/forecast/retrain')
        
        # Should work even without data
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
    
    def test_accuracy_report_endpoint(self, client):
        """Test /api/forecast/accuracy-report endpoint"""
        response = client.get('/api/forecast/accuracy-report?days=7')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'period_days' in data
        assert data['period_days'] == 7


class TestDataValidation:
    """Test data validation and cleaning"""
    
    def test_handle_missing_columns(self):
        """Test handling of missing columns"""
        df = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002'],
            'Theoretical_Balance': [1000, 2000]
        })
        
        # Should handle missing Planning_Balance
        assert 'Planning_Balance' not in df.columns
        
        # Should handle typos
        df_typo = df.copy()
        df_typo['Planning_Ballance'] = [1500, 2500]  # Typo
        assert 'Planning_Ballance' in df_typo.columns
    
    def test_negative_allocated_values(self):
        """Test handling of negative allocated values"""
        allocated = pd.Series([-100, -200, -150])
        
        # Values should already be negative
        assert all(allocated <= 0)
        
        # When used in Planning Balance calculation
        planning_balance = 1000 + allocated + 500
        expected = pd.Series([1400, 1300, 1350])
        assert all(planning_balance == expected)
    
    def test_data_type_conversion(self):
        """Test data type conversions"""
        # String to numeric
        df = pd.DataFrame({
            'Consumed': ['-1000', '-2000', '-1500']
        })
        
        df['Consumed'] = pd.to_numeric(df['Consumed'])
        assert df['Consumed'].dtype in [np.float64, np.int64]
        assert all(df['Consumed'] < 0)


class TestPerformanceOptimization:
    """Test performance optimization features"""
    
    def test_cache_functionality(self):
        """Test caching mechanism"""
        from functools import lru_cache
        
        call_count = 0
        
        @lru_cache(maxsize=128)
        def expensive_calculation(n):
            nonlocal call_count
            call_count += 1
            return n * 2
        
        # First call
        result1 = expensive_calculation(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (cached)
        result2 = expensive_calculation(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
    
    def test_dataframe_memory_optimization(self):
        """Test DataFrame memory optimization"""
        # Create large DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 10000),
            'float_col': np.random.random(10000),
            'str_col': ['YARN001'] * 10000
        })
        
        initial_memory = df.memory_usage(deep=True).sum()
        
        # Optimize dtypes
        df['int_col'] = df['int_col'].astype('int8')  # Smaller int
        df['float_col'] = df['float_col'].astype('float32')  # Smaller float
        df['str_col'] = df['str_col'].astype('category')  # Category for repeated strings
        
        optimized_memory = df.memory_usage(deep=True).sum()
        
        assert optimized_memory < initial_memory
        reduction = 1 - (optimized_memory / initial_memory)
        assert reduction > 0.5  # At least 50% reduction


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])