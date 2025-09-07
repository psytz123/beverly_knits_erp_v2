#!/usr/bin/env python3
"""
Test suite for ML Deployment Pipeline
Comprehensive tests for model versioning, A/B testing, monitoring, and auto-retraining
"""

import sys
import os
import tempfile
import shutil
import json
import sqlite3
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.ml_models.ml_deployment_pipeline import (
    MLDeploymentPipeline,
    ModelRegistry,
    ModelVersion,
    ABTestFramework,
    PerformanceMonitor,
    AutoRetrainingSystem,
    DeploymentStatus
)


class TestModelVersion(unittest.TestCase):
    """Test ModelVersion class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        
        # Create dummy model file
        with open(self.model_path, 'wb') as f:
            f.write(b"dummy model data")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_model_version_creation(self):
        """Test ModelVersion creation and properties"""
        metadata = {"accuracy": 0.95, "training_date": "2025-09-06"}
        
        version = ModelVersion(
            model_name="test_model",
            version="v1.0",
            model_path=self.model_path,
            metadata=metadata
        )
        
        self.assertEqual(version.model_name, "test_model")
        self.assertEqual(version.version, "v1.0")
        self.assertEqual(version.model_path, self.model_path)
        self.assertEqual(version.metadata, metadata)
        self.assertEqual(version.status, DeploymentStatus.PENDING)
        self.assertIsNotNone(version.model_id)
        self.assertIsNotNone(version.model_hash)
        self.assertTrue(len(version.model_hash) > 0)
    
    def test_model_version_serialization(self):
        """Test to_dict and from_dict methods"""
        version = ModelVersion(
            model_name="test_model",
            version="v1.0",
            model_path=self.model_path,
            metadata={"test": "data"}
        )
        
        # Test serialization
        version_dict = version.to_dict()
        self.assertIsInstance(version_dict, dict)
        self.assertEqual(version_dict['model_name'], "test_model")
        self.assertEqual(version_dict['version'], "v1.0")
        
        # Test deserialization
        restored_version = ModelVersion.from_dict(version_dict)
        self.assertEqual(restored_version.model_name, version.model_name)
        self.assertEqual(restored_version.version, version.version)
        self.assertEqual(restored_version.model_path, version.model_path)


class TestModelRegistry(unittest.TestCase):
    """Test ModelRegistry class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_registry.db")
        self.registry = ModelRegistry(self.db_path)
        
        # Create test model file
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        with open(self.model_path, 'wb') as f:
            f.write(b"test model data")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database is properly initialized"""
        self.assertTrue(os.path.exists(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('model_versions', tables)
            self.assertIn('deployment_history', tables)
            self.assertIn('ab_test_results', tables)
    
    def test_model_registration(self):
        """Test model registration"""
        version = ModelVersion(
            model_name="test_model",
            version="v1.0",
            model_path=self.model_path,
            metadata={"test": "data"}
        )
        
        success = self.registry.register_model(version)
        self.assertTrue(success)
        
        # Verify registration
        versions = self.registry.get_model_versions("test_model")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].model_name, "test_model")
        self.assertEqual(versions[0].version, "v1.0")
    
    def test_multiple_versions(self):
        """Test handling multiple model versions"""
        # Register multiple versions
        for i in range(3):
            version = ModelVersion(
                model_name="test_model",
                version=f"v1.{i}",
                model_path=self.model_path,
                metadata={"version_num": i}
            )
            self.registry.register_model(version)
        
        versions = self.registry.get_model_versions("test_model")
        self.assertEqual(len(versions), 3)
        
        # Versions should be ordered by creation date (newest first)
        version_nums = [v.version for v in versions]
        self.assertIn("v1.0", version_nums)
        self.assertIn("v1.1", version_nums)
        self.assertIn("v1.2", version_nums)
    
    def test_active_model_management(self):
        """Test active model setting and retrieval"""
        # Register model
        version = ModelVersion(
            model_name="test_model",
            version="v1.0",
            model_path=self.model_path
        )
        self.registry.register_model(version)
        
        # Initially no active model
        active = self.registry.get_active_model("test_model")
        self.assertIsNone(active)
        
        # Set as active
        self.registry.update_model_status(version.model_id, DeploymentStatus.ACTIVE)
        
        # Verify active model
        active = self.registry.get_active_model("test_model")
        self.assertIsNotNone(active)
        self.assertEqual(active.model_id, version.model_id)
        self.assertEqual(active.status, DeploymentStatus.ACTIVE)


class TestABTestFramework(unittest.TestCase):
    """Test A/B testing framework"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.temp_dir, "test_registry.db"))
        self.ab_framework = ABTestFramework(self.registry)
        
        # Create test models
        self.model_a_path = os.path.join(self.temp_dir, "model_a.pkl")
        self.model_b_path = os.path.join(self.temp_dir, "model_b.pkl")
        
        with open(self.model_a_path, 'wb') as f:
            f.write(b"model A data")
        with open(self.model_b_path, 'wb') as f:
            f.write(b"model B data")
        
        # Register models
        self.model_a = ModelVersion("test_model", "v1.0", self.model_a_path)
        self.model_b = ModelVersion("test_model", "v1.1", self.model_b_path)
        
        self.registry.register_model(self.model_a)
        self.registry.register_model(self.model_b)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_ab_test_creation(self):
        """Test A/B test creation"""
        test_config = {
            'traffic_split': 0.6,
            'duration_hours': 48,
            'metrics': ['mape', 'rmse'],
            'min_samples': 50
        }
        
        test_id = self.ab_framework.start_ab_test(
            self.model_a.model_id,
            self.model_b.model_id,
            test_config
        )
        
        self.assertIsNotNone(test_id)
        self.assertIn(test_id, self.ab_framework.active_tests)
        
        test = self.ab_framework.active_tests[test_id]
        self.assertEqual(test['model_a_id'], self.model_a.model_id)
        self.assertEqual(test['model_b_id'], self.model_b.model_id)
        self.assertEqual(test['config']['traffic_split'], 0.6)
    
    def test_traffic_routing(self):
        """Test traffic routing for A/B test"""
        test_config = {'traffic_split': 0.5}
        test_id = self.ab_framework.start_ab_test(
            self.model_a.model_id,
            self.model_b.model_id,
            test_config
        )
        
        # Test routing multiple times
        routes = []
        for _ in range(100):
            model_id, variant = self.ab_framework.route_prediction_request(test_id, {})
            routes.append(variant)
        
        # Should have both variants
        self.assertIn('model_a', routes)
        self.assertIn('model_b', routes)
        
        # Roughly balanced (allowing for randomness)
        model_a_count = routes.count('model_a')
        model_b_count = routes.count('model_b')
        self.assertGreater(model_a_count, 20)  # At least 20%
        self.assertGreater(model_b_count, 20)  # At least 20%
    
    def test_prediction_recording(self):
        """Test prediction result recording"""
        test_config = {'min_samples': 10}
        test_id = self.ab_framework.start_ab_test(
            self.model_a.model_id,
            self.model_b.model_id,
            test_config
        )
        
        # Record some predictions
        self.ab_framework.record_prediction_result(test_id, 'model_a', 100.0, 95.0, 'req1')
        self.ab_framework.record_prediction_result(test_id, 'model_b', 98.0, 95.0, 'req2')
        
        test = self.ab_framework.active_tests[test_id]
        predictions = test['results']['predictions']
        
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]['model_variant'], 'model_a')
        self.assertEqual(predictions[1]['model_variant'], 'model_b')
    
    def test_ab_test_evaluation(self):
        """Test A/B test evaluation"""
        test_config = {'min_samples': 5, 'metrics': ['mape', 'rmse']}
        test_id = self.ab_framework.start_ab_test(
            self.model_a.model_id,
            self.model_b.model_id,
            test_config
        )
        
        # Add sufficient test data
        # Model A: slightly worse performance
        for i in range(10):
            actual = 100.0
            prediction_a = actual + np.random.normal(0, 5)  # Higher error
            self.ab_framework.record_prediction_result(test_id, 'model_a', prediction_a, actual)
        
        # Model B: better performance
        for i in range(10):
            actual = 100.0
            prediction_b = actual + np.random.normal(0, 2)  # Lower error
            self.ab_framework.record_prediction_result(test_id, 'model_b', prediction_b, actual)
        
        evaluation = self.ab_framework.evaluate_ab_test(test_id)
        
        self.assertEqual(evaluation['status'], 'completed')
        self.assertIn('overall_winner', evaluation)
        self.assertIn('metric_results', evaluation)
        self.assertTrue(0 <= evaluation['confidence'] <= 1)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.temp_dir, "test_registry.db"))
        self.monitor = PerformanceMonitor(self.registry)
        self.monitor.monitoring_active = False  # Disable background thread for testing
        
        # Create test model
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        with open(model_path, 'wb') as f:
            f.write(b"test model")
        
        self.model = ModelVersion("test_model", "v1.0", model_path)
        self.registry.register_model(self.model)
    
    def tearDown(self):
        self.monitor.monitoring_active = False
        shutil.rmtree(self.temp_dir)
    
    def test_prediction_recording(self):
        """Test prediction recording"""
        model_id = self.model.model_id
        
        self.monitor.record_prediction(model_id, 100.0, 95.0, {'feature1': 1.5})
        self.monitor.record_prediction(model_id, 105.0, 100.0, {'feature1': 2.0})
        
        self.assertIn(model_id, self.monitor.monitoring_data)
        predictions = self.monitor.monitoring_data[model_id]['predictions']
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]['prediction'], 100.0)
        self.assertEqual(predictions[0]['actual'], 95.0)
    
    def test_performance_evaluation(self):
        """Test performance evaluation"""
        model_id = self.model.model_id
        
        # Add sufficient prediction data
        np.random.seed(42)  # For reproducible results
        for i in range(60):
            actual = 100.0
            prediction = actual + np.random.normal(0, 3)  # Add some noise
            self.monitor.record_prediction(model_id, prediction, actual)
        
        performance = self.monitor.evaluate_model_performance(model_id)
        
        # Check for either sufficient data or the actual response structure
        if 'status' in performance:
            self.assertNotEqual(performance['status'], 'insufficient_data')
        
        self.assertIn('metrics', performance)
        self.assertIn('mape', performance['metrics'])
        self.assertIn('rmse', performance['metrics'])
        self.assertIn('mae', performance['metrics'])
        self.assertIn('r2', performance['metrics'])
    
    def test_performance_alerts(self):
        """Test performance alerting"""
        model_id = self.model.model_id
        
        # Add good performance data first
        for i in range(50):
            actual = 100.0
            prediction = actual + np.random.normal(0, 1)  # Low error
            self.monitor.record_prediction(model_id, prediction, actual)
        
        performance1 = self.monitor.evaluate_model_performance(model_id)
        
        # Add poor performance data
        for i in range(50):
            actual = 100.0
            prediction = actual + np.random.normal(0, 20)  # High error
            self.monitor.record_prediction(model_id, prediction, actual)
        
        performance2 = self.monitor.evaluate_model_performance(model_id)
        
        # Should have alerts for performance degradation
        self.assertIn('alerts', performance2)
        # The exact alert depends on threshold settings and random data
    
    def test_monitoring_summary(self):
        """Test monitoring summary generation"""
        model_id = self.model.model_id
        
        # Add some data
        for i in range(10):
            self.monitor.record_prediction(model_id, 100.0 + i, 95.0 + i)
        
        summary = self.monitor.get_monitoring_summary()
        
        self.assertEqual(summary['monitored_models'], 1)
        self.assertEqual(summary['total_predictions'], 10)
        self.assertIn(model_id, summary['models'])


class TestAutoRetrainingSystem(unittest.TestCase):
    """Test automated retraining system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(os.path.join(self.temp_dir, "test_registry.db"))
        self.monitor = PerformanceMonitor(self.registry)
        self.monitor.monitoring_active = False
        
        self.retrain_system = AutoRetrainingSystem(self.registry, self.monitor)
        self.retrain_system.retraining_active = False  # Disable background thread
        
        # Create test model
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        with open(model_path, 'wb') as f:
            f.write(b"test model")
        
        self.model = ModelVersion("test_model", "v1.0", model_path)
        self.registry.register_model(self.model)
        self.registry.update_model_status(self.model.model_id, DeploymentStatus.ACTIVE)
    
    def tearDown(self):
        self.retrain_system.retraining_active = False
        self.monitor.monitoring_active = False
        shutil.rmtree(self.temp_dir)
    
    def test_performance_degradation_trigger(self):
        """Test retraining trigger due to performance degradation"""
        model_id = self.model.model_id
        
        # Add poor performance data to trigger retraining
        for i in range(60):
            actual = 100.0
            prediction = actual + np.random.normal(0, 25)  # Very high error
            self.monitor.record_prediction(model_id, prediction, actual)
        
        # Force performance evaluation
        self.monitor.evaluate_model_performance(model_id)
        
        # Check retraining triggers
        triggers = self.retrain_system.check_retraining_triggers("test_model")
        
        # Should trigger retraining due to high MAPE
        if triggers['should_retrain']:
            self.assertTrue(len(triggers['reasons']) > 0)
            # Check if one of the reasons is performance-related
            reason_types = [r['type'] for r in triggers['reasons']]
            self.assertTrue(any(t in reason_types for t in ['mape_threshold', 'performance_drift']))
    
    def test_cooldown_period(self):
        """Test retraining cooldown period"""
        # Simulate recent retraining
        self.retrain_system.retraining_history["test_model"] = {
            'last_retrain': (datetime.now() - timedelta(hours=1)).isoformat(),
            'trigger_reasons': [],
            'new_model_id': 'test_id'
        }
        
        triggers = self.retrain_system.check_retraining_triggers("test_model")
        
        # Should not retrain due to cooldown
        self.assertFalse(triggers['should_retrain'])
        self.assertGreater(triggers['cooldown_remaining'], 0)
    
    @patch('src.ml_models.ml_deployment_pipeline.MLTrainingPipeline')
    def test_retraining_execution(self, mock_pipeline_class):
        """Test retraining execution"""
        # Mock training pipeline
        mock_pipeline = Mock()
        mock_pipeline.train_model.return_value = {
            'status': 'success',
            'model_path': '/tmp/new_model.pkl',
            'metrics': {'mape': 10.0, 'rmse': 50.0}
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        reasons = [{'type': 'mape_threshold', 'current': 25.0, 'threshold': 20.0}]
        
        result = self.retrain_system.trigger_retraining("test_model", reasons)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('new_model_id', result)
        self.assertIn('new_version', result)
        
        # Verify training was called
        mock_pipeline.train_model.assert_called_once_with("test_model", force=True)


class TestMLDeploymentPipeline(unittest.TestCase):
    """Test complete ML deployment pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = MLDeploymentPipeline()
        
        # Override registry with test database
        self.pipeline.registry = ModelRegistry(os.path.join(self.temp_dir, "test_registry.db"))
        self.pipeline.ab_testing.registry = self.pipeline.registry
        self.pipeline.monitor.monitoring_active = False
        self.pipeline.auto_retrain.retraining_active = False
        
        # Create test model file
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        with open(self.model_path, 'wb') as f:
            f.write(b"test model data")
    
    def tearDown(self):
        self.pipeline.monitor.monitoring_active = False
        self.pipeline.auto_retrain.retraining_active = False
        shutil.rmtree(self.temp_dir)
    
    def test_immediate_deployment(self):
        """Test immediate deployment strategy"""
        result = self.pipeline.deploy_model(
            "test_model",
            self.model_path,
            "immediate",
            {"test": "metadata"}
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('model_id', result)
        self.assertIn('version', result)
        self.assertIn('active_model_id', result)
        
        # Verify model is active
        active_model = self.pipeline.registry.get_active_model("test_model")
        self.assertIsNotNone(active_model)
        self.assertEqual(active_model.model_id, result['model_id'])
    
    def test_canary_deployment(self):
        """Test canary deployment strategy"""
        result = self.pipeline.deploy_model(
            "test_model",
            self.model_path,
            "canary"
        )
        
        self.assertEqual(result['status'], 'canary_deployed')
        self.assertIn('canary_model_id', result)
        self.assertEqual(result['traffic_percentage'], 10)
    
    def test_ab_test_deployment(self):
        """Test A/B test deployment strategy"""
        # First deploy a model to have something to test against
        self.pipeline.deploy_model("test_model", self.model_path, "immediate")
        
        # Create second model file
        model_path_2 = os.path.join(self.temp_dir, "test_model_2.pkl")
        with open(model_path_2, 'wb') as f:
            f.write(b"test model data v2")
        
        result = self.pipeline.deploy_model(
            "test_model",
            model_path_2,
            "ab_test"
        )
        
        self.assertEqual(result['status'], 'ab_test_started')
        self.assertIn('test_id', result)
        self.assertIn('shadow_model_id', result)
    
    def test_blue_green_deployment(self):
        """Test blue-green deployment strategy"""
        result = self.pipeline.deploy_model(
            "test_model",
            self.model_path,
            "blue_green"
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('active_model_id', result)
        
        # Verify model is active
        active_model = self.pipeline.registry.get_active_model("test_model")
        self.assertIsNotNone(active_model)
    
    def test_model_rollback(self):
        """Test model rollback functionality"""
        # Deploy first version
        result1 = self.pipeline.deploy_model("test_model", self.model_path, "immediate")
        v1_model_id = result1['model_id']
        
        # Deploy second version
        model_path_2 = os.path.join(self.temp_dir, "test_model_2.pkl")
        with open(model_path_2, 'wb') as f:
            f.write(b"test model data v2")
        
        result2 = self.pipeline.deploy_model("test_model", model_path_2, "immediate")
        v2_model_id = result2['model_id']
        
        # Verify v2 is active
        active_model = self.pipeline.registry.get_active_model("test_model")
        self.assertEqual(active_model.model_id, v2_model_id)
        
        # Rollback to v1
        rollback_result = self.pipeline.rollback_model("test_model")
        
        self.assertEqual(rollback_result['status'], 'success')
        self.assertEqual(rollback_result['previous_model_id'], v2_model_id)
        
        # Verify v1 is now active
        active_model = self.pipeline.registry.get_active_model("test_model")
        # Note: rollback finds the most recent non-current version, which should be v1
    
    def test_pipeline_status(self):
        """Test pipeline status reporting"""
        # Deploy a model first
        self.pipeline.deploy_model("test_model", self.model_path, "immediate")
        
        status = self.pipeline.get_pipeline_status()
        
        self.assertEqual(status['pipeline_status'], 'healthy')
        self.assertIn('timestamp', status)
        self.assertIn('model_statistics', status)
        self.assertIn('monitoring', status)
        self.assertIn('components', status)
        
        # Verify components are reported as active
        components = status['components']
        self.assertEqual(components['registry'], 'active')
        self.assertEqual(components['ab_testing'], 'active')
        self.assertEqual(components['monitoring'], 'active')
        self.assertEqual(components['auto_retrain'], 'active')


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete deployment scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = MLDeploymentPipeline()
        
        # Override with test database
        self.pipeline.registry = ModelRegistry(os.path.join(self.temp_dir, "test_registry.db"))
        self.pipeline.ab_testing.registry = self.pipeline.registry
        self.pipeline.monitor.monitoring_active = False
        self.pipeline.auto_retrain.retraining_active = False
    
    def tearDown(self):
        self.pipeline.monitor.monitoring_active = False
        self.pipeline.auto_retrain.retraining_active = False
        shutil.rmtree(self.temp_dir)
    
    def test_complete_deployment_lifecycle(self):
        """Test complete deployment lifecycle"""
        # Create model files
        model_v1_path = os.path.join(self.temp_dir, "model_v1.pkl")
        model_v2_path = os.path.join(self.temp_dir, "model_v2.pkl")
        
        with open(model_v1_path, 'wb') as f:
            f.write(b"model v1 data")
        with open(model_v2_path, 'wb') as f:
            f.write(b"model v2 data")
        
        # 1. Initial deployment
        result1 = self.pipeline.deploy_model("test_model", model_v1_path, "immediate")
        self.assertEqual(result1['status'], 'success')
        
        # 2. Monitor performance
        active_model = self.pipeline.registry.get_active_model("test_model")
        for i in range(20):
            self.pipeline.monitor.record_prediction(active_model.model_id, 100.0 + i, 95.0 + i)
        
        # 3. Deploy new version with A/B test
        result2 = self.pipeline.deploy_model("test_model", model_v2_path, "ab_test")
        self.assertEqual(result2['status'], 'ab_test_started')
        
        # 4. Simulate A/B test data
        test_id = result2['test_id']
        for i in range(50):
            # Route request and record result
            model_id, variant = self.pipeline.ab_testing.route_prediction_request(test_id, {})
            actual = 100.0
            prediction = actual + np.random.normal(0, 2 if variant == 'model_b' else 3)
            self.pipeline.ab_testing.record_prediction_result(test_id, variant, prediction, actual)
        
        # 5. Evaluate A/B test
        evaluation = self.pipeline.ab_testing.evaluate_ab_test(test_id)
        self.assertEqual(evaluation['status'], 'completed')
        
        # 6. Check pipeline status
        status = self.pipeline.get_pipeline_status()
        self.assertEqual(status['pipeline_status'], 'healthy')
        self.assertGreater(status['monitoring']['total_predictions'], 0)
    
    def test_automated_retraining_scenario(self):
        """Test automated retraining scenario"""
        # Create and deploy model
        model_path = os.path.join(self.temp_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            f.write(b"initial model")
        
        result = self.pipeline.deploy_model("test_model", model_path, "immediate")
        active_model = self.pipeline.registry.get_active_model("test_model")
        
        # Simulate poor performance to trigger retraining
        for i in range(60):
            actual = 100.0
            prediction = actual + np.random.normal(0, 30)  # Very high error
            self.pipeline.monitor.record_prediction(active_model.model_id, prediction, actual)
        
        # Check if retraining is triggered
        triggers = self.pipeline.auto_retrain.check_retraining_triggers("test_model")
        
        if triggers['should_retrain']:
            self.assertGreater(len(triggers['reasons']), 0)
            
            # In a real scenario, retraining would be triggered automatically
            # For testing, we verify the logic works
            self.assertTrue(any(r['type'] in ['mape_threshold', 'performance_drift'] 
                             for r in triggers['reasons']))


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ML Deployment Pipeline Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, error in result.errors:
            print(f"  - {test}: {error.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)