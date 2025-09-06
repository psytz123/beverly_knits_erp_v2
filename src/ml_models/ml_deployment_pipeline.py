#!/usr/bin/env python3
"""
ML Model Deployment Pipeline for Beverly Knits ERP
Complete deployment infrastructure with versioning, A/B testing, monitoring, and automated retraining
Created: 2025-09-06
"""

import sys
import os
import json
import pickle
import shutil
import logging
import hashlib
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sqlite3
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.ml_config import MODEL_REGISTRY, ML_GLOBAL_CONFIG, MODEL_PATH, TRAINING_PATH
    from utils.cache_manager import UnifiedCacheManager
except ImportError:
    MODEL_PATH = Path(__file__).parent.parent.parent / "models"
    TRAINING_PATH = Path(__file__).parent.parent.parent / "training_results"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Model deployment status"""
    PENDING = "pending"
    ACTIVE = "active"
    CANARY = "canary"
    SHADOW = "shadow"
    DEPRECATED = "deprecated"
    ROLLBACK = "rollback"
    FAILED = "failed"


class ModelVersion:
    """Model version metadata"""
    
    def __init__(self, model_name: str, version: str, model_path: str, 
                 metadata: Dict[str, Any] = None):
        self.model_name = model_name
        self.version = version
        self.model_path = model_path
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.status = DeploymentStatus.PENDING
        self.performance_metrics = {}
        self.deployment_config = {}
        
        # Generate unique ID
        self.model_id = str(uuid.uuid4())
        
        # Calculate model hash for integrity
        self.model_hash = self._calculate_model_hash()
    
    def _calculate_model_hash(self) -> str:
        """Calculate hash of model file for integrity checking"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'model_path': self.model_path,
            'model_hash': self.model_hash,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'performance_metrics': self.performance_metrics,
            'deployment_config': self.deployment_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        version = cls(
            model_name=data['model_name'],
            version=data['version'],
            model_path=data['model_path'],
            metadata=data.get('metadata', {})
        )
        version.model_id = data.get('model_id', str(uuid.uuid4()))
        version.model_hash = data.get('model_hash', '')
        version.created_at = datetime.fromisoformat(data['created_at'])
        version.status = DeploymentStatus(data.get('status', 'pending'))
        version.performance_metrics = data.get('performance_metrics', {})
        version.deployment_config = data.get('deployment_config', {})
        return version


class ModelRegistry:
    """Model version registry with SQLite backend"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(MODEL_PATH / "model_registry.db")
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_hash TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    status TEXT,
                    performance_metrics TEXT,
                    deployment_config TEXT,
                    UNIQUE(model_name, version)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    action TEXT,
                    timestamp TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_versions (model_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    model_a_id TEXT,
                    model_b_id TEXT,
                    metric_name TEXT,
                    model_a_score REAL,
                    model_b_score REAL,
                    winner TEXT,
                    confidence REAL,
                    test_start TIMESTAMP,
                    test_end TIMESTAMP,
                    FOREIGN KEY (model_a_id) REFERENCES model_versions (model_id),
                    FOREIGN KEY (model_b_id) REFERENCES model_versions (model_id)
                )
            ''')
    
    def register_model(self, model_version: ModelVersion) -> bool:
        """Register new model version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO model_versions 
                    (model_id, model_name, version, model_path, model_hash, 
                     metadata, created_at, status, performance_metrics, deployment_config)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_version.model_id,
                    model_version.model_name,
                    model_version.version,
                    model_version.model_path,
                    model_version.model_hash,
                    json.dumps(model_version.metadata),
                    model_version.created_at.isoformat(),
                    model_version.status.value,
                    json.dumps(model_version.performance_metrics),
                    json.dumps(model_version.deployment_config)
                ))
                
                # Log registration
                self.log_deployment_action(model_version.model_id, "registered", {
                    "version": model_version.version,
                    "path": model_version.model_path
                })
                
            logger.info(f"Registered model {model_version.model_name} v{model_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        versions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM model_versions 
                WHERE model_name = ? 
                ORDER BY created_at DESC
            ''', (model_name,))
            
            for row in cursor.fetchall():
                version_data = {
                    'model_id': row[0],
                    'model_name': row[1],
                    'version': row[2],
                    'model_path': row[3],
                    'model_hash': row[4],
                    'metadata': json.loads(row[5] or '{}'),
                    'created_at': row[6],
                    'status': row[7],
                    'performance_metrics': json.loads(row[8] or '{}'),
                    'deployment_config': json.loads(row[9] or '{}')
                }
                versions.append(ModelVersion.from_dict(version_data))
        
        return versions
    
    def get_active_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get currently active model version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM model_versions 
                WHERE model_name = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            ''', (model_name,))
            
            row = cursor.fetchone()
            if row:
                version_data = {
                    'model_id': row[0],
                    'model_name': row[1],
                    'version': row[2],
                    'model_path': row[3],
                    'model_hash': row[4],
                    'metadata': json.loads(row[5] or '{}'),
                    'created_at': row[6],
                    'status': row[7],
                    'performance_metrics': json.loads(row[8] or '{}'),
                    'deployment_config': json.loads(row[9] or '{}')
                }
                return ModelVersion.from_dict(version_data)
        
        return None
    
    def update_model_status(self, model_id: str, status: DeploymentStatus, 
                           details: Dict[str, Any] = None) -> bool:
        """Update model deployment status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE model_versions 
                    SET status = ? 
                    WHERE model_id = ?
                ''', (status.value, model_id))
                
                self.log_deployment_action(model_id, f"status_changed_to_{status.value}", details or {})
            
            logger.info(f"Updated model {model_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model status: {e}")
            return False
    
    def log_deployment_action(self, model_id: str, action: str, details: Dict[str, Any]):
        """Log deployment action"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO deployment_history (model_id, action, timestamp, details)
                VALUES (?, ?, ?, ?)
            ''', (model_id, action, datetime.now().isoformat(), json.dumps(details)))


class ABTestFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_tests = {}
        self.test_results = {}
    
    def start_ab_test(self, model_a_id: str, model_b_id: str, 
                      test_config: Dict[str, Any]) -> str:
        """Start A/B test between two models"""
        test_id = str(uuid.uuid4())
        
        test_data = {
            'test_id': test_id,
            'model_a_id': model_a_id,
            'model_b_id': model_b_id,
            'config': test_config,
            'start_time': datetime.now(),
            'traffic_split': test_config.get('traffic_split', 0.5),
            'duration_hours': test_config.get('duration_hours', 24),
            'metrics': test_config.get('metrics', ['mape', 'rmse']),
            'min_samples': test_config.get('min_samples', 100),
            'status': 'running',
            'results': {'model_a': {}, 'model_b': {}, 'predictions': []}
        }
        
        self.active_tests[test_id] = test_data
        
        logger.info(f"Started A/B test {test_id}: {model_a_id} vs {model_b_id}")
        
        return test_id
    
    def route_prediction_request(self, test_id: str, request_data: Dict[str, Any]) -> Tuple[str, str]:
        """Route prediction request to appropriate model for A/B test"""
        if test_id not in self.active_tests:
            return None, "Test not found"
        
        test = self.active_tests[test_id]
        
        # Simple random routing based on traffic split
        if np.random.random() < test['traffic_split']:
            return test['model_a_id'], 'model_a'
        else:
            return test['model_b_id'], 'model_b'
    
    def record_prediction_result(self, test_id: str, model_variant: str, 
                                prediction: float, actual: float, request_id: str = None):
        """Record prediction result for A/B test"""
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        result_record = {
            'request_id': request_id or str(uuid.uuid4()),
            'model_variant': model_variant,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now().isoformat(),
            'error': abs(prediction - actual),
            'relative_error': abs(prediction - actual) / abs(actual) if actual != 0 else float('inf')
        }
        
        test['results']['predictions'].append(result_record)
        
        # Update running metrics
        variant_predictions = [p for p in test['results']['predictions'] 
                              if p['model_variant'] == model_variant]
        
        if len(variant_predictions) > 0:
            predictions = [p['prediction'] for p in variant_predictions]
            actuals = [p['actual'] for p in variant_predictions]
            
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mape = np.mean([p['relative_error'] for p in variant_predictions]) * 100
            
            test['results'][model_variant] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'sample_count': len(variant_predictions),
                'last_updated': datetime.now().isoformat()
            }
    
    def evaluate_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Evaluate A/B test results"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test = self.active_tests[test_id]
        
        # Check if test has enough samples
        model_a_samples = len([p for p in test['results']['predictions'] 
                              if p['model_variant'] == 'model_a'])
        model_b_samples = len([p for p in test['results']['predictions'] 
                              if p['model_variant'] == 'model_b'])
        
        if model_a_samples < test['min_samples'] or model_b_samples < test['min_samples']:
            return {
                'status': 'insufficient_data',
                'model_a_samples': model_a_samples,
                'model_b_samples': model_b_samples,
                'min_required': test['min_samples']
            }
        
        results_a = test['results']['model_a']
        results_b = test['results']['model_b']
        
        # Statistical significance testing (simplified)
        significant_metrics = []
        winner_metrics = {}
        
        for metric in test['metrics']:
            if metric in results_a and metric in results_b:
                score_a = results_a[metric]
                score_b = results_b[metric]
                
                # Lower is better for error metrics
                improvement = (score_a - score_b) / score_a * 100
                winner = 'model_b' if improvement > 0 else 'model_a'
                
                # Simple threshold for significance (5% improvement)
                is_significant = abs(improvement) > 5.0
                
                if is_significant:
                    significant_metrics.append(metric)
                
                winner_metrics[metric] = {
                    'model_a_score': score_a,
                    'model_b_score': score_b,
                    'improvement_percent': improvement,
                    'winner': winner,
                    'significant': is_significant
                }
        
        # Determine overall winner
        significant_wins_a = sum(1 for m in winner_metrics.values() 
                               if m['winner'] == 'model_a' and m['significant'])
        significant_wins_b = sum(1 for m in winner_metrics.values() 
                               if m['winner'] == 'model_b' and m['significant'])
        
        if significant_wins_a > significant_wins_b:
            overall_winner = 'model_a'
            confidence = significant_wins_a / len(significant_metrics) if significant_metrics else 0.5
        elif significant_wins_b > significant_wins_a:
            overall_winner = 'model_b'
            confidence = significant_wins_b / len(significant_metrics) if significant_metrics else 0.5
        else:
            overall_winner = 'tie'
            confidence = 0.5
        
        evaluation = {
            'test_id': test_id,
            'status': 'completed',
            'overall_winner': overall_winner,
            'confidence': confidence,
            'metric_results': winner_metrics,
            'significant_metrics': significant_metrics,
            'test_duration_hours': (datetime.now() - test['start_time']).total_seconds() / 3600,
            'total_samples': model_a_samples + model_b_samples,
            'evaluated_at': datetime.now().isoformat()
        }
        
        # Store results in database
        self._store_ab_test_results(evaluation)
        
        return evaluation
    
    def _store_ab_test_results(self, evaluation: Dict[str, Any]):
        """Store A/B test results in database"""
        test = self.active_tests.get(evaluation['test_id'])
        if not test:
            return
        
        with sqlite3.connect(self.registry.db_path) as conn:
            for metric, result in evaluation['metric_results'].items():
                conn.execute('''
                    INSERT INTO ab_test_results 
                    (test_id, model_a_id, model_b_id, metric_name, 
                     model_a_score, model_b_score, winner, confidence, 
                     test_start, test_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    evaluation['test_id'],
                    test['model_a_id'],
                    test['model_b_id'],
                    metric,
                    result['model_a_score'],
                    result['model_b_score'],
                    result['winner'],
                    evaluation['confidence'],
                    test['start_time'].isoformat(),
                    datetime.now().isoformat()
                ))


class PerformanceMonitor:
    """Real-time model performance monitoring"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.monitoring_data = {}
        self.alerts = []
        self.thresholds = {
            'mape_threshold': 15.0,  # Alert if MAPE > 15%
            'rmse_threshold': 100.0,  # Alert if RMSE > 100
            'drift_threshold': 0.1,   # Alert if performance degrades by 10%
            'min_predictions': 50     # Minimum predictions for reliable monitoring
        }
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def record_prediction(self, model_id: str, prediction: float, 
                         actual: float = None, features: Dict[str, Any] = None):
        """Record a model prediction for monitoring"""
        if model_id not in self.monitoring_data:
            self.monitoring_data[model_id] = {
                'predictions': [],
                'performance_history': [],
                'alerts': [],
                'last_evaluated': None
            }
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'features': features or {},
            'error': abs(prediction - actual) if actual is not None else None,
            'relative_error': abs(prediction - actual) / abs(actual) if actual is not None and actual != 0 else None
        }
        
        self.monitoring_data[model_id]['predictions'].append(record)
        
        # Keep only last 1000 predictions
        if len(self.monitoring_data[model_id]['predictions']) > 1000:
            self.monitoring_data[model_id]['predictions'] = \
                self.monitoring_data[model_id]['predictions'][-1000:]
    
    def evaluate_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Evaluate current model performance"""
        if model_id not in self.monitoring_data:
            return {'error': 'No monitoring data available'}
        
        data = self.monitoring_data[model_id]
        
        # Filter predictions with actual values
        valid_predictions = [p for p in data['predictions'] 
                           if p['actual'] is not None and p['error'] is not None]
        
        if len(valid_predictions) < self.thresholds['min_predictions']:
            return {
                'status': 'insufficient_data',
                'prediction_count': len(valid_predictions),
                'min_required': self.thresholds['min_predictions']
            }
        
        # Calculate metrics for recent period
        recent_predictions = valid_predictions[-100:]  # Last 100 predictions
        
        predictions = [p['prediction'] for p in recent_predictions]
        actuals = [p['actual'] for p in recent_predictions]
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean([p['relative_error'] for p in recent_predictions]) * 100
        r2 = r2_score(actuals, predictions)
        
        performance = {
            'model_id': model_id,
            'evaluation_time': datetime.now().isoformat(),
            'sample_count': len(recent_predictions),
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            },
            'thresholds': self.thresholds
        }
        
        # Check for performance degradation
        performance['alerts'] = self._check_performance_alerts(model_id, performance)
        
        # Store performance history
        data['performance_history'].append(performance)
        data['last_evaluated'] = datetime.now()
        
        # Keep only last 100 evaluations
        if len(data['performance_history']) > 100:
            data['performance_history'] = data['performance_history'][-100:]
        
        return performance
    
    def _check_performance_alerts(self, model_id: str, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        metrics = current_performance['metrics']
        
        # Threshold alerts
        if metrics['mape'] > self.thresholds['mape_threshold']:
            alerts.append({
                'type': 'threshold_exceeded',
                'metric': 'mape',
                'value': metrics['mape'],
                'threshold': self.thresholds['mape_threshold'],
                'severity': 'high'
            })
        
        if metrics['rmse'] > self.thresholds['rmse_threshold']:
            alerts.append({
                'type': 'threshold_exceeded',
                'metric': 'rmse',
                'value': metrics['rmse'],
                'threshold': self.thresholds['rmse_threshold'],
                'severity': 'medium'
            })
        
        # Performance drift alerts
        data = self.monitoring_data[model_id]
        if len(data['performance_history']) > 1:
            previous = data['performance_history'][-2]['metrics']
            
            for metric in ['mape', 'rmse']:
                current_val = metrics[metric]
                previous_val = previous[metric]
                
                # Check for degradation (higher is worse for these metrics)
                if previous_val > 0:
                    change_rate = (current_val - previous_val) / previous_val
                    
                    if change_rate > self.thresholds['drift_threshold']:
                        alerts.append({
                            'type': 'performance_drift',
                            'metric': metric,
                            'current_value': current_val,
                            'previous_value': previous_val,
                            'change_rate': change_rate,
                            'threshold': self.thresholds['drift_threshold'],
                            'severity': 'high'
                        })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            alert['model_id'] = model_id
            data['alerts'].append(alert)
        
        return alerts
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Evaluate all models with monitoring data
                for model_id in list(self.monitoring_data.keys()):
                    data = self.monitoring_data[model_id]
                    
                    # Check if enough time has passed since last evaluation
                    if data['last_evaluated'] is None or \
                       (datetime.now() - data['last_evaluated']).total_seconds() > 3600:  # 1 hour
                        
                        performance = self.evaluate_model_performance(model_id)
                        
                        if 'alerts' in performance and performance['alerts']:
                            logger.warning(f"Performance alerts for model {model_id}: "
                                         f"{len(performance['alerts'])} alerts")
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary for all models"""
        summary = {
            'monitored_models': len(self.monitoring_data),
            'active_alerts': 0,
            'total_predictions': 0,
            'models': {}
        }
        
        for model_id, data in self.monitoring_data.items():
            recent_alerts = [a for a in data['alerts'] 
                           if datetime.fromisoformat(a['timestamp']) > 
                           datetime.now() - timedelta(hours=24)]
            
            summary['active_alerts'] += len(recent_alerts)
            summary['total_predictions'] += len(data['predictions'])
            
            latest_performance = data['performance_history'][-1] if data['performance_history'] else None
            
            summary['models'][model_id] = {
                'prediction_count': len(data['predictions']),
                'recent_alerts': len(recent_alerts),
                'last_evaluation': data['last_evaluated'].isoformat() if data['last_evaluated'] else None,
                'latest_metrics': latest_performance['metrics'] if latest_performance else None
            }
        
        return summary


class AutoRetrainingSystem:
    """Automated model retraining trigger system"""
    
    def __init__(self, registry: ModelRegistry, monitor: PerformanceMonitor):
        self.registry = registry
        self.monitor = monitor
        self.retraining_active = True
        self.retraining_rules = {
            'performance_degradation': {
                'enabled': True,
                'mape_threshold': 20.0,  # Trigger if MAPE > 20%
                'drift_threshold': 0.15,  # Trigger if 15% performance drop
                'cooldown_hours': 24      # Don't retrain more than once per day
            },
            'scheduled': {
                'enabled': True,
                'frequency_days': 7,      # Weekly retraining
                'day_of_week': 0,         # Monday
                'hour': 2                 # 2 AM
            },
            'data_drift': {
                'enabled': True,
                'threshold': 0.1          # Feature drift threshold
            }
        }
        
        # Track retraining history
        self.retraining_history = {}
        
        # Start retraining monitoring thread
        self.retraining_thread = threading.Thread(target=self._retraining_loop, daemon=True)
        self.retraining_thread.start()
    
    def check_retraining_triggers(self, model_name: str) -> Dict[str, Any]:
        """Check if model needs retraining"""
        triggers = {
            'should_retrain': False,
            'reasons': [],
            'last_retrain': None,
            'cooldown_remaining': 0
        }
        
        # Get model information
        active_model = self.registry.get_active_model(model_name)
        if not active_model:
            return triggers
        
        model_id = active_model.model_id
        
        # Check cooldown period
        last_retrain = self.retraining_history.get(model_name, {}).get('last_retrain')
        if last_retrain:
            last_retrain_time = datetime.fromisoformat(last_retrain)
            cooldown_hours = self.retraining_rules['performance_degradation']['cooldown_hours']
            cooldown_remaining = cooldown_hours - (datetime.now() - last_retrain_time).total_seconds() / 3600
            
            if cooldown_remaining > 0:
                triggers['cooldown_remaining'] = cooldown_remaining
                return triggers
            
            triggers['last_retrain'] = last_retrain
        
        # Performance degradation triggers
        if self.retraining_rules['performance_degradation']['enabled']:
            if model_id in self.monitor.monitoring_data:
                data = self.monitor.monitoring_data[model_id]
                
                if data['performance_history']:
                    latest_performance = data['performance_history'][-1]
                    
                    # Check MAPE threshold
                    if latest_performance['metrics']['mape'] > self.retraining_rules['performance_degradation']['mape_threshold']:
                        triggers['should_retrain'] = True
                        triggers['reasons'].append({
                            'type': 'mape_threshold',
                            'current': latest_performance['metrics']['mape'],
                            'threshold': self.retraining_rules['performance_degradation']['mape_threshold']
                        })
                    
                    # Check performance drift
                    if len(data['performance_history']) > 1:
                        previous = data['performance_history'][-2]
                        
                        for metric in ['mape', 'rmse']:
                            current_val = latest_performance['metrics'][metric]
                            baseline_val = previous['metrics'][metric]
                            
                            if baseline_val > 0:
                                drift = (current_val - baseline_val) / baseline_val
                                
                                if drift > self.retraining_rules['performance_degradation']['drift_threshold']:
                                    triggers['should_retrain'] = True
                                    triggers['reasons'].append({
                                        'type': 'performance_drift',
                                        'metric': metric,
                                        'drift': drift,
                                        'threshold': self.retraining_rules['performance_degradation']['drift_threshold']
                                    })
        
        # Scheduled retraining triggers
        if self.retraining_rules['scheduled']['enabled']:
            now = datetime.now()
            
            # Check if it's the scheduled day and time
            if (now.weekday() == self.retraining_rules['scheduled']['day_of_week'] and
                now.hour == self.retraining_rules['scheduled']['hour']):
                
                # Check if we haven't already retrained today
                if not last_retrain or \
                   datetime.fromisoformat(last_retrain).date() != now.date():
                    triggers['should_retrain'] = True
                    triggers['reasons'].append({
                        'type': 'scheduled',
                        'schedule': f"Weekly on {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][self.retraining_rules['scheduled']['day_of_week']]} at {self.retraining_rules['scheduled']['hour']}:00"
                    })
        
        return triggers
    
    def trigger_retraining(self, model_name: str, reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trigger model retraining"""
        result = {
            'model_name': model_name,
            'triggered_at': datetime.now().isoformat(),
            'reasons': reasons,
            'status': 'failed'
        }
        
        try:
            # Import training pipeline
            from scripts.ml_training_pipeline import MLTrainingPipeline
            
            # Initialize training pipeline
            pipeline = MLTrainingPipeline()
            
            # Train model
            training_result = pipeline.train_model(model_name, force=True)
            
            if training_result.get('status') == 'success':
                # Register new model version
                version = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                model_version = ModelVersion(
                    model_name=model_name,
                    version=version,
                    model_path=training_result['model_path'],
                    metadata={
                        'auto_retrain': True,
                        'trigger_reasons': reasons,
                        'training_metrics': training_result.get('metrics', {}),
                        'parent_model': self.registry.get_active_model(model_name).version if self.registry.get_active_model(model_name) else None
                    }
                )
                
                # Register and activate new model
                if self.registry.register_model(model_version):
                    self.registry.update_model_status(model_version.model_id, DeploymentStatus.ACTIVE)
                    
                    # Deactivate old model
                    old_model = self.registry.get_active_model(model_name)
                    if old_model and old_model.model_id != model_version.model_id:
                        self.registry.update_model_status(old_model.model_id, DeploymentStatus.DEPRECATED)
                    
                    result['status'] = 'success'
                    result['new_model_id'] = model_version.model_id
                    result['new_version'] = version
                    result['metrics'] = training_result.get('metrics', {})
                    
                    # Update retraining history
                    self.retraining_history[model_name] = {
                        'last_retrain': datetime.now().isoformat(),
                        'trigger_reasons': reasons,
                        'new_model_id': model_version.model_id
                    }
                    
                    logger.info(f"Auto-retraining successful for {model_name}: {version}")
                
            else:
                result['error'] = training_result.get('error', 'Training failed')
                logger.error(f"Auto-retraining failed for {model_name}: {result['error']}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error during auto-retraining of {model_name}: {e}")
        
        return result
    
    def _retraining_loop(self):
        """Background retraining monitoring loop"""
        while self.retraining_active:
            try:
                # Get all model names from registry
                all_models = set()
                with sqlite3.connect(self.registry.db_path) as conn:
                    cursor = conn.execute('SELECT DISTINCT model_name FROM model_versions')
                    all_models = {row[0] for row in cursor.fetchall()}
                
                # Check each model for retraining triggers
                for model_name in all_models:
                    triggers = self.check_retraining_triggers(model_name)
                    
                    if triggers['should_retrain']:
                        logger.info(f"Retraining triggered for {model_name}: "
                                  f"{len(triggers['reasons'])} reasons")
                        
                        # Trigger retraining
                        result = self.trigger_retraining(model_name, triggers['reasons'])
                        
                        if result['status'] == 'success':
                            logger.info(f"Auto-retraining completed successfully for {model_name}")
                        else:
                            logger.error(f"Auto-retraining failed for {model_name}: "
                                       f"{result.get('error', 'Unknown error')}")
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in retraining loop: {e}")
                time.sleep(300)  # Sleep 5 minutes on error


class MLDeploymentPipeline:
    """Complete ML deployment pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.registry = ModelRegistry()
        self.ab_testing = ABTestFramework(self.registry)
        self.monitor = PerformanceMonitor(self.registry)
        self.auto_retrain = AutoRetrainingSystem(self.registry, self.monitor)
        
        logger.info("ML Deployment Pipeline initialized")
    
    def deploy_model(self, model_name: str, model_path: str, 
                    deployment_strategy: str = "blue_green",
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy a model with specified strategy"""
        result = {
            'model_name': model_name,
            'deployment_strategy': deployment_strategy,
            'status': 'failed',
            'deployed_at': datetime.now().isoformat()
        }
        
        try:
            # Create model version
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                model_path=model_path,
                metadata=metadata or {}
            )
            
            # Register model
            if not self.registry.register_model(model_version):
                result['error'] = 'Failed to register model'
                return result
            
            # Apply deployment strategy
            if deployment_strategy == "immediate":
                # Immediate deployment - activate right away
                self.registry.update_model_status(model_version.model_id, DeploymentStatus.ACTIVE)
                
                # Deactivate old model
                old_model = self.registry.get_active_model(model_name)
                if old_model and old_model.model_id != model_version.model_id:
                    self.registry.update_model_status(old_model.model_id, DeploymentStatus.DEPRECATED)
                
                result['status'] = 'success'
                result['active_model_id'] = model_version.model_id
                
            elif deployment_strategy == "canary":
                # Canary deployment - start with small traffic percentage
                self.registry.update_model_status(model_version.model_id, DeploymentStatus.CANARY)
                
                result['status'] = 'canary_deployed'
                result['canary_model_id'] = model_version.model_id
                result['traffic_percentage'] = 10  # Start with 10% traffic
                
            elif deployment_strategy == "ab_test":
                # A/B test deployment
                old_model = self.registry.get_active_model(model_name)
                if old_model:
                    test_id = self.ab_testing.start_ab_test(
                        old_model.model_id,
                        model_version.model_id,
                        {
                            'duration_hours': 24,
                            'traffic_split': 0.5,
                            'metrics': ['mape', 'rmse'],
                            'min_samples': 100
                        }
                    )
                    
                    self.registry.update_model_status(model_version.model_id, DeploymentStatus.SHADOW)
                    
                    result['status'] = 'ab_test_started'
                    result['test_id'] = test_id
                    result['shadow_model_id'] = model_version.model_id
                else:
                    # No existing model for A/B test, deploy immediately
                    self.registry.update_model_status(model_version.model_id, DeploymentStatus.ACTIVE)
                    result['status'] = 'success'
                    result['active_model_id'] = model_version.model_id
                    
            elif deployment_strategy == "blue_green":
                # Blue-green deployment - default strategy
                self.registry.update_model_status(model_version.model_id, DeploymentStatus.PENDING)
                
                # In a real implementation, this would coordinate with load balancer
                # For now, we simulate by activating the new model
                self.registry.update_model_status(model_version.model_id, DeploymentStatus.ACTIVE)
                
                # Deactivate old model
                old_model = self.registry.get_active_model(model_name)
                if old_model and old_model.model_id != model_version.model_id:
                    self.registry.update_model_status(old_model.model_id, DeploymentStatus.DEPRECATED)
                
                result['status'] = 'success'
                result['active_model_id'] = model_version.model_id
            
            result['model_id'] = model_version.model_id
            result['version'] = version
            
            logger.info(f"Model {model_name} deployed successfully with {deployment_strategy} strategy")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Deployment failed for {model_name}: {e}")
        
        return result
    
    def rollback_model(self, model_name: str, target_version: str = None) -> Dict[str, Any]:
        """Rollback model to previous version"""
        result = {
            'model_name': model_name,
            'rollback_initiated_at': datetime.now().isoformat(),
            'status': 'failed'
        }
        
        try:
            # Get current active model
            current_model = self.registry.get_active_model(model_name)
            if not current_model:
                result['error'] = 'No active model found'
                return result
            
            # Get model versions
            versions = self.registry.get_model_versions(model_name)
            
            if target_version:
                # Rollback to specific version
                target_model = None
                for version in versions:
                    if version.version == target_version:
                        target_model = version
                        break
                
                if not target_model:
                    result['error'] = f'Version {target_version} not found'
                    return result
                    
            else:
                # Rollback to previous version
                active_versions = [v for v in versions if v.status != DeploymentStatus.DEPRECATED]
                if len(active_versions) < 2:
                    result['error'] = 'No previous version available'
                    return result
                
                # Find the most recent version that's not the current one
                target_model = None
                for version in active_versions:
                    if version.model_id != current_model.model_id:
                        target_model = version
                        break
                
                if not target_model:
                    result['error'] = 'No suitable rollback target found'
                    return result
            
            # Perform rollback
            self.registry.update_model_status(target_model.model_id, DeploymentStatus.ACTIVE)
            self.registry.update_model_status(current_model.model_id, DeploymentStatus.ROLLBACK)
            
            result['status'] = 'success'
            result['previous_model_id'] = current_model.model_id
            result['previous_version'] = current_model.version
            result['rollback_model_id'] = target_model.model_id
            result['rollback_version'] = target_model.version
            result['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Rollback successful for {model_name}: "
                       f"{current_model.version} -> {target_model.version}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Rollback failed for {model_name}: {e}")
        
        return result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        # Get model statistics
        model_stats = {}
        with sqlite3.connect(self.registry.db_path) as conn:
            cursor = conn.execute('''
                SELECT model_name, status, COUNT(*) as count 
                FROM model_versions 
                GROUP BY model_name, status
            ''')
            
            for row in cursor.fetchall():
                model_name, status, count = row
                if model_name not in model_stats:
                    model_stats[model_name] = {}
                model_stats[model_name][status] = count
        
        # Get monitoring summary
        monitoring_summary = self.monitor.get_monitoring_summary()
        
        # Get active A/B tests
        active_tests = len([t for t in self.ab_testing.active_tests.values() 
                          if t['status'] == 'running'])
        
        return {
            'pipeline_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_statistics': model_stats,
            'monitoring': monitoring_summary,
            'active_ab_tests': active_tests,
            'auto_retraining_active': self.auto_retrain.retraining_active,
            'components': {
                'registry': 'active',
                'ab_testing': 'active',
                'monitoring': 'active',
                'auto_retrain': 'active'
            }
        }


# CLI Interface
def main():
    """CLI interface for ML deployment pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Deployment Pipeline')
    parser.add_argument('--action', choices=['deploy', 'rollback', 'status', 'monitor'], 
                       required=True, help='Action to perform')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--strategy', choices=['immediate', 'blue_green', 'canary', 'ab_test'], 
                       default='blue_green', help='Deployment strategy')
    parser.add_argument('--version', type=str, help='Target version for rollback')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLDeploymentPipeline()
    
    if args.action == 'deploy':
        if not args.model or not args.model_path:
            print("Error: --model and --model-path are required for deployment")
            return
        
        result = pipeline.deploy_model(args.model, args.model_path, args.strategy)
        
        print(f"\nDeployment Result:")
        print(f"Status: {result['status']}")
        if result['status'] != 'failed':
            print(f"Model ID: {result['model_id']}")
            print(f"Version: {result['version']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    elif args.action == 'rollback':
        if not args.model:
            print("Error: --model is required for rollback")
            return
        
        result = pipeline.rollback_model(args.model, args.version)
        
        print(f"\nRollback Result:")
        print(f"Status: {result['status']}")
        if result['status'] != 'failed':
            print(f"Rolled back from {result['previous_version']} to {result['rollback_version']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    elif args.action == 'status':
        status = pipeline.get_pipeline_status()
        
        print(f"\nML Deployment Pipeline Status:")
        print(f"Overall Status: {status['pipeline_status']}")
        print(f"Timestamp: {status['timestamp']}")
        print(f"\nModel Statistics:")
        for model_name, stats in status['model_statistics'].items():
            print(f"  {model_name}:")
            for status_type, count in stats.items():
                print(f"    {status_type}: {count}")
        
        print(f"\nMonitoring Summary:")
        print(f"  Monitored models: {status['monitoring']['monitored_models']}")
        print(f"  Active alerts: {status['monitoring']['active_alerts']}")
        print(f"  Total predictions: {status['monitoring']['total_predictions']}")
        
        print(f"\nActive A/B tests: {status['active_ab_tests']}")
        print(f"Auto-retraining active: {status['auto_retraining_active']}")
    
    elif args.action == 'monitor':
        if not args.model:
            print("Available models:")
            # List available models
            registry = ModelRegistry()
            with sqlite3.connect(registry.db_path) as conn:
                cursor = conn.execute('SELECT DISTINCT model_name FROM model_versions')
                for row in cursor.fetchall():
                    print(f"  - {row[0]}")
            return
        
        # Monitor specific model
        active_model = pipeline.registry.get_active_model(args.model)
        if not active_model:
            print(f"No active model found for {args.model}")
            return
        
        performance = pipeline.monitor.evaluate_model_performance(active_model.model_id)
        
        print(f"\nMonitoring Report for {args.model}:")
        print(f"Model ID: {active_model.model_id}")
        print(f"Version: {active_model.version}")
        print(f"Status: {active_model.status.value}")
        
        if 'metrics' in performance:
            print(f"\nCurrent Metrics:")
            for metric, value in performance['metrics'].items():
                print(f"  {metric}: {value:.3f}")
        
        if 'alerts' in performance and performance['alerts']:
            print(f"\nActive Alerts:")
            for alert in performance['alerts']:
                print(f"  - {alert['type']}: {alert.get('metric', '')} "
                     f"(severity: {alert['severity']})")


if __name__ == "__main__":
    main()