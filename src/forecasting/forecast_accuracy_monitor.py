"""
Forecast Accuracy Monitor for Beverly Knits ERP
Tracks predictions vs actuals, calculates metrics, and auto-adjusts ensemble weights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule
import time
import threading
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics"""
    yarn_id: str
    forecast_date: datetime
    horizon_weeks: int
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    accuracy: float  # 1 - MAPE
    model_used: str
    actual_values: List[float]
    predicted_values: List[float]


@dataclass
class PerformanceAlert:
    """Alert when accuracy drops below threshold"""
    alert_id: str
    timestamp: datetime
    yarn_id: str
    accuracy: float
    threshold: float
    message: str
    severity: str  # 'warning', 'critical'


class ForecastAccuracyMonitor:
    """
    Monitors forecast accuracy, tracks predictions vs actuals,
    and automatically adjusts ensemble weights based on performance
    """
    
    def __init__(self, 
                 db_path: str = "forecast_accuracy.db",
                 accuracy_threshold: float = 0.90,
                 alert_callback: Optional[callable] = None):
        """
        Initialize the accuracy monitor
        
        Args:
            db_path: Path to SQLite database for storing metrics
            accuracy_threshold: Minimum acceptable accuracy (90% default)
            alert_callback: Function to call when accuracy drops
        """
        self.db_path = db_path
        self.accuracy_threshold = accuracy_threshold
        self.alert_callback = alert_callback
        self.current_metrics = {}
        self.model_performance = {}
        self.monitoring_active = False
        
        # Initialize database
        self._init_database()
        
        # Start monitoring thread
        self.monitor_thread = None
        
        logger.info(f"Forecast Accuracy Monitor initialized with {accuracy_threshold*100}% threshold")
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                yarn_id TEXT NOT NULL,
                forecast_date TIMESTAMP NOT NULL,
                evaluation_date TIMESTAMP NOT NULL,
                horizon_weeks INTEGER NOT NULL,
                mape REAL,
                rmse REAL,
                mae REAL,
                accuracy REAL,
                model_used TEXT,
                actual_values TEXT,
                predicted_values TEXT,
                UNIQUE(yarn_id, forecast_date, horizon_weeks)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                timestamp TIMESTAMP NOT NULL,
                yarn_id TEXT,
                accuracy REAL,
                threshold REAL,
                message TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_date TIMESTAMP NOT NULL,
                prophet_weight REAL,
                xgboost_weight REAL,
                arima_weight REAL,
                average_accuracy REAL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_yarn_forecast ON forecast_metrics(yarn_id, forecast_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy ON forecast_metrics(accuracy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts ON performance_alerts(severity, resolved)")
        
        conn.commit()
        conn.close()
    
    def track_forecast(self,
                       yarn_id: str,
                       forecast_date: datetime,
                       predicted_values: List[float],
                       model_used: str,
                       horizon_weeks: int = 9):
        """
        Track a new forecast for later evaluation
        
        Args:
            yarn_id: Yarn identifier
            forecast_date: Date when forecast was made
            predicted_values: List of predicted values
            model_used: Model or ensemble used
            horizon_weeks: Forecast horizon
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO forecast_metrics 
                (yarn_id, forecast_date, evaluation_date, horizon_weeks, 
                 predicted_values, model_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                yarn_id,
                forecast_date.isoformat(),
                (forecast_date + timedelta(weeks=horizon_weeks)).isoformat(),
                horizon_weeks,
                json.dumps(predicted_values),
                model_used
            ))
            
            conn.commit()
            logger.info(f"Tracked forecast for {yarn_id} with {horizon_weeks}-week horizon")
            
        except Exception as e:
            logger.error(f"Error tracking forecast: {e}")
        finally:
            conn.close()
    
    def evaluate_forecast(self,
                         yarn_id: str,
                         actual_values: List[float],
                         evaluation_date: Optional[datetime] = None) -> Optional[AccuracyMetrics]:
        """
        Evaluate forecast accuracy against actual values
        
        Args:
            yarn_id: Yarn identifier
            actual_values: Actual observed values
            evaluation_date: Date of evaluation (default: now)
        
        Returns:
            AccuracyMetrics or None if no matching forecast
        """
        if evaluation_date is None:
            evaluation_date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Find matching forecast
            cursor.execute("""
                SELECT forecast_date, horizon_weeks, predicted_values, model_used
                FROM forecast_metrics
                WHERE yarn_id = ? 
                AND evaluation_date <= ?
                AND actual_values IS NULL
                ORDER BY forecast_date DESC
                LIMIT 1
            """, (yarn_id, evaluation_date.isoformat()))
            
            result = cursor.fetchone()
            
            if result:
                forecast_date = datetime.fromisoformat(result[0])
                horizon_weeks = result[1]
                predicted_values = json.loads(result[2])
                model_used = result[3]
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    actual_values[:len(predicted_values)],
                    predicted_values[:len(actual_values)]
                )
                
                # Create AccuracyMetrics object
                accuracy_metrics = AccuracyMetrics(
                    yarn_id=yarn_id,
                    forecast_date=forecast_date,
                    horizon_weeks=horizon_weeks,
                    mape=metrics['mape'],
                    rmse=metrics['rmse'],
                    mae=metrics['mae'],
                    accuracy=metrics['accuracy'],
                    model_used=model_used,
                    actual_values=actual_values[:len(predicted_values)],
                    predicted_values=predicted_values[:len(actual_values)]
                )
                
                # Update database with actual values and metrics
                cursor.execute("""
                    UPDATE forecast_metrics
                    SET actual_values = ?, mape = ?, rmse = ?, mae = ?, accuracy = ?
                    WHERE yarn_id = ? AND forecast_date = ?
                """, (
                    json.dumps(actual_values),
                    metrics['mape'],
                    metrics['rmse'],
                    metrics['mae'],
                    metrics['accuracy'],
                    yarn_id,
                    forecast_date.isoformat()
                ))
                
                conn.commit()
                
                # Check for alerts
                self._check_accuracy_alert(accuracy_metrics)
                
                # Update current metrics
                self.current_metrics[yarn_id] = accuracy_metrics
                
                logger.info(f"Evaluated forecast for {yarn_id}: Accuracy = {metrics['accuracy']:.2%}")
                
                return accuracy_metrics
            
            else:
                logger.warning(f"No matching forecast found for {yarn_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating forecast: {e}")
            return None
        finally:
            conn.close()
    
    def _calculate_metrics(self, actual: List[float], predicted: List[float]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        metrics = {}
        
        # MAE
        metrics['mae'] = mean_absolute_error(actual, predicted)
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE (handle zero values)
        non_zero_mask = actual != 0
        if non_zero_mask.any():
            metrics['mape'] = mean_absolute_percentage_error(
                actual[non_zero_mask],
                predicted[non_zero_mask]
            )
        else:
            metrics['mape'] = 0.0
        
        # Accuracy (1 - MAPE)
        metrics['accuracy'] = 1 - metrics['mape'] if metrics['mape'] is not None else 1.0
        
        return metrics
    
    def _check_accuracy_alert(self, metrics: AccuracyMetrics):
        """Check if accuracy drops below threshold and create alert"""
        if metrics.accuracy < self.accuracy_threshold:
            severity = 'critical' if metrics.accuracy < (self.accuracy_threshold - 0.1) else 'warning'
            
            alert = PerformanceAlert(
                alert_id=f"{metrics.yarn_id}_{datetime.now().isoformat()}",
                timestamp=datetime.now(),
                yarn_id=metrics.yarn_id,
                accuracy=metrics.accuracy,
                threshold=self.accuracy_threshold,
                message=f"Forecast accuracy ({metrics.accuracy:.2%}) below threshold ({self.accuracy_threshold:.2%}) for {metrics.yarn_id}",
                severity=severity
            )
            
            # Store alert
            self._store_alert(alert)
            
            # Call callback if provided
            if self.alert_callback:
                self.alert_callback(alert)
            
            logger.warning(f"ALERT: {alert.message}")
    
    def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO performance_alerts
                (alert_id, timestamp, yarn_id, accuracy, threshold, message, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.yarn_id,
                alert.accuracy,
                alert.threshold,
                alert.message,
                alert.severity
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
        finally:
            conn.close()
    
    def get_model_performance(self, 
                             period_days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Get model performance statistics for the specified period
        
        Args:
            period_days: Number of days to analyze
        
        Returns:
            Dictionary of model -> performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        
        cursor.execute("""
            SELECT model_used, 
                   AVG(accuracy) as avg_accuracy,
                   AVG(mape) as avg_mape,
                   AVG(rmse) as avg_rmse,
                   AVG(mae) as avg_mae,
                   COUNT(*) as sample_count
            FROM forecast_metrics
            WHERE evaluation_date >= ?
            AND accuracy IS NOT NULL
            GROUP BY model_used
        """, (start_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        performance = {}
        for row in results:
            model = row[0]
            performance[model] = {
                'avg_accuracy': row[1],
                'avg_mape': row[2],
                'avg_rmse': row[3],
                'avg_mae': row[4],
                'sample_count': row[5]
            }
        
        return performance
    
    def optimize_ensemble_weights(self) -> Dict[str, float]:
        """
        Optimize ensemble weights based on recent performance
        
        Returns:
            Updated ensemble weights
        """
        # Get recent performance
        performance = self.get_model_performance(period_days=30)
        
        if not performance:
            logger.warning("No performance data available for weight optimization")
            return {}
        
        # Calculate weights based on accuracy
        total_accuracy = sum(p.get('avg_accuracy', 0) for p in performance.values())
        
        weights = {}
        if total_accuracy > 0:
            for model, metrics in performance.items():
                weights[model] = metrics.get('avg_accuracy', 0) / total_accuracy
        
        # Store updated weights
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_weights
            (update_date, prophet_weight, xgboost_weight, arima_weight, average_accuracy)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            weights.get('prophet', 0.33),
            weights.get('xgboost', 0.33),
            weights.get('arima', 0.34),
            total_accuracy / len(performance) if performance else 0
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Optimized ensemble weights: {weights}")
        
        return weights
    
    def get_accuracy_report(self, 
                           yarn_id: Optional[str] = None,
                           period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive accuracy report
        
        Args:
            yarn_id: Specific yarn or None for all
            period_days: Period to analyze
        
        Returns:
            Comprehensive accuracy report
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        
        # Build query
        if yarn_id:
            query = """
                SELECT yarn_id, AVG(accuracy), MIN(accuracy), MAX(accuracy), 
                       COUNT(*), AVG(mape), AVG(rmse), AVG(mae)
                FROM forecast_metrics
                WHERE evaluation_date >= ? AND accuracy IS NOT NULL AND yarn_id = ?
                GROUP BY yarn_id
            """
            params = (start_date, yarn_id)
        else:
            query = """
                SELECT yarn_id, AVG(accuracy), MIN(accuracy), MAX(accuracy), 
                       COUNT(*), AVG(mape), AVG(rmse), AVG(mae)
                FROM forecast_metrics
                WHERE evaluation_date >= ? AND accuracy IS NOT NULL
                GROUP BY yarn_id
            """
            params = (start_date,)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get alerts
        cursor.execute("""
            SELECT COUNT(*), severity
            FROM performance_alerts
            WHERE timestamp >= ? AND resolved = FALSE
            GROUP BY severity
        """, (start_date,))
        
        alerts = cursor.fetchall()
        
        conn.close()
        
        # Build report
        report = {
            'period_days': period_days,
            'report_date': datetime.now().isoformat(),
            'accuracy_threshold': self.accuracy_threshold,
            'yarn_metrics': [],
            'overall_metrics': {},
            'alerts': {'warning': 0, 'critical': 0},
            'yarns_meeting_target': 0,
            'yarns_below_target': 0
        }
        
        all_accuracies = []
        
        for row in results:
            yarn_metrics = {
                'yarn_id': row[0],
                'avg_accuracy': row[1],
                'min_accuracy': row[2],
                'max_accuracy': row[3],
                'sample_count': row[4],
                'avg_mape': row[5],
                'avg_rmse': row[6],
                'avg_mae': row[7]
            }
            
            report['yarn_metrics'].append(yarn_metrics)
            all_accuracies.append(row[1])
            
            if row[1] >= self.accuracy_threshold:
                report['yarns_meeting_target'] += 1
            else:
                report['yarns_below_target'] += 1
        
        # Calculate overall metrics
        if all_accuracies:
            report['overall_metrics'] = {
                'average_accuracy': np.mean(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies),
                'std_accuracy': np.std(all_accuracies),
                'target_achievement_rate': report['yarns_meeting_target'] / len(all_accuracies)
            }
        
        # Add alerts
        for count, severity in alerts:
            report['alerts'][severity] = count
        
        return report
    
    def start_continuous_monitoring(self, check_interval_hours: int = 1):
        """
        Start continuous monitoring in background thread
        
        Args:
            check_interval_hours: Hours between checks
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Check for forecasts that need evaluation
                    self._check_pending_evaluations()
                    
                    # Optimize weights weekly
                    if datetime.now().weekday() == 0 and datetime.now().hour == 0:
                        self.optimize_ensemble_weights()
                    
                    # Sleep
                    time.sleep(check_interval_hours * 3600)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started continuous monitoring with {check_interval_hours}-hour interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped continuous monitoring")
    
    def _check_pending_evaluations(self):
        """Check for forecasts that are ready for evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find forecasts past their evaluation date without actuals
        cursor.execute("""
            SELECT yarn_id, forecast_date, horizon_weeks
            FROM forecast_metrics
            WHERE evaluation_date <= ?
            AND actual_values IS NULL
        """, (datetime.now().isoformat(),))
        
        pending = cursor.fetchall()
        conn.close()
        
        if pending:
            logger.info(f"Found {len(pending)} forecasts ready for evaluation")
            # In production, this would trigger data fetch and evaluation
    
    def get_alert_summary(self, unresolved_only: bool = True) -> List[Dict]:
        """Get summary of performance alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if unresolved_only:
            cursor.execute("""
                SELECT alert_id, timestamp, yarn_id, accuracy, threshold, message, severity
                FROM performance_alerts
                WHERE resolved = FALSE
                ORDER BY timestamp DESC
            """)
        else:
            cursor.execute("""
                SELECT alert_id, timestamp, yarn_id, accuracy, threshold, message, severity
                FROM performance_alerts
                ORDER BY timestamp DESC
                LIMIT 100
            """)
        
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            alerts.append({
                'alert_id': row[0],
                'timestamp': row[1],
                'yarn_id': row[2],
                'accuracy': row[3],
                'threshold': row[4],
                'message': row[5],
                'severity': row[6]
            })
        
        return alerts


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = ForecastAccuracyMonitor(
        accuracy_threshold=0.90,
        alert_callback=lambda alert: print(f"ALERT: {alert.message}")
    )
    
    # Track a forecast
    monitor.track_forecast(
        yarn_id="YARN001",
        forecast_date=datetime.now(),
        predicted_values=[100, 110, 105, 115, 120, 125, 130, 135, 140],
        model_used="Ensemble",
        horizon_weeks=9
    )
    
    # Simulate actual values coming in
    actual_values = [98, 112, 103, 118, 119, 128, 132, 134, 142]
    
    # Evaluate forecast
    metrics = monitor.evaluate_forecast("YARN001", actual_values)
    
    if metrics:
        print(f"Accuracy: {metrics.accuracy:.2%}")
        print(f"MAPE: {metrics.mape:.4f}")
        print(f"RMSE: {metrics.rmse:.2f}")
        print(f"MAE: {metrics.mae:.2f}")
    
    # Get performance report
    report = monitor.get_accuracy_report()
    print(f"\nAccuracy Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Optimize weights
    new_weights = monitor.optimize_ensemble_weights()
    print(f"\nOptimized Weights: {new_weights}")
    
    # Start monitoring
    monitor.start_continuous_monitoring(check_interval_hours=1)
    print("Monitoring started...")
    
    # Let it run for a bit
    time.sleep(5)
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("Monitoring stopped.")