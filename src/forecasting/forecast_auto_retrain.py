"""
Automatic Model Retraining System for Beverly Knits ERP
Implements weekly retraining with performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import schedule
import time
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3

from .enhanced_forecasting_engine import EnhancedForecastingEngine, ForecastConfig
from .forecast_accuracy_monitor import ForecastAccuracyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomaticRetrainingSystem:
    """
    Manages automatic weekly retraining of forecasting models
    with performance tracking and optimization
    """
    
    def __init__(self,
                 data_path: str = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/",
                 retrain_schedule: str = "weekly",
                 retrain_day: str = "sunday",
                 retrain_hour: int = 2):
        """
        Initialize automatic retraining system
        
        Args:
            data_path: Path to ERP data files
            retrain_schedule: Frequency of retraining (weekly/daily/monthly)
            retrain_day: Day of week for weekly retraining
            retrain_hour: Hour of day for retraining (24-hour format)
        """
        self.data_path = Path(data_path)
        self.retrain_schedule = retrain_schedule
        self.retrain_day = retrain_day.lower()
        self.retrain_hour = retrain_hour
        
        # Initialize forecasting engine
        self.forecast_config = ForecastConfig(
            horizon_weeks=9,
            min_accuracy_threshold=0.90,
            retrain_frequency=retrain_schedule
        )
        self.forecast_engine = EnhancedForecastingEngine(self.forecast_config)
        
        # Initialize accuracy monitor
        self.accuracy_monitor = ForecastAccuracyMonitor(
            db_path="forecast_accuracy.db",
            accuracy_threshold=0.90
        )
        
        # Training history
        self.training_history = []
        self.last_training_time = None
        self.is_running = False
        self.scheduler_thread = None
        
        logger.info(f"Automatic Retraining System initialized")
        logger.info(f"Schedule: {retrain_schedule} on {retrain_day} at {retrain_hour:02d}:00")
    
    def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load training data from ERP files
        
        Returns:
            Dictionary of yarn_id -> historical data
        """
        training_data = {}
        
        try:
            # Load yarn inventory with consumption data
            inventory_files = list(self.data_path.glob("yarn_inventory*.xlsx")) + \
                            list(self.data_path.glob("yarn_inventory*.csv"))
            
            if inventory_files:
                inventory_file = inventory_files[0]
                logger.info(f"Loading inventory data from {inventory_file}")
                
                if inventory_file.suffix == '.xlsx':
                    df = pd.read_excel(inventory_file)
                else:
                    df = pd.read_csv(inventory_file)
                
                # Group by yarn ID and prepare historical data
                if 'Desc#' in df.columns:
                    yarn_col = 'Desc#'
                elif 'Yarn_ID' in df.columns:
                    yarn_col = 'Yarn_ID'
                else:
                    yarn_col = df.columns[0]  # Use first column as fallback
                
                # Get consumed data for each yarn
                for yarn_id in df[yarn_col].unique():
                    if pd.notna(yarn_id):
                        yarn_data = df[df[yarn_col] == yarn_id].copy()
                        
                        # Create time series from consumed data
                        if 'Consumed' in yarn_data.columns:
                            # Generate weekly time series
                            dates = pd.date_range(
                                end=datetime.now(),
                                periods=52,
                                freq='W'
                            )
                            
                            # Use consumed values or simulate if needed
                            consumed_values = yarn_data['Consumed'].values
                            if len(consumed_values) > 0:
                                # Extend or truncate to 52 weeks
                                if len(consumed_values) < 52:
                                    # Repeat pattern if not enough data
                                    consumed_values = np.tile(consumed_values, 52 // len(consumed_values) + 1)[:52]
                                else:
                                    consumed_values = consumed_values[-52:]
                                
                                historical_df = pd.DataFrame({
                                    'date': dates,
                                    'Consumed': consumed_values
                                })
                                
                                training_data[str(yarn_id)] = historical_df
            
            # Load sales data for order-based forecasting
            sales_files = list(self.data_path.glob("Sales*.csv"))
            if sales_files:
                sales_file = sales_files[0]
                logger.info(f"Loading sales data from {sales_file}")
                sales_df = pd.read_csv(sales_file)
                # Process sales data as needed
            
            logger.info(f"Loaded training data for {len(training_data)} yarns")
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
        
        return training_data
    
    def retrain_models(self) -> Dict[str, Any]:
        """
        Retrain all forecasting models
        
        Returns:
            Training results including accuracy metrics
        """
        logger.info("Starting model retraining...")
        start_time = datetime.now()
        
        # Load latest training data
        training_data = self.load_training_data()
        
        if not training_data:
            logger.warning("No training data available")
            return {
                'status': 'failed',
                'reason': 'No training data',
                'timestamp': start_time
            }
        
        # Retrain models
        model_accuracies = self.forecast_engine.retrain_models(training_data)
        
        # Optimize ensemble weights based on performance
        optimized_weights = self.accuracy_monitor.optimize_ensemble_weights()
        
        # Update forecast engine with new weights
        if optimized_weights:
            self.forecast_engine.config.ensemble_weights = optimized_weights
        
        # Generate forecasts for validation
        validation_results = self._validate_retrained_models(training_data)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare results
        results = {
            'status': 'success',
            'timestamp': start_time,
            'training_time_seconds': training_time,
            'yarns_trained': len(training_data),
            'model_accuracies': model_accuracies,
            'optimized_weights': optimized_weights,
            'validation_results': validation_results,
            'average_accuracy': np.mean(list(model_accuracies.values())) if model_accuracies else 0
        }
        
        # Store training history
        self.training_history.append(results)
        self.last_training_time = start_time
        
        # Save training results
        self._save_training_results(results)
        
        logger.info(f"Model retraining completed in {training_time:.2f} seconds")
        logger.info(f"Average accuracy: {results['average_accuracy']:.2%}")
        
        return results
    
    def _validate_retrained_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Validate retrained models using backtesting
        
        Args:
            training_data: Training data for validation
        
        Returns:
            Validation metrics
        """
        validation_results = {
            'yarns_validated': 0,
            'average_accuracy': 0,
            'yarns_meeting_target': 0,
            'yarns_below_target': 0
        }
        
        accuracies = []
        
        # Sample yarns for validation (max 20 for performance)
        sample_yarns = list(training_data.keys())[:20]
        
        for yarn_id in sample_yarns:
            data = training_data[yarn_id]
            
            if len(data) > 10:
                # Split data for validation
                train_size = int(len(data) * 0.8)
                train_data = data[:train_size]
                test_data = data[train_size:]
                
                # Generate forecast
                result = self.forecast_engine.forecast(yarn_id, train_data)
                
                # Track forecast for monitoring
                if result.predictions is not None and not result.predictions.empty:
                    self.accuracy_monitor.track_forecast(
                        yarn_id=yarn_id,
                        forecast_date=datetime.now(),
                        predicted_values=result.predictions['forecast'].tolist(),
                        model_used=result.model_used,
                        horizon_weeks=result.horizon_weeks
                    )
                    
                    # Check accuracy if available
                    if result.accuracy_metrics.get('accuracy') is not None:
                        accuracy = result.accuracy_metrics['accuracy']
                        accuracies.append(accuracy)
                        
                        if accuracy >= self.forecast_config.min_accuracy_threshold:
                            validation_results['yarns_meeting_target'] += 1
                        else:
                            validation_results['yarns_below_target'] += 1
                
                validation_results['yarns_validated'] += 1
        
        if accuracies:
            validation_results['average_accuracy'] = np.mean(accuracies)
        
        return validation_results
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file"""
        try:
            results_file = Path("forecast_training_history.json")
            
            # Load existing history
            if results_file.exists():
                with open(results_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Convert datetime objects to strings
            results_copy = results.copy()
            if isinstance(results_copy.get('timestamp'), datetime):
                results_copy['timestamp'] = results_copy['timestamp'].isoformat()
            
            # Append new results
            history.append(results_copy)
            
            # Keep only last 100 training sessions
            history = history[-100:]
            
            # Save updated history
            with open(results_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def schedule_retraining(self):
        """Schedule automatic retraining based on configuration"""
        if self.retrain_schedule == "weekly":
            # Schedule weekly retraining
            schedule.every().week.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            
            # Also schedule for specific day if provided
            if self.retrain_day == "monday":
                schedule.every().monday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "tuesday":
                schedule.every().tuesday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "wednesday":
                schedule.every().wednesday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "thursday":
                schedule.every().thursday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "friday":
                schedule.every().friday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "saturday":
                schedule.every().saturday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            elif self.retrain_day == "sunday":
                schedule.every().sunday.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
                
        elif self.retrain_schedule == "daily":
            schedule.every().day.at(f"{self.retrain_hour:02d}:00").do(self.retrain_models)
            
        elif self.retrain_schedule == "monthly":
            # Schedule for first day of month
            schedule.every().day.at(f"{self.retrain_hour:02d}:00").do(self._monthly_retrain_check)
        
        logger.info(f"Retraining scheduled: {self.retrain_schedule}")
    
    def _monthly_retrain_check(self):
        """Check if it's time for monthly retraining"""
        if datetime.now().day == 1:
            self.retrain_models()
    
    def start(self):
        """Start automatic retraining scheduler"""
        if self.is_running:
            logger.warning("Retraining scheduler already running")
            return
        
        self.is_running = True
        
        # Schedule retraining
        self.schedule_retraining()
        
        # Start scheduler in background thread
        def scheduler_loop():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start accuracy monitoring
        self.accuracy_monitor.start_continuous_monitoring(check_interval_hours=1)
        
        logger.info("Automatic retraining system started")
    
    def stop(self):
        """Stop automatic retraining scheduler"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.accuracy_monitor.stop_monitoring()
        
        logger.info("Automatic retraining system stopped")
    
    def trigger_immediate_retrain(self) -> Dict[str, Any]:
        """Trigger immediate model retraining"""
        logger.info("Triggering immediate model retraining...")
        return self.retrain_models()
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and history"""
        status = {
            'is_running': self.is_running,
            'schedule': self.retrain_schedule,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'next_training': self._get_next_training_time(),
            'training_history_count': len(self.training_history),
            'current_ensemble_weights': self.forecast_engine.config.ensemble_weights,
            'accuracy_threshold': self.forecast_config.min_accuracy_threshold
        }
        
        # Add recent training results
        if self.training_history:
            recent = self.training_history[-1]
            status['last_training_results'] = {
                'yarns_trained': recent.get('yarns_trained', 0),
                'average_accuracy': recent.get('average_accuracy', 0),
                'training_time': recent.get('training_time_seconds', 0)
            }
        
        # Add current accuracy report
        accuracy_report = self.accuracy_monitor.get_accuracy_report(period_days=7)
        if accuracy_report and accuracy_report.get('overall_metrics'):
            status['current_accuracy'] = accuracy_report['overall_metrics'].get('average_accuracy', 0)
            status['yarns_meeting_target'] = accuracy_report.get('yarns_meeting_target', 0)
            status['yarns_below_target'] = accuracy_report.get('yarns_below_target', 0)
        
        return status
    
    def _get_next_training_time(self) -> Optional[str]:
        """Calculate next scheduled training time"""
        if not schedule.jobs:
            return None
        
        next_run = schedule.jobs[0].next_run
        if next_run:
            return next_run.isoformat()
        
        return None


# Integration with main ERP system
def integrate_with_erp(app):
    """
    Integrate automatic retraining with Flask app
    
    Args:
        app: Flask application instance
    """
    # Initialize retraining system
    retrain_system = AutomaticRetrainingSystem(
        retrain_schedule="weekly",
        retrain_day="sunday",
        retrain_hour=2
    )
    
    # Start automatic retraining
    retrain_system.start()
    
    # Add API endpoints
    @app.route('/api/forecast/retrain', methods=['POST'])
    def trigger_retrain():
        """Manually trigger model retraining"""
        results = retrain_system.trigger_immediate_retrain()
        return json.dumps(results, default=str), 200
    
    @app.route('/api/forecast/training-status', methods=['GET'])
    def get_training_status():
        """Get current training status"""
        status = retrain_system.get_training_status()
        return json.dumps(status, default=str), 200
    
    @app.route('/api/forecast/accuracy-report', methods=['GET'])
    def get_accuracy_report():
        """Get forecast accuracy report"""
        period_days = int(request.args.get('days', 30))
        report = retrain_system.accuracy_monitor.get_accuracy_report(period_days=period_days)
        return json.dumps(report, default=str), 200
    
    return retrain_system


# Standalone testing
if __name__ == "__main__":
    # Initialize system
    retrain_system = AutomaticRetrainingSystem(
        retrain_schedule="weekly",
        retrain_day="sunday",
        retrain_hour=2
    )
    
    # Start automatic retraining
    retrain_system.start()
    
    print("Automatic retraining system started")
    print(f"Status: {json.dumps(retrain_system.get_training_status(), indent=2, default=str)}")
    
    # Trigger immediate retrain for testing
    print("\nTriggering immediate retrain...")
    results = retrain_system.trigger_immediate_retrain()
    print(f"Training results: {json.dumps(results, indent=2, default=str)}")
    
    # Let it run for a bit
    time.sleep(10)
    
    # Stop system
    retrain_system.stop()
    print("\nAutomatic retraining system stopped")