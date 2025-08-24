"""
Forecast Validation and Backtesting System for Beverly Knits ERP
Comprehensive validation to ensure 90% accuracy at 9-week horizon
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from .enhanced_forecasting_engine import EnhancedForecastingEngine, ForecastConfig, ForecastResult
from .forecast_accuracy_monitor import ForecastAccuracyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results"""
    yarn_id: str
    test_periods: int
    horizon_weeks: int
    model_accuracies: Dict[str, List[float]]
    ensemble_accuracy: float
    best_model: str
    worst_model: str
    confidence_intervals_coverage: float
    forecast_bias: float  # Positive = over-forecasting, Negative = under-forecasting


@dataclass 
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: datetime
    total_yarns_tested: int
    average_accuracy: float
    accuracy_by_week: Dict[int, float]  # Week 1-9 accuracy
    yarns_meeting_target: int
    yarns_below_target: int
    critical_yarns: List[str]  # Yarns with <80% accuracy
    model_rankings: Dict[str, float]
    recommended_weights: Dict[str, float]
    validation_passed: bool


class ForecastValidationSystem:
    """
    Comprehensive validation and backtesting system for forecast models
    Ensures 90% accuracy at 9-week horizon through rigorous testing
    """
    
    def __init__(self,
                 accuracy_target: float = 0.90,
                 confidence_level: float = 0.95):
        """
        Initialize validation system
        
        Args:
            accuracy_target: Target accuracy (90% default)
            confidence_level: Confidence level for intervals
        """
        self.accuracy_target = accuracy_target
        self.confidence_level = confidence_level
        
        # Initialize components
        self.forecast_engine = EnhancedForecastingEngine()
        self.accuracy_monitor = ForecastAccuracyMonitor(accuracy_threshold=accuracy_target)
        
        # Validation results storage
        self.validation_history = []
        self.backtest_results = {}
        
        logger.info(f"Validation System initialized with {accuracy_target*100}% target")
    
    def backtest_yarn(self,
                      yarn_id: str,
                      historical_data: pd.DataFrame,
                      test_periods: int = 5,
                      horizon_weeks: int = 9) -> BacktestResult:
        """
        Perform backtesting for a specific yarn
        
        Args:
            yarn_id: Yarn identifier
            historical_data: Historical consumption data
            test_periods: Number of test periods for validation
            horizon_weeks: Forecast horizon to test
        
        Returns:
            BacktestResult with detailed metrics
        """
        logger.info(f"Backtesting {yarn_id} with {test_periods} test periods")
        
        # Prepare time series split
        tscv = TimeSeriesSplit(n_splits=test_periods)
        
        model_accuracies = {
            'prophet': [],
            'xgboost': [],
            'arima': [],
            'ensemble': []
        }
        
        actual_vs_predicted = []
        confidence_coverage = []
        forecast_errors = []
        
        # Perform walk-forward validation
        for train_idx, test_idx in tscv.split(historical_data):
            # Split data
            train_data = historical_data.iloc[train_idx]
            test_data = historical_data.iloc[test_idx]
            
            # Ensure we have enough test data
            if len(test_data) < horizon_weeks:
                continue
            
            # Generate forecasts
            result = self.forecast_engine.forecast(yarn_id, train_data)
            
            if result.predictions is not None and not result.predictions.empty:
                # Get actual values for comparison
                test_values = test_data.iloc[:horizon_weeks]
                
                # Extract actual demand values
                if 'Consumed' in test_values.columns:
                    actuals = np.abs(test_values['Consumed'].values) / 4.3  # Weekly
                elif 'demand' in test_values.columns:
                    actuals = test_values['demand'].values
                else:
                    actuals = test_values.iloc[:, 0].values
                
                # Get predictions
                predictions = result.predictions['forecast'].values[:len(actuals)]
                
                # Calculate accuracy for this fold
                if len(actuals) > 0 and len(predictions) > 0:
                    # Avoid division by zero
                    non_zero_mask = actuals != 0
                    if non_zero_mask.any():
                        accuracy = 1 - mean_absolute_percentage_error(
                            actuals[non_zero_mask],
                            predictions[non_zero_mask]
                        )
                        model_accuracies['ensemble'].append(accuracy)
                        
                        # Track actual vs predicted
                        actual_vs_predicted.append({
                            'actual': actuals.tolist(),
                            'predicted': predictions.tolist()
                        })
                        
                        # Calculate forecast bias
                        errors = predictions - actuals
                        forecast_errors.extend(errors)
                        
                        # Check confidence interval coverage
                        if result.confidence_intervals is not None:
                            lower = result.confidence_intervals['lower_bound'].values[:len(actuals)]
                            upper = result.confidence_intervals['upper_bound'].values[:len(actuals)]
                            coverage = np.mean((actuals >= lower) & (actuals <= upper))
                            confidence_coverage.append(coverage)
        
        # Calculate overall metrics
        ensemble_accuracy = np.mean(model_accuracies['ensemble']) if model_accuracies['ensemble'] else 0
        
        # Determine best and worst models
        avg_accuracies = {
            model: np.mean(accs) if accs else 0
            for model, accs in model_accuracies.items()
        }
        
        if avg_accuracies:
            best_model = max(avg_accuracies, key=avg_accuracies.get)
            worst_model = min(avg_accuracies, key=avg_accuracies.get)
        else:
            best_model = worst_model = 'unknown'
        
        # Calculate confidence interval coverage
        ci_coverage = np.mean(confidence_coverage) if confidence_coverage else 0
        
        # Calculate forecast bias
        forecast_bias = np.mean(forecast_errors) if forecast_errors else 0
        
        # Create backtest result
        result = BacktestResult(
            yarn_id=yarn_id,
            test_periods=test_periods,
            horizon_weeks=horizon_weeks,
            model_accuracies=model_accuracies,
            ensemble_accuracy=ensemble_accuracy,
            best_model=best_model,
            worst_model=worst_model,
            confidence_intervals_coverage=ci_coverage,
            forecast_bias=forecast_bias
        )
        
        # Store result
        self.backtest_results[yarn_id] = result
        
        logger.info(f"Backtest complete for {yarn_id}: Accuracy = {ensemble_accuracy:.2%}")
        
        return result
    
    def validate_forecast_system(self,
                                test_data: Dict[str, pd.DataFrame],
                                sample_size: Optional[int] = None) -> ValidationReport:
        """
        Perform comprehensive validation of the forecasting system
        
        Args:
            test_data: Dictionary of yarn_id -> historical data
            sample_size: Number of yarns to test (None = all)
        
        Returns:
            ValidationReport with detailed results
        """
        logger.info("Starting comprehensive forecast validation...")
        start_time = datetime.now()
        
        # Sample yarns if needed
        yarn_ids = list(test_data.keys())
        if sample_size and sample_size < len(yarn_ids):
            yarn_ids = np.random.choice(yarn_ids, sample_size, replace=False)
        
        # Track metrics
        all_accuracies = []
        accuracy_by_week = {i: [] for i in range(1, 10)}
        model_performance = {'prophet': [], 'xgboost': [], 'arima': [], 'ensemble': []}
        critical_yarns = []
        
        # Validate each yarn
        for yarn_id in yarn_ids:
            data = test_data[yarn_id]
            
            if len(data) < 20:  # Need minimum data for validation
                continue
            
            # Perform backtesting
            backtest_result = self.backtest_yarn(
                yarn_id=yarn_id,
                historical_data=data,
                test_periods=min(5, len(data) // 20),
                horizon_weeks=9
            )
            
            # Track overall accuracy
            if backtest_result.ensemble_accuracy > 0:
                all_accuracies.append(backtest_result.ensemble_accuracy)
                
                # Check if meets target
                if backtest_result.ensemble_accuracy < self.accuracy_target:
                    if backtest_result.ensemble_accuracy < 0.80:
                        critical_yarns.append(yarn_id)
                
                # Track model-specific performance
                for model, accuracies in backtest_result.model_accuracies.items():
                    if accuracies:
                        model_performance[model].append(np.mean(accuracies))
            
            # Analyze accuracy by forecast week
            self._analyze_weekly_accuracy(yarn_id, data, accuracy_by_week)
        
        # Calculate summary metrics
        if all_accuracies:
            average_accuracy = np.mean(all_accuracies)
            yarns_meeting_target = sum(1 for acc in all_accuracies if acc >= self.accuracy_target)
            yarns_below_target = len(all_accuracies) - yarns_meeting_target
        else:
            average_accuracy = 0
            yarns_meeting_target = 0
            yarns_below_target = 0
        
        # Calculate model rankings
        model_rankings = {}
        for model, accs in model_performance.items():
            if accs:
                model_rankings[model] = np.mean(accs)
        
        # Calculate recommended ensemble weights
        recommended_weights = self._calculate_optimal_weights(model_rankings)
        
        # Calculate weekly accuracy averages
        weekly_accuracy_avg = {}
        for week, accs in accuracy_by_week.items():
            if accs:
                weekly_accuracy_avg[week] = np.mean(accs)
            else:
                weekly_accuracy_avg[week] = 0
        
        # Determine if validation passed
        validation_passed = average_accuracy >= self.accuracy_target
        
        # Create validation report
        report = ValidationReport(
            timestamp=start_time,
            total_yarns_tested=len(all_accuracies),
            average_accuracy=average_accuracy,
            accuracy_by_week=weekly_accuracy_avg,
            yarns_meeting_target=yarns_meeting_target,
            yarns_below_target=yarns_below_target,
            critical_yarns=critical_yarns[:10],  # Top 10 critical
            model_rankings=model_rankings,
            recommended_weights=recommended_weights,
            validation_passed=validation_passed
        )
        
        # Store in history
        self.validation_history.append(report)
        
        # Log results
        logger.info(f"Validation complete: {average_accuracy:.2%} average accuracy")
        logger.info(f"Target achievement: {yarns_meeting_target}/{len(all_accuracies)} yarns")
        logger.info(f"Validation {'PASSED' if validation_passed else 'FAILED'}")
        
        return report
    
    def _analyze_weekly_accuracy(self,
                                yarn_id: str,
                                data: pd.DataFrame,
                                accuracy_by_week: Dict[int, List[float]]):
        """Analyze accuracy for each week of the forecast horizon"""
        if len(data) < 20:
            return
        
        # Split data for testing
        train_size = int(len(data) * 0.7)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        if len(test_data) < 9:
            return
        
        # Generate forecast
        result = self.forecast_engine.forecast(yarn_id, train_data)
        
        if result.predictions is None or result.predictions.empty:
            return
        
        # Compare week by week
        for week in range(1, min(10, len(test_data) + 1)):
            if week <= len(result.predictions):
                # Get actual and predicted for this week
                if 'Consumed' in test_data.columns:
                    actual = np.abs(test_data.iloc[week-1]['Consumed']) / 4.3
                else:
                    actual = test_data.iloc[week-1, 0]
                
                predicted = result.predictions.iloc[week-1]['forecast']
                
                # Calculate accuracy for this week
                if actual != 0:
                    week_accuracy = 1 - abs(predicted - actual) / actual
                    accuracy_by_week[week].append(max(0, week_accuracy))
    
    def _calculate_optimal_weights(self, model_rankings: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on model performance"""
        if not model_rankings:
            return {'prophet': 0.4, 'xgboost': 0.35, 'arima': 0.25}
        
        # Normalize rankings to sum to 1
        total = sum(model_rankings.values())
        if total > 0:
            return {model: rank/total for model, rank in model_rankings.items()}
        
        return model_rankings
    
    def generate_validation_plots(self, output_dir: str = "validation_plots"):
        """Generate validation plots and charts"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.validation_history:
            logger.warning("No validation history to plot")
            return
        
        latest_report = self.validation_history[-1]
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Accuracy by Week Plot
        if latest_report.accuracy_by_week:
            plt.figure()
            weeks = list(latest_report.accuracy_by_week.keys())
            accuracies = list(latest_report.accuracy_by_week.values())
            
            plt.bar(weeks, accuracies, color=['green' if acc >= self.accuracy_target else 'red' for acc in accuracies])
            plt.axhline(y=self.accuracy_target, color='blue', linestyle='--', label=f'Target ({self.accuracy_target:.0%})')
            plt.xlabel('Forecast Week')
            plt.ylabel('Accuracy')
            plt.title('Forecast Accuracy by Week (9-Week Horizon)')
            plt.legend()
            plt.ylim(0, 1.1)
            
            # Add value labels on bars
            for i, (week, acc) in enumerate(zip(weeks, accuracies)):
                plt.text(week, acc + 0.01, f'{acc:.1%}', ha='center')
            
            plt.savefig(output_path / 'accuracy_by_week.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 2. Model Performance Comparison
        if latest_report.model_rankings:
            plt.figure()
            models = list(latest_report.model_rankings.keys())
            performances = list(latest_report.model_rankings.values())
            
            colors = ['blue', 'green', 'orange', 'red']
            plt.bar(models, performances, color=colors[:len(models)])
            plt.axhline(y=self.accuracy_target, color='red', linestyle='--', label=f'Target ({self.accuracy_target:.0%})')
            plt.xlabel('Model')
            plt.ylabel('Average Accuracy')
            plt.title('Model Performance Comparison')
            plt.legend()
            plt.ylim(0, 1.1)
            
            # Add value labels
            for i, (model, perf) in enumerate(zip(models, performances)):
                plt.text(i, perf + 0.01, f'{perf:.1%}', ha='center')
            
            plt.savefig(output_path / 'model_comparison.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 3. Backtest Results Distribution
        if self.backtest_results:
            plt.figure()
            accuracies = [r.ensemble_accuracy for r in self.backtest_results.values()]
            
            plt.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(x=self.accuracy_target, color='red', linestyle='--', 
                       label=f'Target ({self.accuracy_target:.0%})')
            plt.axvline(x=np.mean(accuracies), color='blue', linestyle='-',
                       label=f'Mean ({np.mean(accuracies):.1%})')
            plt.xlabel('Accuracy')
            plt.ylabel('Number of Yarns')
            plt.title('Distribution of Forecast Accuracies')
            plt.legend()
            
            plt.savefig(output_path / 'accuracy_distribution.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 4. Forecast Bias Analysis
        if self.backtest_results:
            plt.figure()
            biases = [r.forecast_bias for r in self.backtest_results.values()]
            
            plt.hist(biases, bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(x=0, color='red', linestyle='--', label='No Bias')
            plt.axvline(x=np.mean(biases), color='blue', linestyle='-',
                       label=f'Mean Bias ({np.mean(biases):.1f})')
            plt.xlabel('Forecast Bias (+ = Over-forecast, - = Under-forecast)')
            plt.ylabel('Number of Yarns')
            plt.title('Forecast Bias Distribution')
            plt.legend()
            
            plt.savefig(output_path / 'forecast_bias.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Validation plots saved to {output_path}")
    
    def export_validation_report(self, filepath: str = "validation_report.json"):
        """Export validation report to JSON"""
        if not self.validation_history:
            logger.warning("No validation history to export")
            return
        
        latest_report = self.validation_history[-1]
        
        # Convert to dictionary
        report_dict = {
            'timestamp': latest_report.timestamp.isoformat(),
            'total_yarns_tested': latest_report.total_yarns_tested,
            'average_accuracy': latest_report.average_accuracy,
            'accuracy_by_week': latest_report.accuracy_by_week,
            'yarns_meeting_target': latest_report.yarns_meeting_target,
            'yarns_below_target': latest_report.yarns_below_target,
            'critical_yarns': latest_report.critical_yarns,
            'model_rankings': latest_report.model_rankings,
            'recommended_weights': latest_report.recommended_weights,
            'validation_passed': latest_report.validation_passed,
            'target_accuracy': self.accuracy_target
        }
        
        # Add backtest summary
        if self.backtest_results:
            backtest_summary = {
                'total_backtests': len(self.backtest_results),
                'average_ensemble_accuracy': np.mean([r.ensemble_accuracy for r in self.backtest_results.values()]),
                'average_confidence_coverage': np.mean([r.confidence_intervals_coverage for r in self.backtest_results.values()]),
                'average_forecast_bias': np.mean([r.forecast_bias for r in self.backtest_results.values()])
            }
            report_dict['backtest_summary'] = backtest_summary
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {filepath}")
    
    def continuous_validation(self,
                             data_source: callable,
                             interval_hours: int = 24,
                             sample_size: int = 50):
        """
        Run continuous validation at specified intervals
        
        Args:
            data_source: Function that returns current data
            interval_hours: Hours between validation runs
            sample_size: Number of yarns to validate each run
        """
        import threading
        import time
        
        def validation_loop():
            while True:
                try:
                    # Get current data
                    test_data = data_source()
                    
                    # Run validation
                    report = self.validate_forecast_system(test_data, sample_size)
                    
                    # Generate plots
                    self.generate_validation_plots()
                    
                    # Export report
                    self.export_validation_report()
                    
                    # Check if action needed
                    if not report.validation_passed:
                        logger.warning(f"Validation failed! Average accuracy: {report.average_accuracy:.2%}")
                        # Could trigger retraining or alert here
                    
                    # Sleep until next run
                    time.sleep(interval_hours * 3600)
                    
                except Exception as e:
                    logger.error(f"Error in continuous validation: {e}")
                    time.sleep(3600)  # Sleep 1 hour on error
        
        # Start in background thread
        thread = threading.Thread(target=validation_loop, daemon=True)
        thread.start()
        
        logger.info(f"Started continuous validation with {interval_hours}-hour interval")


# Example usage and testing
if __name__ == "__main__":
    # Initialize validation system
    validator = ForecastValidationSystem(accuracy_target=0.90)
    
    # Create sample test data
    test_data = {}
    for i in range(20):
        yarn_id = f"YARN{i:03d}"
        dates = pd.date_range(start='2023-01-01', periods=100, freq='W')
        consumed = -np.random.normal(1000, 200, 100)  # Negative consumption
        test_data[yarn_id] = pd.DataFrame({
            'date': dates,
            'Consumed': consumed
        })
    
    # Run validation
    print("Running forecast validation...")
    report = validator.validate_forecast_system(test_data, sample_size=10)
    
    # Print results
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Yarns Tested: {report.total_yarns_tested}")
    print(f"Average Accuracy: {report.average_accuracy:.2%}")
    print(f"Target Achievement: {report.yarns_meeting_target}/{report.total_yarns_tested}")
    print(f"Validation Status: {'PASSED' if report.validation_passed else 'FAILED'}")
    
    print(f"\nAccuracy by Week:")
    for week, acc in report.accuracy_by_week.items():
        status = "✓" if acc >= 0.90 else "✗"
        print(f"  Week {week}: {acc:.1%} {status}")
    
    print(f"\nModel Rankings:")
    for model, score in report.model_rankings.items():
        print(f"  {model}: {score:.2%}")
    
    print(f"\nRecommended Ensemble Weights:")
    for model, weight in report.recommended_weights.items():
        print(f"  {model}: {weight:.2%}")
    
    if report.critical_yarns:
        print(f"\nCritical Yarns (<80% accuracy):")
        for yarn in report.critical_yarns[:5]:
            print(f"  - {yarn}")
    
    # Generate plots
    validator.generate_validation_plots()
    print("\nValidation plots generated in 'validation_plots' directory")
    
    # Export report
    validator.export_validation_report()
    print("Validation report exported to 'validation_report.json'")