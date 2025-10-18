#!/usr/bin/env python3
"""
Weekly Forecast Generator (Enhanced Multi-Source Orchestrator)
Generates comprehensive 13-week sales forecasts by orchestrating:
- ML forecasts from historical sales data
- External forecasts from sales team/customers
- Intelligent blending of all sources
- Integration with time-phased planning

All data from Turso database (no Excel/CSV dependencies)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from src.database.turso_client import get_turso_client
from src.forecasting.enhanced_forecasting_engine import (
    EnhancedForecastingEngine,
    ForecastConfig
)
from src.forecasting.external_forecast_loader import ExternalForecastLoader
from src.forecasting.forecast_blender import ForecastBlender
from src.forecasting.forecast_actual_comparator import ForecastActualComparator
from src.forecasting.forecast_accuracy_tracker import ForecastAccuracyTracker
from src.forecasting.forecast_auto_retrain import AutomaticRetrainingSystem
from src.utils.style_mapper import get_style_mapper

logger = logging.getLogger(__name__)


class WeeklyForecastGenerator:
    """
    Orchestrates multi-source forecast generation for 13-week horizon
    Combines ML forecasts + External forecasts + Actual orders
    Provides complete production planning view
    """

    def __init__(self, forecast_weeks: int = 13):
        """
        Initialize weekly forecast generator with multi-source orchestration

        Args:
            forecast_weeks: Number of weeks to forecast (default 13 for full horizon)
        """
        self.turso_client = get_turso_client()
        self.forecast_weeks = forecast_weeks
        self.forecast_engine = None  # Lazy initialization

        # Multi-source orchestration components
        self.external_loader = ExternalForecastLoader()
        self.blender = ForecastBlender()
        self.comparator = ForecastActualComparator()
        self.style_mapper = get_style_mapper()

        # Training components
        self.accuracy_tracker = ForecastAccuracyTracker()
        self.auto_retrain_system = AutomaticRetrainingSystem()

        # Check if models need training on initialization
        self._check_training_status()

    def generate_comprehensive_forecast(
        self,
        styles: Optional[List[str]] = None,
        start_week: Optional[int] = None,
        external_forecast_files: Optional[List[str]] = None,
        blending_strategy: str = 'weighted_average',
        include_actual_orders: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive multi-source forecast with orchestration

        This is the MAIN method for forecast generation. It:
        1. Generates ML forecasts from historical data
        2. Loads external forecasts from sales team/customers
        3. Blends all sources intelligently
        4. Compares with actual orders
        5. Identifies high-confidence gaps for proactive production

        Args:
            styles: Styles to forecast (None = all with history)
            start_week: Starting week (default = current week)
            external_forecast_files: Paths to external forecast files
            blending_strategy: 'weighted_average', 'highest_confidence', 'conservative', 'aggressive'
            include_actual_orders: Include actual order comparison

        Returns:
            Complete forecast report with:
            {
                'ml_forecast': {style: {week: yards}},
                'external_forecasts': [...],
                'blended_forecast': {style: {week: {blended_yards, confidence, sources}}},
                'actual_orders': {style: {week: yards}},
                'comparison': {...},
                'proactive_production': [...],
                'variance_alerts': [...],
                'combined_schedule': {style: {week: {yards, source, confidence}}}
            }
        """
        try:
            logger.info("=" * 80)
            logger.info("COMPREHENSIVE FORECAST GENERATION")
            logger.info("=" * 80)

            # Determine start week
            if start_week is None:
                start_week = datetime.now().isocalendar()[1]

            logger.info(f"Forecast horizon: Weeks {start_week} to {start_week + self.forecast_weeks - 1}")

            # STEP 1: Generate ML forecasts
            logger.info("\n[1/6] Generating ML forecasts from historical data...")
            ml_forecast = self.generate_weekly_forecasts(
                styles=styles,
                start_week=start_week,
                use_ml=True
            )
            logger.info(f"✓ ML forecasts: {len(ml_forecast)} styles")

            # STEP 2: Load external forecasts
            logger.info("\n[2/6] Loading external forecasts...")
            external_forecasts = []

            if external_forecast_files:
                for file_path in external_forecast_files:
                    try:
                        ext_forecast = self.external_loader.load_sales_team_forecast(
                            file_path=file_path,
                            source_name='sales_team'
                        )
                        external_forecasts.append(ext_forecast)
                        logger.info(f"✓ Loaded external forecast: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")

            # Also load from Turso database (uploaded forecasts)
            db_external = self.external_loader.load_from_turso(
                start_week=start_week,
                weeks=self.forecast_weeks
            )
            if db_external:
                external_forecasts.append(db_external)
                logger.info(f"✓ Loaded external forecasts from Turso")

            logger.info(f"✓ Total external sources: {len(external_forecasts)}")

            # STEP 3: Blend forecasts
            logger.info(f"\n[3/6] Blending forecasts (strategy: {blending_strategy})...")
            blended_forecast = self.blender.blend_forecasts(
                ml_forecast=ml_forecast,
                external_forecasts=external_forecasts,
                blending_strategy=blending_strategy
            )
            logger.info(f"✓ Blended forecast: {len(blended_forecast)} styles")

            # STEP 4: Get actual orders
            actual_orders = {}
            if include_actual_orders:
                logger.info("\n[4/6] Loading actual confirmed orders...")
                actual_orders = self.comparator.get_actual_orders(
                    start_week=start_week,
                    weeks=self.forecast_weeks
                )
                logger.info(f"✓ Actual orders: {len(actual_orders)} styles")

            # STEP 5: Compare forecast vs actual
            comparison = {}
            proactive_production = []
            variance_alerts = []

            if include_actual_orders:
                logger.info("\n[5/6] Comparing forecast vs actual...")
                comparison = self.comparator.compare_forecast_vs_actual(
                    blended_forecast=blended_forecast,
                    actual_orders=actual_orders,
                    start_week=start_week
                )

                # Generate proactive production recommendations
                proactive_production = self.comparator.generate_proactive_production_list(
                    comparison_result=comparison,
                    max_items=50
                )

                # Get variance alerts
                variance_alerts = self.comparator.get_variance_alerts(
                    comparison_result=comparison,
                    threshold_pct=30.0
                )

                logger.info(f"✓ High-confidence gaps: {len(proactive_production)}")
                logger.info(f"✓ Variance alerts: {len(variance_alerts)}")

            # STEP 6: Create combined production schedule
            logger.info("\n[6/6] Creating combined production schedule...")
            combined_schedule = self.comparator.get_combined_production_schedule(
                actual_orders=actual_orders,
                blended_forecast=blended_forecast,
                include_forecast_threshold=0.85
            )
            logger.info(f"✓ Combined schedule: {len(combined_schedule)} styles")

            logger.info("\n" + "=" * 80)
            logger.info("FORECAST GENERATION COMPLETE")
            logger.info("=" * 80)

            return {
                'ml_forecast': ml_forecast,
                'external_forecasts': external_forecasts,
                'blended_forecast': blended_forecast,
                'actual_orders': actual_orders,
                'comparison': comparison,
                'proactive_production': proactive_production,
                'variance_alerts': variance_alerts,
                'combined_schedule': combined_schedule,
                'metadata': {
                    'start_week': start_week,
                    'forecast_weeks': self.forecast_weeks,
                    'blending_strategy': blending_strategy,
                    'generated_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.exception(f"Error in comprehensive forecast generation: {e}")
            return {
                'error': str(e),
                'ml_forecast': {},
                'combined_schedule': {}
            }

    def generate_weekly_forecasts(
        self,
        styles: Optional[List[str]] = None,
        start_week: Optional[int] = None,
        use_ml: bool = True
    ) -> Dict[str, Dict[int, float]]:
        """
        Generate ML-based weekly sales forecasts (internal method)

        Args:
            styles: List of styles to forecast (None = all styles with history)
            start_week: Starting week number for forecast (defaults to current week)
            use_ml: Use ML forecasting (True) or simple moving average (False)

        Returns:
            {style_id: {week_num: yards}} - Weekly sales forecast by style

        Example:
            {
                "STYLE001": {42: 1000, 43: 1050, 44: 1100},
                "STYLE002": {42: 500, 43: 525, 44: 550}
            }
        """
        try:
            # Determine forecast start week
            if start_week is None:
                start_week = datetime.now().isocalendar()[1]

            # Get styles to forecast
            if styles is None:
                styles = self._get_forecastable_styles()
                logger.info(f"Auto-selected {len(styles)} styles with sufficient history")
            else:
                logger.info(f"Forecasting {len(styles)} specified styles")

            # Generate forecasts
            forecasts: Dict[str, Dict[int, float]] = {}

            for style in styles:
                try:
                    # Get historical sales
                    historical_sales = self._get_historical_sales(style)

                    if historical_sales.empty:
                        logger.warning(f"No historical sales for style {style}, skipping")
                        continue

                    # Generate forecast
                    if use_ml and len(historical_sales) >= 10:
                        style_forecast = self._ml_forecast(style, historical_sales, start_week)
                    else:
                        style_forecast = self._simple_forecast(style, historical_sales, start_week)

                    if style_forecast:
                        forecasts[style] = style_forecast

                except Exception as e:
                    logger.error(f"Error forecasting style {style}: {e}")
                    continue

            logger.info(
                f"Generated forecasts for {len(forecasts)}/{len(styles)} styles, "
                f"{self.forecast_weeks} weeks starting week {start_week}"
            )

            return forecasts

        except Exception as e:
            logger.exception(f"Error in generate_weekly_forecasts: {e}")
            return {}

    def _get_forecastable_styles(self, min_records: int = 10) -> List[str]:
        """
        Get list of styles with sufficient historical data for forecasting

        Args:
            min_records: Minimum number of historical records required

        Returns:
            List of style codes
        """
        try:
            sql = """
                SELECT style, COUNT(*) as record_count
                FROM historical_sales
                GROUP BY style
                HAVING record_count >= ?
                ORDER BY record_count DESC
            """

            rows = self.turso_client.execute(sql, [min_records])
            styles = [row['style'] for row in rows]

            return styles

        except Exception as e:
            logger.error(f"Error getting forecastable styles: {e}")
            return []

    def _get_historical_sales(
        self,
        style: str,
        lookback_days: int = 180
    ) -> pd.DataFrame:
        """
        Get historical sales data for a style from Turso

        Args:
            style: Style code
            lookback_days: Number of days to look back

        Returns:
            DataFrame with columns: date, quantity (in yards)
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            sql = """
                SELECT date, SUM(quantity) as quantity
                FROM historical_sales
                WHERE style = ?
                  AND date >= ?
                  AND date <= ?
                  AND units = 'yards'
                GROUP BY date
                ORDER BY date ASC
            """

            rows = self.turso_client.execute(sql, [style, start_date, end_date])

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df['quantity'] = df['quantity'].astype(float)

            return df

        except Exception as e:
            logger.error(f"Error getting historical sales for {style}: {e}")
            return pd.DataFrame()

    def _ml_forecast(
        self,
        style: str,
        historical_sales: pd.DataFrame,
        start_week: int
    ) -> Dict[int, float]:
        """
        Generate ML-based forecast using EnhancedForecastingEngine

        Args:
            style: Style code
            historical_sales: Historical sales DataFrame
            start_week: Starting week number

        Returns:
            {week_num: yards} - Weekly forecast
        """
        try:
            # Initialize forecast engine if needed
            if self.forecast_engine is None:
                config = ForecastConfig(
                    horizon_weeks=self.forecast_weeks,
                    min_accuracy_threshold=0.85,  # Slightly lower for sales forecast
                    use_orders=False  # Only use historical for sales forecast
                )
                self.forecast_engine = EnhancedForecastingEngine(config)

            # Prepare data in format expected by forecast engine
            # Convert daily sales to weekly aggregates
            historical_sales['week'] = historical_sales['date'].dt.isocalendar().week
            historical_sales['year'] = historical_sales['date'].dt.year

            weekly_sales = historical_sales.groupby(['year', 'week'])['quantity'].sum().reset_index()
            weekly_sales['date'] = pd.to_datetime(
                weekly_sales['year'].astype(str) + '-W' +
                weekly_sales['week'].astype(str).str.zfill(2) + '-1',
                format='%Y-W%W-%w'
            )
            weekly_sales = weekly_sales[['date', 'quantity']].set_index('date')

            # Generate forecast
            result = self.forecast_engine.forecast(
                yarn_id=style,  # Using style instead of yarn_id
                historical_data=weekly_sales,
                order_data=None
            )

            # Convert predictions to weekly format
            weekly_forecast = {}
            predictions_df = result.predictions

            for i in range(self.forecast_weeks):
                week_num = (start_week + i) % 53  # Handle year rollover
                if week_num == 0:
                    week_num = 52

                # Get prediction for this week
                if i < len(predictions_df):
                    yards = predictions_df.iloc[i]['forecast']
                    weekly_forecast[week_num] = max(0, float(yards))  # Ensure non-negative
                else:
                    # Fallback to average if not enough predictions
                    avg_weekly = historical_sales['quantity'].mean() / 7  # Daily to weekly
                    weekly_forecast[week_num] = max(0, float(avg_weekly))

            return weekly_forecast

        except Exception as e:
            logger.warning(f"ML forecast failed for {style}, falling back to simple: {e}")
            return self._simple_forecast(style, historical_sales, start_week)

    def _simple_forecast(
        self,
        style: str,
        historical_sales: pd.DataFrame,
        start_week: int
    ) -> Dict[int, float]:
        """
        Generate simple moving average forecast

        Args:
            style: Style code
            historical_sales: Historical sales DataFrame
            start_week: Starting week number

        Returns:
            {week_num: yards} - Weekly forecast
        """
        try:
            # Calculate weekly moving average
            historical_sales['week'] = historical_sales['date'].dt.isocalendar().week

            weekly_totals = historical_sales.groupby('week')['quantity'].sum()

            # Use 4-week moving average for forecast
            if len(weekly_totals) >= 4:
                avg_weekly_sales = weekly_totals.tail(4).mean()
            else:
                avg_weekly_sales = weekly_totals.mean()

            # Apply growth trend if detectable
            growth_factor = self._calculate_growth_trend(weekly_totals)

            # Generate forecast for each week
            weekly_forecast = {}
            for i in range(self.forecast_weeks):
                week_num = (start_week + i) % 53
                if week_num == 0:
                    week_num = 52

                # Apply growth factor for future weeks
                forecasted_yards = avg_weekly_sales * (growth_factor ** i)
                weekly_forecast[week_num] = max(0, float(forecasted_yards))

            logger.debug(
                f"Simple forecast for {style}: avg={avg_weekly_sales:.2f}, "
                f"growth={growth_factor:.3f}"
            )

            return weekly_forecast

        except Exception as e:
            logger.error(f"Error in simple forecast for {style}: {e}")
            return {}

    def _calculate_growth_trend(self, weekly_sales: pd.Series) -> float:
        """
        Calculate growth trend factor from historical data

        Args:
            weekly_sales: Series of weekly sales totals

        Returns:
            Growth factor (1.0 = no growth, 1.05 = 5% growth)
        """
        try:
            if len(weekly_sales) < 4:
                return 1.0  # No trend with insufficient data

            # Calculate linear trend
            x = np.arange(len(weekly_sales))
            y = weekly_sales.values

            # Simple linear regression
            slope, _ = np.polyfit(x, y, 1)

            # Convert slope to growth factor
            avg_sales = y.mean()
            if avg_sales > 0:
                weekly_growth_rate = slope / avg_sales
                growth_factor = 1 + weekly_growth_rate
                # Cap growth factor between 0.9 and 1.1 (±10%)
                growth_factor = max(0.9, min(1.1, growth_factor))
            else:
                growth_factor = 1.0

            return growth_factor

        except Exception as e:
            logger.warning(f"Error calculating growth trend: {e}")
            return 1.0  # Default to no growth

    def validate_forecast_quality(
        self,
        forecasts: Dict[str, Dict[int, float]]
    ) -> Dict[str, Any]:
        """
        Validate quality of generated forecasts

        Args:
            forecasts: Generated forecasts {style: {week: yards}}

        Returns:
            Validation report with quality metrics
        """
        report = {
            'total_styles': len(forecasts),
            'total_weeks_forecasted': 0,
            'total_forecasted_yards': 0,
            'avg_weekly_yards_per_style': 0,
            'styles_with_zero_forecast': [],
            'styles_with_high_variance': []
        }

        total_yards = 0
        total_weeks = 0

        for style, weekly_forecast in forecasts.items():
            style_total = sum(weekly_forecast.values())
            total_yards += style_total
            total_weeks += len(weekly_forecast)

            # Check for zero forecast
            if style_total == 0:
                report['styles_with_zero_forecast'].append(style)

            # Check for high variance
            if len(weekly_forecast) > 1:
                values = list(weekly_forecast.values())
                std_dev = np.std(values)
                mean_val = np.mean(values)
                if mean_val > 0 and (std_dev / mean_val) > 0.5:  # CV > 50%
                    report['styles_with_high_variance'].append(style)

        report['total_weeks_forecasted'] = total_weeks
        report['total_forecasted_yards'] = total_yards
        if report['total_styles'] > 0:
            report['avg_weekly_yards_per_style'] = (
                total_yards / total_weeks * self.forecast_weeks
                if total_weeks > 0 else 0
            )

        logger.info(
            f"Forecast validation: {report['total_styles']} styles, "
            f"{report['total_forecasted_yards']:.0f} yards forecasted"
        )

        return report

    def train_models(
        self,
        styles: Optional[List[str]] = None,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train ML models on historical data from Turso database

        Args:
            styles: List of styles to train (None = all forecastable styles)
            force_retrain: Force retraining even if models are up to date

        Returns:
            Training results with accuracy metrics and weight adjustments
        """
        try:
            logger.info("=" * 80)
            logger.info("TRAINING ML MODELS")
            logger.info("=" * 80)

            # Check if training needed
            if not force_retrain and self.forecast_engine and not self.forecast_engine.needs_retraining():
                logger.info("Models are up to date, skipping training")
                return {
                    "status": "skipped",
                    "reason": "Models up to date",
                    "last_training": self.forecast_engine.last_training_date.isoformat() if self.forecast_engine.last_training_date else None
                }

            # Initialize forecast engine if needed
            if self.forecast_engine is None:
                config = ForecastConfig(
                    horizon_weeks=self.forecast_weeks,
                    min_accuracy_threshold=0.85,
                    use_orders=False
                )
                self.forecast_engine = EnhancedForecastingEngine(config)

            # Load training data from Turso
            logger.info("Loading training data from Turso database...")
            training_data = self.auto_retrain_system.load_training_data(
                min_records=10,
                lookback_days=180
            )

            if not training_data:
                return {
                    "status": "failed",
                    "reason": "No training data available",
                    "styles_found": 0
                }

            logger.info(f"✓ Loaded training data for {len(training_data)} styles")

            # Retrain models
            logger.info("Training models...")
            model_accuracies = self.forecast_engine.retrain_models(training_data)

            # Auto-tune blend weights based on accuracy
            logger.info("Auto-tuning blend weights...")
            weight_adjustments = self._tune_blend_weights()

            # Prepare results
            results = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "styles_trained": len(training_data),
                "model_accuracies": model_accuracies,
                "weight_adjustments": weight_adjustments,
                "average_accuracy": float(np.mean(list(model_accuracies.values()))) if model_accuracies else 0,
                "last_training_date": self.forecast_engine.last_training_date.isoformat() if self.forecast_engine.last_training_date else None
            }

            logger.info("=" * 80)
            logger.info(f"✓ TRAINING COMPLETE")
            logger.info(f"  Styles trained: {results['styles_trained']}")
            logger.info(f"  Avg accuracy: {results['average_accuracy']:.1%}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.exception(f"Error training models: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _tune_blend_weights(self) -> Dict[str, Any]:
        """
        Auto-tune blending weights based on recent forecast accuracy

        Returns:
            Weight adjustment details
        """
        try:
            current_weights = self.blender.source_weights.copy()

            # Get weight recommendations from accuracy tracker
            recommendations = self.accuracy_tracker.recommend_weight_adjustments(
                current_weights=current_weights,
                lookback_weeks=13
            )

            # Apply recommended weights if improvement is significant (>2%)
            if recommendations.get('overall_improvement', 0) > 2.0:
                self.blender.source_weights = recommendations['recommended_weights']

                # Store weight adjustment in database
                self.accuracy_tracker.store_weight_adjustment(
                    weights=recommendations['recommended_weights'],
                    reason=f"Auto-tuned: {recommendations['overall_improvement']:.1f}% improvement"
                )

                logger.info(f"✓ Blend weights auto-tuned: {recommendations['overall_improvement']:.1f}% improvement")

                return {
                    "status": "adjusted",
                    "improvement_pct": recommendations['overall_improvement'],
                    "old_weights": current_weights,
                    "new_weights": recommendations['recommended_weights'],
                    "changes": recommendations.get('changes', [])
                }
            else:
                logger.info(f"Blend weights unchanged (improvement: {recommendations.get('overall_improvement', 0):.1f}%)")
                return {
                    "status": "unchanged",
                    "improvement_pct": recommendations.get('overall_improvement', 0),
                    "current_weights": current_weights
                }

        except Exception as e:
            logger.exception(f"Error tuning blend weights: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def needs_training(self) -> bool:
        """
        Check if models need retraining

        Returns:
            True if training is needed
        """
        if self.forecast_engine is None:
            return True

        return self.forecast_engine.needs_retraining()

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and metrics

        Returns:
            Training status including last training date, accuracy, weights
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "needs_training": self.needs_training(),
                "forecast_engine_initialized": self.forecast_engine is not None
            }

            # Add forecast engine info
            if self.forecast_engine:
                status["last_training_date"] = (
                    self.forecast_engine.last_training_date.isoformat()
                    if self.forecast_engine.last_training_date else None
                )
                status["ensemble_weights"] = self.forecast_engine.config.ensemble_weights

                # Days since training
                if self.forecast_engine.last_training_date:
                    days_since = (datetime.now() - self.forecast_engine.last_training_date).days
                    status["days_since_training"] = days_since
                else:
                    status["days_since_training"] = None
            else:
                status["last_training_date"] = None
                status["ensemble_weights"] = None
                status["days_since_training"] = None

            # Add blend weights
            status["blend_weights"] = self.blender.source_weights

            # Add recent accuracy report
            try:
                accuracy_report = self.accuracy_tracker.generate_accuracy_report(lookback_weeks=4)
                status["accuracy_report"] = accuracy_report
            except Exception as e:
                logger.warning(f"Could not load accuracy report: {e}")
                status["accuracy_report"] = None

            return status

        except Exception as e:
            logger.exception(f"Error getting training status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _check_training_status(self):
        """
        Check training status on initialization and log warning if needed
        """
        try:
            if self.needs_training():
                logger.warning(
                    "⚠️  ML models need training! "
                    "Call train_models() or trigger via API endpoint."
                )
            else:
                logger.info("✓ ML models are up to date")
        except Exception as e:
            logger.debug(f"Could not check training status: {e}")


def main():
    """Test comprehensive forecast generation with multi-source orchestration"""

    print("\n" + "🎯 " * 20)
    print("COMPREHENSIVE FORECAST GENERATOR TEST")
    print("🎯 " * 20 + "\n")

    # Initialize with 13-week horizon
    generator = WeeklyForecastGenerator(forecast_weeks=13)

    # Test 1: Simple ML forecast only
    print("=== Test 1: ML Forecast Only (No External Sources) ===")
    result = generator.generate_comprehensive_forecast(
        styles=None,  # Auto-select styles
        start_week=None,  # Use current week
        external_forecast_files=None,
        blending_strategy='weighted_average',
        include_actual_orders=True
    )

    print(f"\nResults:")
    print(f"  ML forecasts: {len(result.get('ml_forecast', {}))} styles")
    print(f"  Blended forecasts: {len(result.get('blended_forecast', {}))} styles")
    print(f"  Actual orders: {len(result.get('actual_orders', {}))} styles")
    print(f"  Combined schedule: {len(result.get('combined_schedule', {}))} styles")

    # Show summary
    comparison = result.get('comparison', {})
    if 'summary' in comparison:
        print(f"\nComparison Summary:")
        for key, value in comparison['summary'].items():
            print(f"  {key}: {value}")

    # Show top proactive production recommendations
    proactive = result.get('proactive_production', [])
    if proactive:
        print(f"\nTop 5 Proactive Production Recommendations:")
        for rec in proactive[:5]:
            print(f"  {rec['priority']}. {rec['style']} Week {rec['week']}")
            print(f"     Forecasted: {rec['forecasted_yards']:.0f} yards")
            print(f"     Confidence: {rec['confidence']:.1%}")
            print(f"     Action: {rec['action']}")

    # Show variance alerts
    alerts = result.get('variance_alerts', [])
    if alerts:
        print(f"\nTop 3 Variance Alerts:")
        for alert in alerts[:3]:
            print(f"  {alert['style']} Week {alert['week']}: {alert['variance_pct']:+.1f}%")
            print(f"     Forecast: {alert['forecasted_yards']:.0f}, Actual: {alert['actual_yards']:.0f}")
            print(f"     Severity: {alert['severity']}")

    # Test 2: Show sample combined schedule
    print("\n=== Test 2: Sample Combined Production Schedule ===")
    combined = result.get('combined_schedule', {})
    if combined:
        sample_styles = list(combined.keys())[:3]
        for style in sample_styles:
            print(f"\n{style}:")
            weeks = combined[style]
            for week in sorted(weeks.keys())[:5]:  # First 5 weeks
                week_data = weeks[week]
                source_icon = "✅" if week_data['source'] == 'actual_order' else "🔮"
                print(f"  Week {week}: {week_data['yards']:.0f} yards {source_icon} "
                      f"({week_data['source']}, confidence: {week_data['confidence']:.1%})")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
