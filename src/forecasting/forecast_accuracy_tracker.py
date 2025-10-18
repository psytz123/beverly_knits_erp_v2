#!/usr/bin/env python3
"""
Forecast Accuracy Tracker
Tracks forecast vs actual performance to auto-tune blending weights
Stores all data in Turso database (no Excel/CSV files)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from src.database.turso_client import get_turso_client

logger = logging.getLogger(__name__)


class ForecastAccuracyTracker:
    """
    Track accuracy of each forecast source to tune weights
    All data stored in Turso database
    """

    def __init__(self):
        """Initialize forecast accuracy tracker with Turso client"""
        self.turso_client = get_turso_client()

    def track_forecast_vs_actual(
        self,
        forecast_date: datetime,
        forecasts: Dict[str, Dict[int, Dict[str, Any]]],
        actuals: Dict[str, Dict[int, float]]
    ) -> int:
        """
        Store forecast vs actual for later analysis

        Args:
            forecast_date: When forecast was made
            forecasts: {style: {week: {yards, confidence, source}}}
            actuals: {style: {week: actual_yards}}

        Returns:
            Number of records stored

        Example:
            forecasts = {
                "STYLE001": {
                    42: {'yards': 1000, 'confidence': 0.85, 'source': 'ml_historical'}
                }
            }
            actuals = {
                "STYLE001": {42: 950}
            }
        """
        try:
            stored_count = 0
            forecast_date_str = forecast_date.strftime('%Y-%m-%d')

            for style, week_forecasts in forecasts.items():
                for week, forecast_data in week_forecasts.items():
                    # Get actual value for this style-week
                    actual_yards = actuals.get(style, {}).get(week)

                    if actual_yards is None:
                        continue  # Skip if no actual data yet

                    forecasted_yards = forecast_data.get('yards', 0)
                    source = forecast_data.get('source', 'unknown')

                    # Calculate errors
                    error_pct = self._calculate_error_percentage(
                        forecasted_yards, actual_yards
                    )
                    absolute_error = abs(forecasted_yards - actual_yards)

                    # Insert into Turso
                    sql = """
                    INSERT INTO forecast_accuracy
                    (style, week_number, forecast_date, forecasted_quantity,
                     actual_quantity, source, error_pct, absolute_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """

                    params = [
                        style,
                        week,
                        forecast_date_str,
                        forecasted_yards,
                        actual_yards,
                        source,
                        error_pct,
                        absolute_error
                    ]

                    self.turso_client.execute(sql, params)
                    stored_count += 1

            logger.info(f"✓ Stored {stored_count} forecast accuracy records")
            return stored_count

        except Exception as e:
            logger.exception(f"Error tracking forecast vs actual: {e}")
            return 0

    def calculate_source_accuracy(
        self,
        source: str,
        lookback_weeks: int = 13
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics by source

        Args:
            source: Source name ('ml_historical', 'sales_team', etc.)
            lookback_weeks: Number of weeks to look back

        Returns:
            Accuracy metrics:
            - mape: Mean Absolute Percentage Error
            - bias: Average over/under forecasting tendency
            - hit_rate: % within tolerance (±10%)
            - rmse: Root Mean Squared Error

        Example:
            {
                'mape': 12.5,
                'bias': -2.3,  # Negative = under-forecasting
                'hit_rate': 0.78,
                'rmse': 150.2,
                'sample_size': 45
            }
        """
        try:
            cutoff_date = (datetime.now() - timedelta(weeks=lookback_weeks)).strftime('%Y-%m-%d')

            sql = """
            SELECT
                forecasted_quantity,
                actual_quantity,
                error_pct,
                absolute_error
            FROM forecast_accuracy
            WHERE source = ?
              AND forecast_date >= ?
              AND actual_quantity IS NOT NULL
            """

            rows = self.turso_client.execute(sql, [source, cutoff_date])

            if not rows or len(rows) == 0:
                logger.warning(f"No accuracy data found for source: {source}")
                return {
                    'mape': 0.0,
                    'bias': 0.0,
                    'hit_rate': 0.0,
                    'rmse': 0.0,
                    'sample_size': 0
                }

            # Extract values
            forecasted = np.array([row['forecasted_quantity'] for row in rows])
            actuals = np.array([row['actual_quantity'] for row in rows])
            errors_pct = np.array([row['error_pct'] for row in rows])
            abs_errors = np.array([row['absolute_error'] for row in rows])

            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs(errors_pct))

            # Calculate Bias (positive = over-forecasting, negative = under-forecasting)
            bias = np.mean(errors_pct)

            # Calculate Hit Rate (% within ±10%)
            tolerance = 10.0  # 10%
            within_tolerance = np.abs(errors_pct) <= tolerance
            hit_rate = np.sum(within_tolerance) / len(errors_pct)

            # Calculate RMSE (Root Mean Squared Error)
            squared_errors = (forecasted - actuals) ** 2
            rmse = np.sqrt(np.mean(squared_errors))

            return {
                'mape': float(mape),
                'bias': float(bias),
                'hit_rate': float(hit_rate),
                'rmse': float(rmse),
                'sample_size': len(rows)
            }

        except Exception as e:
            logger.exception(f"Error calculating source accuracy for {source}: {e}")
            return {
                'mape': 0.0,
                'bias': 0.0,
                'hit_rate': 0.0,
                'rmse': 0.0,
                'sample_size': 0
            }

    def recommend_weight_adjustments(
        self,
        current_weights: Dict[str, float],
        lookback_weeks: int = 13
    ) -> Dict[str, Any]:
        """
        Analyze accuracy and suggest weight changes

        Args:
            current_weights: Current blend weights {source: weight}
            lookback_weeks: Weeks to analyze

        Returns:
            Recommendations with new weights and reasoning

        Example:
            {
                'recommended_weights': {'ml_historical': 0.35, 'sales_team': 0.45, ...},
                'changes': [{'source': 'sales_team', 'old': 0.35, 'new': 0.45, 'reason': '...'}],
                'overall_improvement': 5.2  # % improvement in MAPE
            }
        """
        try:
            # Get accuracy for each source
            source_metrics = {}
            for source in current_weights.keys():
                metrics = self.calculate_source_accuracy(source, lookback_weeks)
                source_metrics[source] = metrics

            # Calculate new weights based on performance
            # Lower MAPE = better performance = higher weight
            recommended_weights = {}
            changes = []

            # Calculate inverse MAPE scores (lower MAPE = higher score)
            total_inverse_mape = 0
            for source, metrics in source_metrics.items():
                if metrics['sample_size'] > 0:
                    # Inverse MAPE: 1 / (1 + MAPE)
                    # This gives higher scores to lower MAPE
                    inverse_mape = 1.0 / (1.0 + metrics['mape'])
                    source_metrics[source]['inverse_mape'] = inverse_mape
                    total_inverse_mape += inverse_mape
                else:
                    source_metrics[source]['inverse_mape'] = 0

            # Calculate recommended weights proportional to inverse MAPE
            for source in current_weights.keys():
                if total_inverse_mape > 0:
                    new_weight = source_metrics[source]['inverse_mape'] / total_inverse_mape
                else:
                    new_weight = current_weights[source]  # No change if no data

                # Apply smoothing: don't change weights too drastically
                # New weight = 70% new + 30% old (prevents wild swings)
                smoothed_weight = 0.7 * new_weight + 0.3 * current_weights[source]

                recommended_weights[source] = smoothed_weight

                # Track changes
                if abs(smoothed_weight - current_weights[source]) > 0.02:  # >2% change
                    reason = self._explain_weight_change(
                        source,
                        source_metrics[source],
                        current_weights[source],
                        smoothed_weight
                    )
                    changes.append({
                        'source': source,
                        'old_weight': current_weights[source],
                        'new_weight': smoothed_weight,
                        'reason': reason,
                        'mape': source_metrics[source]['mape']
                    })

            # Calculate overall improvement
            current_weighted_mape = sum(
                current_weights[s] * source_metrics[s]['mape']
                for s in current_weights
                if source_metrics[s]['sample_size'] > 0
            )

            recommended_weighted_mape = sum(
                recommended_weights[s] * source_metrics[s]['mape']
                for s in recommended_weights
                if source_metrics[s]['sample_size'] > 0
            )

            improvement_pct = 0
            if current_weighted_mape > 0:
                improvement_pct = (
                    (current_weighted_mape - recommended_weighted_mape) /
                    current_weighted_mape * 100
                )

            return {
                'recommended_weights': recommended_weights,
                'changes': changes,
                'overall_improvement': improvement_pct,
                'source_metrics': source_metrics
            }

        except Exception as e:
            logger.exception(f"Error recommending weight adjustments: {e}")
            return {
                'recommended_weights': current_weights,
                'changes': [],
                'overall_improvement': 0,
                'source_metrics': {}
            }

    def generate_accuracy_report(
        self,
        lookback_weeks: int = 13
    ) -> Dict[str, Any]:
        """
        Generate weekly accuracy report for stakeholders

        Args:
            lookback_weeks: Weeks to include

        Returns:
            Comprehensive accuracy report
        """
        try:
            # Get all unique sources
            sql = """
            SELECT DISTINCT source
            FROM forecast_accuracy
            WHERE forecast_date >= ?
            """

            cutoff_date = (datetime.now() - timedelta(weeks=lookback_weeks)).strftime('%Y-%m-%d')
            sources_result = self.turso_client.execute(sql, [cutoff_date])

            if not sources_result:
                return {'message': 'No forecast accuracy data available'}

            sources = [row['source'] for row in sources_result]

            # Get accuracy for each source
            source_performance = {}
            for source in sources:
                metrics = self.calculate_source_accuracy(source, lookback_weeks)
                source_performance[source] = metrics

            # Find best and worst performers
            sources_with_data = {
                s: m for s, m in source_performance.items()
                if m['sample_size'] > 0
            }

            if sources_with_data:
                best_source = min(sources_with_data.items(), key=lambda x: x[1]['mape'])
                worst_source = max(sources_with_data.items(), key=lambda x: x[1]['mape'])
            else:
                best_source = ('none', {'mape': 0})
                worst_source = ('none', {'mape': 0})

            # Get styles with highest forecast error
            sql = """
            SELECT
                style,
                AVG(ABS(error_pct)) as avg_error,
                COUNT(*) as sample_size
            FROM forecast_accuracy
            WHERE forecast_date >= ?
              AND actual_quantity IS NOT NULL
            GROUP BY style
            HAVING sample_size >= 3
            ORDER BY avg_error DESC
            LIMIT 10
            """

            problematic_styles = self.turso_client.execute(sql, [cutoff_date])

            return {
                'period': f'Last {lookback_weeks} weeks',
                'sources_analyzed': len(sources),
                'source_performance': source_performance,
                'best_performer': {
                    'source': best_source[0],
                    'mape': best_source[1]['mape']
                },
                'worst_performer': {
                    'source': worst_source[0],
                    'mape': worst_source[1]['mape']
                },
                'problematic_styles': [
                    {
                        'style': row['style'],
                        'avg_error_pct': row['avg_error'],
                        'sample_size': row['sample_size']
                    }
                    for row in (problematic_styles or [])
                ],
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.exception(f"Error generating accuracy report: {e}")
            return {'error': str(e)}

    def store_weight_adjustment(
        self,
        weights: Dict[str, float],
        reason: str
    ) -> bool:
        """
        Store weight adjustment in history

        Args:
            weights: New weights {source: weight}
            reason: Reason for change

        Returns:
            Success status
        """
        try:
            effective_date = datetime.now().strftime('%Y-%m-%d')

            for source, weight in weights.items():
                sql = """
                INSERT INTO forecast_blend_weights
                (effective_date, source, weight, reason)
                VALUES (?, ?, ?, ?)
                """

                self.turso_client.execute(
                    sql,
                    [effective_date, source, weight, reason]
                )

            logger.info(f"✓ Stored weight adjustments for {len(weights)} sources")
            return True

        except Exception as e:
            logger.exception(f"Error storing weight adjustment: {e}")
            return False

    def _calculate_error_percentage(
        self,
        forecasted: float,
        actual: float
    ) -> float:
        """Calculate percentage error"""
        if actual == 0:
            return 0 if forecasted == 0 else 100

        return ((forecasted - actual) / actual) * 100

    def _explain_weight_change(
        self,
        source: str,
        metrics: Dict[str, float],
        old_weight: float,
        new_weight: float
    ) -> str:
        """Generate human-readable explanation for weight change"""
        change_pct = ((new_weight - old_weight) / old_weight) * 100

        if new_weight > old_weight:
            direction = "increased"
        else:
            direction = "decreased"

        return (
            f"{source} weight {direction} by {abs(change_pct):.1f}% "
            f"(MAPE: {metrics['mape']:.1f}%, Hit Rate: {metrics['hit_rate']*100:.1f}%)"
        )


def main():
    """Test forecast accuracy tracker"""

    tracker = ForecastAccuracyTracker()

    # Test 1: Track some forecasts
    print("=== Test 1: Track Forecast vs Actual ===")

    forecast_date = datetime.now() - timedelta(weeks=2)
    forecasts = {
        "STYLE001": {
            42: {'yards': 1000, 'confidence': 0.85, 'source': 'ml_historical'},
            43: {'yards': 1050, 'confidence': 0.82, 'source': 'ml_historical'}
        }
    }

    actuals = {
        "STYLE001": {
            42: 950,  # ML was off by 5%
            43: 1100  # ML was off by -5%
        }
    }

    stored = tracker.track_forecast_vs_actual(forecast_date, forecasts, actuals)
    print(f"Stored {stored} accuracy records")

    # Test 2: Calculate accuracy
    print("\n=== Test 2: Calculate Source Accuracy ===")

    metrics = tracker.calculate_source_accuracy('ml_historical', lookback_weeks=4)
    print(f"ML Historical Performance:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Bias: {metrics['bias']:.2f}%")
    print(f"  Hit Rate: {metrics['hit_rate']*100:.1f}%")
    print(f"  Sample Size: {metrics['sample_size']}")

    # Test 3: Recommend weight adjustments
    print("\n=== Test 3: Weight Recommendations ===")

    current_weights = {
        'ml_historical': 0.40,
        'sales_team': 0.35,
        'customer_commitment': 0.20,
        'market_intelligence': 0.05
    }

    recommendations = tracker.recommend_weight_adjustments(current_weights)

    print("Current weights → Recommended weights:")
    for source in current_weights:
        old = current_weights[source]
        new = recommendations['recommended_weights'].get(source, old)
        change = new - old
        print(f"  {source}: {old:.2f} → {new:.2f} ({change:+.2f})")

    if recommendations['changes']:
        print("\nRecommended changes:")
        for change in recommendations['changes']:
            print(f"  - {change['reason']}")

    # Test 4: Generate report
    print("\n=== Test 4: Accuracy Report ===")

    report = tracker.generate_accuracy_report(lookback_weeks=4)
    if 'best_performer' in report:
        print(f"Best performer: {report['best_performer']['source']} "
              f"(MAPE: {report['best_performer']['mape']:.2f}%)")
        print(f"Worst performer: {report['worst_performer']['source']} "
              f"(MAPE: {report['worst_performer']['mape']:.2f}%)")


if __name__ == "__main__":
    main()
