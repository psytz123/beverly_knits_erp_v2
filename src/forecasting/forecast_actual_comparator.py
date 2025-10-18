#!/usr/bin/env python3
"""
Forecast-Actual Comparator
Compares forecasted demand with actual confirmed orders
Identifies high-confidence gaps for proactive production planning
All data from Turso database
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from src.database.turso_client import get_turso_client
from src.utils.style_mapper import get_style_mapper

logger = logging.getLogger(__name__)


class ForecastActualComparator:
    """
    Compare forecasted demand with actual confirmed orders
    Identify high-confidence gaps for proactive production
    """

    def __init__(self, confidence_threshold: float = 0.85):
        """
        Initialize comparator with Turso client

        Args:
            confidence_threshold: Minimum confidence to include forecasted demand (default 0.85)
        """
        self.turso_client = get_turso_client()
        self.style_mapper = get_style_mapper()
        self.confidence_threshold = confidence_threshold

    def get_actual_orders(
        self,
        start_week: Optional[int] = None,
        weeks: int = 13
    ) -> Dict[str, Dict[int, float]]:
        """
        Get actual confirmed orders from Turso database

        Args:
            start_week: Starting ISO week number (defaults to current week)
            weeks: Number of weeks to retrieve (default 13)

        Returns:
            {style: {week_number: yards}}

        Example:
            {
                "STYLE001": {42: 1000, 43: 1200},
                "STYLE002": {42: 500}
            }
        """
        try:
            if start_week is None:
                start_week = datetime.now().isocalendar()[1]

            # Calculate week range
            end_week = start_week + weeks

            # Query knit_orders for actual confirmed orders
            # Map fStyle to Style using style_mapper
            sql = """
            SELECT
                style,
                week_number,
                SUM(quantity) as total_yards
            FROM knit_orders
            WHERE week_number >= ?
              AND week_number < ?
              AND status IN ('confirmed', 'in_production', 'completed')
            GROUP BY style, week_number
            ORDER BY style, week_number
            """

            rows = self.turso_client.execute(sql, [start_week, end_week])

            if not rows:
                logger.warning(f"No actual orders found for weeks {start_week}-{end_week-1}")
                return {}

            # Organize by style and week
            actuals = {}
            for row in rows:
                fstyle = row.get('style', '')  # This is fStyle from knit_orders
                week = row.get('week_number')
                yards = row.get('total_yards', 0)

                if not fstyle or week is None:
                    continue

                # Map fStyle to gBase (standard style identifier)
                gbase = self.style_mapper.fstyle_to_gbase.get(fstyle, fstyle)

                if gbase not in actuals:
                    actuals[gbase] = {}

                actuals[gbase][week] = yards

            logger.info(f"✓ Loaded actual orders: {len(actuals)} styles, weeks {start_week}-{end_week-1}")
            return actuals

        except Exception as e:
            logger.exception(f"Error getting actual orders: {e}")
            return {}

    def compare_forecast_vs_actual(
        self,
        blended_forecast: Dict[str, Dict[int, Dict[str, Any]]],
        actual_orders: Optional[Dict[str, Dict[int, float]]] = None,
        start_week: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare forecasted demand with actual orders

        Args:
            blended_forecast: Output from ForecastBlender.blend_forecasts()
                {style: {week: {blended_yards, confidence, sources, ...}}}
            actual_orders: Actual confirmed orders (auto-loaded if None)
            start_week: Starting ISO week (defaults to current)

        Returns:
            Comprehensive comparison report with gaps and variances

        Example:
            {
                'summary': {
                    'total_styles': 50,
                    'styles_with_orders': 35,
                    'styles_forecast_only': 15,
                    'high_confidence_gaps': 12,
                    'total_gap_yards': 15000
                },
                'gaps': [
                    {
                        'style': 'STYLE001',
                        'week': 42,
                        'forecasted_yards': 1000,
                        'confidence': 0.92,
                        'dominant_source': 'sales_team',
                        'reason': 'High confidence forecast without order'
                    }
                ],
                'variances': [
                    {
                        'style': 'STYLE002',
                        'week': 43,
                        'forecasted_yards': 800,
                        'actual_yards': 1200,
                        'variance_pct': -50.0,
                        'variance_abs': -400
                    }
                ]
            }
        """
        try:
            if start_week is None:
                start_week = datetime.now().isocalendar()[1]

            # Load actual orders if not provided
            if actual_orders is None:
                actual_orders = self.get_actual_orders(start_week=start_week)

            gaps = []
            variances = []

            styles_with_orders = set()
            styles_forecast_only = set()
            total_gap_yards = 0.0

            # Compare each forecasted style-week
            for style, week_forecasts in blended_forecast.items():
                for week, forecast_data in week_forecasts.items():
                    forecasted_yards = forecast_data.get('blended_yards', 0)
                    confidence = forecast_data.get('confidence', 0)
                    dominant_source = forecast_data.get('dominant_source', 'unknown')

                    # Get actual orders for this style-week
                    actual_yards = actual_orders.get(style, {}).get(week, 0)

                    if actual_yards > 0:
                        # We have both forecast and actual - calculate variance
                        styles_with_orders.add(style)

                        variance_abs = forecasted_yards - actual_yards
                        variance_pct = (variance_abs / actual_yards) * 100 if actual_yards > 0 else 0

                        variances.append({
                            'style': style,
                            'week': week,
                            'forecasted_yards': forecasted_yards,
                            'actual_yards': actual_yards,
                            'variance_pct': variance_pct,
                            'variance_abs': variance_abs,
                            'confidence': confidence,
                            'dominant_source': dominant_source
                        })

                    else:
                        # We have forecast but NO actual order
                        if confidence >= self.confidence_threshold:
                            # High-confidence gap - flag for proactive production
                            gaps.append({
                                'style': style,
                                'week': week,
                                'forecasted_yards': forecasted_yards,
                                'confidence': confidence,
                                'dominant_source': dominant_source,
                                'reason': 'High confidence forecast without order',
                                'source_agreement': forecast_data.get('source_agreement', 0),
                                'sources': forecast_data.get('sources', {})
                            })

                            styles_forecast_only.add(style)
                            total_gap_yards += forecasted_yards

            # Sort gaps by confidence (highest first)
            gaps.sort(key=lambda x: x['confidence'], reverse=True)

            # Sort variances by absolute variance (largest differences first)
            variances.sort(key=lambda x: abs(x['variance_pct']), reverse=True)

            # Build summary
            summary = {
                'total_styles': len(blended_forecast),
                'styles_with_orders': len(styles_with_orders),
                'styles_forecast_only': len(styles_forecast_only),
                'high_confidence_gaps': len(gaps),
                'total_gap_yards': total_gap_yards,
                'total_variances': len(variances),
                'avg_confidence': sum(g['confidence'] for g in gaps) / len(gaps) if gaps else 0
            }

            logger.info(f"✓ Comparison complete: {summary['high_confidence_gaps']} high-confidence gaps identified")

            return {
                'summary': summary,
                'gaps': gaps,
                'variances': variances,
                'comparison_date': datetime.now().isoformat(),
                'confidence_threshold': self.confidence_threshold
            }

        except Exception as e:
            logger.exception(f"Error comparing forecast vs actual: {e}")
            return {
                'summary': {'error': str(e)},
                'gaps': [],
                'variances': []
            }

    def generate_proactive_production_list(
        self,
        comparison_result: Dict[str, Any],
        max_items: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate prioritized list of items for proactive production

        Args:
            comparison_result: Output from compare_forecast_vs_actual()
            max_items: Maximum items to include (top N by confidence)

        Returns:
            Prioritized production recommendations

        Example:
            [
                {
                    'priority': 1,
                    'style': 'STYLE001',
                    'week': 42,
                    'forecasted_yards': 1000,
                    'confidence': 0.95,
                    'dominant_source': 'customer_commitment',
                    'action': 'Schedule for production',
                    'risk_level': 'low'
                }
            ]
        """
        try:
            gaps = comparison_result.get('gaps', [])

            if not gaps:
                logger.info("No high-confidence gaps found - no proactive production needed")
                return []

            # Take top N gaps by confidence
            top_gaps = gaps[:max_items]

            # Build production recommendations
            recommendations = []
            for idx, gap in enumerate(top_gaps, start=1):
                # Determine risk level based on confidence and source agreement
                confidence = gap['confidence']
                source_agreement = gap.get('source_agreement', 0)

                if confidence >= 0.95 and source_agreement >= 0.9:
                    risk_level = 'very_low'
                    action = 'Schedule for production immediately'
                elif confidence >= 0.90:
                    risk_level = 'low'
                    action = 'Schedule for production'
                elif confidence >= 0.85:
                    risk_level = 'medium'
                    action = 'Consider for production - monitor'
                else:
                    risk_level = 'high'
                    action = 'Monitor only'

                recommendations.append({
                    'priority': idx,
                    'style': gap['style'],
                    'week': gap['week'],
                    'forecasted_yards': gap['forecasted_yards'],
                    'confidence': confidence,
                    'dominant_source': gap['dominant_source'],
                    'source_agreement': source_agreement,
                    'action': action,
                    'risk_level': risk_level,
                    'sources': gap.get('sources', {})
                })

            logger.info(f"✓ Generated {len(recommendations)} proactive production recommendations")
            return recommendations

        except Exception as e:
            logger.exception(f"Error generating production list: {e}")
            return []

    def get_variance_alerts(
        self,
        comparison_result: Dict[str, Any],
        threshold_pct: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Get alerts for significant forecast vs actual variances

        Args:
            comparison_result: Output from compare_forecast_vs_actual()
            threshold_pct: Alert threshold (default 30%)

        Returns:
            List of variance alerts

        Example:
            [
                {
                    'style': 'STYLE002',
                    'week': 43,
                    'variance_pct': -50.0,
                    'forecasted_yards': 800,
                    'actual_yards': 1200,
                    'alert_type': 'under_forecast',
                    'severity': 'high'
                }
            ]
        """
        try:
            variances = comparison_result.get('variances', [])

            alerts = []
            for var in variances:
                variance_pct = var['variance_pct']

                # Check if variance exceeds threshold
                if abs(variance_pct) >= threshold_pct:
                    # Determine alert type
                    if variance_pct < 0:
                        alert_type = 'under_forecast'  # Actual > Forecast
                    else:
                        alert_type = 'over_forecast'   # Forecast > Actual

                    # Determine severity
                    if abs(variance_pct) >= 50:
                        severity = 'high'
                    elif abs(variance_pct) >= 40:
                        severity = 'medium'
                    else:
                        severity = 'low'

                    alerts.append({
                        'style': var['style'],
                        'week': var['week'],
                        'variance_pct': variance_pct,
                        'forecasted_yards': var['forecasted_yards'],
                        'actual_yards': var['actual_yards'],
                        'alert_type': alert_type,
                        'severity': severity,
                        'dominant_source': var.get('dominant_source', 'unknown')
                    })

            # Sort by severity and variance magnitude
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            alerts.sort(key=lambda x: (severity_order[x['severity']], -abs(x['variance_pct'])))

            logger.info(f"✓ Found {len(alerts)} variance alerts (>{threshold_pct}% threshold)")
            return alerts

        except Exception as e:
            logger.exception(f"Error getting variance alerts: {e}")
            return []

    def get_combined_production_schedule(
        self,
        actual_orders: Dict[str, Dict[int, float]],
        blended_forecast: Dict[str, Dict[int, Dict[str, Any]]],
        include_forecast_threshold: float = 0.85
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Create combined production schedule (actual orders + high-confidence forecasts)

        Args:
            actual_orders: Confirmed orders {style: {week: yards}}
            blended_forecast: Forecast data from ForecastBlender
            include_forecast_threshold: Minimum confidence to include forecasts

        Returns:
            Combined schedule with source tracking

        Example:
            {
                "STYLE001": {
                    42: {
                        'yards': 1000,
                        'source': 'actual_order',
                        'confidence': 1.0
                    },
                    43: {
                        'yards': 800,
                        'source': 'forecast',
                        'confidence': 0.92,
                        'forecast_source': 'sales_team'
                    }
                }
            }
        """
        try:
            combined_schedule = {}

            # First, add all actual orders
            for style, week_orders in actual_orders.items():
                if style not in combined_schedule:
                    combined_schedule[style] = {}

                for week, yards in week_orders.items():
                    combined_schedule[style][week] = {
                        'yards': yards,
                        'source': 'actual_order',
                        'confidence': 1.0
                    }

            # Then, add high-confidence forecasts where NO actual order exists
            for style, week_forecasts in blended_forecast.items():
                if style not in combined_schedule:
                    combined_schedule[style] = {}

                for week, forecast_data in week_forecasts.items():
                    # Skip if actual order already exists for this week
                    if week in combined_schedule[style]:
                        continue

                    confidence = forecast_data.get('confidence', 0)

                    # Only include if confidence meets threshold
                    if confidence >= include_forecast_threshold:
                        combined_schedule[style][week] = {
                            'yards': forecast_data.get('blended_yards', 0),
                            'source': 'forecast',
                            'confidence': confidence,
                            'forecast_source': forecast_data.get('dominant_source', 'unknown'),
                            'source_agreement': forecast_data.get('source_agreement', 0)
                        }

            logger.info(f"✓ Combined schedule: {len(combined_schedule)} styles")
            return combined_schedule

        except Exception as e:
            logger.exception(f"Error creating combined schedule: {e}")
            return {}


def main():
    """Test forecast-actual comparator"""

    comparator = ForecastActualComparator(confidence_threshold=0.85)

    # Test 1: Get actual orders
    print("=== Test 1: Get Actual Orders ===")
    actual_orders = comparator.get_actual_orders()
    print(f"Loaded {len(actual_orders)} styles with actual orders")

    if actual_orders:
        sample_style = list(actual_orders.keys())[0]
        print(f"\nSample: {sample_style}")
        for week, yards in list(actual_orders[sample_style].items())[:3]:
            print(f"  Week {week}: {yards} yards")

    # Test 2: Mock blended forecast for comparison
    print("\n=== Test 2: Compare Forecast vs Actual ===")

    # Create mock blended forecast
    mock_forecast = {
        "C1B4014": {
            42: {
                'blended_yards': 1000,
                'confidence': 0.92,
                'dominant_source': 'sales_team',
                'source_agreement': 0.88
            },
            43: {
                'blended_yards': 1200,
                'confidence': 0.78,  # Below threshold
                'dominant_source': 'ml_historical',
                'source_agreement': 0.65
            }
        }
    }

    comparison = comparator.compare_forecast_vs_actual(
        blended_forecast=mock_forecast,
        actual_orders=actual_orders
    )

    print(f"\nSummary:")
    for key, value in comparison['summary'].items():
        print(f"  {key}: {value}")

    # Test 3: Proactive production recommendations
    print("\n=== Test 3: Proactive Production Recommendations ===")

    recommendations = comparator.generate_proactive_production_list(comparison)
    print(f"Found {len(recommendations)} recommendations")

    if recommendations:
        print("\nTop 3 recommendations:")
        for rec in recommendations[:3]:
            print(f"  Priority {rec['priority']}: {rec['style']} Week {rec['week']}")
            print(f"    Forecasted: {rec['forecasted_yards']} yards")
            print(f"    Confidence: {rec['confidence']:.2%}")
            print(f"    Action: {rec['action']}")

    # Test 4: Variance alerts
    print("\n=== Test 4: Variance Alerts ===")

    alerts = comparator.get_variance_alerts(comparison, threshold_pct=30.0)
    print(f"Found {len(alerts)} variance alerts")

    if alerts:
        print("\nTop 3 alerts:")
        for alert in alerts[:3]:
            print(f"  {alert['style']} Week {alert['week']}: {alert['variance_pct']:+.1f}%")
            print(f"    Forecast: {alert['forecasted_yards']}, Actual: {alert['actual_yards']}")
            print(f"    Severity: {alert['severity']}")


if __name__ == "__main__":
    main()
