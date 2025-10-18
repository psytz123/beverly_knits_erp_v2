#!/usr/bin/env python3
"""
Forecast Blender
Intelligently blends multiple forecast sources using weighted ensemble
Combines ML historical forecasts with external sales team and customer forecasts
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class BlendingStrategy(Enum):
    """Available forecast blending strategies"""
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class ForecastBlender:
    """
    Intelligently blend multiple forecast sources into single master forecast
    Supports configurable weights and multiple blending strategies
    """

    def __init__(self):
        """Initialize forecast blender with default weights"""
        # Default source weights (tunable based on historical accuracy)
        self.source_weights = {
            'ml_historical': 0.40,      # ML from historical sales
            'sales_team': 0.35,          # Sales team forecast
            'customer_commitment': 0.20, # Customer forward orders
            'market_intelligence': 0.05  # Industry data
        }

        # Historical accuracy tracking (for adaptive weighting)
        self.source_accuracy_history: Dict[str, List[float]] = {}

    def blend_forecasts(
        self,
        ml_forecast: Dict[str, Dict[int, float]],
        external_forecasts: List[Dict[str, Dict[int, Dict[str, Any]]]],
        blending_strategy: str = 'weighted_average',
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Combine multiple forecast sources into single master forecast

        Args:
            ml_forecast: {style: {week: yards}} - ML historical forecast
            external_forecasts: List of external forecast dicts with metadata
            blending_strategy: Strategy to use for blending
            custom_weights: Optional custom weights (overrides defaults)

        Returns:
            Master forecast with blended values and metadata:
            {
                style: {
                    week: {
                        'blended_yards': float,
                        'confidence': float,
                        'sources': {source_name: {yards, weight}},
                        'source_agreement': float,
                        'dominant_source': str
                    }
                }
            }
        """
        try:
            # Use custom weights if provided
            weights = custom_weights if custom_weights else self.source_weights.copy()

            # Normalize weights to sum to 1.0
            weights = self._normalize_weights(weights)

            # Convert ML forecast to standard format with metadata
            ml_forecast_standard = self._standardize_ml_forecast(ml_forecast)

            # Combine all forecasts
            all_forecasts = [ml_forecast_standard] + external_forecasts

            # Get all unique style-week combinations
            style_week_combinations = self._get_all_combinations(all_forecasts)

            # Blend each style-week combination
            master_forecast: Dict[str, Dict[int, Dict[str, Any]]] = {}

            for style, week in style_week_combinations:
                # Collect all source forecasts for this style-week
                source_forecasts = self._collect_source_forecasts(
                    style, week, all_forecasts
                )

                if not source_forecasts:
                    continue

                # Apply blending strategy
                blended_data = self._apply_blending_strategy(
                    source_forecasts,
                    weights,
                    blending_strategy
                )

                # Store in master forecast
                if style not in master_forecast:
                    master_forecast[style] = {}

                master_forecast[style][week] = blended_data

            logger.info(
                f"Blended forecasts for {len(master_forecast)} styles using "
                f"{blending_strategy} strategy"
            )

            return master_forecast

        except Exception as e:
            logger.exception(f"Error blending forecasts: {e}")
            return {}

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0

        Args:
            weights: Dict of source weights

        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        if total == 0:
            logger.warning("Total weight is zero, using equal weights")
            return {k: 1.0/len(weights) for k in weights}

        return {k: v/total for k, v in weights.items()}

    def _standardize_ml_forecast(
        self,
        ml_forecast: Dict[str, Dict[int, float]]
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Convert ML forecast to standard format with metadata

        Args:
            ml_forecast: {style: {week: yards}}

        Returns:
            Standardized format with metadata
        """
        standardized = {}

        for style, weeks in ml_forecast.items():
            if style not in standardized:
                standardized[style] = {}

            for week, yards in weeks.items():
                standardized[style][week] = {
                    'yards': float(yards),
                    'confidence': 0.85,  # Default ML confidence
                    'source': 'ml_historical',
                    'notes': 'ML ensemble forecast'
                }

        return standardized

    def _get_all_combinations(
        self,
        forecasts: List[Dict[str, Dict[int, Dict[str, Any]]]]
    ) -> List[Tuple[str, int]]:
        """
        Get all unique style-week combinations across all forecasts

        Args:
            forecasts: List of forecast dicts

        Returns:
            List of (style, week) tuples
        """
        combinations = set()

        for forecast in forecasts:
            for style, weeks in forecast.items():
                for week in weeks.keys():
                    combinations.add((style, week))

        return list(combinations)

    def _collect_source_forecasts(
        self,
        style: str,
        week: int,
        forecasts: List[Dict[str, Dict[int, Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Collect all source forecasts for a specific style-week

        Args:
            style: Style code
            week: Week number
            forecasts: All forecast dicts

        Returns:
            List of source forecast data
        """
        source_forecasts = []

        for forecast in forecasts:
            if style in forecast and week in forecast[style]:
                source_data = forecast[style][week].copy()
                source_forecasts.append(source_data)

        return source_forecasts

    def _apply_blending_strategy(
        self,
        source_forecasts: List[Dict[str, Any]],
        weights: Dict[str, float],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Apply blending strategy to combine source forecasts

        Args:
            source_forecasts: List of source forecast data
            weights: Source weights
            strategy: Blending strategy name

        Returns:
            Blended forecast data
        """
        if strategy == BlendingStrategy.WEIGHTED_AVERAGE.value:
            return self._weighted_average_blend(source_forecasts, weights)
        elif strategy == BlendingStrategy.HIGHEST_CONFIDENCE.value:
            return self._highest_confidence_blend(source_forecasts)
        elif strategy == BlendingStrategy.CONSERVATIVE.value:
            return self._conservative_blend(source_forecasts)
        elif strategy == BlendingStrategy.AGGRESSIVE.value:
            return self._aggressive_blend(source_forecasts)
        elif strategy == BlendingStrategy.ADAPTIVE.value:
            return self._adaptive_blend(source_forecasts, weights)
        else:
            logger.warning(f"Unknown strategy {strategy}, using weighted_average")
            return self._weighted_average_blend(source_forecasts, weights)

    def _weighted_average_blend(
        self,
        source_forecasts: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Blend using weighted average based on configured weights

        Args:
            source_forecasts: Source forecast data
            weights: Source weights

        Returns:
            Blended forecast
        """
        total_weighted_yards = 0.0
        total_weight = 0.0
        sources_used = {}
        confidences = []

        for source in source_forecasts:
            source_name = source.get('source', 'unknown')
            yards = source.get('yards', 0)
            confidence = source.get('confidence', 0.5)

            # Get weight for this source
            weight = weights.get(source_name, 0.1)  # Default 0.1 for unknown sources

            total_weighted_yards += yards * weight
            total_weight += weight
            confidences.append(confidence)

            sources_used[source_name] = {
                'yards': yards,
                'weight': weight,
                'confidence': confidence
            }

        # Calculate blended values
        blended_yards = total_weighted_yards / total_weight if total_weight > 0 else 0
        blended_confidence = self.calculate_blended_confidence(source_forecasts)
        source_agreement = self._calculate_source_agreement(source_forecasts)
        dominant_source = max(sources_used.items(), key=lambda x: x[1]['weight'])[0] if sources_used else 'unknown'

        return {
            'blended_yards': blended_yards,
            'confidence': blended_confidence,
            'sources': sources_used,
            'source_agreement': source_agreement,
            'dominant_source': dominant_source,
            'blending_strategy': 'weighted_average'
        }

    def _highest_confidence_blend(
        self,
        source_forecasts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use forecast from source with highest confidence

        Args:
            source_forecasts: Source forecast data

        Returns:
            Blended forecast (taking highest confidence source)
        """
        if not source_forecasts:
            return self._empty_blend()

        # Find source with highest confidence
        best_source = max(source_forecasts, key=lambda x: x.get('confidence', 0))

        sources_used = {
            best_source.get('source', 'unknown'): {
                'yards': best_source.get('yards', 0),
                'weight': 1.0,
                'confidence': best_source.get('confidence', 0)
            }
        }

        return {
            'blended_yards': best_source.get('yards', 0),
            'confidence': best_source.get('confidence', 0),
            'sources': sources_used,
            'source_agreement': self._calculate_source_agreement(source_forecasts),
            'dominant_source': best_source.get('source', 'unknown'),
            'blending_strategy': 'highest_confidence'
        }

    def _conservative_blend(
        self,
        source_forecasts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Take minimum forecast (lower inventory risk)

        Args:
            source_forecasts: Source forecast data

        Returns:
            Conservative blended forecast
        """
        if not source_forecasts:
            return self._empty_blend()

        min_source = min(source_forecasts, key=lambda x: x.get('yards', 0))

        sources_used = {s.get('source', 'unknown'): {
            'yards': s.get('yards', 0),
            'weight': 1.0 if s == min_source else 0.0,
            'confidence': s.get('confidence', 0)
        } for s in source_forecasts}

        return {
            'blended_yards': min_source.get('yards', 0),
            'confidence': min_source.get('confidence', 0) * 0.9,  # Reduce confidence for conservative
            'sources': sources_used,
            'source_agreement': self._calculate_source_agreement(source_forecasts),
            'dominant_source': min_source.get('source', 'unknown'),
            'blending_strategy': 'conservative'
        }

    def _aggressive_blend(
        self,
        source_forecasts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Take maximum forecast (higher service level)

        Args:
            source_forecasts: Source forecast data

        Returns:
            Aggressive blended forecast
        """
        if not source_forecasts:
            return self._empty_blend()

        max_source = max(source_forecasts, key=lambda x: x.get('yards', 0))

        sources_used = {s.get('source', 'unknown'): {
            'yards': s.get('yards', 0),
            'weight': 1.0 if s == max_source else 0.0,
            'confidence': s.get('confidence', 0)
        } for s in source_forecasts}

        return {
            'blended_yards': max_source.get('yards', 0),
            'confidence': max_source.get('confidence', 0) * 0.9,  # Reduce confidence for aggressive
            'sources': sources_used,
            'source_agreement': self._calculate_source_agreement(source_forecasts),
            'dominant_source': max_source.get('source', 'unknown'),
            'blending_strategy': 'aggressive'
        }

    def _adaptive_blend(
        self,
        source_forecasts: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Adjust weights based on recent accuracy (requires accuracy history)

        Args:
            source_forecasts: Source forecast data
            weights: Base weights

        Returns:
            Adaptively blended forecast
        """
        # For now, use weighted average with confidence adjustment
        # TODO: Implement true adaptive weighting based on accuracy tracker
        adjusted_weights = weights.copy()

        # Adjust weights by confidence
        for source in source_forecasts:
            source_name = source.get('source', 'unknown')
            confidence = source.get('confidence', 0.5)
            if source_name in adjusted_weights:
                adjusted_weights[source_name] *= confidence

        return self._weighted_average_blend(source_forecasts, adjusted_weights)

    def calculate_blended_confidence(
        self,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate combined confidence from multiple sources

        Logic:
        - Agreement increases confidence (all sources similar)
        - Disagreement decreases confidence (sources diverge)
        - More sources = higher confidence (if aligned)

        Args:
            sources: List of source forecast data

        Returns:
            Blended confidence score (0-1)
        """
        if not sources:
            return 0.0

        if len(sources) == 1:
            return sources[0].get('confidence', 0.5)

        # Base confidence: average of source confidences
        avg_confidence = np.mean([s.get('confidence', 0.5) for s in sources])

        # Agreement factor: how similar are the forecasts
        yards_values = [s.get('yards', 0) for s in sources]
        if len(yards_values) > 1 and np.mean(yards_values) > 0:
            cv = np.std(yards_values) / np.mean(yards_values)  # Coefficient of variation
            agreement_factor = max(0, 1 - cv)  # Lower CV = higher agreement
        else:
            agreement_factor = 0.5

        # Source count bonus: more aligned sources = higher confidence
        source_count_bonus = min(0.1, (len(sources) - 1) * 0.03)

        # Combined confidence
        blended_confidence = avg_confidence * 0.7 + agreement_factor * 0.2 + source_count_bonus

        return min(1.0, max(0.0, blended_confidence))

    def detect_source_conflicts(
        self,
        forecasts: Dict[str, Dict[int, Dict[str, Any]]],
        variance_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Flag when sources significantly disagree

        Args:
            forecasts: Master blended forecast
            variance_threshold: Threshold for flagging conflicts (default 30%)

        Returns:
            List of conflicts with details
        """
        conflicts = []

        for style, weeks in forecasts.items():
            for week, data in weeks.items():
                sources = data.get('sources', {})

                if len(sources) < 2:
                    continue

                # Calculate variance across sources
                yards_values = [s['yards'] for s in sources.values()]
                mean_yards = np.mean(yards_values)

                if mean_yards > 0:
                    std_yards = np.std(yards_values)
                    variance_pct = std_yards / mean_yards

                    if variance_pct > variance_threshold:
                        conflict = {
                            'style': style,
                            'week': week,
                            'variance_pct': variance_pct * 100,
                            'mean_yards': mean_yards,
                            'std_yards': std_yards,
                            'sources': {
                                name: data['yards']
                                for name, data in sources.items()
                            },
                            'severity': 'high' if variance_pct > 0.5 else 'medium'
                        }
                        conflicts.append(conflict)

        logger.info(f"Detected {len(conflicts)} source conflicts")
        return conflicts

    def apply_business_rules(
        self,
        blended_forecast: Dict[str, Dict[int, Dict[str, Any]]],
        rules: Dict[str, Any]
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Apply business logic overrides to blended forecast

        Rules:
        - Customer commitments override ML (when conf > threshold)
        - Sales team can veto ML forecast
        - Minimum/maximum order quantities
        - Strategic customer priorities

        Args:
            blended_forecast: Master blended forecast
            rules: Business rules configuration

        Returns:
            Forecast with business rules applied
        """
        adjusted_forecast = blended_forecast.copy()

        # Rule 1: Customer commitments override when high confidence
        customer_override_threshold = rules.get('customer_override_confidence', 0.90)

        for style, weeks in adjusted_forecast.items():
            for week, data in weeks.items():
                sources = data.get('sources', {})

                # Check for customer commitment source
                for source_name, source_data in sources.items():
                    if 'customer' in source_name.lower():
                        if source_data['confidence'] >= customer_override_threshold:
                            # Override with customer commitment
                            data['blended_yards'] = source_data['yards']
                            data['confidence'] = source_data['confidence']
                            data['dominant_source'] = source_name
                            data['override_reason'] = 'customer_commitment_high_confidence'

        # Rule 2: Apply min/max constraints
        min_order_qty = rules.get('min_order_quantity', 0)
        max_order_qty = rules.get('max_order_quantity', float('inf'))

        for style, weeks in adjusted_forecast.items():
            for week, data in weeks.items():
                blended_yards = data['blended_yards']

                if blended_yards < min_order_qty:
                    data['blended_yards'] = 0  # Below minimum, don't order
                    data['adjustment'] = 'below_minimum'
                elif blended_yards > max_order_qty:
                    data['blended_yards'] = max_order_qty
                    data['adjustment'] = 'capped_at_maximum'

        return adjusted_forecast

    def _calculate_source_agreement(
        self,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how much sources agree (0-1)

        Args:
            sources: List of source forecast data

        Returns:
            Agreement score (1.0 = perfect agreement, 0.0 = high disagreement)
        """
        if len(sources) <= 1:
            return 1.0

        yards_values = [s.get('yards', 0) for s in sources]
        mean_yards = np.mean(yards_values)

        if mean_yards == 0:
            return 1.0

        # Calculate coefficient of variation
        cv = np.std(yards_values) / mean_yards

        # Convert to agreement score (inverse of CV, capped at 1.0)
        agreement = max(0.0, 1.0 - cv)

        return agreement

    def _empty_blend(self) -> Dict[str, Any]:
        """Return empty blend structure"""
        return {
            'blended_yards': 0.0,
            'confidence': 0.0,
            'sources': {},
            'source_agreement': 0.0,
            'dominant_source': 'none',
            'blending_strategy': 'empty'
        }


def main():
    """Test forecast blender"""

    blender = ForecastBlender()

    # Test 1: Simple weighted average
    print("=== Test 1: Weighted Average Blending ===")

    ml_forecast = {
        "STYLE001": {42: 1000, 43: 1050}
    }

    external_forecasts = [
        {
            "STYLE001": {
                42: {'yards': 1100, 'confidence': 0.90, 'source': 'sales_team', 'notes': 'Q4 pipeline'},
                43: {'yards': 1150, 'confidence': 0.88, 'source': 'sales_team', 'notes': 'Growing demand'}
            }
        },
        {
            "STYLE001": {
                42: {'yards': 1050, 'confidence': 0.95, 'source': 'customer_commitment', 'notes': 'Confirmed PO'},
            }
        }
    ]

    blended = blender.blend_forecasts(ml_forecast, external_forecasts, 'weighted_average')

    for style, weeks in blended.items():
        print(f"\n{style}:")
        for week, data in sorted(weeks.items()):
            print(f"  Week {week}: {data['blended_yards']:.2f} yards (conf: {data['confidence']:.2f})")
            print(f"    Agreement: {data['source_agreement']:.2f}, Dominant: {data['dominant_source']}")
            for source, source_data in data['sources'].items():
                print(f"      {source}: {source_data['yards']:.0f} yards (weight: {source_data['weight']:.2f})")

    # Test 2: Conflict detection
    print("\n=== Test 2: Conflict Detection ===")
    conflicts = blender.detect_source_conflicts(blended, variance_threshold=0.2)
    if conflicts:
        for conflict in conflicts:
            print(f"Conflict: {conflict['style']} week {conflict['week']}")
            print(f"  Variance: {conflict['variance_pct']:.1f}%")
            print(f"  Sources: {conflict['sources']}")
    else:
        print("No significant conflicts detected")

    # Test 3: Different strategies
    print("\n=== Test 3: Different Blending Strategies ===")
    strategies = ['weighted_average', 'highest_confidence', 'conservative', 'aggressive']

    for strategy in strategies:
        result = blender.blend_forecasts(ml_forecast, external_forecasts, strategy)
        week_42_data = result['STYLE001'][42]
        print(f"{strategy}: {week_42_data['blended_yards']:.2f} yards (conf: {week_42_data['confidence']:.2f})")


if __name__ == "__main__":
    main()
