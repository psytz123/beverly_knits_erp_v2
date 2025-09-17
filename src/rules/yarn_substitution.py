"""Yarn Substitution Engine for Beverly Knits ERP.

Identifies and ranks compatible yarn substitutions when shortages occur,
displaying only relevant inventory information.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YarnSubstitute:
    """Container for yarn substitute information."""

    original_yarn: str
    substitute_yarn: str
    compatibility_score: float
    on_hand: float
    on_order: float
    allocated: float
    available: float
    reason: str

    def can_fulfill(self, required_qty: float) -> bool:
        """Check if substitute can fulfill requirement."""
        return self.available >= required_qty


class YarnSubstitutionEngine:
    """Engine for finding compatible yarn substitutions.

    When shortages are detected, this engine identifies compatible
    alternatives and displays their inventory information.
    """

    # Compatibility rules
    COMPATIBILITY_RULES = {
        'exact_match': 1.0,      # Same yarn different supplier
        'same_weight': 0.9,      # Same weight/denier
        'same_color': 0.8,       # Same color family
        'same_material': 0.7,    # Same material composition
        'similar_weight': 0.5,   # Within 10% weight
        'similar_color': 0.4,    # Similar color
        'different': 0.0        # Not compatible
    }

    def __init__(self) -> None:
        """Initialize substitution engine."""
        self.substitution_history: List[Dict[str, Any]] = []
        self.compatibility_matrix: Optional[pd.DataFrame] = None
        logger.info("YarnSubstitutionEngine initialized")

    def find_substitutes(
        self,
        yarn_id: str,
        required_qty: float,
        yarn_inventory: pd.DataFrame,
        compatibility_data: Optional[pd.DataFrame] = None
    ) -> List[YarnSubstitute]:
        """Find compatible yarn substitutes.

        Args:
            yarn_id: Yarn with shortage
            required_qty: Quantity needed
            yarn_inventory: Current yarn inventory
            compatibility_data: Optional compatibility matrix

        Returns:
            List of compatible substitutes ranked by score
        """
        substitutes = []

        # Get original yarn properties
        original = yarn_inventory[yarn_inventory['YarnID'] == yarn_id]
        if original.empty:
            logger.warning(f"Original yarn {yarn_id} not found")
            return []

        original_row = original.iloc[0]

        # Check all other yarns
        for _, yarn in yarn_inventory.iterrows():
            if yarn['YarnID'] == yarn_id:
                continue  # Skip original

            # Calculate compatibility
            score, reason = self._calculate_compatibility(
                original_row, yarn, compatibility_data
            )

            if score > 0:
                # Calculate availability
                available = yarn['On Hand'] + yarn['On Order'] - yarn['Allocated']

                if available > 0:
                    substitute = YarnSubstitute(
                        original_yarn=yarn_id,
                        substitute_yarn=yarn['YarnID'],
                        compatibility_score=score,
                        on_hand=yarn['On Hand'],
                        on_order=yarn['On Order'],
                        allocated=yarn['Allocated'],
                        available=available,
                        reason=reason
                    )
                    substitutes.append(substitute)

        # Sort by score and availability
        substitutes.sort(
            key=lambda x: (x.compatibility_score, x.available),
            reverse=True
        )

        # Log substitution search
        self.substitution_history.append({
            'yarn_id': yarn_id,
            'required_qty': required_qty,
            'substitutes_found': len(substitutes),
            'best_match': substitutes[0].substitute_yarn if substitutes else None
        })

        return substitutes

    def _calculate_compatibility(
        self,
        original: pd.Series,
        candidate: pd.Series,
        compatibility_data: Optional[pd.DataFrame]
    ) -> tuple[float, str]:
        """Calculate compatibility score between yarns.

        Args:
            original: Original yarn properties
            candidate: Candidate substitute properties
            compatibility_data: Optional compatibility matrix

        Returns:
            Tuple of (score, reason)
        """
        # Check explicit compatibility matrix first
        if compatibility_data is not None:
            score = self._check_compatibility_matrix(
                original['YarnID'],
                candidate['YarnID'],
                compatibility_data
            )
            if score is not None:
                return score, "Predefined compatibility"

        # Rule-based compatibility
        scores = []
        reasons = []

        # Check weight/denier
        if 'Weight' in original.index and 'Weight' in candidate.index:
            if original['Weight'] == candidate['Weight']:
                scores.append(self.COMPATIBILITY_RULES['same_weight'])
                reasons.append("Same weight")
            elif abs(original['Weight'] - candidate['Weight']) / original['Weight'] < 0.1:
                scores.append(self.COMPATIBILITY_RULES['similar_weight'])
                reasons.append("Similar weight (±10%)")

        # Check color
        if 'Color' in original.index and 'Color' in candidate.index:
            if original['Color'] == candidate['Color']:
                scores.append(self.COMPATIBILITY_RULES['same_color'])
                reasons.append("Same color")
            elif self._similar_color(original['Color'], candidate['Color']):
                scores.append(self.COMPATIBILITY_RULES['similar_color'])
                reasons.append("Similar color")

        # Check material
        if 'Material' in original.index and 'Material' in candidate.index:
            if original['Material'] == candidate['Material']:
                scores.append(self.COMPATIBILITY_RULES['same_material'])
                reasons.append("Same material")

        # Return best score
        if scores:
            best_idx = np.argmax(scores)
            return scores[best_idx], reasons[best_idx]

        return 0.0, "Not compatible"

    def _check_compatibility_matrix(
        self,
        yarn1: str,
        yarn2: str,
        compatibility_data: pd.DataFrame
    ) -> Optional[float]:
        """Check explicit compatibility matrix.

        Args:
            yarn1: First yarn ID
            yarn2: Second yarn ID
            compatibility_data: Compatibility matrix

        Returns:
            Compatibility score or None
        """
        if yarn1 in compatibility_data.index and yarn2 in compatibility_data.columns:
            return compatibility_data.loc[yarn1, yarn2]
        if yarn2 in compatibility_data.index and yarn1 in compatibility_data.columns:
            return compatibility_data.loc[yarn2, yarn1]
        return None

    def _similar_color(self, color1: str, color2: str) -> bool:
        """Check if colors are similar.

        Args:
            color1: First color
            color2: Second color

        Returns:
            True if colors are similar
        """
        # Simple color family matching
        color_families = {
            'red': ['red', 'crimson', 'scarlet', 'burgundy'],
            'blue': ['blue', 'navy', 'azure', 'cobalt'],
            'green': ['green', 'olive', 'emerald', 'lime'],
            'black': ['black', 'charcoal', 'ebony'],
            'white': ['white', 'ivory', 'cream'],
            'gray': ['gray', 'grey', 'silver', 'ash']
        }

        color1_lower = str(color1).lower()
        color2_lower = str(color2).lower()

        for family, colors in color_families.items():
            if color1_lower in colors and color2_lower in colors:
                return True

        return False

    def generate_substitution_report(
        self,
        shortages: List[Dict[str, Any]],
        yarn_inventory: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate comprehensive substitution report.

        Args:
            shortages: List of yarn shortages
            yarn_inventory: Current inventory

        Returns:
            DataFrame with substitution recommendations
        """
        report_data = []

        for shortage in shortages:
            yarn_id = shortage['yarn_id']
            required = shortage['required']

            # Find substitutes
            substitutes = self.find_substitutes(
                yarn_id, required, yarn_inventory
            )

            # Add to report
            if substitutes:
                best = substitutes[0]
                report_data.append({
                    'Original_Yarn': yarn_id,
                    'Required_Qty': required,
                    'Best_Substitute': best.substitute_yarn,
                    'Compatibility': best.compatibility_score,
                    'Available': best.available,
                    'Can_Fulfill': best.can_fulfill(required),
                    'Reason': best.reason,
                    'Alternative_Count': len(substitutes)
                })
            else:
                report_data.append({
                    'Original_Yarn': yarn_id,
                    'Required_Qty': required,
                    'Best_Substitute': 'None',
                    'Compatibility': 0.0,
                    'Available': 0.0,
                    'Can_Fulfill': False,
                    'Reason': 'No substitutes found',
                    'Alternative_Count': 0
                })

        return pd.DataFrame(report_data)

    def get_substitution_summary(self) -> Dict[str, Any]:
        """Get summary of substitution searches.

        Returns:
            Summary statistics
        """
        if not self.substitution_history:
            return {'total_searches': 0}

        total = len(self.substitution_history)
        with_substitutes = sum(
            1 for h in self.substitution_history
            if h['substitutes_found'] > 0
        )

        return {
            'total_searches': total,
            'with_substitutes': with_substitutes,
            'without_substitutes': total - with_substitutes,
            'success_rate': (with_substitutes / total * 100) if total > 0 else 0,
            'avg_substitutes': sum(
                h['substitutes_found'] for h in self.substitution_history
            ) / total if total > 0 else 0
        }


if __name__ == "__main__":
    """Validation of yarn substitution engine."""

    engine = YarnSubstitutionEngine()

    # Create sample inventory
    inventory = pd.DataFrame([
        {'YarnID': 'Y001', 'On Hand': 0, 'On Order': 0, 'Allocated': 100,
         'Weight': 2.5, 'Color': 'Blue', 'Material': 'Cotton'},
        {'YarnID': 'Y002', 'On Hand': 500, 'On Order': 100, 'Allocated': 200,
         'Weight': 2.5, 'Color': 'Navy', 'Material': 'Cotton'},
        {'YarnID': 'Y003', 'On Hand': 300, 'On Order': 0, 'Allocated': 100,
         'Weight': 2.6, 'Color': 'Blue', 'Material': 'Cotton'},
        {'YarnID': 'Y004', 'On Hand': 1000, 'On Order': 0, 'Allocated': 500,
         'Weight': 3.0, 'Color': 'Red', 'Material': 'Polyester'}
    ])

    # Test Case 1: Find substitutes for Y001 (shortage)
    substitutes = engine.find_substitutes('Y001', 100, inventory)
    assert len(substitutes) > 0
    assert substitutes[0].substitute_yarn == 'Y002'  # Same weight, similar color
    logger.info(f"Test 1 passed: Found {len(substitutes)} substitutes")

    # Test Case 2: Check best substitute properties
    best = substitutes[0]
    assert best.compatibility_score > 0.5
    assert best.available > 0
    logger.info(f"Test 2 passed: Best substitute = {best.substitute_yarn} (score={best.compatibility_score})")

    # Test Case 3: Generate report
    shortages = [
        {'yarn_id': 'Y001', 'required': 100}
    ]
    report = engine.generate_substitution_report(shortages, inventory)
    assert len(report) == 1
    assert report.iloc[0]['Can_Fulfill'] is True
    logger.info("Test 3 passed: Report generated")

    # Test Case 4: Get summary
    summary = engine.get_substitution_summary()
    assert summary['total_searches'] == 1
    assert summary['with_substitutes'] == 1
    logger.info(f"Test 4 passed: Summary = {summary}")

    print("All validations passed!")