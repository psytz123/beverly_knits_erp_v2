"""BOM Style/Variant Resolver.

Handles parsing of Style/Variant format from knit orders and
maps them to the correct BOM entries for accurate yarn requirements.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StyleVariant:
    """Container for parsed style/variant information."""

    full_style: str
    base_style: str
    variant: Optional[str]
    has_variant: bool

    def __str__(self) -> str:
        """String representation."""
        if self.has_variant:
            return f"{self.base_style}/{self.variant}"
        return self.base_style


class BOMVariantResolver:
    """Resolver for BOM style/variant mappings.

    Parses Style/Variant format from knit orders and finds
    the correct BOM entries for specific yarn requirements.
    """

    def __init__(self) -> None:
        """Initialize resolver."""
        self.resolution_cache: Dict[str, StyleVariant] = {}
        self.bom_cache: Optional[pd.DataFrame] = None
        logger.info("BOMVariantResolver initialized")

    def parse_style_variant(self, style_string: str) -> StyleVariant:
        """Parse style/variant from string format.

        Args:
            style_string: Style in format "BASE/VARIANT" or just "BASE"

        Returns:
            StyleVariant with parsed components

        Examples:
            >>> resolver = BOMVariantResolver()
            >>> sv = resolver.parse_style_variant("ST123/A")
            >>> assert sv.base_style == "ST123"
            >>> assert sv.variant == "A"
        """
        # Check cache
        if style_string in self.resolution_cache:
            return self.resolution_cache[style_string]

        # Clean input
        style_string = str(style_string).strip()

        # Parse style/variant
        if '/' in style_string:
            parts = style_string.split('/', 1)
            base_style = parts[0].strip()
            variant = parts[1].strip() if len(parts) > 1 else None
            has_variant = True
        else:
            base_style = style_string
            variant = None
            has_variant = False

        # Create result
        result = StyleVariant(
            full_style=style_string,
            base_style=base_style,
            variant=variant,
            has_variant=has_variant
        )

        # Cache result
        self.resolution_cache[style_string] = result

        return result

    def resolve_bom_entries(
        self,
        style_string: str,
        bom_df: pd.DataFrame,
        style_column: str = 'Style#'
    ) -> pd.DataFrame:
        """Resolve BOM entries for a style/variant.

        Args:
            style_string: Style in format "BASE/VARIANT" or "BASE"
            bom_df: BOM DataFrame
            style_column: Name of style column in BOM

        Returns:
            DataFrame with BOM entries for this style/variant

        The resolution logic:
        1. If variant exists, look for exact "BASE/VARIANT" match
        2. If no exact match, look for base style only
        3. Return all matching BOM entries
        """
        # Parse style/variant
        style_variant = self.parse_style_variant(style_string)

        # First try exact match if variant exists
        if style_variant.has_variant:
            exact_match = bom_df[bom_df[style_column] == style_variant.full_style]
            if not exact_match.empty:
                logger.debug(f"Found exact match for {style_variant.full_style}")
                return exact_match

            # Try variant as separate column if exists
            if 'Variant' in bom_df.columns:
                variant_match = bom_df[
                    (bom_df[style_column] == style_variant.base_style) &
                    (bom_df['Variant'] == style_variant.variant)
                ]
                if not variant_match.empty:
                    logger.debug(f"Found variant match for {style_variant}")
                    return variant_match

        # Fall back to base style
        base_match = bom_df[bom_df[style_column] == style_variant.base_style]
        if not base_match.empty:
            logger.debug(f"Using base style match for {style_variant.base_style}")
            return base_match

        # No match found
        logger.warning(f"No BOM entries found for {style_string}")
        return pd.DataFrame()

    def get_yarn_requirements(
        self,
        style_string: str,
        bom_df: pd.DataFrame,
        quantity: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Get yarn requirements for a style/variant.

        Args:
            style_string: Style in format "BASE/VARIANT" or "BASE"
            bom_df: BOM DataFrame
            quantity: Production quantity multiplier

        Returns:
            List of yarn requirements with quantities
        """
        # Get BOM entries
        bom_entries = self.resolve_bom_entries(style_string, bom_df)

        if bom_entries.empty:
            return []

        # Build requirements list
        requirements = []
        for _, row in bom_entries.iterrows():
            yarn_id = row.get('YarnID') or row.get('Desc#')
            usage = row.get('Usage') or row.get('Qty') or 0

            requirements.append({
                'style': style_string,
                'yarn_id': yarn_id,
                'usage_per_unit': float(usage),
                'total_required': float(usage) * quantity,
                'uom': row.get('UOM', 'LBS'),
                'description': row.get('Description', '')
            })

        return requirements

    def validate_bom_coverage(
        self,
        orders_df: pd.DataFrame,
        bom_df: pd.DataFrame,
        style_column: str = 'Style#'
    ) -> Dict[str, Any]:
        """Validate BOM coverage for all orders.

        Args:
            orders_df: Orders DataFrame with styles
            bom_df: BOM DataFrame
            style_column: Name of style column

        Returns:
            Coverage analysis results
        """
        if style_column not in orders_df.columns:
            raise ValueError(f"Column {style_column} not found in orders")

        unique_styles = orders_df[style_column].unique()
        covered = []
        missing = []
        partial = []

        for style in unique_styles:
            if pd.isna(style):
                continue

            entries = self.resolve_bom_entries(str(style), bom_df)

            if entries.empty:
                missing.append(style)
            elif len(entries) < 2:  # Assuming most styles need multiple yarns
                partial.append(style)
            else:
                covered.append(style)

        total = len(unique_styles)
        coverage_rate = (len(covered) / total * 100) if total > 0 else 0

        return {
            'total_styles': total,
            'covered': len(covered),
            'missing': len(missing),
            'partial': len(partial),
            'coverage_rate': coverage_rate,
            'missing_styles': missing[:10],  # First 10 missing
            'partial_styles': partial[:10]   # First 10 partial
        }

    def merge_orders_with_bom(
        self,
        orders_df: pd.DataFrame,
        bom_df: pd.DataFrame,
        quantity_column: str = 'Qty'
    ) -> pd.DataFrame:
        """Merge orders with BOM to get yarn requirements.

        Args:
            orders_df: Orders DataFrame
            bom_df: BOM DataFrame
            quantity_column: Name of quantity column in orders

        Returns:
            DataFrame with orders and their yarn requirements
        """
        results = []

        for _, order in orders_df.iterrows():
            style = order.get('Style#')
            quantity = order.get(quantity_column, 0)

            if pd.isna(style):
                continue

            # Get yarn requirements
            requirements = self.get_yarn_requirements(
                str(style), bom_df, float(quantity)
            )

            # Add to results
            for req in requirements:
                results.append({
                    'Order#': order.get('Order#'),
                    'Style': style,
                    'Order_Qty': quantity,
                    'YarnID': req['yarn_id'],
                    'Usage_Per_Unit': req['usage_per_unit'],
                    'Total_Yarn_Required': req['total_required'],
                    'UOM': req['uom']
                })

        return pd.DataFrame(results)

    def get_resolution_summary(self) -> Dict[str, Any]:
        """Get summary of style/variant resolutions.

        Returns:
            Summary statistics
        """
        total = len(self.resolution_cache)
        with_variant = sum(1 for sv in self.resolution_cache.values()
                          if sv.has_variant)
        without_variant = total - with_variant

        return {
            'total_resolved': total,
            'with_variant': with_variant,
            'without_variant': without_variant,
            'variant_rate': (with_variant / total * 100) if total > 0 else 0
        }


if __name__ == "__main__":
    """Validation with sample data."""

    resolver = BOMVariantResolver()

    # Test Case 1: Style with variant
    sv1 = resolver.parse_style_variant("ST123/A")
    assert sv1.base_style == "ST123"
    assert sv1.variant == "A"
    assert sv1.has_variant is True
    logger.info(f"Test 1 passed: {sv1}")

    # Test Case 2: Style without variant
    sv2 = resolver.parse_style_variant("ST456")
    assert sv2.base_style == "ST456"
    assert sv2.variant is None
    assert sv2.has_variant is False
    logger.info(f"Test 2 passed: {sv2}")

    # Test Case 3: Complex variant
    sv3 = resolver.parse_style_variant("STYLE-789/VAR-B")
    assert sv3.base_style == "STYLE-789"
    assert sv3.variant == "VAR-B"
    logger.info(f"Test 3 passed: {sv3}")

    # Test Case 4: BOM resolution with sample data
    sample_bom = pd.DataFrame([
        {'Style#': 'ST123', 'YarnID': 'Y001', 'Usage': 2.5, 'UOM': 'LBS'},
        {'Style#': 'ST123', 'YarnID': 'Y002', 'Usage': 1.0, 'UOM': 'LBS'},
        {'Style#': 'ST123/A', 'YarnID': 'Y003', 'Usage': 0.5, 'UOM': 'LBS'},
        {'Style#': 'ST456', 'YarnID': 'Y001', 'Usage': 3.0, 'UOM': 'LBS'}
    ])

    # Test exact variant match
    entries1 = resolver.resolve_bom_entries("ST123/A", sample_bom)
    assert len(entries1) == 1
    assert entries1.iloc[0]['YarnID'] == 'Y003'
    logger.info("Test 4a passed: Exact variant match")

    # Test fallback to base style
    entries2 = resolver.resolve_bom_entries("ST123/B", sample_bom)
    assert len(entries2) == 2  # Falls back to base ST123
    logger.info("Test 4b passed: Fallback to base style")

    # Test yarn requirements calculation
    reqs = resolver.get_yarn_requirements("ST456", sample_bom, quantity=10.0)
    assert len(reqs) == 1
    assert reqs[0]['total_required'] == 30.0  # 3.0 * 10
    logger.info("Test 5 passed: Yarn requirements calculation")

    print("All validations passed!")