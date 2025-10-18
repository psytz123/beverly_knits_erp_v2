#!/usr/bin/env python3
"""
Turso BOM Explosion Module
Converts sales forecasts to yarn requirements using BOM data from Turso database
Implements multi-level explosion: Sales (yards) → Fabric (lbs) → Yarn (lbs)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from src.database.turso_client import get_turso_client

logger = logging.getLogger(__name__)


class TursoBOMExplosion:
    """
    Explodes sales forecasts through BOM to calculate yarn requirements
    Uses Turso database for BOM and fabric specifications
    """

    def __init__(self):
        """Initialize BOM explosion engine with Turso client"""
        self.turso_client = get_turso_client()
        self._bom_cache: Dict[str, Dict[str, float]] = {}
        self._fabric_specs_cache: Dict[str, Dict[str, Any]] = {}

    def explode_sales_to_yarn_weekly(
        self,
        sales_forecast: Dict[str, Dict[int, float]],
        weeks_to_explode: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Explode weekly sales forecast to weekly yarn requirements

        Args:
            sales_forecast: {style_id: {week_num: yards}} - Sales by style and week
            weeks_to_explode: Optional list of week numbers to process (defaults to all)

        Returns:
            {yarn_id: {week_num: lbs}} - Yarn requirements by week

        Example:
            sales_forecast = {
                "STYLE001": {42: 1000, 43: 1500},  # yards
                "STYLE002": {42: 500, 43: 800}
            }

            Returns:
            {
                "18884": {42: 250.5, 43: 375.2},  # lbs
                "18885": {42: 125.3, 43: 187.6}
            }
        """
        yarn_requirements: Dict[str, Dict[int, float]] = {}

        try:
            # Pre-load BOM and fabric specs for all styles
            styles = list(sales_forecast.keys())
            self._preload_bom_and_specs(styles)

            # Process each style's weekly demand
            for style, weekly_sales in sales_forecast.items():
                # Get fabric specs for this style
                fabric_specs = self._get_fabric_specs(style)
                if not fabric_specs:
                    logger.warning(f"No fabric specs found for style {style}, skipping")
                    continue

                # Get BOM for this style
                bom = self._get_bom(style)
                if not bom:
                    logger.warning(f"No BOM found for style {style}, skipping")
                    continue

                # Convert each week's sales to yarn requirements
                for week_num, yards in weekly_sales.items():
                    if weeks_to_explode and week_num not in weeks_to_explode:
                        continue

                    # Step 1: Convert yards to fabric lbs
                    fabric_lbs = self._yards_to_fabric_lbs(yards, fabric_specs)

                    # Step 2: Explode fabric lbs to yarn lbs using BOM percentages
                    yarn_lbs_breakdown = self._fabric_to_yarn_lbs(fabric_lbs, bom)

                    # Step 3: Accumulate yarn requirements by week
                    for yarn_id, lbs_needed in yarn_lbs_breakdown.items():
                        if yarn_id not in yarn_requirements:
                            yarn_requirements[yarn_id] = {}

                        if week_num not in yarn_requirements[yarn_id]:
                            yarn_requirements[yarn_id][week_num] = 0.0

                        yarn_requirements[yarn_id][week_num] += lbs_needed

            logger.info(
                f"Exploded sales for {len(sales_forecast)} styles into "
                f"{len(yarn_requirements)} yarn requirements"
            )

            return yarn_requirements

        except Exception as e:
            logger.exception(f"Error in BOM explosion: {e}")
            return {}

    def _preload_bom_and_specs(self, styles: List[str]) -> None:
        """
        Pre-load BOM and fabric specs for multiple styles to minimize queries

        Args:
            styles: List of style codes to load
        """
        try:
            # Query all BOM entries for these styles
            if not styles:
                return

            # Build SQL with proper parameter binding
            placeholders = ','.join(['?' for _ in styles])
            sql = f"""
                SELECT style, yarn_id, percentage
                FROM bom
                WHERE style IN ({placeholders})
            """

            bom_rows = self.turso_client.execute(sql, styles)

            # Organize by style
            for row in bom_rows:
                style = row['style']
                yarn_id = row['yarn_id']
                percentage = row['percentage']

                if style not in self._bom_cache:
                    self._bom_cache[style] = {}

                self._bom_cache[style][yarn_id] = percentage

            # Query all fabric specs for these styles
            sql = f"""
                SELECT style, yds_per_lb, gsm, width, fabric_type
                FROM fabric_specs
                WHERE style IN ({placeholders})
            """

            specs_rows = self.turso_client.execute(sql, styles)

            # Store in cache
            for row in specs_rows:
                style = row['style']
                self._fabric_specs_cache[style] = dict(row)

            logger.info(
                f"Pre-loaded BOM for {len(self._bom_cache)} styles, "
                f"specs for {len(self._fabric_specs_cache)} styles"
            )

        except Exception as e:
            logger.exception(f"Error pre-loading data: {e}")

    def _get_bom(self, style: str) -> Dict[str, float]:
        """
        Get BOM percentages for a style from cache or database

        Args:
            style: Style code

        Returns:
            {yarn_id: percentage} - Yarn percentages (e.g., 0.6 for 60%)
        """
        if style in self._bom_cache:
            return self._bom_cache[style]

        # Query from database if not cached
        try:
            sql = "SELECT yarn_id, percentage FROM bom WHERE style = ?"
            rows = self.turso_client.execute(sql, [style])

            bom_dict = {row['yarn_id']: row['percentage'] for row in rows}
            self._bom_cache[style] = bom_dict

            return bom_dict

        except Exception as e:
            logger.error(f"Error getting BOM for {style}: {e}")
            return {}

    def _get_fabric_specs(self, style: str) -> Optional[Dict[str, Any]]:
        """
        Get fabric specifications for a style from cache or database

        Args:
            style: Style code

        Returns:
            Fabric specs dict with yds_per_lb, gsm, width, fabric_type
        """
        if style in self._fabric_specs_cache:
            return self._fabric_specs_cache[style]

        # Query from database if not cached
        try:
            sql = "SELECT * FROM fabric_specs WHERE style = ?"
            rows = self.turso_client.execute(sql, [style])

            if rows:
                specs = dict(rows[0])
                self._fabric_specs_cache[style] = specs
                return specs
            else:
                logger.warning(f"No fabric specs found for style {style}")
                return None

        except Exception as e:
            logger.error(f"Error getting fabric specs for {style}: {e}")
            return None

    def _yards_to_fabric_lbs(
        self,
        yards: float,
        fabric_specs: Dict[str, Any]
    ) -> float:
        """
        Convert yards of fabric to pounds using yards-per-pound ratio

        Args:
            yards: Yards of fabric needed
            fabric_specs: Fabric specifications including yds_per_lb

        Returns:
            Pounds of fabric needed
        """
        yds_per_lb = fabric_specs.get('yds_per_lb')

        if not yds_per_lb or yds_per_lb <= 0:
            logger.warning(
                f"Invalid yds_per_lb: {yds_per_lb}, using default 2.5"
            )
            yds_per_lb = 2.5  # Default fallback

        fabric_lbs = yards / yds_per_lb
        return fabric_lbs

    def _fabric_to_yarn_lbs(
        self,
        fabric_lbs: float,
        bom: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Explode fabric pounds to yarn pounds using BOM percentages

        Args:
            fabric_lbs: Pounds of fabric needed
            bom: {yarn_id: percentage} - BOM breakdown

        Returns:
            {yarn_id: lbs} - Yarn requirements in pounds
        """
        yarn_breakdown = {}

        for yarn_id, percentage in bom.items():
            # Convert percentage to decimal if needed (handle both 0.6 and 60 formats)
            if percentage > 1:
                percentage = percentage / 100

            yarn_lbs = fabric_lbs * percentage
            yarn_breakdown[yarn_id] = yarn_lbs

        return yarn_breakdown

    def explode_single_style(
        self,
        style: str,
        quantity_yards: float
    ) -> Dict[str, float]:
        """
        Explode a single style/quantity to yarn requirements (one-time calculation)

        Args:
            style: Style code
            quantity_yards: Quantity in yards

        Returns:
            {yarn_id: lbs} - Yarn requirements

        Example:
            explode_single_style("STYLE001", 1000)
            Returns: {"18884": 250.5, "18885": 125.3}
        """
        try:
            # Get fabric specs
            fabric_specs = self._get_fabric_specs(style)
            if not fabric_specs:
                logger.error(f"No fabric specs for style {style}")
                return {}

            # Get BOM
            bom = self._get_bom(style)
            if not bom:
                logger.error(f"No BOM for style {style}")
                return {}

            # Convert yards to fabric lbs
            fabric_lbs = self._yards_to_fabric_lbs(quantity_yards, fabric_specs)

            # Explode to yarn lbs
            yarn_requirements = self._fabric_to_yarn_lbs(fabric_lbs, bom)

            return yarn_requirements

        except Exception as e:
            logger.exception(f"Error exploding single style {style}: {e}")
            return {}

    def validate_bom_coverage(
        self,
        styles: List[str]
    ) -> Dict[str, Any]:
        """
        Validate BOM and fabric specs coverage for given styles

        Args:
            styles: List of style codes to validate

        Returns:
            Validation report with coverage statistics
        """
        report = {
            'total_styles': len(styles),
            'styles_with_bom': 0,
            'styles_with_specs': 0,
            'styles_complete': 0,
            'missing_bom': [],
            'missing_specs': [],
            'missing_both': []
        }

        for style in styles:
            has_bom = bool(self._get_bom(style))
            has_specs = bool(self._get_fabric_specs(style))

            if has_bom:
                report['styles_with_bom'] += 1
            else:
                report['missing_bom'].append(style)

            if has_specs:
                report['styles_with_specs'] += 1
            else:
                report['missing_specs'].append(style)

            if has_bom and has_specs:
                report['styles_complete'] += 1
            elif not has_bom and not has_specs:
                report['missing_both'].append(style)

        report['coverage_pct'] = (
            report['styles_complete'] / report['total_styles'] * 100
            if report['total_styles'] > 0 else 0
        )

        logger.info(
            f"BOM Coverage: {report['styles_complete']}/{report['total_styles']} "
            f"({report['coverage_pct']:.1f}%)"
        )

        return report


def main():
    """Test BOM explosion with sample data"""

    explosion = TursoBOMExplosion()

    # Test single style explosion
    print("=== Single Style Explosion Test ===")
    result = explosion.explode_single_style("STYLE001", 1000)
    print(f"1000 yards of STYLE001 requires:")
    for yarn_id, lbs in result.items():
        print(f"  {yarn_id}: {lbs:.2f} lbs")

    # Test weekly explosion
    print("\n=== Weekly Forecast Explosion Test ===")
    sales_forecast = {
        "STYLE001": {42: 1000, 43: 1500},
        "STYLE002": {42: 500, 43: 800}
    }

    yarn_requirements = explosion.explode_sales_to_yarn_weekly(sales_forecast)

    print(f"Weekly yarn requirements:")
    for yarn_id, weekly_reqs in yarn_requirements.items():
        print(f"\nYarn {yarn_id}:")
        for week_num, lbs in sorted(weekly_reqs.items()):
            print(f"  Week {week_num}: {lbs:.2f} lbs")

    # Validation test
    print("\n=== BOM Coverage Validation ===")
    test_styles = ["STYLE001", "STYLE002", "STYLE003"]
    validation = explosion.validate_bom_coverage(test_styles)
    print(f"Coverage: {validation['coverage_pct']:.1f}%")
    print(f"Complete: {validation['styles_complete']}/{validation['total_styles']}")
    if validation['missing_bom']:
        print(f"Missing BOM: {validation['missing_bom']}")
    if validation['missing_specs']:
        print(f"Missing Specs: {validation['missing_specs']}")


if __name__ == "__main__":
    main()
