"""Column Name Resolver for Beverly Knits ERP.

Centralizes column name mapping to handle variations across different
data sources and formats.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import logging
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)


class ColumnResolver:
    """Centralized column name resolver.

    Handles all column name variations across different data sources,
    ensuring consistent access to data regardless of naming conventions.
    """

    # Master column mappings
    COLUMN_MAPPINGS = {
        # Yarn related
        'yarn_id': [
            'YarnID', 'Desc#', 'desc_num', 'Yarn_ID', 'yarn_code',
            'YarnCode', 'Yarn ID', 'YARNID', 'yarn_id', 'Yarn'
        ],
        'yarn_description': [
            'YarnDescription', 'Yarn_Description', 'Description',
            'Yarn Desc', 'YarnDesc', 'yarn_desc', 'Desc'
        ],

        # Style related
        'style': [
            'Style#', 'fStyle#', 'style_num', 'StyleNumber',
            'Style', 'style_code', 'StyleCode', 'STYLE', 'Style #'
        ],

        # Inventory related
        'planning_balance': [
            'Planning Balance', 'Planning_Balance', 'PlanningBalance',
            'Plan Balance', 'plan_balance', 'PLANNING_BALANCE', 'Planning Bal'
        ],
        'on_hand': [
            'On Hand', 'On_Hand', 'OnHand', 'Balance (lbs)',
            'Current Balance', 'on_hand', 'OH', 'On-Hand', 'Balance'
        ],
        'allocated': [
            'Allocated', 'allocated', 'ALLOCATED', 'Reserved',
            'Alloc', 'allocation', 'Allocated Qty'
        ],
        'on_order': [
            'On Order', 'On_Order', 'OnOrder', 'PO Qty',
            'Purchase Order', 'on_order', 'OO', 'On-Order', 'PO'
        ],

        # BOM related
        'usage': [
            'Usage', 'Qty', 'Quantity', 'Amount', 'Required',
            'Requirement', 'usage', 'QTY', 'Consumption'
        ],
        'uom': [
            'UOM', 'Unit', 'UnitOfMeasure', 'Unit_of_Measure',
            'Units', 'unit', 'UNIT', 'Unit of Measure'
        ],

        # Machine related
        'work_center': [
            'Work Center', 'WorkCenter', 'WC', 'work_center',
            'Work_Center', 'WORK_CENTER', 'Wc', 'Work-Center'
        ],
        'machine': [
            'Machine', 'MACH', 'MachineID', 'machine_id',
            'Machine_ID', 'machine', 'MACHINE', 'Machine #'
        ],

        # Order related
        'order_number': [
            'Order#', 'OrderNumber', 'Order_Number', 'OrderID',
            'Order ID', 'order_id', 'ORDER', 'Order #', 'Order'
        ],
        'quantity': [
            'Quantity', 'Qty', 'QTY', 'quantity', 'Amount',
            'Quanity',  # Common typo
            'Qty.', 'Quan'
        ],
        'due_date': [
            'Due Date', 'DueDate', 'Due_Date', 'due_date',
            'Delivery Date', 'DeliveryDate', 'Due'
        ],

        # Additional fields
        'supplier': [
            'Supplier', 'Vendor', 'supplier', 'SUPPLIER',
            'Supplier Name', 'Vendor Name'
        ],
        'color': [
            'Color', 'Colour', 'COLOR', 'color', 'Color Code'
        ]
    }

    def __init__(self) -> None:
        """Initialize resolver with caching."""
        self.resolution_cache: Dict[str, str] = {}
        self.resolution_stats: Dict[str, int] = {}
        logger.info("ColumnResolver initialized")

    @lru_cache(maxsize=128)
    def resolve_column(
        self,
        df_columns: tuple,
        standard_name: str
    ) -> Optional[str]:
        """Resolve standard name to actual column name.

        Args:
            df_columns: Tuple of DataFrame column names (for caching)
            standard_name: Standard column name to resolve

        Returns:
            Actual column name or None if not found
        """
        # Get possible variations
        if standard_name not in self.COLUMN_MAPPINGS:
            logger.warning(f"Unknown standard column: {standard_name}")
            return None

        possible_names = self.COLUMN_MAPPINGS[standard_name]

        # Search for column
        for name in possible_names:
            if name in df_columns:
                # Track statistics
                stat_key = f"{standard_name}->{name}"
                self.resolution_stats[stat_key] = \
                    self.resolution_stats.get(stat_key, 0) + 1
                return name

        return None

    def resolve_required(
        self,
        df: pd.DataFrame,
        standard_name: str
    ) -> str:
        """Resolve column name, raising error if not found.

        Args:
            df: DataFrame to search
            standard_name: Standard column name

        Returns:
            Actual column name

        Raises:
            KeyError: If column not found
        """
        column = self.resolve_column(tuple(df.columns), standard_name)

        if column is None:
            possible = self.COLUMN_MAPPINGS.get(standard_name, [])
            raise KeyError(
                f"Required column '{standard_name}' not found. "
                f"Searched for: {possible}. "
                f"Available: {list(df.columns)[:10]}..."
            )

        return column

    def standardize_dataframe(
        self,
        df: pd.DataFrame,
        inplace: bool = False
    ) -> pd.DataFrame:
        """Standardize all column names in DataFrame.

        Args:
            df: DataFrame to standardize
            inplace: Whether to modify in place

        Returns:
            DataFrame with standardized column names
        """
        if not inplace:
            df = df.copy()

        # Map columns to standard names
        rename_map = {}

        for col in df.columns:
            for standard_name, variations in self.COLUMN_MAPPINGS.items():
                if col in variations:
                    # Use first variation as standard
                    standard_col = variations[0]
                    if col != standard_col:
                        rename_map[col] = standard_col
                    break

        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.info(f"Standardized {len(rename_map)} column names")

        return df

    def safe_get(
        self,
        df: pd.DataFrame,
        row_index: int,
        standard_name: str,
        default: Any = None
    ) -> Any:
        """Safely get value from DataFrame.

        Args:
            df: DataFrame
            row_index: Row index
            standard_name: Standard column name
            default: Default value if not found

        Returns:
            Value or default
        """
        try:
            column = self.resolve_column(tuple(df.columns), standard_name)
            if column is None:
                return default

            value = df.iloc[row_index][column]

            # Clean common issues
            if pd.isna(value):
                return default

            # Handle text in numeric fields
            if isinstance(value, str):
                if value.upper() in ['N/A', 'NA', 'NULL', 'NONE', '-']:
                    return default

                # Remove currency symbols and commas
                if standard_name in ['on_hand', 'on_order', 'allocated',
                                     'planning_balance', 'quantity', 'usage']:
                    value = value.replace('$', '').replace(',', '')
                    try:
                        return float(value)
                    except ValueError:
                        return default

            return value

        except (IndexError, KeyError):
            return default

    def get_mapping_report(self) -> Dict[str, Any]:
        """Get column mapping statistics.

        Returns:
            Report of resolution statistics
        """
        return {
            'total_resolutions': sum(self.resolution_stats.values()),
            'unique_mappings': len(self.resolution_stats),
            'top_mappings': sorted(
                self.resolution_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate that required columns exist.

        Args:
            df: DataFrame to validate
            required: List of required standard column names

        Returns:
            Tuple of (all_found, missing_columns)
        """
        missing = []

        for standard_name in required:
            column = self.resolve_column(tuple(df.columns), standard_name)
            if column is None:
                missing.append(standard_name)

        return len(missing) == 0, missing


if __name__ == "__main__":
    """Validation with sample data."""

    resolver = ColumnResolver()

    # Test Case 1: Create sample DataFrame with variations
    sample_df = pd.DataFrame({
        'Desc#': ['Y001', 'Y002'],
        'Planning_Balance': [100, -50],
        'On-Hand': [200, 100],
        'PO Qty': [50, 0],
        'Allocated Qty': [150, 150]
    })

    # Test column resolution
    yarn_col = resolver.resolve_column(tuple(sample_df.columns), 'yarn_id')
    assert yarn_col == 'Desc#'
    logger.info(f"Test 1 passed: yarn_id -> {yarn_col}")

    balance_col = resolver.resolve_column(tuple(sample_df.columns), 'planning_balance')
    assert balance_col == 'Planning_Balance'
    logger.info(f"Test 2 passed: planning_balance -> {balance_col}")

    # Test standardization
    std_df = resolver.standardize_dataframe(sample_df)
    assert 'YarnID' in std_df.columns  # Standardized to first variation
    logger.info("Test 3 passed: DataFrame standardized")

    # Test safe get
    value = resolver.safe_get(sample_df, 0, 'on_hand', default=0)
    assert value == 200
    logger.info(f"Test 4 passed: Safe get value = {value}")

    # Test validation
    valid, missing = resolver.validate_required_columns(
        sample_df,
        ['yarn_id', 'planning_balance', 'style']  # style missing
    )
    assert valid is False
    assert 'style' in missing
    logger.info("Test 5 passed: Validation detected missing column")

    print("All validations passed!")