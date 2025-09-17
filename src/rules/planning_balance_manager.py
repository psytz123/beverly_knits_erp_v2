"""Planning Balance Manager with Shortage Detection.

This module handles planning balance calculations and shortage detection
for yarn inventory. Negative balances indicate future shortages and are
properly categorized by severity for action planning.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlanningBalanceResult:
    """Result container for planning balance calculation."""

    yarn_id: str
    on_hand: float
    on_order: float
    allocated: float
    planning_balance: float
    is_shortage: bool
    shortage_amount: float
    action_required: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "OK"]
    weeks_coverage: float

    def __post_init__(self) -> None:
        """Validate and categorize shortage severity."""
        if self.on_hand < 0 or self.on_order < 0 or self.allocated < 0:
            raise ValueError("Input values cannot be negative")

        # Categorize action level based on shortage amount
        if self.planning_balance >= 0:
            self.action_required = "OK"
        elif self.planning_balance > -50:
            self.action_required = "LOW"
        elif self.planning_balance > -100:
            self.action_required = "MEDIUM"
        elif self.planning_balance > -500:
            self.action_required = "HIGH"
        else:
            self.action_required = "CRITICAL"

    def format_display(self) -> str:
        """Format balance for display (negative in parentheses)."""
        if self.planning_balance < 0:
            return f"({abs(self.planning_balance):.2f})"
        return f"{self.planning_balance:.2f}"


class PlanningBalanceAnalyzer:
    """Analyzer for planning balance and shortage detection.

    This class correctly handles negative planning balances as shortage
    indicators, not errors. Negative values show future inventory shortfalls
    that require action.
    """

    # Shortage severity thresholds (in lbs)
    SHORTAGE_THRESHOLDS = {
        'critical': -500,    # More than 500 lbs short
        'high': -100,        # 100-500 lbs short
        'medium': -50,       # 50-100 lbs short
        'low': 0,           # Any shortage
        'warning': 50,      # Getting low (positive but concerning)
        'safe': 200         # Comfortable inventory level
    }

    # Weekly usage estimate for coverage calculation
    DEFAULT_WEEKLY_USAGE = 50.0  # lbs per week default

    def __init__(self) -> None:
        """Initialize analyzer with tracking lists."""
        self.shortage_log: List[PlanningBalanceResult] = []
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("PlanningBalanceAnalyzer initialized")

    def calculate_planning_balance(
        self,
        yarn_id: str,
        on_hand: float,
        on_order: float,
        allocated: float,
        *,
        weekly_usage: Optional[float] = None,
        log_shortage: bool = True
    ) -> PlanningBalanceResult:
        """Calculate planning balance with shortage detection.

        Formula: Planning Balance = On Hand + On Order - Allocated

        Negative values indicate future shortages (this is VALID and expected).
        Positive values indicate available inventory.

        Args:
            yarn_id: Yarn identifier
            on_hand: Current physical inventory (lbs)
            on_order: Incoming inventory from POs (lbs)
            allocated: Reserved for production orders (lbs)
            weekly_usage: Average weekly usage for coverage calc
            log_shortage: Whether to log shortages

        Returns:
            PlanningBalanceResult with shortage analysis

        Raises:
            ValueError: If input values are negative
        """
        # Validate inputs (these cannot be negative)
        if on_hand < 0:
            raise ValueError(f"On Hand cannot be negative: {on_hand}")
        if on_order < 0:
            raise ValueError(f"On Order cannot be negative: {on_order}")
        if allocated < 0:
            raise ValueError(f"Allocated cannot be negative: {allocated}")

        # Calculate planning balance
        planning_balance = on_hand + on_order - allocated

        # Determine shortage status
        is_shortage = planning_balance < 0
        shortage_amount = abs(planning_balance) if is_shortage else 0.0

        # Calculate weeks of coverage
        usage = weekly_usage or self.DEFAULT_WEEKLY_USAGE
        weeks_coverage = max(0, planning_balance / usage) if usage > 0 else 999

        # Create result
        result = PlanningBalanceResult(
            yarn_id=yarn_id,
            on_hand=on_hand,
            on_order=on_order,
            allocated=allocated,
            planning_balance=planning_balance,
            is_shortage=is_shortage,
            shortage_amount=shortage_amount,
            action_required="OK",  # Will be set in __post_init__
            weeks_coverage=weeks_coverage
        )

        # Log if shortage detected
        if is_shortage and log_shortage:
            self.shortage_log.append(result)
            logger.warning(
                f"SHORTAGE - {yarn_id}: {result.format_display()} lbs "
                f"(Action: {result.action_required})"
            )

        # Track in history
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'yarn_id': yarn_id,
            'planning_balance': planning_balance,
            'is_shortage': is_shortage,
            'action': result.action_required
        })

        return result

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze planning balance for entire DataFrame.

        Args:
            df: DataFrame with columns: YarnID, On Hand, On Order, Allocated

        Returns:
            DataFrame with added shortage analysis columns
        """
        required_cols = ['YarnID', 'On Hand', 'On Order', 'Allocated']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate planning balance for each row
        results = []
        for _, row in df.iterrows():
            result = self.calculate_planning_balance(
                yarn_id=row['YarnID'],
                on_hand=row['On Hand'],
                on_order=row['On Order'],
                allocated=row['Allocated'],
                log_shortage=False  # Don't log each one
            )
            results.append({
                'YarnID': result.yarn_id,
                'Planning Balance': result.planning_balance,
                'Display Balance': result.format_display(),
                'Has Shortage': result.is_shortage,
                'Shortage Amount': result.shortage_amount,
                'Action Required': result.action_required,
                'Weeks Coverage': result.weeks_coverage
            })

        # Merge with original DataFrame
        result_df = pd.DataFrame(results)
        return df.merge(result_df, on='YarnID', how='left')

    def get_shortage_summary(self) -> Dict[str, Any]:
        """Get summary of all logged shortages.

        Returns:
            Dictionary with shortage statistics
        """
        if not self.shortage_log:
            return {
                'total_shortages': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'total_shortage_lbs': 0.0
            }

        # Count by severity
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }

        total_shortage = 0.0
        for result in self.shortage_log:
            severity_counts[result.action_required] += 1
            total_shortage += result.shortage_amount

        return {
            'total_shortages': len(self.shortage_log),
            'critical': severity_counts['CRITICAL'],
            'high': severity_counts['HIGH'],
            'medium': severity_counts['MEDIUM'],
            'low': severity_counts['LOW'],
            'total_shortage_lbs': total_shortage,
            'avg_shortage_lbs': total_shortage / len(self.shortage_log)
        }

    def get_critical_shortages(self) -> List[PlanningBalanceResult]:
        """Get list of critical shortages requiring immediate action.

        Returns:
            List of PlanningBalanceResult with CRITICAL or HIGH severity
        """
        return [
            r for r in self.shortage_log
            if r.action_required in ['CRITICAL', 'HIGH']
        ]

    def export_shortage_report(self) -> pd.DataFrame:
        """Export shortage report as DataFrame.

        Returns:
            DataFrame with all shortage details
        """
        if not self.shortage_log:
            return pd.DataFrame()

        data = []
        for result in self.shortage_log:
            data.append({
                'YarnID': result.yarn_id,
                'Planning Balance': result.planning_balance,
                'Display': result.format_display(),
                'Shortage Amount': result.shortage_amount,
                'Action Required': result.action_required,
                'On Hand': result.on_hand,
                'On Order': result.on_order,
                'Allocated': result.allocated,
                'Weeks Coverage': result.weeks_coverage
            })

        df = pd.DataFrame(data)
        # Sort by severity (most critical first)
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        df['severity_rank'] = df['Action Required'].map(severity_order)
        df = df.sort_values(['severity_rank', 'Shortage Amount'],
                          ascending=[True, False])
        df = df.drop('severity_rank', axis=1)

        return df


if __name__ == "__main__":
    """Validation with real data patterns."""

    analyzer = PlanningBalanceAnalyzer()

    # Test Case 1: Normal shortage (as shown in screenshot)
    result1 = analyzer.calculate_planning_balance(
        yarn_id="Y18771",
        on_hand=1291.0,
        on_order=0.0,
        allocated=1450.0
    )
    assert result1.planning_balance == -159.0
    assert result1.is_shortage is True
    assert result1.format_display() == "(159.00)"
    logger.info(f"Test 1 passed: {result1.format_display()}")

    # Test Case 2: Positive balance
    result2 = analyzer.calculate_planning_balance(
        yarn_id="Y14415",
        on_hand=500.0,
        on_order=100.0,
        allocated=400.0
    )
    assert result2.planning_balance == 200.0
    assert result2.is_shortage is False
    logger.info(f"Test 2 passed: {result2.format_display()}")

    # Test Case 3: Critical shortage
    result3 = analyzer.calculate_planning_balance(
        yarn_id="Y19069",
        on_hand=0.0,
        on_order=0.0,
        allocated=945.0
    )
    assert result3.planning_balance == -945.0
    assert result3.action_required == "CRITICAL"
    logger.info(f"Test 3 passed: Critical shortage detected")

    # Get summary
    summary = analyzer.get_shortage_summary()
    logger.info(f"Shortage summary: {summary}")

    print("All validations passed!")