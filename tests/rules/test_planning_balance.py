"""Tests for Planning Balance Manager."""

import pytest
from typing import Any, Dict
import pandas as pd

from src.rules.planning_balance_manager import (
    PlanningBalanceAnalyzer,
    PlanningBalanceResult
)


class TestPlanningBalanceResult:
    """Test suite for PlanningBalanceResult."""

    def test_normal_shortage(self) -> None:
        """Test normal shortage detection."""
        result = PlanningBalanceResult(
            yarn_id="Y001",
            on_hand=100.0,
            on_order=50.0,
            allocated=200.0,
            planning_balance=-50.0,
            is_shortage=True,
            shortage_amount=50.0,
            action_required="OK",
            weeks_coverage=0.0
        )

        assert result.planning_balance == -50.0
        assert result.is_shortage is True
        assert result.action_required == "MEDIUM"  # Set in __post_init__
        assert result.format_display() == "(50.00)"

    def test_critical_shortage(self) -> None:
        """Test critical shortage categorization."""
        result = PlanningBalanceResult(
            yarn_id="Y002",
            on_hand=0.0,
            on_order=0.0,
            allocated=600.0,
            planning_balance=-600.0,
            is_shortage=True,
            shortage_amount=600.0,
            action_required="OK",
            weeks_coverage=0.0
        )

        assert result.action_required == "CRITICAL"

    def test_no_shortage(self) -> None:
        """Test positive balance (no shortage)."""
        result = PlanningBalanceResult(
            yarn_id="Y003",
            on_hand=500.0,
            on_order=200.0,
            allocated=400.0,
            planning_balance=300.0,
            is_shortage=False,
            shortage_amount=0.0,
            action_required="OK",
            weeks_coverage=6.0
        )

        assert result.action_required == "OK"
        assert result.format_display() == "300.00"

    def test_invalid_inputs(self) -> None:
        """Test that negative inputs raise error."""
        with pytest.raises(ValueError, match="Input values cannot be negative"):
            PlanningBalanceResult(
                yarn_id="Y004",
                on_hand=-100.0,  # Invalid!
                on_order=50.0,
                allocated=200.0,
                planning_balance=0.0,
                is_shortage=False,
                shortage_amount=0.0,
                action_required="OK",
                weeks_coverage=0.0
            )


class TestPlanningBalanceAnalyzer:
    """Test suite for PlanningBalanceAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> PlanningBalanceAnalyzer:
        """Provide analyzer instance."""
        return PlanningBalanceAnalyzer()

    def test_calculate_planning_balance_shortage(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test planning balance calculation with shortage."""
        result = analyzer.calculate_planning_balance(
            yarn_id="Y001",
            on_hand=100.0,
            on_order=50.0,
            allocated=200.0
        )

        assert result.planning_balance == -50.0
        assert result.is_shortage is True
        assert result.shortage_amount == 50.0
        assert result.action_required == "MEDIUM"

    def test_calculate_planning_balance_positive(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test planning balance calculation with positive result."""
        result = analyzer.calculate_planning_balance(
            yarn_id="Y002",
            on_hand=500.0,
            on_order=100.0,
            allocated=400.0
        )

        assert result.planning_balance == 200.0
        assert result.is_shortage is False
        assert result.shortage_amount == 0.0
        assert result.action_required == "OK"

    def test_weeks_coverage_calculation(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test weeks of coverage calculation."""
        result = analyzer.calculate_planning_balance(
            yarn_id="Y003",
            on_hand=200.0,
            on_order=0.0,
            allocated=0.0,
            weekly_usage=50.0
        )

        assert result.weeks_coverage == 4.0  # 200 / 50

    def test_shortage_logging(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test that shortages are logged."""
        # Create shortage
        analyzer.calculate_planning_balance(
            yarn_id="Y001",
            on_hand=0.0,
            on_order=0.0,
            allocated=100.0,
            log_shortage=True
        )

        assert len(analyzer.shortage_log) == 1
        assert analyzer.shortage_log[0].yarn_id == "Y001"

    def test_analyze_dataframe(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test DataFrame analysis."""
        df = pd.DataFrame([
            {'YarnID': 'Y001', 'On Hand': 100, 'On Order': 50, 'Allocated': 200},
            {'YarnID': 'Y002', 'On Hand': 500, 'On Order': 0, 'Allocated': 300},
            {'YarnID': 'Y003', 'On Hand': 0, 'On Order': 0, 'Allocated': 1000}
        ])

        result_df = analyzer.analyze_dataframe(df)

        assert 'Planning Balance' in result_df.columns
        assert 'Has Shortage' in result_df.columns
        assert 'Action Required' in result_df.columns

        # Check specific results
        y001 = result_df[result_df['YarnID'] == 'Y001'].iloc[0]
        assert y001['Planning Balance'] == -50.0
        assert y001['Has Shortage'] is True

        y002 = result_df[result_df['YarnID'] == 'Y002'].iloc[0]
        assert y002['Planning Balance'] == 200.0
        assert y002['Has Shortage'] is False

    def test_get_shortage_summary(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test shortage summary generation."""
        # Create various shortages
        analyzer.calculate_planning_balance("Y001", 0, 0, 50)    # Low
        analyzer.calculate_planning_balance("Y002", 0, 0, 100)   # Medium
        analyzer.calculate_planning_balance("Y003", 0, 0, 200)   # High
        analyzer.calculate_planning_balance("Y004", 0, 0, 600)   # Critical

        summary = analyzer.get_shortage_summary()

        assert summary['total_shortages'] == 4
        assert summary['critical'] == 1
        assert summary['high'] == 1
        assert summary['medium'] == 1
        assert summary['low'] == 1
        assert summary['total_shortage_lbs'] == 950.0

    def test_get_critical_shortages(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test critical shortage filtering."""
        # Create shortages of different severities
        analyzer.calculate_planning_balance("Y001", 0, 0, 50)    # Low
        analyzer.calculate_planning_balance("Y002", 0, 0, 600)   # Critical
        analyzer.calculate_planning_balance("Y003", 0, 0, 200)   # High

        critical = analyzer.get_critical_shortages()

        assert len(critical) == 2  # Critical and High
        assert any(r.yarn_id == "Y002" for r in critical)
        assert any(r.yarn_id == "Y003" for r in critical)

    def test_export_shortage_report(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test shortage report export."""
        # Create shortages
        analyzer.calculate_planning_balance("Y001", 100, 50, 200)
        analyzer.calculate_planning_balance("Y002", 0, 0, 600)

        report = analyzer.export_shortage_report()

        assert len(report) == 2
        assert 'YarnID' in report.columns
        assert 'Planning Balance' in report.columns
        assert 'Action Required' in report.columns

        # Check sorting (most critical first)
        assert report.iloc[0]['YarnID'] == 'Y002'  # Critical
        assert report.iloc[1]['YarnID'] == 'Y001'  # Medium

    @pytest.mark.parametrize("on_hand,on_order,allocated,expected_balance", [
        (100.0, 50.0, 150.0, 0.0),     # Exact balance
        (200.0, 100.0, 100.0, 200.0),  # Positive
        (50.0, 0.0, 100.0, -50.0),     # Shortage
        (0.0, 0.0, 1000.0, -1000.0),   # Critical shortage
    ])
    def test_various_balance_scenarios(
        self,
        analyzer: PlanningBalanceAnalyzer,
        on_hand: float,
        on_order: float,
        allocated: float,
        expected_balance: float
    ) -> None:
        """Test various balance calculation scenarios."""
        result = analyzer.calculate_planning_balance(
            yarn_id="TEST",
            on_hand=on_hand,
            on_order=on_order,
            allocated=allocated
        )

        assert result.planning_balance == expected_balance

    def test_negative_input_validation(
        self,
        analyzer: PlanningBalanceAnalyzer
    ) -> None:
        """Test that negative inputs are rejected."""
        with pytest.raises(ValueError, match="On Hand cannot be negative"):
            analyzer.calculate_planning_balance("Y001", -100, 50, 200)

        with pytest.raises(ValueError, match="On Order cannot be negative"):
            analyzer.calculate_planning_balance("Y001", 100, -50, 200)

        with pytest.raises(ValueError, match="Allocated cannot be negative"):
            analyzer.calculate_planning_balance("Y001", 100, 50, -200)