"""Beverly Knits Rules Engine - Master Orchestrator.

This module orchestrates all business rules and provides a unified
interface for rule validation and enforcement across the ERP system.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

import pandas as pd

from .planning_balance_manager import PlanningBalanceAnalyzer
from .work_center_validator import WorkCenterValidator
from .bom_variant_resolver import BOMVariantResolver
from .knit_order_processor import KnitOrderProcessor
from .cache_manager import CacheManager
from .efab_integration import eFabIntegration

logger = logging.getLogger(__name__)


class BeverlyKnitsRulesEngine:
    """Master orchestrator for all Beverly Knits business rules.

    Provides a unified interface for:
    - Planning balance and shortage detection
    - Work center pattern validation
    - BOM style/variant resolution
    - Machine assignment validation
    - Column name resolution
    - Cache management
    - eFab integration
    - Yarn substitution
    """

    def __init__(self) -> None:
        """Initialize all rule components."""
        self.planning_balance = PlanningBalanceAnalyzer()
        self.work_center = WorkCenterValidator()
        self.bom_resolver = BOMVariantResolver()

        # Initialize cache and eFab integration
        self.cache_manager = CacheManager()
        self.efab_integration = eFabIntegration()

        # Initialize knit order processor with dependencies
        # Default to using eFab API for knit orders
        self.knit_order_processor = KnitOrderProcessor(
            efab_integration=self.efab_integration,
            cache_manager=self.cache_manager,
            use_efab_api=True  # Use eFab API endpoint /api/knitorder/list
        )

        # Tracking
        self.execution_log: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []

        logger.info("BeverlyKnitsRulesEngine initialized")

    def validate_yarn_inventory(
        self,
        yarn_df: pd.DataFrame,
        detect_shortages: bool = True
    ) -> Dict[str, Any]:
        """Validate yarn inventory and detect shortages.

        Args:
            yarn_df: Yarn inventory DataFrame
            detect_shortages: Whether to detect shortages

        Returns:
            Validation results with shortage analysis
        """
        start_time = datetime.now()

        try:
            # Analyze planning balance
            result_df = self.planning_balance.analyze_dataframe(yarn_df)

            # Get shortage summary
            shortage_summary = self.planning_balance.get_shortage_summary()

            # Get critical shortages
            critical = self.planning_balance.get_critical_shortages()

            result = {
                'status': 'success',
                'total_yarns': len(yarn_df),
                'shortages': shortage_summary,
                'critical_count': len(critical),
                'critical_yarns': [r.yarn_id for r in critical[:5]],
                'data': result_df
            }

            self._log_execution('validate_yarn_inventory', 'success', start_time)
            return result

        except Exception as e:
            logger.exception(f"Yarn inventory validation failed: {e}")
            self._log_error('validate_yarn_inventory', str(e))
            return {
                'status': 'error',
                'error': str(e),
                'data': yarn_df
            }

    def validate_machine_assignments(
        self,
        machine_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate all machine work center assignments.

        Args:
            machine_df: Machine mapping DataFrame

        Returns:
            Validation results
        """
        start_time = datetime.now()

        try:
            # Validate work centers
            result_df = self.work_center.validate_machine_mapping(machine_df)

            # Get validation summary
            summary = self.work_center.get_validation_summary()

            # Find invalid assignments
            invalid = result_df[result_df['Valid'] == False]

            result = {
                'status': 'success',
                'total_machines': len(machine_df),
                'validation_summary': summary,
                'invalid_count': len(invalid),
                'invalid_work_centers': invalid['WC'].unique().tolist()[:10],
                'data': result_df
            }

            self._log_execution('validate_machine_assignments', 'success', start_time)
            return result

        except Exception as e:
            logger.exception(f"Machine assignment validation failed: {e}")
            self._log_error('validate_machine_assignments', str(e))
            return {
                'status': 'error',
                'error': str(e),
                'data': machine_df
            }

    def resolve_order_requirements(
        self,
        orders_df: pd.DataFrame,
        bom_df: pd.DataFrame,
        yarn_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Resolve yarn requirements for production orders.

        Args:
            orders_df: Production orders DataFrame
            bom_df: BOM DataFrame
            yarn_df: Optional yarn inventory for shortage check

        Returns:
            Yarn requirements with shortage analysis
        """
        start_time = datetime.now()

        try:
            # Resolve BOM requirements
            requirements_df = self.bom_resolver.merge_orders_with_bom(
                orders_df, bom_df
            )

            # Validate BOM coverage
            coverage = self.bom_resolver.validate_bom_coverage(
                orders_df, bom_df
            )

            # Check yarn availability if provided
            shortage_analysis = None
            if yarn_df is not None:
                shortage_analysis = self._analyze_requirement_shortages(
                    requirements_df, yarn_df
                )

            result = {
                'status': 'success',
                'total_orders': len(orders_df),
                'bom_coverage': coverage,
                'requirements': requirements_df,
                'shortage_analysis': shortage_analysis
            }

            self._log_execution('resolve_order_requirements', 'success', start_time)
            return result

        except Exception as e:
            logger.exception(f"Order requirements resolution failed: {e}")
            self._log_error('resolve_order_requirements', str(e))
            return {
                'status': 'error',
                'error': str(e)
            }

    def process_knit_orders(
        self,
        use_api: bool = True
    ) -> Dict[str, Any]:
        """Process knit orders with intelligent machine assignment.

        Args:
            use_api: Whether to use API for data fetching

        Returns:
            Processing results with machine assignments
        """
        start_time = datetime.now()

        try:
            # Process knit orders
            results = self.knit_order_processor.process_knit_orders()

            # Get summary
            summary = self.knit_order_processor.get_processing_summary()

            # Combine results
            result = {
                'status': results['status'],
                'message': results['message'],
                'stats': results['stats'],
                'summary': summary,
                'machine_workloads': results.get('machine_workloads', {}),
                'total_machines': results.get('total_machines', 0),
                'total_workload_lbs': results.get('total_workload_lbs', 0)
            }

            self._log_execution('process_knit_orders', 'success', start_time)
            return result

        except Exception as e:
            logger.exception(f"Knit order processing failed: {e}")
            self._log_error('process_knit_orders', str(e))
            return {
                'status': 'error',
                'error': str(e)
            }

    def suggest_machine_assignments(
        self,
        style: str,
        quantity_lbs: float
    ) -> List[Dict[str, Any]]:
        """Suggest machine assignments for a style and quantity.

        Args:
            style: Style code
            quantity_lbs: Quantity in pounds

        Returns:
            List of machine suggestions
        """
        try:
            suggestions = self.knit_order_processor.suggest_machine_assignments(
                style, quantity_lbs
            )
            return suggestions

        except Exception as e:
            logger.error(f"Failed to suggest machine assignments: {e}")
            return []

    def _analyze_requirement_shortages(
        self,
        requirements_df: pd.DataFrame,
        yarn_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze shortages for yarn requirements.

        Args:
            requirements_df: Yarn requirements
            yarn_df: Yarn inventory

        Returns:
            Shortage analysis
        """
        # Group requirements by yarn
        yarn_totals = requirements_df.groupby('YarnID')['Total_Yarn_Required'].sum()

        shortages = []
        for yarn_id, required in yarn_totals.items():
            # Get yarn inventory
            yarn_row = yarn_df[yarn_df['YarnID'] == yarn_id]

            if yarn_row.empty:
                shortages.append({
                    'yarn_id': yarn_id,
                    'required': required,
                    'available': 0,
                    'shortage': required,
                    'status': 'NOT_FOUND'
                })
                continue

            # Calculate availability
            on_hand = yarn_row['On Hand'].iloc[0]
            on_order = yarn_row['On Order'].iloc[0]
            allocated = yarn_row['Allocated'].iloc[0]

            available = on_hand + on_order - allocated
            shortage = max(0, required - available)

            if shortage > 0:
                shortages.append({
                    'yarn_id': yarn_id,
                    'required': required,
                    'available': available,
                    'shortage': shortage,
                    'status': 'SHORTAGE'
                })

        return {
            'total_yarns_required': len(yarn_totals),
            'yarns_with_shortage': len(shortages),
            'total_shortage_amount': sum(s['shortage'] for s in shortages),
            'shortages': shortages[:10]  # Top 10
        }

    def validate_all(
        self,
        yarn_df: Optional[pd.DataFrame] = None,
        machine_df: Optional[pd.DataFrame] = None,
        orders_df: Optional[pd.DataFrame] = None,
        bom_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Run all validations.

        Args:
            yarn_df: Yarn inventory
            machine_df: Machine mappings
            orders_df: Production orders
            bom_df: Bill of Materials

        Returns:
            Comprehensive validation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {}
        }

        # Validate yarn inventory
        if yarn_df is not None:
            results['validations']['yarn_inventory'] = self.validate_yarn_inventory(
                yarn_df
            )

        # Validate machine assignments
        if machine_df is not None:
            results['validations']['machine_assignments'] = self.validate_machine_assignments(
                machine_df
            )

        # Resolve order requirements
        if orders_df is not None and bom_df is not None:
            results['validations']['order_requirements'] = self.resolve_order_requirements(
                orders_df, bom_df, yarn_df
            )

        # Add execution summary
        results['summary'] = {
            'total_validations': len(results['validations']),
            'successful': sum(
                1 for v in results['validations'].values()
                if v.get('status') == 'success'
            ),
            'errors': sum(
                1 for v in results['validations'].values()
                if v.get('status') == 'error'
            )
        }

        return results

    def _log_execution(
        self,
        operation: str,
        status: str,
        start_time: datetime
    ) -> None:
        """Log operation execution."""
        duration = (datetime.now() - start_time).total_seconds()

        self.execution_log.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'status': status,
            'duration_seconds': duration
        })

        logger.info(f"{operation} completed in {duration:.2f}s")

    def _log_error(self, operation: str, error: str) -> None:
        """Log operation error."""
        self.error_log.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'error': error
        })

        logger.error(f"{operation} failed: {error}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions.

        Returns:
            Execution statistics
        """
        if not self.execution_log:
            return {'total_executions': 0}

        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e['status'] == 'success')

        avg_duration = sum(
            e['duration_seconds'] for e in self.execution_log
        ) / total if total > 0 else 0

        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'avg_duration_seconds': avg_duration,
            'total_errors': len(self.error_log)
        }


if __name__ == "__main__":
    """Validation of rules engine."""

    engine = BeverlyKnitsRulesEngine()

    # Create sample data
    sample_yarn = pd.DataFrame([
        {'YarnID': 'Y001', 'On Hand': 100, 'On Order': 50, 'Allocated': 200},
        {'YarnID': 'Y002', 'On Hand': 500, 'On Order': 0, 'Allocated': 300}
    ])

    sample_machines = pd.DataFrame([
        {'WC': '1.30.20.F', 'MACH': '161'},
        {'WC': '1.30.20.M', 'MACH': '210'},
        {'WC': 'INVALID', 'MACH': '999'}
    ])

    # Test yarn validation
    yarn_result = engine.validate_yarn_inventory(sample_yarn)
    assert yarn_result['status'] == 'success'
    assert yarn_result['shortages']['total_shortages'] == 1
    logger.info("Yarn validation test passed")

    # Test machine validation
    machine_result = engine.validate_machine_assignments(sample_machines)
    assert machine_result['status'] == 'success'
    assert machine_result['invalid_count'] == 1
    logger.info("Machine validation test passed")

    # Get execution summary
    summary = engine.get_execution_summary()
    assert summary['total_executions'] == 2
    assert summary['successful'] == 2
    logger.info(f"Execution summary: {summary}")

    print("All validations passed!")