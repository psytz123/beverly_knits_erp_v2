"""Machine Assignment Validator for Beverly Knits ERP.

Validates machine assignments against work center patterns and capacity,
suggesting alternatives when primary machines are unavailable.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MachineAssignment:
    """Container for machine assignment information."""

    order_id: str
    machine_id: str
    work_center: str
    style: str
    quantity: float
    start_time: datetime
    end_time: datetime
    utilization: float
    is_valid: bool
    validation_errors: List[str]


class MachineAssignmentValidator:
    """Validates and optimizes machine assignments for production orders.

    Ensures machines are properly assigned to compatible work centers
    and have available capacity.
    """

    # Machine capacity configuration
    CAPACITY_CONFIG = {
        'shifts_per_day': 3,
        'hours_per_shift': 8,
        'efficiency_factor': 0.85,
        'maintenance_hours_per_day': 2,
        'max_utilization': 0.85
    }

    # Production rates (units per hour by machine type)
    PRODUCTION_RATES = {
        '1': 50,   # Single Jersey
        '2': 45,   # Double Jersey
        '3': 40,   # Rib
        '4': 35,   # Interlock
        '5': 30,   # Fleece
        '6': 25,   # Terry
        '7': 20,   # Jacquard
        '8': 35,   # Special Knit
        '9': 40    # Warp Knit
    }

    def __init__(self) -> None:
        """Initialize validator."""
        self.machine_schedule: Dict[str, List[MachineAssignment]] = {}
        self.validation_log: List[Dict[str, Any]] = []
        logger.info("MachineAssignmentValidator initialized")

    def validate_assignment(
        self,
        order_id: str,
        machine_id: str,
        style: str,
        quantity: float,
        machine_mapping: pd.DataFrame,
        work_center_validator: Optional[Any] = None
    ) -> Tuple[bool, List[str]]:
        """Validate a machine assignment.

        Args:
            order_id: Production order ID
            machine_id: Machine to validate
            style: Style being produced
            quantity: Production quantity
            machine_mapping: Machine to work center mapping
            work_center_validator: Optional work center validator

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check machine exists
        if machine_id not in machine_mapping['MACH'].values:
            errors.append(f"Machine {machine_id} not found in system")
            return False, errors

        # Get work center
        machine_row = machine_mapping[machine_mapping['MACH'] == machine_id]
        if machine_row.empty:
            errors.append(f"No work center mapping for machine {machine_id}")
            return False, errors

        work_center = machine_row['WC'].iloc[0]

        # Validate work center format if validator provided
        if work_center_validator:
            valid, components = work_center_validator.validate_work_center(work_center)
            if not valid:
                errors.append(f"Invalid work center format: {work_center}")

        # Check capacity
        if not self._check_capacity(machine_id, quantity):
            errors.append(f"Machine {machine_id} at capacity")

        # Check style compatibility
        if not self._check_style_compatibility(style, work_center):
            errors.append(f"Style {style} incompatible with work center {work_center}")

        # Log validation
        self.validation_log.append({
            'timestamp': datetime.now(),
            'order_id': order_id,
            'machine_id': machine_id,
            'valid': len(errors) == 0,
            'errors': errors
        })

        return len(errors) == 0, errors

    def suggest_alternative_machines(
        self,
        style: str,
        quantity: float,
        machine_mapping: pd.DataFrame,
        preferred_work_center: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Suggest alternative machines for an order.

        Args:
            style: Style to produce
            quantity: Production quantity
            machine_mapping: Machine mapping DataFrame
            preferred_work_center: Preferred work center pattern

        Returns:
            List of alternative machine suggestions
        """
        suggestions = []

        for _, row in machine_mapping.iterrows():
            machine_id = str(row['MACH'])
            work_center = row['WC']

            # Filter by preferred work center if specified
            if preferred_work_center and not work_center.startswith(preferred_work_center):
                continue

            # Check compatibility
            if not self._check_style_compatibility(style, work_center):
                continue

            # Check capacity
            if not self._check_capacity(machine_id, quantity):
                continue

            # Calculate utilization
            current_util = self._get_machine_utilization(machine_id)
            available_capacity = self._get_available_capacity(machine_id)

            suggestions.append({
                'machine_id': machine_id,
                'work_center': work_center,
                'current_utilization': current_util,
                'available_capacity': available_capacity,
                'estimated_hours': self._estimate_production_time(quantity, work_center),
                'priority': self._calculate_priority(current_util, work_center)
            })

        # Sort by priority
        suggestions.sort(key=lambda x: x['priority'], reverse=True)

        return suggestions[:5]  # Top 5 suggestions

    def _check_capacity(
        self,
        machine_id: str,
        quantity: float
    ) -> bool:
        """Check if machine has capacity.

        Args:
            machine_id: Machine ID
            quantity: Required capacity

        Returns:
            True if capacity available
        """
        current_util = self._get_machine_utilization(machine_id)
        max_util = self.CAPACITY_CONFIG['max_utilization']

        return current_util < max_util

    def _check_style_compatibility(
        self,
        style: str,
        work_center: str
    ) -> bool:
        """Check style compatibility with work center.

        Args:
            style: Style code
            work_center: Work center code

        Returns:
            True if compatible
        """
        # Extract machine type from work center (first digit)
        if work_center and len(work_center) > 0:
            machine_type = work_center[0]

            # Simple compatibility check (can be enhanced)
            # Certain styles work better on certain machine types
            style_prefix = style[:2] if len(style) >= 2 else ''

            compatibility_map = {
                'ST': ['1', '2', '3'],  # Standard styles
                'FL': ['5'],            # Fleece styles
                'JQ': ['7'],            # Jacquard styles
                'TR': ['6'],            # Terry styles
            }

            if style_prefix in compatibility_map:
                return machine_type in compatibility_map[style_prefix]

        # Default: compatible
        return True

    def _get_machine_utilization(self, machine_id: str) -> float:
        """Get current machine utilization.

        Args:
            machine_id: Machine ID

        Returns:
            Utilization percentage (0-1)
        """
        if machine_id not in self.machine_schedule:
            return 0.0

        # Calculate total scheduled hours
        total_hours = sum(
            (a.end_time - a.start_time).total_seconds() / 3600
            for a in self.machine_schedule[machine_id]
        )

        # Calculate available hours per day
        available_hours = (
            self.CAPACITY_CONFIG['shifts_per_day'] *
            self.CAPACITY_CONFIG['hours_per_shift'] -
            self.CAPACITY_CONFIG['maintenance_hours_per_day']
        )

        return min(total_hours / available_hours, 1.0)

    def _get_available_capacity(self, machine_id: str) -> float:
        """Get available capacity in hours.

        Args:
            machine_id: Machine ID

        Returns:
            Available hours
        """
        current_util = self._get_machine_utilization(machine_id)
        max_util = self.CAPACITY_CONFIG['max_utilization']

        available_util = max_util - current_util
        available_hours = (
            self.CAPACITY_CONFIG['shifts_per_day'] *
            self.CAPACITY_CONFIG['hours_per_shift'] *
            self.CAPACITY_CONFIG['efficiency_factor']
        )

        return available_hours * available_util

    def _estimate_production_time(
        self,
        quantity: float,
        work_center: str
    ) -> float:
        """Estimate production time.

        Args:
            quantity: Production quantity
            work_center: Work center code

        Returns:
            Estimated hours
        """
        # Get machine type from work center
        machine_type = work_center[0] if work_center else '1'

        # Get production rate
        rate = self.PRODUCTION_RATES.get(machine_type, 40)

        # Calculate time
        return quantity / rate / self.CAPACITY_CONFIG['efficiency_factor']

    def _calculate_priority(
        self,
        utilization: float,
        work_center: str
    ) -> float:
        """Calculate machine priority score.

        Args:
            utilization: Current utilization
            work_center: Work center code

        Returns:
            Priority score (higher is better)
        """
        # Prefer machines with moderate utilization (40-60%)
        util_score = 1.0 - abs(utilization - 0.5) * 2

        # Prefer certain machine types
        machine_type = work_center[0] if work_center else '1'
        type_score = {
            '1': 1.0,  # Single Jersey (most versatile)
            '2': 0.9,  # Double Jersey
            '3': 0.8,  # Rib
            '4': 0.7,  # Interlock
            '5': 0.6,  # Fleece
            '6': 0.5,  # Terry
            '7': 0.4,  # Jacquard
            '8': 0.7,  # Special
            '9': 0.8   # Warp
        }.get(machine_type, 0.5)

        return util_score * 0.6 + type_score * 0.4

    def schedule_order(
        self,
        order_id: str,
        machine_id: str,
        work_center: str,
        style: str,
        quantity: float,
        start_time: Optional[datetime] = None
    ) -> MachineAssignment:
        """Schedule an order on a machine.

        Args:
            order_id: Order ID
            machine_id: Machine ID
            work_center: Work center code
            style: Style code
            quantity: Production quantity
            start_time: Optional start time

        Returns:
            Machine assignment
        """
        if start_time is None:
            start_time = datetime.now()

        # Estimate production time
        production_hours = self._estimate_production_time(quantity, work_center)
        end_time = start_time + timedelta(hours=production_hours)

        # Create assignment
        assignment = MachineAssignment(
            order_id=order_id,
            machine_id=machine_id,
            work_center=work_center,
            style=style,
            quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            utilization=self._get_machine_utilization(machine_id),
            is_valid=True,
            validation_errors=[]
        )

        # Add to schedule
        if machine_id not in self.machine_schedule:
            self.machine_schedule[machine_id] = []
        self.machine_schedule[machine_id].append(assignment)

        logger.info(f"Scheduled order {order_id} on machine {machine_id}")

        return assignment

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary.

        Returns:
            Summary statistics
        """
        if not self.validation_log:
            return {'total_validations': 0}

        total = len(self.validation_log)
        valid = sum(1 for v in self.validation_log if v['valid'])

        return {
            'total_validations': total,
            'valid': valid,
            'invalid': total - valid,
            'validity_rate': (valid / total * 100) if total > 0 else 0,
            'scheduled_machines': len(self.machine_schedule),
            'total_assignments': sum(
                len(assignments) for assignments in self.machine_schedule.values()
            )
        }


if __name__ == "__main__":
    """Validation of machine assignment validator."""

    validator = MachineAssignmentValidator()

    # Create sample machine mapping
    machine_mapping = pd.DataFrame([
        {'WC': '1.30.20.F', 'MACH': '161'},
        {'WC': '1.30.20.M', 'MACH': '210'},
        {'WC': '5.38.18.F', 'MACH': '177'},
        {'WC': '7.26.16.J', 'MACH': '225'}
    ])

    # Test Case 1: Valid assignment
    valid, errors = validator.validate_assignment(
        'ORD001', '161', 'ST123', 1000, machine_mapping
    )
    assert valid is True
    assert len(errors) == 0
    logger.info("Test 1 passed: Valid assignment")

    # Test Case 2: Invalid machine
    valid, errors = validator.validate_assignment(
        'ORD002', '999', 'ST123', 1000, machine_mapping
    )
    assert valid is False
    assert 'not found' in errors[0]
    logger.info("Test 2 passed: Invalid machine detected")

    # Test Case 3: Suggest alternatives
    suggestions = validator.suggest_alternative_machines(
        'ST456', 500, machine_mapping
    )
    assert len(suggestions) > 0
    logger.info(f"Test 3 passed: Found {len(suggestions)} alternatives")

    # Test Case 4: Schedule order
    assignment = validator.schedule_order(
        'ORD003', '161', '1.30.20.F', 'ST789', 2000
    )
    assert assignment.is_valid is True
    assert assignment.machine_id == '161'
    logger.info("Test 4 passed: Order scheduled")

    # Test Case 5: Get summary
    summary = validator.get_validation_summary()
    assert summary['total_validations'] == 2
    logger.info(f"Test 5 passed: Summary = {summary}")

    print("All validations passed!")