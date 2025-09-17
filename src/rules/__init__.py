"""Beverly Knits ERP Rules Engine.

This module provides comprehensive business rule validation and enforcement
for the Beverly Knits ERP system, ensuring data consistency and business
logic compliance across all operations.

Key Features:
- Planning balance with shortage detection (negative = future shortage)
- Work center pattern validation (TYPE.DIAMETER.MANUFACTURER.CUT)
- BOM style/variant resolution
- Machine assignment validation with capacity checking
- Column name resolution for data consistency
- API caching with file change detection
- eFab integration with retry logic
- Yarn substitution recommendations
"""

from .planning_balance_manager import PlanningBalanceAnalyzer, PlanningBalanceResult
from .work_center_validator import WorkCenterValidator
from .bom_variant_resolver import BOMVariantResolver, StyleVariant
from .machine_assignment_validator import MachineAssignmentValidator, MachineAssignment
from .column_resolver import ColumnResolver
from .cache_manager import CacheManager
from .efab_integration import eFabIntegration
from .yarn_substitution import YarnSubstitutionEngine, YarnSubstitute
from .knit_order_processor import KnitOrderProcessor, KnitOrderAssignment
from .rules_engine import BeverlyKnitsRulesEngine

__all__ = [
    # Planning Balance
    'PlanningBalanceAnalyzer',
    'PlanningBalanceResult',

    # Work Center
    'WorkCenterValidator',

    # BOM
    'BOMVariantResolver',
    'StyleVariant',

    # Machine Assignment
    'MachineAssignmentValidator',
    'MachineAssignment',

    # Column Resolution
    'ColumnResolver',

    # Caching
    'CacheManager',

    # eFab Integration
    'eFabIntegration',

    # Yarn Substitution
    'YarnSubstitutionEngine',
    'YarnSubstitute',

    # Knit Order Processing
    'KnitOrderProcessor',
    'KnitOrderAssignment',

    # Master Engine
    'BeverlyKnitsRulesEngine',
]

__version__ = '2.0.0'