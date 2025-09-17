"""Knit Order Processor with API-based Machine Assignment.

Processes knit orders from API endpoints with intelligent machine assignment,
filtering completed orders, and distributing unassigned workloads.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np

from .efab_integration import eFabIntegration
from .cache_manager import CacheManager
from .machine_assignment_validator import MachineAssignment

logger = logging.getLogger(__name__)


@dataclass
class KnitOrderAssignment:
    """Container for knit order machine assignment."""

    order_id: str
    style: str
    machine_id: str
    balance_lbs: float
    g00_lbs: float
    shipped_lbs: float
    order_status: str  # "Complete", "In Production", or "Active"
    is_assigned: bool
    is_suggested: bool
    assignment_reason: str


class KnitOrderProcessor:
    """Processes knit orders from API with intelligent machine assignment.

    Features:
    - API-based data fetching with caching
    - Filters completed orders (balance <= 0)
    - Intelligent machine inheritance for same styles
    - Equal distribution of unassigned orders
    - Machine utilization calculation
    """

    # Configuration
    DEFAULT_DAILY_CAPACITY = 616.0  # lbs per day
    UTILIZATION_DAYS = 5  # Days for 100% utilization
    API_ENDPOINT = "/api/knitorder/list"  # eFab knit order endpoint
    LOCAL_API_ENDPOINT = "/api/knit-orders"  # Local ERP endpoint (fallback)

    def __init__(
        self,
        efab_integration: Optional[eFabIntegration] = None,
        cache_manager: Optional[CacheManager] = None,
        api_base_url: Optional[str] = None,
        use_efab_api: bool = True
    ) -> None:
        """Initialize knit order processor.

        Args:
            efab_integration: eFab API integration instance
            cache_manager: Cache manager instance
            api_base_url: Base URL for API calls
            use_efab_api: Whether to use eFab API (True) or local API (False)
        """
        self.efab = efab_integration or eFabIntegration()
        self.cache = cache_manager or CacheManager()
        self.api_base_url = api_base_url or "http://localhost:5006"
        self.use_efab_api = use_efab_api

        self.machine_workloads: Dict[str, float] = {}
        self.machine_assignments: Dict[str, str] = {}
        self.suggested_workloads: Dict[str, float] = {}
        self.order_assignments: List[KnitOrderAssignment] = []
        self.processing_stats = {
            'total_orders': 0,
            'completed_filtered': 0,
            'in_production_count': 0,
            'active_count': 0,
            'assigned_orders': 0,
            'unassigned_orders': 0,
            'suggested_assignments': 0
        }

        logger.info("KnitOrderProcessor initialized")

    def fetch_knit_orders_from_api(self) -> pd.DataFrame:
        """Fetch knit orders from API.

        Returns:
            DataFrame containing knit orders
        """
        try:
            # Try cache first
            cache_key = self.cache.cache_key("knit_orders_api")
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                logger.debug("Using cached knit orders data")
                return pd.DataFrame(cached_data)

            # Determine which API to use
            if self.use_efab_api:
                # Use eFab API directly
                logger.info(f"Fetching knit orders from eFab API: {self.API_ENDPOINT}")
                response = self.efab.fetch_with_retry(self.API_ENDPOINT)
            else:
                # Use local API endpoint as fallback
                import requests
                endpoint = self.LOCAL_API_ENDPOINT
                url = f"{self.api_base_url}{endpoint}"
                logger.info(f"Fetching knit orders from local API: {url}")
                response = requests.get(url, timeout=30).json()

            # Handle different response formats
            if response:
                if isinstance(response, list):
                    # Direct list of orders from eFab API
                    orders_data = response
                elif isinstance(response, dict):
                    # Wrapped response from local API
                    if 'data' in response:
                        orders_data = response['data']
                    elif 'orders' in response:
                        orders_data = response['orders']
                    elif 'results' in response:
                        orders_data = response['results']
                    else:
                        # Assume the response itself is the data
                        orders_data = [response] if response else []
                else:
                    orders_data = []
            else:
                orders_data = []

            # Convert to DataFrame
            df = pd.DataFrame(orders_data)

            # Standardize column names if coming from eFab API
            if self.use_efab_api and not df.empty:
                # Map eFab column names to expected names
                column_mapping = {
                    'order_number': 'Order #',
                    'style_number': 'Style #',
                    'machine_id': 'Machine',
                    'balance': 'Balance (lbs)',
                    'g00': 'G00 (lbs)',
                    'shipped': 'Shipped (lbs)',
                    'qty_ordered': 'Qty Ordered (lbs)',
                    'start_date': 'Start Date',
                    'quoted_date': 'Quoted Date'
                }
                df = df.rename(columns=column_mapping)

            # Cache the data
            self.cache.set(cache_key, df.to_dict('records'), ttl=300)  # 5 min cache

            logger.info(f"Fetched {len(df)} knit orders from {'eFab' if self.use_efab_api else 'local'} API")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch knit orders from API: {e}")

            # Try fallback to cached data with longer TTL
            cached_data = self.cache.get(cache_key, ttl=3600)  # 1 hour fallback
            if cached_data is not None:
                logger.warning("Using fallback cached data")
                return pd.DataFrame(cached_data)

            # Return empty DataFrame if all fails
            return pd.DataFrame()

    def categorize_order_status(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Categorize orders by status and filter out completed ones.

        Status Rules:
        - Complete: Balance <= 0
        - In Production: G00 + Shipped > 0
        - Active: All others

        Args:
            orders: DataFrame of knit orders

        Returns:
            DataFrame with status column and only non-complete orders
        """
        if orders.empty:
            return orders

        # Identify columns
        balance_col = None
        for col in ['Balance (lbs)', 'Balance_lbs', 'balance_lbs', 'Balance']:
            if col in orders.columns:
                balance_col = col
                break

        g00_col = None
        for col in ['G00 (lbs)', 'G00_lbs', 'g00_lbs', 'G00']:
            if col in orders.columns:
                g00_col = col
                break

        shipped_col = None
        for col in ['Shipped (lbs)', 'Shipped_lbs', 'shipped_lbs', 'Shipped']:
            if col in orders.columns:
                shipped_col = col
                break

        if balance_col is None:
            logger.warning("No balance column found in orders")
            return orders

        # Clean values helper
        def clean_value(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, str):
                return float(val.replace(',', ''))
            return float(val)

        # Add cleaned columns
        orders['_balance_clean'] = orders[balance_col].apply(clean_value)
        orders['_g00_clean'] = orders[g00_col].apply(clean_value) if g00_col else 0
        orders['_shipped_clean'] = orders[shipped_col].apply(clean_value) if shipped_col else 0

        # Determine order status
        # Priority: 1) Complete (balance <= 0), 2) In Production (G00+Shipped > 0), 3) Active
        def determine_status(row):
            balance = row['_balance_clean']
            g00 = row['_g00_clean']
            shipped = row['_shipped_clean']

            # Complete takes priority - if balance <= 0, order is complete
            if balance <= 0:
                return "Complete"
            # Only check production status if not complete
            elif g00 + shipped > 0:
                return "In Production"
            else:
                return "Active"

        orders['order_status'] = orders.apply(determine_status, axis=1)

        # Count by status
        status_counts = orders['order_status'].value_counts()
        completed_count = status_counts.get('Complete', 0)
        in_production_count = status_counts.get('In Production', 0)
        active_count = status_counts.get('Active', 0)

        # Filter out completed orders but keep their count
        initial_count = len(orders)
        active_orders = orders[orders['order_status'] != 'Complete'].copy()

        # Update stats
        self.processing_stats['total_orders'] = initial_count
        self.processing_stats['completed_filtered'] = completed_count
        self.processing_stats['in_production_count'] = in_production_count
        self.processing_stats['active_count'] = active_count

        logger.info(f"Order Status: {completed_count} complete, {in_production_count} in production, "
                   f"{active_count} active")

        return active_orders

    def assign_machines_intelligently(
        self,
        orders: pd.DataFrame
    ) -> Dict[str, List[KnitOrderAssignment]]:
        """Assign machines intelligently based on style patterns.

        Implements:
        1. Orders of same style with some machines inherit those machines
        2. Multiple machines for a style distribute unassigned orders equally

        Args:
            orders: DataFrame of active knit orders

        Returns:
            Dictionary of machine_id -> list of assignments
        """
        if orders.empty:
            return {}

        assignments = defaultdict(list)
        style_machines = defaultdict(set)
        style_orders = defaultdict(list)

        # Identify columns
        style_col = None
        for col in ['Style #', 'Style', 'style']:
            if col in orders.columns:
                style_col = col
                break

        machine_col = None
        for col in ['Machine', 'machine', 'Machine_ID']:
            if col in orders.columns:
                machine_col = col
                break

        balance_col = None
        for col in ['Balance (lbs)', 'Balance_lbs', 'balance_lbs']:
            if col in orders.columns:
                balance_col = col
                break

        g00_col = None
        for col in ['G00 (lbs)', 'G00_lbs', 'g00_lbs', 'G00']:
            if col in orders.columns:
                g00_col = col
                break

        shipped_col = None
        for col in ['Shipped (lbs)', 'Shipped_lbs', 'shipped_lbs', 'Shipped']:
            if col in orders.columns:
                shipped_col = col
                break

        if not all([style_col, machine_col, balance_col]):
            logger.error(f"Missing required columns. Found: style={style_col}, "
                        f"machine={machine_col}, balance={balance_col}")
            return {}

        # First pass: collect machine assignments by style
        for idx, row in orders.iterrows():
            style = row.get(style_col)
            if pd.notna(style):
                style_orders[str(style)].append(row)
                machine_id = row.get(machine_col)
                if pd.notna(machine_id):
                    style_machines[str(style)].add(str(int(machine_id)))

        # Second pass: process orders with intelligent assignment
        for style, orders_list in style_orders.items():
            machines = list(style_machines[style])

            # Separate orders with and without machines
            orders_with_machine = []
            orders_without_machine = []

            for order in orders_list:
                if pd.notna(order.get(machine_col)):
                    orders_with_machine.append(order)
                else:
                    orders_without_machine.append(order)

            # Process orders that already have machines
            for order in orders_with_machine:
                machine_id = str(int(order[machine_col]))
                balance = self._clean_balance_value(order[balance_col])
                g00 = self._clean_balance_value(order.get(g00_col, 0)) if g00_col else 0
                shipped = self._clean_balance_value(order.get(shipped_col, 0)) if shipped_col else 0
                order_status = order.get('order_status', 'Active')

                # Only process non-complete orders
                if order_status != 'Complete' and balance > 0:
                    # Create assignment
                    assignment = KnitOrderAssignment(
                        order_id=str(order.get('Order #', order.name)),
                        style=str(style),
                        machine_id=machine_id,
                        balance_lbs=balance,
                        g00_lbs=g00,
                        shipped_lbs=shipped,
                        order_status=order_status,
                        is_assigned=True,
                        is_suggested=False,
                        assignment_reason="Original assignment"
                    )

                    assignments[machine_id].append(assignment)
                    self.order_assignments.append(assignment)

                    # Update workloads
                    self.machine_workloads[machine_id] = \
                        self.machine_workloads.get(machine_id, 0) + balance
                    self.machine_assignments[machine_id] = str(style)

            # Process orders without machines - distribute across known machines
            if orders_without_machine and machines:
                # Calculate total unassigned workload (excluding complete orders)
                unassigned_workload = sum(
                    self._clean_balance_value(order[balance_col])
                    for order in orders_without_machine
                    if order.get('order_status', 'Active') != 'Complete' and
                       self._clean_balance_value(order[balance_col]) > 0
                )

                if unassigned_workload > 0:
                    # Distribute equally across machines
                    workload_per_machine = unassigned_workload / len(machines)

                    for order in orders_without_machine:
                        balance = self._clean_balance_value(order[balance_col])
                        g00 = self._clean_balance_value(order.get(g00_col, 0)) if g00_col else 0
                        shipped = self._clean_balance_value(order.get(shipped_col, 0)) if shipped_col else 0
                        order_status = order.get('order_status', 'Active')

                        if order_status != 'Complete' and balance > 0:
                            # Assign proportionally to each machine
                            order_share = balance / unassigned_workload

                            for machine_id in machines:
                                machine_balance = workload_per_machine * order_share

                                assignment = KnitOrderAssignment(
                                    order_id=str(order.get('Order #', order.name)),
                                    style=str(style),
                                    machine_id=machine_id,
                                    balance_lbs=machine_balance,
                                    g00_lbs=g00,
                                    shipped_lbs=shipped,
                                    order_status=order_status,
                                    is_assigned=False,
                                    is_suggested=True,
                                    assignment_reason=f"Distributed from {len(machines)} machines"
                                )

                                assignments[machine_id].append(assignment)
                                self.order_assignments.append(assignment)

                                # Update suggested workloads
                                self.suggested_workloads[machine_id] = \
                                    self.suggested_workloads.get(machine_id, 0) + machine_balance

                                # Also add to main workload for utilization
                                self.machine_workloads[machine_id] = \
                                    self.machine_workloads.get(machine_id, 0) + machine_balance
                                self.machine_assignments[machine_id] = str(style)

                    logger.info(f"Distributed {unassigned_workload:.0f} lbs for style "
                              f"{style} across {len(machines)} machines")

        # Update stats
        self.processing_stats['assigned_orders'] = sum(
            1 for a in self.order_assignments if a.is_assigned
        )
        self.processing_stats['unassigned_orders'] = sum(
            1 for a in self.order_assignments if not a.is_assigned
        )
        self.processing_stats['suggested_assignments'] = sum(
            1 for a in self.order_assignments if a.is_suggested
        )

        return dict(assignments)

    def _clean_balance_value(self, val: Any) -> float:
        """Clean balance value to float.

        Args:
            val: Raw balance value

        Returns:
            Cleaned float value
        """
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.replace(',', ''))
        return float(val)

    def calculate_machine_workloads(self) -> Dict[str, Dict[str, float]]:
        """Calculate machine workloads and utilization.

        Returns:
            Dictionary with machine workload details
        """
        workloads = {}

        for machine_id, total_lbs in self.machine_workloads.items():
            # Calculate utilization
            days_of_work = total_lbs / self.DEFAULT_DAILY_CAPACITY
            utilization = min(100.0, (days_of_work / self.UTILIZATION_DAYS) * 100.0)

            workloads[machine_id] = {
                'total_lbs': total_lbs,
                'assigned_lbs': total_lbs - self.suggested_workloads.get(machine_id, 0),
                'suggested_lbs': self.suggested_workloads.get(machine_id, 0),
                'days_of_work': days_of_work,
                'utilization_percent': utilization,
                'assigned_style': self.machine_assignments.get(machine_id, ''),
                'is_overloaded': utilization > 85.0
            }

        return workloads

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
            List of machine suggestions with scores
        """
        suggestions = []

        # Find machines already assigned to this style
        style_machines = [
            m for m, s in self.machine_assignments.items()
            if s == style
        ]

        if style_machines:
            # Prefer machines already working on this style
            for machine_id in style_machines:
                workload = self.machine_workloads.get(machine_id, 0)
                days_of_work = workload / self.DEFAULT_DAILY_CAPACITY
                utilization = (days_of_work / self.UTILIZATION_DAYS) * 100.0

                suggestions.append({
                    'machine_id': machine_id,
                    'current_utilization': utilization,
                    'available_capacity': max(0, self.DEFAULT_DAILY_CAPACITY *
                                            self.UTILIZATION_DAYS - workload),
                    'priority_score': 100 - abs(utilization - 50),  # Prefer 50% util
                    'reason': 'Already assigned to this style'
                })
        else:
            # Find machines with lowest utilization
            for machine_id, workload in self.machine_workloads.items():
                days_of_work = workload / self.DEFAULT_DAILY_CAPACITY
                utilization = (days_of_work / self.UTILIZATION_DAYS) * 100.0

                if utilization < 85:  # Not overloaded
                    suggestions.append({
                        'machine_id': machine_id,
                        'current_utilization': utilization,
                        'available_capacity': max(0, self.DEFAULT_DAILY_CAPACITY *
                                                self.UTILIZATION_DAYS - workload),
                        'priority_score': 100 - utilization,  # Prefer lower util
                        'reason': 'Available capacity'
                    })

        # Sort by priority score
        suggestions.sort(key=lambda x: x['priority_score'], reverse=True)

        return suggestions[:5]  # Top 5 suggestions

    def process_knit_orders(self) -> Dict[str, Any]:
        """Main processing method - fetch, filter, and assign.

        Returns:
            Processing results summary
        """
        logger.info("Starting knit order processing")

        # Reset state
        self.machine_workloads.clear()
        self.machine_assignments.clear()
        self.suggested_workloads.clear()
        self.order_assignments.clear()

        # Fetch orders from API
        orders = self.fetch_knit_orders_from_api()

        if orders.empty:
            logger.warning("No orders fetched from API")
            return {
                'status': 'no_data',
                'message': 'No knit orders available',
                'stats': self.processing_stats
            }

        # Categorize orders by status and filter completed ones
        active_orders = self.categorize_order_status(orders)

        if active_orders.empty:
            logger.info("All orders are completed")
            return {
                'status': 'all_completed',
                'message': 'All knit orders are completed',
                'stats': self.processing_stats
            }

        # Assign machines intelligently
        assignments = self.assign_machines_intelligently(active_orders)

        # Calculate workloads
        workloads = self.calculate_machine_workloads()

        return {
            'status': 'success',
            'message': f'Processed {len(active_orders)} active orders',
            'stats': self.processing_stats,
            'machine_assignments': assignments,
            'machine_workloads': workloads,
            'total_machines': len(self.machine_workloads),
            'total_workload_lbs': sum(self.machine_workloads.values())
        }

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of last processing run.

        Returns:
            Processing summary with statistics
        """
        # Calculate production statistics
        in_production_lbs = sum(
            a.balance_lbs for a in self.order_assignments
            if a.order_status == 'In Production'
        )
        active_lbs = sum(
            a.balance_lbs for a in self.order_assignments
            if a.order_status == 'Active'
        )

        return {
            'processing_stats': self.processing_stats,
            'machine_count': len(self.machine_workloads),
            'total_workload_lbs': sum(self.machine_workloads.values()),
            'assigned_workload_lbs': sum(self.machine_workloads.values()) -
                                   sum(self.suggested_workloads.values()),
            'suggested_workload_lbs': sum(self.suggested_workloads.values()),
            'in_production_lbs': in_production_lbs,
            'active_lbs': active_lbs,
            'order_status_breakdown': {
                'completed': self.processing_stats.get('completed_filtered', 0),
                'in_production': self.processing_stats.get('in_production_count', 0),
                'active': self.processing_stats.get('active_count', 0)
            },
            'average_utilization': np.mean([
                min(100, (w / self.DEFAULT_DAILY_CAPACITY / self.UTILIZATION_DAYS) * 100)
                for w in self.machine_workloads.values()
            ]) if self.machine_workloads else 0,
            'overloaded_machines': sum(
                1 for w in self.machine_workloads.values()
                if (w / self.DEFAULT_DAILY_CAPACITY / self.UTILIZATION_DAYS) * 100 > 85
            ),
            'styles_processed': len(set(self.machine_assignments.values()))
        }


if __name__ == "__main__":
    """Validation of knit order processor."""
    import os

    # Create processor with eFab API
    print("Testing Knit Order Processor with eFab API")
    print("-" * 50)

    # Check if eFab session is available
    efab_session = os.getenv('EFAB_SESSION')
    if efab_session:
        print(f"eFab session cookie: Set ({len(efab_session)} chars)")
        use_efab = True
    else:
        print("eFab session cookie: Not set, using local API")
        use_efab = False

    # Create processor
    processor = KnitOrderProcessor(use_efab_api=use_efab)

    # Test Case 1: Process orders
    print("\nTest 1: Processing knit orders...")
    results = processor.process_knit_orders()
    print(f"  Status: {results['status']}")
    print(f"  Stats: {results['stats']}")

    # Test Case 2: Get summary
    print("\nTest 2: Getting processing summary...")
    summary = processor.get_processing_summary()
    print(f"  Machine count: {summary['machine_count']}")
    print(f"  Total workload: {summary['total_workload_lbs']:.0f} lbs")
    print(f"  Order status breakdown: {summary.get('order_status_breakdown', {})}")

    # Test Case 3: Suggest machines for a style
    if processor.machine_assignments:
        test_style = list(processor.machine_assignments.values())[0]
        print(f"\nTest 3: Suggesting machines for style {test_style}...")
        suggestions = processor.suggest_machine_assignments(test_style, 1000)
        print(f"  Found {len(suggestions)} machine suggestions")
        for i, sugg in enumerate(suggestions[:3], 1):
            print(f"    {i}. Machine {sugg['machine_id']}: {sugg['reason']}")

    print("\n" + "=" * 50)
    print("Validation complete!")