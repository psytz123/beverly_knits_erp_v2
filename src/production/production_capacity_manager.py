"""
Production Capacity Manager for Beverly Knits ERP
Manages machine production capacity data and scheduling constraints
Enhanced with machine-level tracking and work center integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

# Import machine mapper for work center and machine tracking
try:
    from .machine_mapper import get_machine_mapper, MachineWorkCenterMapper
except ImportError:
    try:
        from src.production.machine_mapper import get_machine_mapper, MachineWorkCenterMapper
    except ImportError:
        logger.warning("MachineWorkCenterMapper not available - machine-level features disabled")
        get_machine_mapper = None
        MachineWorkCenterMapper = None

logger = logging.getLogger(__name__)

class ProductionCapacityManager:
    """
    Manages production capacity data from Production Lbs.xlsx
    Provides capacity planning and scheduling capabilities
    """
    
    def __init__(self, capacity_file_path: Optional[str] = None):
        """Initialize with production capacity data"""
        self.capacity_data = None
        self.style_capacities = {}
        self.default_capacity = 616  # Average from data analysis
        
        # Machine-level tracking
        self.machine_mapper = None
        self.machine_utilization = {}  # machine_id -> utilization %
        self.machine_assignments = {}  # machine_id -> current_style
        self.work_center_loads = {}    # work_center -> current load
        self.machine_workload_lbs = {}  # machine_id -> remaining work in lbs
        self.machine_suggested_workload_lbs = {}  # machine_id -> suggested/unassigned work in lbs
        self.machine_forecasted_workload_lbs = {}  # machine_id -> forecasted demand in lbs
        
        if capacity_file_path:
            self.load_capacity_data(capacity_file_path)
        else:
            # Default path
            default_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/Production Lbs.xlsx"
            if Path(default_path).exists():
                self.load_capacity_data(default_path)
        
        # Initialize machine mapper if available
        if get_machine_mapper:
            try:
                self.machine_mapper = get_machine_mapper()
                logger.info("Machine mapper initialized successfully")
                # Load machine workloads after mapper is initialized
                self.load_machine_workloads()
            except Exception as e:
                logger.warning(f"Failed to initialize machine mapper: {e}")
    
    def load_machine_workloads(self):
        """Load actual machine workloads from eFab Knit Orders

        Implements smart assignment logic:
        1. If multiple orders of same style and only some have machines,
           unassigned orders inherit the same machine(s)
        2. If multiple machines for a style, distribute unassigned orders equally
        """
        try:
            import pandas as pd
            from pathlib import Path
            from collections import defaultdict

            # Try to load knit orders file
            knit_orders_path = Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/eFab_Knit_Orders.csv")

            if not knit_orders_path.exists():
                logger.warning("Knit orders file not found, using default machine utilization")
                return

            df = pd.read_csv(knit_orders_path)

            # Clear existing assignments
            self.machine_workload_lbs.clear()
            self.machine_suggested_workload_lbs.clear()
            self.machine_assignments.clear()
            self.machine_utilization.clear()

            # First pass: collect machine assignments by style
            style_machines = defaultdict(set)  # style -> set of machine IDs
            style_orders = defaultdict(list)   # style -> list of order rows

            for idx, row in df.iterrows():
                style = row.get('Style #')
                if pd.notna(style):
                    style_orders[style].append(row)
                    machine_id = row.get('Machine')
                    if pd.notna(machine_id):
                        style_machines[style].add(int(machine_id))

            # Second pass: process orders with smart machine assignment
            for style, orders in style_orders.items():
                machines = list(style_machines[style])

                # Separate orders with and without machines
                orders_with_machine = []
                orders_without_machine = []

                for order in orders:
                    if pd.notna(order.get('Machine')):
                        orders_with_machine.append(order)
                    else:
                        orders_without_machine.append(order)

                # Process orders that already have machines
                for order in orders_with_machine:
                    machine_id = str(int(order['Machine']))
                    balance_lbs = order.get('Balance (lbs)')

                    if pd.notna(balance_lbs):
                        # Clean balance value (remove commas)
                        if isinstance(balance_lbs, str):
                            balance_lbs = float(balance_lbs.replace(',', ''))

                        # Skip completed orders (balance <= 0)
                        if balance_lbs <= 0:
                            continue

                        # Accumulate workload
                        if machine_id in self.machine_workload_lbs:
                            self.machine_workload_lbs[machine_id] += balance_lbs
                        else:
                            self.machine_workload_lbs[machine_id] = balance_lbs

                        # Assign style
                        self.machine_assignments[machine_id] = str(style)

                # Process orders without machines - distribute across known machines for this style
                if orders_without_machine and machines:
                    # Calculate total unassigned workload (excluding completed orders)
                    unassigned_workload = 0
                    for order in orders_without_machine:
                        balance_lbs = order.get('Balance (lbs)')
                        if pd.notna(balance_lbs):
                            if isinstance(balance_lbs, str):
                                balance_lbs = float(balance_lbs.replace(',', ''))
                            # Only add if not completed (balance > 0)
                            if balance_lbs > 0:
                                unassigned_workload += balance_lbs

                    # Distribute equally across machines
                    if unassigned_workload > 0:
                        workload_per_machine = unassigned_workload / len(machines)
                        for machine_id in machines:
                            machine_id_str = str(machine_id)

                            # Add to suggested workload (to distinguish from assigned)
                            if machine_id_str in self.machine_suggested_workload_lbs:
                                self.machine_suggested_workload_lbs[machine_id_str] += workload_per_machine
                            else:
                                self.machine_suggested_workload_lbs[machine_id_str] = workload_per_machine

                            # Also add to main workload for utilization calculation
                            if machine_id_str in self.machine_workload_lbs:
                                self.machine_workload_lbs[machine_id_str] += workload_per_machine
                            else:
                                self.machine_workload_lbs[machine_id_str] = workload_per_machine

                            # Ensure style assignment
                            self.machine_assignments[machine_id_str] = str(style)

                        logger.info(f"Distributed {unassigned_workload:.0f} lbs for style {style} across {len(machines)} machines")

            # Calculate utilization based on total workload
            for machine_id, workload_lbs in self.machine_workload_lbs.items():
                # Assume machine capacity is ~616 lbs/day and calculate utilization
                daily_capacity = self.default_capacity
                days_of_work = workload_lbs / daily_capacity

                # Convert days of work to utilization percentage (cap at 100%)
                # More than 5 days = 100% utilization
                utilization = min(100.0, (days_of_work / 5.0) * 100.0)
                self.machine_utilization[machine_id] = utilization

            # Log summary
            total_assigned = len([m for m in self.machine_workload_lbs if m not in self.machine_suggested_workload_lbs])
            total_suggested = len(self.machine_suggested_workload_lbs)
            logger.info(f"Loaded workloads for {len(self.machine_workload_lbs)} machines from knit orders")
            logger.info(f"  - {total_assigned} machines with direct assignments")
            logger.info(f"  - {total_suggested} machines with distributed/suggested workload")

        except Exception as e:
            logger.error(f"Error loading machine workloads: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with default behavior
    
    def load_capacity_data(self, file_path: str) -> bool:
        """Load production capacity data from Excel file"""
        try:
            self.capacity_data = pd.read_excel(file_path)
            
            # Build style capacity dictionary
            for _, row in self.capacity_data.iterrows():
                style = str(row['Style'])
                capacity = row['Average of lbs/day']
                
                # Handle negative capacities (likely data issues)
                if capacity < 0:
                    logger.warning(f"Negative capacity for style {style}: {capacity}. Using default.")
                    capacity = self.default_capacity
                
                self.style_capacities[style] = capacity
            
            logger.info(f"Loaded {len(self.style_capacities)} style capacities from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading capacity data: {e}")
            return False
    
    def get_style_capacity(self, style: str, use_mapper: bool = True) -> float:
        """
        Get production capacity for a style in lbs/day
        Returns actual capacity if available, otherwise returns default
        """
        # Direct lookup
        if style in self.style_capacities:
            return self.style_capacities[style]
        
        # Try without spaces
        style_no_space = style.replace(' ', '')
        if style_no_space in self.style_capacities:
            return self.style_capacities[style_no_space]
        
        # Try to find similar styles (same base)
        if use_mapper and '/' in style:
            base = style.split('/')[0]
            for cap_style, capacity in self.style_capacities.items():
                if cap_style.startswith(base):
                    return capacity
        
        # Return default if not found
        return self.default_capacity
    
    def calculate_production_time(self, style: str, quantity_lbs: float) -> Dict:
        """
        Calculate production time for a given style and quantity
        Returns dict with production metrics
        """
        capacity_per_day = self.get_style_capacity(style)
        
        if capacity_per_day <= 0:
            capacity_per_day = self.default_capacity
        
        # Calculate days needed
        days_needed = quantity_lbs / capacity_per_day
        
        # Calculate calendar days (accounting for weekends)
        calendar_days = self.working_days_to_calendar_days(days_needed)
        
        # Calculate machine hours (assuming 8-hour shifts)
        hours_per_day = 8
        total_hours = days_needed * hours_per_day
        
        return {
            'style': style,
            'quantity_lbs': quantity_lbs,
            'capacity_per_day': capacity_per_day,
            'production_days': round(days_needed, 1),
            'calendar_days': calendar_days,
            'total_hours': round(total_hours, 1),
            'efficiency_rating': self.get_efficiency_rating(capacity_per_day)
        }
    
    def working_days_to_calendar_days(self, working_days: float) -> int:
        """Convert working days to calendar days (accounting for weekends)"""
        # Assume 5 working days per week
        weeks = working_days / 5
        calendar_days = weeks * 7
        return int(np.ceil(calendar_days))
    
    def get_efficiency_rating(self, capacity: float) -> str:
        """
        Rate production efficiency based on capacity
        """
        if capacity >= 2000:
            return "EXCELLENT"
        elif capacity >= 1000:
            return "GOOD"
        elif capacity >= 500:
            return "AVERAGE"
        elif capacity >= 100:
            return "BELOW_AVERAGE"
        else:
            return "POOR"
    
    def optimize_production_schedule(self, production_requests: List[Dict], 
                                    max_daily_capacity: float = 10000) -> List[Dict]:
        """
        Optimize production schedule based on capacity constraints
        Groups similar styles and prioritizes by efficiency
        """
        schedule = []
        current_date = datetime.now()
        daily_load = {}  # Track daily capacity usage
        
        # Sort by priority and efficiency
        for request in production_requests:
            style = request.get('style')
            quantity = request.get('quantity_lbs', 0)
            priority = request.get('priority_score', 0.5)
            
            if quantity <= 0:
                continue
            
            # Get style capacity
            capacity = self.get_style_capacity(style)
            
            # Calculate efficiency score
            efficiency_score = capacity / self.default_capacity
            
            # Combined score (priority + efficiency)
            combined_score = (priority * 0.7) + (efficiency_score * 0.3)
            
            request['capacity_per_day'] = capacity
            request['efficiency_score'] = efficiency_score
            request['combined_score'] = combined_score
        
        # Sort by combined score
        sorted_requests = sorted(production_requests, 
                                key=lambda x: x.get('combined_score', 0), 
                                reverse=True)
        
        # Schedule production
        for request in sorted_requests:
            style = request.get('style')
            quantity = request.get('quantity_lbs', 0)
            capacity = request.get('capacity_per_day', self.default_capacity)
            
            # Find earliest available slot
            scheduled = False
            check_date = current_date
            
            while not scheduled:
                date_key = check_date.strftime('%Y-%m-%d')
                current_load = daily_load.get(date_key, 0)
                
                if current_load + capacity <= max_daily_capacity:
                    # Can schedule on this day
                    days_needed = quantity / capacity
                    
                    schedule_item = {
                        'style': style,
                        'quantity_lbs': quantity,
                        'start_date': check_date.strftime('%Y-%m-%d'),
                        'end_date': (check_date + timedelta(days=days_needed)).strftime('%Y-%m-%d'),
                        'capacity_per_day': capacity,
                        'days_needed': round(days_needed, 1),
                        'priority': request.get('priority_rank', 'MEDIUM'),
                        'efficiency': self.get_efficiency_rating(capacity)
                    }
                    
                    schedule.append(schedule_item)
                    
                    # Update daily load
                    for day in range(int(np.ceil(days_needed))):
                        day_date = (check_date + timedelta(days=day)).strftime('%Y-%m-%d')
                        daily_load[day_date] = daily_load.get(day_date, 0) + capacity
                    
                    scheduled = True
                else:
                    # Try next day
                    check_date += timedelta(days=1)
                    
                    # Stop if too far in future
                    if (check_date - current_date).days > 180:
                        break
        
        return schedule
    
    def get_capacity_summary(self) -> Dict:
        """Get summary statistics of production capacity"""
        if not self.style_capacities:
            return {
                'total_styles': 0,
                'avg_capacity': self.default_capacity,
                'total_daily_capacity': 0
            }
        
        capacities = list(self.style_capacities.values())
        # Filter out negative values for statistics
        positive_capacities = [c for c in capacities if c > 0]
        
        return {
            'total_styles': len(self.style_capacities),
            'avg_capacity': np.mean(positive_capacities) if positive_capacities else 0,
            'min_capacity': min(positive_capacities) if positive_capacities else 0,
            'max_capacity': max(positive_capacities) if positive_capacities else 0,
            'median_capacity': np.median(positive_capacities) if positive_capacities else 0,
            'total_daily_capacity': sum(positive_capacities),
            'excellent_efficiency_styles': len([c for c in positive_capacities if c >= 2000]),
            'good_efficiency_styles': len([c for c in positive_capacities if 1000 <= c < 2000]),
            'average_efficiency_styles': len([c for c in positive_capacities if 500 <= c < 1000]),
            'below_average_efficiency_styles': len([c for c in positive_capacities if 100 <= c < 500]),
            'poor_efficiency_styles': len([c for c in positive_capacities if c < 100])
        }
    
    def validate_production_request(self, style: str, quantity_lbs: float, 
                                   deadline_days: int = None) -> Tuple[bool, str]:
        """
        Validate if a production request is feasible
        """
        capacity = self.get_style_capacity(style)
        
        if capacity <= 0:
            return False, f"Style {style} has invalid production capacity"
        
        days_needed = quantity_lbs / capacity
        
        if deadline_days and days_needed > deadline_days:
            return False, f"Cannot produce {quantity_lbs} lbs in {deadline_days} days. Need {days_needed:.1f} days"
        
        if quantity_lbs > capacity * 30:  # More than a month's production
            return False, f"Quantity {quantity_lbs} lbs exceeds monthly capacity of {capacity*30:.0f} lbs"
        
        return True, f"Can produce {quantity_lbs} lbs in {days_needed:.1f} days"
    
    # === Machine-Level Tracking Methods ===
    
    def get_work_center_for_style(self, style: str) -> Optional[str]:
        """Get work center assigned to a style"""
        if self.machine_mapper:
            return self.machine_mapper.get_work_center_for_style(style)
        return None
    
    def get_machine_ids_for_style(self, style: str) -> List[str]:
        """Get all machine IDs that can process a style"""
        if self.machine_mapper:
            return self.machine_mapper.get_machine_ids_for_style(style)
        return []
    
    def get_machine_ids_for_work_center(self, work_center: str) -> List[str]:
        """Get all machine IDs in a work center"""
        if self.machine_mapper:
            return self.machine_mapper.get_machine_ids_for_work_center(work_center)
        return []
    
    def get_machine_utilization(self, machine_id: str) -> float:
        """Get current utilization percentage for a machine"""
        return self.machine_utilization.get(machine_id, 0.0)
    
    def update_machine_utilization(self, machine_id: str, utilization: float):
        """Update machine utilization percentage"""
        self.machine_utilization[machine_id] = max(0.0, min(100.0, utilization))
    
    def assign_style_to_machine(self, machine_id: str, style: str) -> bool:
        """Assign a style to a specific machine"""
        if self.machine_mapper and machine_id in self.machine_mapper.machine_info:
            self.machine_assignments[machine_id] = style
            # Update utilization (placeholder logic)
            self.update_machine_utilization(machine_id, 75.0)  # Assume 75% when assigned
            return True
        return False
    
    def get_machine_assignment(self, machine_id: str) -> Optional[str]:
        """Get currently assigned style for a machine"""
        return self.machine_assignments.get(machine_id)
    
    def get_work_center_capacity_summary(self, work_center: str) -> Dict[str, Any]:
        """Get capacity summary for a work center"""
        if not self.machine_mapper:
            return {}
        
        machine_ids = self.get_machine_ids_for_work_center(work_center)
        styles = self.machine_mapper.get_styles_for_work_center(work_center)
        
        # Calculate total capacity for this work center
        total_capacity = 0.0
        avg_utilization = 0.0
        active_machines = 0
        
        for machine_id in machine_ids:
            # Get style capacity (use default if no specific style assigned)
            assigned_style = self.get_machine_assignment(machine_id)
            if assigned_style:
                machine_capacity = self.get_style_capacity(assigned_style)
            else:
                machine_capacity = self.default_capacity
            
            total_capacity += machine_capacity
            utilization = self.get_machine_utilization(machine_id)
            avg_utilization += utilization
            
            if utilization > 0:
                active_machines += 1
        
        # Calculate averages
        machine_count = len(machine_ids)
        if machine_count > 0:
            avg_utilization = avg_utilization / machine_count
        
        return {
            'work_center': work_center,
            'machine_count': machine_count,
            'active_machines': active_machines,
            'total_capacity_lbs_day': total_capacity,
            'avg_capacity_per_machine': total_capacity / max(1, machine_count),
            'avg_utilization_percent': avg_utilization,
            'assigned_styles_count': len(styles),
            'assigned_styles': styles,
            'machine_ids': machine_ids
        }
    
    def get_all_work_centers_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get capacity summary for all work centers"""
        if not self.machine_mapper:
            return {}
        
        summaries = {}
        for work_center in self.machine_mapper.work_center_to_machines.keys():
            summaries[work_center] = self.get_work_center_capacity_summary(work_center)
        
        return summaries
    
    def get_machine_level_status(self, active_only: bool = False) -> Dict[str, Any]:
        """Get complete machine-level status overview

        Args:
            active_only: If True, only return machines with workload > 0 or forecasted demand
        """
        if not self.machine_mapper:
            return {'error': 'Machine mapper not available'}

        machine_status = []
        work_center_status = []

        # Collect machine-level data
        for machine_id, machine_info in self.machine_mapper.machine_info.items():
            utilization = self.get_machine_utilization(machine_id)
            workload_lbs = self.machine_workload_lbs.get(machine_id, 0)
            suggested_workload_lbs = self.machine_suggested_workload_lbs.get(machine_id, 0)
            forecasted_workload_lbs = self.machine_forecasted_workload_lbs.get(machine_id, 0)

            # Calculate total potential workload
            total_workload = workload_lbs + suggested_workload_lbs + forecasted_workload_lbs

            # Skip machines with no workload if active_only is True
            if active_only and total_workload <= 0:
                continue

            status = {
                'machine_id': machine_id,
                'work_center': self.machine_mapper.get_work_center_for_machine(machine_id),
                'utilization': utilization,
                'assigned_style': self.get_machine_assignment(machine_id),
                'capacity_lbs_day': self.get_style_capacity(
                    self.get_machine_assignment(machine_id) or 'default'
                ),
                'workload_lbs': workload_lbs,
                'suggested_workload_lbs': suggested_workload_lbs,
                'forecasted_workload_lbs': forecasted_workload_lbs,
                'total_workload_lbs': total_workload,
                'days_of_work': round(total_workload / self.default_capacity, 1) if total_workload > 0 else 0,
                'status': 'RUNNING' if workload_lbs > 0 else ('PENDING' if suggested_workload_lbs > 0 else 'IDLE')
            }
            machine_status.append(status)

        # Collect work center summaries (filter if active_only)
        for work_center, summary in self.get_all_work_centers_summary().items():
            if active_only:
                # Check if work center has any active machines
                wc_machines = [m for m in machine_status if m['work_center'] == work_center]
                if not wc_machines:
                    continue
            work_center_status.append(summary)

        # Overall statistics
        total_machines = len(machine_status)
        running_machines = len([m for m in machine_status if m['status'] == 'RUNNING'])
        pending_machines = len([m for m in machine_status if m['status'] == 'PENDING'])
        total_capacity = sum(m['capacity_lbs_day'] for m in machine_status)
        avg_utilization = np.mean([m['utilization'] for m in machine_status]) if machine_status else 0

        return {
            'total_machines': total_machines,
            'running_machines': running_machines,
            'pending_machines': pending_machines,
            'idle_machines': total_machines - running_machines - pending_machines,
            'total_work_centers': len(work_center_status),
            'total_capacity_lbs_day': total_capacity,
            'avg_utilization_percent': avg_utilization,
            'machine_status': machine_status,
            'work_center_status': work_center_status,
            'filtered': active_only,
            'last_updated': datetime.now().isoformat()
        }
    
    def optimize_machine_assignments(self, production_requests: List[Dict]) -> List[Dict]:
        """
        Optimize style assignments to machines based on capacity and work center mapping
        Enhanced version of existing optimize_production_schedule with machine-level detail
        """
        if not self.machine_mapper:
            # Fall back to original method if no machine mapping
            return self.optimize_production_schedule(production_requests)
        
        optimized_assignments = []
        current_date = datetime.now()
        
        # Sort requests by priority and capacity requirements
        sorted_requests = sorted(production_requests, 
                                key=lambda x: x.get('priority_score', 0.5), 
                                reverse=True)
        
        for request in sorted_requests:
            style = request.get('style', '')
            quantity = request.get('quantity_lbs', 0)
            
            # Get work center and machines for this style
            work_center = self.get_work_center_for_style(style)
            machine_ids = self.get_machine_ids_for_style(style)
            
            if not work_center or not machine_ids:
                # Style not mapped - use original scheduling
                continue
            
            # Find best available machine in the work center
            best_machine = None
            best_utilization = 100.0  # Start with max, find lowest
            
            for machine_id in machine_ids:
                utilization = self.get_machine_utilization(machine_id)
                if utilization < best_utilization:
                    best_machine = machine_id
                    best_utilization = utilization
            
            if best_machine:
                # Calculate production metrics
                capacity = self.get_style_capacity(style)
                days_needed = quantity / capacity if capacity > 0 else quantity / self.default_capacity
                
                assignment = {
                    'style': style,
                    'quantity_lbs': quantity,
                    'work_center': work_center,
                    'assigned_machine': best_machine,
                    'machine_utilization_before': best_utilization,
                    'estimated_days': round(days_needed, 1),
                    'capacity_lbs_day': capacity,
                    'start_date': current_date.strftime('%Y-%m-%d'),
                    'estimated_completion': (current_date + timedelta(days=days_needed)).strftime('%Y-%m-%d'),
                    'priority': request.get('priority_score', 0.5),
                    'assignment_reason': f"Optimal machine in work center {work_center}"
                }
                
                optimized_assignments.append(assignment)
                
                # Update machine utilization for next iteration
                new_utilization = min(100.0, best_utilization + (days_needed * 10))  # Rough estimate
                self.update_machine_utilization(best_machine, new_utilization)
                self.assign_style_to_machine(best_machine, style)
        
        return optimized_assignments


# Global instance
_capacity_manager = None

def get_capacity_manager(capacity_file_path: Optional[str] = None) -> ProductionCapacityManager:
    """Get or create the global capacity manager instance"""
    global _capacity_manager
    
    if _capacity_manager is None:
        _capacity_manager = ProductionCapacityManager(capacity_file_path)
    
    return _capacity_manager