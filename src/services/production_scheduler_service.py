"""
Production Scheduler Service
Handles production scheduling, machine assignment, and workload optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProductionSchedulerService:
    """Service for production scheduling and machine assignment"""
    
    def __init__(self, data_loader, capacity_engine=None):
        """
        Initialize production scheduler service
        
        Args:
            data_loader: Data loading service
            capacity_engine: Capacity planning engine (optional)
        """
        self.data_loader = data_loader
        self.capacity_engine = capacity_engine
        self.machine_assignments = {}
        self.work_center_mappings = {}
        self._load_work_center_mappings()
    
    def _load_work_center_mappings(self):
        """Load work center to machine mappings"""
        try:
            # Load QuadS mappings for style to work center
            quad_df = self.data_loader.load_quad_s_data()
            if quad_df is not None and not quad_df.empty:
                # Assuming columns C=style, D=work_center
                self.work_center_mappings = dict(zip(
                    quad_df.iloc[:, 2],  # Column C (style)
                    quad_df.iloc[:, 3]   # Column D (work_center)
                ))
            
            logger.info(f"Loaded {len(self.work_center_mappings)} work center mappings")
        except Exception as e:
            logger.error(f"Error loading work center mappings: {e}")
            self.work_center_mappings = {}
    
    def schedule_production(self, orders: pd.DataFrame, 
                          priority_rules: Optional[Dict] = None) -> pd.DataFrame:
        """
        Schedule production orders based on priority and constraints
        
        Args:
            orders: DataFrame of production orders
            priority_rules: Optional priority rules for scheduling
            
        Returns:
            Scheduled orders DataFrame
        """
        if orders.empty:
            return orders
        
        # Apply default priority rules if not provided
        if priority_rules is None:
            priority_rules = {
                'critical': 1,
                'high': 2,
                'normal': 3,
                'low': 4
            }
        
        # Copy orders to avoid modifying original
        scheduled_orders = orders.copy()
        
        # Add priority scoring
        scheduled_orders['priority_score'] = scheduled_orders.apply(
            lambda row: self._calculate_priority_score(row, priority_rules),
            axis=1
        )
        
        # Sort by priority score (lower is higher priority)
        scheduled_orders = scheduled_orders.sort_values('priority_score')
        
        # Assign scheduled dates based on capacity
        scheduled_orders = self._assign_scheduled_dates(scheduled_orders)
        
        # Assign machines if not already assigned
        unassigned = scheduled_orders[
            scheduled_orders['machine_id'].isna() | 
            (scheduled_orders['machine_id'] == '')
        ]
        
        if not unassigned.empty:
            scheduled_orders = self._assign_machines(scheduled_orders)
        
        return scheduled_orders
    
    def _calculate_priority_score(self, order: pd.Series, 
                                 priority_rules: Dict) -> float:
        """Calculate priority score for an order"""
        base_priority = priority_rules.get(
            order.get('priority', 'normal').lower(), 
            3
        )
        
        # Adjust for order age (older orders get higher priority)
        if 'created_date' in order and pd.notna(order['created_date']):
            age_days = (datetime.now() - pd.to_datetime(order['created_date'])).days
            age_factor = max(0, 1 - (age_days / 30))  # Decrease priority over 30 days
        else:
            age_factor = 0
        
        # Adjust for quantity (larger orders may need more lead time)
        quantity_factor = min(1, order.get('quantity', 0) / 10000)
        
        return base_priority - age_factor + (quantity_factor * 0.5)
    
    def _assign_scheduled_dates(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Assign scheduled dates based on capacity and lead times"""
        current_date = datetime.now()
        capacity_per_day = self._get_daily_capacity()
        
        daily_load = {}
        
        for idx, order in orders.iterrows():
            # Find available date with capacity
            scheduled_date = current_date
            days_ahead = 0
            
            while True:
                date_key = scheduled_date.strftime('%Y-%m-%d')
                current_load = daily_load.get(date_key, 0)
                
                if current_load + order.get('quantity', 0) <= capacity_per_day:
                    # Assign this date
                    orders.at[idx, 'scheduled_date'] = scheduled_date
                    daily_load[date_key] = current_load + order.get('quantity', 0)
                    break
                
                # Try next day
                scheduled_date += timedelta(days=1)
                days_ahead += 1
                
                # Prevent infinite loop
                if days_ahead > 365:
                    orders.at[idx, 'scheduled_date'] = scheduled_date
                    logger.warning(f"Order {order.get('order_id')} scheduled far in future")
                    break
        
        return orders
    
    def _get_daily_capacity(self) -> float:
        """Get daily production capacity"""
        if self.capacity_engine:
            return self.capacity_engine.get_daily_capacity()
        
        # Default capacity if no engine available
        return 10000.0  # lbs per day
    
    def _assign_machines(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Assign machines to unassigned orders"""
        machine_data = self.data_loader.load_machine_data()
        
        if machine_data is None or machine_data.empty:
            logger.warning("No machine data available for assignment")
            return orders
        
        for idx, order in orders.iterrows():
            if pd.isna(order.get('machine_id')) or order.get('machine_id') == '':
                # Get work center for this style
                style = order.get('style_id', '')
                work_center = self.work_center_mappings.get(style)
                
                if work_center:
                    # Find available machines for this work center
                    available_machines = self._find_available_machines(
                        work_center, 
                        machine_data,
                        order.get('scheduled_date')
                    )
                    
                    if available_machines:
                        # Assign to machine with lowest load
                        best_machine = min(
                            available_machines, 
                            key=lambda m: m.get('current_load', 0)
                        )
                        orders.at[idx, 'machine_id'] = best_machine['machine_id']
                        orders.at[idx, 'work_center'] = work_center
                        
                        logger.info(f"Assigned order {order.get('order_id')} to machine {best_machine['machine_id']}")
        
        return orders
    
    def _find_available_machines(self, work_center: str, 
                                machine_data: pd.DataFrame,
                                scheduled_date: Any) -> List[Dict]:
        """Find available machines for a work center"""
        available = []
        
        # Filter machines by work center pattern
        wc_pattern = work_center.split('.')[0] if '.' in work_center else work_center
        
        for _, machine in machine_data.iterrows():
            machine_wc = str(machine.get('work_center', ''))
            if machine_wc.startswith(wc_pattern):
                available.append({
                    'machine_id': machine.get('machine_id'),
                    'current_load': self._get_machine_load(
                        machine.get('machine_id'), 
                        scheduled_date
                    )
                })
        
        return available
    
    def _get_machine_load(self, machine_id: str, date: Any) -> float:
        """Get current load for a machine on a specific date"""
        # This would query actual production data
        # For now, return a random load for simulation
        return np.random.uniform(0, 1000)
    
    def get_machine_utilization(self) -> Dict[str, float]:
        """Get utilization metrics for all machines"""
        utilization = {}
        
        try:
            machine_data = self.data_loader.load_machine_data()
            orders = self.data_loader.load_production_orders()
            
            if machine_data is not None and orders is not None:
                for _, machine in machine_data.iterrows():
                    machine_id = machine.get('machine_id')
                    
                    # Calculate load for this machine
                    machine_orders = orders[orders['machine_id'] == machine_id]
                    total_load = machine_orders['quantity'].sum() if not machine_orders.empty else 0
                    
                    # Assume capacity of 1000 lbs per day, 30 days
                    capacity = 30000
                    utilization[str(machine_id)] = min(100, (total_load / capacity) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating machine utilization: {e}")
        
        return utilization
    
    def optimize_schedule(self, current_schedule: pd.DataFrame,
                         optimization_goals: Optional[Dict] = None) -> pd.DataFrame:
        """
        Optimize existing schedule based on goals
        
        Args:
            current_schedule: Current production schedule
            optimization_goals: Goals for optimization (e.g., minimize changeover)
            
        Returns:
            Optimized schedule
        """
        if optimization_goals is None:
            optimization_goals = {
                'minimize_changeover': True,
                'balance_workload': True,
                'prioritize_urgent': True
            }
        
        optimized = current_schedule.copy()
        
        if optimization_goals.get('minimize_changeover'):
            optimized = self._minimize_changeover(optimized)
        
        if optimization_goals.get('balance_workload'):
            optimized = self._balance_workload(optimized)
        
        if optimization_goals.get('prioritize_urgent'):
            optimized = self._prioritize_urgent_orders(optimized)
        
        return optimized
    
    def _minimize_changeover(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Minimize changeover time by grouping similar products"""
        # Group by style to minimize changeovers
        if 'style_id' in schedule.columns:
            schedule = schedule.sort_values(['machine_id', 'style_id', 'scheduled_date'])
        
        return schedule
    
    def _balance_workload(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Balance workload across machines"""
        if 'machine_id' not in schedule.columns:
            return schedule
        
        # Calculate load per machine
        machine_loads = schedule.groupby('machine_id')['quantity'].sum()
        
        # Find overloaded and underloaded machines
        mean_load = machine_loads.mean()
        std_load = machine_loads.std()
        
        overloaded = machine_loads[machine_loads > mean_load + std_load].index
        underloaded = machine_loads[machine_loads < mean_load - std_load].index
        
        # Reassign some orders from overloaded to underloaded machines
        for over_machine in overloaded:
            if len(underloaded) == 0:
                break
            
            # Find orders that can be moved
            movable = schedule[
                (schedule['machine_id'] == over_machine) & 
                (schedule.get('locked', False) != True)
            ]
            
            if not movable.empty:
                # Move some orders to underloaded machine
                orders_to_move = movable.head(len(movable) // 4)  # Move 25%
                under_machine = underloaded[0]
                
                schedule.loc[orders_to_move.index, 'machine_id'] = under_machine
                
                # Recalculate loads
                machine_loads = schedule.groupby('machine_id')['quantity'].sum()
                underloaded = machine_loads[machine_loads < mean_load - std_load].index
        
        return schedule
    
    def _prioritize_urgent_orders(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Ensure urgent orders are scheduled first"""
        if 'priority' in schedule.columns:
            # Sort by priority and adjust scheduled dates
            priority_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
            schedule['priority_rank'] = schedule['priority'].map(
                lambda x: priority_order.get(str(x).lower(), 2)
            )
            schedule = schedule.sort_values(['priority_rank', 'scheduled_date'])
            schedule = schedule.drop('priority_rank', axis=1)
        
        return schedule
    
    def get_schedule_metrics(self, schedule: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a production schedule"""
        metrics = {
            'total_orders': len(schedule),
            'assigned_orders': len(schedule[schedule['machine_id'].notna()]),
            'unassigned_orders': len(schedule[schedule['machine_id'].isna()]),
            'total_quantity': schedule['quantity'].sum() if 'quantity' in schedule.columns else 0,
            'unique_styles': schedule['style_id'].nunique() if 'style_id' in schedule.columns else 0,
            'machines_used': schedule['machine_id'].nunique() if 'machine_id' in schedule.columns else 0,
            'average_lead_time': 0,
            'utilization': {}
        }
        
        # Calculate average lead time
        if 'scheduled_date' in schedule.columns:
            current_date = datetime.now()
            lead_times = []
            for _, order in schedule.iterrows():
                if pd.notna(order['scheduled_date']):
                    lead_time = (pd.to_datetime(order['scheduled_date']) - current_date).days
                    lead_times.append(max(0, lead_time))
            
            if lead_times:
                metrics['average_lead_time'] = np.mean(lead_times)
        
        # Add machine utilization
        metrics['utilization'] = self.get_machine_utilization()
        
        return metrics