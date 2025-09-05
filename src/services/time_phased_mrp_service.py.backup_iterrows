"""
Time-Phased MRP Service
Implements Material Requirements Planning with time-phased requirements calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimePhasedMRPService:
    """Service for time-phased material requirements planning"""
    
    def __init__(self, data_loader, inventory_service=None):
        """
        Initialize time-phased MRP service
        
        Args:
            data_loader: Data loading service
            inventory_service: Inventory management service (optional)
        """
        self.data_loader = data_loader
        self.inventory_service = inventory_service
        self.bom_data = None
        self.lead_times = {}
        self.safety_stock_levels = {}
        self._load_master_data()
    
    def _load_master_data(self):
        """Load BOM and master data"""
        try:
            # Load BOM data
            self.bom_data = self.data_loader.load_bom_data()
            
            # Load lead times (would come from supplier data)
            self._load_lead_times()
            
            # Load safety stock levels
            self._load_safety_stock_levels()
            
            logger.info(f"Loaded BOM with {len(self.bom_data) if self.bom_data is not None else 0} entries")
        except Exception as e:
            logger.error(f"Error loading master data: {e}")
    
    def _load_lead_times(self):
        """Load material lead times"""
        # In production, this would load from database
        # For now, use default lead times
        if self.bom_data is not None and not self.bom_data.empty:
            unique_materials = self.bom_data['yarn_id'].unique() if 'yarn_id' in self.bom_data.columns else []
            for material in unique_materials:
                # Default lead time based on material type
                self.lead_times[material] = np.random.randint(7, 30)  # 7-30 days
    
    def _load_safety_stock_levels(self):
        """Load safety stock levels for materials"""
        # In production, this would be calculated based on demand variability
        for material, lead_time in self.lead_times.items():
            # Simple safety stock calculation
            self.safety_stock_levels[material] = lead_time * 10  # 10 units per day as safety
    
    def calculate_requirements(self, demand_forecast: pd.DataFrame, 
                             planning_horizon_days: int = 90,
                             include_safety_stock: bool = True) -> Dict[str, Any]:
        """
        Calculate time-phased material requirements
        
        Args:
            demand_forecast: DataFrame with demand forecast by product and date
            planning_horizon_days: Number of days to plan ahead
            include_safety_stock: Whether to include safety stock in calculations
            
        Returns:
            Time-phased requirements plan
        """
        if demand_forecast.empty:
            return {'error': 'No demand forecast provided'}
        
        # Initialize MRP table
        mrp_table = self._initialize_mrp_table(planning_horizon_days)
        
        # Process each product in demand forecast
        for _, demand_row in demand_forecast.iterrows():
            product = demand_row.get('product_id', demand_row.get('style_id'))
            quantity = demand_row.get('quantity', 0)
            due_date = pd.to_datetime(demand_row.get('date', datetime.now()))
            
            # Calculate material requirements for this product
            material_reqs = self._explode_bom(product, quantity)
            
            # Schedule requirements considering lead times
            for material, req_qty in material_reqs.items():
                self._schedule_requirement(
                    mrp_table, 
                    material, 
                    req_qty, 
                    due_date,
                    include_safety_stock
                )
        
        # Calculate net requirements
        mrp_result = self._calculate_net_requirements(mrp_table)
        
        # Generate purchase orders
        purchase_orders = self._generate_purchase_orders(mrp_result)
        
        return {
            'mrp_table': mrp_result,
            'purchase_orders': purchase_orders,
            'summary': self._generate_mrp_summary(mrp_result, purchase_orders),
            'critical_materials': self._identify_critical_materials(mrp_result)
        }
    
    def _initialize_mrp_table(self, horizon_days: int) -> pd.DataFrame:
        """Initialize empty MRP table"""
        dates = pd.date_range(
            start=datetime.now().date(),
            periods=horizon_days,
            freq='D'
        )
        
        # Get all unique materials
        materials = []
        if self.bom_data is not None and 'yarn_id' in self.bom_data.columns:
            materials = self.bom_data['yarn_id'].unique().tolist()
        
        # Create multi-index for materials and dates
        index = pd.MultiIndex.from_product(
            [materials, dates],
            names=['material', 'date']
        )
        
        mrp_table = pd.DataFrame(
            index=index,
            columns=[
                'gross_requirements',
                'scheduled_receipts',
                'on_hand',
                'net_requirements',
                'planned_receipts',
                'planned_orders'
            ]
        ).fillna(0)
        
        return mrp_table
    
    def _explode_bom(self, product: str, quantity: float) -> Dict[str, float]:
        """
        Explode BOM to get material requirements
        
        Args:
            product: Product/style ID
            quantity: Required quantity
            
        Returns:
            Dictionary of material requirements
        """
        requirements = {}
        
        if self.bom_data is None or self.bom_data.empty:
            return requirements
        
        # Find BOM entries for this product
        product_bom = self.bom_data[
            self.bom_data.get('style_id', self.bom_data.get('fStyle#', '')) == product
        ]
        
        for _, bom_row in product_bom.iterrows():
            material = bom_row.get('yarn_id', bom_row.get('Desc#'))
            qty_per_unit = bom_row.get('quantity_per_unit', bom_row.get('Qty', 1))
            
            if material:
                requirements[material] = requirements.get(material, 0) + (quantity * qty_per_unit)
        
        return requirements
    
    def _schedule_requirement(self, mrp_table: pd.DataFrame, 
                            material: str, 
                            quantity: float,
                            due_date: pd.Timestamp,
                            include_safety_stock: bool):
        """Schedule material requirement considering lead time"""
        lead_time = self.lead_times.get(material, 14)  # Default 14 days
        
        # Calculate order date (due date minus lead time)
        order_date = due_date - timedelta(days=lead_time)
        
        # Ensure order date is not in the past
        if order_date < datetime.now():
            order_date = datetime.now()
            logger.warning(f"Order date for {material} adjusted to today due to lead time constraints")
        
        # Add to gross requirements
        if (material, due_date.date()) in mrp_table.index:
            mrp_table.loc[(material, due_date.date()), 'gross_requirements'] += quantity
        
        # Add safety stock if required
        if include_safety_stock and material in self.safety_stock_levels:
            safety_stock = self.safety_stock_levels[material]
            if (material, order_date.date()) in mrp_table.index:
                mrp_table.loc[(material, order_date.date()), 'gross_requirements'] += safety_stock
    
    def _calculate_net_requirements(self, mrp_table: pd.DataFrame) -> pd.DataFrame:
        """Calculate net requirements considering on-hand inventory"""
        result_table = mrp_table.copy()
        
        # Get current inventory levels
        current_inventory = self._get_current_inventory()
        
        # Process each material
        for material in mrp_table.index.get_level_values('material').unique():
            material_data = result_table.xs(material, level='material')
            on_hand = current_inventory.get(material, 0)
            
            for date in material_data.index:
                gross_req = material_data.loc[date, 'gross_requirements']
                scheduled_receipts = material_data.loc[date, 'scheduled_receipts']
                
                # Update on-hand inventory
                result_table.loc[(material, date), 'on_hand'] = on_hand + scheduled_receipts
                
                # Calculate net requirements
                net_req = max(0, gross_req - on_hand - scheduled_receipts)
                result_table.loc[(material, date), 'net_requirements'] = net_req
                
                # Plan order if net requirements > 0
                if net_req > 0:
                    # Lot sizing logic (for now, order exact quantity)
                    order_qty = self._calculate_lot_size(material, net_req)
                    result_table.loc[(material, date), 'planned_orders'] = order_qty
                    
                    # This becomes a receipt later
                    lead_time = self.lead_times.get(material, 14)
                    receipt_date = date + timedelta(days=lead_time)
                    if (material, receipt_date) in result_table.index:
                        result_table.loc[(material, receipt_date), 'planned_receipts'] = order_qty
                
                # Update on-hand for next period
                on_hand = on_hand + scheduled_receipts - gross_req
                on_hand = max(0, on_hand)  # Can't have negative on-hand
        
        return result_table
    
    def _get_current_inventory(self) -> Dict[str, float]:
        """Get current inventory levels"""
        inventory = {}
        
        if self.inventory_service:
            try:
                inv_data = self.inventory_service.get_current_inventory()
                if isinstance(inv_data, pd.DataFrame):
                    for _, row in inv_data.iterrows():
                        material = row.get('yarn_id', row.get('Desc#'))
                        balance = row.get('planning_balance', row.get('Planning Balance', 0))
                        if material:
                            inventory[material] = max(0, balance)  # Only positive inventory
            except Exception as e:
                logger.error(f"Error getting inventory: {e}")
        
        return inventory
    
    def _calculate_lot_size(self, material: str, net_requirement: float) -> float:
        """
        Calculate lot size for ordering
        
        Args:
            material: Material ID
            net_requirement: Net requirement quantity
            
        Returns:
            Lot size to order
        """
        # Simple lot sizing strategies
        # In production, this would use EOQ, POQ, or other methods
        
        # Minimum order quantity
        min_order_qty = 100  # Default minimum
        
        # Round up to nearest lot size
        lot_size = 50  # Standard lot size
        
        if net_requirement <= min_order_qty:
            return min_order_qty
        else:
            # Round up to nearest lot size
            return np.ceil(net_requirement / lot_size) * lot_size
    
    def _generate_purchase_orders(self, mrp_result: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate purchase orders from MRP results"""
        purchase_orders = []
        
        # Get all planned orders
        planned_orders = mrp_result[mrp_result['planned_orders'] > 0]
        
        for (material, date), row in planned_orders.iterrows():
            quantity = row['planned_orders']
            lead_time = self.lead_times.get(material, 14)
            
            purchase_order = {
                'material': material,
                'quantity': quantity,
                'order_date': date.strftime('%Y-%m-%d'),
                'due_date': (date + timedelta(days=lead_time)).strftime('%Y-%m-%d'),
                'lead_time_days': lead_time,
                'priority': self._calculate_priority(material, date),
                'estimated_cost': self._estimate_cost(material, quantity)
            }
            
            purchase_orders.append(purchase_order)
        
        # Sort by order date and priority
        purchase_orders.sort(key=lambda x: (x['order_date'], -x['priority']))
        
        return purchase_orders
    
    def _calculate_priority(self, material: str, order_date: pd.Timestamp) -> int:
        """Calculate order priority (1-5, 5 being highest)"""
        days_until_order = (order_date - datetime.now()).days
        
        if days_until_order <= 0:
            return 5  # Urgent
        elif days_until_order <= 7:
            return 4  # High
        elif days_until_order <= 14:
            return 3  # Medium
        elif days_until_order <= 30:
            return 2  # Low
        else:
            return 1  # Planning
    
    def _estimate_cost(self, material: str, quantity: float) -> float:
        """Estimate cost for material order"""
        # In production, this would use actual pricing data
        unit_cost = np.random.uniform(5, 50)  # Random unit cost for demo
        return quantity * unit_cost
    
    def _generate_mrp_summary(self, mrp_result: pd.DataFrame, 
                            purchase_orders: List[Dict]) -> Dict[str, Any]:
        """Generate MRP summary statistics"""
        return {
            'total_materials': mrp_result.index.get_level_values('material').nunique(),
            'total_requirements': mrp_result['gross_requirements'].sum(),
            'total_planned_orders': len(purchase_orders),
            'total_order_value': sum(po['estimated_cost'] for po in purchase_orders),
            'urgent_orders': len([po for po in purchase_orders if po['priority'] >= 4]),
            'materials_with_shortages': len(
                mrp_result[mrp_result['net_requirements'] > 0]
                .index.get_level_values('material').unique()
            )
        }
    
    def _identify_critical_materials(self, mrp_result: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify critical materials that need attention"""
        critical = []
        
        # Materials with high net requirements
        high_requirements = mrp_result[mrp_result['net_requirements'] > 1000]
        for (material, date), row in high_requirements.iterrows():
            critical.append({
                'material': material,
                'date': date.strftime('%Y-%m-%d'),
                'net_requirement': row['net_requirements'],
                'reason': 'High requirement volume'
            })
        
        # Materials with urgent orders (within 7 days)
        urgent_date = datetime.now() + timedelta(days=7)
        urgent_orders = mrp_result[
            (mrp_result['planned_orders'] > 0) & 
            (mrp_result.index.get_level_values('date') <= urgent_date)
        ]
        
        for (material, date), row in urgent_orders.iterrows():
            critical.append({
                'material': material,
                'date': date.strftime('%Y-%m-%d'),
                'net_requirement': row['planned_orders'],
                'reason': 'Urgent order required'
            })
        
        # Remove duplicates and sort by date
        seen = set()
        unique_critical = []
        for item in sorted(critical, key=lambda x: x['date']):
            key = (item['material'], item['date'])
            if key not in seen:
                seen.add(key)
                unique_critical.append(item)
        
        return unique_critical[:20]  # Return top 20 critical items
    
    def perform_pegging_analysis(self, material: str) -> Dict[str, Any]:
        """
        Perform pegging analysis to trace material requirements to demand
        
        Args:
            material: Material ID to analyze
            
        Returns:
            Pegging analysis results
        """
        pegging_results = {
            'material': material,
            'total_requirement': 0,
            'demand_sources': [],
            'timeline': []
        }
        
        if self.bom_data is None or self.bom_data.empty:
            return pegging_results
        
        # Find all products using this material
        products_using_material = self.bom_data[
            self.bom_data.get('yarn_id', self.bom_data.get('Desc#')) == material
        ]['style_id'].unique() if 'style_id' in self.bom_data.columns else []
        
        for product in products_using_material:
            # Get quantity per unit for this product
            qty_per_unit = self.bom_data[
                (self.bom_data['style_id'] == product) & 
                (self.bom_data.get('yarn_id', self.bom_data.get('Desc#')) == material)
            ].get('quantity_per_unit', [1]).iloc[0] if not self.bom_data.empty else 1
            
            pegging_results['demand_sources'].append({
                'product': product,
                'quantity_per_unit': qty_per_unit,
                'demand_type': 'production'
            })
        
        pegging_results['total_requirement'] = sum(
            source['quantity_per_unit'] 
            for source in pegging_results['demand_sources']
        )
        
        return pegging_results
    
    def calculate_capacity_requirements(self, production_plan: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate capacity requirements based on production plan
        
        Args:
            production_plan: Production plan DataFrame
            
        Returns:
            Capacity requirements by work center and period
        """
        capacity_requirements = {
            'by_work_center': {},
            'by_period': {},
            'total_hours_required': 0,
            'bottlenecks': []
        }
        
        # This would integrate with actual routing and work center data
        # For now, provide a simplified calculation
        
        for _, order in production_plan.iterrows():
            work_center = order.get('work_center', 'default')
            quantity = order.get('quantity', 0)
            
            # Assume standard hours per unit (would come from routing)
            hours_per_unit = 0.1  # 6 minutes per unit
            total_hours = quantity * hours_per_unit
            
            if work_center not in capacity_requirements['by_work_center']:
                capacity_requirements['by_work_center'][work_center] = 0
            
            capacity_requirements['by_work_center'][work_center] += total_hours
            capacity_requirements['total_hours_required'] += total_hours
        
        # Identify bottlenecks (work centers over 80% capacity)
        available_capacity = 160  # 160 hours per work center per week
        for wc, required_hours in capacity_requirements['by_work_center'].items():
            utilization = (required_hours / available_capacity) * 100
            if utilization > 80:
                capacity_requirements['bottlenecks'].append({
                    'work_center': wc,
                    'required_hours': required_hours,
                    'available_hours': available_capacity,
                    'utilization': utilization
                })
        
        return capacity_requirements