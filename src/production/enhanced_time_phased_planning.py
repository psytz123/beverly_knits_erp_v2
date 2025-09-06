#!/usr/bin/env python3
"""
Enhanced Time-Phased Planning Engine
Provides comprehensive weekly planning with demand vs supply analysis,
shortage predictions, and PO suggestions with lead time considerations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedTimePhasedPlanning:
    """
    Advanced time-phased planning with improved visualization and PO suggestions
    """
    
    def __init__(self, lead_time_days: int = 14, safety_stock_weeks: int = 2):
        """
        Initialize Enhanced Time-Phased Planning
        
        Args:
            lead_time_days: Default lead time for PO delivery
            safety_stock_weeks: Weeks of safety stock to maintain
        """
        self.lead_time_days = lead_time_days
        self.safety_stock_weeks = safety_stock_weeks
        self.current_week = self._get_current_week()
        self.planning_horizon_weeks = 12  # Extended to 12 weeks
        
        # Supplier lead times (in days)
        self.supplier_lead_times = {
            'default': 14,
            'domestic': 7,
            'international': 21,
            'express': 3
        }
        
        # Week mapping
        self.week_dates = self._generate_week_dates()
    
    def _get_current_week(self) -> int:
        """Get current week number"""
        return datetime.now().isocalendar()[1]
    
    def _generate_week_dates(self) -> Dict[int, str]:
        """Generate week number to date mapping"""
        week_dates = {}
        current_date = datetime.now()
        
        for i in range(self.planning_horizon_weeks):
            week_date = current_date + timedelta(weeks=i)
            week_num = week_date.isocalendar()[1]
            week_dates[week_num] = week_date.strftime('%m/%d/%Y')
        
        return week_dates
    
    def calculate_enhanced_balance(
        self,
        yarn_id: str,
        current_balance: float,
        safety_stock: float,
        weekly_demand: Dict[str, float],
        weekly_receipts: Dict[str, float],
        production_schedule: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate enhanced weekly balance with multiple metrics
        
        Args:
            yarn_id: Yarn identifier
            current_balance: Current inventory balance
            safety_stock: Safety stock level
            weekly_demand: Demand by week
            weekly_receipts: Scheduled PO receipts by week
            production_schedule: Optional production orders by week
            
        Returns:
            Comprehensive weekly analysis with balances, shortages, and metrics
        """
        result = {
            'yarn_id': yarn_id,
            'current_balance': current_balance,
            'safety_stock': safety_stock,
            'weekly_analysis': [],
            'shortage_weeks': [],
            'coverage_weeks': 0,
            'total_demand': 0,
            'total_receipts': 0,
            'suggested_po': None,
            'risk_level': 'LOW'
        }
        
        balance = current_balance
        cumulative_demand = 0
        cumulative_receipts = 0
        first_shortage_week = None
        
        for week_offset in range(self.planning_horizon_weeks):
            week_num = self.current_week + week_offset
            week_key = f'week_{week_num}'
            
            # Get demand and receipts
            demand = weekly_demand.get(week_key, 0)
            receipt = weekly_receipts.get(week_key, 0)
            
            # Calculate new balance
            balance = balance + receipt - demand
            cumulative_demand += demand
            cumulative_receipts += receipt
            
            # Check for shortage
            is_shortage = balance < safety_stock
            is_critical = balance < 0
            
            # Production orders for this week (if provided)
            production_orders = []
            if production_schedule:
                production_orders = production_schedule.get(week_key, [])
            
            week_data = {
                'week': week_num,
                'week_key': week_key,
                'date': self.week_dates.get(week_num, ''),
                'demand': demand,
                'receipt': receipt,
                'balance': balance,
                'safety_stock': safety_stock,
                'is_shortage': is_shortage,
                'is_critical': is_critical,
                'shortage_amount': max(0, safety_stock - balance) if is_shortage else 0,
                'production_orders': production_orders,
                'coverage_days': (balance / (demand / 7)) if demand > 0 else float('inf')
            }
            
            result['weekly_analysis'].append(week_data)
            
            # Track shortage weeks
            if is_shortage:
                result['shortage_weeks'].append(week_num)
                if first_shortage_week is None:
                    first_shortage_week = week_num
            
            # Calculate coverage weeks
            if balance > 0 and demand > 0:
                result['coverage_weeks'] = max(result['coverage_weeks'], week_offset)
        
        result['total_demand'] = cumulative_demand
        result['total_receipts'] = cumulative_receipts
        
        # Determine risk level
        if first_shortage_week:
            weeks_until_shortage = first_shortage_week - self.current_week
            if weeks_until_shortage <= 1:
                result['risk_level'] = 'CRITICAL'
            elif weeks_until_shortage <= 2:
                result['risk_level'] = 'HIGH'
            elif weeks_until_shortage <= 4:
                result['risk_level'] = 'MEDIUM'
            else:
                result['risk_level'] = 'LOW'
        
        # Generate PO suggestion if needed
        if result['shortage_weeks']:
            result['suggested_po'] = self._generate_po_suggestion(
                yarn_id, result['weekly_analysis'], first_shortage_week
            )
        
        return result
    
    def _generate_po_suggestion(
        self,
        yarn_id: str,
        weekly_analysis: List[Dict],
        first_shortage_week: int
    ) -> Dict[str, Any]:
        """
        Generate intelligent PO suggestion based on shortage analysis
        
        Args:
            yarn_id: Yarn identifier
            weekly_analysis: Weekly balance analysis
            first_shortage_week: First week with shortage
            
        Returns:
            PO suggestion with quantity, timing, and supplier recommendation
        """
        # Calculate required quantity
        shortage_data = [w for w in weekly_analysis if w['is_shortage']]
        if not shortage_data:
            return None
        
        # Sum shortage amounts for next 4 weeks
        required_qty = sum(w['shortage_amount'] for w in shortage_data[:4])
        
        # Add buffer for safety
        required_qty *= 1.2
        
        # Calculate order date based on lead time
        weeks_until_shortage = max(0, first_shortage_week - self.current_week)
        lead_time_weeks = self.lead_time_days / 7
        order_by_week = self.current_week + max(0, weeks_until_shortage - lead_time_weeks)
        
        # Determine supplier based on urgency
        if weeks_until_shortage <= 1:
            supplier_type = 'express'
            lead_time = self.supplier_lead_times['express']
        elif weeks_until_shortage <= 2:
            supplier_type = 'domestic'
            lead_time = self.supplier_lead_times['domestic']
        else:
            supplier_type = 'default'
            lead_time = self.supplier_lead_times['default']
        
        delivery_week = self.current_week + (lead_time / 7)
        
        return {
            'yarn_id': yarn_id,
            'suggested_quantity': round(required_qty, 0),
            'order_by_week': round(order_by_week, 0),
            'delivery_week': round(delivery_week, 0),
            'supplier_type': supplier_type,
            'lead_time_days': lead_time,
            'urgency': 'HIGH' if weeks_until_shortage <= 2 else 'MEDIUM',
            'estimated_cost': required_qty * 5.0  # Assuming $5/lb average
        }
    
    def generate_procurement_plan(
        self,
        yarn_shortages: List[Dict],
        budget_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive procurement plan for multiple yarns
        
        Args:
            yarn_shortages: List of yarns with shortage data
            budget_limit: Optional budget constraint
            
        Returns:
            Prioritized procurement plan with PO suggestions
        """
        procurement_plan = {
            'po_suggestions': [],
            'total_cost': 0,
            'total_quantity': 0,
            'critical_pos': [],
            'budget_exceeded': False
        }
        
        # Sort by urgency
        sorted_shortages = sorted(
            yarn_shortages,
            key=lambda x: (x.get('risk_level', 'LOW'), x.get('first_shortage_week', 999))
        )
        
        running_cost = 0
        
        for shortage in sorted_shortages:
            if shortage.get('suggested_po'):
                po = shortage['suggested_po']
                po_cost = po['estimated_cost']
                
                # Check budget constraint
                if budget_limit and running_cost + po_cost > budget_limit:
                    procurement_plan['budget_exceeded'] = True
                    po['status'] = 'BUDGET_EXCEEDED'
                else:
                    po['status'] = 'APPROVED'
                    running_cost += po_cost
                
                procurement_plan['po_suggestions'].append(po)
                
                if po.get('urgency') == 'HIGH':
                    procurement_plan['critical_pos'].append(po)
        
        procurement_plan['total_cost'] = running_cost
        procurement_plan['total_quantity'] = sum(
            po['suggested_quantity'] for po in procurement_plan['po_suggestions']
            if po.get('status') == 'APPROVED'
        )
        
        return procurement_plan
    
    def analyze_supply_chain_risk(
        self,
        yarn_data: List[Dict],
        supplier_performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze supply chain risks across all yarns
        
        Args:
            yarn_data: List of yarn analysis data
            supplier_performance: Optional supplier reliability scores
            
        Returns:
            Supply chain risk assessment with recommendations
        """
        risk_assessment = {
            'overall_risk': 'LOW',
            'risk_factors': [],
            'recommendations': [],
            'metrics': {
                'yarns_at_risk': 0,
                'total_shortage_value': 0,
                'avg_coverage_weeks': 0,
                'critical_yarns': []
            }
        }
        
        critical_count = 0
        total_coverage = 0
        total_shortage_value = 0
        
        for yarn in yarn_data:
            if yarn.get('risk_level') == 'CRITICAL':
                critical_count += 1
                risk_assessment['metrics']['critical_yarns'].append(yarn['yarn_id'])
            
            if yarn.get('risk_level') in ['CRITICAL', 'HIGH']:
                risk_assessment['metrics']['yarns_at_risk'] += 1
            
            total_coverage += yarn.get('coverage_weeks', 0)
            
            # Calculate shortage value
            if yarn.get('suggested_po'):
                total_shortage_value += yarn['suggested_po'].get('estimated_cost', 0)
        
        risk_assessment['metrics']['total_shortage_value'] = total_shortage_value
        risk_assessment['metrics']['avg_coverage_weeks'] = (
            total_coverage / len(yarn_data) if yarn_data else 0
        )
        
        # Determine overall risk level
        if critical_count > 5:
            risk_assessment['overall_risk'] = 'CRITICAL'
            risk_assessment['risk_factors'].append('Multiple critical yarn shortages')
            risk_assessment['recommendations'].append('Expedite PO processing for critical yarns')
        elif critical_count > 2:
            risk_assessment['overall_risk'] = 'HIGH'
            risk_assessment['risk_factors'].append('Several yarns at critical levels')
            risk_assessment['recommendations'].append('Review and adjust safety stock levels')
        elif risk_assessment['metrics']['yarns_at_risk'] > 10:
            risk_assessment['overall_risk'] = 'MEDIUM'
            risk_assessment['risk_factors'].append('Elevated number of at-risk yarns')
            risk_assessment['recommendations'].append('Implement preventive ordering strategy')
        
        # Add supplier-based recommendations
        if supplier_performance:
            low_performers = [s for s, score in supplier_performance.items() if score < 0.8]
            if low_performers:
                risk_assessment['risk_factors'].append(f'Supplier reliability issues: {", ".join(low_performers)}')
                risk_assessment['recommendations'].append('Diversify supplier base for critical yarns')
        
        return risk_assessment


# Example usage
if __name__ == "__main__":
    planner = EnhancedTimePhasedPlanning()
    
    # Sample yarn analysis
    result = planner.calculate_enhanced_balance(
        yarn_id='Y001-BLK',
        current_balance=1000,
        safety_stock=500,
        weekly_demand={f'week_{i}': 200 for i in range(36, 48)},
        weekly_receipts={'week_38': 1500, 'week_41': 2000}
    )
    
    print(f"Yarn {result['yarn_id']} Risk Level: {result['risk_level']}")
    print(f"Coverage Weeks: {result['coverage_weeks']}")
    if result['suggested_po']:
        print(f"Suggested PO: {result['suggested_po']['suggested_quantity']} lbs")
        print(f"Order by Week: {result['suggested_po']['order_by_week']}")