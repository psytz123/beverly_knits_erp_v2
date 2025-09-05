"""
Manufacturing Supply Chain Service
Integrates inventory, forecasting, and planning for end-to-end supply chain management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ManufacturingSupplyChainService:
    """Service for integrated manufacturing and supply chain operations"""
    
    def __init__(self, inventory_service, forecasting_service, planning_service):
        """
        Initialize manufacturing supply chain service
        
        Args:
            inventory_service: Inventory management service
            forecasting_service: Sales forecasting service
            planning_service: Production planning service
        """
        self.inventory = inventory_service
        self.forecasting = forecasting_service
        self.planning = planning_service
        self.supply_chain_metrics = {}
        self.optimization_history = []
    
    def analyze_supply_chain(self) -> Dict[str, Any]:
        """
        Perform comprehensive supply chain analysis
        
        Returns:
            Supply chain analysis results
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'inventory_status': self._analyze_inventory_status(),
            'demand_forecast': self._analyze_demand_forecast(),
            'production_capacity': self._analyze_production_capacity(),
            'supply_chain_risks': self._identify_supply_chain_risks(),
            'recommendations': self._generate_recommendations(),
            'kpis': self._calculate_supply_chain_kpis()
        }
        
        # Store metrics for trending
        self.supply_chain_metrics = analysis['kpis']
        
        return analysis
    
    def _analyze_inventory_status(self) -> Dict[str, Any]:
        """Analyze current inventory status"""
        try:
            inventory_data = self.inventory.get_enhanced_intelligence()
            
            return {
                'total_skus': inventory_data.get('total_items', 0),
                'shortage_count': inventory_data.get('shortage_count', 0),
                'critical_shortages': inventory_data.get('critical_shortages', []),
                'excess_inventory': self._identify_excess_inventory(),
                'inventory_value': self._calculate_inventory_value(),
                'turnover_rate': self._calculate_turnover_rate()
            }
        except Exception as e:
            logger.error(f"Error analyzing inventory status: {e}")
            return {}
    
    def _analyze_demand_forecast(self) -> Dict[str, Any]:
        """Analyze demand forecast"""
        try:
            forecast_data = self.forecasting.generate_comprehensive_forecast()
            
            return {
                'forecast_horizon': forecast_data.get('horizon', 30),
                'total_demand': forecast_data.get('total_forecasted_demand', 0),
                'confidence_level': forecast_data.get('confidence', 0),
                'top_products': forecast_data.get('top_products', []),
                'demand_variability': self._calculate_demand_variability(forecast_data)
            }
        except Exception as e:
            logger.error(f"Error analyzing demand forecast: {e}")
            return {}
    
    def _analyze_production_capacity(self) -> Dict[str, Any]:
        """Analyze production capacity"""
        try:
            capacity_data = self.planning.get_capacity_analysis()
            
            return {
                'total_capacity': capacity_data.get('total_capacity', 0),
                'utilized_capacity': capacity_data.get('utilized_capacity', 0),
                'utilization_rate': capacity_data.get('utilization_percentage', 0),
                'bottlenecks': capacity_data.get('bottlenecks', []),
                'available_capacity': capacity_data.get('available_capacity', 0)
            }
        except Exception as e:
            logger.error(f"Error analyzing production capacity: {e}")
            return {}
    
    def _identify_supply_chain_risks(self) -> List[Dict[str, Any]]:
        """Identify risks in the supply chain"""
        risks = []
        
        # Check for material shortages
        shortages = self.inventory.detect_shortages()
        if len(shortages) > 0:
            risks.append({
                'type': 'material_shortage',
                'severity': 'high' if len(shortages) > 10 else 'medium',
                'description': f"{len(shortages)} materials with negative planning balance",
                'impact': 'Production delays possible',
                'mitigation': 'Expedite purchase orders or find substitutes'
            })
        
        # Check for capacity constraints
        capacity = self._analyze_production_capacity()
        if capacity.get('utilization_rate', 0) > 90:
            risks.append({
                'type': 'capacity_constraint',
                'severity': 'high',
                'description': f"Capacity utilization at {capacity['utilization_rate']:.1f}%",
                'impact': 'Limited ability to handle demand spikes',
                'mitigation': 'Consider overtime or additional shifts'
            })
        
        # Check for demand variability
        forecast = self._analyze_demand_forecast()
        if forecast.get('demand_variability', 0) > 30:
            risks.append({
                'type': 'demand_uncertainty',
                'severity': 'medium',
                'description': f"High demand variability: {forecast['demand_variability']:.1f}%",
                'impact': 'Difficulty in planning inventory levels',
                'mitigation': 'Increase safety stock or improve forecasting'
            })
        
        # Check for supplier dependencies
        supplier_concentration = self._analyze_supplier_concentration()
        if supplier_concentration > 0.5:
            risks.append({
                'type': 'supplier_concentration',
                'severity': 'medium',
                'description': f"High supplier concentration: {supplier_concentration:.1%}",
                'impact': 'Vulnerable to supplier disruptions',
                'mitigation': 'Diversify supplier base'
            })
        
        return risks
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate supply chain recommendations"""
        recommendations = []
        
        # Analyze current state
        inventory_status = self._analyze_inventory_status()
        forecast = self._analyze_demand_forecast()
        capacity = self._analyze_production_capacity()
        
        # Inventory recommendations
        if inventory_status.get('shortage_count', 0) > 5:
            recommendations.append({
                'category': 'inventory',
                'priority': 'high',
                'action': 'Address material shortages',
                'details': f"Expedite orders for {inventory_status['shortage_count']} materials with shortages",
                'expected_impact': 'Prevent production delays'
            })
        
        if inventory_status.get('turnover_rate', 0) < 4:
            recommendations.append({
                'category': 'inventory',
                'priority': 'medium',
                'action': 'Improve inventory turnover',
                'details': 'Review slow-moving inventory and adjust ordering patterns',
                'expected_impact': 'Reduce carrying costs'
            })
        
        # Capacity recommendations
        if capacity.get('utilization_rate', 0) > 85:
            recommendations.append({
                'category': 'capacity',
                'priority': 'high',
                'action': 'Increase production capacity',
                'details': 'Consider adding shifts or equipment',
                'expected_impact': 'Meet growing demand'
            })
        elif capacity.get('utilization_rate', 0) < 60:
            recommendations.append({
                'category': 'capacity',
                'priority': 'medium',
                'action': 'Optimize capacity utilization',
                'details': 'Consolidate production runs or seek additional orders',
                'expected_impact': 'Reduce unit costs'
            })
        
        # Forecast recommendations
        if forecast.get('confidence_level', 0) < 0.8:
            recommendations.append({
                'category': 'planning',
                'priority': 'medium',
                'action': 'Improve forecast accuracy',
                'details': 'Review forecasting models and incorporate more data sources',
                'expected_impact': 'Better inventory planning'
            })
        
        return recommendations
    
    def optimize_supply_chain(self, optimization_goals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize supply chain based on specified goals
        
        Args:
            optimization_goals: Goals for optimization
            
        Returns:
            Optimization results and recommendations
        """
        if optimization_goals is None:
            optimization_goals = {
                'minimize_cost': True,
                'maximize_service_level': True,
                'reduce_lead_time': True,
                'balance_inventory': True
            }
        
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'goals': optimization_goals,
            'current_state': self._get_current_state(),
            'optimized_state': {},
            'improvements': {},
            'actions': []
        }
        
        # Perform optimization based on goals
        if optimization_goals.get('minimize_cost'):
            cost_optimization = self._optimize_costs()
            optimization_result['optimized_state']['costs'] = cost_optimization
            optimization_result['actions'].extend(cost_optimization.get('actions', []))
        
        if optimization_goals.get('maximize_service_level'):
            service_optimization = self._optimize_service_level()
            optimization_result['optimized_state']['service_level'] = service_optimization
            optimization_result['actions'].extend(service_optimization.get('actions', []))
        
        if optimization_goals.get('reduce_lead_time'):
            lead_time_optimization = self._optimize_lead_times()
            optimization_result['optimized_state']['lead_times'] = lead_time_optimization
            optimization_result['actions'].extend(lead_time_optimization.get('actions', []))
        
        if optimization_goals.get('balance_inventory'):
            inventory_optimization = self._optimize_inventory_levels()
            optimization_result['optimized_state']['inventory'] = inventory_optimization
            optimization_result['actions'].extend(inventory_optimization.get('actions', []))
        
        # Calculate improvements
        optimization_result['improvements'] = self._calculate_improvements(
            optimization_result['current_state'],
            optimization_result['optimized_state']
        )
        
        # Store in history
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current supply chain state"""
        return {
            'total_inventory_value': self._calculate_inventory_value(),
            'service_level': self._calculate_service_level(),
            'average_lead_time': self._calculate_average_lead_time(),
            'inventory_turnover': self._calculate_turnover_rate(),
            'total_costs': self._calculate_total_costs()
        }
    
    def _optimize_costs(self) -> Dict[str, Any]:
        """Optimize supply chain costs"""
        current_costs = self._calculate_total_costs()
        
        optimization = {
            'current_costs': current_costs,
            'optimized_costs': {},
            'savings': 0,
            'actions': []
        }
        
        # Identify cost reduction opportunities
        if current_costs.get('inventory_holding', 0) > current_costs.get('total', 0) * 0.3:
            optimization['actions'].append({
                'type': 'reduce_inventory',
                'description': 'Reduce excess inventory levels',
                'expected_savings': current_costs['inventory_holding'] * 0.2
            })
            optimization['optimized_costs']['inventory_holding'] = current_costs['inventory_holding'] * 0.8
        
        if current_costs.get('expediting', 0) > 0:
            optimization['actions'].append({
                'type': 'improve_planning',
                'description': 'Better planning to reduce expediting costs',
                'expected_savings': current_costs['expediting'] * 0.5
            })
            optimization['optimized_costs']['expediting'] = current_costs['expediting'] * 0.5
        
        optimization['savings'] = sum(
            action['expected_savings'] 
            for action in optimization['actions']
        )
        
        return optimization
    
    def _optimize_service_level(self) -> Dict[str, Any]:
        """Optimize service level"""
        current_service = self._calculate_service_level()
        
        optimization = {
            'current_level': current_service,
            'target_level': 0.95,
            'actions': []
        }
        
        if current_service < 0.95:
            gap = 0.95 - current_service
            
            optimization['actions'].append({
                'type': 'increase_safety_stock',
                'description': f"Increase safety stock for critical items",
                'expected_improvement': gap * 0.5
            })
            
            optimization['actions'].append({
                'type': 'improve_forecast',
                'description': 'Improve demand forecasting accuracy',
                'expected_improvement': gap * 0.3
            })
        
        return optimization
    
    def _optimize_lead_times(self) -> Dict[str, Any]:
        """Optimize lead times"""
        current_lead_time = self._calculate_average_lead_time()
        
        optimization = {
            'current_lead_time': current_lead_time,
            'optimized_lead_time': current_lead_time * 0.8,
            'actions': []
        }
        
        optimization['actions'].append({
            'type': 'streamline_processes',
            'description': 'Streamline production processes',
            'expected_reduction': current_lead_time * 0.1
        })
        
        optimization['actions'].append({
            'type': 'improve_scheduling',
            'description': 'Optimize production scheduling',
            'expected_reduction': current_lead_time * 0.1
        })
        
        return optimization
    
    def _optimize_inventory_levels(self) -> Dict[str, Any]:
        """Optimize inventory levels"""
        return {
            'current_turnover': self._calculate_turnover_rate(),
            'target_turnover': 6,
            'actions': [
                {
                    'type': 'adjust_ordering',
                    'description': 'Adjust ordering quantities and frequencies',
                    'expected_improvement': 'Increase turnover by 20%'
                },
                {
                    'type': 'implement_jit',
                    'description': 'Implement JIT for fast-moving items',
                    'expected_improvement': 'Reduce inventory by 15%'
                }
            ]
        }
    
    def _calculate_improvements(self, current_state: Dict, optimized_state: Dict) -> Dict:
        """Calculate improvements from optimization"""
        improvements = {}
        
        if 'costs' in optimized_state:
            current_costs = current_state.get('total_costs', {}).get('total', 0)
            optimized_costs = sum(optimized_state['costs'].get('optimized_costs', {}).values())
            if current_costs > 0:
                improvements['cost_reduction'] = (current_costs - optimized_costs) / current_costs
        
        if 'service_level' in optimized_state:
            current_service = current_state.get('service_level', 0)
            target_service = optimized_state['service_level'].get('target_level', 0)
            improvements['service_improvement'] = target_service - current_service
        
        if 'lead_times' in optimized_state:
            current_lead = current_state.get('average_lead_time', 0)
            optimized_lead = optimized_state['lead_times'].get('optimized_lead_time', 0)
            if current_lead > 0:
                improvements['lead_time_reduction'] = (current_lead - optimized_lead) / current_lead
        
        return improvements
    
    def _calculate_supply_chain_kpis(self) -> Dict[str, float]:
        """Calculate key supply chain KPIs"""
        return {
            'perfect_order_rate': self._calculate_perfect_order_rate(),
            'inventory_turnover': self._calculate_turnover_rate(),
            'cash_to_cash_cycle': self._calculate_cash_cycle(),
            'supply_chain_costs_ratio': self._calculate_cost_ratio(),
            'forecast_accuracy': self._calculate_forecast_accuracy(),
            'fill_rate': self._calculate_fill_rate(),
            'on_time_delivery': self._calculate_on_time_delivery()
        }
    
    # Helper calculation methods
    def _identify_excess_inventory(self) -> List[Dict]:
        """Identify excess inventory items"""
        # This would analyze inventory vs demand
        return []
    
    def _calculate_inventory_value(self) -> float:
        """Calculate total inventory value"""
        # This would sum up inventory value
        return 0.0
    
    def _calculate_turnover_rate(self) -> float:
        """Calculate inventory turnover rate"""
        # Annual COGS / Average inventory value
        return 4.5  # Placeholder
    
    def _calculate_demand_variability(self, forecast_data: Dict) -> float:
        """Calculate demand variability percentage"""
        # Coefficient of variation of demand
        return 15.0  # Placeholder
    
    def _analyze_supplier_concentration(self) -> float:
        """Analyze supplier concentration risk"""
        # Herfindahl index or top supplier percentage
        return 0.3  # Placeholder
    
    def _calculate_service_level(self) -> float:
        """Calculate current service level"""
        # Orders fulfilled / Total orders
        return 0.92  # Placeholder
    
    def _calculate_average_lead_time(self) -> float:
        """Calculate average lead time in days"""
        return 7.5  # Placeholder
    
    def _calculate_total_costs(self) -> Dict[str, float]:
        """Calculate total supply chain costs"""
        return {
            'inventory_holding': 50000,
            'transportation': 30000,
            'procurement': 20000,
            'expediting': 5000,
            'total': 105000
        }
    
    def _calculate_perfect_order_rate(self) -> float:
        """Calculate perfect order rate"""
        return 0.85  # Placeholder
    
    def _calculate_cash_cycle(self) -> float:
        """Calculate cash-to-cash cycle time in days"""
        return 45.0  # Placeholder
    
    def _calculate_cost_ratio(self) -> float:
        """Calculate supply chain costs as percentage of revenue"""
        return 0.12  # Placeholder
    
    def _calculate_forecast_accuracy(self) -> float:
        """Calculate forecast accuracy percentage"""
        return 0.82  # Placeholder
    
    def _calculate_fill_rate(self) -> float:
        """Calculate order fill rate"""
        return 0.94  # Placeholder
    
    def _calculate_on_time_delivery(self) -> float:
        """Calculate on-time delivery rate"""
        return 0.89  # Placeholder