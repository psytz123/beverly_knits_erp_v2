#!/usr/bin/env python3
"""
Beverly Knits ERP - Inventory Management Pipeline Service
Extracted from beverly_comprehensive_erp.py (lines 419-586)
Complete inventory management pipeline orchestration
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for inventory pipeline service"""
    waste_factor: float = 1.2  # 20% waste factor for material requirements
    growth_forecast: float = 1.1  # 10% growth for simple forecasting
    critical_threshold: int = 5  # Number of critical items to trigger urgent action


class InventoryManagementPipelineService:
    """
    Complete inventory management pipeline orchestration
    Extracted from monolith for modular architecture
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize inventory management pipeline service
        
        Args:
            config: Optional configuration for pipeline
        """
        self.config = config or PipelineConfig()
        self.supply_chain_ai = None
        self.inventory_analyzer = None
        
        logger.info(f"InventoryManagementPipelineService initialized")
    
    def set_dependencies(self, inventory_analyzer=None, supply_chain_ai=None):
        """
        Set service dependencies
        
        Args:
            inventory_analyzer: Inventory analyzer service instance
            supply_chain_ai: Supply chain AI instance (optional)
        """
        self.inventory_analyzer = inventory_analyzer
        self.supply_chain_ai = supply_chain_ai
        logger.debug("Dependencies set for InventoryManagementPipelineService")
    
    def run_complete_analysis(self, 
                            sales_data: Optional[pd.DataFrame] = None, 
                            inventory_data: Optional[pd.DataFrame] = None, 
                            yarn_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute complete inventory analysis pipeline
        
        Args:
            sales_data: Historical sales data
            inventory_data: Current inventory levels
            yarn_data: Yarn/material inventory data
            
        Returns:
            Dictionary with complete analysis results
        """
        results = {}
        
        try:
            # Step 1: Use existing forecast or generate new one
            if self.supply_chain_ai and hasattr(self.supply_chain_ai, 'demand_forecast'):
                sales_forecast = self.supply_chain_ai.demand_forecast
                logger.info("Using existing demand forecast from supply chain AI")
            else:
                # Simple forecast based on historical data
                sales_forecast = self._generate_simple_forecast(sales_data)
                logger.info(f"Generated simple forecast for {len(sales_forecast)} items")
            
            results['sales_forecast'] = sales_forecast
            
            # Step 2: Analyze inventory levels
            if inventory_data is not None:
                current_inventory = self._prepare_inventory_data(inventory_data)
                
                if self.inventory_analyzer:
                    inventory_analysis = self.inventory_analyzer.analyze_inventory_levels(
                        current_inventory=current_inventory,
                        forecast=sales_forecast
                    )
                else:
                    # Fallback if analyzer not available
                    inventory_analysis = self._simple_inventory_analysis(
                        current_inventory, sales_forecast
                    )
                
                results['inventory_analysis'] = inventory_analysis
                
                # Step 3: Generate production plan
                production_plan = self.generate_production_plan(
                    inventory_analysis=inventory_analysis,
                    forecast=sales_forecast
                )
                results['production_plan'] = production_plan
                
                # Step 4: Calculate material requirements
                if yarn_data is not None:
                    yarn_requirements = self._calculate_material_requirements(
                        production_plan, yarn_data
                    )
                    results['yarn_requirements'] = yarn_requirements
                    
                    # Step 5: Detect shortages
                    shortage_analysis = self._analyze_material_shortages(
                        yarn_requirements, yarn_data
                    )
                    results['shortage_analysis'] = shortage_analysis
            
            # Step 6: Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Add summary
            results['summary'] = self._generate_summary(results)
            
            logger.info(f"Pipeline analysis complete with {len(results)} components")
            
        except Exception as e:
            logger.error(f"Error in pipeline analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def generate_production_plan(self, 
                                inventory_analysis: List[Dict], 
                                forecast: Dict[str, float]) -> Dict[str, Any]:
        """
        Create production plan based on inventory gaps
        
        Args:
            inventory_analysis: Results from inventory analysis
            forecast: Sales forecast dictionary
            
        Returns:
            Production plan with quantities and priorities
        """
        production_plan = {}
        total_production_value = 0
        
        for item in inventory_analysis:
            if item.get('reorder_needed', False):
                # Calculate production quantity
                product_id = item['product_id']
                reorder_qty = item.get('reorder_quantity', 0)
                forecast_qty = forecast.get(product_id, 0)
                
                production_qty = reorder_qty + forecast_qty
                
                # Determine priority based on risk
                risk_level = item.get('shortage_risk', 'LOW')
                priority = 'HIGH' if risk_level in ['CRITICAL', 'HIGH'] else 'NORMAL'
                
                production_plan[product_id] = {
                    'quantity': production_qty,
                    'priority': priority,
                    'risk_level': risk_level,
                    'reorder_point': item.get('reorder_point', 0),
                    'current_stock': item.get('current_stock', 0)
                }
                
                total_production_value += production_qty
        
        # Add plan summary
        production_plan['_summary'] = {
            'total_items': len(production_plan) - 1,  # Exclude summary itself
            'total_quantity': total_production_value,
            'high_priority_items': sum(1 for k, v in production_plan.items() 
                                      if k != '_summary' and v['priority'] == 'HIGH')
        }
        
        logger.info(f"Generated production plan for {production_plan['_summary']['total_items']} items")
        
        return production_plan
    
    def _prepare_inventory_data(self, inventory_data: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to list format for analyzer
        
        Args:
            inventory_data: Inventory DataFrame
            
        Returns:
            List of inventory dictionaries
        """
        if hasattr(inventory_data, 'iterrows'):
            # It's a DataFrame
            inventory_list = []
            for idx, row in inventory_data.iterrows():
                inventory_list.append({
                    'product_id': str(row.get('Description', row.get('Item', idx))),
                    'quantity': float(row.get('Planning Balance', row.get('Stock', 0)))
                })
            return inventory_list
        return inventory_data
    
    def _generate_simple_forecast(self, sales_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Generate simple forecast if no advanced forecasting available
        
        Args:
            sales_data: Historical sales data
            
        Returns:
            Forecast dictionary
        """
        if sales_data is None:
            return {}
        
        # Simple moving average forecast
        forecast = {}
        if hasattr(sales_data, 'iterrows'):
            for _, row in sales_data.iterrows():
                item_id = str(row.get('Description', row.get('Item', '')))
                # Use last month's consumption as forecast with growth factor
                consumed = abs(row.get('Consumed', row.get('Sales', 0)))
                forecast[item_id] = consumed * self.config.growth_forecast
        
        return forecast
    
    def _calculate_material_requirements(self, 
                                        production_plan: Dict, 
                                        yarn_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate material requirements based on production plan
        
        Args:
            production_plan: Production plan dictionary
            yarn_data: Yarn inventory data
            
        Returns:
            Material requirements dictionary
        """
        requirements = {}
        
        # Apply waste factor to production quantities
        for product_id, plan in production_plan.items():
            if product_id == '_summary':
                continue
                
            requirements[product_id] = {
                'quantity_needed': plan['quantity'] * self.config.waste_factor,
                'priority': plan['priority'],
                'base_quantity': plan['quantity'],
                'waste_factor': self.config.waste_factor
            }
        
        return requirements
    
    def _analyze_material_shortages(self, 
                                   requirements: Dict, 
                                   yarn_data: pd.DataFrame) -> List[Dict]:
        """
        Analyze material shortages against current inventory
        
        Args:
            requirements: Material requirements
            yarn_data: Current yarn inventory
            
        Returns:
            List of shortage items
        """
        shortages = []
        
        for material_id, req in requirements.items():
            # Find current stock
            current_stock = 0
            if hasattr(yarn_data, 'iterrows'):
                for _, row in yarn_data.iterrows():
                    if str(row.get('Description', '')) == material_id:
                        current_stock = float(row.get('Planning Balance', 0))
                        break
            
            if current_stock < req['quantity_needed']:
                shortage_amount = req['quantity_needed'] - current_stock
                shortages.append({
                    'material_id': material_id,
                    'current_stock': current_stock,
                    'required': req['quantity_needed'],
                    'shortage': shortage_amount,
                    'priority': req['priority'],
                    'coverage_days': int(current_stock / (req['quantity_needed'] / 30)) 
                                   if req['quantity_needed'] > 0 else 0
                })
        
        # Sort by priority and shortage amount
        shortages.sort(key=lambda x: (x['priority'] != 'HIGH', -x['shortage']))
        
        return shortages
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """
        Generate actionable recommendations based on analysis
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check inventory analysis
        if 'inventory_analysis' in analysis_results:
            critical_items = [
                item for item in analysis_results['inventory_analysis']
                if item.get('shortage_risk') in ['CRITICAL', 'HIGH']
            ]
            
            if len(critical_items) >= self.config.critical_threshold:
                recommendations.append({
                    'type': 'URGENT',
                    'severity': 'CRITICAL',
                    'message': f'{len(critical_items)} items at critical/high stockout risk',
                    'action': 'Expedite production and procurement immediately',
                    'items': [item['product_id'] for item in critical_items[:5]]  # Top 5
                })
            elif critical_items:
                recommendations.append({
                    'type': 'WARNING',
                    'severity': 'HIGH',
                    'message': f'{len(critical_items)} items need attention',
                    'action': 'Review and prioritize production schedule',
                    'items': [item['product_id'] for item in critical_items]
                })
        
        # Check shortage analysis
        if 'shortage_analysis' in analysis_results:
            shortages = analysis_results['shortage_analysis']
            if shortages:
                high_priority_shortages = [s for s in shortages if s['priority'] == 'HIGH']
                
                if high_priority_shortages:
                    recommendations.append({
                        'type': 'PROCUREMENT',
                        'severity': 'HIGH',
                        'message': f'{len(high_priority_shortages)} high-priority material shortages',
                        'action': 'Place urgent material orders',
                        'total_shortage_value': sum(s['shortage'] for s in high_priority_shortages)
                    })
                else:
                    recommendations.append({
                        'type': 'PROCUREMENT',
                        'severity': 'MEDIUM',
                        'message': f'{len(shortages)} material shortages detected',
                        'action': 'Schedule material orders within lead time'
                    })
        
        # Check production plan
        if 'production_plan' in analysis_results:
            plan = analysis_results['production_plan']
            if '_summary' in plan and plan['_summary']['high_priority_items'] > 10:
                recommendations.append({
                    'type': 'CAPACITY',
                    'severity': 'MEDIUM',
                    'message': f'{plan["_summary"]["high_priority_items"]} high-priority production items',
                    'action': 'Review production capacity and consider overtime'
                })
        
        return recommendations
    
    def _simple_inventory_analysis(self, 
                                  current_inventory: List[Dict], 
                                  forecast: Dict[str, float]) -> List[Dict]:
        """
        Fallback simple inventory analysis if analyzer not available
        
        Args:
            current_inventory: Current inventory levels
            forecast: Sales forecast
            
        Returns:
            Simple analysis results
        """
        analysis = []
        
        for item in current_inventory:
            product_id = item['product_id']
            current_stock = item['quantity']
            forecast_demand = forecast.get(product_id, 0)
            
            # Simple reorder logic
            reorder_point = forecast_demand * 2  # 2 months of stock
            reorder_needed = current_stock < reorder_point
            
            # Simple risk assessment
            if current_stock <= 0:
                risk = 'CRITICAL'
            elif current_stock < forecast_demand:
                risk = 'HIGH'
            elif current_stock < forecast_demand * 2:
                risk = 'MEDIUM'
            else:
                risk = 'LOW'
            
            analysis.append({
                'product_id': product_id,
                'current_stock': current_stock,
                'forecast_demand': forecast_demand,
                'reorder_point': reorder_point,
                'reorder_needed': reorder_needed,
                'reorder_quantity': max(0, reorder_point - current_stock),
                'shortage_risk': risk
            })
        
        return analysis
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """
        Generate executive summary of pipeline results
        
        Args:
            results: Complete analysis results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'components_analyzed': len([k for k in results.keys() if k not in ['error', 'summary']]),
            'has_errors': 'error' in results
        }
        
        if 'inventory_analysis' in results:
            inventory = results['inventory_analysis']
            summary['inventory'] = {
                'total_items': len(inventory),
                'critical_items': sum(1 for i in inventory if i.get('shortage_risk') == 'CRITICAL'),
                'reorder_needed': sum(1 for i in inventory if i.get('reorder_needed'))
            }
        
        if 'production_plan' in results and '_summary' in results['production_plan']:
            summary['production'] = results['production_plan']['_summary']
        
        if 'shortage_analysis' in results:
            shortages = results['shortage_analysis']
            summary['shortages'] = {
                'total_shortages': len(shortages),
                'high_priority': sum(1 for s in shortages if s['priority'] == 'HIGH')
            }
        
        if 'recommendations' in results:
            summary['recommendations'] = {
                'total': len(results['recommendations']),
                'urgent': sum(1 for r in results['recommendations'] if r.get('severity') == 'CRITICAL')
            }
        
        return summary


def test_inventory_pipeline_service():
    """Test the inventory pipeline service"""
    print("=" * 80)
    print("Testing InventoryManagementPipelineService")
    print("=" * 80)
    
    # Create service
    config = PipelineConfig(
        waste_factor=1.25,
        growth_forecast=1.15,
        critical_threshold=3
    )
    service = InventoryManagementPipelineService(config)
    
    # Create sample data
    sales_data = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'],
        'Consumed': [-100, -50, -200]  # Negative values as consumed
    })
    
    inventory_data = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'],
        'Planning Balance': [80, 30, 250]
    })
    
    yarn_data = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'],
        'Planning Balance': [80, 30, 250]
    })
    
    # Run complete analysis
    print("\n1. Running Complete Pipeline Analysis:")
    results = service.run_complete_analysis(sales_data, inventory_data, yarn_data)
    
    print(f"\n  Components analyzed: {len(results)}")
    
    if 'sales_forecast' in results:
        print(f"  ✓ Sales forecast generated for {len(results['sales_forecast'])} items")
    
    if 'inventory_analysis' in results:
        print(f"  ✓ Inventory analyzed for {len(results['inventory_analysis'])} items")
    
    if 'production_plan' in results:
        plan = results['production_plan']
        if '_summary' in plan:
            print(f"  ✓ Production plan created for {plan['_summary']['total_items']} items")
            print(f"    High priority: {plan['_summary']['high_priority_items']}")
    
    if 'shortage_analysis' in results:
        print(f"  ✓ Shortages detected: {len(results['shortage_analysis'])}")
    
    if 'recommendations' in results:
        print(f"  ✓ Recommendations generated: {len(results['recommendations'])}")
        for rec in results['recommendations']:
            print(f"    - [{rec['type']}] {rec['message']}")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\n2. Executive Summary:")
        print(f"  Timestamp: {summary['timestamp']}")
        if 'inventory' in summary:
            print(f"  Inventory: {summary['inventory']['total_items']} items, "
                  f"{summary['inventory']['critical_items']} critical")
        if 'shortages' in summary:
            print(f"  Shortages: {summary['shortages']['total_shortages']} total, "
                  f"{summary['shortages']['high_priority']} high priority")
    
    print("\n" + "=" * 80)
    print("✅ InventoryManagementPipelineService test complete")


if __name__ == "__main__":
    test_inventory_pipeline_service()