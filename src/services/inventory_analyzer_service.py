"""
Inventory Analyzer Service
Extracted from beverly_comprehensive_erp.py (lines 957-1256)
PRESERVED EXACTLY - all methods, all calculations
"""

import math
import pandas as pd


class InventoryAnalyzer:
    """Inventory analysis as per INVENTORY_FORECASTING_IMPLEMENTATION.md spec"""

    def __init__(self, data_path=None):
        self.data_path = data_path  # Accept data_path for test compatibility
        self.safety_stock_multiplier = 1.5
        self.lead_time_days = 30

    def analyze_inventory_levels(self, current_inventory, forecast):
        """Compare current inventory against forecasted demand"""
        analysis = []

        for product in current_inventory:
            product_id = product.get('id', product.get('product_id', ''))
            quantity = product.get('quantity', product.get('stock', 0))

            # Get forecast for this product
            forecasted_demand = forecast.get(product_id, 0)

            # Calculate days of supply
            daily_demand = forecasted_demand / 30 if forecasted_demand > 0 else 0
            days_of_supply = quantity / daily_demand if daily_demand > 0 else 999

            # Calculate required inventory with safety stock
            required_inventory = (
                daily_demand * self.lead_time_days *
                self.safety_stock_multiplier
            )

            # Identify risk level using spec criteria
            risk_level = self.calculate_risk(
                current=quantity,
                required=required_inventory,
                days_supply=days_of_supply
            )

            analysis.append({
                'product_id': product_id,
                'current_stock': quantity,
                'forecasted_demand': forecasted_demand,
                'days_of_supply': days_of_supply,
                'required_inventory': required_inventory,
                'shortage_risk': risk_level,
                'reorder_needed': quantity < required_inventory,
                'reorder_quantity': max(0, required_inventory - quantity)
            })

        return analysis

    def calculate_risk(self, current, required, days_supply):
        """Calculate stockout risk level per spec"""
        if days_supply < 7:
            return 'CRITICAL'
        elif days_supply < 14:
            return 'HIGH'
        elif days_supply < 30:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_inventory(self, inventory_data=None):
        """Analyze inventory and return insights"""
        if inventory_data is None:
            # Use default empty data
            return {
                'total_items': 0,
                'critical_items': [],
                'recommendations': [],
                'summary': {
                    'critical_count': 0,
                    'high_risk_count': 0,
                    'healthy_count': 0
                }
            }
        
        # Convert to list of dicts if it's a DataFrame
        if hasattr(inventory_data, 'to_dict'):
            inventory_list = inventory_data.to_dict('records')
        else:
            inventory_list = inventory_data
        
        # Analyze each item
        critical_items = []
        high_risk_items = []
        healthy_items = []
        
        for item in inventory_list:
            balance = item.get('Planning Balance', item.get('quantity', 0))
            if balance < 0:
                critical_items.append(item)
            elif balance < 100:
                high_risk_items.append(item)
            else:
                healthy_items.append(item)
        
        return {
            'total_items': len(inventory_list),
            'critical_items': critical_items[:10],  # Top 10 critical
            'recommendations': self._generate_recommendations(critical_items, high_risk_items),
            'summary': {
                'critical_count': len(critical_items),
                'high_risk_count': len(high_risk_items),
                'healthy_count': len(healthy_items)
            }
        }
    
    def _calculate_eoq(self, annual_demand, ordering_cost=100, holding_cost_rate=0.25, unit_cost=10):
        """Calculate Economic Order Quantity"""
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
            return 0
        
        holding_cost = holding_cost_rate * unit_cost
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return round(eoq, 2)
    
    def _generate_recommendations(self, critical_items, high_risk_items):
        """Generate inventory recommendations"""
        recommendations = []
        
        for item in critical_items[:5]:  # Top 5 critical
            recommendations.append({
                'item': item.get('Desc#', item.get('Item', 'Unknown')),
                'action': 'URGENT ORDER',
                'quantity': abs(item.get('Planning Balance', 0)),
                'priority': 'CRITICAL'
            })
        
        for item in high_risk_items[:3]:  # Top 3 high risk
            recommendations.append({
                'item': item.get('Desc#', item.get('Item', 'Unknown')),
                'action': 'REORDER SOON',
                'quantity': 100 - item.get('Planning Balance', 0),
                'priority': 'HIGH'
            })
        
        return recommendations


class InventoryManagementPipeline:
    """Complete inventory management pipeline as per spec"""

    def __init__(self, supply_chain_ai=None):
        self.supply_chain_ai = supply_chain_ai
        self.inventory_analyzer = InventoryAnalyzer()

    def run_complete_analysis(self, sales_data=None, inventory_data=None, yarn_data=None):
        """Execute complete inventory analysis pipeline"""
        results = {}

        try:
            # Step 1: Use existing forecast or generate new one
            if self.supply_chain_ai and hasattr(self.supply_chain_ai, 'demand_forecast'):
                sales_forecast = self.supply_chain_ai.demand_forecast
            else:
                # Simple forecast based on historical data
                sales_forecast = self._generate_simple_forecast(sales_data)
            results['sales_forecast'] = sales_forecast

            # Step 2: Analyze inventory levels
            if inventory_data is not None:
                current_inventory = self._prepare_inventory_data(inventory_data)
                inventory_analysis = self.inventory_analyzer.analyze_inventory_levels(
                    current_inventory=current_inventory,
                    forecast=sales_forecast
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

        except Exception as e:
            print(f"Error in pipeline analysis: {e}")
            results['error'] = str(e)

        return results

    def generate_production_plan(self, inventory_analysis, forecast):
        """Create production plan based on inventory gaps"""
        production_plan = {}

        for item in inventory_analysis:
            if item['reorder_needed']:
                # Calculate production quantity
                product_id = item['product_id']
                production_qty = item['reorder_quantity'] + forecast.get(product_id, 0)
                production_plan[product_id] = {
                    'quantity': production_qty,
                    'priority': 'HIGH' if item['shortage_risk'] in ['CRITICAL', 'HIGH'] else 'NORMAL',
                    'risk_level': item['shortage_risk']
                }

        return production_plan

    def _prepare_inventory_data(self, inventory_data):
        """Convert DataFrame to list format for analyzer"""
        if hasattr(inventory_data, 'iterrows'):
            # It's a DataFrame
            inventory_list = []
            for idx, row in inventory_data.iterrows():
                inventory_list.append({
                    'id': str(row.get('Description', row.get('Item', idx))),
                    'quantity': row.get('Planning Balance', row.get('Stock', 0))
                })
            return inventory_list
        return inventory_data

    def _generate_simple_forecast(self, sales_data):
        """Generate simple forecast if no advanced forecasting available"""
        if sales_data is None:
            return {}

        # Simple moving average forecast
        forecast = {}
        if hasattr(sales_data, 'iterrows'):
            for _, row in sales_data.iterrows():
                item_id = str(row.get('Description', row.get('Item', '')))
                # Use last month's consumption as forecast
                forecast[item_id] = row.get('Consumed', row.get('Sales', 0)) * 1.1  # 10% growth

        return forecast

    def _calculate_material_requirements(self, production_plan, yarn_data):
        """Calculate material requirements based on production plan"""
        requirements = {}

        # Simple BOM assumption: 1 unit of product requires materials
        for product_id, plan in production_plan.items():
            requirements[product_id] = {
                'quantity_needed': plan['quantity'] * 1.2,  # 20% waste factor
                'priority': plan['priority']
            }

        return requirements

    def _analyze_material_shortages(self, requirements, yarn_data):
        """Analyze material shortages"""
        shortages = []

        for material_id, req in requirements.items():
            # Find current stock
            current_stock = 0
            if hasattr(yarn_data, 'iterrows'):
                for _, row in yarn_data.iterrows():
                    if str(row.get('Description', '')) == material_id:
                        current_stock = row.get('Planning Balance', 0)
                        break

            if current_stock < req['quantity_needed']:
                shortages.append({
                    'material_id': material_id,
                    'current_stock': current_stock,
                    'required': req['quantity_needed'],
                    'shortage': req['quantity_needed'] - current_stock,
                    'priority': req['priority']
                })

        return shortages

    def _generate_recommendations(self, analysis_results):
        """Generate actionable recommendations"""
        recommendations = []

        # Check inventory analysis
        if 'inventory_analysis' in analysis_results:
            critical_items = [
                item for item in analysis_results['inventory_analysis']
                if item['shortage_risk'] in ['CRITICAL', 'HIGH']
            ]
            if critical_items:
                recommendations.append({
                    'type': 'URGENT',
                    'message': f'{len(critical_items)} items at critical/high stockout risk',
                    'action': 'Expedite production and procurement'
                })

        # Check shortage analysis
        if 'shortage_analysis' in analysis_results:
            if analysis_results['shortage_analysis']:
                recommendations.append({
                    'type': 'PROCUREMENT',
                    'message': f'{len(analysis_results["shortage_analysis"])} material shortages detected',
                    'action': 'Review and order materials urgently'
                })

        # Check production plan
        if 'production_plan' in analysis_results:
            high_priority = [
                p for p in analysis_results['production_plan'].values()
                if p['priority'] == 'HIGH'
            ]
            if high_priority:
                recommendations.append({
                    'type': 'PRODUCTION',
                    'message': f'{len(high_priority)} high-priority production orders',
                    'action': 'Schedule priority production runs'
                })

        return recommendations