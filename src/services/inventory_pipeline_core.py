"""
Core Inventory Management Pipeline Service
Extracted from beverly_comprehensive_erp.py for better modularity
"""
from .inventory_analyzer_core import InventoryAnalyzer


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
                    'action': 'Place urgent material orders'
                })

        return recommendations