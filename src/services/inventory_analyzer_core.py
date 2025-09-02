"""
Core Inventory Analyzer Service
Extracted from beverly_comprehensive_erp.py for better modularity
"""
import math


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