"""
Enhanced AI Production Suggestions Module
Provides intelligent, context-aware production recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedProductionSuggestions:
    """Advanced AI-driven production suggestion engine"""
    
    def __init__(self, analyzer=None):
        """Initialize with analyzer instance for data access"""
        self.analyzer = analyzer
        self.suggestions_cache = {}
        self.last_update = None
        
        # Enhanced thresholds and parameters
        self.min_production_batch = 500  # Minimum economic batch size
        self.safety_stock_days = 14  # Target safety stock
        self.lead_time_days = 21  # Production lead time
        self.forecast_horizon_days = 60  # Extended forecast horizon
        
        # Advanced scoring weights
        self.scoring_weights = {
            'demand_urgency': 0.30,
            'inventory_coverage': 0.25,
            'material_availability': 0.20,
            'profitability': 0.15,
            'customer_priority': 0.10
        }
        
    def calculate_demand_forecast(self, style: str, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced demand forecast using multiple methods"""
        forecast = {
            'point_forecast': 0,
            'upper_bound': 0,
            'lower_bound': 0,
            'confidence': 0,
            'method': 'none'
        }
        
        if historical_data.empty:
            # No history - use industry defaults
            forecast['point_forecast'] = self.min_production_batch
            forecast['upper_bound'] = self.min_production_batch * 1.5
            forecast['lower_bound'] = self.min_production_batch * 0.5
            forecast['confidence'] = 0.3
            forecast['method'] = 'default'
            return forecast
        
        # Calculate various statistics
        daily_sales = historical_data.groupby('date')['quantity'].sum() if 'date' in historical_data.columns else pd.Series()
        
        if not daily_sales.empty:
            # Moving averages for trend analysis
            ma_7 = daily_sales.rolling(7, min_periods=1).mean()
            ma_30 = daily_sales.rolling(30, min_periods=1).mean()
            
            # Trend detection
            recent_trend = ma_7.iloc[-1] / ma_30.iloc[-1] if ma_30.iloc[-1] > 0 else 1
            
            # Seasonality detection (simplified)
            seasonal_factor = 1.0
            if len(daily_sales) > 90:
                # Compare last 30 days to previous 30 days
                recent_30 = daily_sales.iloc[-30:].mean()
                previous_30 = daily_sales.iloc[-60:-30].mean()
                if previous_30 > 0:
                    seasonal_factor = recent_30 / previous_30
            
            # Calculate forecast
            base_forecast = ma_7.iloc[-1] * self.forecast_horizon_days
            
            # Apply adjustments
            forecast['point_forecast'] = base_forecast * recent_trend * seasonal_factor
            
            # Calculate confidence bounds based on historical volatility
            std_dev = daily_sales.std()
            forecast['upper_bound'] = forecast['point_forecast'] + (2 * std_dev * np.sqrt(self.forecast_horizon_days))
            forecast['lower_bound'] = max(0, forecast['point_forecast'] - (2 * std_dev * np.sqrt(self.forecast_horizon_days)))
            
            # Confidence based on data quality and consistency
            cv = std_dev / daily_sales.mean() if daily_sales.mean() > 0 else 1
            forecast['confidence'] = max(0.1, min(0.9, 1 - cv))
            forecast['method'] = 'time_series_analysis'
        else:
            # Fallback to simple average
            total_demand = historical_data['quantity'].sum() if 'quantity' in historical_data.columns else 0
            days_of_data = 30  # Assume 30 days of data
            daily_avg = total_demand / days_of_data if days_of_data > 0 else 0
            
            forecast['point_forecast'] = daily_avg * self.forecast_horizon_days
            forecast['upper_bound'] = forecast['point_forecast'] * 1.3
            forecast['lower_bound'] = forecast['point_forecast'] * 0.7
            forecast['confidence'] = 0.5
            forecast['method'] = 'simple_average'
        
        return forecast
    
    def analyze_inventory_position(self, style: str) -> Dict[str, Any]:
        """Comprehensive inventory position analysis"""
        position = {
            'on_hand': 0,
            'in_production': 0,
            'in_transit': 0,
            'allocated': 0,
            'available': 0,
            'days_of_supply': 0,
            'stockout_risk': 'unknown'
        }
        
        if not self.analyzer:
            return position
        
        # Check all inventory stages
        if hasattr(self.analyzer, 'inventory_data') and self.analyzer.inventory_data:
            for stage, inv_data in self.analyzer.inventory_data.items():
                if isinstance(inv_data, dict) and 'data' in inv_data:
                    inv_df = inv_data['data']
                else:
                    inv_df = inv_data
                
                if inv_df is not None and isinstance(inv_df, pd.DataFrame):
                    if 'Style#' in inv_df.columns:
                        style_inv = inv_df[inv_df['Style#'] == style]
                        if not style_inv.empty:
                            # Map stages to position categories
                            if stage in ['F01', 'Finished']:
                                position['on_hand'] += style_inv.get('Balance', style_inv.get('On Hand (lbs)', pd.Series())).sum()
                            elif stage in ['G00', 'G02', 'WIP']:
                                position['in_production'] += style_inv.get('Balance', style_inv.get('On Hand (lbs)', pd.Series())).sum()
                            elif stage in ['I01', 'Transit']:
                                position['in_transit'] += style_inv.get('Balance', style_inv.get('On Hand (lbs)', pd.Series())).sum()
        
        # Check knit orders for production pipeline
        if hasattr(self.analyzer, 'knit_orders_data') and self.analyzer.knit_orders_data is not None:
            style_orders = self.analyzer.knit_orders_data[
                self.analyzer.knit_orders_data['Style#'] == style
            ] if 'Style#' in self.analyzer.knit_orders_data.columns else pd.DataFrame()
            
            if not style_orders.empty:
                position['in_production'] += style_orders.get('Balance (lbs)', pd.Series()).sum()
        
        # Check sales orders for allocations
        if hasattr(self.analyzer, 'sales_orders_data') and self.analyzer.sales_orders_data is not None:
            style_sales = self.analyzer.sales_orders_data[
                self.analyzer.sales_orders_data['Style#'] == style
            ] if 'Style#' in self.analyzer.sales_orders_data.columns else pd.DataFrame()
            
            if not style_sales.empty:
                position['allocated'] = style_sales.get('Balance', pd.Series()).sum()
        
        # Calculate available inventory
        position['available'] = position['on_hand'] + position['in_production'] + position['in_transit'] - position['allocated']
        
        # Calculate days of supply
        historical_demand = self.get_historical_demand(style)
        if historical_demand > 0:
            daily_demand = historical_demand / 30  # Assume 30-day historical period
            position['days_of_supply'] = position['available'] / daily_demand if daily_demand > 0 else 999
            
            # Assess stockout risk
            if position['days_of_supply'] < 7:
                position['stockout_risk'] = 'critical'
            elif position['days_of_supply'] < 14:
                position['stockout_risk'] = 'high'
            elif position['days_of_supply'] < 30:
                position['stockout_risk'] = 'medium'
            else:
                position['stockout_risk'] = 'low'
        
        return position
    
    def check_material_availability(self, style: str, quantity: float) -> Tuple[bool, List[str], float]:
        """Check if materials are available for production"""
        available = True
        issues = []
        availability_score = 1.0
        
        if not self.analyzer or not hasattr(self.analyzer, 'bom_data'):
            return True, [], 1.0  # Assume available if no BOM data
        
        bom_df = self.analyzer.bom_data
        if isinstance(bom_df, dict) and 'data' in bom_df:
            bom_df = bom_df['data']
        
        if not isinstance(bom_df, pd.DataFrame) or bom_df.empty:
            return True, [], 1.0
        
        # Get BOM for this style
        style_bom = bom_df[bom_df['Style#'] == style] if 'Style#' in bom_df.columns else pd.DataFrame()
        
        if style_bom.empty:
            issues.append(f"No BOM found for style {style}")
            return False, issues, 0.0
        
        # Get yarn availability
        yarn_availability = {}
        if hasattr(self.analyzer, 'yarn_data') and self.analyzer.yarn_data is not None:
            for idx, yarn_row in self.analyzer.yarn_data.iterrows():
                yarn_id = yarn_row.get('Desc#', '')
                planning_balance = yarn_row.get('Planning_Balance', 0)
                yarn_availability[yarn_id] = planning_balance
        
        # Check each component
        total_components = len(style_bom)
        available_components = 0
        
        for _, bom_row in style_bom.iterrows():
            yarn_id = bom_row.get('Desc#', bom_row.get('Yarn_ID', ''))
            bom_percent = bom_row.get('bom_percentage', bom_row.get('BOM_Percent', 100))
            
            if yarn_id and str(yarn_id) != 'nan':
                yarn_needed = (quantity * bom_percent / 100) if bom_percent > 0 else quantity
                available_yarn = yarn_availability.get(yarn_id, 0)
                
                if available_yarn >= yarn_needed:
                    available_components += 1
                else:
                    shortage = yarn_needed - available_yarn
                    issues.append(f"Yarn {yarn_id}: need {yarn_needed:.0f} lbs, short {shortage:.0f} lbs")
                    available = False
        
        # Calculate availability score
        if total_components > 0:
            availability_score = available_components / total_components
        
        return available, issues, availability_score
    
    def calculate_profitability_score(self, style: str) -> float:
        """Calculate profitability score based on historical data"""
        score = 0.5  # Default middle score
        
        if not self.analyzer or not hasattr(self.analyzer, 'sales_data'):
            return score
        
        sales_df = self.analyzer.sales_data
        if sales_df is None or sales_df.empty:
            return score
        
        # Get sales data for this style
        if 'Style#' in sales_df.columns:
            style_sales = sales_df[sales_df['Style#'] == style]
            
            if not style_sales.empty:
                # Calculate average price if available
                if 'Unit Price' in style_sales.columns:
                    avg_price = style_sales['Unit Price'].mean()
                    # Normalize price score (assume $10-50 range)
                    score = min(1.0, max(0.0, (avg_price - 10) / 40))
                
                # Adjust for order frequency
                order_count = len(style_sales)
                if order_count > 10:
                    score = min(1.0, score + 0.2)
                elif order_count < 3:
                    score = max(0.0, score - 0.2)
        
        return score
    
    def get_customer_priority(self, style: str) -> float:
        """Determine customer priority for a style"""
        priority = 0.5  # Default priority
        
        if not self.analyzer or not hasattr(self.analyzer, 'sales_orders_data'):
            return priority
        
        sales_orders = self.analyzer.sales_orders_data
        if sales_orders is None or sales_orders.empty:
            return priority
        
        # Check for urgent or overdue orders
        if 'Style#' in sales_orders.columns:
            style_orders = sales_orders[sales_orders['Style#'] == style]
            
            if not style_orders.empty:
                # Check for priority indicators
                if 'Priority' in style_orders.columns:
                    high_priority = style_orders['Priority'].str.contains('High|Urgent|Rush', case=False, na=False).any()
                    if high_priority:
                        priority = 0.9
                
                # Check for overdue orders
                if 'Start_Ship' in style_orders.columns:
                    try:
                        style_orders['Start_Ship_Date'] = pd.to_datetime(style_orders['Start_Ship'])
                        overdue = (style_orders['Start_Ship_Date'] < datetime.now()).any()
                        if overdue:
                            priority = max(priority, 0.8)
                    except:
                        pass
        
        return priority
    
    def get_historical_demand(self, style: str) -> float:
        """Get historical demand for a style"""
        demand = 0
        
        if not self.analyzer:
            return demand
        
        # Check sales data
        if hasattr(self.analyzer, 'sales_data') and self.analyzer.sales_data is not None:
            sales_df = self.analyzer.sales_data
            if 'Style#' in sales_df.columns:
                style_sales = sales_df[sales_df['Style#'] == style]
                if not style_sales.empty:
                    if 'Ordered' in style_sales.columns:
                        demand = style_sales['Ordered'].sum()
                    elif 'Quantity' in style_sales.columns:
                        demand = style_sales['Quantity'].sum()
        
        return float(demand) if demand else 0
    
    def calculate_priority_score(self, suggestion: Dict[str, Any]) -> float:
        """Calculate comprehensive priority score"""
        score = 0
        
        # Demand urgency (0-100)
        if suggestion['stockout_risk'] == 'critical':
            score += self.scoring_weights['demand_urgency'] * 100
        elif suggestion['stockout_risk'] == 'high':
            score += self.scoring_weights['demand_urgency'] * 75
        elif suggestion['stockout_risk'] == 'medium':
            score += self.scoring_weights['demand_urgency'] * 50
        else:
            score += self.scoring_weights['demand_urgency'] * 25
        
        # Inventory coverage (0-100)
        days_of_supply = suggestion.get('current_coverage_days', 0)
        if days_of_supply < 7:
            coverage_score = 100
        elif days_of_supply < 14:
            coverage_score = 75
        elif days_of_supply < 30:
            coverage_score = 50
        else:
            coverage_score = 25
        score += self.scoring_weights['inventory_coverage'] * coverage_score
        
        # Material availability (0-100)
        material_score = suggestion.get('material_availability_score', 0) * 100
        score += self.scoring_weights['material_availability'] * material_score
        
        # Profitability (0-100)
        profit_score = suggestion.get('profitability_score', 0.5) * 100
        score += self.scoring_weights['profitability'] * profit_score
        
        # Customer priority (0-100)
        customer_score = suggestion.get('customer_priority', 0.5) * 100
        score += self.scoring_weights['customer_priority'] * customer_score
        
        return round(score, 1)
    
    def generate_suggestions(self, limit: int = 20) -> Dict[str, Any]:
        """Generate comprehensive production suggestions"""
        suggestions = []
        
        try:
            # Get all styles from various data sources
            all_styles = set()
            
            # From BOM data
            if hasattr(self.analyzer, 'bom_data') and self.analyzer.bom_data is not None:
                bom_df = self.analyzer.bom_data
                if isinstance(bom_df, dict) and 'data' in bom_df:
                    bom_df = bom_df['data']
                if isinstance(bom_df, pd.DataFrame) and 'Style#' in bom_df.columns:
                    all_styles.update(bom_df['Style#'].unique())
            
            # From sales orders
            if hasattr(self.analyzer, 'sales_orders_data') and self.analyzer.sales_orders_data is not None:
                if 'Style#' in self.analyzer.sales_orders_data.columns:
                    all_styles.update(self.analyzer.sales_orders_data['Style#'].unique())
            
            # From inventory
            if hasattr(self.analyzer, 'inventory_data') and self.analyzer.inventory_data:
                for stage, inv_data in self.analyzer.inventory_data.items():
                    if isinstance(inv_data, dict) and 'data' in inv_data:
                        inv_df = inv_data['data']
                    else:
                        inv_df = inv_data
                    if inv_df is not None and isinstance(inv_df, pd.DataFrame) and 'Style#' in inv_df.columns:
                        all_styles.update(inv_df['Style#'].unique())
            
            # Process each style
            for style in list(all_styles)[:100]:  # Limit to 100 styles for performance
                if not style or pd.isna(style):
                    continue
                
                # Get inventory position
                inventory_position = self.analyze_inventory_position(style)
                
                # Skip if already well-stocked
                if inventory_position['days_of_supply'] > self.forecast_horizon_days:
                    continue
                
                # Get historical demand
                historical_demand = self.get_historical_demand(style)
                
                # Calculate forecast
                historical_data = pd.DataFrame()  # Would need actual historical data
                forecast = self.calculate_demand_forecast(style, historical_data)
                
                # Calculate production requirement
                current_total = (inventory_position['on_hand'] + 
                               inventory_position['in_production'] + 
                               inventory_position['in_transit'])
                
                # Account for safety stock
                target_inventory = forecast['point_forecast'] + (forecast['point_forecast'] * 0.2)  # 20% safety stock
                net_requirement = target_inventory - current_total
                
                # Apply minimum batch size
                if 0 < net_requirement < self.min_production_batch:
                    net_requirement = self.min_production_batch
                
                if net_requirement <= 0:
                    continue
                
                # Check material availability
                material_available, material_issues, material_score = self.check_material_availability(style, net_requirement)
                
                # Get additional scoring factors
                profitability = self.calculate_profitability_score(style)
                customer_priority = self.get_customer_priority(style)
                
                # Build suggestion
                suggestion = {
                    'style': style,
                    'suggested_quantity_lbs': round(net_requirement, 0),
                    'current_inventory': round(inventory_position['on_hand'], 0),
                    'current_production': round(inventory_position['in_production'], 0),
                    'in_transit': round(inventory_position['in_transit'], 0),
                    'allocated': round(inventory_position['allocated'], 0),
                    'available': round(inventory_position['available'], 0),
                    'historical_demand': round(historical_demand, 0),
                    'forecasted_demand': round(forecast['point_forecast'], 0),
                    'forecast_confidence': round(forecast['confidence'], 2),
                    'forecast_method': forecast['method'],
                    'current_coverage_days': round(inventory_position['days_of_supply'], 1),
                    'target_coverage_days': round(self.forecast_horizon_days, 1),
                    'stockout_risk': inventory_position['stockout_risk'],
                    'yarn_available': material_available,
                    'material_availability_score': material_score,
                    'material_status': 'All materials available' if material_available else f"{len(material_issues)} material issues",
                    'material_issues': material_issues[:3],  # Top 3 issues
                    'profitability_score': profitability,
                    'customer_priority': customer_priority,
                    'priority_score': 0,  # Will be calculated
                    'rationale': ''  # Will be generated
                }
                
                # Calculate priority score
                suggestion['priority_score'] = self.calculate_priority_score(suggestion)
                
                # Generate detailed rationale
                rationale_parts = []
                
                if suggestion['stockout_risk'] in ['critical', 'high']:
                    rationale_parts.append(f"{suggestion['stockout_risk'].capitalize()} stockout risk with {suggestion['current_coverage_days']:.1f} days of supply")
                
                if historical_demand > 0:
                    rationale_parts.append(f"Historical demand: {historical_demand:,.0f} lbs")
                
                if forecast['confidence'] > 0.7:
                    rationale_parts.append(f"High confidence forecast ({forecast['confidence']:.0%})")
                elif forecast['confidence'] < 0.3:
                    rationale_parts.append(f"Low confidence forecast - monitor closely")
                
                if not material_available:
                    rationale_parts.append(f"Material constraints: {len(material_issues)} issues")
                
                if customer_priority > 0.7:
                    rationale_parts.append("High priority customer orders")
                
                if profitability > 0.7:
                    rationale_parts.append("High margin product")
                
                suggestion['rationale'] = '. '.join(rationale_parts) if rationale_parts else 'Standard replenishment requirement'
                
                suggestions.append(suggestion)
            
            # Sort by priority
            suggestions.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Filter to top suggestions
            top_suggestions = suggestions[:limit]
            
            # Calculate summary statistics
            summary = {
                'total_suggestions': len(top_suggestions),
                'critical_stockout_risk': len([s for s in top_suggestions if s['stockout_risk'] == 'critical']),
                'high_stockout_risk': len([s for s in top_suggestions if s['stockout_risk'] == 'high']),
                'material_available': len([s for s in top_suggestions if s['yarn_available']]),
                'material_shortage': len([s for s in top_suggestions if not s['yarn_available']]),
                'total_suggested_production': sum(s['suggested_quantity_lbs'] for s in top_suggestions),
                'average_priority_score': round(np.mean([s['priority_score'] for s in top_suggestions]), 1) if top_suggestions else 0,
                'average_confidence': round(np.mean([s['forecast_confidence'] for s in top_suggestions]), 2) if top_suggestions else 0
            }
            
            # Add recommendations
            recommendations = []
            
            if summary['critical_stockout_risk'] > 0:
                recommendations.append(f"URGENT: {summary['critical_stockout_risk']} styles at critical stockout risk - prioritize immediately")
            
            if summary['material_shortage'] > summary['material_available']:
                recommendations.append("Material constraints affecting majority of suggestions - review yarn procurement")
            
            if summary['average_confidence'] < 0.5:
                recommendations.append("Low forecast confidence - consider gathering more historical data")
            
            if summary['total_suggested_production'] > 50000:
                recommendations.append(f"Large production volume ({summary['total_suggested_production']:,.0f} lbs) - verify capacity availability")
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'suggestions': top_suggestions,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'summary': {
                    'total_suggestions': 0,
                    'material_available': 0,
                    'material_shortage': 0,
                    'total_suggested_production': 0
                },
                'suggestions': []
            }

def create_enhanced_suggestions_endpoint(analyzer):
    """Factory function to create suggestions with analyzer instance"""
    engine = EnhancedProductionSuggestions(analyzer)
    return engine.generate_suggestions()