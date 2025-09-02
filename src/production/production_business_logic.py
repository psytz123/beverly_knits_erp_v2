"""
Production Business Logic for Beverly Knits ERP
Implements comprehensive business rules for production planning and suggestions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ProductionBusinessLogic:
    """
    Encapsulates all business logic for production planning decisions
    """
    
    def __init__(self):
        # Business Constants
        self.MIN_PRODUCTION_BATCH = 100  # Minimum batch size in lbs
        self.MAX_PRODUCTION_BATCH = 5000  # Maximum batch size in lbs
        self.SAFETY_STOCK_DAYS = 14  # Days of safety stock to maintain
        self.LEAD_TIME_DAYS = 21  # Standard production lead time
        self.RUSH_LEAD_TIME_DAYS = 7  # Rush order lead time
        self.FORECAST_HORIZON_DAYS = 90  # Planning horizon
        
        # Priority Weights (sum to 1.0)
        self.WEIGHT_CUSTOMER_PRIORITY = 0.25
        self.WEIGHT_PROFITABILITY = 0.20
        self.WEIGHT_INVENTORY_URGENCY = 0.25
        self.WEIGHT_MATERIAL_AVAILABILITY = 0.15
        self.WEIGHT_PRODUCTION_EFFICIENCY = 0.15
        
        # Customer Tiers
        self.customer_tiers = {
            'PLATINUM': {'priority': 1.0, 'min_fill_rate': 0.98},
            'GOLD': {'priority': 0.8, 'min_fill_rate': 0.95},
            'SILVER': {'priority': 0.6, 'min_fill_rate': 0.90},
            'BRONZE': {'priority': 0.4, 'min_fill_rate': 0.85}
        }
        
        # Style Categories
        self.style_categories = {
            'SEASONAL': {'lead_time_factor': 1.2, 'safety_stock_factor': 1.5},
            'CORE': {'lead_time_factor': 1.0, 'safety_stock_factor': 1.0},
            'CUSTOM': {'lead_time_factor': 1.5, 'safety_stock_factor': 0.8},
            'DISCONTINUED': {'lead_time_factor': 0.5, 'safety_stock_factor': 0.0}
        }
    
    def calculate_demand_forecast(self, style: str, sales_history: pd.DataFrame, 
                                 forecast_horizon: int = None) -> Dict:
        """
        Calculate demand forecast for a style based on historical sales
        """
        if forecast_horizon is None:
            forecast_horizon = self.FORECAST_HORIZON_DAYS
        
        forecast = {
            'style': style,
            'forecast_qty': 0,
            'confidence': 0,
            'trend': 'STABLE',
            'seasonality': 'NONE',
            'method': 'NONE'
        }
        
        if sales_history is None or sales_history.empty:
            return forecast
        
        # Calculate basic statistics
        daily_avg = 0
        weekly_avg = 0
        monthly_avg = 0
        
        # Group by date if date column exists
        date_cols = ['Invoice Date', 'Date', 'Order Date', 'date']
        date_col = None
        for col in date_cols:
            if col in sales_history.columns:
                date_col = col
                break
        
        qty_cols = ['Yds_ordered', 'Qty', 'Quantity', 'Qty Shipped']
        qty_col = None
        for col in qty_cols:
            if col in sales_history.columns:
                qty_col = col
                break
        
        if date_col and qty_col:
            try:
                # Convert to datetime
                sales_history[date_col] = pd.to_datetime(sales_history[date_col], errors='coerce')
                sales_history = sales_history.dropna(subset=[date_col])
                
                if not sales_history.empty:
                    # Calculate date range
                    date_range = (sales_history[date_col].max() - sales_history[date_col].min()).days
                    
                    if date_range > 0:
                        # Total quantity sold
                        total_qty = sales_history[qty_col].sum()
                        
                        # Daily average
                        daily_avg = total_qty / max(1, date_range)
                        
                        # Weekly pattern
                        sales_history['week'] = sales_history[date_col].dt.isocalendar().week
                        weekly_sales = sales_history.groupby('week')[qty_col].sum()
                        weekly_avg = weekly_sales.mean() if not weekly_sales.empty else 0
                        
                        # Monthly pattern
                        sales_history['month'] = sales_history[date_col].dt.month
                        monthly_sales = sales_history.groupby('month')[qty_col].sum()
                        monthly_avg = monthly_sales.mean() if not monthly_sales.empty else 0
                        
                        # Calculate trend
                        if date_range >= 30:
                            first_month = sales_history[sales_history[date_col] <= sales_history[date_col].min() + timedelta(days=30)]
                            last_month = sales_history[sales_history[date_col] >= sales_history[date_col].max() - timedelta(days=30)]
                            
                            first_month_qty = first_month[qty_col].sum() if not first_month.empty else 0
                            last_month_qty = last_month[qty_col].sum() if not last_month.empty else 0
                            
                            if first_month_qty > 0:
                                growth_rate = (last_month_qty - first_month_qty) / first_month_qty
                                if growth_rate > 0.1:
                                    forecast['trend'] = 'GROWING'
                                elif growth_rate < -0.1:
                                    forecast['trend'] = 'DECLINING'
                        
                        # Detect seasonality
                        if len(monthly_sales) >= 3:
                            cv = monthly_sales.std() / monthly_sales.mean() if monthly_sales.mean() > 0 else 0
                            if cv > 0.3:
                                forecast['seasonality'] = 'HIGH'
                            elif cv > 0.15:
                                forecast['seasonality'] = 'MODERATE'
                        
                        # Calculate forecast
                        if daily_avg > 0:
                            # Base forecast on daily average
                            forecast['forecast_qty'] = daily_avg * forecast_horizon
                            
                            # Adjust for trend
                            if forecast['trend'] == 'GROWING':
                                forecast['forecast_qty'] *= 1.1
                            elif forecast['trend'] == 'DECLINING':
                                forecast['forecast_qty'] *= 0.9
                            
                            # Calculate confidence based on data points and variability
                            data_points = len(sales_history)
                            if data_points >= 30:
                                forecast['confidence'] = 0.9
                            elif data_points >= 10:
                                forecast['confidence'] = 0.7
                            elif data_points >= 5:
                                forecast['confidence'] = 0.5
                            else:
                                forecast['confidence'] = 0.3
                            
                            # Adjust confidence for variability
                            if sales_history[qty_col].std() > 0:
                                cv = sales_history[qty_col].std() / sales_history[qty_col].mean()
                                if cv > 0.5:
                                    forecast['confidence'] *= 0.8
                            
                            forecast['method'] = 'STATISTICAL'
                
            except Exception as e:
                logger.error(f"Error calculating forecast for {style}: {e}")
        
        # Fallback to simple average if date analysis fails
        if forecast['forecast_qty'] == 0 and qty_col:
            avg_order = sales_history[qty_col].mean()
            order_count = len(sales_history)
            
            if avg_order > 0 and order_count > 0:
                # Estimate based on order frequency
                forecast['forecast_qty'] = avg_order * order_count * (forecast_horizon / 30)
                forecast['confidence'] = min(0.5, order_count / 10)
                forecast['method'] = 'SIMPLE_AVERAGE'
        
        return forecast
    
    def calculate_inventory_urgency(self, style: str, current_inventory: float,
                                   allocated_qty: float, forecast: Dict) -> Dict:
        """
        Calculate inventory urgency for a style
        """
        urgency = {
            'style': style,
            'current_inventory': current_inventory,
            'allocated': allocated_qty,
            'available': max(0, current_inventory - allocated_qty),
            'days_of_supply': 0,
            'urgency_level': 'LOW',
            'urgency_score': 0.0,
            'reorder_point': 0,
            'reorder_qty': 0
        }
        
        # Calculate daily demand rate
        daily_demand = forecast.get('forecast_qty', 0) / max(1, self.FORECAST_HORIZON_DAYS)
        
        # Calculate safety stock
        safety_stock = daily_demand * self.SAFETY_STOCK_DAYS
        
        # Calculate reorder point (lead time demand + safety stock)
        urgency['reorder_point'] = (daily_demand * self.LEAD_TIME_DAYS) + safety_stock
        
        # Calculate days of supply
        if daily_demand > 0:
            urgency['days_of_supply'] = urgency['available'] / daily_demand
        else:
            urgency['days_of_supply'] = 999  # Infinite if no demand
        
        # Determine urgency level
        if urgency['available'] <= 0:
            urgency['urgency_level'] = 'CRITICAL'
            urgency['urgency_score'] = 1.0
        elif urgency['available'] < safety_stock:
            urgency['urgency_level'] = 'HIGH'
            urgency['urgency_score'] = 0.8
        elif urgency['available'] < urgency['reorder_point']:
            urgency['urgency_level'] = 'MEDIUM'
            urgency['urgency_score'] = 0.6
        elif urgency['days_of_supply'] < self.FORECAST_HORIZON_DAYS:
            urgency['urgency_level'] = 'LOW'
            urgency['urgency_score'] = 0.4
        else:
            urgency['urgency_level'] = 'NONE'
            urgency['urgency_score'] = 0.0
        
        # Calculate reorder quantity (economic order quantity simplified)
        if urgency['available'] < urgency['reorder_point']:
            # Order up to maximum of forecast horizon demand
            max_stock = daily_demand * self.FORECAST_HORIZON_DAYS
            urgency['reorder_qty'] = max(self.MIN_PRODUCTION_BATCH, 
                                        max_stock - urgency['available'])
            
            # Round to batch size
            batch_size = self.MIN_PRODUCTION_BATCH
            urgency['reorder_qty'] = np.ceil(urgency['reorder_qty'] / batch_size) * batch_size
            
            # Cap at maximum batch
            urgency['reorder_qty'] = min(urgency['reorder_qty'], self.MAX_PRODUCTION_BATCH)
        
        return urgency
    
    def calculate_production_priority(self, style: str, customer: str,
                                     forecast: Dict, urgency: Dict,
                                     material_availability: float,
                                     profitability_score: float = 0.5) -> Dict:
        """
        Calculate overall production priority score
        """
        priority = {
            'style': style,
            'customer': customer,
            'score': 0.0,
            'rank': 'LOW',
            'factors': {}
        }
        
        # Get customer tier
        customer_tier = self.classify_customer(customer)
        customer_priority = self.customer_tiers[customer_tier]['priority']
        
        # Calculate weighted score
        priority['factors'] = {
            'customer_priority': customer_priority * self.WEIGHT_CUSTOMER_PRIORITY,
            'profitability': profitability_score * self.WEIGHT_PROFITABILITY,
            'inventory_urgency': urgency.get('urgency_score', 0) * self.WEIGHT_INVENTORY_URGENCY,
            'material_availability': material_availability * self.WEIGHT_MATERIAL_AVAILABILITY,
            'production_efficiency': self.calculate_production_efficiency(style) * self.WEIGHT_PRODUCTION_EFFICIENCY
        }
        
        priority['score'] = sum(priority['factors'].values())
        
        # Determine rank
        if priority['score'] >= 0.8:
            priority['rank'] = 'CRITICAL'
        elif priority['score'] >= 0.6:
            priority['rank'] = 'HIGH'
        elif priority['score'] >= 0.4:
            priority['rank'] = 'MEDIUM'
        else:
            priority['rank'] = 'LOW'
        
        return priority
    
    def classify_customer(self, customer: str) -> str:
        """
        Classify customer into tier based on business rules
        """
        # Major customers (simplified - in production would use customer database)
        platinum_customers = ['Serta Simmons', 'Tempur Sealy', 'Purple']
        gold_customers = ['Ashley', 'La-Z-Boy', 'Rooms To Go']
        silver_customers = ['Regional Retailer', 'Online Store']
        
        if customer:
            customer_upper = customer.upper()
            for plat in platinum_customers:
                if plat.upper() in customer_upper:
                    return 'PLATINUM'
            for gold in gold_customers:
                if gold.upper() in customer_upper:
                    return 'GOLD'
            for silver in silver_customers:
                if silver.upper() in customer_upper:
                    return 'SILVER'
        
        return 'BRONZE'
    
    def classify_style_category(self, style: str, sales_pattern: Dict = None) -> str:
        """
        Classify style into category based on patterns
        """
        if not style:
            return 'CORE'
        
        # Check for seasonal patterns
        if sales_pattern and sales_pattern.get('seasonality') == 'HIGH':
            return 'SEASONAL'
        
        # Check for custom indicators
        if 'CUSTOM' in style.upper() or 'CS' in style.upper():
            return 'CUSTOM'
        
        # Check for discontinued (would check against product database)
        if 'DISC' in style.upper() or 'OLD' in style.upper():
            return 'DISCONTINUED'
        
        return 'CORE'
    
    def calculate_production_efficiency(self, style: str) -> float:
        """
        Calculate production efficiency score for a style based on actual capacity
        """
        # Try to get actual production capacity
        try:
            from production.production_capacity_manager import get_capacity_manager
            capacity_mgr = get_capacity_manager()
            
            # Get style's actual production capacity
            capacity = capacity_mgr.get_style_capacity(style)
            
            # Normalize capacity to efficiency score (0-1)
            # Excellent (2000+): 1.0
            # Good (1000-2000): 0.8
            # Average (500-1000): 0.6
            # Below Average (100-500): 0.4
            # Poor (<100): 0.2
            
            if capacity >= 2000:
                efficiency = 1.0
            elif capacity >= 1000:
                efficiency = 0.8
            elif capacity >= 500:
                efficiency = 0.6
            elif capacity >= 100:
                efficiency = 0.4
            else:
                efficiency = 0.2
                
            return efficiency
            
        except:
            # Fallback to simple heuristic if capacity data not available
            efficiency = 0.5  # Base efficiency
            
            if style:
                if 'STANDARD' in style.upper() or 'STD' in style.upper():
                    efficiency = 0.8
                elif 'CUSTOM' in style.upper():
                    efficiency = 0.3
                elif style.startswith('C1B'):  # Common prefix
                    efficiency = 0.7
            
            return efficiency
    
    def calculate_production_schedule(self, suggestions: List[Dict],
                                     max_capacity_per_day: float = 10000) -> List[Dict]:
        """
        Create production schedule from suggestions using actual production capacity
        """
        schedule = []
        current_date = datetime.now()
        accumulated_time = 0
        
        # Get capacity manager for actual production rates
        try:
            from production.production_capacity_manager import get_capacity_manager
            capacity_mgr = get_capacity_manager()
        except:
            capacity_mgr = None
        
        # Sort by priority
        sorted_suggestions = sorted(suggestions, 
                                  key=lambda x: x.get('priority_score', 0),
                                  reverse=True)
        
        for suggestion in sorted_suggestions:
            style = suggestion.get('style')
            qty = suggestion.get('suggested_quantity_lbs', 0)
            if qty <= 0:
                continue
            
            # Get actual production capacity for this style
            if capacity_mgr:
                style_capacity = capacity_mgr.get_style_capacity(style)
                # Use the lower of style capacity or max daily capacity
                effective_capacity = min(style_capacity, max_capacity_per_day)
            else:
                effective_capacity = max_capacity_per_day
            
            # Calculate production time with actual capacity
            production_days = qty / effective_capacity
            
            # Determine lead time based on urgency
            if suggestion.get('urgency_level') == 'CRITICAL':
                lead_time = self.RUSH_LEAD_TIME_DAYS
            else:
                lead_time = self.LEAD_TIME_DAYS
            
            # Calculate start and end dates
            start_date = current_date + timedelta(days=accumulated_time)
            end_date = start_date + timedelta(days=production_days)
            due_date = start_date + timedelta(days=lead_time)
            
            schedule_item = {
                'style': suggestion.get('style'),
                'quantity_lbs': qty,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'due_date': due_date.strftime('%Y-%m-%d'),
                'priority': suggestion.get('priority_rank', 'MEDIUM'),
                'status': 'PLANNED'
            }
            
            schedule.append(schedule_item)
            accumulated_time += production_days
            
            # Stop if we exceed planning horizon
            if accumulated_time > self.FORECAST_HORIZON_DAYS:
                break
        
        return schedule
    
    def validate_production_suggestion(self, suggestion: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a production suggestion against business rules
        """
        issues = []
        is_valid = True
        
        qty = suggestion.get('suggested_quantity_lbs', 0)
        
        # Check minimum batch size
        if qty < self.MIN_PRODUCTION_BATCH:
            issues.append(f"Quantity {qty} below minimum batch {self.MIN_PRODUCTION_BATCH}")
            is_valid = False
        
        # Check maximum batch size
        if qty > self.MAX_PRODUCTION_BATCH:
            issues.append(f"Quantity {qty} exceeds maximum batch {self.MAX_PRODUCTION_BATCH}")
            suggestion['suggested_quantity_lbs'] = self.MAX_PRODUCTION_BATCH
        
        # Check material availability
        if not suggestion.get('yarn_available', True):
            issues.append("Insufficient yarn/materials available")
            is_valid = False
        
        # Check if style is discontinued
        category = self.classify_style_category(suggestion.get('style'))
        if category == 'DISCONTINUED':
            issues.append("Style is discontinued")
            is_valid = False
        
        return is_valid, issues


# Helper function to integrate with existing system
def apply_business_logic(suggestions: List[Dict], analyzer=None) -> List[Dict]:
    """
    Apply business logic to production suggestions
    """
    logic = ProductionBusinessLogic()
    enhanced_suggestions = []
    
    for suggestion in suggestions:
        style = suggestion.get('style')
        
        # Get sales history if analyzer available
        sales_history = None
        if analyzer and hasattr(analyzer, 'sales_data') and analyzer.sales_data is not None:
            if 'Style#' in analyzer.sales_data.columns:
                sales_history = analyzer.sales_data[analyzer.sales_data['Style#'] == style]
        
        # Calculate forecast
        forecast = logic.calculate_demand_forecast(style, sales_history)
        
        # Calculate urgency
        current_inv = suggestion.get('current_inventory', 0)
        allocated = suggestion.get('allocated', 0)
        urgency = logic.calculate_inventory_urgency(style, current_inv, allocated, forecast)
        
        # Get customer (from first sales record if available)
        customer = ''
        if sales_history is not None and not sales_history.empty:
            if 'Customer' in sales_history.columns:
                customer = sales_history['Customer'].iloc[0]
        
        # Calculate priority
        material_availability = 1.0 if suggestion.get('yarn_available', True) else 0.0
        profitability = suggestion.get('profitability_score', 0.5)
        
        priority = logic.calculate_production_priority(
            style, customer, forecast, urgency,
            material_availability, profitability
        )
        
        # Enhance suggestion with business logic
        suggestion.update({
            'forecast': forecast,
            'urgency': urgency,
            'priority': priority,
            'customer_tier': logic.classify_customer(customer),
            'style_category': logic.classify_style_category(style),
            'suggested_quantity_lbs': urgency.get('reorder_qty', 0),
            'priority_score': priority['score'],
            'priority_rank': priority['rank'],
            'urgency_level': urgency['urgency_level']
        })
        
        # Validate
        is_valid, issues = logic.validate_production_suggestion(suggestion)
        suggestion['is_valid'] = is_valid
        suggestion['validation_issues'] = issues
        
        if is_valid or len(issues) == 1:  # Allow if only issue is materials
            enhanced_suggestions.append(suggestion)
    
    return enhanced_suggestions