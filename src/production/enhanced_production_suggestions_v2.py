"""
Enhanced AI Production Suggestions Module V2
With improved demand confidence levels and priority ranking system
Priority: Sales Orders > High-confidence forecast > Low-confidence forecast
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedProductionSuggestionsV2:
    """Advanced AI-driven production suggestion engine with confidence-based prioritization"""
    
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
        
        # Confidence thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.75
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.50
        
        # Priority categories with weights
        self.PRIORITY_CATEGORIES = {
            'SALES_ORDER': 100,      # Actual sales orders - highest priority
            'HIGH_CONFIDENCE': 75,    # High confidence forecast (>75%)
            'MEDIUM_CONFIDENCE': 50,  # Medium confidence forecast (50-75%)
            'LOW_CONFIDENCE': 25,     # Low confidence forecast (<50%)
            'NO_DEMAND': 10          # No demand history
        }
        
        # Enhanced scoring weights - adjusted for confidence-based prioritization
        self.scoring_weights = {
            'demand_source': 0.35,        # Sales order vs forecast (increased)
            'demand_confidence': 0.25,    # Confidence level of demand (increased)
            'inventory_urgency': 0.20,    # How urgent based on current stock
            'material_availability': 0.15, # Material availability
            'profitability': 0.05         # Profitability (reduced)
        }
        
    def calculate_demand_confidence(self, style: str) -> Dict[str, Any]:
        """
        Calculate comprehensive demand confidence for a style
        Returns confidence level, demand source, and metrics
        """
        result = {
            'confidence_score': 0.0,
            'confidence_level': 'LOW',
            'demand_source': 'NO_DEMAND',
            'has_sales_order': False,
            'sales_order_qty': 0,
            'historical_demand': 0,
            'demand_variability': 1.0,
            'data_points': 0,
            'forecast_accuracy': 0.0,
            'trend_stability': 0.0
        }
        
        if not self.analyzer:
            return result
        
        # Check for active sales orders (highest confidence)
        active_sales_orders = 0
        if hasattr(self.analyzer, 'sales_orders_data') and self.analyzer.sales_orders_data is not None:
            if 'Style#' in self.analyzer.sales_orders_data.columns:
                style_orders = self.analyzer.sales_orders_data[
                    self.analyzer.sales_orders_data['Style#'] == style
                ]
                if not style_orders.empty:
                    # Check for open/active orders
                    if 'Status' in style_orders.columns:
                        active_orders = style_orders[
                            style_orders['Status'].str.contains('Open|Active|Pending', case=False, na=False)
                        ]
                        if not active_orders.empty:
                            result['has_sales_order'] = True
                            result['demand_source'] = 'SALES_ORDER'
                            # Convert to float to avoid numpy type issues
                            result['sales_order_qty'] = float(active_orders.get('Balance', active_orders.get('Ordered', pd.Series())).sum())
                            result['confidence_score'] = 1.0  # 100% confidence for actual orders
                            result['confidence_level'] = 'SALES_ORDER'
                            active_sales_orders = len(active_orders)
        
        # Analyze historical sales data for forecast confidence
        if hasattr(self.analyzer, 'sales_data') and self.analyzer.sales_data is not None:
            # Handle both Style# and fStyle# columns
            style_col = None
            if 'Style#' in self.analyzer.sales_data.columns:
                style_col = 'Style#'
            elif 'fStyle#' in self.analyzer.sales_data.columns:
                style_col = 'fStyle#'
            
            if style_col:
                style_sales = self.analyzer.sales_data[self.analyzer.sales_data[style_col] == style]
                
                if not style_sales.empty:
                    result['data_points'] = int(len(style_sales))
                    
                    # Get demand column and handle comma-formatted numbers
                    demand_col = style_sales.get('Ordered', style_sales.get('Quantity', pd.Series()))
                    if not demand_col.empty:
                        # Convert string values with commas to numeric
                        if demand_col.dtype == 'object':
                            # Remove commas, dollar signs, and any non-numeric characters
                            demand_col = demand_col.astype(str).str.replace(',', '').str.replace('$', '')
                            # Extract numeric values
                            demand_col = demand_col.str.extract(r'([\d.]+)', expand=False)
                            demand_col = pd.to_numeric(demand_col, errors='coerce').fillna(0)
                        result['historical_demand'] = float(demand_col.sum())
                    else:
                        result['historical_demand'] = 0.0
                    
                    # Calculate demand variability (coefficient of variation)
                    if 'Ordered' in style_sales.columns or 'Quantity' in style_sales.columns:
                        demand_series = style_sales.get('Ordered', style_sales.get('Quantity', pd.Series()))
                        # Convert string values with commas to numeric
                        if demand_series.dtype == 'object':
                            # Remove commas, dollar signs, and any non-numeric characters
                            demand_series = demand_series.astype(str).str.replace(',', '').str.replace('$', '')
                            # Extract numeric values
                            demand_series = demand_series.str.extract(r'([\d.]+)', expand=False)
                            demand_series = pd.to_numeric(demand_series, errors='coerce').fillna(0)
                        if len(demand_series) > 1:
                            mean_demand = demand_series.mean()
                            std_demand = demand_series.std()
                            if mean_demand > 0:
                                cv = std_demand / mean_demand
                                result['demand_variability'] = cv
                                
                                # Lower variability = higher confidence
                                variability_confidence = max(0, min(1, 1 - (cv / 2)))
                            else:
                                variability_confidence = 0.3
                        else:
                            variability_confidence = 0.4
                    else:
                        variability_confidence = 0.3
                    
                    # Calculate trend stability if we have date information
                    trend_confidence = 0.5  # Default
                    if 'Invoice Date' in style_sales.columns or 'Date' in style_sales.columns:
                        date_col = 'Invoice Date' if 'Invoice Date' in style_sales.columns else 'Date'
                        try:
                            style_sales['parsed_date'] = pd.to_datetime(style_sales[date_col])
                            style_sales = style_sales.sort_values('parsed_date')
                            
                            # Check for consistent ordering pattern
                            date_diffs = style_sales['parsed_date'].diff().dt.days.dropna()
                            if len(date_diffs) > 0:
                                avg_interval = date_diffs.mean()
                                std_interval = date_diffs.std()
                                
                                # More consistent intervals = higher confidence
                                if avg_interval > 0 and not pd.isna(std_interval):
                                    interval_cv = std_interval / avg_interval if avg_interval > 0 else 1
                                    trend_confidence = max(0, min(1, 1 - (interval_cv / 2)))
                                    result['trend_stability'] = trend_confidence
                        except:
                            pass
                    
                    # Data recency factor - more recent data = higher confidence
                    recency_confidence = 0.5
                    if 'Invoice Date' in style_sales.columns or 'Date' in style_sales.columns:
                        date_col = 'Invoice Date' if 'Invoice Date' in style_sales.columns else 'Date'
                        try:
                            latest_date = pd.to_datetime(style_sales[date_col]).max()
                            days_since_last = (datetime.now() - latest_date).days
                            
                            if days_since_last < 30:
                                recency_confidence = 0.9
                            elif days_since_last < 60:
                                recency_confidence = 0.7
                            elif days_since_last < 90:
                                recency_confidence = 0.5
                            else:
                                recency_confidence = 0.3
                        except:
                            pass
                    
                    # Data volume factor - more data points = higher confidence
                    volume_confidence = min(1.0, result['data_points'] / 10)  # Max confidence at 10+ orders
                    
                    # If no sales order, calculate forecast confidence
                    if not result['has_sales_order']:
                        # Weighted average of confidence factors
                        forecast_confidence = (
                            variability_confidence * 0.30 +
                            trend_confidence * 0.25 +
                            recency_confidence * 0.25 +
                            volume_confidence * 0.20
                        )
                        
                        result['confidence_score'] = forecast_confidence
                        result['forecast_accuracy'] = forecast_confidence
                        
                        # Determine confidence level
                        if forecast_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                            result['confidence_level'] = 'HIGH'
                            result['demand_source'] = 'HIGH_CONFIDENCE'
                        elif forecast_confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                            result['confidence_level'] = 'MEDIUM'
                            result['demand_source'] = 'MEDIUM_CONFIDENCE'
                        else:
                            result['confidence_level'] = 'LOW'
                            result['demand_source'] = 'LOW_CONFIDENCE'
        
        # No historical data and no sales orders
        if result['data_points'] == 0 and not result['has_sales_order']:
            result['confidence_score'] = 0.1
            result['confidence_level'] = 'NO_HISTORY'
            result['demand_source'] = 'NO_DEMAND'
        
        return result
    
    def calculate_inventory_urgency(self, style: str) -> Dict[str, Any]:
        """Calculate urgency based on current inventory position"""
        urgency = {
            'urgency_score': 0.5,
            'days_of_supply': 999,
            'stockout_risk': 'unknown',
            'on_hand': 0,
            'in_production': 0,
            'allocated': 0,
            'available': 0
        }
        
        if not self.analyzer:
            return urgency
        
        # Get current inventory levels
        total_inventory = 0
        if hasattr(self.analyzer, 'inventory_data') and self.analyzer.inventory_data:
            for stage, inv_data in self.analyzer.inventory_data.items():
                if isinstance(inv_data, dict) and 'data' in inv_data:
                    inv_df = inv_data['data']
                else:
                    inv_df = inv_data
                
                if inv_df is not None and isinstance(inv_df, pd.DataFrame) and 'Style#' in inv_df.columns:
                    style_inv = inv_df[inv_df['Style#'] == style]
                    if not style_inv.empty:
                        balance = style_inv.get('Balance', style_inv.get('On Hand (lbs)', pd.Series())).sum()
                        total_inventory += balance
                        
                        if stage in ['F01', 'Finished']:
                            urgency['on_hand'] += balance
                        elif stage in ['G00', 'G02', 'WIP']:
                            urgency['in_production'] += balance
        
        # Check allocations
        if hasattr(self.analyzer, 'sales_orders_data') and self.analyzer.sales_orders_data is not None:
            if 'Style#' in self.analyzer.sales_orders_data.columns:
                style_orders = self.analyzer.sales_orders_data[
                    self.analyzer.sales_orders_data['Style#'] == style
                ]
                if not style_orders.empty:
                    urgency['allocated'] = float(style_orders.get('Balance', pd.Series()).sum())
        
        urgency['available'] = total_inventory - urgency['allocated']
        
        # Calculate days of supply based on historical demand
        demand_confidence = self.calculate_demand_confidence(style)
        daily_demand = 0
        
        if demand_confidence['sales_order_qty'] > 0:
            # Use sales order quantity over lead time
            daily_demand = demand_confidence['sales_order_qty'] / self.lead_time_days
        elif demand_confidence['historical_demand'] > 0:
            # Use historical demand
            daily_demand = demand_confidence['historical_demand'] / 30  # Assume 30-day history
        
        if daily_demand > 0:
            urgency['days_of_supply'] = urgency['available'] / daily_demand
            
            # Calculate urgency score
            if urgency['days_of_supply'] < 7:
                urgency['urgency_score'] = 1.0
                urgency['stockout_risk'] = 'critical'
            elif urgency['days_of_supply'] < 14:
                urgency['urgency_score'] = 0.8
                urgency['stockout_risk'] = 'high'
            elif urgency['days_of_supply'] < 30:
                urgency['urgency_score'] = 0.5
                urgency['stockout_risk'] = 'medium'
            else:
                urgency['urgency_score'] = 0.2
                urgency['stockout_risk'] = 'low'
        
        return urgency
    
    def calculate_priority_score_v2(self, style: str, demand_confidence: Dict, 
                                   inventory_urgency: Dict, material_availability: float,
                                   profitability: float) -> Dict[str, Any]:
        """
        Calculate priority score with emphasis on demand confidence
        Priority: Sales Orders > High-confidence forecast > Low-confidence forecast
        """
        # Base priority from demand source
        base_priority = self.PRIORITY_CATEGORIES.get(demand_confidence['demand_source'], 10)
        
        # Calculate weighted score
        weighted_score = 0
        
        # 1. Demand source score (35% weight)
        demand_source_score = base_priority / 100  # Normalize to 0-1
        weighted_score += demand_source_score * self.scoring_weights['demand_source'] * 100
        
        # 2. Demand confidence score (25% weight)
        confidence_score = demand_confidence['confidence_score']
        weighted_score += confidence_score * self.scoring_weights['demand_confidence'] * 100
        
        # 3. Inventory urgency score (20% weight)
        urgency_score = inventory_urgency['urgency_score']
        weighted_score += urgency_score * self.scoring_weights['inventory_urgency'] * 100
        
        # 4. Material availability score (15% weight)
        weighted_score += material_availability * self.scoring_weights['material_availability'] * 100
        
        # 5. Profitability score (5% weight)
        weighted_score += profitability * self.scoring_weights['profitability'] * 100
        
        # Apply multipliers for critical conditions
        multiplier = 1.0
        
        # Boost for actual sales orders
        if demand_confidence['has_sales_order']:
            multiplier *= 1.5
        
        # Boost for critical stockout risk
        if inventory_urgency['stockout_risk'] == 'critical':
            multiplier *= 1.3
        elif inventory_urgency['stockout_risk'] == 'high':
            multiplier *= 1.1
        
        # Penalty for low confidence without sales orders
        if not demand_confidence['has_sales_order'] and confidence_score < 0.5:
            multiplier *= 0.7
        
        final_score = min(100, weighted_score * multiplier)
        
        return {
            'priority_score': round(final_score, 1),
            'base_priority': base_priority,
            'confidence_score': round(confidence_score, 2),
            'urgency_score': round(urgency_score, 2),
            'material_score': round(material_availability, 2),
            'multiplier': round(multiplier, 2)
        }
    
    def check_material_availability(self, style: str, quantity: float) -> Tuple[bool, List[str], float]:
        """Check if materials are available for production"""
        available = True
        issues = []
        availability_score = 1.0
        
        if not self.analyzer or not hasattr(self.analyzer, 'bom_data'):
            return True, [], 1.0
        
        bom_df = self.analyzer.bom_data
        if isinstance(bom_df, dict) and 'data' in bom_df:
            bom_df = bom_df['data']
        
        if not isinstance(bom_df, pd.DataFrame) or bom_df.empty:
            return True, [], 1.0
        
        # Try to map sales style to BOM style using style mapper
        bom_styles_to_check = [style]  # Default to original style
        
        if hasattr(self.analyzer, 'style_mapper') and self.analyzer.style_mapper:
            mapped_styles = self.analyzer.style_mapper.map_sales_to_bom(style)
            if mapped_styles:
                bom_styles_to_check = mapped_styles
        
        # Get BOM for this style (check all mapped styles)
        style_bom = pd.DataFrame()
        for bom_style in bom_styles_to_check:
            if 'Style#' in bom_df.columns:
                bom_match = bom_df[bom_df['Style#'] == bom_style]
                if not bom_match.empty:
                    style_bom = bom_match
                    break
        
        if style_bom.empty:
            issues.append(f"No BOM found for style {style} or mapped styles")
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
        
        # Get sales data for this style - handle both Style# and fStyle#
        style_col = None
        if 'Style#' in sales_df.columns:
            style_col = 'Style#'
        elif 'fStyle#' in sales_df.columns:
            style_col = 'fStyle#'
        
        if style_col:
            style_sales = sales_df[sales_df[style_col] == style]
            
            if not style_sales.empty:
                # Calculate average price if available
                if 'Unit Price' in style_sales.columns:
                    price_series = style_sales['Unit Price']
                    # Handle string prices with dollar signs
                    if price_series.dtype == 'object':
                        price_series = price_series.astype(str).str.replace('$', '').str.replace(',', '')
                        price_series = price_series.str.extract(r'([\d.]+)', expand=False)
                        price_series = pd.to_numeric(price_series, errors='coerce').fillna(0)
                    avg_price = price_series.mean()
                    # Normalize price score (assume $10-50 range)
                    score = min(1.0, max(0.0, (avg_price - 10) / 40))
                
                # Adjust for order frequency
                order_count = len(style_sales)
                if order_count > 10:
                    score = min(1.0, score + 0.2)
                elif order_count < 3:
                    score = max(0.0, score - 0.2)
        
        return score
    
    def calculate_suggested_quantity(self, style: str, demand_confidence: Dict, 
                                    inventory_urgency: Dict) -> float:
        """Calculate suggested production quantity based on demand and inventory"""
        
        # For sales orders, produce the order quantity plus safety stock
        if demand_confidence['has_sales_order']:
            base_qty = demand_confidence['sales_order_qty']
            safety_stock = base_qty * 0.2  # 20% safety stock
            suggested_qty = base_qty + safety_stock
        
        # For forecasted demand, adjust based on confidence
        elif demand_confidence['historical_demand'] > 0:
            # Project demand for forecast horizon
            daily_demand = demand_confidence['historical_demand'] / 30
            base_qty = daily_demand * self.forecast_horizon_days
            
            # Adjust based on confidence level
            if demand_confidence['confidence_level'] == 'HIGH':
                confidence_factor = 1.0
            elif demand_confidence['confidence_level'] == 'MEDIUM':
                confidence_factor = 0.7
            else:
                confidence_factor = 0.5
            
            suggested_qty = base_qty * confidence_factor
            
            # Add safety stock based on variability
            if demand_confidence['demand_variability'] > 0.5:
                safety_stock = suggested_qty * 0.3  # High variability = more safety stock
            else:
                safety_stock = suggested_qty * 0.15
            
            suggested_qty += safety_stock
        
        # No demand history - minimum batch
        else:
            suggested_qty = self.min_production_batch
        
        # Subtract available inventory
        suggested_qty -= inventory_urgency['available']
        
        # Apply minimum batch size
        if suggested_qty > 0 and suggested_qty < self.min_production_batch:
            suggested_qty = self.min_production_batch
        
        # Don't suggest negative quantities
        return max(0, round(suggested_qty, 0))
    
    def generate_suggestions(self, limit: int = 20) -> Dict[str, Any]:
        """Generate comprehensive production suggestions with confidence-based prioritization"""
        suggestions = []
        
        try:
            # Get all styles from various data sources
            all_styles = set()
            
            # Priority 1: Styles with active sales orders
            sales_order_styles = set()
            if hasattr(self.analyzer, 'sales_orders_data') and self.analyzer.sales_orders_data is not None:
                if 'Style#' in self.analyzer.sales_orders_data.columns:
                    # Filter for active orders
                    if 'Status' in self.analyzer.sales_orders_data.columns:
                        active_orders = self.analyzer.sales_orders_data[
                            self.analyzer.sales_orders_data['Status'].str.contains('Open|Active|Pending', 
                                                                                   case=False, na=False)
                        ]
                        sales_order_styles.update(active_orders['Style#'].unique())
                    else:
                        sales_order_styles.update(self.analyzer.sales_orders_data['Style#'].unique())
            
            # Priority 2: Styles with historical sales
            historical_styles = set()
            if hasattr(self.analyzer, 'sales_data') and self.analyzer.sales_data is not None:
                # Handle both Style# and fStyle# columns
                if 'Style#' in self.analyzer.sales_data.columns:
                    sales_styles = self.analyzer.sales_data['Style#'].dropna().unique()
                    historical_styles.update(sales_styles)
                    
                    # If we have a style mapper, also check for mapped BOM styles
                    if hasattr(self.analyzer, 'style_mapper') and self.analyzer.style_mapper:
                        for style in sales_styles[:50]:  # Limit for performance
                            mapped = self.analyzer.style_mapper.map_sales_to_bom(str(style))
                            if mapped:
                                # Add the first mapped BOM style as a candidate
                                historical_styles.add(mapped[0])
                                
                elif 'fStyle#' in self.analyzer.sales_data.columns:
                    # Add all fStyle# values
                    fstyles = self.analyzer.sales_data['fStyle#'].unique()
                    historical_styles.update(fstyles)
            
            # Priority 3: Styles in BOM (could be new products)
            bom_styles = set()
            if hasattr(self.analyzer, 'bom_data') and self.analyzer.bom_data is not None:
                bom_df = self.analyzer.bom_data
                if isinstance(bom_df, dict) and 'data' in bom_df:
                    bom_df = bom_df['data']
                if isinstance(bom_df, pd.DataFrame) and 'Style#' in bom_df.columns:
                    bom_styles.update(bom_df['Style#'].unique())
            
            # Combine all styles with priority order
            all_styles = list(sales_order_styles) + \
                        [s for s in historical_styles if s not in sales_order_styles] + \
                        [s for s in bom_styles if s not in sales_order_styles and s not in historical_styles]
            
            # Process each style (limit to 200 for performance)
            for style in all_styles[:200]:
                if not style or pd.isna(style):
                    continue
                
                # Calculate demand confidence
                demand_confidence = self.calculate_demand_confidence(style)
                
                # Calculate inventory urgency
                inventory_urgency = self.calculate_inventory_urgency(style)
                
                # Skip if well-stocked and no urgent demand
                # But always include if we have sales orders
                if (inventory_urgency['days_of_supply'] > self.forecast_horizon_days and 
                    not demand_confidence['has_sales_order'] and
                    demand_confidence['confidence_score'] < 0.3):
                    continue
                
                # Calculate suggested quantity
                suggested_qty = self.calculate_suggested_quantity(style, demand_confidence, inventory_urgency)
                
                # Skip if no production needed
                if suggested_qty <= 0:
                    continue
                
                # Check material availability
                material_available, material_issues, material_score = self.check_material_availability(style, suggested_qty)
                
                # Calculate profitability
                profitability = self.calculate_profitability_score(style)
                
                # Calculate priority score
                priority_details = self.calculate_priority_score_v2(
                    style, demand_confidence, inventory_urgency, material_score, profitability
                )
                
                # Build detailed rationale
                rationale_parts = []
                
                # Lead with demand source
                if demand_confidence['has_sales_order']:
                    rationale_parts.append(f"ACTIVE SALES ORDER: {demand_confidence['sales_order_qty']:.0f} lbs")
                elif demand_confidence['confidence_level'] == 'HIGH':
                    rationale_parts.append(f"High confidence forecast ({demand_confidence['confidence_score']:.0%})")
                elif demand_confidence['confidence_level'] == 'MEDIUM':
                    rationale_parts.append(f"Medium confidence forecast ({demand_confidence['confidence_score']:.0%})")
                else:
                    rationale_parts.append(f"Low confidence forecast ({demand_confidence['confidence_score']:.0%})")
                
                # Add urgency information
                if inventory_urgency['stockout_risk'] in ['critical', 'high']:
                    rationale_parts.append(f"{inventory_urgency['stockout_risk'].upper()} stockout risk - {inventory_urgency['days_of_supply']:.1f} days supply")
                
                # Add demand metrics
                if demand_confidence['historical_demand'] > 0:
                    rationale_parts.append(f"Historical: {demand_confidence['historical_demand']:.0f} lbs")
                    if demand_confidence['data_points'] > 0:
                        rationale_parts.append(f"{demand_confidence['data_points']} orders")
                
                # Add material status
                if not material_available:
                    rationale_parts.append(f"Material constraints: {len(material_issues)} issues")
                
                # Create suggestion object
                suggestion = {
                    'style': style,
                    'priority_category': demand_confidence['demand_source'],
                    'priority_score': priority_details['priority_score'],
                    'confidence_level': demand_confidence['confidence_level'],
                    'confidence_score': demand_confidence['confidence_score'],
                    'has_sales_order': demand_confidence['has_sales_order'],
                    'sales_order_qty': round(demand_confidence['sales_order_qty'], 0),
                    'suggested_quantity_lbs': suggested_qty,
                    'current_inventory': round(inventory_urgency['on_hand'], 0),
                    'in_production': round(inventory_urgency['in_production'], 0),
                    'allocated': round(inventory_urgency['allocated'], 0),
                    'available': round(inventory_urgency['available'], 0),
                    'days_of_supply': round(inventory_urgency['days_of_supply'], 1),
                    'stockout_risk': inventory_urgency['stockout_risk'],
                    'historical_demand': round(demand_confidence['historical_demand'], 0),
                    'data_points': demand_confidence['data_points'],
                    'demand_variability': round(demand_confidence['demand_variability'], 2),
                    'trend_stability': round(demand_confidence['trend_stability'], 2),
                    'yarn_available': material_available,
                    'material_availability_score': material_score,
                    'material_status': 'All materials available' if material_available else f"{len(material_issues)} material issues",
                    'material_issues': material_issues[:3],
                    'profitability_score': profitability,
                    'rationale': '. '.join(rationale_parts)
                }
                
                suggestions.append(suggestion)
            
            # Sort by priority score (which already considers demand source priority)
            suggestions.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Take top suggestions
            top_suggestions = suggestions[:limit]
            
            # Group suggestions by category for summary
            category_counts = {
                'SALES_ORDER': 0,
                'HIGH_CONFIDENCE': 0,
                'MEDIUM_CONFIDENCE': 0,
                'LOW_CONFIDENCE': 0,
                'NO_DEMAND': 0
            }
            
            for s in top_suggestions:
                category_counts[s['priority_category']] = category_counts.get(s['priority_category'], 0) + 1
            
            # Calculate summary statistics
            summary = {
                'total_suggestions': len(top_suggestions),
                'sales_order_driven': category_counts['SALES_ORDER'],
                'high_confidence_forecast': category_counts['HIGH_CONFIDENCE'],
                'medium_confidence_forecast': category_counts['MEDIUM_CONFIDENCE'],
                'low_confidence_forecast': category_counts['LOW_CONFIDENCE'],
                'no_demand_history': category_counts['NO_DEMAND'],
                'critical_stockout_risk': len([s for s in top_suggestions if s['stockout_risk'] == 'critical']),
                'high_stockout_risk': len([s for s in top_suggestions if s['stockout_risk'] == 'high']),
                'material_available': len([s for s in top_suggestions if s['yarn_available']]),
                'material_shortage': len([s for s in top_suggestions if not s['yarn_available']]),
                'total_suggested_production': sum(s['suggested_quantity_lbs'] for s in top_suggestions),
                'average_priority_score': round(np.mean([s['priority_score'] for s in top_suggestions]), 1) if top_suggestions else 0,
                'average_confidence': round(np.mean([s['confidence_score'] for s in top_suggestions]), 2) if top_suggestions else 0
            }
            
            # Generate strategic recommendations
            recommendations = []
            
            # Priority-based recommendations
            if summary['sales_order_driven'] > 0:
                recommendations.append(f"PRIORITY: {summary['sales_order_driven']} styles have active sales orders - produce immediately")
            
            if summary['high_confidence_forecast'] > 0:
                recommendations.append(f"High confidence: {summary['high_confidence_forecast']} styles with reliable demand forecast")
            
            if summary['low_confidence_forecast'] > summary['high_confidence_forecast']:
                recommendations.append("Many low-confidence forecasts - consider gathering more sales data before production")
            
            # Risk-based recommendations
            if summary['critical_stockout_risk'] > 0:
                recommendations.append(f"URGENT: {summary['critical_stockout_risk']} styles at critical stockout risk")
            
            # Material recommendations
            if summary['material_shortage'] > summary['material_available']:
                recommendations.append("Material constraints affecting majority - prioritize yarn procurement")
            
            # Confidence recommendations
            avg_conf = summary['average_confidence']
            if avg_conf < 0.5:
                recommendations.append("Overall low demand confidence - focus on sales order fulfillment")
            elif avg_conf > 0.75:
                recommendations.append("High overall confidence - production plans are reliable")
            
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
                    'sales_order_driven': 0,
                    'high_confidence_forecast': 0,
                    'medium_confidence_forecast': 0,
                    'low_confidence_forecast': 0,
                    'material_available': 0,
                    'material_shortage': 0,
                    'total_suggested_production': 0
                },
                'suggestions': []
            }

def create_enhanced_suggestions_v2(analyzer):
    """Factory function to create suggestions with analyzer instance"""
    engine = EnhancedProductionSuggestionsV2(analyzer)
    suggestions_result = engine.generate_suggestions()
    
    # Apply business logic if available
    try:
        from production.production_business_logic import apply_business_logic
        if suggestions_result.get('status') == 'success' and suggestions_result.get('suggestions'):
            # Enhance suggestions with business logic
            enhanced = apply_business_logic(suggestions_result['suggestions'], analyzer)
            suggestions_result['suggestions'] = enhanced
            suggestions_result['summary']['total_suggestions'] = len(enhanced)
            
            # Update summary with business logic metrics
            if enhanced:
                suggestions_result['summary']['critical_priority'] = len([s for s in enhanced if s.get('priority_rank') == 'CRITICAL'])
                suggestions_result['summary']['high_priority'] = len([s for s in enhanced if s.get('priority_rank') == 'HIGH'])
                suggestions_result['summary']['total_suggested_qty'] = sum(s.get('suggested_quantity_lbs', 0) for s in enhanced)
                
                # Add business logic recommendations
                business_recommendations = []
                
                critical_count = suggestions_result['summary'].get('critical_priority', 0)
                if critical_count > 0:
                    business_recommendations.append(f"URGENT: {critical_count} critical priority items need immediate production")
                
                material_shortage = len([s for s in enhanced if not s.get('yarn_available', True)])
                if material_shortage > 0:
                    business_recommendations.append(f"Material shortage affecting {material_shortage} styles - prioritize procurement")
                
                seasonal = len([s for s in enhanced if s.get('style_category') == 'SEASONAL'])
                if seasonal > 0:
                    business_recommendations.append(f"{seasonal} seasonal items detected - adjust lead times accordingly")
                
                if business_recommendations:
                    suggestions_result['recommendations'].extend(business_recommendations)
                    
    except ImportError as e:
        print(f"Business logic not available: {e}")
    
    return suggestions_result