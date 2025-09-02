"""
Enhanced Production Pipeline Module for Beverly Knits ERP
Uses knit orders as the primary data source for production tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class EnhancedProductionPipeline:
    """Enhanced production pipeline using knit orders data"""
    
    def __init__(self, knit_orders_data: Optional[pd.DataFrame] = None):
        self.knit_orders = knit_orders_data
        self.stage_mapping = {
            'KNT': 'Knitting',
            'DYE': 'Dyeing', 
            'FIN': 'Finishing',
            'QC': 'Quality Control',
            'PKG': 'Packaging',
            'SHP': 'Shipping'
        }
        
    def process_knit_orders(self) -> Dict[str, Any]:
        """Process knit orders data and return production pipeline metrics"""
        
        if self.knit_orders is None or self.knit_orders.empty:
            return self._empty_response()
            
        result = {
            'pipeline': [],
            'summary': {},
            'critical_orders': [],
            'bottlenecks': [],
            'recommendations': [],
            'data_source': 'knit_orders',
            'last_updated': datetime.now().isoformat()
        }
        
        # Process pipeline stages
        result['pipeline'] = self._calculate_stage_metrics()
        
        # Calculate summary statistics
        result['summary'] = self._calculate_summary()
        
        # Identify critical orders
        result['critical_orders'] = self._identify_critical_orders()
        
        # Identify bottlenecks
        result['bottlenecks'] = self._identify_bottlenecks(result['pipeline'])
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result['pipeline'], result['bottlenecks'])
        
        return result
    
    def _calculate_stage_metrics(self) -> List[Dict[str, Any]]:
        """Calculate metrics for each production stage"""
        stages = []
        
        # Find the stage/status column
        stage_col = self._find_column(['Stage', 'Status', 'Production Stage', 'Current Stage'])
        if not stage_col:
            # If no stage column, analyze by order status patterns
            return self._analyze_by_status_patterns()
            
        # Group by stages
        for stage in self.knit_orders[stage_col].unique():
            if pd.isna(stage):
                continue
                
            stage_df = self.knit_orders[self.knit_orders[stage_col] == stage]
            
            # Calculate metrics
            metrics = {
                'stage': str(stage),
                'order_count': len(stage_df),
                'total_quantity': self._calculate_total_quantity(stage_df),
                'avg_completion': self._calculate_completion_rate(stage_df),
                'efficiency': self._calculate_efficiency(stage_df),
                'utilization': self._calculate_utilization(len(stage_df)),
                'on_time_rate': self._calculate_on_time_rate(stage_df),
                'late_orders': self._count_late_orders(stage_df),
                'bottleneck_status': self._determine_bottleneck_status(stage_df),
                'avg_cycle_time': self._calculate_avg_cycle_time(stage_df),
                'wip_value': self._calculate_wip_value(stage_df)
            }
            
            stages.append(metrics)
            
        return sorted(stages, key=lambda x: self._get_stage_order(x['stage']))
    
    def _analyze_by_status_patterns(self) -> List[Dict[str, Any]]:
        """Analyze production by status patterns when no explicit stage column exists"""
        stages = []
        
        # Define status patterns for each stage
        patterns = {
            'Knitting': ['knit', 'knt', 'knitting'],
            'Dyeing': ['dye', 'dyeing', 'color'],
            'Finishing': ['finish', 'fin', 'finishing'],
            'Quality Control': ['qc', 'inspect', 'quality', 'check'],
            'Packaging': ['pack', 'pkg', 'packing'],
            'Shipping': ['ship', 'shp', 'dispatch', 'deliver']
        }
        
        # Try to find a status column
        status_col = self._find_column(['Status', 'Order Status', 'Current Status', 'Stage'])
        
        if status_col:
            for stage_name, keywords in patterns.items():
                # Filter orders matching this stage
                mask = pd.Series([False] * len(self.knit_orders))
                for keyword in keywords:
                    mask |= self.knit_orders[status_col].str.contains(keyword, case=False, na=False)
                
                stage_df = self.knit_orders[mask]
                
                if not stage_df.empty:
                    metrics = {
                        'stage': stage_name,
                        'order_count': len(stage_df),
                        'total_quantity': self._calculate_total_quantity(stage_df),
                        'avg_completion': 50,  # Default when no completion data
                        'efficiency': self._calculate_efficiency(stage_df),
                        'utilization': self._calculate_utilization(len(stage_df)),
                        'on_time_rate': self._calculate_on_time_rate(stage_df),
                        'late_orders': self._count_late_orders(stage_df),
                        'bottleneck_status': self._determine_bottleneck_status(stage_df),
                        'avg_cycle_time': 'N/A',
                        'wip_value': self._calculate_wip_value(stage_df)
                    }
                    stages.append(metrics)
        
        # If no status patterns found, return all as "In Process"
        if not stages:
            stages.append({
                'stage': 'All Orders',
                'order_count': len(self.knit_orders),
                'total_quantity': self._calculate_total_quantity(self.knit_orders),
                'avg_completion': 50,
                'efficiency': 85,
                'utilization': 75,
                'on_time_rate': 80,
                'late_orders': 0,
                'bottleneck_status': 'Normal',
                'avg_cycle_time': 'N/A',
                'wip_value': self._calculate_wip_value(self.knit_orders)
            })
            
        return stages
    
    def _calculate_total_quantity(self, df: pd.DataFrame) -> float:
        """Calculate total quantity from dataframe - use Balance (lbs) for accurate remaining work"""
        # For knit orders, use Balance (lbs) as it represents work remaining
        balance_col = self._find_column(['Balance (lbs)', 'Balance', 'Remaining'], df)
        if balance_col:
            return float(df[balance_col].sum())
        # Fallback to quantity columns
        qty_col = self._find_column(['Qty Ordered (lbs)', 'Qty', 'Quantity', 'Order Qty'], df)
        if qty_col:
            return float(df[qty_col].sum())
        return 0
    
    def _calculate_completion_rate(self, df: pd.DataFrame) -> float:
        """Calculate average completion rate"""
        comp_col = self._find_column(['Completion', 'Progress', '%Complete', 'Completion %'], df)
        if comp_col:
            return float(df[comp_col].mean())
        return 50  # Default
    
    def _calculate_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate production efficiency - G00 produced vs Qty Ordered"""
        # Correct efficiency calculation: (G00 / Qty Ordered) * 100
        g00_col = self._find_column(['G00 (lbs)', 'G00', 'Produced'], df)
        qty_col = self._find_column(['Qty Ordered (lbs)', 'Qty', 'Quantity'], df)
        
        if g00_col and qty_col:
            # Remove NaN values for calculation
            valid_rows = df[[g00_col, qty_col]].dropna()
            if not valid_rows.empty:
                total_g00 = valid_rows[g00_col].sum()
                total_qty = valid_rows[qty_col].sum()
                if total_qty > 0:
                    efficiency = (total_g00 / total_qty) * 100
                    return min(100, efficiency)  # Cap at 100%
        
        # Fallback to on-time rate if production data not available
        on_time = self._calculate_on_time_rate(df)
        return on_time
    
    def _calculate_utilization(self, order_count: int) -> float:
        """Calculate capacity utilization based on realistic capacity"""
        # More realistic: 50 active orders represents full utilization
        # This accounts for machine capacity and workforce limitations
        FULL_CAPACITY_ORDERS = 50
        return min(100, (order_count / FULL_CAPACITY_ORDERS) * 100)
    
    def _calculate_on_time_rate(self, df: pd.DataFrame) -> float:
        """Calculate percentage of orders on time"""
        # Use Start Date or Quoted Date for knit orders
        date_col = self._find_column(['Start Date', 'Quoted Date', 'Due Date', 'Delivery Date'], df)
        if date_col:
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                today = pd.Timestamp.now()
                # Orders with future start date or no overdue balance are on-time
                on_time = len(df_copy[(df_copy[date_col] >= today) | df_copy[date_col].isna()])
                return (on_time / len(df_copy) * 100) if len(df_copy) > 0 else 100
            except:
                pass
        return 85  # Default
    
    def _count_late_orders(self, df: pd.DataFrame) -> int:
        """Count number of late orders"""
        # Use Start Date for knit orders
        date_col = self._find_column(['Start Date', 'Quoted Date', 'Due Date'], df)
        if date_col:
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                today = pd.Timestamp.now()
                # Count orders with past start date and remaining balance
                balance_col = self._find_column(['Balance (lbs)', 'Balance'], df_copy)
                if balance_col:
                    late_orders = df_copy[(df_copy[date_col] < today) & (df_copy[balance_col] > 0)]
                    return len(late_orders)
                else:
                    return len(df_copy[df_copy[date_col] < today])
            except:
                pass
        return 0
    
    def _determine_bottleneck_status(self, df: pd.DataFrame) -> str:
        """Determine if stage is a bottleneck"""
        late_count = self._count_late_orders(df)
        total_count = len(df)
        
        if total_count == 0:
            return 'Normal'
            
        late_ratio = late_count / total_count
        
        if late_ratio > 0.3:
            return 'Critical'
        elif late_ratio > 0.15:
            return 'Warning'
        else:
            return 'Normal'
    
    def _calculate_avg_cycle_time(self, df: pd.DataFrame) -> str:
        """Calculate average cycle time for orders in stage"""
        # Would need start/end timestamps for accurate calculation
        return 'N/A'
    
    def _calculate_wip_value(self, df: pd.DataFrame) -> float:
        """Calculate work in process value based on actual production data"""
        # WIP = (G00 - Shipped) * estimated cost per lb
        g00_col = self._find_column(['G00 (lbs)', 'G00', 'Produced'], df)
        shipped_col = self._find_column(['Shipped (lbs)', 'Shipped', 'Delivered'], df)
        
        if g00_col and shipped_col:
            valid_rows = df[[g00_col, shipped_col]].fillna(0)
            wip_quantity = (valid_rows[g00_col].sum() - valid_rows[shipped_col].sum())
            # Use $5 per lb as standard cost estimate for knitted fabric
            return max(0, wip_quantity * 5)
        
        # Fallback calculation
        qty = self._calculate_total_quantity(df)
        return qty * 5  # $5 per lb estimate
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics with correct formulas"""
        # Get column names for calculations
        qty_col = self._find_column(['Qty Ordered (lbs)', 'Qty', 'Quantity'])
        g00_col = self._find_column(['G00 (lbs)', 'G00', 'Produced'])
        shipped_col = self._find_column(['Shipped (lbs)', 'Shipped'])
        seconds_col = self._find_column(['Seconds (lbs)', 'Seconds', 'Defects'])
        balance_col = self._find_column(['Balance (lbs)', 'Balance'])
        
        # Validate balance calculations
        balance_errors = 0
        if all([qty_col, g00_col, shipped_col, balance_col]):
            for idx, row in self.knit_orders.iterrows():
                qty = row.get(qty_col, 0) or 0
                g00 = row.get(g00_col, 0) or 0
                shipped = row.get(shipped_col, 0) or 0
                seconds = row.get(seconds_col, 0) or 0 if seconds_col else 0
                recorded_balance = row.get(balance_col, 0) or 0
                
                # Correct formula: Balance = Qty Ordered - (G00 + Shipped + Seconds)
                calculated_balance = qty - (g00 + shipped + seconds)
                if abs(calculated_balance - recorded_balance) > 1:  # Allow 1 lb tolerance
                    balance_errors += 1
        
        summary = {
            'total_orders': len(self.knit_orders),
            'total_quantity_ordered': self._sum_column(qty_col) if qty_col else 0,
            'total_g00_produced': self._sum_column(g00_col) if g00_col else 0,
            'total_shipped': self._sum_column(shipped_col) if shipped_col else 0,
            'total_balance': self._sum_column(balance_col) if balance_col else 0,
            'balance_calculation_errors': balance_errors,
            'data_accuracy_note': f'{balance_errors} orders have incorrect balance calculations' if balance_errors > 0 else 'All balance calculations verified',
            'unique_styles': 0,
            'unique_customers': 0,
            'avg_order_size': 0,
            'total_wip_value': 0
        }
        
        # Count unique styles
        style_col = self._find_column(['Style', 'Style #', 'Style Number'])
        if style_col:
            summary['unique_styles'] = self.knit_orders[style_col].nunique()
            
        # Count unique customers
        customer_col = self._find_column(['Customer', 'Client', 'Account'])
        if customer_col:
            summary['unique_customers'] = self.knit_orders[customer_col].nunique()
            
        # Calculate average order size
        if summary['total_orders'] > 0:
            summary['avg_order_size'] = summary['total_quantity_ordered'] / summary['total_orders']
            
        # Calculate total WIP value
        summary['total_wip_value'] = self._calculate_wip_value(self.knit_orders)
        
        return summary
    
    def _sum_column(self, col_name: str) -> float:
        """Safely sum a column handling NaN values"""
        if col_name and col_name in self.knit_orders.columns:
            return float(self.knit_orders[col_name].fillna(0).sum())
        return 0
    
    def _identify_critical_orders(self) -> List[Dict[str, Any]]:
        """Identify orders that need immediate attention"""
        critical = []
        
        # Find date column
        date_col = self._find_column(['Due Date', 'Delivery Date', 'Ship Date'])
        if not date_col:
            return critical
            
        try:
            self.knit_orders[date_col] = pd.to_datetime(self.knit_orders[date_col], errors='coerce')
            today = pd.Timestamp.now()
            next_week = today + pd.Timedelta(days=7)
            
            # Orders due within next 7 days
            critical_df = self.knit_orders[
                (self.knit_orders[date_col] >= today) & 
                (self.knit_orders[date_col] <= next_week)
            ].copy()
            
            # Sort by due date
            critical_df = critical_df.sort_values(date_col)
            
            # Get order details
            for _, row in critical_df.head(10).iterrows():
                order = {
                    'order_id': self._get_order_id(row),
                    'style': self._get_style(row),
                    'customer': self._get_customer(row),
                    'quantity': self._get_quantity(row),
                    'due_date': row[date_col].strftime('%Y-%m-%d') if pd.notna(row[date_col]) else 'N/A',
                    'days_until_due': (row[date_col] - today).days if pd.notna(row[date_col]) else 0,
                    'status': self._get_status(row),
                    'priority': 'HIGH'
                }
                critical.append(order)
                
        except Exception as e:
            print(f"Error identifying critical orders: {e}")
            
        return critical
    
    def _identify_bottlenecks(self, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """Identify production bottlenecks"""
        bottlenecks = []
        
        for stage in pipeline:
            if stage['bottleneck_status'] in ['Warning', 'Critical']:
                bottlenecks.append({
                    'stage': stage['stage'],
                    'severity': stage['bottleneck_status'],
                    'late_orders': stage['late_orders'],
                    'utilization': stage['utilization'],
                    'impact': f"{stage['late_orders']} orders delayed",
                    'recommendation': self._get_bottleneck_recommendation(stage)
                })
                
        return bottlenecks
    
    def _generate_recommendations(self, pipeline: List[Dict], bottlenecks: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for critical bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b['severity'] == 'Critical']
        if critical_bottlenecks:
            recommendations.append(f"URGENT: Address critical bottleneck in {critical_bottlenecks[0]['stage']} - {critical_bottlenecks[0]['impact']}")
            
        # Check overall efficiency
        avg_efficiency = np.mean([s['efficiency'] for s in pipeline])
        if avg_efficiency < 70:
            recommendations.append(f"Overall efficiency is low ({avg_efficiency:.1f}%). Review production scheduling and resource allocation.")
            
        # Check for high WIP
        total_wip = sum(s['order_count'] for s in pipeline)
        if total_wip > 500:
            recommendations.append(f"High WIP levels ({total_wip} orders). Consider implementing pull systems to reduce inventory.")
            
        # Check for late orders
        total_late = sum(s['late_orders'] for s in pipeline)
        if total_late > 10:
            recommendations.append(f"{total_late} orders are late. Expedite processing and communicate with customers.")
            
        if not recommendations:
            recommendations.append("Production pipeline is operating normally. Continue monitoring for changes.")
            
        return recommendations
    
    def _get_bottleneck_recommendation(self, stage: Dict) -> str:
        """Get specific recommendation for bottleneck with validation"""
        try:
            utilization = float(stage.get('utilization', 0))
            late_orders = int(stage.get('late_orders', 0))
            
            if utilization > 90:
                return "Add capacity or shift resources to this stage"
            elif late_orders > 5:
                return "Expedite late orders and review scheduling"
            else:
                return "Monitor closely and prepare contingency plans"
        except (ValueError, TypeError):
            return "Review stage performance and data quality"
    
    # Helper methods for finding columns and extracting data
    def _find_column(self, possible_names: List[str], df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Find column by possible names with validation"""
        if df is None:
            df = self.knit_orders
            
        if df is None:
            return None
        
        # Validate dataframe has columns attribute and is not empty
        if not hasattr(df, 'columns') or df.empty:
            return None
            
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _get_order_id(self, row: pd.Series) -> str:
        """Extract order ID from row"""
        for col in ['Order #', 'Order ID', 'SO#', 'Order Number']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return 'N/A'
    
    def _get_style(self, row: pd.Series) -> str:
        """Extract style from row"""
        for col in ['Style', 'Style #', 'Style Number', 'Product']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return 'N/A'
    
    def _get_customer(self, row: pd.Series) -> str:
        """Extract customer from row"""
        for col in ['Customer', 'Client', 'Account', 'Buyer']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return 'N/A'
    
    def _get_quantity(self, row: pd.Series) -> float:
        """Extract quantity from row"""
        for col in ['Qty', 'Quantity', 'Order Qty', 'Qty (yds)', 'Qty (lbs)']:
            if col in row.index and pd.notna(row[col]):
                try:
                    return float(row[col])
                except:
                    pass
        return 0
    
    def _get_status(self, row: pd.Series) -> str:
        """Extract status from row"""
        for col in ['Status', 'Order Status', 'Stage', 'Current Status']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return 'In Process'
    
    def _get_stage_order(self, stage: str) -> int:
        """Get sort order for stage"""
        order = {
            'Knitting': 1,
            'Dyeing': 2,
            'Finishing': 3,
            'Quality Control': 4,
            'Packaging': 5,
            'Shipping': 6
        }
        return order.get(stage, 99)
    
    def _empty_response(self) -> Dict[str, Any]:
        """Return empty response structure"""
        return {
            'pipeline': [],
            'summary': {
                'total_orders': 0,
                'total_quantity': 0,
                'unique_styles': 0,
                'unique_customers': 0
            },
            'critical_orders': [],
            'bottlenecks': [],
            'recommendations': ['No knit orders data available'],
            'data_source': 'none',
            'last_updated': datetime.now().isoformat()
        }


def integrate_with_flask(analyzer):
    """
    Integration function to be added to beverly_comprehensive_erp.py
    This replaces the existing production pipeline endpoint
    """
    from flask import jsonify
    
    # Create enhanced pipeline processor
    if hasattr(analyzer, 'knit_orders_data') and analyzer.knit_orders_data is not None:
        processor = EnhancedProductionPipeline(analyzer.knit_orders_data)
        result = processor.process_knit_orders()
    else:
        processor = EnhancedProductionPipeline()
        result = processor._empty_response()
        
    return jsonify(result)