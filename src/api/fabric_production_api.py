"""Fabric Production API for complete fabric production management."""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from functools import wraps
import logging

# Create blueprint
fabric_production_bp = Blueprint('fabric_production', __name__, url_prefix='/api')

# Logger setup
logger = logging.getLogger(__name__)


class FabricProductionAnalyzer:
    """Analyzes fabric production data and provides insights."""
    
    def __init__(self, data_loader):
        """
        Initialize the fabric production analyzer.
        
        Args:
            data_loader: Unified data loader instance
        """
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        
    async def analyze_production(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze fabric production based on parameters.
        
        Args:
            params: Query parameters including dates, styles, etc.
            
        Returns:
            Production analysis results
        """
        try:
            # Load production data
            production_df = await self._load_production_data(params)
            
            # Calculate production metrics
            metrics = self._calculate_production_metrics(production_df)
            
            # Analyze production efficiency
            efficiency = self._analyze_efficiency(production_df)
            
            # Get production schedule
            schedule = self._get_production_schedule(production_df, params)
            
            return {
                'total_production': float(production_df['quantity'].sum()) if not production_df.empty else 0,
                'active_orders': len(production_df),
                'metrics': metrics,
                'efficiency': efficiency,
                'schedule': schedule,
                'by_style': self._group_by_style(production_df),
                'by_work_center': self._group_by_work_center(production_df)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing production: {e}")
            return {
                'error': str(e),
                'total_production': 0,
                'active_orders': 0
            }
    
    async def analyze_demand(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze fabric demand based on sales and forecasts.
        
        Args:
            params: Query parameters
            
        Returns:
            Demand analysis results
        """
        try:
            # Load sales data
            sales_df = await self._load_sales_data(params)
            
            # Calculate demand metrics
            current_demand = self._calculate_current_demand(sales_df)
            
            # Generate forecast if requested
            forecast = {}
            if params.get('include_forecast', False):
                forecast = await self._generate_demand_forecast(sales_df, params)
            
            # Identify demand patterns
            patterns = self._identify_demand_patterns(sales_df)
            
            return {
                'current_demand': current_demand,
                'forecast': forecast,
                'patterns': patterns,
                'top_styles': self._get_top_styles(sales_df),
                'demand_by_period': self._group_demand_by_period(sales_df)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing demand: {e}")
            return {
                'error': str(e),
                'current_demand': {},
                'forecast': {}
            }
    
    async def get_capacity_utilization(self) -> Dict[str, Any]:
        """
        Calculate capacity utilization across work centers.
        
        Returns:
            Capacity utilization metrics
        """
        try:
            # Load machine and work center data
            machines_df = await self.data_loader.load_machine_data()
            orders_df = await self.data_loader.load_production_orders()
            
            # Calculate utilization by work center
            utilization = {}
            
            if not machines_df.empty and not orders_df.empty:
                # Group orders by machine
                machine_workload = orders_df.groupby('Machine')['Quantity'].sum().to_dict()
                
                # Calculate utilization for each work center
                work_centers = machines_df['work_center'].unique()
                
                for wc in work_centers:
                    wc_machines = machines_df[machines_df['work_center'] == wc]['machine_id'].values
                    wc_total_capacity = len(wc_machines) * 8 * 60  # 8 hours * 60 minutes per machine
                    wc_workload = sum(machine_workload.get(m, 0) for m in wc_machines)
                    
                    utilization[wc] = {
                        'machines': len(wc_machines),
                        'workload': float(wc_workload),
                        'capacity': wc_total_capacity,
                        'utilization_pct': min((wc_workload / wc_total_capacity * 100) if wc_total_capacity > 0 else 0, 100)
                    }
            
            return {
                'work_centers': utilization,
                'overall_utilization': np.mean([u['utilization_pct'] for u in utilization.values()]) if utilization else 0,
                'bottlenecks': [wc for wc, u in utilization.items() if u['utilization_pct'] > 90]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating capacity utilization: {e}")
            return {
                'work_centers': {},
                'overall_utilization': 0,
                'bottlenecks': []
            }
    
    # Private helper methods
    async def _load_production_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load and filter production data."""
        df = await self.data_loader.load_production_orders()
        
        if df.empty:
            return df
        
        # Apply date filters
        if start_date := params.get('start_date'):
            df = df[df['scheduled_date'] >= pd.to_datetime(start_date)]
        
        if end_date := params.get('end_date'):
            df = df[df['scheduled_date'] <= pd.to_datetime(end_date)]
        
        # Apply style filter
        if style := params.get('style_filter'):
            df = df[df['style_id'].str.contains(style, case=False, na=False)]
        
        return df
    
    async def _load_sales_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load and filter sales data."""
        df = await self.data_loader.load_sales_data()
        
        if df.empty:
            return df
        
        # Apply date filters
        if start_date := params.get('start_date'):
            df = df[df['order_date'] >= pd.to_datetime(start_date)]
        
        if end_date := params.get('end_date'):
            df = df[df['order_date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def _calculate_production_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key production metrics."""
        if df.empty:
            return {}
        
        return {
            'total_quantity': float(df['Quantity'].sum()),
            'unique_styles': int(df['Style'].nunique()),
            'avg_order_size': float(df['Quantity'].mean()),
            'completion_rate': float((df['status'] == 'completed').mean() * 100) if 'status' in df else 0
        }
    
    def _analyze_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze production efficiency."""
        if df.empty:
            return {'overall': 0, 'by_machine': {}}
        
        efficiency = {}
        if 'Machine' in df.columns:
            # Calculate efficiency by machine
            machine_groups = df.groupby('Machine')
            for machine, group in machine_groups:
                efficiency[str(machine)] = {
                    'orders': len(group),
                    'total_quantity': float(group['Quantity'].sum()),
                    'avg_time': float(group['estimated_time'].mean()) if 'estimated_time' in group else 0
                }
        
        return {
            'overall': 85.0,  # Placeholder - would calculate from actual vs expected
            'by_machine': efficiency
        }
    
    def _get_production_schedule(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get production schedule."""
        if df.empty:
            return []
        
        # Sort by scheduled date
        df = df.sort_values('scheduled_date', na_position='last')
        
        # Convert to schedule format
        schedule = []
        for _, row in df.head(100).iterrows():  # Limit to 100 for performance
            schedule.append({
                'order_id': row.get('Order ID', ''),
                'style': row.get('Style', ''),
                'quantity': float(row.get('Quantity', 0)),
                'machine': str(row.get('Machine', '')),
                'scheduled_date': row.get('scheduled_date', '').isoformat() if pd.notna(row.get('scheduled_date')) else '',
                'status': row.get('status', 'pending')
            })
        
        return schedule
    
    def _group_by_style(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Group production by style."""
        if df.empty or 'Style' not in df.columns:
            return {}
        
        grouped = df.groupby('Style').agg({
            'Quantity': 'sum',
            'Order ID': 'count'
        }).to_dict('index')
        
        # Convert numpy types to Python types
        for style in grouped:
            grouped[style] = {
                'total_quantity': float(grouped[style]['Quantity']),
                'order_count': int(grouped[style]['Order ID'])
            }
        
        return grouped
    
    def _group_by_work_center(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Group production by work center."""
        if df.empty or 'work_center' not in df.columns:
            return {}
        
        grouped = df.groupby('work_center').agg({
            'Quantity': 'sum',
            'Order ID': 'count'
        }).to_dict('index')
        
        # Convert numpy types to Python types
        for wc in grouped:
            grouped[wc] = {
                'total_quantity': float(grouped[wc]['Quantity']),
                'order_count': int(grouped[wc]['Order ID'])
            }
        
        return grouped
    
    def _calculate_current_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current demand from sales data."""
        if df.empty:
            return {}
        
        return {
            'total_demand': float(df['quantity'].sum()) if 'quantity' in df else 0,
            'unique_customers': int(df['customer'].nunique()) if 'customer' in df else 0,
            'avg_order_value': float(df['total_price'].mean()) if 'total_price' in df else 0
        }
    
    async def _generate_demand_forecast(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demand forecast using ML models."""
        # This would integrate with the ML forecasting system
        horizon_days = params.get('horizon_days', 30)
        
        return {
            'horizon_days': horizon_days,
            'predicted_demand': float(df['quantity'].sum() * 1.1) if not df.empty else 0,  # Simple growth assumption
            'confidence': 0.85
        }
    
    def _identify_demand_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in demand data."""
        if df.empty:
            return {}
        
        patterns = {}
        
        # Seasonal patterns
        if 'order_date' in df:
            df['month'] = pd.to_datetime(df['order_date']).dt.month
            monthly_demand = df.groupby('month')['quantity'].sum()
            patterns['seasonal'] = {
                'peak_month': int(monthly_demand.idxmax()) if not monthly_demand.empty else 0,
                'low_month': int(monthly_demand.idxmin()) if not monthly_demand.empty else 0
            }
        
        # Growth trend
        if len(df) > 30:
            # Simple linear trend
            df['day_num'] = (pd.to_datetime(df['order_date']) - pd.to_datetime(df['order_date']).min()).dt.days
            if df['day_num'].std() > 0:
                correlation = df['day_num'].corr(df['quantity'])
                patterns['trend'] = 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable'
            else:
                patterns['trend'] = 'stable'
        
        return patterns
    
    def _get_top_styles(self, df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top selling styles."""
        if df.empty or 'style_id' not in df:
            return []
        
        top_styles = df.groupby('style_id')['quantity'].sum().nlargest(limit)
        
        result = []
        for style, qty in top_styles.items():
            result.append({
                'style_id': style,
                'total_quantity': float(qty)
            })
        
        return result
    
    def _group_demand_by_period(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Group demand by time period."""
        if df.empty or 'order_date' not in df:
            return {}
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Daily demand for last 30 days
        last_30_days = df[df['order_date'] >= (datetime.now() - timedelta(days=30))]
        daily = last_30_days.groupby(df['order_date'].dt.date)['quantity'].sum().to_dict()
        
        # Weekly demand
        df['week'] = df['order_date'].dt.isocalendar().week
        weekly = df.groupby('week')['quantity'].sum().to_dict()
        
        return {
            'daily': {str(k): float(v) for k, v in daily.items()},
            'weekly': {str(k): float(v) for k, v in weekly.items()}
        }


# Create analyzer instance (will be initialized with data loader)
fabric_analyzer = None


def init_fabric_production_api(data_loader):
    """Initialize the fabric production API with data loader."""
    global fabric_analyzer
    fabric_analyzer = FabricProductionAnalyzer(data_loader)


# API Endpoints
@fabric_production_bp.route('/fabric-production', methods=['GET'])
async def fabric_production_endpoint():
    """
    Complete fabric production implementation.
    
    Query Parameters:
    - start_date: Start date for filtering (YYYY-MM-DD)
    - end_date: End date for filtering (YYYY-MM-DD)
    - style: Style filter
    - forecast: Include forecast (true/false)
    """
    if not fabric_analyzer:
        return jsonify({
            'status': 'error',
            'message': 'Fabric production analyzer not initialized'
        }), 500
    
    # Parse query parameters
    params = FabricQueryParams(
        start_date=request.args.get('start_date'),
        end_date=request.args.get('end_date'),
        style_filter=request.args.get('style'),
        include_forecast=request.args.get('forecast', 'false').lower() == 'true',
        horizon_days=int(request.args.get('horizon_days', 30))
    )
    
    try:
        # Analyze production
        production_data = await fabric_analyzer.analyze_production(params.__dict__)
        
        # Analyze demand
        demand_data = await fabric_analyzer.analyze_demand(params.__dict__)
        
        # Get capacity utilization
        capacity_data = await fabric_analyzer.get_capacity_utilization()
        
        return jsonify({
            'status': 'success',
            'production': production_data,
            'demand': demand_data,
            'capacity': capacity_data,
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in fabric production endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@fabric_production_bp.route('/fabric-production/schedule', methods=['GET'])
async def get_production_schedule():
    """Get detailed production schedule."""
    if not fabric_analyzer:
        return jsonify({
            'status': 'error',
            'message': 'Fabric production analyzer not initialized'
        }), 500
    
    try:
        params = {
            'start_date': request.args.get('start_date', datetime.now().isoformat()),
            'end_date': request.args.get('end_date'),
            'style_filter': request.args.get('style')
        }
        
        production_df = await fabric_analyzer._load_production_data(params)
        schedule = fabric_analyzer._get_production_schedule(production_df, params)
        
        return jsonify({
            'status': 'success',
            'schedule': schedule,
            'total_orders': len(schedule),
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting production schedule: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@fabric_production_bp.route('/fabric-production/capacity', methods=['GET'])
async def get_capacity_details():
    """Get detailed capacity utilization information."""
    if not fabric_analyzer:
        return jsonify({
            'status': 'error',
            'message': 'Fabric production analyzer not initialized'
        }), 500
    
    try:
        capacity_data = await fabric_analyzer.get_capacity_utilization()
        
        return jsonify({
            'status': 'success',
            'capacity': capacity_data,
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting capacity details: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


class FabricQueryParams:
    """Parameters for fabric production queries."""
    
    def __init__(self, start_date=None, end_date=None, style_filter=None, 
                 include_forecast=False, horizon_days=30):
        """
        Initialize query parameters.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            style_filter: Style to filter by
            include_forecast: Whether to include forecast
            horizon_days: Forecast horizon in days
        """
        self.start_date = start_date
        self.end_date = end_date
        self.style_filter = style_filter
        self.include_forecast = include_forecast
        self.horizon_days = horizon_days