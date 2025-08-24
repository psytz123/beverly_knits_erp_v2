#!/usr/bin/env python3
"""
Integrated Inventory Analysis Module
Combines inventory analysis with forecasting and optimization for comprehensive insights
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies
try:
    from .inventory_analyzer_service import InventoryAnalyzerService, InventoryConfig
    from .inventory_pipeline_service import InventoryManagementPipelineService
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    logger.warning("Inventory services not available")


@dataclass
class AnalysisConfig:
    """Configuration for integrated analysis"""
    analysis_horizon_days: int = 90
    risk_threshold_critical: float = 0.8
    risk_threshold_high: float = 0.6
    risk_threshold_medium: float = 0.4
    include_seasonality: bool = True
    include_trends: bool = True
    confidence_threshold: float = 0.7


class IntegratedInventoryAnalysis:
    """
    Integrated inventory analysis combining multiple data sources and analytics
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.analyzer = InventoryAnalyzerService() if SERVICES_AVAILABLE else None
        self.pipeline = InventoryManagementPipelineService() if SERVICES_AVAILABLE else None
        self.analysis_cache = {}
        logger.info("IntegratedInventoryAnalysis initialized")
    
    def run_comprehensive_analysis(self,
                                 inventory_data: pd.DataFrame,
                                 sales_data: pd.DataFrame,
                                 forecast_data: Optional[pd.DataFrame] = None,
                                 bom_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive inventory analysis across all products
        """
        try:
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'summary': {},
                'detailed_analysis': [],
                'risk_assessment': {},
                'recommendations': [],
                'kpis': {}
            }
            
            # Step 1: Basic inventory metrics
            inventory_metrics = self._calculate_inventory_metrics(inventory_data)
            analysis_results['summary']['inventory_metrics'] = inventory_metrics
            
            # Step 2: Demand analysis
            demand_analysis = self._analyze_demand_patterns(sales_data)
            analysis_results['summary']['demand_patterns'] = demand_analysis
            
            # Step 3: Risk assessment
            risk_assessment = self._assess_inventory_risks(
                inventory_data, sales_data, forecast_data
            )
            analysis_results['risk_assessment'] = risk_assessment
            
            # Step 4: Product-level analysis
            product_analysis = self._analyze_by_product(
                inventory_data, sales_data, forecast_data, bom_data
            )
            analysis_results['detailed_analysis'] = product_analysis
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                inventory_metrics, demand_analysis, risk_assessment
            )
            analysis_results['recommendations'] = recommendations
            
            # Step 6: Calculate KPIs
            kpis = self._calculate_kpis(inventory_data, sales_data)
            analysis_results['kpis'] = kpis
            
            # Cache results
            self.analysis_cache['latest'] = analysis_results
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_inventory_metrics(self, inventory_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic inventory metrics"""
        metrics = {
            'total_products': len(inventory_data),
            'total_value': inventory_data['quantity'].sum() if 'quantity' in inventory_data else 0,
            'avg_stock_level': inventory_data['quantity'].mean() if 'quantity' in inventory_data else 0,
            'zero_stock_items': len(inventory_data[inventory_data['quantity'] == 0]) if 'quantity' in inventory_data else 0,
            'low_stock_items': 0,
            'overstocked_items': 0
        }
        
        if 'quantity' in inventory_data:
            # Define thresholds
            low_threshold = metrics['avg_stock_level'] * 0.3
            high_threshold = metrics['avg_stock_level'] * 2.0
            
            metrics['low_stock_items'] = len(inventory_data[
                (inventory_data['quantity'] > 0) & 
                (inventory_data['quantity'] < low_threshold)
            ])
            metrics['overstocked_items'] = len(inventory_data[
                inventory_data['quantity'] > high_threshold
            ])
        
        return metrics
    
    def _analyze_demand_patterns(self, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand patterns from sales data"""
        if sales_data.empty:
            return {'status': 'no_data'}
        
        patterns = {
            'total_demand': sales_data['quantity'].sum() if 'quantity' in sales_data else 0,
            'unique_products': sales_data['product_id'].nunique() if 'product_id' in sales_data else 0,
            'demand_volatility': {},
            'seasonal_patterns': {},
            'trends': {}
        }
        
        if 'date' in sales_data.columns and 'quantity' in sales_data.columns:
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Calculate daily demand
            daily_demand = sales_data.groupby('date')['quantity'].sum()
            
            # Volatility metrics
            patterns['demand_volatility'] = {
                'std_dev': daily_demand.std(),
                'coefficient_variation': daily_demand.std() / daily_demand.mean() if daily_demand.mean() > 0 else 0,
                'max_daily': daily_demand.max(),
                'min_daily': daily_demand.min()
            }
            
            # Detect weekly patterns
            if len(daily_demand) > 7:
                sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
                weekly_pattern = sales_data.groupby('day_of_week')['quantity'].mean()
                patterns['seasonal_patterns']['weekly'] = weekly_pattern.to_dict()
            
            # Trend analysis
            if len(daily_demand) > 30:
                recent_30_days = daily_demand.tail(30).mean()
                previous_30_days = daily_demand.iloc[-60:-30].mean() if len(daily_demand) > 60 else daily_demand.head(30).mean()
                trend_pct = ((recent_30_days - previous_30_days) / previous_30_days * 100) if previous_30_days > 0 else 0
                
                patterns['trends'] = {
                    'recent_avg': recent_30_days,
                    'previous_avg': previous_30_days,
                    'trend_percentage': trend_pct,
                    'trend_direction': 'increasing' if trend_pct > 5 else 'decreasing' if trend_pct < -5 else 'stable'
                }
        
        return patterns
    
    def _assess_inventory_risks(self, 
                              inventory_data: pd.DataFrame,
                              sales_data: pd.DataFrame,
                              forecast_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Assess inventory-related risks"""
        risks = {
            'critical_items': [],
            'high_risk_items': [],
            'medium_risk_items': [],
            'risk_summary': {}
        }
        
        # Calculate average daily demand by product
        if 'product_id' in sales_data.columns and 'quantity' in sales_data.columns:
            daily_demand = sales_data.groupby('product_id')['quantity'].mean()
            
            for _, item in inventory_data.iterrows():
                product_id = item.get('product_id', item.get('id', ''))
                current_stock = item.get('quantity', 0)
                
                if product_id in daily_demand.index:
                    avg_demand = daily_demand[product_id]
                    days_of_supply = current_stock / avg_demand if avg_demand > 0 else 999
                    
                    risk_item = {
                        'product_id': product_id,
                        'current_stock': current_stock,
                        'avg_daily_demand': avg_demand,
                        'days_of_supply': days_of_supply
                    }
                    
                    # Categorize risk
                    if days_of_supply < 7:
                        risk_item['risk_level'] = 'critical'
                        risks['critical_items'].append(risk_item)
                    elif days_of_supply < 14:
                        risk_item['risk_level'] = 'high'
                        risks['high_risk_items'].append(risk_item)
                    elif days_of_supply < 30:
                        risk_item['risk_level'] = 'medium'
                        risks['medium_risk_items'].append(risk_item)
        
        # Risk summary
        risks['risk_summary'] = {
            'total_at_risk': len(risks['critical_items']) + len(risks['high_risk_items']) + len(risks['medium_risk_items']),
            'critical_count': len(risks['critical_items']),
            'high_count': len(risks['high_risk_items']),
            'medium_count': len(risks['medium_risk_items']),
            'risk_percentage': (len(risks['critical_items']) + len(risks['high_risk_items'])) / len(inventory_data) * 100 if len(inventory_data) > 0 else 0
        }
        
        return risks
    
    def _analyze_by_product(self,
                          inventory_data: pd.DataFrame,
                          sales_data: pd.DataFrame,
                          forecast_data: Optional[pd.DataFrame] = None,
                          bom_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Detailed analysis for each product"""
        product_analysis = []
        
        # Get unique products
        products = inventory_data['product_id'].unique() if 'product_id' in inventory_data else []
        
        for product_id in products[:50]:  # Limit to top 50 for performance
            analysis = {
                'product_id': product_id,
                'current_stock': 0,
                'demand_stats': {},
                'forecast': {},
                'recommendations': []
            }
            
            # Current stock
            product_inv = inventory_data[inventory_data['product_id'] == product_id]
            if not product_inv.empty:
                analysis['current_stock'] = product_inv.iloc[0]['quantity'] if 'quantity' in product_inv else 0
            
            # Demand statistics
            if 'product_id' in sales_data.columns:
                product_sales = sales_data[sales_data['product_id'] == product_id]
                if not product_sales.empty and 'quantity' in product_sales:
                    analysis['demand_stats'] = {
                        'total_demand': product_sales['quantity'].sum(),
                        'avg_daily': product_sales['quantity'].mean(),
                        'max_daily': product_sales['quantity'].max(),
                        'demand_days': len(product_sales)
                    }
            
            # Add forecast if available
            if forecast_data is not None and 'product_id' in forecast_data.columns:
                product_forecast = forecast_data[forecast_data['product_id'] == product_id]
                if not product_forecast.empty:
                    analysis['forecast'] = {
                        'next_30_days': product_forecast['forecast_30'].iloc[0] if 'forecast_30' in product_forecast else 0,
                        'next_60_days': product_forecast['forecast_60'].iloc[0] if 'forecast_60' in product_forecast else 0,
                        'next_90_days': product_forecast['forecast_90'].iloc[0] if 'forecast_90' in product_forecast else 0
                    }
            
            # Generate product-specific recommendations
            if analysis['demand_stats'] and analysis['current_stock'] > 0:
                days_of_supply = analysis['current_stock'] / analysis['demand_stats'].get('avg_daily', 1)
                
                if days_of_supply < 7:
                    analysis['recommendations'].append("URGENT: Reorder immediately - less than 7 days of supply")
                elif days_of_supply < 14:
                    analysis['recommendations'].append("Schedule reorder within 3-5 days")
                elif days_of_supply > 90:
                    analysis['recommendations'].append("Consider reducing stock levels - over 90 days of supply")
            
            product_analysis.append(analysis)
        
        return product_analysis
    
    def _generate_recommendations(self,
                                inventory_metrics: Dict[str, Any],
                                demand_analysis: Dict[str, Any],
                                risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical stock recommendations
        if risk_assessment['risk_summary']['critical_count'] > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'inventory',
                'recommendation': f"Immediate action required for {risk_assessment['risk_summary']['critical_count']} critical items with less than 7 days of supply",
                'impact': 'Prevent stockouts and lost sales',
                'items': [item['product_id'] for item in risk_assessment['critical_items'][:5]]
            })
        
        # Overstock recommendations
        if inventory_metrics.get('overstocked_items', 0) > inventory_metrics.get('total_products', 1) * 0.2:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'inventory',
                'recommendation': f"Review and reduce inventory for {inventory_metrics['overstocked_items']} overstocked items",
                'impact': 'Reduce holding costs and free up working capital'
            })
        
        # Demand trend recommendations
        if demand_analysis.get('trends', {}).get('trend_direction') == 'increasing':
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'planning',
                'recommendation': f"Demand trending up {demand_analysis['trends']['trend_percentage']:.1f}% - consider increasing safety stock levels",
                'impact': 'Maintain service levels with growing demand'
            })
        
        # Zero stock recommendations
        if inventory_metrics.get('zero_stock_items', 0) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'inventory',
                'recommendation': f"Review {inventory_metrics['zero_stock_items']} items with zero stock - determine if discontinued or need reorder",
                'impact': 'Optimize inventory mix and prevent stockouts'
            })
        
        return recommendations
    
    def _calculate_kpis(self, inventory_data: pd.DataFrame, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        kpis = {
            'inventory_turnover': 0,
            'days_inventory_outstanding': 0,
            'stockout_rate': 0,
            'service_level': 0,
            'inventory_accuracy': 95.0  # Default assumption
        }
        
        if not inventory_data.empty and not sales_data.empty:
            # Inventory turnover
            if 'quantity' in inventory_data and 'quantity' in sales_data:
                total_sales = sales_data['quantity'].sum()
                avg_inventory = inventory_data['quantity'].mean()
                
                if avg_inventory > 0:
                    # Annualized turnover
                    days_in_period = (sales_data['date'].max() - sales_data['date'].min()).days if 'date' in sales_data else 30
                    annual_sales = total_sales * (365 / days_in_period) if days_in_period > 0 else total_sales * 12
                    kpis['inventory_turnover'] = annual_sales / avg_inventory
                    kpis['days_inventory_outstanding'] = 365 / kpis['inventory_turnover'] if kpis['inventory_turnover'] > 0 else 0
            
            # Service level (simplified - percentage of items with stock)
            if 'quantity' in inventory_data:
                items_with_stock = len(inventory_data[inventory_data['quantity'] > 0])
                kpis['service_level'] = (items_with_stock / len(inventory_data) * 100) if len(inventory_data) > 0 else 0
                kpis['stockout_rate'] = 100 - kpis['service_level']
        
        return kpis


def run_inventory_analysis(inventory_data: pd.DataFrame,
                         sales_data: pd.DataFrame,
                         forecast_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Convenience function to run inventory analysis
    """
    analyzer = IntegratedInventoryAnalysis()
    return analyzer.run_comprehensive_analysis(
        inventory_data=inventory_data,
        sales_data=sales_data,
        forecast_data=forecast_data
    )


def get_yarn_shortage_report(inventory_data: pd.DataFrame,
                           demand_forecast: pd.DataFrame,
                           bom_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate yarn shortage report based on forecast and BOM
    """
    shortage_report = {
        'timestamp': datetime.now().isoformat(),
        'shortages': [],
        'at_risk': [],
        'summary': {}
    }
    
    # Simple shortage calculation
    for _, item in inventory_data.iterrows():
        product_id = item.get('product_id', '')
        current_stock = item.get('quantity', 0)
        
        # Get forecasted demand
        if product_id in demand_forecast['product_id'].values:
            forecast = demand_forecast[demand_forecast['product_id'] == product_id].iloc[0]
            required_qty = forecast.get('forecast_30', 0)
            
            if current_stock < required_qty:
                shortage = {
                    'product_id': product_id,
                    'current_stock': current_stock,
                    'required_quantity': required_qty,
                    'shortage_quantity': required_qty - current_stock,
                    'shortage_percentage': ((required_qty - current_stock) / required_qty * 100) if required_qty > 0 else 0
                }
                
                if shortage['shortage_percentage'] > 50:
                    shortage_report['shortages'].append(shortage)
                else:
                    shortage_report['at_risk'].append(shortage)
    
    # Summary
    shortage_report['summary'] = {
        'total_shortages': len(shortage_report['shortages']),
        'total_at_risk': len(shortage_report['at_risk']),
        'critical_items': [s['product_id'] for s in shortage_report['shortages'][:10]]
    }
    
    return shortage_report


def get_inventory_risk_report(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate risk report from analysis results
    """
    risk_report = {
        'timestamp': datetime.now().isoformat(),
        'risk_summary': analysis_results.get('risk_assessment', {}).get('risk_summary', {}),
        'critical_actions': [],
        'risk_mitigation': []
    }
    
    # Extract critical actions
    for rec in analysis_results.get('recommendations', []):
        if rec.get('priority') == 'CRITICAL':
            risk_report['critical_actions'].append(rec)
    
    # Add mitigation strategies
    if risk_report['risk_summary'].get('critical_count', 0) > 0:
        risk_report['risk_mitigation'].append({
            'strategy': 'Emergency procurement',
            'target': 'Critical items',
            'timeline': 'Immediate (1-2 days)'
        })
    
    if risk_report['risk_summary'].get('high_count', 0) > 0:
        risk_report['risk_mitigation'].append({
            'strategy': 'Expedited orders',
            'target': 'High risk items',
            'timeline': '3-5 days'
        })
    
    return risk_report


# Module exports
__all__ = [
    'IntegratedInventoryAnalysis',
    'run_inventory_analysis',
    'get_yarn_shortage_report',
    'get_inventory_risk_report',
    'AnalysisConfig'
]