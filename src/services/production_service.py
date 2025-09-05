#!/usr/bin/env python3
"""
Production Service - Phase 4
Provides production management functionality for API v2
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ProductionService:
    """Service for production management operations"""
    
    def __init__(self, get_planning_func, get_status_func):
        """
        Initialize production service
        
        Args:
            get_planning_func: Function to get production planning
            get_status_func: Function to get production status
        """
        self.get_planning = get_planning_func
        self.get_status = get_status_func
        self.logger = logger
    
    def get_production_data(self, view: str, **kwargs) -> Dict[str, Any]:
        """
        Get production data based on view
        
        Args:
            view: Type of view (status, planning, recommendations, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Production data
        """
        try:
            if view == 'planning':
                result = self._get_planning_data()
            elif view == 'recommendations':
                result = self._get_recommendations()
            elif view == 'machines':
                result = self._get_machine_assignments()
            elif view == 'pipeline':
                result = self._get_production_pipeline()
            else:  # status
                result = self._get_status_data()
            
            # Apply filters if provided
            if 'machine_id' in kwargs and kwargs['machine_id']:
                result = self._filter_by_machine(result, kwargs['machine_id'])
            
            if 'status' in kwargs and kwargs['status']:
                result = self._filter_by_status(result, kwargs['status'])
            
            # Add forecast if requested
            if kwargs.get('include_forecast', False):
                result['forecast'] = self._get_production_forecast()
            
            # Add metadata
            result['metadata'] = {
                'view': view,
                'filters_applied': {
                    k: v for k, v in kwargs.items() 
                    if k in ['machine_id', 'status']
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting production data: {str(e)}")
            raise
    
    def _get_status_data(self) -> Dict[str, Any]:
        """Get production status"""
        try:
            status = self.get_status()
            
            if not isinstance(status, dict):
                status = {'data': status}
            
            # Calculate summary metrics
            if 'orders' in status:
                orders = status['orders']
                total = len(orders)
                assigned = len([o for o in orders if o.get('machine_id')])
                unassigned = total - assigned
                
                status['summary'] = {
                    'total_orders': total,
                    'assigned': assigned,
                    'unassigned': unassigned,
                    'assignment_rate': (assigned / total * 100) if total > 0 else 0
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            return {
                'orders': [],
                'summary': {
                    'total_orders': 0,
                    'assigned': 0,
                    'unassigned': 0,
                    'assignment_rate': 0
                }
            }
    
    def _get_planning_data(self) -> Dict[str, Any]:
        """Get production planning data"""
        try:
            planning = self.get_planning()
            
            if not isinstance(planning, dict):
                planning = {'data': planning}
            
            # Add planning metrics
            if 'schedule' in planning:
                schedule = planning['schedule']
                
                # Calculate capacity utilization
                total_capacity = sum(s.get('capacity', 0) for s in schedule)
                used_capacity = sum(s.get('allocated', 0) for s in schedule)
                
                planning['metrics'] = {
                    'total_capacity': total_capacity,
                    'used_capacity': used_capacity,
                    'utilization_rate': (used_capacity / total_capacity * 100) if total_capacity > 0 else 0
                }
            
            return planning
            
        except Exception as e:
            self.logger.error(f"Error getting planning: {str(e)}")
            return {'schedule': [], 'metrics': {}}
    
    def _get_recommendations(self) -> Dict[str, Any]:
        """Get ML-powered production recommendations"""
        try:
            from src.production.enhanced_production_suggestions_v2 import get_ml_recommendations
            recommendations = get_ml_recommendations()
            
            if not isinstance(recommendations, dict):
                recommendations = {'recommendations': recommendations}
            
            return recommendations
            
        except ImportError:
            self.logger.warning("ML recommendations not available")
            return {'recommendations': [], 'message': 'ML recommendations module not available'}
    
    def _get_machine_assignments(self) -> Dict[str, Any]:
        """Get machine assignment suggestions"""
        try:
            from src.core.beverly_comprehensive_erp import get_machine_assignment_suggestions
            assignments = get_machine_assignment_suggestions()
            
            if not isinstance(assignments, dict):
                assignments = {'assignments': assignments}
            
            # Group by work center
            if 'suggestions' in assignments:
                work_centers = {}
                for suggestion in assignments['suggestions']:
                    wc = suggestion.get('work_center', 'Unknown')
                    if wc not in work_centers:
                        work_centers[wc] = []
                    work_centers[wc].append(suggestion)
                
                assignments['by_work_center'] = work_centers
            
            return assignments
            
        except ImportError:
            return {'assignments': [], 'message': 'Machine assignment module not available'}
    
    def _get_production_pipeline(self) -> Dict[str, Any]:
        """Get production pipeline status"""
        try:
            from src.production.enhanced_production_pipeline import get_production_pipeline
            pipeline = get_production_pipeline()
            
            if not isinstance(pipeline, dict):
                pipeline = {'pipeline': pipeline}
            
            # Add stage metrics
            if 'stages' in pipeline:
                for stage in pipeline['stages']:
                    items = stage.get('items', [])
                    stage['metrics'] = {
                        'count': len(items),
                        'total_quantity': sum(i.get('quantity', 0) for i in items)
                    }
            
            return pipeline
            
        except ImportError:
            return {
                'pipeline': [],
                'stages': [
                    {'name': 'G00', 'items': [], 'metrics': {'count': 0}},
                    {'name': 'G02', 'items': [], 'metrics': {'count': 0}},
                    {'name': 'I01', 'items': [], 'metrics': {'count': 0}},
                    {'name': 'F01', 'items': [], 'metrics': {'count': 0}}
                ]
            }
    
    def _get_production_forecast(self) -> Dict[str, Any]:
        """Get production forecast"""
        try:
            from src.forecasting.enhanced_forecasting_engine import get_production_forecast
            return get_production_forecast()
        except ImportError:
            return {'forecast': 'Not available'}
    
    def _filter_by_machine(self, data: Dict[str, Any], machine_id: str) -> Dict[str, Any]:
        """Filter production data by machine ID"""
        if 'orders' in data:
            data['orders'] = [
                o for o in data['orders']
                if str(o.get('machine_id', '')) == str(machine_id)
            ]
            
            # Update summary
            if 'summary' in data:
                total = len(data['orders'])
                assigned = len([o for o in data['orders'] if o.get('machine_id')])
                data['summary']['total_orders'] = total
                data['summary']['assigned'] = assigned
                data['summary']['unassigned'] = total - assigned
        
        return data
    
    def _filter_by_status(self, data: Dict[str, Any], status: str) -> Dict[str, Any]:
        """Filter production data by status"""
        if 'orders' in data:
            if status == 'assigned':
                data['orders'] = [o for o in data['orders'] if o.get('machine_id')]
            elif status == 'unassigned':
                data['orders'] = [o for o in data['orders'] if not o.get('machine_id')]
            elif status == 'completed':
                data['orders'] = [o for o in data['orders'] if o.get('status') == 'completed']
            
            # Update summary
            if 'summary' in data:
                total = len(data['orders'])
                assigned = len([o for o in data['orders'] if o.get('machine_id')])
                data['summary']['total_orders'] = total
                data['summary']['assigned'] = assigned
                data['summary']['unassigned'] = total - assigned
        
        return data
    
    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new production order
        
        Args:
            order_data: Order details
            
        Returns:
            Created order
        """
        try:
            # Validate required fields
            required_fields = ['style_id', 'quantity', 'deadline']
            missing = [f for f in required_fields if f not in order_data]
            
            if missing:
                return {
                    'success': False,
                    'error': f"Missing required fields: {', '.join(missing)}"
                }
            
            # Generate order ID
            import uuid
            order_id = f"PO-{uuid.uuid4().hex[:8].upper()}"
            
            # Create order
            order = {
                'id': order_id,
                'status': 'pending',
                'created_at': pd.Timestamp.now().isoformat(),
                **order_data
            }
            
            # This would normally save to database
            self.logger.info(f"Created production order: {order_id}")
            
            return {
                'success': True,
                'order': order
            }
            
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing production order
        
        Args:
            order_id: Order identifier
            updates: Fields to update
            
        Returns:
            Update result
        """
        try:
            # This would normally update the database
            self.logger.info(f"Updating order {order_id}: {updates}")
            
            return {
                'success': True,
                'order_id': order_id,
                'updates_applied': updates,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating order: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Export the service class
__all__ = ['ProductionService']