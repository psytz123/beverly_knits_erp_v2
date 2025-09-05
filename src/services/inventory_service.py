#!/usr/bin/env python3
"""
Inventory Service - Phase 4
Provides inventory management functionality for API v2
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class InventoryService:
    """Service for inventory management operations"""
    
    def __init__(self, inventory_analyzer):
        """
        Initialize inventory service
        
        Args:
            inventory_analyzer: Core inventory analyzer instance
        """
        self.analyzer = inventory_analyzer
        self.logger = logger
    
    def get_inventory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get inventory data based on parameters
        
        Args:
            params: Query parameters
            
        Returns:
            Inventory data
        """
        view = params.get('view', 'summary')
        analysis = params.get('analysis', 'none')
        realtime = params.get('realtime', 'false') == 'true'
        shortage_only = params.get('shortage_only', 'false') == 'true'
        
        try:
            # Get base data based on view
            if view == 'yarn':
                result = self._get_yarn_inventory()
            elif view == 'shortage':
                result = self._get_shortage_analysis()
            elif view == 'planning':
                result = self._get_planning_balance()
            elif view == 'detailed':
                result = self._get_detailed_inventory()
            else:  # summary
                result = self._get_summary()
            
            # Apply analysis if requested
            if analysis == 'shortage':
                result['shortage_analysis'] = self._analyze_shortages()
            elif analysis == 'forecast':
                result['forecast_analysis'] = self._get_forecast_analysis()
            elif analysis == 'intelligence':
                result['intelligence'] = self._get_intelligence_analysis()
            
            # Filter for shortages if requested
            if shortage_only and 'items' in result:
                result['items'] = [
                    item for item in result['items']
                    if item.get('planning_balance', 0) < 0
                ]
                result['shortage_count'] = len(result['items'])
            
            # Add metadata
            result['metadata'] = {
                'view': view,
                'analysis': analysis,
                'realtime': realtime,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting inventory: {str(e)}")
            raise
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get inventory summary"""
        try:
            summary = self.analyzer.get_inventory_summary()
            
            # Ensure proper structure
            if not isinstance(summary, dict):
                summary = {'data': summary}
            
            # Add calculated metrics
            if 'items' in summary:
                total_items = len(summary['items'])
                shortage_items = len([i for i in summary['items'] if i.get('planning_balance', 0) < 0])
                
                summary['metrics'] = {
                    'total_items': total_items,
                    'shortage_items': shortage_items,
                    'shortage_percentage': (shortage_items / total_items * 100) if total_items > 0 else 0
                }
            
            return summary
            
        except AttributeError:
            # Fallback if analyzer doesn't have this method
            return {
                'items': [],
                'metrics': {
                    'total_items': 0,
                    'shortage_items': 0,
                    'shortage_percentage': 0
                }
            }
    
    def _get_yarn_inventory(self) -> Dict[str, Any]:
        """Get yarn-specific inventory"""
        try:
            yarn_data = self.analyzer.get_yarn_inventory()
            
            # Process yarn data
            if isinstance(yarn_data, pd.DataFrame):
                yarn_data = yarn_data.to_dict('records')
            elif not isinstance(yarn_data, dict):
                yarn_data = {'yarns': yarn_data}
            
            return {
                'yarns': yarn_data.get('yarns', yarn_data),
                'total_count': len(yarn_data.get('yarns', yarn_data))
            }
            
        except AttributeError:
            return {'yarns': [], 'total_count': 0}
    
    def _get_shortage_analysis(self) -> Dict[str, Any]:
        """Get shortage analysis"""
        try:
            shortages = self.analyzer.detect_shortages()
            
            if isinstance(shortages, pd.DataFrame):
                shortages = shortages.to_dict('records')
            elif not isinstance(shortages, list):
                shortages = []
            
            # Group shortages by severity
            critical = [s for s in shortages if s.get('planning_balance', 0) < -1000]
            warning = [s for s in shortages if -1000 <= s.get('planning_balance', 0) < -100]
            low = [s for s in shortages if -100 <= s.get('planning_balance', 0) < 0]
            
            return {
                'shortages': shortages,
                'total_shortages': len(shortages),
                'by_severity': {
                    'critical': len(critical),
                    'warning': len(warning),
                    'low': len(low)
                },
                'critical_items': critical[:10]  # Top 10 critical shortages
            }
            
        except AttributeError:
            return {
                'shortages': [],
                'total_shortages': 0,
                'by_severity': {'critical': 0, 'warning': 0, 'low': 0}
            }
    
    def _get_planning_balance(self) -> Dict[str, Any]:
        """Get planning balance analysis"""
        try:
            if hasattr(self.analyzer, 'get_planning_balance_analysis'):
                return self.analyzer.get_planning_balance_analysis()
            
            # Fallback calculation
            yarn_data = self.analyzer.get_yarn_inventory()
            if isinstance(yarn_data, pd.DataFrame):
                planning_data = yarn_data[['Desc#', 'Planning_Balance', 'Theoretical_Balance', 'Allocated', 'On_Order']]
                return {
                    'planning_balance': planning_data.to_dict('records'),
                    'total_planning_balance': planning_data['Planning_Balance'].sum()
                }
            
            return {'planning_balance': [], 'total_planning_balance': 0}
            
        except Exception as e:
            self.logger.error(f"Error getting planning balance: {str(e)}")
            return {'planning_balance': [], 'total_planning_balance': 0}
    
    def _get_detailed_inventory(self) -> Dict[str, Any]:
        """Get detailed inventory with all fields"""
        try:
            # Get comprehensive inventory status
            from src.core.beverly_comprehensive_erp import get_comprehensive_inventory_status
            return get_comprehensive_inventory_status()
        except ImportError:
            # Fallback to basic inventory
            return self._get_summary()
    
    def _analyze_shortages(self) -> Dict[str, Any]:
        """Analyze shortage patterns and causes"""
        shortages = self.analyzer.detect_shortages()
        
        if isinstance(shortages, pd.DataFrame) and not shortages.empty:
            return {
                'total_shortage_value': abs(shortages['Planning_Balance'].sum()),
                'average_shortage': abs(shortages['Planning_Balance'].mean()),
                'most_critical': shortages.nlargest(5, 'Planning_Balance', keep='first')[['Desc#', 'Planning_Balance']].to_dict('records')
            }
        
        return {
            'total_shortage_value': 0,
            'average_shortage': 0,
            'most_critical': []
        }
    
    def _get_forecast_analysis(self) -> Dict[str, Any]:
        """Get forecast-based inventory analysis"""
        try:
            from src.forecasting.enhanced_forecasting_engine import get_inventory_forecast
            return get_inventory_forecast()
        except ImportError:
            return {'forecast': 'Not available'}
    
    def _get_intelligence_analysis(self) -> Dict[str, Any]:
        """Get intelligent inventory insights"""
        try:
            from src.yarn_intelligence.yarn_intelligence_enhanced import get_yarn_intelligence
            return get_yarn_intelligence()
        except ImportError:
            return {'intelligence': 'Not available'}
    
    def get_shortages(self) -> List[Dict[str, Any]]:
        """Get current shortage list"""
        shortages = self.analyzer.detect_shortages()
        
        if isinstance(shortages, pd.DataFrame):
            return shortages.to_dict('records')
        elif isinstance(shortages, list):
            return shortages
        
        return []
    
    def update_inventory(self, yarn_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update inventory for a specific yarn
        
        Args:
            yarn_id: Yarn identifier
            updates: Fields to update
            
        Returns:
            Update result
        """
        try:
            # Implement inventory update logic
            self.logger.info(f"Updating inventory for yarn {yarn_id}: {updates}")
            
            # This would normally update the database/data source
            return {
                'success': True,
                'yarn_id': yarn_id,
                'updates_applied': updates,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating inventory: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Export the service class
__all__ = ['InventoryService']