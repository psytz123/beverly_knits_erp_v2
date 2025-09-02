"""
Inventory Blueprint - Consolidates all inventory-related API endpoints
Extracted from beverly_comprehensive_erp.py for better organization
"""
from flask import Blueprint, jsonify, request
from services.inventory_analyzer_core import InventoryAnalyzer
from services.inventory_pipeline_core import InventoryManagementPipeline
import logging

logger = logging.getLogger(__name__)

# Create the blueprint
inventory_bp = Blueprint('inventory', __name__)


class InventoryAPIHandler:
    """Handler class for inventory-related operations"""
    
    def __init__(self, analyzer=None, pipeline=None, data_loader=None):
        self.analyzer = analyzer or InventoryAnalyzer()
        self.pipeline = pipeline or InventoryManagementPipeline()
        self.data_loader = data_loader
        self._cache = {}
    
    def get_inventory_data(self):
        """Get current inventory data from data loader or cache"""
        if self.data_loader:
            try:
                return self.data_loader.load_yarn_inventory()
            except Exception as e:
                logger.error(f"Error loading inventory data: {e}")
        return None
    
    def get_sales_data(self):
        """Get sales data from data loader or cache"""
        if self.data_loader:
            try:
                return self.data_loader.load_sales_data()
            except Exception as e:
                logger.error(f"Error loading sales data: {e}")
        return None


# Initialize handler (will be set by main app)
handler = None


def init_blueprint(analyzer_instance, pipeline_instance, data_loader_instance):
    """Initialize the blueprint with required instances"""
    global handler
    handler = InventoryAPIHandler(analyzer_instance, pipeline_instance, data_loader_instance)


# --- Core Inventory Endpoints ---

@inventory_bp.route("/inventory-intelligence-enhanced")
def inventory_intelligence_enhanced():
    """Enhanced inventory intelligence with multiple views"""
    try:
        view = request.args.get('view', 'summary')
        realtime = request.args.get('realtime', 'false').lower() == 'true'
        analysis_type = request.args.get('analysis', 'all')
        
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get inventory data
        inventory_data = handler.get_inventory_data()
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Perform analysis
        analysis_results = handler.analyzer.analyze_inventory(inventory_data)
        
        # Format response based on view
        response = {
            'status': 'success',
            'view': view,
            'realtime': realtime,
            'analysis_type': analysis_type,
            'data': analysis_results
        }
        
        # Add additional analysis if requested
        if analysis_type == 'shortage':
            critical_items = [
                item for item in analysis_results.get('critical_items', [])
                if item.get('Planning Balance', 0) < 0
            ]
            response['shortage_analysis'] = {
                'critical_count': len(critical_items),
                'items': critical_items[:10]
            }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in inventory intelligence: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-analysis")
def inventory_analysis():
    """Basic inventory analysis endpoint"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        inventory_data = handler.get_inventory_data()
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        analysis = handler.analyzer.analyze_inventory(inventory_data)
        return jsonify(analysis)
    
    except Exception as e:
        logger.error(f"Error in inventory analysis: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-analysis/complete")
def inventory_analysis_complete():
    """Complete inventory analysis with all components"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get all required data
        inventory_data = handler.get_inventory_data()
        sales_data = handler.get_sales_data()
        
        # Run complete pipeline analysis
        results = handler.pipeline.run_complete_analysis(
            sales_data=sales_data,
            inventory_data=inventory_data,
            yarn_data=inventory_data  # Using same data for yarn
        )
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in complete analysis: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-analysis/yarn-shortages")
def yarn_shortages():
    """Analyze yarn shortages"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        inventory_data = handler.get_inventory_data()
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Find items with negative planning balance
        shortages = []
        if hasattr(inventory_data, 'iterrows'):
            for _, row in inventory_data.iterrows():
                if row.get('Planning Balance', 0) < 0:
                    shortages.append({
                        'yarn_id': row.get('Desc#', 'Unknown'),
                        'description': row.get('Description', ''),
                        'shortage_amount': abs(row.get('Planning Balance', 0)),
                        'on_order': row.get('On_Order', 0),
                        'allocated': row.get('Allocated', 0)
                    })
        
        # Sort by shortage amount
        shortages.sort(key=lambda x: x['shortage_amount'], reverse=True)
        
        return jsonify({
            'total_shortages': len(shortages),
            'shortages': shortages[:20],  # Top 20 shortages
            'summary': {
                'critical': len([s for s in shortages if s['shortage_amount'] > 1000]),
                'high': len([s for s in shortages if 500 < s['shortage_amount'] <= 1000]),
                'medium': len([s for s in shortages if 100 < s['shortage_amount'] <= 500]),
                'low': len([s for s in shortages if s['shortage_amount'] <= 100])
            }
        })
    
    except Exception as e:
        logger.error(f"Error analyzing yarn shortages: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-analysis/stock-risks")
def stock_risks():
    """Analyze stock risk levels"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        inventory_data = handler.get_inventory_data()
        sales_data = handler.get_sales_data()
        
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Generate forecast
        forecast = {}
        if sales_data is not None and hasattr(sales_data, 'iterrows'):
            for _, row in sales_data.iterrows():
                item_id = str(row.get('Style#', ''))
                forecast[item_id] = row.get('Total_Sold', 0)
        
        # Prepare inventory for analysis
        current_inventory = []
        if hasattr(inventory_data, 'iterrows'):
            for _, row in inventory_data.iterrows():
                current_inventory.append({
                    'id': row.get('Desc#', ''),
                    'quantity': row.get('Planning Balance', 0)
                })
        
        # Analyze risk levels
        analysis = handler.analyzer.analyze_inventory_levels(current_inventory, forecast)
        
        # Group by risk level
        risk_summary = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        for item in analysis:
            risk_level = item.get('shortage_risk', 'LOW')
            risk_summary[risk_level].append(item)
        
        return jsonify({
            'risk_summary': {
                level: len(items) for level, items in risk_summary.items()
            },
            'critical_items': risk_summary['CRITICAL'][:10],
            'high_risk_items': risk_summary['HIGH'][:10]
        })
    
    except Exception as e:
        logger.error(f"Error analyzing stock risks: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-netting")
def inventory_netting():
    """Multi-level inventory netting calculations"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        inventory_data = handler.get_inventory_data()
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Calculate netting at different levels
        netting_results = {
            'yarn_level': {},
            'style_level': {},
            'order_level': {},
            'summary': {}
        }
        
        total_inventory = 0
        total_allocated = 0
        total_on_order = 0
        
        if hasattr(inventory_data, 'iterrows'):
            for _, row in inventory_data.iterrows():
                yarn_id = row.get('Desc#', 'Unknown')
                planning_balance = row.get('Planning Balance', 0)
                allocated = row.get('Allocated', 0)
                on_order = row.get('On_Order', 0)
                
                netting_results['yarn_level'][yarn_id] = {
                    'planning_balance': planning_balance,
                    'allocated': allocated,
                    'on_order': on_order,
                    'net_available': planning_balance - allocated + on_order
                }
                
                total_inventory += planning_balance
                total_allocated += abs(allocated)
                total_on_order += on_order
        
        netting_results['summary'] = {
            'total_inventory_value': total_inventory,
            'total_allocated': total_allocated,
            'total_on_order': total_on_order,
            'net_position': total_inventory - total_allocated + total_on_order
        }
        
        return jsonify(netting_results)
    
    except Exception as e:
        logger.error(f"Error in inventory netting: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/real-time-inventory")
def real_time_inventory():
    """Get real-time inventory status"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Force reload for real-time data
        if handler.data_loader and hasattr(handler.data_loader, 'clear_cache'):
            handler.data_loader.clear_cache()
        
        inventory_data = handler.get_inventory_data()
        if inventory_data is None:
            return jsonify({'error': 'No inventory data available'}), 404
        
        # Get current inventory status
        status = {
            'last_updated': 'Now',
            'total_items': len(inventory_data) if hasattr(inventory_data, '__len__') else 0,
            'critical_items': 0,
            'healthy_items': 0,
            'items': []
        }
        
        if hasattr(inventory_data, 'iterrows'):
            for _, row in inventory_data.iterrows():
                balance = row.get('Planning Balance', 0)
                if balance < 0:
                    status['critical_items'] += 1
                elif balance > 100:
                    status['healthy_items'] += 1
                
                # Add first 10 items
                if len(status['items']) < 10:
                    status['items'].append({
                        'id': row.get('Desc#', ''),
                        'description': row.get('Description', ''),
                        'balance': balance,
                        'status': 'Critical' if balance < 0 else 'Healthy' if balance > 100 else 'Warning'
                    })
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting real-time inventory: {e}")
        return jsonify({'error': str(e)}), 500


@inventory_bp.route("/inventory-overview")
def inventory_overview():
    """Get inventory overview - redirects to enhanced endpoint"""
    # This is a deprecated endpoint that redirects
    return inventory_intelligence_enhanced()