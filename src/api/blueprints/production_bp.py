"""
Production Blueprint - Consolidates all production-related API endpoints
Uses existing service modules for business logic
"""
from flask import Blueprint, jsonify, request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Create the blueprint
production_bp = Blueprint('production', __name__)

# Global handler (will be initialized by main app)
handler = None


class ProductionAPIHandler:
    """Handler class for production-related operations"""
    
    def __init__(self, capacity_service=None, pipeline_service=None, 
                 planning_engine=None, data_loader=None):
        self.capacity = capacity_service
        self.pipeline = pipeline_service
        self.planning = planning_engine
        self.data_loader = data_loader
    
    def get_production_data(self):
        """Get current production data"""
        if self.data_loader:
            try:
                return self.data_loader.load_knit_orders()
            except Exception as e:
                logger.error(f"Error loading production data: {e}")
        return None


def init_blueprint(capacity_service, pipeline_service, planning_engine, data_loader):
    """Initialize the blueprint with required services"""
    global handler
    handler = ProductionAPIHandler(
        capacity_service=capacity_service,
        pipeline_service=pipeline_service,
        planning_engine=planning_engine,
        data_loader=data_loader
    )


# --- Production Planning Endpoints ---

@production_bp.route("/production-planning")
def production_planning():
    """Production planning with forecasting integration"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get production data
        production_data = handler.get_production_data()
        if production_data is None:
            return jsonify({'error': 'No production data available'}), 404
        
        # Get parameters
        view = request.args.get('view', 'summary')
        forecast = request.args.get('forecast', 'false').lower() == 'true'
        
        # Basic production planning response
        planning_data = {
            'total_orders': len(production_data) if hasattr(production_data, '__len__') else 0,
            'view': view,
            'forecast_enabled': forecast,
            'orders': []
        }
        
        # Add order details if available
        if hasattr(production_data, 'iterrows'):
            for idx, row in production_data.head(10).iterrows():
                planning_data['orders'].append({
                    'order_id': row.get('KO#', idx),
                    'style': row.get('Style#', ''),
                    'quantity': row.get('Order Qty', 0),
                    'status': row.get('Status', 'Active')
                })
        
        # If capacity service available, add capacity analysis
        if handler.capacity and hasattr(handler.capacity, 'calculate_capacity_utilization'):
            capacity_data = handler.capacity.calculate_capacity_utilization(production_data)
            planning_data['capacity'] = capacity_data
        
        return jsonify(planning_data)
    
    except Exception as e:
        logger.error(f"Error in production planning: {e}")
        return jsonify({'error': str(e)}), 500


@production_bp.route("/production-pipeline")
def production_pipeline():
    """Real-time production pipeline status"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        production_data = handler.get_production_data()
        if production_data is None:
            return jsonify({'error': 'No production data available'}), 404
        
        # Pipeline stages
        pipeline_status = {
            'stages': {
                'G00': {'name': 'Greige Stage 1', 'count': 0, 'items': []},
                'G02': {'name': 'Greige Stage 2', 'count': 0, 'items': []},
                'I01': {'name': 'QC Inspection', 'count': 0, 'items': []},
                'F01': {'name': 'Finished Goods', 'count': 0, 'items': []}
            },
            'total_in_pipeline': 0,
            'bottlenecks': []
        }
        
        # Count items in each stage
        if hasattr(production_data, 'iterrows'):
            for _, row in production_data.iterrows():
                stage = row.get('Stage', 'G00')
                if stage in pipeline_status['stages']:
                    pipeline_status['stages'][stage]['count'] += 1
                    pipeline_status['total_in_pipeline'] += 1
        
        # Identify bottlenecks using capacity service
        if handler.capacity and hasattr(handler.capacity, 'identify_capacity_bottlenecks'):
            bottlenecks = handler.capacity.identify_capacity_bottlenecks(production_data)
            pipeline_status['bottlenecks'] = bottlenecks
        
        return jsonify(pipeline_status)
    
    except Exception as e:
        logger.error(f"Error in production pipeline: {e}")
        return jsonify({'error': str(e)}), 500


@production_bp.route("/production-suggestions")
def production_suggestions():
    """AI-powered production suggestions"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get all required data
        production_data = handler.get_production_data()
        yarn_data = handler.data_loader.load_yarn_inventory() if handler.data_loader else None
        
        suggestions = {
            'optimization_opportunities': [],
            'risk_mitigation': [],
            'efficiency_improvements': []
        }
        
        # Use pipeline service for complete analysis
        if handler.pipeline and hasattr(handler.pipeline, 'run_complete_analysis'):
            analysis = handler.pipeline.run_complete_analysis(
                inventory_data=yarn_data,
                yarn_data=yarn_data
            )
            
            # Generate suggestions based on analysis
            if 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    if rec.get('type') == 'URGENT':
                        suggestions['risk_mitigation'].append({
                            'priority': 'HIGH',
                            'action': rec.get('action', ''),
                            'reason': rec.get('message', '')
                        })
                    else:
                        suggestions['optimization_opportunities'].append({
                            'priority': 'MEDIUM',
                            'action': rec.get('action', ''),
                            'reason': rec.get('message', '')
                        })
        
        # Add capacity-based suggestions
        if handler.capacity and production_data is not None:
            capacity_metrics = handler.capacity.get_capacity_metrics(production_data)
            if capacity_metrics:
                utilization = capacity_metrics.get('overall_utilization', 0)
                if utilization > 0.9:
                    suggestions['efficiency_improvements'].append({
                        'priority': 'HIGH',
                        'action': 'Consider adding production capacity',
                        'reason': f'Capacity utilization at {utilization*100:.1f}%'
                    })
                elif utilization < 0.5:
                    suggestions['efficiency_improvements'].append({
                        'priority': 'MEDIUM',
                        'action': 'Optimize production scheduling',
                        'reason': f'Capacity utilization only {utilization*100:.1f}%'
                    })
        
        return jsonify(suggestions)
    
    except Exception as e:
        logger.error(f"Error generating production suggestions: {e}")
        return jsonify({'error': str(e)}), 500


@production_bp.route("/po-risk-analysis")
def po_risk_analysis():
    """Purchase order risk analysis"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        production_data = handler.get_production_data()
        yarn_data = handler.data_loader.load_yarn_inventory() if handler.data_loader else None
        
        risk_analysis = {
            'high_risk_orders': [],
            'medium_risk_orders': [],
            'low_risk_orders': [],
            'risk_factors': {
                'material_shortage': [],
                'capacity_constraints': [],
                'timeline_risks': []
            }
        }
        
        if production_data is not None and hasattr(production_data, 'iterrows'):
            for _, order in production_data.iterrows():
                order_id = order.get('KO#', 'Unknown')
                risk_score = 0
                risk_factors = []
                
                # Check material availability
                if yarn_data is not None:
                    # Simple check - in reality would check BOM requirements
                    if len(yarn_data[yarn_data['Planning Balance'] < 0]) > 0:
                        risk_score += 30
                        risk_factors.append('Material shortage risk')
                
                # Check timeline
                import pandas as pd
                if 'Delivery Date' in order:
                    try:
                        delivery_date = pd.to_datetime(order['Delivery Date'])
                        days_until = (delivery_date - pd.Timestamp.now()).days
                        if days_until < 7:
                            risk_score += 40
                            risk_factors.append('Tight deadline')
                        elif days_until < 14:
                            risk_score += 20
                            risk_factors.append('Approaching deadline')
                    except:
                        pass
                
                # Categorize risk
                risk_item = {
                    'order_id': order_id,
                    'style': order.get('Style#', ''),
                    'quantity': order.get('Order Qty', 0),
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                }
                
                if risk_score >= 50:
                    risk_analysis['high_risk_orders'].append(risk_item)
                elif risk_score >= 20:
                    risk_analysis['medium_risk_orders'].append(risk_item)
                else:
                    risk_analysis['low_risk_orders'].append(risk_item)
        
        # Summary
        risk_analysis['summary'] = {
            'total_orders': len(production_data) if production_data is not None else 0,
            'high_risk_count': len(risk_analysis['high_risk_orders']),
            'medium_risk_count': len(risk_analysis['medium_risk_orders']),
            'low_risk_count': len(risk_analysis['low_risk_orders'])
        }
        
        return jsonify(risk_analysis)
    
    except Exception as e:
        logger.error(f"Error in PO risk analysis: {e}")
        return jsonify({'error': str(e)}), 500


@production_bp.route("/fabric/yarn-requirements", methods=['POST'])
def fabric_yarn_requirements():
    """Calculate yarn requirements from fabric specifications"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get fabric specifications from request
        fabric_specs = request.get_json() or {}
        
        # Default values if not provided
        fabric_weight = fabric_specs.get('weight', 200)  # gsm
        fabric_width = fabric_specs.get('width', 60)  # inches
        fabric_length = fabric_specs.get('length', 1000)  # yards
        waste_factor = fabric_specs.get('waste_factor', 1.1)  # 10% waste
        
        # Calculate yarn requirements
        # Simplified calculation - in reality would use more complex formulas
        total_area = fabric_width * fabric_length * 0.0254  # Convert to square meters
        fabric_weight_kg = (fabric_weight * total_area) / 1000
        yarn_required = fabric_weight_kg * waste_factor
        
        # Convert to pounds
        yarn_required_lbs = yarn_required * 2.20462
        
        requirements = {
            'fabric_specs': {
                'weight_gsm': fabric_weight,
                'width_inches': fabric_width,
                'length_yards': fabric_length
            },
            'yarn_requirements': {
                'total_kg': round(yarn_required, 2),
                'total_lbs': round(yarn_required_lbs, 2),
                'with_waste_factor': waste_factor
            },
            'breakdown': {
                'base_requirement_kg': round(fabric_weight_kg, 2),
                'waste_kg': round(yarn_required - fabric_weight_kg, 2)
            }
        }
        
        return jsonify(requirements)
    
    except Exception as e:
        logger.error(f"Error calculating yarn requirements: {e}")
        return jsonify({'error': str(e)}), 500


@production_bp.route("/planning-status")
def planning_status():
    """Get current planning status"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        status = {
            'planning_engine': 'Available' if handler.planning else 'Not Available',
            'capacity_service': 'Available' if handler.capacity else 'Not Available',
            'pipeline_service': 'Available' if handler.pipeline else 'Not Available',
            'data_loader': 'Available' if handler.data_loader else 'Not Available',
            'status': 'Ready'
        }
        
        # Get current metrics if services available
        if handler.capacity and handler.get_production_data():
            production_data = handler.get_production_data()
            metrics = handler.capacity.get_capacity_metrics(production_data)
            status['current_metrics'] = metrics
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting planning status: {e}")
        return jsonify({'error': str(e)}), 500