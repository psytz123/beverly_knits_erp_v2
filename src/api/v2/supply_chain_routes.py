"""
Supply Chain API Routes v2
Handles end-to-end supply chain and MRP endpoints
"""

from flask import Blueprint, jsonify, request
from src.services.service_container import get_service
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
supply_chain_bp = Blueprint('supply_chain_v2', __name__)


@supply_chain_bp.route('/supply-chain/analysis', methods=['GET'])
def get_supply_chain_analysis():
    """Get comprehensive supply chain analysis"""
    try:
        supply_chain_service = get_service('supply_chain')
        
        # Perform analysis
        analysis = supply_chain_service.analyze_supply_chain()
        
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error in supply chain analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/supply-chain/optimize', methods=['POST'])
def optimize_supply_chain():
    """Optimize supply chain based on goals"""
    try:
        supply_chain_service = get_service('supply_chain')
        
        data = request.get_json() or {}
        
        # Get optimization goals
        goals = data.get('goals', {
            'minimize_cost': True,
            'maximize_service_level': True,
            'reduce_lead_time': True,
            'balance_inventory': True
        })
        
        # Run optimization
        optimization_result = supply_chain_service.optimize_supply_chain(goals)
        
        return jsonify({
            'status': 'success',
            'optimization': optimization_result
        })
        
    except Exception as e:
        logger.error(f"Error optimizing supply chain: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/supply-chain/kpis', methods=['GET'])
def get_supply_chain_kpis():
    """Get supply chain KPIs"""
    try:
        supply_chain_service = get_service('supply_chain')
        
        # Get comprehensive KPIs
        kpis = supply_chain_service._calculate_supply_chain_kpis()
        
        # Add trend data if available
        cache = get_service('cache')
        historical_kpis = cache.get('supply_chain:kpis:history')
        
        if historical_kpis:
            kpis['trends'] = historical_kpis
        
        return jsonify({
            'status': 'success',
            'kpis': kpis
        })
        
    except Exception as e:
        logger.error(f"Error getting KPIs: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/mrp/requirements', methods=['POST'])
def calculate_mrp_requirements():
    """Calculate time-phased MRP requirements"""
    try:
        mrp_service = get_service('mrp')
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No demand forecast provided'
            }), 400
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            if 'demand' in data:
                demand_df = pd.DataFrame(data['demand'])
            else:
                demand_df = pd.DataFrame([data])
        else:
            demand_df = pd.DataFrame(data)
        
        # Get parameters
        horizon = int(request.args.get('horizon', 90))
        include_safety_stock = request.args.get('safety_stock', 'true').lower() == 'true'
        
        # Calculate requirements
        mrp_result = mrp_service.calculate_requirements(
            demand_forecast=demand_df,
            planning_horizon_days=horizon,
            include_safety_stock=include_safety_stock
        )
        
        # Convert MRP table to serializable format
        if 'mrp_table' in mrp_result and isinstance(mrp_result['mrp_table'], pd.DataFrame):
            mrp_result['mrp_table'] = mrp_result['mrp_table'].reset_index().to_dict('records')
        
        return jsonify({
            'status': 'success',
            'mrp': mrp_result
        })
        
    except Exception as e:
        logger.error(f"Error calculating MRP: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/mrp/pegging/<material>', methods=['GET'])
def get_pegging_analysis(material):
    """Get pegging analysis for a material"""
    try:
        mrp_service = get_service('mrp')
        
        # Perform pegging analysis
        pegging_result = mrp_service.perform_pegging_analysis(material)
        
        return jsonify({
            'status': 'success',
            'pegging': pegging_result
        })
        
    except Exception as e:
        logger.error(f"Error in pegging analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/mrp/capacity-requirements', methods=['POST'])
def calculate_capacity_requirements():
    """Calculate capacity requirements from production plan"""
    try:
        mrp_service = get_service('mrp')
        
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No production plan provided'
            }), 400
        
        # Convert to DataFrame
        if isinstance(data, list):
            production_plan = pd.DataFrame(data)
        else:
            production_plan = pd.DataFrame([data])
        
        # Calculate capacity requirements
        capacity_req = mrp_service.calculate_capacity_requirements(production_plan)
        
        return jsonify({
            'status': 'success',
            'capacity_requirements': capacity_req
        })
        
    except Exception as e:
        logger.error(f"Error calculating capacity requirements: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/supply-chain/risks', methods=['GET'])
def get_supply_chain_risks():
    """Get supply chain risk assessment"""
    try:
        supply_chain_service = get_service('supply_chain')
        
        # Identify risks
        risks = supply_chain_service._identify_supply_chain_risks()
        
        # Categorize by severity
        risk_summary = {
            'high': [r for r in risks if r.get('severity') == 'high'],
            'medium': [r for r in risks if r.get('severity') == 'medium'],
            'low': [r for r in risks if r.get('severity') == 'low']
        }
        
        return jsonify({
            'status': 'success',
            'risks': risks,
            'summary': risk_summary,
            'total_risks': len(risks)
        })
        
    except Exception as e:
        logger.error(f"Error identifying risks: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/supply-chain/recommendations', methods=['GET'])
def get_supply_chain_recommendations():
    """Get supply chain improvement recommendations"""
    try:
        supply_chain_service = get_service('supply_chain')
        
        # Generate recommendations
        recommendations = supply_chain_service._generate_recommendations()
        
        # Sort by priority
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
        low_priority = [r for r in recommendations if r.get('priority') == 'low']
        
        return jsonify({
            'status': 'success',
            'recommendations': {
                'high_priority': high_priority,
                'medium_priority': medium_priority,
                'low_priority': low_priority
            },
            'total': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@supply_chain_bp.route('/po-risk-analysis', methods=['GET'])
def get_po_risk_analysis():
    """Get purchase order risk analysis"""
    try:
        supply_chain_service = get_service('supply_chain')
        mrp_service = get_service('mrp')
        
        # Get current POs and assess risks
        risk_analysis = {
            'at_risk_orders': [],
            'delayed_orders': [],
            'urgent_orders': [],
            'risk_mitigation': []
        }
        
        # Get MRP data for critical materials
        demand_df = pd.DataFrame()  # Would get from actual demand
        mrp_result = mrp_service.calculate_requirements(demand_df)
        
        if 'purchase_orders' in mrp_result:
            for po in mrp_result['purchase_orders']:
                if po.get('priority', 0) >= 4:
                    risk_analysis['urgent_orders'].append(po)
        
        # Add mitigation strategies
        if risk_analysis['urgent_orders']:
            risk_analysis['risk_mitigation'].append({
                'risk': 'Urgent material requirements',
                'action': 'Expedite purchase orders',
                'impact': 'Prevent production delays'
            })
        
        return jsonify({
            'status': 'success',
            'risk_analysis': risk_analysis
        })
        
    except Exception as e:
        logger.error(f"Error in PO risk analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500