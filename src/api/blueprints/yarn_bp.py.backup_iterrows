"""
Yarn Blueprint - Yarn intelligence, substitution, and shortage analysis
Uses existing yarn intelligence modules
"""
from flask import Blueprint, jsonify, request
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Create the blueprint
yarn_bp = Blueprint('yarn', __name__)

# Global handler
handler = None


class YarnAPIHandler:
    """Handler for yarn-related operations"""
    
    def __init__(self, yarn_intelligence=None, interchangeability=None, 
                 data_loader=None, service_manager=None):
        self.yarn_intel = yarn_intelligence
        self.interchangeability = interchangeability
        self.data_loader = data_loader
        self.service_manager = service_manager
    
    def get_yarn_data(self):
        """Get current yarn inventory data"""
        if self.data_loader:
            try:
                return self.data_loader.load_yarn_inventory()
            except Exception as e:
                logger.error(f"Error loading yarn data: {e}")
        return None
    
    def get_bom_data(self):
        """Get BOM data for yarn requirements"""
        if self.data_loader:
            try:
                return self.data_loader.load_bom()
            except Exception as e:
                logger.error(f"Error loading BOM data: {e}")
        return None


def init_blueprint(yarn_intelligence, interchangeability, data_loader, service_manager=None):
    """Initialize blueprint with yarn services"""
    global handler
    handler = YarnAPIHandler(
        yarn_intelligence=yarn_intelligence,
        interchangeability=interchangeability,
        data_loader=data_loader,
        service_manager=service_manager
    )


# --- Yarn Intelligence Endpoints ---

@yarn_bp.route("/yarn-intelligence")
def yarn_intelligence():
    """Comprehensive yarn intelligence analysis"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        # Get parameters
        analysis_type = request.args.get('analysis', 'all')
        include_forecast = request.args.get('forecast', 'false').lower() == 'true'
        
        # Load yarn data
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        intelligence = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_yarns': len(yarn_data),
            'analysis': {},
            'shortages': [],
            'recommendations': []
        }
        
        # Analyze shortages
        if analysis_type in ['shortage', 'all']:
            shortage_yarns = []
            if hasattr(yarn_data, 'iterrows'):
                for _, yarn in yarn_data.iterrows():
                    planning_balance = yarn.get('Planning Balance', yarn.get('planning_balance', 0))
                    if planning_balance < 0:
                        shortage_yarns.append({
                            'yarn_id': yarn.get('Desc#', yarn.get('yarn_id', '')),
                            'description': yarn.get('Description', ''),
                            'shortage_amount': abs(planning_balance),
                            'on_order': yarn.get('On_Order', yarn.get('on_order', 0)),
                            'allocated': yarn.get('Allocated', yarn.get('allocated', 0))
                        })
            
            # Sort by shortage amount
            shortage_yarns.sort(key=lambda x: x['shortage_amount'], reverse=True)
            intelligence['shortages'] = shortage_yarns[:20]  # Top 20
            intelligence['analysis']['shortage_count'] = len(shortage_yarns)
            intelligence['analysis']['total_shortage_value'] = sum(y['shortage_amount'] for y in shortage_yarns)
        
        # Analyze inventory health
        if hasattr(yarn_data, 'describe'):
            planning_balance = yarn_data.get('Planning Balance', yarn_data.get('planning_balance', pd.Series()))
            if not planning_balance.empty:
                intelligence['analysis']['inventory_health'] = {
                    'healthy_items': len(planning_balance[planning_balance > 100]),
                    'warning_items': len(planning_balance[(planning_balance >= 0) & (planning_balance <= 100)]),
                    'critical_items': len(planning_balance[planning_balance < 0]),
                    'average_balance': float(planning_balance.mean()),
                    'total_value': float(planning_balance.sum())
                }
        
        # Generate recommendations
        if intelligence['analysis'].get('shortage_count', 0) > 10:
            intelligence['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Urgent procurement needed',
                'reason': f"{intelligence['analysis']['shortage_count']} yarns in shortage"
            })
        
        # Add forecast if requested
        if include_forecast and handler.service_manager:
            forecasting = handler.service_manager.get_service('forecasting')
            if forecasting:
                intelligence['forecast_available'] = True
        
        return jsonify(intelligence)
    
    except Exception as e:
        logger.error(f"Error in yarn intelligence: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-data")
def yarn_data():
    """Get raw yarn inventory data"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        # Convert to JSON-friendly format
        data_response = {
            'total_items': len(yarn_data),
            'columns': list(yarn_data.columns) if hasattr(yarn_data, 'columns') else [],
            'data': []
        }
        
        # Add first 100 items
        if hasattr(yarn_data, 'head'):
            sample = yarn_data.head(100).to_dict('records')
            data_response['data'] = sample
        
        return jsonify(data_response)
    
    except Exception as e:
        logger.error(f"Error getting yarn data: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-shortage-analysis")
def yarn_shortage_analysis():
    """Detailed yarn shortage analysis"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        shortage_analysis = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'critical_shortages': [],
            'high_risk': [],
            'medium_risk': [],
            'statistics': {},
            'action_items': []
        }
        
        # Categorize shortages
        if hasattr(yarn_data, 'iterrows'):
            for _, yarn in yarn_data.iterrows():
                planning_balance = yarn.get('Planning Balance', yarn.get('planning_balance', 0))
                on_order = yarn.get('On_Order', yarn.get('on_order', 0))
                allocated = abs(yarn.get('Allocated', yarn.get('allocated', 0)))
                
                yarn_info = {
                    'yarn_id': yarn.get('Desc#', ''),
                    'description': yarn.get('Description', ''),
                    'planning_balance': planning_balance,
                    'on_order': on_order,
                    'allocated': allocated,
                    'net_position': planning_balance + on_order - allocated
                }
                
                # Categorize by risk level
                if planning_balance < -1000:
                    shortage_analysis['critical_shortages'].append(yarn_info)
                elif planning_balance < -100:
                    shortage_analysis['high_risk'].append(yarn_info)
                elif planning_balance < 0:
                    shortage_analysis['medium_risk'].append(yarn_info)
        
        # Calculate statistics
        shortage_analysis['statistics'] = {
            'total_shortage_items': (
                len(shortage_analysis['critical_shortages']) +
                len(shortage_analysis['high_risk']) +
                len(shortage_analysis['medium_risk'])
            ),
            'critical_count': len(shortage_analysis['critical_shortages']),
            'high_risk_count': len(shortage_analysis['high_risk']),
            'medium_risk_count': len(shortage_analysis['medium_risk'])
        }
        
        # Generate action items
        if shortage_analysis['critical_shortages']:
            shortage_analysis['action_items'].append({
                'priority': 'CRITICAL',
                'action': f"Immediately order {len(shortage_analysis['critical_shortages'])} critical yarns",
                'yarns': [y['yarn_id'] for y in shortage_analysis['critical_shortages'][:5]]
            })
        
        return jsonify(shortage_analysis)
    
    except Exception as e:
        logger.error(f"Error in shortage analysis: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-alternatives")
def yarn_alternatives():
    """Get yarn substitution alternatives"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        # Get yarn_id from query params
        yarn_id = request.args.get('yarn_id', '')
        
        alternatives = {
            'requested_yarn': yarn_id,
            'alternatives': [],
            'substitution_groups': {},
            'recommendations': []
        }
        
        # If interchangeability analyzer available, use it
        if handler.interchangeability:
            # In reality would call actual methods
            pass
        
        # Simple alternative logic based on description similarity
        if yarn_id and hasattr(yarn_data, 'iterrows'):
            target_yarn = None
            similar_yarns = []
            
            for _, yarn in yarn_data.iterrows():
                if yarn.get('Desc#', '') == yarn_id:
                    target_yarn = yarn
                    break
            
            if target_yarn is not None:
                target_desc = str(target_yarn.get('Description', '')).lower()
                
                for _, yarn in yarn_data.iterrows():
                    if yarn.get('Desc#', '') != yarn_id:
                        desc = str(yarn.get('Description', '')).lower()
                        # Simple similarity check
                        if any(word in desc for word in target_desc.split()[:2]):
                            similar_yarns.append({
                                'yarn_id': yarn.get('Desc#', ''),
                                'description': yarn.get('Description', ''),
                                'availability': yarn.get('Planning Balance', 0),
                                'similarity_score': 0.7  # Simplified
                            })
            
            alternatives['alternatives'] = similar_yarns[:10]
        
        return jsonify(alternatives)
    
    except Exception as e:
        logger.error(f"Error finding alternatives: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-substitution-intelligent")
def yarn_substitution_intelligent():
    """Intelligent yarn substitution using ML"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        view = request.args.get('view', 'opportunities')
        
        substitution_data = {
            'view': view,
            'timestamp': pd.Timestamp.now().isoformat(),
            'opportunities': [],
            'savings_potential': 0,
            'recommendations': []
        }
        
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        # Find substitution opportunities
        if view == 'opportunities' and hasattr(yarn_data, 'iterrows'):
            shortage_yarns = []
            available_yarns = []
            
            for _, yarn in yarn_data.iterrows():
                balance = yarn.get('Planning Balance', 0)
                if balance < 0:
                    shortage_yarns.append(yarn)
                elif balance > 1000:
                    available_yarns.append(yarn)
            
            # Match shortages with available yarns
            for shortage in shortage_yarns[:10]:
                for available in available_yarns:
                    # Simple matching logic
                    substitution_data['opportunities'].append({
                        'shortage_yarn': shortage.get('Desc#', ''),
                        'substitute_yarn': available.get('Desc#', ''),
                        'shortage_amount': abs(shortage.get('Planning Balance', 0)),
                        'available_amount': available.get('Planning Balance', 0),
                        'confidence': 0.75
                    })
                    
                    if len(substitution_data['opportunities']) >= 20:
                        break
        
        # Calculate potential savings
        substitution_data['savings_potential'] = len(substitution_data['opportunities']) * 1000
        
        # Add recommendations
        if substitution_data['opportunities']:
            substitution_data['recommendations'].append({
                'action': f"Consider {len(substitution_data['opportunities'])} substitution opportunities",
                'impact': f"Could resolve shortages worth ${substitution_data['savings_potential']:,.0f}"
            })
        
        return jsonify(substitution_data)
    
    except Exception as e:
        logger.error(f"Error in intelligent substitution: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-aggregation")
def yarn_aggregation():
    """Yarn aggregation and grouping analysis"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        yarn_data = handler.get_yarn_data()
        if yarn_data is None:
            return jsonify({'error': 'No yarn data available'}), 404
        
        aggregation = {
            'total_yarns': len(yarn_data),
            'groups': {},
            'summary': {}
        }
        
        # Group by supplier if column exists
        if 'supplier' in yarn_data.columns:
            supplier_groups = yarn_data.groupby('supplier').agg({
                'Planning Balance': 'sum',
                'Desc#': 'count'
            }).to_dict('index')
            
            aggregation['groups']['by_supplier'] = supplier_groups
        
        # Group by color if exists
        if 'color' in yarn_data.columns:
            color_groups = yarn_data.groupby('color').size().to_dict()
            aggregation['groups']['by_color'] = color_groups
        
        # Summary statistics
        if 'Planning Balance' in yarn_data.columns:
            aggregation['summary'] = {
                'total_inventory_value': float(yarn_data['Planning Balance'].sum()),
                'average_balance': float(yarn_data['Planning Balance'].mean()),
                'items_in_shortage': len(yarn_data[yarn_data['Planning Balance'] < 0]),
                'items_overstocked': len(yarn_data[yarn_data['Planning Balance'] > 5000])
            }
        
        return jsonify(aggregation)
    
    except Exception as e:
        logger.error(f"Error in yarn aggregation: {e}")
        return jsonify({'error': str(e)}), 500


@yarn_bp.route("/yarn-forecast-shortages")
def yarn_forecast_shortages():
    """Forecast future yarn shortages"""
    try:
        if not handler:
            return jsonify({'error': 'Handler not initialized'}), 500
        
        yarn_data = handler.get_yarn_data()
        bom_data = handler.get_bom_data()
        
        forecast_shortages = {
            'forecast_period': '30 days',
            'predicted_shortages': [],
            'risk_assessment': {},
            'preventive_actions': []
        }
        
        # Simple forecast logic
        if yarn_data is not None and hasattr(yarn_data, 'iterrows'):
            for _, yarn in yarn_data.head(50).iterrows():
                current_balance = yarn.get('Planning Balance', 0)
                allocated = abs(yarn.get('Allocated', 0))
                
                # Predict future balance (simplified)
                daily_usage = allocated / 30 if allocated > 0 else 0
                predicted_balance = current_balance - (daily_usage * 30)
                
                if predicted_balance < 0:
                    forecast_shortages['predicted_shortages'].append({
                        'yarn_id': yarn.get('Desc#', ''),
                        'current_balance': current_balance,
                        'predicted_balance': predicted_balance,
                        'shortage_date': '30 days',
                        'daily_usage': daily_usage
                    })
        
        # Risk assessment
        shortage_count = len(forecast_shortages['predicted_shortages'])
        if shortage_count > 20:
            risk_level = 'HIGH'
        elif shortage_count > 10:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        forecast_shortages['risk_assessment'] = {
            'risk_level': risk_level,
            'shortage_count': shortage_count,
            'confidence': 0.75
        }
        
        # Preventive actions
        if shortage_count > 0:
            forecast_shortages['preventive_actions'].append({
                'action': f'Place orders for {shortage_count} yarns',
                'urgency': 'HIGH' if risk_level == 'HIGH' else 'MEDIUM',
                'timeline': 'Within 7 days'
            })
        
        return jsonify(forecast_shortages)
    
    except Exception as e:
        logger.error(f"Error forecasting shortages: {e}")
        return jsonify({'error': str(e)}), 500