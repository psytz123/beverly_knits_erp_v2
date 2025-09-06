#!/usr/bin/env python3
"""
Eva Avatar API Blueprint
Handles all Eva avatar interactions and chat functionality
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime
import logging
import json
import asyncio
from typing import Dict, Any, Optional

# Import Eva components
from ...ai_agents.interface.eva_avatar_agent import (
    EvaAvatarAgent, EvaState, EvaEmotion
)
from ...ai_agents.interface.customer_manager_agent import CustomerManagerAgent
from ...ai_agents.core.state_manager import system_state

# Setup logging
logger = logging.getLogger(__name__)

# Create blueprint
eva_bp = Blueprint('eva', __name__, url_prefix='/api/eva')

# Initialize Eva agent (singleton)
eva_agent = None
customer_manager = None


def get_eva_agent() -> EvaAvatarAgent:
    """Get or create Eva agent instance"""
    global eva_agent
    if eva_agent is None:
        eva_agent = EvaAvatarAgent()
        # Run initialization in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(eva_agent.initialize())
    return eva_agent


def get_customer_manager() -> CustomerManagerAgent:
    """Get or create customer manager instance"""
    global customer_manager
    if customer_manager is None:
        customer_manager = CustomerManagerAgent()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(customer_manager.initialize())
    return customer_manager


@eva_bp.route('/chat', methods=['POST'])
def eva_chat():
    """
    Handle chat messages with Eva
    
    Request body:
    {
        "message": "User message",
        "customer_id": "Optional customer ID",
        "context": {optional context object}
    }
    """
    try:
        data = request.get_json()
        message = data.get('message', '')
        customer_id = data.get('customer_id', session.get('customer_id', 'default'))
        context = data.get('context', {})
        
        # Get Eva agent
        eva = get_eva_agent()
        
        # Create async task for Eva to process message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Process message
        result = loop.run_until_complete(eva._handle_chat_message({
            'customer_id': customer_id,
            'message': message,
            'context': context
        }))
        
        # Store customer_id in session
        session['customer_id'] = customer_id
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'conversation_id': result.get('conversation_id'),
            'state': result.get('state'),
            'emotion': result.get('emotion')
        })
        
    except Exception as e:
        logger.error(f"Error in Eva chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/greeting', methods=['GET'])
def eva_greeting():
    """Get Eva's greeting message"""
    try:
        customer_id = request.args.get('customer_id', session.get('customer_id', 'default'))
        is_returning = request.args.get('returning', 'false').lower() == 'true'
        
        eva = get_eva_agent()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(eva._generate_greeting({
            'customer_id': customer_id,
            'is_returning': is_returning
        }))
        
        return jsonify({
            'success': True,
            'response': result['response']
        })
        
    except Exception as e:
        logger.error(f"Error getting Eva greeting: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/state', methods=['GET'])
def eva_state():
    """Get Eva's current state and progress"""
    try:
        customer_id = request.args.get('customer_id', session.get('customer_id', 'default'))
        
        eva = get_eva_agent()
        state_info = eva._get_current_state({'customer_id': customer_id})
        
        return jsonify({
            'success': True,
            'state': state_info['state'],
            'emotion': state_info['emotion'],
            'phase_progress': state_info['phase_progress'],
            'overall_progress': state_info['overall_progress'],
            'personality': state_info['personality']
        })
        
    except Exception as e:
        logger.error(f"Error getting Eva state: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/progress', methods=['POST'])
def update_progress():
    """Update implementation progress"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id', session.get('customer_id', 'default'))
        phase = data.get('phase')
        progress = data.get('progress', 0)
        milestone = data.get('milestone')
        
        eva = get_eva_agent()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(eva._handle_progress_update({
            'customer_id': customer_id,
            'phase': phase,
            'progress': progress,
            'milestone': milestone
        }))
        
        return jsonify({
            'success': True,
            'response': result['response']
        })
        
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/upload', methods=['POST'])
def handle_upload():
    """Handle file upload through Eva"""
    try:
        customer_id = request.form.get('customer_id', session.get('customer_id', 'default'))
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        # Process files through customer manager
        manager = get_customer_manager()
        results = []
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for file in files:
            # Process each file
            result = loop.run_until_complete(manager._process_document_upload({
                'customer_id': customer_id,
                'file_data': {
                    'size': len(file.read()),
                    'type': file.content_type
                },
                'filename': file.filename,
                'document_context': {
                    'description': f"File uploaded through Eva: {file.filename}",
                    'priority': 'HIGH'
                }
            }))
            results.append(result)
            file.seek(0)  # Reset file pointer
        
        # Generate Eva's response about the upload
        eva = get_eva_agent()
        eva_response = loop.run_until_complete(eva._handle_chat_message({
            'customer_id': customer_id,
            'message': f"I've uploaded {len(files)} file(s)",
            'context': {
                'has_file_upload': True,
                'file_count': len(files),
                'file_type': 'data files'
            }
        }))
        
        return jsonify({
            'success': True,
            'files_processed': len(results),
            'upload_results': results,
            'eva_response': eva_response['response']
        })
        
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/milestone', methods=['POST'])
def celebrate_milestone():
    """Celebrate a milestone achievement"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id', session.get('customer_id', 'default'))
        milestone_type = data.get('milestone_type')
        achievement = data.get('achievement', '')
        
        eva = get_eva_agent()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(eva._celebrate_milestone({
            'customer_id': customer_id,
            'milestone_type': milestone_type,
            'achievement': achievement
        }))
        
        return jsonify({
            'success': True,
            'response': result['response']
        })
        
    except Exception as e:
        logger.error(f"Error celebrating milestone: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/quick-action', methods=['POST'])
def handle_quick_action():
    """Handle quick action button clicks"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id', session.get('customer_id', 'default'))
        action = data.get('action')
        params = data.get('params', {})
        
        # Map actions to responses
        action_responses = {
            'start_data_collection': "Perfect! Let's begin collecting your data. Please upload your sales forecasts, inventory records, BOMs, and supplier details. I can work with any format you have.",
            'show_timeline': "Based on typical implementations, here's our timeline:\n• Week 1-2: Data Collection & Cleaning\n• Week 2-3: Process Mapping\n• Week 3-4: System Configuration\n• Week 4-5: Training & Testing\n• Week 5-6: Go Live & Support",
            'show_features': "eFab's key features include:\n• AI-powered inventory optimization\n• Real-time production planning\n• Automated demand forecasting\n• Custom dashboard configuration\n• Seamless integration capabilities\n• 24/7 AI support through agents like myself",
            'upload_files': "Ready to receive your files! You can upload spreadsheets, CSVs, or any data exports you have. I'll clean and organize everything automatically.",
            'check_progress': "Let me check your current progress...",
            'show_requirements': "For a smooth implementation, I'll need:\n• Sales/order history (12+ months preferred)\n• Current inventory levels\n• Bill of Materials (BOMs)\n• Supplier/vendor information\n• Customer lists\n• Any existing process documentation"
        }
        
        response_text = action_responses.get(action, "I'll help you with that right away!")
        
        # Get Eva's formatted response
        eva = get_eva_agent()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(eva._handle_chat_message({
            'customer_id': customer_id,
            'message': action,
            'context': {'quick_action': True, 'action': action}
        }))
        
        # Override with specific response if available
        if action in action_responses:
            result['response']['text'] = response_text
        
        return jsonify({
            'success': True,
            'response': result['response']
        })
        
    except Exception as e:
        logger.error(f"Error handling quick action: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/implementation-status', methods=['GET'])
def get_implementation_status():
    """Get detailed implementation status"""
    try:
        customer_id = request.args.get('customer_id', session.get('customer_id', 'default'))
        
        # Get status from customer manager
        manager = get_customer_manager()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        status = loop.run_until_complete(manager._get_customer_implementation_status({
            'customer_id': customer_id
        }))
        
        # Get Eva's interpretation
        eva = get_eva_agent()
        eva_state = eva._get_current_state({'customer_id': customer_id})
        
        return jsonify({
            'success': True,
            'implementation_status': status,
            'eva_summary': {
                'overall_progress': eva_state['overall_progress'],
                'current_phase': eva_state['phase_progress'],
                'next_steps': status.get('next_actions', [])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting implementation status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@eva_bp.route('/voice-config', methods=['GET'])
def get_voice_config():
    """Get Eva's voice configuration for text-to-speech"""
    try:
        eva = get_eva_agent()
        
        return jsonify({
            'success': True,
            'voice_config': eva.voice_config,
            'available_emotions': [e.value for e in EvaEmotion],
            'speech_enabled': True
        })
        
    except Exception as e:
        logger.error(f"Error getting voice config: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Export blueprint
__all__ = ['eva_bp']