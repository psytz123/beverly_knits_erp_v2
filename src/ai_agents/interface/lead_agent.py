#!/usr/bin/env python3
"""
Lead Agent for eFab AI Agent System
Primary customer interface and chatbot orchestrator for ERP implementations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import uuid

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import system_state, CustomerProfile, ImplementationPhase, IndustryType, CompanySize

# Setup logging
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Customer conversation states"""
    GREETING = "GREETING"                           # Initial customer interaction
    DISCOVERY = "DISCOVERY"                         # Learning about customer needs
    ASSESSMENT = "ASSESSMENT"                       # Assessing implementation requirements
    PLANNING = "PLANNING"                           # Implementation planning phase
    ACTIVE_IMPLEMENTATION = "ACTIVE_IMPLEMENTATION" # During implementation
    SUPPORT = "SUPPORT"                             # Post-implementation support
    ESCALATION = "ESCALATION"                       # Escalated to human
    COMPLETED = "COMPLETED"                         # Conversation concluded


class MessageIntent(Enum):
    """Customer message intent classification"""
    GREETING = "GREETING"                           # Hello, hi, good morning
    QUESTION = "QUESTION"                           # Questions about features, process
    REQUEST = "REQUEST"                             # Specific requests for action
    CONCERN = "CONCERN"                             # Problems, issues, complaints
    FEEDBACK = "FEEDBACK"                           # Feedback about implementation
    STATUS_CHECK = "STATUS_CHECK"                   # Implementation status inquiry
    TECHNICAL_ISSUE = "TECHNICAL_ISSUE"             # Technical problems
    BUSINESS_REQUIREMENT = "BUSINESS_REQUIREMENT"   # Business needs discussion
    SCHEDULE_INQUIRY = "SCHEDULE_INQUIRY"           # Timeline questions
    COST_INQUIRY = "COST_INQUIRY"                   # Pricing and cost questions


class ResponseType(Enum):
    """Types of responses the Lead Agent can provide"""
    INFORMATIONAL = "INFORMATIONAL"                 # Providing information
    CLARIFYING_QUESTION = "CLARIFYING_QUESTION"     # Asking for clarification
    ACTION_CONFIRMATION = "ACTION_CONFIRMATION"     # Confirming actions taken
    ESCALATION_NOTICE = "ESCALATION_NOTICE"         # Notifying of escalation
    STATUS_UPDATE = "STATUS_UPDATE"                 # Implementation status
    RECOMMENDATION = "RECOMMENDATION"               # Recommending actions
    TROUBLESHOOTING = "TROUBLESHOOTING"             # Technical assistance


@dataclass
class CustomerConversation:
    """Customer conversation context"""
    conversation_id: str
    customer_id: str
    customer_profile: Optional[CustomerProfile] = None
    conversation_state: ConversationState = ConversationState.GREETING
    intent_history: List[MessageIntent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    escalation_reasons: List[str] = field(default_factory=list)
    satisfaction_score: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    resolved_issues: List[str] = field(default_factory=list)


@dataclass
class ChatbotResponse:
    """Structured chatbot response"""
    response_text: str
    response_type: ResponseType
    confidence: float
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)
    escalation_needed: bool = False
    follow_up_questions: List[str] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)


class LeadAgent(BaseAgent):
    """
    Lead Agent for eFab AI Agent System
    
    Primary customer interface providing:
    - Natural language customer interaction via chatbot
    - Intent recognition and conversation management
    - Implementation status communication
    - Issue escalation and resolution coordination
    - Customer satisfaction monitoring
    - Multi-lingual support (future)
    - Integration with all other agents for comprehensive responses
    - Proactive customer communication and updates
    """
    
    def __init__(self, agent_id: str = "lead_agent"):
        """Initialize Lead Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Lead Agent",
            agent_description="Primary customer interface and conversation orchestrator"
        )
        
        # Conversation management
        self.active_conversations: Dict[str, CustomerConversation] = {}
        self.conversation_templates: Dict[ConversationState, Dict[str, Any]] = {}
        
        # Intent recognition patterns
        self.intent_patterns = {
            MessageIntent.GREETING: [
                r"hello|hi|hey|good morning|good afternoon|good evening",
                r"how are you|greetings"
            ],
            MessageIntent.QUESTION: [
                r"what|how|when|where|why|can you|could you|do you",
                r"\?",  # Question mark
                r"tell me about|explain|help me understand"
            ],
            MessageIntent.REQUEST: [
                r"please|can you|could you|i need|i want|i would like",
                r"start|begin|initiate|create|setup|configure"
            ],
            MessageIntent.CONCERN: [
                r"problem|issue|error|bug|not working|broken|failed",
                r"worried|concerned|frustrated|disappointed"
            ],
            MessageIntent.STATUS_CHECK: [
                r"status|progress|update|how is|where are we",
                r"when will|completion|timeline|schedule"
            ],
            MessageIntent.TECHNICAL_ISSUE: [
                r"error|exception|crash|slow|timeout|connection",
                r"technical|system|database|server"
            ]
        }
        
        # Response templates
        self.response_templates = {
            ConversationState.GREETING: {
                "welcome": "Hello! I'm your eFab implementation assistant. I'm here to help you with your ERP implementation journey. How can I assist you today?",
                "returning": "Welcome back! I see we've been working on your {industry} ERP implementation. How can I help you today?"
            },
            ConversationState.DISCOVERY: {
                "industry_inquiry": "I'd love to learn more about your business. What industry are you in? We specialize in furniture manufacturing, injection molding, and electrical equipment manufacturing.",
                "size_inquiry": "Could you tell me about the size of your company? How many employees do you have?",
                "requirements": "What are your main goals for implementing an ERP system?"
            }
        }
        
        # Escalation triggers
        self.escalation_triggers = {
            "complex_technical_issue": 0.8,
            "customer_dissatisfaction": 0.3,
            "repeated_failed_responses": 3,
            "implementation_critical_issue": 0.9,
            "explicit_human_request": 1.0
        }
        
        # Customer satisfaction indicators
        self.satisfaction_keywords = {
            "positive": ["great", "excellent", "perfect", "amazing", "wonderful", "fantastic", "love", "impressed"],
            "negative": ["terrible", "awful", "horrible", "hate", "disappointed", "frustrated", "angry", "upset"],
            "neutral": ["okay", "fine", "alright", "decent", "acceptable"]
        }
    
    def _initialize(self):
        """Initialize Lead Agent capabilities"""
        # Register customer interface capabilities
        self.register_capability(AgentCapability(
            name="handle_customer_conversation",
            description="Handle natural language customer conversations",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_message": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "conversation_context": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "object"},
                    "conversation_state": {"type": "string"},
                    "escalation_needed": {"type": "boolean"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="provide_status_update",
            description="Provide implementation status updates to customers",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "status_type": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status_summary": {"type": "object"},
                    "next_steps": {"type": "array"},
                    "timeline_update": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="escalate_to_human",
            description="Escalate conversation to human support",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "escalation_reason": {"type": "string"},
                    "conversation_summary": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "escalation_ticket": {"type": "object"},
                    "estimated_response_time": {"type": "string"},
                    "interim_support": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="coordinate_agent_responses",
            description="Coordinate with other agents to provide comprehensive responses",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_query": {"type": "object"},
                    "required_expertise": {"type": "array"},
                    "context": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "coordinated_response": {"type": "object"},
                    "source_agents": {"type": "array"},
                    "confidence_score": {"type": "number"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_customer_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_system_notification)
        
        # Initialize conversation templates
        self._initialize_conversation_templates()
        
        # Start background tasks
        asyncio.create_task(self._conversation_monitor_loop())
        asyncio.create_task(self._proactive_communication_loop())
    
    def _initialize_conversation_templates(self):
        """Initialize conversation state templates"""
        self.conversation_templates = {
            ConversationState.GREETING: {
                "system_prompts": [
                    "Greet the customer warmly",
                    "Identify if this is a new or returning customer",
                    "Ask how you can help today"
                ],
                "typical_responses": [
                    "Welcome message",
                    "Service overview", 
                    "Next steps guidance"
                ]
            },
            ConversationState.DISCOVERY: {
                "system_prompts": [
                    "Learn about customer's business and industry",
                    "Understand current challenges",
                    "Identify implementation requirements"
                ],
                "key_information": [
                    "Industry type",
                    "Company size",
                    "Current systems",
                    "Business goals"
                ]
            },
            ConversationState.ACTIVE_IMPLEMENTATION: {
                "system_prompts": [
                    "Provide implementation status updates",
                    "Address concerns or questions",
                    "Coordinate with implementation team"
                ],
                "available_actions": [
                    "Status check",
                    "Schedule update",
                    "Issue escalation",
                    "Progress report"
                ]
            }
        }
    
    async def _handle_customer_request(self, message: AgentMessage) -> AgentMessage:
        """Handle customer interaction requests"""
        action = message.payload.get("action")
        
        try:
            if action == "handle_customer_conversation":
                result = await self._handle_customer_conversation(message.payload)
            elif action == "provide_status_update":
                result = await self._provide_status_update(message.payload)
            elif action == "escalate_to_human":
                result = await self._escalate_to_human(message.payload)
            elif action == "coordinate_agent_responses":
                result = await self._coordinate_agent_responses(message.payload)
            elif action == "start_conversation":
                result = await self._start_conversation(message.payload)
            elif action == "end_conversation":
                result = await self._end_conversation(message.payload)
            else:
                result = {"error": "Unsupported action", "action": action}
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error handling customer request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _handle_system_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle system notifications for proactive customer communication"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "IMPLEMENTATION_MILESTONE":
            milestone_data = message.payload.get("milestone_data", {})
            await self._notify_milestone_completion(milestone_data)
        
        elif notification_type == "ISSUE_DETECTED":
            issue_data = message.payload.get("issue_data", {})
            await self._proactive_issue_communication(issue_data)
        
        elif notification_type == "IMPLEMENTATION_DELAY":
            delay_data = message.payload.get("delay_data", {})
            await self._communicate_delay(delay_data)
        
        elif notification_type == "SUCCESS_METRIC_ACHIEVED":
            success_data = message.payload.get("success_data", {})
            await self._celebrate_success(success_data)
        
        return None
    
    async def _handle_customer_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer conversation interaction"""
        customer_message = payload.get("customer_message", "")
        customer_id = payload.get("customer_id")
        context = payload.get("conversation_context", {})
        
        # Get or create conversation
        conversation_id = context.get("conversation_id", str(uuid.uuid4()))
        conversation = await self._get_or_create_conversation(conversation_id, customer_id, context)
        
        # Classify intent
        intent = self._classify_intent(customer_message)
        conversation.intent_history.append(intent)
        conversation.message_count += 1
        conversation.last_activity = datetime.now()
        
        # Update satisfaction score
        satisfaction_delta = self._analyze_sentiment(customer_message)
        conversation.satisfaction_score = max(0, min(1.0, conversation.satisfaction_score + satisfaction_delta))
        
        # Generate response based on conversation state and intent
        response = await self._generate_response(conversation, customer_message, intent)
        
        # Update conversation state if needed
        new_state = self._determine_next_state(conversation, intent, response)
        if new_state != conversation.conversation_state:
            conversation.conversation_state = new_state
        
        # Check for escalation needs
        escalation_needed = self._should_escalate(conversation, response)
        
        # Update conversation context
        conversation.context.update(response.context_updates)
        
        return {
            "response": {
                "text": response.response_text,
                "type": response.response_type.value,
                "confidence": response.confidence,
                "suggested_actions": response.suggested_actions,
                "follow_up_questions": response.follow_up_questions
            },
            "conversation_state": conversation.conversation_state.value,
            "escalation_needed": escalation_needed,
            "conversation_context": {
                "conversation_id": conversation_id,
                "message_count": conversation.message_count,
                "satisfaction_score": conversation.satisfaction_score,
                "context": conversation.context
            },
            "intent_classification": intent.value
        }
    
    async def _get_or_create_conversation(
        self, 
        conversation_id: str, 
        customer_id: str,
        context: Dict[str, Any]
    ) -> CustomerConversation:
        """Get existing conversation or create new one"""
        
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Get customer profile
        customer_profile = system_state.get_customer_profile(customer_id)
        
        # Create new conversation
        conversation = CustomerConversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
            customer_profile=customer_profile,
            context=context
        )
        
        self.active_conversations[conversation_id] = conversation
        
        # Initialize conversation state based on customer profile
        if customer_profile:
            current_phase = system_state.implementation_phases.get(customer_id)
            if current_phase:
                if current_phase in [ImplementationPhase.DISCOVERY, ImplementationPhase.CONFIGURATION]:
                    conversation.conversation_state = ConversationState.PLANNING
                elif current_phase in [ImplementationPhase.DATA_MIGRATION, ImplementationPhase.TRAINING, ImplementationPhase.TESTING]:
                    conversation.conversation_state = ConversationState.ACTIVE_IMPLEMENTATION
                elif current_phase == ImplementationPhase.COMPLETED:
                    conversation.conversation_state = ConversationState.SUPPORT
        
        return conversation
    
    def _classify_intent(self, message: str) -> MessageIntent:
        """Classify customer message intent using pattern matching"""
        message_lower = message.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent, default to QUESTION
        if intent_scores:
            return max(intent_scores.keys(), key=lambda k: intent_scores[k])
        
        # Fallback classification based on sentence structure
        if message.strip().endswith('?'):
            return MessageIntent.QUESTION
        elif any(word in message_lower for word in ['please', 'can you', 'could you']):
            return MessageIntent.REQUEST
        elif any(word in message_lower for word in ['problem', 'issue', 'error']):
            return MessageIntent.CONCERN
        
        return MessageIntent.QUESTION
    
    def _analyze_sentiment(self, message: str) -> float:
        """Analyze sentiment and return satisfaction score delta"""
        message_lower = message.lower()
        
        positive_score = sum(1 for word in self.satisfaction_keywords["positive"] if word in message_lower)
        negative_score = sum(1 for word in self.satisfaction_keywords["negative"] if word in message_lower)
        
        if positive_score > negative_score:
            return 0.1 * positive_score
        elif negative_score > positive_score:
            return -0.15 * negative_score
        else:
            return 0.0
    
    async def _generate_response(
        self, 
        conversation: CustomerConversation, 
        customer_message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Generate appropriate response based on conversation context and intent"""
        
        # Handle different conversation states
        if conversation.conversation_state == ConversationState.GREETING:
            return await self._handle_greeting_state(conversation, customer_message, intent)
        
        elif conversation.conversation_state == ConversationState.DISCOVERY:
            return await self._handle_discovery_state(conversation, customer_message, intent)
        
        elif conversation.conversation_state == ConversationState.ASSESSMENT:
            return await self._handle_assessment_state(conversation, customer_message, intent)
        
        elif conversation.conversation_state == ConversationState.PLANNING:
            return await self._handle_planning_state(conversation, customer_message, intent)
        
        elif conversation.conversation_state == ConversationState.ACTIVE_IMPLEMENTATION:
            return await self._handle_implementation_state(conversation, customer_message, intent)
        
        elif conversation.conversation_state == ConversationState.SUPPORT:
            return await self._handle_support_state(conversation, customer_message, intent)
        
        else:
            return await self._handle_general_inquiry(conversation, customer_message, intent)
    
    async def _handle_greeting_state(
        self, 
        conversation: CustomerConversation, 
        message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Handle greeting state conversation"""
        
        if intent == MessageIntent.GREETING:
            if conversation.customer_profile:
                response_text = f"Welcome back, {conversation.customer_profile.company_name}! I'm here to help with your {conversation.customer_profile.industry.value.replace('_', ' ').title()} ERP implementation. How can I assist you today?"
            else:
                response_text = "Hello! I'm your eFab implementation assistant. I'm here to help you with your ERP implementation journey. To get started, could you tell me a bit about your business?"
            
            suggested_actions = [
                {"action": "check_status", "label": "Check Implementation Status"},
                {"action": "ask_question", "label": "Ask a Question"},
                {"action": "report_issue", "label": "Report an Issue"}
            ]
            
            return ChatbotResponse(
                response_text=response_text,
                response_type=ResponseType.INFORMATIONAL,
                confidence=0.9,
                suggested_actions=suggested_actions,
                follow_up_questions=["What would you like to know about your implementation?"] if conversation.customer_profile else ["What industry is your company in?"]
            )
        
        else:
            # Customer jumped straight to a question or request
            response = await self._handle_general_inquiry(conversation, message, intent)
            response.context_updates["skip_greeting"] = True
            return response
    
    async def _handle_discovery_state(
        self, 
        conversation: CustomerConversation, 
        message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Handle discovery state conversation"""
        
        # Extract information from customer message
        context_updates = {}
        
        # Try to extract industry information
        industry_keywords = {
            "furniture": IndustryType.FURNITURE,
            "wood": IndustryType.FURNITURE,
            "cabinet": IndustryType.FURNITURE,
            "plastic": IndustryType.INJECTION_MOLDING,
            "molding": IndustryType.INJECTION_MOLDING,
            "injection": IndustryType.INJECTION_MOLDING,
            "electrical": IndustryType.ELECTRICAL_EQUIPMENT,
            "electronics": IndustryType.ELECTRICAL_EQUIPMENT,
            "equipment": IndustryType.ELECTRICAL_EQUIPMENT
        }
        
        message_lower = message.lower()
        detected_industry = None
        for keyword, industry in industry_keywords.items():
            if keyword in message_lower:
                detected_industry = industry
                break
        
        if detected_industry:
            context_updates["industry"] = detected_industry.value
            
            response_text = f"Excellent! I see you're in {detected_industry.value.replace('_', ' ').title()} manufacturing. We have specialized expertise in your industry. "
            response_text += "Could you tell me more about your current challenges or what you're hoping to achieve with an ERP implementation?"
            
            follow_up_questions = [
                "What's your biggest operational challenge right now?",
                "How many employees does your company have?",
                "Are you currently using any ERP or management software?"
            ]
        
        else:
            response_text = "Thank you for that information. To better understand how we can help, could you tell me what industry your company operates in? "
            response_text += "We specialize in furniture manufacturing, injection molding, and electrical equipment manufacturing."
            
            follow_up_questions = [
                "What industry best describes your business?",
                "What products does your company manufacture?"
            ]
        
        return ChatbotResponse(
            response_text=response_text,
            response_type=ResponseType.CLARIFYING_QUESTION,
            confidence=0.8,
            follow_up_questions=follow_up_questions,
            context_updates=context_updates
        )
    
    async def _handle_implementation_state(
        self, 
        conversation: CustomerConversation, 
        message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Handle active implementation state conversation"""
        
        if intent == MessageIntent.STATUS_CHECK:
            return await self._provide_implementation_status(conversation)
        
        elif intent == MessageIntent.CONCERN or intent == MessageIntent.TECHNICAL_ISSUE:
            return await self._handle_implementation_issue(conversation, message, intent)
        
        elif intent == MessageIntent.QUESTION:
            return await self._answer_implementation_question(conversation, message)
        
        else:
            # General implementation support
            response_text = "I'm here to help with your ongoing implementation. "
            
            # Get current implementation phase
            if conversation.customer_id:
                current_phase = system_state.implementation_phases.get(conversation.customer_id)
                if current_phase:
                    phase_name = current_phase.value.replace('_', ' ').title()
                    response_text += f"You're currently in the {phase_name} phase. "
            
            response_text += "How can I assist you today?"
            
            suggested_actions = [
                {"action": "status_update", "label": "Get Status Update"},
                {"action": "schedule_meeting", "label": "Schedule a Meeting"},
                {"action": "report_issue", "label": "Report an Issue"},
                {"action": "ask_technical_question", "label": "Ask Technical Question"}
            ]
            
            return ChatbotResponse(
                response_text=response_text,
                response_type=ResponseType.INFORMATIONAL,
                confidence=0.8,
                suggested_actions=suggested_actions
            )
    
    async def _provide_implementation_status(self, conversation: CustomerConversation) -> ChatbotResponse:
        """Provide implementation status update"""
        
        if not conversation.customer_id:
            return ChatbotResponse(
                response_text="I don't have your customer information available. Could you provide your customer ID or company name?",
                response_type=ResponseType.CLARIFYING_QUESTION,
                confidence=0.9
            )
        
        # Get implementation status from system
        customer_dashboard = system_state.get_customer_dashboard(conversation.customer_id)
        
        if not customer_dashboard:
            return ChatbotResponse(
                response_text="I'm having trouble accessing your implementation status right now. Let me escalate this to our technical team for immediate assistance.",
                response_type=ResponseType.ESCALATION_NOTICE,
                confidence=0.8,
                escalation_needed=True
            )
        
        current_phase = customer_dashboard.get("current_phase", "Unknown")
        progress_percentage = customer_dashboard.get("progress_percentage", 0)
        
        response_text = f"Here's your current implementation status:\n\n"
        response_text += f"• Current Phase: {current_phase.replace('_', ' ').title()}\n"
        response_text += f"• Overall Progress: {progress_percentage:.1f}%\n"
        
        # Add phase-specific information
        if current_phase == "DATA_MIGRATION":
            response_text += f"• Status: Data migration is in progress\n"
            response_text += f"• Next: User training will begin after data validation\n"
        elif current_phase == "TRAINING":
            response_text += f"• Status: User training sessions are underway\n"
            response_text += f"• Next: System testing and validation\n"
        elif current_phase == "TESTING":
            response_text += f"• Status: System testing and validation in progress\n"
            response_text += f"• Next: Go-live preparation\n"
        
        suggested_actions = [
            {"action": "detailed_report", "label": "Get Detailed Report"},
            {"action": "schedule_update", "label": "Schedule Status Meeting"},
            {"action": "next_steps", "label": "What Are Next Steps?"}
        ]
        
        return ChatbotResponse(
            response_text=response_text,
            response_type=ResponseType.STATUS_UPDATE,
            confidence=0.9,
            suggested_actions=suggested_actions
        )
    
    async def _handle_implementation_issue(
        self, 
        conversation: CustomerConversation, 
        message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Handle implementation issues and concerns"""
        
        # Classify issue severity
        severity_keywords = {
            "critical": ["critical", "urgent", "emergency", "down", "stopped", "broken"],
            "high": ["important", "blocking", "major", "significant"],
            "medium": ["issue", "problem", "concern", "question"],
            "low": ["minor", "cosmetic", "suggestion", "enhancement"]
        }
        
        message_lower = message.lower()
        issue_severity = "medium"  # default
        
        for severity, keywords in severity_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                issue_severity = severity
                break
        
        # Generate appropriate response based on severity
        if issue_severity == "critical":
            response_text = "I understand this is a critical issue that needs immediate attention. I'm escalating this to our technical team right now for urgent resolution. "
            response_text += "You should receive a response within 30 minutes. In the meantime, here are some immediate steps you can take:\n\n"
            
            # Provide immediate troubleshooting steps
            if "login" in message_lower or "access" in message_lower:
                response_text += "• Try clearing your browser cache and cookies\n"
                response_text += "• Verify your username and password\n"
                response_text += "• Try accessing from a different browser\n"
            elif "performance" in message_lower or "slow" in message_lower:
                response_text += "• Check your internet connection\n"
                response_text += "• Close unnecessary browser tabs\n"
                response_text += "• Try refreshing the page\n"
            else:
                response_text += "• Document any error messages you're seeing\n"
                response_text += "• Note what you were doing when the issue occurred\n"
                response_text += "• Try restarting your browser if possible\n"
            
            return ChatbotResponse(
                response_text=response_text,
                response_type=ResponseType.ESCALATION_NOTICE,
                confidence=0.9,
                escalation_needed=True,
                context_updates={"issue_severity": "critical", "issue_description": message}
            )
        
        elif issue_severity in ["high", "medium"]:
            response_text = f"I understand you're experiencing an issue. Let me help you resolve this. "
            
            # Try to provide relevant troubleshooting
            if "data" in message_lower:
                response_text += "For data-related issues, I can help coordinate with our data migration specialist. "
                response_text += "Could you provide more details about what specific data problem you're seeing?"
            
            elif "training" in message_lower or "user" in message_lower:
                response_text += "For training-related concerns, I can arrange additional training sessions or resources. "
                response_text += "What specific area would you like more help with?"
            
            else:
                response_text += "Could you provide more specific details about the issue? "
                response_text += "This will help me route it to the right specialist for quick resolution."
            
            suggested_actions = [
                {"action": "escalate_technical", "label": "Escalate to Technical Team"},
                {"action": "schedule_support", "label": "Schedule Support Call"},
                {"action": "provide_details", "label": "Provide More Details"}
            ]
            
            return ChatbotResponse(
                response_text=response_text,
                response_type=ResponseType.TROUBLESHOOTING,
                confidence=0.8,
                suggested_actions=suggested_actions,
                follow_up_questions=["Can you describe exactly what happened?", "What error messages are you seeing?"],
                context_updates={"issue_severity": issue_severity, "issue_category": "technical"}
            )
        
        else:  # Low severity
            response_text = "Thank you for the feedback. I'll make sure to pass this along to the appropriate team. "
            response_text += "Is there anything else I can help you with regarding your implementation?"
            
            return ChatbotResponse(
                response_text=response_text,
                response_type=ResponseType.ACTION_CONFIRMATION,
                confidence=0.8,
                context_updates={"feedback_logged": True}
            )
    
    async def _answer_implementation_question(
        self, 
        conversation: CustomerConversation, 
        message: str
    ) -> ChatbotResponse:
        """Answer implementation-related questions"""
        
        message_lower = message.lower()
        
        # Common implementation questions
        if any(word in message_lower for word in ["timeline", "schedule", "when", "how long"]):
            response_text = "Based on your implementation plan, here's the typical timeline:\n\n"
            response_text += "• Total Duration: 6-9 weeks (depending on complexity)\n"
            response_text += "• Discovery & Planning: Week 1\n"
            response_text += "• System Configuration: Week 2\n"
            response_text += "• Data Migration: Week 3-4\n"
            response_text += "• User Training: Week 5\n"
            response_text += "• Testing & Go-Live: Week 6\n\n"
            response_text += "Your specific timeline may vary based on your company's unique requirements."
            
        elif any(word in message_lower for word in ["cost", "price", "budget", "expense"]):
            response_text = "Implementation costs vary based on your company size, complexity, and specific requirements. "
            response_text += "Our AI-powered approach typically reduces implementation costs by 85% compared to traditional methods. "
            response_text += "For specific pricing information, I can connect you with our sales team who can provide a customized quote."
            
        elif any(word in message_lower for word in ["training", "learn", "how to use"]):
            response_text = "We provide comprehensive training as part of your implementation:\n\n"
            response_text += "• Role-specific training for different user types\n"
            response_text += "• Interactive training sessions\n"
            response_text += "• Documentation and video tutorials\n"
            response_text += "• Post-implementation support\n\n"
            response_text += "Training is customized for your industry and specific workflows."
            
        elif any(word in message_lower for word in ["support", "help", "assistance"]):
            response_text = "You'll have continuous support throughout and after implementation:\n\n"
            response_text += "• Dedicated implementation team\n"
            response_text += "• 24/7 chatbot support (that's me!)\n"
            response_text += "• Technical support escalation\n"
            response_text += "• Post-go-live optimization\n\n"
            response_text += "I'm always here to help answer questions and coordinate support."
            
        else:
            # General question - try to route to appropriate specialist
            response_text = "That's a great question! To give you the most accurate answer, I'd like to coordinate with our specialist team. "
            response_text += "Could you provide a bit more context about what specifically you'd like to know?"
            
        suggested_actions = [
            {"action": "more_details", "label": "Get More Details"},
            {"action": "speak_to_specialist", "label": "Speak to Specialist"},
            {"action": "schedule_call", "label": "Schedule a Call"}
        ]
        
        return ChatbotResponse(
            response_text=response_text,
            response_type=ResponseType.INFORMATIONAL,
            confidence=0.8,
            suggested_actions=suggested_actions
        )
    
    async def _handle_general_inquiry(
        self, 
        conversation: CustomerConversation, 
        message: str,
        intent: MessageIntent
    ) -> ChatbotResponse:
        """Handle general inquiries and fallback responses"""
        
        if intent == MessageIntent.GREETING:
            response_text = "Hello! I'm your eFab implementation assistant. How can I help you today?"
            confidence = 0.9
        
        elif intent == MessageIntent.QUESTION:
            response_text = "I'd be happy to help answer your question. Could you provide more details about what you'd like to know?"
            confidence = 0.7
        
        elif intent == MessageIntent.REQUEST:
            response_text = "I'll do my best to help with your request. What specifically would you like me to assist you with?"
            confidence = 0.7
        
        else:
            response_text = "I'm here to help with your ERP implementation. Could you tell me more about what you need assistance with?"
            confidence = 0.6
        
        suggested_actions = [
            {"action": "implementation_status", "label": "Check Implementation Status"},
            {"action": "general_info", "label": "General Information"},
            {"action": "technical_support", "label": "Technical Support"},
            {"action": "speak_to_human", "label": "Speak to Human"}
        ]
        
        return ChatbotResponse(
            response_text=response_text,
            response_type=ResponseType.CLARIFYING_QUESTION,
            confidence=confidence,
            suggested_actions=suggested_actions
        )
    
    def _determine_next_state(
        self, 
        conversation: CustomerConversation, 
        intent: MessageIntent,
        response: ChatbotResponse
    ) -> ConversationState:
        """Determine next conversation state"""
        
        current_state = conversation.conversation_state
        
        # State transition logic
        if current_state == ConversationState.GREETING:
            if intent in [MessageIntent.QUESTION, MessageIntent.REQUEST]:
                return ConversationState.DISCOVERY
            elif intent == MessageIntent.STATUS_CHECK:
                return ConversationState.ACTIVE_IMPLEMENTATION
        
        elif current_state == ConversationState.DISCOVERY:
            if "industry" in response.context_updates:
                return ConversationState.ASSESSMENT
        
        elif current_state == ConversationState.ASSESSMENT:
            if conversation.message_count > 5:
                return ConversationState.PLANNING
        
        # Check for escalation state
        if response.escalation_needed:
            return ConversationState.ESCALATION
        
        return current_state
    
    def _should_escalate(self, conversation: CustomerConversation, response: ChatbotResponse) -> bool:
        """Determine if conversation should be escalated to human"""
        
        # Already marked for escalation in response
        if response.escalation_needed:
            return True
        
        # Check satisfaction score
        if conversation.satisfaction_score < self.escalation_triggers["customer_dissatisfaction"]:
            conversation.escalation_reasons.append("Low customer satisfaction")
            return True
        
        # Check for repeated issues
        recent_concerns = sum(
            1 for intent in conversation.intent_history[-5:] 
            if intent in [MessageIntent.CONCERN, MessageIntent.TECHNICAL_ISSUE]
        )
        
        if recent_concerns >= 3:
            conversation.escalation_reasons.append("Multiple unresolved concerns")
            return True
        
        # Check for explicit human request
        if any(word in conversation.intent_history[-1].value.lower() for word in ["human", "person", "representative", "agent"]):
            conversation.escalation_reasons.append("Explicit human request")
            return True
        
        # Low confidence responses repeatedly
        if response.confidence < 0.5 and len([r for r in conversation.context.get("low_confidence_responses", [])]) >= 2:
            conversation.escalation_reasons.append("Repeated low confidence responses")
            return True
        
        return False
    
    async def _provide_status_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Provide implementation status update"""
        customer_id = payload.get("customer_id")
        status_type = payload.get("status_type", "general")
        
        if not customer_id:
            return {"error": "Customer ID required"}
        
        # Get customer dashboard data
        dashboard_data = system_state.get_customer_dashboard(customer_id)
        
        if not dashboard_data:
            return {"error": "Customer data not found"}
        
        current_phase = dashboard_data.get("current_phase", "Unknown")
        progress = dashboard_data.get("progress_percentage", 0)
        
        status_summary = {
            "current_phase": current_phase,
            "progress_percentage": progress,
            "phase_description": self._get_phase_description(current_phase),
            "estimated_completion": self._estimate_completion_date(dashboard_data)
        }
        
        next_steps = self._get_next_steps(current_phase)
        
        timeline_update = {
            "current_week": self._get_current_week(dashboard_data),
            "total_weeks": dashboard_data.get("implementation_plan", {}).get("estimated_duration_weeks", 6),
            "milestones_completed": self._get_completed_milestones(dashboard_data),
            "upcoming_milestones": self._get_upcoming_milestones(dashboard_data)
        }
        
        return {
            "status_summary": status_summary,
            "next_steps": next_steps,
            "timeline_update": timeline_update
        }
    
    def _get_phase_description(self, phase: str) -> str:
        """Get human-readable phase description"""
        descriptions = {
            "PRE_ASSESSMENT": "Initial assessment and planning",
            "DISCOVERY": "Business requirements discovery",
            "CONFIGURATION": "System configuration and setup",
            "DATA_MIGRATION": "Data migration from legacy systems",
            "TRAINING": "User training and knowledge transfer",
            "TESTING": "System testing and validation",
            "GO_LIVE": "Production deployment and go-live",
            "STABILIZATION": "Post go-live stabilization",
            "OPTIMIZATION": "Performance optimization",
            "COMPLETED": "Implementation completed successfully"
        }
        return descriptions.get(phase, "Implementation in progress")
    
    def _estimate_completion_date(self, dashboard_data: Dict[str, Any]) -> str:
        """Estimate implementation completion date"""
        # Simplified estimation - in real implementation this would be more sophisticated
        progress = dashboard_data.get("progress_percentage", 0)
        if progress > 0:
            remaining_percentage = 100 - progress
            estimated_days = (remaining_percentage / 100) * 42  # 6 weeks * 7 days
            completion_date = datetime.now() + timedelta(days=estimated_days)
            return completion_date.strftime("%Y-%m-%d")
        
        return (datetime.now() + timedelta(weeks=6)).strftime("%Y-%m-%d")
    
    def _get_next_steps(self, current_phase: str) -> List[str]:
        """Get next steps for current phase"""
        next_steps_map = {
            "PRE_ASSESSMENT": [
                "Complete business requirements assessment",
                "Finalize implementation timeline",
                "Set up project team communications"
            ],
            "DISCOVERY": [
                "Document current business processes",
                "Identify integration requirements",
                "Define success criteria"
            ],
            "CONFIGURATION": [
                "Complete system configuration",
                "Set up user roles and permissions", 
                "Configure workflows and approvals"
            ],
            "DATA_MIGRATION": [
                "Validate migrated data accuracy",
                "Complete remaining data transfers",
                "Prepare for user training"
            ],
            "TRAINING": [
                "Complete user training sessions",
                "Distribute training materials",
                "Schedule go-live preparation"
            ],
            "TESTING": [
                "Complete user acceptance testing",
                "Address any identified issues",
                "Finalize go-live checklist"
            ]
        }
        
        return next_steps_map.get(current_phase, ["Continue with current implementation phase"])
    
    def _get_current_week(self, dashboard_data: Dict[str, Any]) -> int:
        """Get current implementation week"""
        progress = dashboard_data.get("progress_percentage", 0)
        total_weeks = dashboard_data.get("implementation_plan", {}).get("estimated_duration_weeks", 6)
        return max(1, int((progress / 100) * total_weeks))
    
    def _get_completed_milestones(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Get completed milestones"""
        # Simplified - would integrate with actual milestone tracking
        current_phase = dashboard_data.get("current_phase", "")
        completed = []
        
        phase_order = ["PRE_ASSESSMENT", "DISCOVERY", "CONFIGURATION", "DATA_MIGRATION", "TRAINING", "TESTING"]
        
        for phase in phase_order:
            if phase == current_phase:
                break
            completed.append(phase.replace("_", " ").title())
        
        return completed
    
    def _get_upcoming_milestones(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Get upcoming milestones"""
        current_phase = dashboard_data.get("current_phase", "")
        upcoming = []
        
        phase_order = ["PRE_ASSESSMENT", "DISCOVERY", "CONFIGURATION", "DATA_MIGRATION", "TRAINING", "TESTING", "GO_LIVE"]
        
        try:
            current_index = phase_order.index(current_phase)
            upcoming = [phase.replace("_", " ").title() for phase in phase_order[current_index + 1:current_index + 3]]
        except ValueError:
            upcoming = ["Next Phase", "Final Deployment"]
        
        return upcoming
    
    async def _escalate_to_human(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate conversation to human support"""
        customer_id = payload.get("customer_id")
        escalation_reason = payload.get("escalation_reason", "Customer requested human support")
        conversation_summary = payload.get("conversation_summary", {})
        
        # Create escalation ticket
        escalation_ticket = {
            "ticket_id": f"ESC_{int(datetime.now().timestamp())}",
            "customer_id": customer_id,
            "escalation_reason": escalation_reason,
            "priority": self._determine_escalation_priority(escalation_reason),
            "conversation_context": conversation_summary,
            "created_at": datetime.now().isoformat(),
            "status": "OPEN"
        }
        
        # Determine response time based on priority
        priority_response_times = {
            "CRITICAL": "15 minutes",
            "HIGH": "1 hour",
            "MEDIUM": "4 hours", 
            "LOW": "24 hours"
        }
        
        response_time = priority_response_times.get(escalation_ticket["priority"], "4 hours")
        
        # Provide interim support suggestions
        interim_support = [
            "I've escalated your request to our specialist team",
            "You should receive a response within " + response_time,
            "In the meantime, you can check our knowledge base for common solutions",
            "Feel free to continue the conversation with me for any other questions"
        ]
        
        return {
            "escalation_ticket": escalation_ticket,
            "estimated_response_time": response_time,
            "interim_support": interim_support
        }
    
    def _determine_escalation_priority(self, reason: str) -> str:
        """Determine escalation priority based on reason"""
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ["critical", "urgent", "emergency", "down", "stopped"]):
            return "CRITICAL"
        elif any(word in reason_lower for word in ["important", "blocking", "major", "significant"]):
            return "HIGH"
        elif any(word in reason_lower for word in ["issue", "problem", "concern"]):
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _coordinate_agent_responses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with other agents to provide comprehensive responses"""
        customer_query = payload.get("customer_query", {})
        required_expertise = payload.get("required_expertise", [])
        context = payload.get("context", {})
        
        coordinated_response = {
            "primary_response": "",
            "supporting_information": {},
            "recommended_actions": []
        }
        
        source_agents = []
        confidence_scores = []
        
        # Route to appropriate specialist agents based on expertise needed
        if "technical" in required_expertise:
            # Would coordinate with technical agents
            coordinated_response["supporting_information"]["technical"] = "Technical analysis coordinated with specialist team"
            source_agents.append("technical_specialist")
            confidence_scores.append(0.8)
        
        if "implementation" in required_expertise:
            # Would coordinate with implementation agents
            coordinated_response["supporting_information"]["implementation"] = "Implementation guidance from project management team"
            source_agents.append("implementation_manager")
            confidence_scores.append(0.9)
        
        if "industry_specific" in required_expertise:
            # Would coordinate with industry-specific agents
            industry = context.get("industry", "GENERIC")
            coordinated_response["supporting_information"]["industry"] = f"Industry-specific guidance for {industry} manufacturing"
            source_agents.append(f"{industry.lower()}_specialist")
            confidence_scores.append(0.85)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
        
        # Generate primary response
        coordinated_response["primary_response"] = "I've coordinated with our specialist team to provide you with comprehensive information. "
        coordinated_response["primary_response"] += "Based on the combined expertise, here's what we recommend..."
        
        return {
            "coordinated_response": coordinated_response,
            "source_agents": source_agents,
            "confidence_score": overall_confidence
        }
    
    async def _start_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Start new conversation with customer"""
        customer_id = payload.get("customer_id")
        initial_context = payload.get("initial_context", {})
        
        conversation_id = str(uuid.uuid4())
        conversation = await self._get_or_create_conversation(conversation_id, customer_id, initial_context)
        
        # Generate welcome message
        welcome_response = await self._generate_response(
            conversation, 
            "Hello", 
            MessageIntent.GREETING
        )
        
        return {
            "conversation_id": conversation_id,
            "welcome_message": welcome_response.response_text,
            "suggested_actions": welcome_response.suggested_actions,
            "conversation_state": conversation.conversation_state.value
        }
    
    async def _end_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """End conversation with customer"""
        conversation_id = payload.get("conversation_id")
        
        if conversation_id in self.active_conversations:
            conversation = self.active_conversations[conversation_id]
            conversation.conversation_state = ConversationState.COMPLETED
            
            # Generate conversation summary
            summary = {
                "conversation_id": conversation_id,
                "duration_minutes": (datetime.now() - conversation.started_at).total_seconds() / 60,
                "message_count": conversation.message_count,
                "satisfaction_score": conversation.satisfaction_score,
                "issues_resolved": conversation.resolved_issues,
                "escalation_needed": len(conversation.escalation_reasons) > 0
            }
            
            # Archive conversation
            del self.active_conversations[conversation_id]
            
            return {
                "conversation_summary": summary,
                "status": "COMPLETED"
            }
        
        return {"error": "Conversation not found"}
    
    async def _conversation_monitor_loop(self):
        """Background loop to monitor conversation health"""
        while self.status != "SHUTDOWN":
            try:
                current_time = datetime.now()
                
                # Check for stale conversations
                stale_conversations = []
                for conv_id, conversation in self.active_conversations.items():
                    time_since_activity = (current_time - conversation.last_activity).total_seconds()
                    
                    if time_since_activity > 1800:  # 30 minutes
                        stale_conversations.append(conv_id)
                
                # Handle stale conversations
                for conv_id in stale_conversations:
                    conversation = self.active_conversations[conv_id]
                    conversation.conversation_state = ConversationState.COMPLETED
                    del self.active_conversations[conv_id]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in conversation monitor loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def _proactive_communication_loop(self):
        """Background loop for proactive customer communication"""
        while self.status != "SHUTDOWN":
            try:
                # Check for customers who might need proactive communication
                # This would integrate with implementation monitoring
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in proactive communication loop: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _notify_milestone_completion(self, milestone_data: Dict[str, Any]):
        """Notify customer of milestone completion"""
        # Implementation for proactive milestone notifications
        pass
    
    async def _proactive_issue_communication(self, issue_data: Dict[str, Any]):
        """Proactively communicate issues to customers"""
        # Implementation for proactive issue communication
        pass
    
    async def _communicate_delay(self, delay_data: Dict[str, Any]):
        """Communicate implementation delays"""
        # Implementation for delay communication
        pass
    
    async def _celebrate_success(self, success_data: Dict[str, Any]):
        """Celebrate implementation successes with customers"""
        # Implementation for success celebration
        pass


# Export main component
__all__ = ["LeadAgent", "ConversationState", "MessageIntent", "ResponseType"]