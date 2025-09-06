#!/usr/bin/env python3
"""
Base Agent Architecture for eFab AI Agent System
Defines core agent interfaces, message protocols, and capability frameworks
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Agent message types for communication protocol"""
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE" 
    NOTIFICATION = "NOTIFICATION"
    ERROR = "ERROR"
    HEARTBEAT = "HEARTBEAT"
    SHUTDOWN = "SHUTDOWN"


class Priority(Enum):
    """Message priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    
    @property
    def score(self):
        """Numeric priority for sorting"""
        scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return scores.get(self.value, 1)


class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    BUSY = "BUSY" 
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class AgentMessage:
    """Standardized message format for agent communication"""
    agent_id: str
    target_agent_id: Optional[str]
    message_type: MessageType
    payload: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "agent_id": self.agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            agent_id=data["agent_id"],
            target_agent_id=data.get("target_agent_id"),
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            priority=Priority(data.get("priority", "MEDIUM")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"],
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return datetime.now() > (self.timestamp + timedelta(seconds=self.timeout_seconds))
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries


@dataclass
class AgentCapability:
    """Defines specific capabilities of an agent"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    estimated_duration_seconds: int = 60
    requires_human_approval: bool = False
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "requires_human_approval": self.requires_human_approval,
            "risk_level": self.risk_level
        }


@dataclass
class AgentMetrics:
    """Agent performance and operational metrics"""
    agent_id: str
    messages_processed: int = 0
    messages_failed: int = 0
    average_response_time_ms: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    uptime_seconds: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    
    def update_response_time(self, response_time_ms: float):
        """Update average response time with new measurement"""
        if self.messages_processed == 0:
            self.average_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.average_response_time_ms
            )
    
    def calculate_success_rate(self):
        """Calculate and update success rate"""
        total_messages = self.messages_processed + self.messages_failed
        if total_messages > 0:
            self.success_rate = (self.messages_processed / total_messages) * 100
        else:
            self.success_rate = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all eFab AI agents
    Provides common functionality and enforces agent contract
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_description: str,
        capabilities: List[AgentCapability] = None
    ):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for this agent
            agent_name: Human-readable name
            agent_description: Description of agent's purpose
            capabilities: List of capabilities this agent provides
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.capabilities = capabilities or []
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics(agent_id=agent_id)
        
        # Communication components
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_requests: Dict[str, AgentMessage] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Configuration and state
        self.config: Dict[str, Any] = {}
        self.shared_state: Dict[str, Any] = {}
        
        # Logging
        self.logger = logging.getLogger(f"Agent.{agent_name}")
        self.logger.info(f"Agent {agent_name} ({agent_id}) initializing...")
        
        # Register default message handlers
        self._register_default_handlers()
        
        # Initialize agent-specific components
        self._initialize()
        
        self.status = AgentStatus.READY
        self.logger.info(f"Agent {agent_name} ready")
    
    @abstractmethod
    def _initialize(self):
        """Agent-specific initialization - must be implemented by subclasses"""
        pass
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.SHUTDOWN] = self._handle_shutdown
        self.message_handlers[MessageType.ERROR] = self._handle_error
    
    async def _handle_heartbeat(self, message: AgentMessage) -> AgentMessage:
        """Handle heartbeat messages"""
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "status": self.status.value,
                "metrics": {
                    "messages_processed": self.metrics.messages_processed,
                    "success_rate": self.metrics.success_rate,
                    "average_response_time_ms": self.metrics.average_response_time_ms,
                    "uptime_seconds": self.metrics.uptime_seconds
                }
            },
            correlation_id=message.correlation_id
        )
    
    async def _handle_shutdown(self, message: AgentMessage) -> AgentMessage:
        """Handle shutdown messages"""
        self.logger.info(f"Agent {self.agent_name} shutting down...")
        self.status = AgentStatus.SHUTDOWN
        await self._cleanup()
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={"status": "SHUTDOWN_COMPLETE"},
            correlation_id=message.correlation_id
        )
    
    async def _handle_error(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle error messages"""
        error_info = message.payload
        self.logger.error(f"Received error: {error_info}")
        self.metrics.error_count += 1
        
        # Implement error recovery logic if needed
        await self._handle_error_recovery(error_info)
        
        return None
    
    async def _handle_error_recovery(self, error_info: Dict[str, Any]):
        """Implement agent-specific error recovery"""
        # Base implementation - can be overridden
        pass
    
    async def _cleanup(self):
        """Cleanup resources before shutdown"""
        # Base cleanup - can be overridden
        pass
    
    def register_capability(self, capability: AgentCapability):
        """Register a new capability for this agent"""
        self.capabilities.append(capability)
        self.logger.info(f"Registered capability: {capability.name}")
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get capability by name"""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has specific capability"""
        return self.get_capability(capability_name) is not None
    
    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Send message to another agent (implemented by orchestrator)
        This is a placeholder - actual implementation depends on message router
        """
        raise NotImplementedError("Message sending must be implemented by orchestrator")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming message and return response if needed
        
        Args:
            message: Incoming message to process
            
        Returns:
            Optional response message
        """
        start_time = datetime.now()
        
        try:
            # Check if message has expired
            if message.is_expired():
                self.logger.warning(f"Received expired message: {message.correlation_id}")
                return AgentMessage(
                    agent_id=self.agent_id,
                    target_agent_id=message.agent_id,
                    message_type=MessageType.ERROR,
                    payload={"error": "MESSAGE_EXPIRED"},
                    correlation_id=message.correlation_id
                )
            
            # Find appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                self.logger.error(f"No handler for message type: {message.message_type}")
                return AgentMessage(
                    agent_id=self.agent_id,
                    target_agent_id=message.agent_id,
                    message_type=MessageType.ERROR,
                    payload={"error": "UNSUPPORTED_MESSAGE_TYPE"},
                    correlation_id=message.correlation_id
                )
            
            # Process message
            response = await handler(message)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.update_response_time(processing_time)
            self.metrics.messages_processed += 1
            self.metrics.last_activity = datetime.now()
            self.metrics.calculate_success_rate()
            
            return response
            
        except Exception as e:
            # Handle processing errors
            self.logger.error(f"Error processing message: {str(e)}")
            self.metrics.messages_failed += 1
            self.metrics.error_count += 1
            self.metrics.calculate_success_rate()
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={
                    "error": "PROCESSING_ERROR",
                    "details": str(e)
                },
                correlation_id=message.correlation_id
            )
    
    def register_message_handler(
        self, 
        message_type: MessageType, 
        handler: Callable[[AgentMessage], AgentMessage]
    ):
        """Register a message handler for specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "messages_failed": self.metrics.messages_failed,
                "success_rate": self.metrics.success_rate,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "error_count": self.metrics.error_count,
                "uptime_seconds": self.metrics.uptime_seconds,
                "last_activity": self.metrics.last_activity.isoformat()
            }
        }
    
    async def start(self):
        """Start agent operation"""
        self.logger.info(f"Starting agent {self.agent_name}")
        self.status = AgentStatus.READY
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.status != AgentStatus.SHUTDOWN:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process message
                await self.process_message(message)
                
            except asyncio.TimeoutError:
                # No message received - continue
                continue
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Stop agent operation"""
        self.logger.info(f"Stopping agent {self.agent_name}")
        self.status = AgentStatus.SHUTDOWN
        await self._cleanup()


# Utility functions for agent management
def create_capability(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    **kwargs
) -> AgentCapability:
    """Utility function to create agent capabilities"""
    return AgentCapability(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        **kwargs
    )


def validate_message(message: AgentMessage) -> bool:
    """Validate message format and required fields"""
    try:
        # Check required fields
        required_fields = ["agent_id", "message_type", "payload"]
        for field in required_fields:
            if not hasattr(message, field) or getattr(message, field) is None:
                return False
        
        # Validate message type
        if not isinstance(message.message_type, MessageType):
            return False
        
        # Validate priority
        if not isinstance(message.priority, Priority):
            return False
        
        return True
    except Exception:
        return False


# Export key components
__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentCapability",
    "AgentMetrics",
    "MessageType",
    "Priority", 
    "AgentStatus",
    "create_capability",
    "validate_message"
]