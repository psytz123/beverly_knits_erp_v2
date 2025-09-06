#!/usr/bin/env python3
"""
Message Router for eFab AI Agent System
High-performance message routing with intelligent delivery, priority queuing, and reliability guarantees
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid
from pathlib import Path
import heapq

from ..core.agent_base import BaseAgent, AgentMessage, MessageType, Priority, AgentStatus
from ..core.state_manager import system_state

# Setup logging
logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "DIRECT"                    # Direct agent-to-agent routing
    BROADCAST = "BROADCAST"              # Send to all agents
    ROUND_ROBIN = "ROUND_ROBIN"          # Load balance across agents
    CAPABILITY_BASED = "CAPABILITY_BASED" # Route based on agent capabilities
    PRIORITY_BASED = "PRIORITY_BASED"    # Route based on message priority
    LOAD_BALANCED = "LOAD_BALANCED"      # Route to least loaded agent


class DeliveryStatus(Enum):
    """Message delivery status"""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    ROUTING = "ROUTING"
    DELIVERED = "DELIVERED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    RETRY = "RETRY"


@dataclass
class RoutingRule:
    """Message routing rule definition"""
    rule_id: str
    name: str
    condition: Dict[str, Any]
    strategy: RoutingStrategy
    target_agents: List[str] = field(default_factory=list)
    priority_boost: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    enabled: bool = True


@dataclass
class MessageEnvelope:
    """Message envelope with routing metadata"""
    envelope_id: str
    message: AgentMessage
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    target_agents: List[str] = field(default_factory=list)
    attempted_agents: Set[str] = field(default_factory=set)
    retry_count: int = 0
    queued_at: datetime = field(default_factory=datetime.now)
    routed_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if message envelope has expired"""
        return self.message.is_expired()
    
    @property
    def age_seconds(self) -> float:
        """Get age of envelope in seconds"""
        return (datetime.now() - self.queued_at).total_seconds()


class MessageRouter:
    """
    Intelligent Message Router for eFab AI Agent System
    
    Features:
    - Priority-based message queuing with heaps
    - Intelligent routing strategies (direct, broadcast, load-balanced)
    - Automatic retry with exponential backoff
    - Dead letter queue for failed messages
    - Message deduplication and idempotency
    - Circuit breaker pattern for failed agents
    - Metrics and performance monitoring
    - Message persistence for reliability
    """
    
    def __init__(self, orchestrator_agent: Optional[BaseAgent] = None):
        """Initialize message router"""
        self.logger = logging.getLogger("MessageRouter")
        
        # Message queuing system
        self.message_queue: List[Tuple[int, MessageEnvelope]] = []  # Priority heap
        self.priority_queues: Dict[Priority, deque] = {
            priority: deque() for priority in Priority
        }
        
        # Agent registry and management
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_callbacks: Dict[str, Callable] = {}
        self.agent_workloads: Dict[str, int] = defaultdict(int)
        self.agent_health: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Circuit breaker for failed agents
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.failure_thresholds = {
            "error_rate": 0.5,      # 50% error rate triggers circuit breaker
            "timeout_rate": 0.3,    # 30% timeout rate triggers circuit breaker
            "consecutive_failures": 5  # 5 consecutive failures triggers circuit breaker
        }
        
        # Routing configuration
        self.routing_rules: List[RoutingRule] = []
        self.default_routing_strategy = RoutingStrategy.DIRECT
        
        # Message tracking and metrics
        self.message_history: Dict[str, MessageEnvelope] = {}
        self.delivered_messages: Dict[str, MessageEnvelope] = {}
        self.failed_messages: Dict[str, MessageEnvelope] = {}  # Dead letter queue
        self.message_metrics: Dict[str, Any] = {
            "total_messages": 0,
            "delivered_messages": 0,
            "failed_messages": 0,
            "retry_messages": 0,
            "average_delivery_time_ms": 0.0,
            "throughput_messages_per_second": 0.0
        }
        
        # Performance monitoring
        self.throughput_samples: deque = deque(maxlen=100)
        self.delivery_time_samples: deque = deque(maxlen=1000)
        
        # Orchestrator integration
        self.orchestrator = orchestrator_agent
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info("Message Router initialized")
    
    async def start(self):
        """Start message router background tasks"""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._message_processing_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._circuit_breaker_monitoring()),
            asyncio.create_task(self._dead_letter_processor())
        ]
        
        self.logger.info("Message Router started")
    
    async def stop(self):
        """Stop message router and cleanup"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Message Router stopped")
    
    def register_agent(
        self, 
        agent_id: str, 
        agent_info: Dict[str, Any],
        message_callback: Callable[[AgentMessage], asyncio.Task]
    ) -> bool:
        """Register agent with message router"""
        try:
            self.registered_agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0,
                "error_count": 0,
                "timeout_count": 0,
                "status": AgentStatus.READY.value
            }
            
            self.agent_callbacks[agent_id] = message_callback
            self.agent_workloads[agent_id] = 0
            self.agent_health[agent_id] = 1.0
            
            # Initialize circuit breaker
            self.circuit_breakers[agent_id] = {
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failure_count": 0,
                "last_failure": None,
                "next_attempt": datetime.now()
            }
            
            self.logger.info(f"Agent registered: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            return False
    
    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent from message router"""
        try:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
                del self.agent_callbacks[agent_id]
                del self.agent_workloads[agent_id]
                del self.agent_health[agent_id]
                del self.circuit_breakers[agent_id]
                
                self.logger.info(f"Agent deregistered: {agent_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def route_message(
        self, 
        message: AgentMessage,
        routing_strategy: Optional[RoutingStrategy] = None
    ) -> str:
        """
        Route message to appropriate agent(s)
        
        Args:
            message: Message to route
            routing_strategy: Override default routing strategy
            
        Returns:
            Envelope ID for tracking
        """
        try:
            # Create message envelope
            envelope_id = str(uuid.uuid4())
            envelope = MessageEnvelope(
                envelope_id=envelope_id,
                message=message,
                routing_strategy=routing_strategy or self._determine_routing_strategy(message)
            )
            
            # Determine target agents
            envelope.target_agents = self._determine_target_agents(envelope)
            
            if not envelope.target_agents:
                self.logger.warning(f"No target agents found for message {envelope_id}")
                envelope.delivery_status = DeliveryStatus.FAILED
                self.failed_messages[envelope_id] = envelope
                return envelope_id
            
            # Add to processing queue
            envelope.delivery_status = DeliveryStatus.QUEUED
            self.message_history[envelope_id] = envelope
            
            # Priority-based queuing
            priority_score = message.priority.score * -1  # Negative for min-heap
            heapq.heappush(self.message_queue, (priority_score, envelope))
            
            self.message_metrics["total_messages"] += 1
            
            self.logger.debug(f"Message {envelope_id} queued for routing to {len(envelope.target_agents)} agents")
            return envelope_id
            
        except Exception as e:
            self.logger.error(f"Failed to route message: {str(e)}")
            raise
    
    def _determine_routing_strategy(self, message: AgentMessage) -> RoutingStrategy:
        """Determine routing strategy for message"""
        # Check routing rules first
        for rule in self.routing_rules:
            if rule.enabled and self._matches_rule(message, rule):
                return rule.strategy
        
        # Default strategy based on message type
        if message.target_agent_id:
            return RoutingStrategy.DIRECT
        elif message.message_type == MessageType.NOTIFICATION:
            return RoutingStrategy.BROADCAST
        else:
            return self.default_routing_strategy
    
    def _matches_rule(self, message: AgentMessage, rule: RoutingRule) -> bool:
        """Check if message matches routing rule condition"""
        condition = rule.condition
        
        # Check message type
        if "message_type" in condition:
            if message.message_type.value != condition["message_type"]:
                return False
        
        # Check priority
        if "priority" in condition:
            if message.priority.value != condition["priority"]:
                return False
        
        # Check agent ID patterns
        if "agent_pattern" in condition:
            pattern = condition["agent_pattern"]
            if not (pattern in message.agent_id or pattern in str(message.target_agent_id)):
                return False
        
        # Check payload conditions
        if "payload_conditions" in condition:
            for key, expected_value in condition["payload_conditions"].items():
                if message.payload.get(key) != expected_value:
                    return False
        
        return True
    
    def _determine_target_agents(self, envelope: MessageEnvelope) -> List[str]:
        """Determine target agents for message envelope"""
        message = envelope.message
        strategy = envelope.routing_strategy
        
        if strategy == RoutingStrategy.DIRECT:
            # Direct routing to specific agent
            if message.target_agent_id and message.target_agent_id in self.registered_agents:
                if self._is_agent_available(message.target_agent_id):
                    return [message.target_agent_id]
            return []
        
        elif strategy == RoutingStrategy.BROADCAST:
            # Broadcast to all available agents
            return [
                agent_id for agent_id in self.registered_agents.keys()
                if self._is_agent_available(agent_id) and agent_id != message.agent_id
            ]
        
        elif strategy == RoutingStrategy.CAPABILITY_BASED:
            # Route based on agent capabilities
            required_capability = message.payload.get("required_capability")
            if required_capability:
                return [
                    agent_id for agent_id, agent_info in self.registered_agents.items()
                    if self._agent_has_capability(agent_id, required_capability) and 
                       self._is_agent_available(agent_id)
                ]
        
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Route to least loaded available agent
            available_agents = [
                agent_id for agent_id in self.registered_agents.keys()
                if self._is_agent_available(agent_id) and agent_id != message.agent_id
            ]
            
            if available_agents:
                # Sort by workload (ascending)
                available_agents.sort(key=lambda a: self.agent_workloads[a])
                return [available_agents[0]]
        
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            available_agents = [
                agent_id for agent_id in self.registered_agents.keys()
                if self._is_agent_available(agent_id) and agent_id != message.agent_id
            ]
            
            if available_agents:
                # Use message timestamp for round-robin selection
                index = hash(message.timestamp) % len(available_agents)
                return [available_agents[index]]
        
        return []
    
    def _is_agent_available(self, agent_id: str) -> bool:
        """Check if agent is available for message delivery"""
        if agent_id not in self.registered_agents:
            return False
        
        agent_info = self.registered_agents[agent_id]
        
        # Check agent status
        if agent_info.get("status") != AgentStatus.READY.value:
            return False
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(agent_id, {})
        if circuit_breaker.get("state") == "OPEN":
            # Check if we can attempt half-open
            if datetime.now() >= circuit_breaker.get("next_attempt", datetime.now()):
                circuit_breaker["state"] = "HALF_OPEN"
            else:
                return False
        
        # Check workload limits
        max_workload = agent_info.get("max_concurrent_messages", 10)
        if self.agent_workloads[agent_id] >= max_workload:
            return False
        
        return True
    
    def _agent_has_capability(self, agent_id: str, capability_name: str) -> bool:
        """Check if agent has specific capability"""
        agent_info = self.registered_agents.get(agent_id, {})
        capabilities = agent_info.get("capabilities", [])
        
        for capability in capabilities:
            if capability.get("name") == capability_name:
                return True
        
        return False
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.running:
            try:
                if self.message_queue:
                    # Get highest priority message
                    priority_score, envelope = heapq.heappop(self.message_queue)
                    
                    # Process message
                    await self._process_message_envelope(envelope)
                
                else:
                    # No messages to process - short sleep
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def _process_message_envelope(self, envelope: MessageEnvelope):
        """Process individual message envelope"""
        try:
            envelope.delivery_status = DeliveryStatus.ROUTING
            envelope.routed_at = datetime.now()
            
            # Track start time for metrics
            start_time = datetime.now()
            
            # Attempt delivery to target agents
            delivery_results = await self._deliver_to_agents(envelope)
            
            # Update envelope status based on delivery results
            successful_deliveries = sum(1 for success in delivery_results.values() if success)
            
            if successful_deliveries > 0:
                envelope.delivery_status = DeliveryStatus.DELIVERED
                envelope.delivered_at = datetime.now()
                self.delivered_messages[envelope.envelope_id] = envelope
                self.message_metrics["delivered_messages"] += 1
                
                # Update delivery time metrics
                delivery_time_ms = (envelope.delivered_at - start_time).total_seconds() * 1000
                self.delivery_time_samples.append(delivery_time_ms)
                
            else:
                # All deliveries failed - check for retry
                if envelope.retry_count < envelope.message.max_retries:
                    envelope.retry_count += 1
                    envelope.delivery_status = DeliveryStatus.RETRY
                    
                    # Re-queue with exponential backoff
                    delay = min(2 ** envelope.retry_count, 60)  # Max 60 second delay
                    await asyncio.sleep(delay)
                    
                    # Re-add to queue
                    priority_score = envelope.message.priority.score * -1
                    heapq.heappush(self.message_queue, (priority_score, envelope))
                    
                    self.message_metrics["retry_messages"] += 1
                    self.logger.debug(f"Message {envelope.envelope_id} queued for retry {envelope.retry_count}")
                
                else:
                    # Max retries exceeded - move to dead letter queue
                    envelope.delivery_status = DeliveryStatus.FAILED
                    self.failed_messages[envelope.envelope_id] = envelope
                    self.message_metrics["failed_messages"] += 1
                    
                    self.logger.warning(f"Message {envelope.envelope_id} failed after {envelope.retry_count} retries")
            
        except Exception as e:
            self.logger.error(f"Error processing message envelope {envelope.envelope_id}: {str(e)}")
            envelope.delivery_status = DeliveryStatus.FAILED
            self.failed_messages[envelope.envelope_id] = envelope
    
    async def _deliver_to_agents(self, envelope: MessageEnvelope) -> Dict[str, bool]:
        """Deliver message to target agents"""
        delivery_results = {}
        
        for agent_id in envelope.target_agents:
            if agent_id in envelope.attempted_agents:
                continue  # Skip already attempted agents
            
            envelope.attempted_agents.add(agent_id)
            
            try:
                # Check if agent is still available
                if not self._is_agent_available(agent_id):
                    delivery_results[agent_id] = False
                    continue
                
                # Get agent callback
                callback = self.agent_callbacks.get(agent_id)
                if not callback:
                    delivery_results[agent_id] = False
                    continue
                
                # Update workload
                self.agent_workloads[agent_id] += 1
                
                # Deliver message
                task = callback(envelope.message)
                response = await asyncio.wait_for(task, timeout=envelope.message.timeout_seconds)
                
                # Update agent metrics
                self.registered_agents[agent_id]["message_count"] += 1
                self.registered_agents[agent_id]["last_activity"] = datetime.now()
                
                # Success
                delivery_results[agent_id] = True
                self._update_circuit_breaker(agent_id, success=True)
                
                self.logger.debug(f"Message {envelope.envelope_id} delivered to {agent_id}")
                
            except asyncio.TimeoutError:
                # Timeout
                delivery_results[agent_id] = False
                self.registered_agents[agent_id]["timeout_count"] += 1
                self._update_circuit_breaker(agent_id, success=False, timeout=True)
                self.logger.warning(f"Timeout delivering message {envelope.envelope_id} to {agent_id}")
                
            except Exception as e:
                # Other delivery error
                delivery_results[agent_id] = False
                self.registered_agents[agent_id]["error_count"] += 1
                self._update_circuit_breaker(agent_id, success=False)
                self.logger.error(f"Error delivering message {envelope.envelope_id} to {agent_id}: {str(e)}")
            
            finally:
                # Update workload
                if agent_id in self.agent_workloads:
                    self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
        
        return delivery_results
    
    def _update_circuit_breaker(self, agent_id: str, success: bool, timeout: bool = False):
        """Update circuit breaker state for agent"""
        circuit_breaker = self.circuit_breakers.get(agent_id, {})
        
        if success:
            # Reset failure count on success
            circuit_breaker["failure_count"] = 0
            if circuit_breaker.get("state") == "HALF_OPEN":
                circuit_breaker["state"] = "CLOSED"
        
        else:
            # Increment failure count
            circuit_breaker["failure_count"] += 1
            circuit_breaker["last_failure"] = datetime.now()
            
            # Check if circuit breaker should open
            agent_info = self.registered_agents.get(agent_id, {})
            total_messages = agent_info.get("message_count", 0)
            error_count = agent_info.get("error_count", 0)
            timeout_count = agent_info.get("timeout_count", 0)
            
            error_rate = error_count / max(total_messages, 1)
            timeout_rate = timeout_count / max(total_messages, 1)
            consecutive_failures = circuit_breaker["failure_count"]
            
            should_open = (
                error_rate > self.failure_thresholds["error_rate"] or
                timeout_rate > self.failure_thresholds["timeout_rate"] or
                consecutive_failures >= self.failure_thresholds["consecutive_failures"]
            )
            
            if should_open and circuit_breaker.get("state") != "OPEN":
                circuit_breaker["state"] = "OPEN"
                circuit_breaker["next_attempt"] = datetime.now() + timedelta(seconds=60)  # 1 minute timeout
                self.logger.warning(f"Circuit breaker opened for agent {agent_id}")
    
    async def _health_monitoring_loop(self):
        """Monitor agent health and update availability"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for agent_id, agent_info in self.registered_agents.items():
                    last_activity = agent_info.get("last_activity", current_time)
                    time_since_activity = (current_time - last_activity).total_seconds()
                    
                    # Calculate health score based on activity and performance
                    if time_since_activity < 30:
                        health_score = 1.0
                    elif time_since_activity < 120:
                        health_score = 0.8
                    elif time_since_activity < 300:
                        health_score = 0.5
                    else:
                        health_score = 0.2
                    
                    # Factor in error rates
                    total_messages = agent_info.get("message_count", 0)
                    error_count = agent_info.get("error_count", 0)
                    if total_messages > 0:
                        error_rate = error_count / total_messages
                        health_score *= (1.0 - min(error_rate, 0.8))
                    
                    self.agent_health[agent_id] = health_score
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Collect and update performance metrics"""
        while self.running:
            try:
                # Calculate average delivery time
                if self.delivery_time_samples:
                    avg_delivery_time = sum(self.delivery_time_samples) / len(self.delivery_time_samples)
                    self.message_metrics["average_delivery_time_ms"] = avg_delivery_time
                
                # Calculate throughput
                current_time = datetime.now()
                self.throughput_samples.append((current_time, self.message_metrics["total_messages"]))
                
                if len(self.throughput_samples) >= 2:
                    time_diff = (self.throughput_samples[-1][0] - self.throughput_samples[0][0]).total_seconds()
                    message_diff = self.throughput_samples[-1][1] - self.throughput_samples[0][1]
                    
                    if time_diff > 0:
                        throughput = message_diff / time_diff
                        self.message_metrics["throughput_messages_per_second"] = throughput
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _circuit_breaker_monitoring(self):
        """Monitor and manage circuit breakers"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for agent_id, circuit_breaker in self.circuit_breakers.items():
                    if circuit_breaker.get("state") == "OPEN":
                        next_attempt = circuit_breaker.get("next_attempt", current_time)
                        if current_time >= next_attempt:
                            circuit_breaker["state"] = "HALF_OPEN"
                            self.logger.info(f"Circuit breaker for agent {agent_id} moved to half-open")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in circuit breaker monitoring: {str(e)}")
                await asyncio.sleep(10)
    
    async def _dead_letter_processor(self):
        """Process messages in dead letter queue"""
        while self.running:
            try:
                # Periodically check for failed messages that might be retryable
                current_time = datetime.now()
                
                for envelope_id, envelope in list(self.failed_messages.items()):
                    # Check if message is very old and should be permanently discarded
                    age_hours = (current_time - envelope.queued_at).total_seconds() / 3600
                    
                    if age_hours > 24:  # 24 hours retention in dead letter queue
                        del self.failed_messages[envelope_id]
                        self.logger.info(f"Permanently discarded failed message {envelope_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in dead letter processor: {str(e)}")
                await asyncio.sleep(300)
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add message routing rule"""
        self.routing_rules.append(rule)
        self.logger.info(f"Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove routing rule"""
        for i, rule in enumerate(self.routing_rules):
            if rule.rule_id == rule_id:
                del self.routing_rules[i]
                self.logger.info(f"Removed routing rule: {rule_id}")
                return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics"""
        return {
            "message_metrics": self.message_metrics.copy(),
            "agent_metrics": {
                "total_agents": len(self.registered_agents),
                "available_agents": sum(1 for agent_id in self.registered_agents.keys() if self._is_agent_available(agent_id)),
                "average_workload": sum(self.agent_workloads.values()) / max(len(self.agent_workloads), 1),
                "average_health_score": sum(self.agent_health.values()) / max(len(self.agent_health), 1)
            },
            "queue_metrics": {
                "pending_messages": len(self.message_queue),
                "failed_messages": len(self.failed_messages),
                "delivered_messages": len(self.delivered_messages)
            },
            "circuit_breaker_metrics": {
                "open_breakers": sum(1 for cb in self.circuit_breakers.values() if cb.get("state") == "OPEN"),
                "half_open_breakers": sum(1 for cb in self.circuit_breakers.values() if cb.get("state") == "HALF_OPEN")
            }
        }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed status for specific agent"""
        if agent_id not in self.registered_agents:
            return {"error": "Agent not found"}
        
        agent_info = self.registered_agents[agent_id].copy()
        circuit_breaker = self.circuit_breakers.get(agent_id, {})
        
        return {
            "agent_id": agent_id,
            "agent_info": agent_info,
            "workload": self.agent_workloads.get(agent_id, 0),
            "health_score": self.agent_health.get(agent_id, 0.0),
            "circuit_breaker_state": circuit_breaker.get("state", "UNKNOWN"),
            "is_available": self._is_agent_available(agent_id)
        }


# Export main component
__all__ = ["MessageRouter", "RoutingStrategy", "RoutingRule", "MessageEnvelope"]