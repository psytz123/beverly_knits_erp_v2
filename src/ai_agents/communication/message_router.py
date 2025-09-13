#!/usr/bin/env python3
"""
Message Router for eFab AI Agent System
Handles routing and delivery of messages between agents
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
import uuid
import json

from ..core.agent_base import AgentMessage, MessageType, Priority, validate_message

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_info: Dict[str, Any]
    message_callback: Callable[[AgentMessage], Any]
    registered_at: datetime
    last_activity: datetime
    message_count: int = 0


class MessageRouter:
    """
    Message Router for AI Agent Communication
    
    Features:
    - Agent registration and discovery
    - Message routing and delivery
    - Broadcast and multicast messaging
    - Message queuing and retry logic
    - Performance monitoring
    - Dead letter handling
    """
    
    def __init__(self):
        """Initialize message router"""
        self.logger = logging.getLogger("MessageRouter")
        
        # Agent registry
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Message queues
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.dead_letter_queue: List[AgentMessage] = []
        self.pending_messages: Dict[str, AgentMessage] = {}
        
        # Routing tables
        self.capability_routing: Dict[str, List[str]] = {}  # capability -> agent_ids
        self.message_handlers: Dict[str, Callable] = {}
        
        # Performance metrics
        self.router_metrics = {
            "messages_routed": 0,
            "messages_failed": 0,
            "average_routing_time_ms": 0.0,
            "active_agents": 0,
            "queue_size": 0
        }
        
        # Router state
        self.is_running = False
        self.router_task: Optional[asyncio.Task] = None
        
        self.logger.info("Message Router initialized")
    
    async def start(self):
        """Start the message router"""
        if self.is_running:
            self.logger.warning("Message router is already running")
            return
        
        self.is_running = True
        self.router_task = asyncio.create_task(self._message_processing_loop())
        self.logger.info("Message router started")
    
    async def stop(self):
        """Stop the message router"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.router_task:
            self.router_task.cancel()
            try:
                await self.router_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Message router stopped")
    
    def register_agent(
        self, 
        agent_id: str, 
        agent_info: Dict[str, Any], 
        message_callback: Callable[[AgentMessage], Any]
    ) -> bool:
        """
        Register an agent with the router
        
        Args:
            agent_id: Unique agent identifier
            agent_info: Agent metadata and capabilities
            message_callback: Callback function to deliver messages
            
        Returns:
            True if registration successful
        """
        try:
            if agent_id in self.registered_agents:
                self.logger.warning(f"Agent {agent_id} is already registered")
                return False
            
            # Create registration
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_info=agent_info,
                message_callback=message_callback,
                registered_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.registered_agents[agent_id] = registration
            
            # Register capabilities
            capabilities = agent_info.get("capabilities", [])
            if isinstance(capabilities, list) and capabilities:
                self.agent_capabilities[agent_id] = [
                    cap if isinstance(cap, str) else cap.get("name", str(cap))
                    for cap in capabilities
                ]
                
                # Update capability routing
                for capability in self.agent_capabilities[agent_id]:
                    if capability not in self.capability_routing:
                        self.capability_routing[capability] = []
                    self.capability_routing[capability].append(agent_id)
            
            self.router_metrics["active_agents"] = len(self.registered_agents)
            
            self.logger.info(f"Agent registered: {agent_id} with {len(capabilities)} capabilities")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            return False
    
    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the router"""
        try:
            if agent_id not in self.registered_agents:
                self.logger.warning(f"Agent {agent_id} is not registered")
                return False
            
            # Remove from capability routing
            capabilities = self.agent_capabilities.get(agent_id, [])
            for capability in capabilities:
                if capability in self.capability_routing:
                    if agent_id in self.capability_routing[capability]:
                        self.capability_routing[capability].remove(agent_id)
                    if not self.capability_routing[capability]:
                        del self.capability_routing[capability]
            
            # Clean up
            del self.registered_agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            
            self.router_metrics["active_agents"] = len(self.registered_agents)
            
            self.logger.info(f"Agent deregistered: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def route_message(self, message: AgentMessage) -> bool:
        """
        Route a message to its destination
        
        Args:
            message: Message to route
            
        Returns:
            True if message was queued successfully
        """
        try:
            # Validate message
            if not validate_message(message):
                self.logger.error(f"Invalid message format: {message.correlation_id}")
                return False
            
            # Check if message has expired
            if message.is_expired():
                self.logger.warning(f"Message expired: {message.correlation_id}")
                self.dead_letter_queue.append(message)
                return False
            
            # Queue message for processing
            await self.message_queue.put(message)
            self.router_metrics["queue_size"] = self.message_queue.qsize()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to route message: {str(e)}")
            return False
    
    async def broadcast_message(self, message: AgentMessage, agent_filter: Optional[Callable] = None) -> int:
        """
        Broadcast message to all registered agents
        
        Args:
            message: Message to broadcast
            agent_filter: Optional filter function for agents
            
        Returns:
            Number of agents message was sent to
        """
        sent_count = 0
        
        for agent_id, registration in self.registered_agents.items():
            # Apply filter if provided
            if agent_filter and not agent_filter(registration):
                continue
            
            # Create copy of message for each agent
            agent_message = AgentMessage(
                agent_id=message.agent_id,
                target_agent_id=agent_id,
                message_type=message.message_type,
                payload=message.payload.copy(),
                priority=message.priority,
                correlation_id=f"{message.correlation_id}_{agent_id}"
            )
            
            if await self.route_message(agent_message):
                sent_count += 1
        
        self.logger.info(f"Broadcast message sent to {sent_count} agents")
        return sent_count
    
    async def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability"""
        return self.capability_routing.get(capability, [])
    
    async def send_to_capability(self, message: AgentMessage, capability: str) -> bool:
        """
        Send message to an agent that has a specific capability
        
        Args:
            message: Message to send
            capability: Required capability
            
        Returns:
            True if message was sent successfully
        """
        capable_agents = await self.find_agents_by_capability(capability)
        
        if not capable_agents:
            self.logger.warning(f"No agents found with capability: {capability}")
            return False
        
        # Select agent (simple round-robin for now)
        selected_agent = capable_agents[0]  # TODO: Implement better selection logic
        
        # Update target
        message.target_agent_id = selected_agent
        
        return await self.route_message(message)
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        self.logger.info("Message processing loop started")
        
        while self.is_running:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message
                await self._deliver_message(message)
                
            except asyncio.TimeoutError:
                # No message received - continue
                continue
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Message processing loop stopped")
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver message to target agent"""
        start_time = datetime.now()
        
        try:
            target_agent_id = message.target_agent_id
            
            # Handle broadcast messages (no specific target)
            if not target_agent_id:
                await self.broadcast_message(message)
                return
            
            # Check if target agent is registered
            if target_agent_id not in self.registered_agents:
                self.logger.warning(f"Target agent not found: {target_agent_id}")
                self.dead_letter_queue.append(message)
                self.router_metrics["messages_failed"] += 1
                return
            
            # Get agent registration
            registration = self.registered_agents[target_agent_id]
            
            # Deliver message
            try:
                # Call agent's message callback
                if asyncio.iscoroutinefunction(registration.message_callback):
                    await registration.message_callback(message)
                else:
                    registration.message_callback(message)
                
                # Update metrics
                registration.message_count += 1
                registration.last_activity = datetime.now()
                
                self.router_metrics["messages_routed"] += 1
                
                # Update average routing time
                routing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._update_average_routing_time(routing_time_ms)
                
                self.logger.debug(f"Message delivered: {message.correlation_id} -> {target_agent_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to deliver message to {target_agent_id}: {str(e)}")
                self.dead_letter_queue.append(message)
                self.router_metrics["messages_failed"] += 1
                
        except Exception as e:
            self.logger.error(f"Error delivering message: {str(e)}")
            self.dead_letter_queue.append(message)
            self.router_metrics["messages_failed"] += 1
    
    def _update_average_routing_time(self, routing_time_ms: float):
        """Update average routing time metric"""
        current_avg = self.router_metrics["average_routing_time_ms"]
        total_routed = self.router_metrics["messages_routed"]
        
        if total_routed == 1:
            self.router_metrics["average_routing_time_ms"] = routing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.router_metrics["average_routing_time_ms"] = (
                alpha * routing_time_ms + (1 - alpha) * current_avg
            )
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get comprehensive router status"""
        return {
            "is_running": self.is_running,
            "registered_agents": len(self.registered_agents),
            "queue_size": self.message_queue.qsize(),
            "dead_letter_count": len(self.dead_letter_queue),
            "capabilities_registered": len(self.capability_routing),
            "metrics": self.router_metrics.copy(),
            "agent_list": list(self.registered_agents.keys())
        }
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific agent"""
        if agent_id not in self.registered_agents:
            return None
        
        registration = self.registered_agents[agent_id]
        return {
            "agent_id": agent_id,
            "agent_info": registration.agent_info,
            "registered_at": registration.registered_at.isoformat(),
            "last_activity": registration.last_activity.isoformat(),
            "message_count": registration.message_count,
            "capabilities": self.agent_capabilities.get(agent_id, [])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform router health check"""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "uptime_seconds": (datetime.now() - datetime.now()).total_seconds(),  # TODO: Track actual uptime
            "performance": {
                "average_routing_time_ms": self.router_metrics["average_routing_time_ms"],
                "success_rate": self._calculate_success_rate(),
                "queue_utilization": min(self.message_queue.qsize() / 1000, 1.0)  # Assume max 1000
            },
            "agents": {
                "total_registered": len(self.registered_agents),
                "active_in_last_hour": self._count_active_agents(3600),
                "capabilities_available": len(self.capability_routing)
            }
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate message routing success rate"""
        total_messages = self.router_metrics["messages_routed"] + self.router_metrics["messages_failed"]
        if total_messages == 0:
            return 1.0
        return self.router_metrics["messages_routed"] / total_messages
    
    def _count_active_agents(self, seconds: int) -> int:
        """Count agents active within specified seconds"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return sum(
            1 for reg in self.registered_agents.values()
            if reg.last_activity > cutoff_time
        )


# Export main component
__all__ = ["MessageRouter", "AgentRegistration"]