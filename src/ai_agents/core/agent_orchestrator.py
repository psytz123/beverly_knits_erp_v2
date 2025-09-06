"""
Agent Orchestrator
==================

Central coordination system for managing multiple AI agents, message routing,
task distribution, and system-wide performance monitoring.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
import heapq

from .base_agent import BaseAgent, Message, Task, MessagePriority, AgentState


@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_type: str
    name: str
    capabilities: List[str]
    agent_instance: BaseAgent
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class WorkloadMetrics:
    """Agent workload tracking"""
    pending_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_completion_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    @property
    def load_score(self) -> float:
        """Calculate load score for task assignment"""
        return (
            self.pending_tasks * 1.0 +
            self.active_tasks * 2.0 +
            self.cpu_usage * 0.5 +
            self.memory_usage * 0.3
        )


class AgentOrchestrator:
    """
    Central orchestrator for the eFab AI Agent System
    
    Responsibilities:
    - Agent lifecycle management
    - Message routing and delivery
    - Task distribution and load balancing
    - Performance monitoring and optimization
    - Fault tolerance and recovery
    - Training coordination
    """
    
    def __init__(self, orchestrator_id: str = "orchestrator"):
        self.orchestrator_id = orchestrator_id
        self.logger = logging.getLogger(f"eFab.Orchestrator.{orchestrator_id}")
        
        # Agent management
        self.agents: Dict[str, AgentRegistration] = {}
        self.agent_types: Dict[str, List[str]] = defaultdict(list)
        self.workload_metrics: Dict[str, WorkloadMetrics] = {}
        
        # Message routing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.delivery_callbacks: Dict[str, Callable] = {}
        self.message_history: List[Message] = []
        
        # Task management
        self.task_queue = []  # Priority queue using heapq
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.active_tasks: Dict[str, Task] = {}
        
        # System state
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # Performance tracking
        self.system_metrics = {
            "messages_routed": 0,
            "tasks_distributed": 0,
            "agents_registered": 0,
            "uptime": 0.0,
            "average_response_time": 0.0
        }
        
        # Training support
        self.training_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Agent Orchestrator {orchestrator_id} initialized")
    
    # =============================================================================
    # Orchestrator Lifecycle
    # =============================================================================
    
    async def start(self):
        """Start the orchestrator"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start core processing loops
        asyncio.create_task(self._message_routing_loop())
        asyncio.create_task(self._task_distribution_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._metrics_update_loop())
        
        self.logger.info("Agent Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Agent Orchestrator")
        
        # Stop all agents
        stop_tasks = []
        for registration in self.agents.values():
            if registration.is_active:
                stop_tasks.append(registration.agent_instance.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.is_running = False
        self.logger.info("Agent Orchestrator stopped")
    
    # =============================================================================
    # Agent Management
    # =============================================================================
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the orchestrator"""
        try:
            if agent.agent_id in self.agents:
                self.logger.warning(f"Agent {agent.agent_id} already registered")
                return False
            
            # Create registration
            registration = AgentRegistration(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                name=agent.name,
                capabilities=agent.capabilities.copy(),
                agent_instance=agent
            )
            
            # Store registration
            self.agents[agent.agent_id] = registration
            self.agent_types[agent.agent_type].append(agent.agent_id)
            self.workload_metrics[agent.agent_id] = WorkloadMetrics()
            
            # Start the agent if not already started
            if agent.state == AgentState.INITIALIZING:
                await agent.start()
            
            self.system_metrics["agents_registered"] += 1
            
            self.logger.info(f"Agent registered: {agent.name} ({agent.agent_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id not in self.agents:
                return False
            
            registration = self.agents[agent_id]
            
            # Stop the agent
            if registration.is_active:
                await registration.agent_instance.stop()
            
            # Remove from tracking
            self.agent_types[registration.agent_type].remove(agent_id)
            del self.agents[agent_id]
            del self.workload_metrics[agent_id]
            
            self.logger.info(f"Agent unregistered: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentRegistration]:
        """Get all agents of a specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids 
                if self.agents[agent_id].is_active]
    
    def get_agent_by_capability(self, capability: str) -> Optional[AgentRegistration]:
        """Find the best agent for a specific capability"""
        candidates = []
        
        for registration in self.agents.values():
            if capability in registration.capabilities and registration.is_active:
                load_score = self.workload_metrics[registration.agent_id].load_score
                candidates.append((load_score, registration))
        
        if candidates:
            # Return agent with lowest load score
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return None
    
    # =============================================================================
    # Message Routing
    # =============================================================================
    
    async def route_message(self, message: Message) -> bool:
        """Route message to target agent"""
        try:
            # Add to routing queue
            await self.message_queue.put(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue message {message.id}: {str(e)}")
            return False
    
    async def _message_routing_loop(self):
        """Main message routing loop"""
        while self.is_running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                
                await self._deliver_message(message)
                self.system_metrics["messages_routed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message routing loop: {str(e)}")
    
    async def _deliver_message(self, message: Message):
        """Deliver message to target agent"""
        start_time = time.time()
        
        try:
            # Find target agent
            if message.recipient_id not in self.agents:
                self.logger.error(f"Target agent {message.recipient_id} not found")
                return
            
            registration = self.agents[message.recipient_id]
            
            if not registration.is_active:
                self.logger.error(f"Target agent {message.recipient_id} is not active")
                return
            
            # Deliver message
            await registration.agent_instance.receive_message(message)
            
            # Store in history
            self.message_history.append(message)
            
            # Keep history bounded
            if len(self.message_history) > 10000:
                self.message_history = self.message_history[-5000:]
            
            # Update response time metrics
            response_time = time.time() - start_time
            self._update_response_time_metric(response_time)
            
        except Exception as e:
            self.logger.error(f"Failed to deliver message {message.id}: {str(e)}")
    
    # =============================================================================
    # Task Distribution
    # =============================================================================
    
    async def distribute_task(self, task: Task, 
                            preferred_agent_type: Optional[str] = None,
                            required_capability: Optional[str] = None) -> bool:
        """Distribute task to the best available agent"""
        try:
            # Add to distribution queue
            priority = task.priority.value
            heapq.heappush(self.task_queue, (priority, time.time(), task))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue task {task.id}: {str(e)}")
            return False
    
    async def _task_distribution_loop(self):
        """Main task distribution loop"""
        while self.is_running:
            try:
                if self.task_queue:
                    # Get highest priority task
                    priority, timestamp, task = heapq.heappop(self.task_queue)
                    
                    # Find best agent for the task
                    agent_id = await self._find_best_agent_for_task(task)
                    
                    if agent_id:
                        await self._assign_task_to_agent(task, agent_id)
                        self.system_metrics["tasks_distributed"] += 1
                    else:
                        # No agent available, requeue with delay
                        heapq.heappush(self.task_queue, 
                                     (priority, time.time() + 5, task))
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                self.logger.error(f"Error in task distribution loop: {str(e)}")
    
    async def _find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best agent to handle a specific task"""
        candidates = []
        
        # Get task requirements
        task_type = task.parameters.get("type")
        required_capability = task.parameters.get("capability")
        preferred_agent_type = task.parameters.get("agent_type")
        
        # Score all agents
        for agent_id, registration in self.agents.items():
            if not registration.is_active:
                continue
            
            score = 0.0
            workload = self.workload_metrics[agent_id]
            
            # Type preference
            if preferred_agent_type and registration.agent_type == preferred_agent_type:
                score += 10.0
            
            # Capability match
            if required_capability and required_capability in registration.capabilities:
                score += 8.0
            
            # Load balancing (lower load = higher score)
            load_penalty = min(workload.load_score, 10.0)
            score += (10.0 - load_penalty)
            
            # Historical performance
            if workload.completed_tasks > 0:
                success_rate = workload.completed_tasks / (
                    workload.completed_tasks + workload.failed_tasks
                )
                score += success_rate * 5.0
            
            if score > 0:
                candidates.append((score, agent_id))
        
        if candidates:
            # Return agent with highest score
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None
    
    async def _assign_task_to_agent(self, task: Task, agent_id: str):
        """Assign task to specific agent"""
        try:
            registration = self.agents[agent_id]
            
            # Update tracking
            self.task_assignments[task.id] = agent_id
            self.active_tasks[task.id] = task
            
            # Update workload metrics
            workload = self.workload_metrics[agent_id]
            workload.pending_tasks += 1
            
            # Assign to agent
            await registration.agent_instance.assign_task(task)
            
            self.logger.info(f"Task {task.id} assigned to agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.id} to {agent_id}: {str(e)}")
    
    # =============================================================================
    # Health Monitoring
    # =============================================================================
    
    async def _health_monitoring_loop(self):
        """Monitor agent health and system status"""
        while self.is_running:
            try:
                # Check each agent
                for agent_id, registration in self.agents.items():
                    await self._check_agent_health(registration)
                
                # Update system health
                await self._update_system_health()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}")
    
    async def _check_agent_health(self, registration: AgentRegistration):
        """Check individual agent health"""
        try:
            # Send ping message
            ping_message = Message(
                sender_id=self.orchestrator_id,
                recipient_id=registration.agent_id,
                message_type="ping",
                content={"timestamp": datetime.now().isoformat()}
            )
            
            # This would need timeout handling in a real implementation
            response = await self._send_ping_and_wait(registration, ping_message)
            
            if response:
                registration.last_heartbeat = datetime.now()
                if not registration.is_active:
                    registration.is_active = True
                    self.logger.info(f"Agent {registration.agent_id} is back online")
            else:
                # Agent not responding
                if registration.is_active:
                    registration.is_active = False
                    self.logger.warning(f"Agent {registration.agent_id} is not responding")
                
        except Exception as e:
            self.logger.error(f"Health check failed for agent {registration.agent_id}: {str(e)}")
    
    async def _send_ping_and_wait(self, registration: AgentRegistration, 
                                  ping_message: Message, timeout: float = 5.0):
        """Send ping and wait for response"""
        try:
            # This is a simplified implementation
            # In a real system, this would use proper request/response handling
            await registration.agent_instance.receive_message(ping_message)
            return True
        except Exception:
            return False
    
    async def _update_system_health(self):
        """Update overall system health metrics"""
        active_agents = sum(1 for reg in self.agents.values() if reg.is_active)
        total_agents = len(self.agents)
        
        if total_agents > 0:
            system_health = active_agents / total_agents * 100
            
            if system_health < 80:
                self.logger.warning(f"System health degraded: {system_health:.1f}%")
            
            # Update metrics
            self.system_metrics["active_agents"] = active_agents
            self.system_metrics["total_agents"] = total_agents
            self.system_metrics["system_health"] = system_health
    
    # =============================================================================
    # Performance Monitoring
    # =============================================================================
    
    async def _metrics_update_loop(self):
        """Update system metrics periodically"""
        while self.is_running:
            try:
                # Update uptime
                if self.start_time:
                    self.system_metrics["uptime"] = time.time() - self.start_time
                
                # Update workload metrics for all agents
                for agent_id in self.agents:
                    await self._update_agent_workload_metrics(agent_id)
                
                # Log system status
                self._log_system_status()
                
                # Sleep for metrics interval
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {str(e)}")
    
    async def _update_agent_workload_metrics(self, agent_id: str):
        """Update workload metrics for a specific agent"""
        try:
            registration = self.agents[agent_id]
            workload = self.workload_metrics[agent_id]
            
            # Get metrics from agent
            metrics_message = Message(
                sender_id=self.orchestrator_id,
                recipient_id=agent_id,
                message_type="metrics_request",
                content={}
            )
            
            # Update basic counters
            agent_metrics = registration.agent_instance.metrics
            workload.completed_tasks = agent_metrics.tasks_completed
            workload.failed_tasks = agent_metrics.tasks_failed
            workload.average_completion_time = agent_metrics.average_response_time
            
            # Count active tasks
            active_count = 1 if registration.agent_instance.current_task else 0
            workload.active_tasks = active_count
            
            # Estimate pending tasks (this would be more sophisticated in practice)
            workload.pending_tasks = registration.agent_instance.task_queue.qsize()
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics for agent {agent_id}: {str(e)}")
    
    def _log_system_status(self):
        """Log current system status"""
        active_agents = sum(1 for reg in self.agents.values() if reg.is_active)
        total_tasks = sum(w.completed_tasks + w.failed_tasks 
                         for w in self.workload_metrics.values())
        
        self.logger.info(
            f"System Status: {active_agents}/{len(self.agents)} agents active, "
            f"{len(self.task_queue)} queued tasks, "
            f"{total_tasks} total tasks processed, "
            f"{self.system_metrics['uptime']:.0f}s uptime"
        )
    
    def _update_response_time_metric(self, response_time: float):
        """Update system-wide response time metric"""
        current_avg = self.system_metrics.get("average_response_time", 0.0)
        messages_count = self.system_metrics.get("messages_routed", 0)
        
        if messages_count == 0:
            self.system_metrics["average_response_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.system_metrics["average_response_time"] = (
                alpha * response_time + (1 - alpha) * current_avg
            )
    
    # =============================================================================
    # Training Support
    # =============================================================================
    
    async def start_training_session(self, session_id: str, 
                                   training_config: Dict[str, Any]) -> bool:
        """Start a system-wide training session"""
        try:
            # Store training session
            self.training_sessions[session_id] = {
                "config": training_config,
                "start_time": datetime.now(),
                "participating_agents": [],
                "status": "starting"
            }
            
            # Notify all agents to start training
            target_agent_types = training_config.get("agent_types", [])
            
            for agent_id, registration in self.agents.items():
                if not target_agent_types or registration.agent_type in target_agent_types:
                    training_message = Message(
                        sender_id=self.orchestrator_id,
                        recipient_id=agent_id,
                        message_type="training_start",
                        content={
                            "session_id": session_id,
                            "config": training_config
                        }
                    )
                    
                    await self.route_message(training_message)
                    self.training_sessions[session_id]["participating_agents"].append(agent_id)
            
            self.training_sessions[session_id]["status"] = "active"
            
            self.logger.info(f"Training session {session_id} started with "
                           f"{len(self.training_sessions[session_id]['participating_agents'])} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start training session {session_id}: {str(e)}")
            return False
    
    async def stop_training_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a training session and collect results"""
        try:
            if session_id not in self.training_sessions:
                return {"error": f"Training session {session_id} not found"}
            
            session = self.training_sessions[session_id]
            
            # Notify agents to stop training
            results = {}
            for agent_id in session["participating_agents"]:
                stop_message = Message(
                    sender_id=self.orchestrator_id,
                    recipient_id=agent_id,
                    message_type="training_stop",
                    content={"session_id": session_id}
                )
                
                await self.route_message(stop_message)
                
                # Collect results (this would be more sophisticated in practice)
                agent = self.agents[agent_id].agent_instance
                results[agent_id] = agent.training_metrics.copy()
            
            # Update session
            session["status"] = "completed"
            session["end_time"] = datetime.now()
            session["results"] = results
            
            self.logger.info(f"Training session {session_id} completed")
            
            return {
                "session_id": session_id,
                "status": "completed",
                "results": results,
                "duration": (session["end_time"] - session["start_time"]).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop training session {session_id}: {str(e)}")
            return {"error": str(e)}
    
    # =============================================================================
    # Public Interface
    # =============================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_agents = sum(1 for reg in self.agents.values() if reg.is_active)
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "is_running": self.is_running,
            "agents": {
                "total": len(self.agents),
                "active": active_agents,
                "by_type": {
                    agent_type: len(agent_ids)
                    for agent_type, agent_ids in self.agent_types.items()
                }
            },
            "tasks": {
                "queued": len(self.task_queue),
                "active": len(self.active_tasks),
                "total_distributed": self.system_metrics["tasks_distributed"]
            },
            "performance": {
                "uptime": self.system_metrics.get("uptime", 0.0),
                "messages_routed": self.system_metrics["messages_routed"],
                "average_response_time": self.system_metrics.get("average_response_time", 0.0)
            },
            "training": {
                "active_sessions": len([s for s in self.training_sessions.values() 
                                     if s["status"] == "active"])
            }
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific agent"""
        if agent_id not in self.agents:
            return None
        
        registration = self.agents[agent_id]
        workload = self.workload_metrics[agent_id]
        
        return {
            "agent_id": agent_id,
            "agent_type": registration.agent_type,
            "name": registration.name,
            "is_active": registration.is_active,
            "state": registration.agent_instance.state.value,
            "capabilities": registration.capabilities,
            "workload": {
                "pending_tasks": workload.pending_tasks,
                "active_tasks": workload.active_tasks,
                "completed_tasks": workload.completed_tasks,
                "failed_tasks": workload.failed_tasks,
                "load_score": workload.load_score
            },
            "last_heartbeat": registration.last_heartbeat.isoformat(),
            "registered_at": registration.registered_at.isoformat()
        }