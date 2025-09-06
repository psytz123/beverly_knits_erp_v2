"""
Base Agent Class
================

Foundation class for all eFab AI agents providing core functionality including
communication protocols, state management, error handling, and training interfaces.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum


class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing" 
    TRAINING = "training"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class Message:
    """Inter-agent message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Task:
    """Agent task representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    assigned_agent: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Agent performance tracking"""
    messages_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    uptime: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Base class for all eFab AI agents
    
    Provides core functionality:
    - Message handling and routing
    - Task management and coordination
    - State management and monitoring
    - Error handling and recovery
    - Performance metrics tracking
    - Training interface compliance
    """
    
    def __init__(self, agent_id: str, agent_type: str, name: str = ""):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name or f"{agent_type}_{agent_id}"
        
        # Agent state
        self.state = AgentState.INITIALIZING
        self.capabilities: List[str] = []
        self.configuration: Dict[str, Any] = {}
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.response_callbacks: Dict[str, Callable] = {}
        
        # Task management
        self.current_task: Optional[Task] = None
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: List[Task] = []
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        
        # Training interface
        self.training_mode = False
        self.training_session_id: Optional[str] = None
        self.training_metrics: Dict[str, Any] = {}
        
        # Logging
        self.logger = logging.getLogger(f"eFab.Agent.{self.name}")
        
        # Register core message handlers
        self._register_core_handlers()
        
        self.logger.info(f"Agent {self.name} ({self.agent_type}) initialized")
    
    def _register_core_handlers(self):
        """Register core message handlers"""
        self.message_handlers.update({
            "ping": self._handle_ping,
            "task_assignment": self._handle_task_assignment,
            "task_status_request": self._handle_task_status_request,
            "configuration_update": self._handle_configuration_update,
            "shutdown": self._handle_shutdown,
            "training_start": self._handle_training_start,
            "training_stop": self._handle_training_stop,
            "metrics_request": self._handle_metrics_request
        })
    
    # =============================================================================
    # Core Agent Lifecycle
    # =============================================================================
    
    async def start(self):
        """Start the agent"""
        try:
            await self.initialize()
            self.state = AgentState.IDLE
            
            # Start core processing loops
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._task_processing_loop())
            asyncio.create_task(self._metrics_update_loop())
            
            self.logger.info(f"Agent {self.name} started successfully")
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to start agent {self.name}: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the agent gracefully"""
        self.logger.info(f"Stopping agent {self.name}")
        
        # Complete current task if possible
        if self.current_task and self.current_task.status == "in_progress":
            try:
                await asyncio.wait_for(self._complete_current_task(), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.warning("Failed to complete current task within timeout")
        
        self.state = AgentState.SHUTDOWN
        await self.cleanup()
        
        self.logger.info(f"Agent {self.name} stopped")
    
    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up agent-specific resources"""
        pass
    
    # =============================================================================
    # Message Handling
    # =============================================================================
    
    async def send_message(self, recipient_id: str, message_type: str, 
                          content: Dict[str, Any], 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          timeout_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Send message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            priority=priority,
            timeout=datetime.now() + timedelta(seconds=timeout_seconds)
        )
        
        try:
            # This would integrate with the agent orchestrator
            response = await self._send_message_via_orchestrator(message)
            self.metrics.messages_processed += 1
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
            return None
    
    async def _send_message_via_orchestrator(self, message: Message) -> Optional[Dict[str, Any]]:
        """Send message through the orchestrator (placeholder)"""
        # This will be implemented when orchestrator is created
        pass
    
    async def receive_message(self, message: Message):
        """Receive and queue message for processing"""
        await self.message_queue.put(message)
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
    
    async def _process_message(self, message: Message):
        """Process individual message"""
        start_time = time.time()
        
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                response = await handler(message)
                
                # Send response if needed
                if response and message.sender_id:
                    await self.send_message(
                        message.sender_id,
                        f"{message.message_type}_response",
                        response
                    )
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {str(e)}")
            
            # Send error response
            if message.sender_id:
                await self.send_message(
                    message.sender_id,
                    "error",
                    {"error": str(e), "original_message_id": message.id}
                )
        finally:
            # Update response time metrics
            response_time = time.time() - start_time
            self._update_response_time_metric(response_time)
    
    # =============================================================================
    # Task Management
    # =============================================================================
    
    async def assign_task(self, task: Task):
        """Assign task to this agent"""
        task.assigned_agent = self.agent_id
        await self.task_queue.put(task)
        self.logger.info(f"Task {task.id} assigned: {task.name}")
    
    async def _task_processing_loop(self):
        """Main task processing loop"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # Get next task
                task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {str(e)}")
    
    async def _process_task(self, task: Task):
        """Process individual task"""
        self.current_task = task
        task.status = "in_progress"
        task.started_at = datetime.now()
        
        self.state = AgentState.PROCESSING
        
        try:
            # Execute the task
            result = await self.execute_task(task)
            
            # Mark as completed
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            self.completed_tasks.append(task)
            self.metrics.tasks_completed += 1
            
            self.logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error_message = str(e)
            
            self.completed_tasks.append(task)
            self.metrics.tasks_failed += 1
            
            self.logger.error(f"Task {task.id} failed: {str(e)}")
            
        finally:
            self.current_task = None
            self.state = AgentState.IDLE
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute agent-specific task logic"""
        pass
    
    # =============================================================================
    # Core Message Handlers
    # =============================================================================
    
    async def _handle_ping(self, message: Message) -> Dict[str, Any]:
        """Handle ping message"""
        return {
            "status": "ok",
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_task_assignment(self, message: Message) -> Dict[str, Any]:
        """Handle task assignment"""
        task_data = message.content.get("task", {})
        
        task = Task(
            id=task_data.get("id", str(uuid.uuid4())),
            name=task_data.get("name", ""),
            description=task_data.get("description", ""),
            parameters=task_data.get("parameters", {}),
            priority=MessagePriority(task_data.get("priority", 3))
        )
        
        await self.assign_task(task)
        
        return {
            "status": "accepted",
            "task_id": task.id,
            "estimated_completion": self._estimate_completion_time(task)
        }
    
    async def _handle_task_status_request(self, message: Message) -> Dict[str, Any]:
        """Handle task status request"""
        task_id = message.content.get("task_id")
        
        if self.current_task and self.current_task.id == task_id:
            task = self.current_task
        else:
            task = next((t for t in self.completed_tasks if t.id == task_id), None)
        
        if not task:
            return {"error": f"Task {task_id} not found"}
        
        return {
            "task_id": task.id,
            "status": task.status,
            "progress": self._get_task_progress(task),
            "estimated_completion": self._estimate_completion_time(task)
        }
    
    async def _handle_configuration_update(self, message: Message) -> Dict[str, Any]:
        """Handle configuration update"""
        new_config = message.content.get("configuration", {})
        self.configuration.update(new_config)
        
        # Apply configuration changes
        await self._apply_configuration_changes(new_config)
        
        return {"status": "configuration_updated"}
    
    async def _handle_shutdown(self, message: Message) -> Dict[str, Any]:
        """Handle shutdown request"""
        asyncio.create_task(self.stop())
        return {"status": "shutdown_initiated"}
    
    async def _handle_training_start(self, message: Message) -> Dict[str, Any]:
        """Handle training session start"""
        self.training_mode = True
        self.training_session_id = message.content.get("session_id")
        self.training_metrics = {}
        
        return {"status": "training_started", "session_id": self.training_session_id}
    
    async def _handle_training_stop(self, message: Message) -> Dict[str, Any]:
        """Handle training session stop"""
        self.training_mode = False
        session_id = self.training_session_id
        self.training_session_id = None
        
        return {
            "status": "training_stopped", 
            "session_id": session_id,
            "metrics": self.training_metrics
        }
    
    async def _handle_metrics_request(self, message: Message) -> Dict[str, Any]:
        """Handle metrics request"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "average_response_time": self.metrics.average_response_time,
                "uptime": time.time() - self.start_time,
                "error_rate": self.metrics.error_rate,
                "current_state": self.state.value,
                "training_mode": self.training_mode
            }
        }
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    def _estimate_completion_time(self, task: Task) -> Optional[datetime]:
        """Estimate task completion time"""
        # Basic implementation - can be overridden by specific agents
        if task.status == "completed":
            return task.completed_at
        elif task.status == "in_progress":
            # Estimate based on average task time
            avg_time = 300  # 5 minutes default
            return datetime.now() + timedelta(seconds=avg_time)
        else:
            return None
    
    def _get_task_progress(self, task: Task) -> float:
        """Get task progress percentage"""
        if task.status == "completed":
            return 100.0
        elif task.status == "in_progress":
            # Basic time-based progress estimation
            if task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                estimated_total = 300  # 5 minutes default
                return min(elapsed / estimated_total * 100, 95.0)
        return 0.0
    
    async def _apply_configuration_changes(self, config: Dict[str, Any]):
        """Apply configuration changes"""
        # Override in specific agents
        pass
    
    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""
        if self.metrics.messages_processed == 0:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
    
    async def _metrics_update_loop(self):
        """Update metrics periodically"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # Update uptime
                self.metrics.uptime = time.time() - self.start_time
                
                # Update error rate
                total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
                if total_tasks > 0:
                    self.metrics.error_rate = self.metrics.tasks_failed / total_tasks * 100
                
                self.metrics.last_updated = datetime.now()
                
                # Sleep for 60 seconds
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {str(e)}")
    
    async def _complete_current_task(self):
        """Complete current task gracefully"""
        if self.current_task:
            # Try to complete quickly
            pass
    
    # =============================================================================
    # Training Interface
    # =============================================================================
    
    def record_training_metric(self, metric_name: str, value: Any):
        """Record training metric"""
        if self.training_mode:
            self.training_metrics[metric_name] = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
    
    async def training_scenario_completed(self, scenario_id: str, 
                                        results: Dict[str, Any]):
        """Called when a training scenario is completed"""
        if self.training_mode and self.training_session_id:
            self.record_training_metric(f"scenario_{scenario_id}", results)
            
            # Notify training orchestrator
            await self.send_message(
                "training_orchestrator",
                "scenario_completed",
                {
                    "agent_id": self.agent_id,
                    "session_id": self.training_session_id,
                    "scenario_id": scenario_id,
                    "results": results
                }
            )