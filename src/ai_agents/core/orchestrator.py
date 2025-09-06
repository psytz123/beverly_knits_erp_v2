#!/usr/bin/env python3
"""
Central Orchestrator for eFab AI Agent System
Master coordinator that manages all other agents and system-wide operations
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority, AgentStatus
from .state_manager import SystemState, CustomerProfile, ImplementationPhase, system_state

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestratorStatus(Enum):
    """Orchestrator-specific status states"""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    LOAD_BALANCING = "LOAD_BALANCING"
    EMERGENCY = "EMERGENCY"
    MAINTENANCE = "MAINTENANCE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class TaskAssignment:
    """Task assignment to agents"""
    task_id: str
    customer_id: str
    agent_id: str
    task_type: str
    priority: Priority
    assigned_at: datetime
    estimated_completion: datetime
    dependencies: List[str] = field(default_factory=list)
    status: str = "ASSIGNED"  # ASSIGNED, IN_PROGRESS, COMPLETED, FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "customer_id": self.customer_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "assigned_at": self.assigned_at.isoformat(),
            "estimated_completion": self.estimated_completion.isoformat(),
            "dependencies": self.dependencies,
            "status": self.status
        }


class CentralOrchestrator(BaseAgent):
    """
    Central Orchestrator Agent - Master coordinator for eFab AI Agent System
    
    Responsibilities:
    - Agent lifecycle management and health monitoring
    - Task routing and load balancing
    - Implementation orchestration and timeline management
    - System-wide decision making and conflict resolution
    - Performance monitoring and optimization
    - Emergency response and recovery procedures
    """
    
    def __init__(self):
        """Initialize Central Orchestrator"""
        super().__init__(
            agent_id="orchestrator",
            agent_name="Central Orchestrator",
            agent_description="Master coordinator managing all eFab AI agents and system operations"
        )
        
        # Orchestrator-specific status
        self.orchestrator_status = OrchestratorStatus.INITIALIZING
        
        # Agent management
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_workloads: Dict[str, int] = {}
        self.agent_health_scores: Dict[str, float] = {}
        
        # Task management
        self.active_tasks: Dict[str, TaskAssignment] = {}
        self.task_queue: List[TaskAssignment] = []
        self.completed_tasks: List[TaskAssignment] = []
        
        # Implementation orchestration
        self.customer_workflows: Dict[str, Dict[str, Any]] = {}
        self.implementation_timelines: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.system_performance: Dict[str, Any] = {}
        self.alert_thresholds: Dict[str, float] = {
            "agent_response_time_ms": 1000,
            "agent_error_rate": 0.05,
            "system_load": 0.8,
            "implementation_delay_hours": 24
        }
        
        # Communication channels
        self.message_router: Optional[Callable] = None
        self.notification_handlers: Dict[str, Callable] = {}
        
        self.logger.info("Central Orchestrator initialized")
    
    def _initialize(self):
        """Initialize orchestrator-specific components"""
        # Register orchestrator capabilities
        self.register_capability(AgentCapability(
            name="agent_management",
            description="Register, monitor, and manage AI agents",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["register", "deregister", "status", "health_check"]},
                    "agent_id": {"type": "string"},
                    "agent_info": {"type": "object"}
                }
            },
            output_schema={
                "type": "object", 
                "properties": {
                    "success": {"type": "boolean"},
                    "agent_status": {"type": "object"},
                    "message": {"type": "string"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="implementation_orchestration",
            description="Orchestrate customer implementation workflows",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "implementation_phase": {"type": "string"},
                    "action": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "workflow_status": {"type": "object"},
                    "next_actions": {"type": "array"},
                    "timeline_update": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="task_coordination",
            description="Coordinate task assignment and execution across agents",
            input_schema={
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "customer_id": {"type": "string"}, 
                    "priority": {"type": "string"},
                    "requirements": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "assigned_agent": {"type": "string"},
                    "estimated_completion": {"type": "string"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_notification)
        
        # Initialize system state connection
        self.system_state = system_state
        
        # Start background tasks
        asyncio.create_task(self._agent_health_monitor())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._performance_monitor())
        
        self.orchestrator_status = OrchestratorStatus.ACTIVE
    
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """
        Register a new agent with the orchestrator
        
        Args:
            agent_id: Unique identifier for the agent
            agent_info: Agent information including capabilities
            
        Returns:
            True if registration successful
        """
        try:
            # Validate agent info
            required_fields = ["agent_name", "agent_description", "capabilities"]
            if not all(field in agent_info for field in required_fields):
                self.logger.error(f"Missing required fields for agent {agent_id}")
                return False
            
            # Register agent
            self.registered_agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "status": AgentStatus.READY.value,
                "message_count": 0,
                "error_count": 0
            }
            
            # Register capabilities
            capabilities = [AgentCapability(**cap) for cap in agent_info.get("capabilities", [])]
            self.agent_capabilities[agent_id] = capabilities
            
            # Initialize workload tracking
            self.agent_workloads[agent_id] = 0
            self.agent_health_scores[agent_id] = 1.0
            
            # Register with system state
            self.system_state.register_agent(agent_id, agent_info)
            
            self.logger.info(f"Agent registered: {agent_id} ({agent_info['agent_name']})")
            
            # Send welcome message to agent
            welcome_message = AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=agent_id,
                message_type=MessageType.NOTIFICATION,
                payload={
                    "message": "REGISTRATION_SUCCESSFUL",
                    "orchestrator_id": self.agent_id,
                    "system_config": self.system_state.system_config
                }
            )
            
            if self.message_router:
                await self.message_router(welcome_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent"""
        try:
            if agent_id in self.registered_agents:
                # Cancel any assigned tasks
                await self._reassign_agent_tasks(agent_id)
                
                # Remove from tracking
                del self.registered_agents[agent_id]
                del self.agent_capabilities[agent_id]
                del self.agent_workloads[agent_id]
                del self.agent_health_scores[agent_id]
                
                self.logger.info(f"Agent deregistered: {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def assign_task(
        self, 
        task_type: str, 
        customer_id: str, 
        requirements: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> Optional[str]:
        """
        Assign task to most suitable agent
        
        Args:
            task_type: Type of task to assign
            customer_id: Customer for whom task is being performed
            requirements: Task-specific requirements
            priority: Task priority
            
        Returns:
            Task ID if assignment successful
        """
        try:
            # Find suitable agents
            suitable_agents = self._find_suitable_agents(task_type, requirements)
            
            if not suitable_agents:
                self.logger.warning(f"No suitable agents found for task type: {task_type}")
                return None
            
            # Select best agent based on workload and capability match
            selected_agent = self._select_best_agent(suitable_agents, requirements)
            
            # Create task assignment
            task_id = str(uuid.uuid4())
            estimated_completion = datetime.now() + timedelta(hours=1)  # Default estimate
            
            task = TaskAssignment(
                task_id=task_id,
                customer_id=customer_id,
                agent_id=selected_agent,
                task_type=task_type,
                priority=priority,
                assigned_at=datetime.now(),
                estimated_completion=estimated_completion
            )
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            self.agent_workloads[selected_agent] += 1
            
            # Send task to agent
            task_message = AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=selected_agent,
                message_type=MessageType.REQUEST,
                payload={
                    "action": "EXECUTE_TASK",
                    "task_id": task_id,
                    "task_type": task_type,
                    "customer_id": customer_id,
                    "requirements": requirements,
                    "priority": priority.value
                },
                priority=priority
            )
            
            if self.message_router:
                await self.message_router(task_message)
            
            self.logger.info(f"Task {task_id} assigned to agent {selected_agent}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to assign task: {str(e)}")
            return None
    
    def _find_suitable_agents(self, task_type: str, requirements: Dict[str, Any]) -> List[str]:
        """Find agents capable of handling specific task type"""
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            # Check if agent is available
            if (agent_id not in self.registered_agents or 
                self.registered_agents[agent_id]["status"] != AgentStatus.READY.value):
                continue
            
            # Check if agent has required capability
            for capability in capabilities:
                if capability.name == task_type or task_type in capability.description.lower():
                    suitable_agents.append(agent_id)
                    break
        
        return suitable_agents
    
    def _select_best_agent(self, suitable_agents: List[str], requirements: Dict[str, Any]) -> str:
        """Select best agent based on workload and capability match"""
        if len(suitable_agents) == 1:
            return suitable_agents[0]
        
        # Score agents based on workload, health, and capability match
        agent_scores = []
        
        for agent_id in suitable_agents:
            # Lower workload is better
            workload_score = 1.0 / (1.0 + self.agent_workloads.get(agent_id, 0))
            
            # Health score (0-1, higher is better)
            health_score = self.agent_health_scores.get(agent_id, 0.5)
            
            # Combined score
            total_score = (workload_score * 0.4) + (health_score * 0.6)
            
            agent_scores.append((agent_id, total_score))
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        return agent_scores[0][0]
    
    async def orchestrate_implementation(self, customer_id: str) -> Dict[str, Any]:
        """
        Orchestrate complete customer implementation workflow
        
        Args:
            customer_id: Customer to orchestrate implementation for
            
        Returns:
            Implementation workflow status and next steps
        """
        try:
            # Get customer profile and current phase
            customer_profile = self.system_state.get_customer_profile(customer_id)
            if not customer_profile:
                return {"error": "Customer not found"}
            
            current_phase = self.system_state.implementation_phases.get(customer_id)
            if not current_phase:
                current_phase = ImplementationPhase.PRE_ASSESSMENT
                self.system_state.update_implementation_phase(customer_id, current_phase)
            
            # Get implementation plan
            implementation_plan = self.system_state.get_implementation_plan(customer_id)
            
            # Orchestrate current phase
            workflow_status = await self._orchestrate_phase(customer_id, current_phase)
            
            # Determine next actions
            next_actions = self._determine_next_actions(customer_id, current_phase, workflow_status)
            
            # Update implementation timeline
            timeline_update = self._update_implementation_timeline(customer_id, workflow_status)
            
            return {
                "customer_id": customer_id,
                "current_phase": current_phase.value,
                "workflow_status": workflow_status,
                "next_actions": next_actions,
                "timeline_update": timeline_update,
                "estimated_completion": implementation_plan.estimated_duration_weeks if implementation_plan else 6
            }
            
        except Exception as e:
            self.logger.error(f"Failed to orchestrate implementation for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _orchestrate_phase(self, customer_id: str, phase: ImplementationPhase) -> Dict[str, Any]:
        """Orchestrate specific implementation phase"""
        phase_orchestration = {
            ImplementationPhase.PRE_ASSESSMENT: self._orchestrate_pre_assessment,
            ImplementationPhase.DISCOVERY: self._orchestrate_discovery,
            ImplementationPhase.CONFIGURATION: self._orchestrate_configuration,
            ImplementationPhase.DATA_MIGRATION: self._orchestrate_data_migration,
            ImplementationPhase.TRAINING: self._orchestrate_training,
            ImplementationPhase.TESTING: self._orchestrate_testing,
            ImplementationPhase.GO_LIVE: self._orchestrate_go_live,
            ImplementationPhase.STABILIZATION: self._orchestrate_stabilization,
            ImplementationPhase.OPTIMIZATION: self._orchestrate_optimization
        }
        
        orchestrator_func = phase_orchestration.get(phase)
        if orchestrator_func:
            return await orchestrator_func(customer_id)
        else:
            return {"status": "PHASE_NOT_IMPLEMENTED", "phase": phase.value}
    
    async def _orchestrate_pre_assessment(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate pre-assessment phase"""
        # Assign customer assessment task to Implementation Project Manager Agent
        task_id = await self.assign_task(
            task_type="customer_assessment",
            customer_id=customer_id,
            requirements={"phase": "pre_assessment"},
            priority=Priority.HIGH
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 8
        }
    
    async def _orchestrate_discovery(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate discovery phase"""
        tasks = []
        
        # Legacy system analysis
        task_id = await self.assign_task(
            task_type="legacy_system_analysis",
            customer_id=customer_id,
            requirements={"phase": "discovery"},
            priority=Priority.HIGH
        )
        if task_id:
            tasks.append(task_id)
        
        # Business process mapping
        task_id = await self.assign_task(
            task_type="process_mapping",
            customer_id=customer_id,
            requirements={"phase": "discovery"},
            priority=Priority.MEDIUM
        )
        if task_id:
            tasks.append(task_id)
        
        return {
            "status": "IN_PROGRESS",
            "tasks": tasks,
            "estimated_completion_hours": 40
        }
    
    async def _orchestrate_configuration(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate configuration phase"""
        # System configuration generation
        task_id = await self.assign_task(
            task_type="system_configuration",
            customer_id=customer_id,
            requirements={"phase": "configuration"},
            priority=Priority.HIGH
        )
        
        return {
            "status": "IN_PROGRESS", 
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 24
        }
    
    async def _orchestrate_data_migration(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate data migration phase"""
        # Data migration and validation
        task_id = await self.assign_task(
            task_type="data_migration",
            customer_id=customer_id,
            requirements={"phase": "data_migration"},
            priority=Priority.CRITICAL
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 32
        }
    
    async def _orchestrate_training(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate training phase"""
        # User training and documentation
        task_id = await self.assign_task(
            task_type="user_training", 
            customer_id=customer_id,
            requirements={"phase": "training"},
            priority=Priority.MEDIUM
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 24
        }
    
    async def _orchestrate_testing(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate testing phase"""
        # System testing and validation
        task_id = await self.assign_task(
            task_type="system_testing",
            customer_id=customer_id,
            requirements={"phase": "testing"},
            priority=Priority.HIGH
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 32
        }
    
    async def _orchestrate_go_live(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate go-live phase"""
        # Production deployment and monitoring
        task_id = await self.assign_task(
            task_type="production_deployment",
            customer_id=customer_id,
            requirements={"phase": "go_live"},
            priority=Priority.CRITICAL
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 16
        }
    
    async def _orchestrate_stabilization(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate stabilization phase"""
        # Post go-live support and optimization
        task_id = await self.assign_task(
            task_type="system_stabilization",
            customer_id=customer_id,
            requirements={"phase": "stabilization"},
            priority=Priority.HIGH
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 40
        }
    
    async def _orchestrate_optimization(self, customer_id: str) -> Dict[str, Any]:
        """Orchestrate optimization phase"""
        # Performance optimization and feature enhancement
        task_id = await self.assign_task(
            task_type="performance_optimization",
            customer_id=customer_id,
            requirements={"phase": "optimization"},
            priority=Priority.MEDIUM
        )
        
        return {
            "status": "IN_PROGRESS",
            "tasks": [task_id] if task_id else [],
            "estimated_completion_hours": 24
        }
    
    def _determine_next_actions(
        self, 
        customer_id: str, 
        current_phase: ImplementationPhase,
        workflow_status: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Determine next actions for implementation"""
        next_actions = []
        
        if workflow_status.get("status") == "COMPLETED":
            # Move to next phase
            next_phase = self._get_next_phase(current_phase)
            if next_phase:
                next_actions.append({
                    "action": "ADVANCE_PHASE",
                    "target_phase": next_phase.value,
                    "estimated_start": (datetime.now() + timedelta(hours=2)).isoformat()
                })
        
        elif workflow_status.get("status") == "IN_PROGRESS":
            # Monitor current tasks
            next_actions.append({
                "action": "MONITOR_PROGRESS",
                "tasks": workflow_status.get("tasks", []),
                "check_in_hours": 4
            })
        
        return next_actions
    
    def _get_next_phase(self, current_phase: ImplementationPhase) -> Optional[ImplementationPhase]:
        """Get next implementation phase"""
        phase_sequence = [
            ImplementationPhase.PRE_ASSESSMENT,
            ImplementationPhase.DISCOVERY,
            ImplementationPhase.CONFIGURATION,
            ImplementationPhase.DATA_MIGRATION,
            ImplementationPhase.TRAINING,
            ImplementationPhase.TESTING,
            ImplementationPhase.GO_LIVE,
            ImplementationPhase.STABILIZATION,
            ImplementationPhase.OPTIMIZATION,
            ImplementationPhase.COMPLETED
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _update_implementation_timeline(self, customer_id: str, workflow_status: Dict[str, Any]) -> Dict[str, Any]:
        """Update implementation timeline based on progress"""
        timeline = self.implementation_timelines.get(customer_id, {})
        
        # Update current phase status
        current_phase = self.system_state.implementation_phases.get(customer_id)
        if current_phase:
            timeline[current_phase.value] = {
                "status": workflow_status.get("status"),
                "updated_at": datetime.now().isoformat(),
                "estimated_completion": workflow_status.get("estimated_completion_hours")
            }
        
        self.implementation_timelines[customer_id] = timeline
        return timeline
    
    async def _reassign_agent_tasks(self, failed_agent_id: str):
        """Reassign tasks from failed agent to other agents"""
        tasks_to_reassign = [
            task for task in self.active_tasks.values()
            if task.agent_id == failed_agent_id and task.status in ["ASSIGNED", "IN_PROGRESS"]
        ]
        
        for task in tasks_to_reassign:
            # Remove from failed agent
            if failed_agent_id in self.agent_workloads:
                self.agent_workloads[failed_agent_id] -= 1
            
            # Find new agent
            suitable_agents = self._find_suitable_agents(task.task_type, {})
            if suitable_agents:
                new_agent = self._select_best_agent(suitable_agents, {})
                
                # Reassign task
                task.agent_id = new_agent
                task.assigned_at = datetime.now()
                task.status = "ASSIGNED"
                
                self.agent_workloads[new_agent] += 1
                
                # Notify new agent
                task_message = AgentMessage(
                    agent_id=self.agent_id,
                    target_agent_id=new_agent,
                    message_type=MessageType.REQUEST,
                    payload={
                        "action": "EXECUTE_TASK",
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "customer_id": task.customer_id,
                        "priority": task.priority.value,
                        "reassigned": True
                    },
                    priority=task.priority
                )
                
                if self.message_router:
                    await self.message_router(task_message)
                
                self.logger.info(f"Task {task.task_id} reassigned from {failed_agent_id} to {new_agent}")
    
    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle incoming requests"""
        action = message.payload.get("action")
        
        if action == "REGISTER_AGENT":
            agent_id = message.payload.get("agent_id")
            agent_info = message.payload.get("agent_info", {})
            
            success = await self.register_agent(agent_id, agent_info)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "success": success,
                    "message": "Agent registered successfully" if success else "Registration failed"
                },
                correlation_id=message.correlation_id
            )
        
        elif action == "ASSIGN_TASK":
            task_type = message.payload.get("task_type")
            customer_id = message.payload.get("customer_id")
            requirements = message.payload.get("requirements", {})
            priority = Priority(message.payload.get("priority", "MEDIUM"))
            
            task_id = await self.assign_task(task_type, customer_id, requirements, priority)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "task_id": task_id,
                    "success": task_id is not None,
                    "message": "Task assigned successfully" if task_id else "Task assignment failed"
                },
                correlation_id=message.correlation_id
            )
        
        elif action == "ORCHESTRATE_IMPLEMENTATION":
            customer_id = message.payload.get("customer_id")
            result = await self.orchestrate_implementation(customer_id)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
        
        elif action == "GET_SYSTEM_STATUS":
            status = self.system_state.get_system_status()
            status.update({
                "orchestrator_status": self.orchestrator_status.value,
                "registered_agents": len(self.registered_agents),
                "active_tasks": len(self.active_tasks),
                "task_queue_length": len(self.task_queue)
            })
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=status,
                correlation_id=message.correlation_id
            )
        
        else:
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": "UNSUPPORTED_ACTION", "action": action},
                correlation_id=message.correlation_id
            )
    
    async def _handle_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming notifications"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "TASK_COMPLETED":
            task_id = message.payload.get("task_id")
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = "COMPLETED"
                
                # Update agent workload
                self.agent_workloads[task.agent_id] -= 1
                
                # Move to completed tasks
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
                
                self.logger.info(f"Task {task_id} completed by agent {task.agent_id}")
        
        elif notification_type == "TASK_FAILED":
            task_id = message.payload.get("task_id")
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = "FAILED"
                
                # Update agent workload
                self.agent_workloads[task.agent_id] -= 1
                
                # Try to reassign if possible
                error_reason = message.payload.get("error_reason", "Unknown error")
                self.logger.error(f"Task {task_id} failed: {error_reason}")
                
                # Attempt reassignment for critical tasks
                if task.priority == Priority.CRITICAL:
                    await self._reassign_agent_tasks(task.agent_id)
        
        elif notification_type == "AGENT_HEARTBEAT":
            agent_id = message.agent_id
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id]["last_heartbeat"] = datetime.now()
                self.system_state.update_agent_heartbeat(agent_id, message.payload)
        
        return None
    
    async def _agent_health_monitor(self):
        """Monitor agent health and availability"""
        while self.orchestrator_status != OrchestratorStatus.SHUTDOWN:
            try:
                current_time = datetime.now()
                
                for agent_id, agent_info in self.registered_agents.items():
                    last_heartbeat = agent_info.get("last_heartbeat")
                    if last_heartbeat:
                        time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
                        
                        # Calculate health score
                        if time_since_heartbeat < 30:
                            health_score = 1.0
                        elif time_since_heartbeat < 120:
                            health_score = 0.8
                        elif time_since_heartbeat < 300:
                            health_score = 0.5
                        else:
                            health_score = 0.0
                            
                        self.agent_health_scores[agent_id] = health_score
                        
                        # Handle unhealthy agents
                        if health_score == 0.0:
                            self.logger.warning(f"Agent {agent_id} appears unhealthy - no heartbeat for {time_since_heartbeat}s")
                            await self._reassign_agent_tasks(agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent health monitor: {str(e)}")
                await asyncio.sleep(30)
    
    async def _task_scheduler(self):
        """Schedule and manage task execution"""
        while self.orchestrator_status != OrchestratorStatus.SHUTDOWN:
            try:
                # Process task queue
                if self.task_queue:
                    # Sort by priority and age
                    self.task_queue.sort(
                        key=lambda t: (t.priority.score, t.assigned_at),
                        reverse=True
                    )
                    
                    # Process high-priority tasks first
                    for task in self.task_queue[:]:
                        if task.task_id not in self.active_tasks:
                            # Find available agent
                            suitable_agents = self._find_suitable_agents(task.task_type, {})
                            available_agents = [
                                agent_id for agent_id in suitable_agents
                                if self.agent_workloads.get(agent_id, 0) < 5  # Max 5 concurrent tasks
                            ]
                            
                            if available_agents:
                                selected_agent = self._select_best_agent(available_agents, {})
                                task.agent_id = selected_agent
                                
                                # Move to active tasks
                                self.active_tasks[task.task_id] = task
                                self.task_queue.remove(task)
                                self.agent_workloads[selected_agent] += 1
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {str(e)}")
                await asyncio.sleep(10)
    
    async def _performance_monitor(self):
        """Monitor system performance and trigger alerts"""
        while self.orchestrator_status != OrchestratorStatus.SHUTDOWN:
            try:
                # Calculate system performance metrics
                total_agents = len(self.registered_agents)
                healthy_agents = sum(
                    1 for score in self.agent_health_scores.values() 
                    if score > 0.5
                )
                
                system_health = healthy_agents / total_agents if total_agents > 0 else 0
                
                # Update system metrics
                self.system_state.update_metrics({
                    "system_uptime_percentage": 99.9,  # Calculate actual uptime
                    "average_response_time_ms": sum(
                        agent["message_count"] for agent in self.registered_agents.values()
                    ) / len(self.registered_agents) if self.registered_agents else 0
                })
                
                # Check for alerts
                if system_health < 0.7:
                    self.logger.warning(f"System health degraded: {system_health:.2f}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
                await asyncio.sleep(60)
    
    def set_message_router(self, router: Callable):
        """Set message router for agent communication"""
        self.message_router = router
        self.logger.info("Message router configured")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            "orchestrator_id": self.agent_id,
            "status": self.orchestrator_status.value,
            "registered_agents": len(self.registered_agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "task_queue_length": len(self.task_queue),
            "system_health": {
                "healthy_agents": sum(1 for score in self.agent_health_scores.values() if score > 0.5),
                "total_agents": len(self.registered_agents),
                "average_health_score": sum(self.agent_health_scores.values()) / len(self.agent_health_scores) if self.agent_health_scores else 0
            }
        }


# Export main component
__all__ = ["CentralOrchestrator", "TaskAssignment", "OrchestratorStatus"]