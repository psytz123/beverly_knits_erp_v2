#!/usr/bin/env python3
"""
Agent Factory for eFab AI Agent System
Dynamic agent creation, lifecycle management, and deployment orchestration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import importlib
import inspect
import json
import uuid

from ..core.agent_base import BaseAgent, AgentCapability
from ..core.state_manager import CustomerProfile, IndustryType, system_state
from ..communication.message_router import MessageRouter

# Setup logging
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent type categories"""
    CORE = "CORE"                           # Core infrastructure agents
    IMPLEMENTATION = "IMPLEMENTATION"       # Implementation workflow agents
    INDUSTRY = "INDUSTRY"                   # Industry-specific agents
    OPTIMIZATION = "OPTIMIZATION"           # Performance and optimization agents
    LEARNING = "LEARNING"                   # Knowledge and learning agents
    MONITORING = "MONITORING"               # Monitoring and alerting agents
    INTEGRATION = "INTEGRATION"             # External system integration agents


class DeploymentStrategy(Enum):
    """Agent deployment strategies"""
    IMMEDIATE = "IMMEDIATE"                 # Deploy immediately
    LAZY = "LAZY"                          # Deploy when first needed
    SCHEDULED = "SCHEDULED"                # Deploy at scheduled time
    CONDITIONAL = "CONDITIONAL"            # Deploy based on conditions
    CUSTOMER_SPECIFIC = "CUSTOMER_SPECIFIC" # Deploy for specific customers


class AgentTemplate:
    """Agent template for factory creation"""
    
    def __init__(
        self,
        agent_class: Type[BaseAgent],
        agent_type: AgentType,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.LAZY,
        resource_requirements: Dict[str, Any] = None,
        dependencies: List[str] = None,
        configuration_template: Dict[str, Any] = None
    ):
        self.agent_class = agent_class
        self.agent_type = agent_type
        self.deployment_strategy = deployment_strategy
        self.resource_requirements = resource_requirements or {}
        self.dependencies = dependencies or []
        self.configuration_template = configuration_template or {}
        self.created_instances: Dict[str, BaseAgent] = {}
        
    def can_create_instance(self, context: Dict[str, Any]) -> bool:
        """Check if agent instance can be created in given context"""
        # Check dependencies
        for dep in self.dependencies:
            if dep not in context.get("available_agents", []):
                return False
        
        # Check resource requirements
        available_resources = context.get("available_resources", {})
        for resource, requirement in self.resource_requirements.items():
            if available_resources.get(resource, 0) < requirement:
                return False
        
        return True
    
    async def create_instance(
        self, 
        agent_id: str, 
        configuration: Dict[str, Any] = None
    ) -> BaseAgent:
        """Create new agent instance"""
        if agent_id in self.created_instances:
            return self.created_instances[agent_id]
        
        # Merge configuration with template
        final_config = {**self.configuration_template}
        if configuration:
            final_config.update(configuration)
        
        # Create agent instance
        if self._requires_custom_initialization():
            instance = await self._create_with_custom_init(agent_id, final_config)
        else:
            instance = self.agent_class(
                agent_id=agent_id,
                configuration=final_config
            )
        
        self.created_instances[agent_id] = instance
        return instance
    
    def _requires_custom_initialization(self) -> bool:
        """Check if agent class requires custom initialization"""
        init_signature = inspect.signature(self.agent_class.__init__)
        return len(init_signature.parameters) > 2  # self + agent_id
    
    async def _create_with_custom_init(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create agent with custom initialization logic"""
        # This method handles complex agent initialization
        # Can be overridden by subclasses for specific agent types
        return self.agent_class(agent_id=agent_id, **config)


class AgentFactory:
    """
    Agent Factory for eFab AI Agent System
    
    Features:
    - Dynamic agent creation and lifecycle management
    - Template-based agent configuration
    - Resource-aware deployment strategies
    - Customer-specific agent provisioning
    - Automatic dependency resolution
    - Performance monitoring and scaling
    - Hot-swapping and updates
    """
    
    def __init__(self, message_router: Optional[MessageRouter] = None):
        """Initialize agent factory"""
        self.logger = logging.getLogger("AgentFactory")
        
        # Agent management
        self.agent_templates: Dict[str, AgentTemplate] = {}
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Customer-agent assignments
        self.customer_agents: Dict[str, List[str]] = {}  # customer_id -> agent_ids
        self.agent_customers: Dict[str, List[str]] = {}  # agent_id -> customer_ids
        
        # Resource management
        self.resource_pool: Dict[str, Any] = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "storage_gb": 100,
            "network_bandwidth_mbps": 1000
        }
        self.resource_usage: Dict[str, Any] = {
            "cpu_cores": 0,
            "memory_gb": 0,
            "storage_gb": 0,
            "network_bandwidth_mbps": 0
        }
        
        # Integration components
        self.message_router = message_router
        self.system_state = system_state
        
        # Factory metrics
        self.factory_metrics = {
            "agents_created": 0,
            "agents_destroyed": 0,
            "creation_failures": 0,
            "average_creation_time_ms": 0.0,
            "resource_utilization": 0.0
        }
        
        # Initialize built-in agent templates
        self._initialize_builtin_templates()
        
        self.logger.info("Agent Factory initialized")
    
    def _initialize_builtin_templates(self):
        """Initialize built-in agent templates"""
        from ..implementation.project_manager_agent import ImplementationProjectManagerAgent
        from ..implementation.data_migration_agent import DataMigrationIntelligenceAgent
        from ..implementation.configuration_agent import ConfigurationGenerationAgent
        
        # Implementation agents
        self.register_template(
            "project_manager",
            AgentTemplate(
                agent_class=ImplementationProjectManagerAgent,
                agent_type=AgentType.IMPLEMENTATION,
                deployment_strategy=DeploymentStrategy.CUSTOMER_SPECIFIC,
                resource_requirements={"memory_gb": 1, "cpu_cores": 0.5}
            )
        )
        
        self.register_template(
            "data_migration",
            AgentTemplate(
                agent_class=DataMigrationIntelligenceAgent,
                agent_type=AgentType.IMPLEMENTATION,
                deployment_strategy=DeploymentStrategy.CONDITIONAL,
                resource_requirements={"memory_gb": 2, "cpu_cores": 1}
            )
        )
        
        self.register_template(
            "configuration_generator",
            AgentTemplate(
                agent_class=ConfigurationGenerationAgent,
                agent_type=AgentType.IMPLEMENTATION,
                deployment_strategy=DeploymentStrategy.LAZY,
                resource_requirements={"memory_gb": 1, "cpu_cores": 0.5}
            )
        )
    
    def register_template(self, template_name: str, template: AgentTemplate):
        """Register new agent template"""
        self.agent_templates[template_name] = template
        self.logger.info(f"Registered agent template: {template_name}")
    
    def unregister_template(self, template_name: str) -> bool:
        """Unregister agent template"""
        if template_name in self.agent_templates:
            # Stop all instances of this template
            template = self.agent_templates[template_name]
            for instance_id, instance in list(template.created_instances.items()):
                asyncio.create_task(self.destroy_agent(instance_id))
            
            del self.agent_templates[template_name]
            self.logger.info(f"Unregistered agent template: {template_name}")
            return True
        return False
    
    async def create_agent(
        self, 
        template_name: str,
        agent_id: Optional[str] = None,
        configuration: Dict[str, Any] = None,
        customer_id: Optional[str] = None
    ) -> Optional[BaseAgent]:
        """
        Create new agent instance
        
        Args:
            template_name: Name of agent template to use
            agent_id: Unique ID for agent (generated if not provided)
            configuration: Agent-specific configuration
            customer_id: Customer this agent will serve
            
        Returns:
            Created agent instance or None if creation failed
        """
        start_time = datetime.now()
        
        try:
            # Generate agent ID if not provided
            if not agent_id:
                agent_id = f"{template_name}_{uuid.uuid4().hex[:8]}"
            
            # Check if agent already exists
            if agent_id in self.active_agents:
                self.logger.warning(f"Agent {agent_id} already exists")
                return self.active_agents[agent_id]
            
            # Get template
            template = self.agent_templates.get(template_name)
            if not template:
                self.logger.error(f"Unknown agent template: {template_name}")
                self.factory_metrics["creation_failures"] += 1
                return None
            
            # Check if creation is possible
            context = {
                "available_agents": list(self.active_agents.keys()),
                "available_resources": self._calculate_available_resources(),
                "customer_id": customer_id
            }
            
            if not template.can_create_instance(context):
                self.logger.error(f"Cannot create agent {agent_id} - requirements not met")
                self.factory_metrics["creation_failures"] += 1
                return None
            
            # Reserve resources
            if not self._reserve_resources(template.resource_requirements):
                self.logger.error(f"Cannot create agent {agent_id} - insufficient resources")
                self.factory_metrics["creation_failures"] += 1
                return None
            
            # Create agent instance
            agent = await template.create_instance(agent_id, configuration)
            
            # Initialize agent
            await agent.start()
            
            # Register with message router if available
            if self.message_router and hasattr(agent, 'process_message'):
                self.message_router.register_agent(
                    agent_id=agent_id,
                    agent_info={
                        "agent_name": agent.agent_name,
                        "agent_type": template.agent_type.value,
                        "capabilities": [cap.__dict__ for cap in agent.capabilities],
                        "template_name": template_name
                    },
                    message_callback=agent.process_message
                )
            
            # Store agent and metadata
            self.active_agents[agent_id] = agent
            self.agent_metadata[agent_id] = {
                "template_name": template_name,
                "agent_type": template.agent_type.value,
                "created_at": datetime.now(),
                "customer_id": customer_id,
                "resource_requirements": template.resource_requirements.copy(),
                "configuration": configuration or {}
            }
            
            # Update customer-agent assignments
            if customer_id:
                if customer_id not in self.customer_agents:
                    self.customer_agents[customer_id] = []
                self.customer_agents[customer_id].append(agent_id)
                
                if agent_id not in self.agent_customers:
                    self.agent_customers[agent_id] = []
                self.agent_customers[agent_id].append(customer_id)
            
            # Update metrics
            creation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.factory_metrics["agents_created"] += 1
            self._update_average_creation_time(creation_time_ms)
            
            self.logger.info(f"Created agent: {agent_id} ({template_name})")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {str(e)}")
            self.factory_metrics["creation_failures"] += 1
            
            # Release reserved resources on failure
            if template_name in self.agent_templates:
                template = self.agent_templates[template_name]
                self._release_resources(template.resource_requirements)
            
            return None
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy agent instance and cleanup resources"""
        try:
            if agent_id not in self.active_agents:
                self.logger.warning(f"Agent {agent_id} not found")
                return False
            
            agent = self.active_agents[agent_id]
            metadata = self.agent_metadata.get(agent_id, {})
            
            # Stop agent
            await agent.stop()
            
            # Deregister from message router
            if self.message_router:
                self.message_router.deregister_agent(agent_id)
            
            # Release resources
            resource_requirements = metadata.get("resource_requirements", {})
            self._release_resources(resource_requirements)
            
            # Update customer-agent assignments
            customer_id = metadata.get("customer_id")
            if customer_id and customer_id in self.customer_agents:
                if agent_id in self.customer_agents[customer_id]:
                    self.customer_agents[customer_id].remove(agent_id)
                    if not self.customer_agents[customer_id]:
                        del self.customer_agents[customer_id]
            
            if agent_id in self.agent_customers:
                del self.agent_customers[agent_id]
            
            # Remove from active agents
            del self.active_agents[agent_id]
            del self.agent_metadata[agent_id]
            
            # Remove from template instances
            template_name = metadata.get("template_name")
            if template_name and template_name in self.agent_templates:
                template = self.agent_templates[template_name]
                if agent_id in template.created_instances:
                    del template.created_instances[agent_id]
            
            self.factory_metrics["agents_destroyed"] += 1
            self.logger.info(f"Destroyed agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent {agent_id}: {str(e)}")
            return False
    
    async def create_customer_agent_stack(
        self, 
        customer_id: str,
        industry_type: IndustryType = IndustryType.GENERIC_MANUFACTURING
    ) -> List[str]:
        """
        Create complete agent stack for customer
        
        Args:
            customer_id: Customer to create agents for
            industry_type: Industry specialization
            
        Returns:
            List of created agent IDs
        """
        created_agents = []
        
        try:
            # Core implementation agents
            core_agents = [
                ("project_manager", {}),
                ("data_migration", {}),
                ("configuration_generator", {})
            ]
            
            for template_name, config in core_agents:
                agent = await self.create_agent(
                    template_name=template_name,
                    configuration=config,
                    customer_id=customer_id
                )
                
                if agent:
                    created_agents.append(agent.agent_id)
                else:
                    self.logger.warning(f"Failed to create {template_name} agent for customer {customer_id}")
            
            # Industry-specific agents
            industry_agent_map = {
                IndustryType.FURNITURE: "furniture_manufacturing",
                IndustryType.INJECTION_MOLDING: "injection_molding",
                IndustryType.ELECTRICAL_EQUIPMENT: "electrical_equipment"
            }
            
            industry_template = industry_agent_map.get(industry_type)
            if industry_template and industry_template in self.agent_templates:
                agent = await self.create_agent(
                    template_name=industry_template,
                    customer_id=customer_id
                )
                
                if agent:
                    created_agents.append(agent.agent_id)
            
            self.logger.info(f"Created {len(created_agents)} agents for customer {customer_id}")
            return created_agents
            
        except Exception as e:
            self.logger.error(f"Failed to create customer agent stack for {customer_id}: {str(e)}")
            
            # Cleanup partially created agents
            for agent_id in created_agents:
                await self.destroy_agent(agent_id)
            
            return []
    
    async def scale_agents(self, template_name: str, target_count: int) -> List[str]:
        """Scale agents of specific template to target count"""
        if template_name not in self.agent_templates:
            return []
        
        template = self.agent_templates[template_name]
        current_count = len(template.created_instances)
        
        if current_count == target_count:
            return list(template.created_instances.keys())
        
        elif current_count < target_count:
            # Scale up
            created_agents = []
            for i in range(target_count - current_count):
                agent = await self.create_agent(template_name)
                if agent:
                    created_agents.append(agent.agent_id)
            
            return list(template.created_instances.keys())
        
        else:
            # Scale down
            agents_to_remove = list(template.created_instances.keys())[target_count:]
            for agent_id in agents_to_remove:
                await self.destroy_agent(agent_id)
            
            return list(template.created_instances.keys())
    
    def get_customer_agents(self, customer_id: str) -> List[BaseAgent]:
        """Get all agents assigned to customer"""
        agent_ids = self.customer_agents.get(customer_id, [])
        return [self.active_agents[aid] for aid in agent_ids if aid in self.active_agents]
    
    def get_agent_customers(self, agent_id: str) -> List[str]:
        """Get all customers assigned to agent"""
        return self.agent_customers.get(agent_id, [])
    
    def _calculate_available_resources(self) -> Dict[str, Any]:
        """Calculate available resources"""
        available = {}
        for resource, total in self.resource_pool.items():
            used = self.resource_usage.get(resource, 0)
            available[resource] = max(0, total - used)
        return available
    
    def _reserve_resources(self, requirements: Dict[str, Any]) -> bool:
        """Reserve resources for agent creation"""
        available = self._calculate_available_resources()
        
        # Check if resources are available
        for resource, required in requirements.items():
            if available.get(resource, 0) < required:
                return False
        
        # Reserve resources
        for resource, required in requirements.items():
            self.resource_usage[resource] += required
        
        return True
    
    def _release_resources(self, requirements: Dict[str, Any]):
        """Release resources from destroyed agent"""
        for resource, amount in requirements.items():
            self.resource_usage[resource] = max(0, self.resource_usage[resource] - amount)
    
    def _update_average_creation_time(self, creation_time_ms: float):
        """Update average creation time metric"""
        current_avg = self.factory_metrics["average_creation_time_ms"]
        total_created = self.factory_metrics["agents_created"]
        
        if total_created == 1:
            self.factory_metrics["average_creation_time_ms"] = creation_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.factory_metrics["average_creation_time_ms"] = (
                alpha * creation_time_ms + (1 - alpha) * current_avg
            )
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get comprehensive factory status"""
        # Calculate resource utilization
        total_resources = sum(self.resource_pool.values())
        used_resources = sum(self.resource_usage.values())
        utilization = (used_resources / total_resources * 100) if total_resources > 0 else 0
        
        self.factory_metrics["resource_utilization"] = utilization
        
        return {
            "active_agents": len(self.active_agents),
            "registered_templates": len(self.agent_templates),
            "customer_assignments": len(self.customer_agents),
            "resource_pool": self.resource_pool.copy(),
            "resource_usage": self.resource_usage.copy(),
            "resource_utilization_percentage": utilization,
            "factory_metrics": self.factory_metrics.copy(),
            "agent_breakdown": {
                template_name: len(template.created_instances)
                for template_name, template in self.agent_templates.items()
            }
        }
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific agent"""
        if agent_id not in self.active_agents:
            return None
        
        agent = self.active_agents[agent_id]
        metadata = self.agent_metadata.get(agent_id, {})
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.agent_name,
            "agent_description": agent.agent_description,
            "status": agent.status.value if hasattr(agent, 'status') else "UNKNOWN",
            "template_name": metadata.get("template_name"),
            "agent_type": metadata.get("agent_type"),
            "created_at": metadata.get("created_at").isoformat() if metadata.get("created_at") else None,
            "customer_id": metadata.get("customer_id"),
            "capabilities": [cap.name for cap in agent.capabilities],
            "resource_usage": metadata.get("resource_requirements", {}),
            "configuration": metadata.get("configuration", {})
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of factory and all agents"""
        healthy_agents = 0
        unhealthy_agents = 0
        agent_health_details = {}
        
        for agent_id, agent in self.active_agents.items():
            try:
                # Basic health check
                if hasattr(agent, 'get_status'):
                    status = agent.get_status()
                    agent_status = status.get("status", "UNKNOWN")
                    
                    if agent_status in ["READY", "BUSY"]:
                        healthy_agents += 1
                        agent_health_details[agent_id] = "HEALTHY"
                    else:
                        unhealthy_agents += 1
                        agent_health_details[agent_id] = f"UNHEALTHY ({agent_status})"
                
                else:
                    # Agent doesn't support health checks
                    agent_health_details[agent_id] = "NO_HEALTH_CHECK"
                    
            except Exception as e:
                unhealthy_agents += 1
                agent_health_details[agent_id] = f"ERROR: {str(e)}"
        
        return {
            "factory_healthy": unhealthy_agents == 0,
            "total_agents": len(self.active_agents),
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "resource_utilization": self.factory_metrics["resource_utilization"],
            "agent_health_details": agent_health_details,
            "timestamp": datetime.now().isoformat()
        }


# Export main component
__all__ = ["AgentFactory", "AgentTemplate", "AgentType", "DeploymentStrategy"]