#!/usr/bin/env python3
"""
AI Agent System Initializer for Beverly Knits ERP
Initializes and configures the complete AI agent infrastructure
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Core agent system imports
from .core.agent_base import BaseAgent, AgentCapability, create_capability
from .core.orchestrator import CentralOrchestrator
from .core.state_manager import system_state, CustomerProfile, IndustryType
from .deployment.agent_factory import AgentFactory, AgentTemplate, AgentType, DeploymentStrategy
from .communication.message_router import MessageRouter
from ..agents.training_framework import AgentTrainingOrchestrator, AgentConfig, AgentRole
from ..agents.role_definitions import AGENT_ROLES_REGISTRY, get_agent_role

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentSystemInitializer:
    """
    Comprehensive AI Agent System Initializer
    
    Responsibilities:
    - Initialize core agent infrastructure
    - Set up agent factory and orchestrator
    - Configure message routing
    - Load agent configurations
    - Deploy initial agent stack
    - Validate system readiness
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the agent system initializer"""
        self.base_path = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.base_path / "config" / "agent_training_config.json"
        
        # Core components
        self.message_router: Optional[MessageRouter] = None
        self.agent_factory: Optional[AgentFactory] = None
        self.orchestrator: Optional[CentralOrchestrator] = None
        self.training_orchestrator: Optional[AgentTrainingOrchestrator] = None
        
        # System state
        self.initialization_status = {
            "started_at": datetime.now(),
            "components_initialized": [],
            "agents_deployed": [],
            "errors": [],
            "is_ready": False
        }
        
        # Load configuration
        self.config = self._load_configuration()
        
        logger.info("Agent System Initializer created")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load agent training configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            "global_settings": {
                "enable_parallel_training": True,
                "max_parallel_agents": 3,
                "verbose_logging": True
            },
            "agent_configurations": {
                "inventory_intelligence": {"enabled": True, "priority": 1},
                "forecast_intelligence": {"enabled": True, "priority": 2},
                "production_planning": {"enabled": True, "priority": 3},
                "yarn_substitution": {"enabled": True, "priority": 4},
                "quality_assurance": {"enabled": True, "priority": 5}
            }
        }
    
    async def initialize_system(self, mode: str = "full") -> Dict[str, Any]:
        """
        Initialize the complete AI agent system
        
        Args:
            mode: Initialization mode ('full', 'minimal', 'training_only')
            
        Returns:
            Initialization status and results
        """
        logger.info(f"Starting AI agent system initialization (mode: {mode})")
        
        try:
            # Step 1: Initialize core infrastructure
            await self._initialize_core_infrastructure()
            
            # Step 2: Set up agent factory and templates
            await self._setup_agent_factory()
            
            # Step 3: Initialize orchestrator
            await self._initialize_orchestrator()
            
            # Step 4: Set up training system
            if mode in ['full', 'training_only']:
                await self._setup_training_system()
            
            # Step 5: Deploy initial agents
            if mode == 'full':
                await self._deploy_initial_agents()
            
            # Step 6: Validate system readiness
            await self._validate_system_readiness()
            
            self.initialization_status["is_ready"] = True
            self.initialization_status["completed_at"] = datetime.now()
            
            logger.info("AI agent system initialization completed successfully")
            return self.initialization_status
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            logger.error(error_msg)
            self.initialization_status["errors"].append(error_msg)
            self.initialization_status["is_ready"] = False
            return self.initialization_status
    
    async def _initialize_core_infrastructure(self):
        """Initialize core agent infrastructure"""
        logger.info("Initializing core infrastructure...")
        
        # Initialize message router
        self.message_router = MessageRouter()
        await self.message_router.start()
        
        # Initialize system state
        system_state.initialize_system({
            "system_name": "Beverly Knits ERP AI Agents",
            "version": "2.0.0",
            "environment": "production",
            "initialized_at": datetime.now().isoformat()
        })
        
        self.initialization_status["components_initialized"].append("core_infrastructure")
        logger.info("Core infrastructure initialized")
    
    async def _setup_agent_factory(self):
        """Set up agent factory with templates"""
        logger.info("Setting up agent factory...")
        
        # Create agent factory
        self.agent_factory = AgentFactory(message_router=self.message_router)
        
        # Register agent templates based on role definitions
        for role_id, role_def in AGENT_ROLES_REGISTRY.items():
            if self._is_agent_enabled(role_id):
                await self._register_agent_template(role_id, role_def)
        
        self.initialization_status["components_initialized"].append("agent_factory")
        logger.info("Agent factory setup completed")
    
    def _is_agent_enabled(self, role_id: str) -> bool:
        """Check if agent is enabled in configuration"""
        agent_config = self.config.get("agent_configurations", {}).get(role_id, {})
        return agent_config.get("enabled", False)
    
    async def _register_agent_template(self, role_id: str, role_def):
        """Register agent template with factory"""
        try:
            # Create agent template based on role definition
            template = AgentTemplate(
                agent_class=self._get_agent_class_for_role(role_id),
                agent_type=self._get_agent_type_for_role(role_id),
                deployment_strategy=DeploymentStrategy.LAZY,
                resource_requirements=self._get_resource_requirements(role_id),
                configuration_template=self._get_configuration_template(role_id)
            )
            
            self.agent_factory.register_template(role_id, template)
            logger.info(f"Registered template for {role_id}")
            
        except Exception as e:
            logger.error(f"Failed to register template for {role_id}: {e}")
    
    def _get_agent_class_for_role(self, role_id: str):
        """Get agent class for specific role (placeholder implementation)"""
        # This would import and return the actual agent implementation classes
        # For now, return BaseAgent as placeholder
        
        class RoleSpecificAgent(BaseAgent):
            def __init__(self, agent_id: str, **kwargs):
                super().__init__(
                    agent_id=agent_id,
                    agent_name=f"{role_id.replace('_', ' ').title()} Agent",
                    agent_description=f"AI agent for {role_id} operations"
                )
            
            def _initialize(self):
                # Add role-specific capabilities
                role_def = get_agent_role(role_id)
                if role_def:
                    for capability in role_def.capabilities:
                        self.register_capability(AgentCapability(
                            name=capability.name,
                            description=capability.description,
                            input_schema={"type": "object"},
                            output_schema={"type": "object"}
                        ))
        
        return RoleSpecificAgent
    
    def _get_agent_type_for_role(self, role_id: str) -> AgentType:
        """Map role to agent type"""
        role_type_mapping = {
            "inventory_intelligence": AgentType.CORE,
            "forecast_intelligence": AgentType.CORE,
            "production_planning": AgentType.CORE,
            "yarn_substitution": AgentType.OPTIMIZATION,
            "quality_assurance": AgentType.MONITORING
        }
        return role_type_mapping.get(role_id, AgentType.CORE)
    
    def _get_resource_requirements(self, role_id: str) -> Dict[str, Any]:
        """Get resource requirements for agent role"""
        agent_config = self.config.get("agent_configurations", {}).get(role_id, {})
        specific_settings = agent_config.get("specific_settings", {})
        
        return {
            "memory_gb": 1,
            "cpu_cores": 0.5,
            "storage_gb": 1,
            "network_bandwidth_mbps": 10
        }
    
    def _get_configuration_template(self, role_id: str) -> Dict[str, Any]:
        """Get configuration template for agent role"""
        agent_config = self.config.get("agent_configurations", {}).get(role_id, {})
        return agent_config.get("specific_settings", {})
    
    async def _initialize_orchestrator(self):
        """Initialize central orchestrator"""
        logger.info("Initializing orchestrator...")
        
        # Create orchestrator
        self.orchestrator = CentralOrchestrator()
        
        # Set message router
        self.orchestrator.set_message_router(self.message_router.route_message)
        
        # Start orchestrator
        await self.orchestrator.start()
        
        # Register orchestrator with message router
        if hasattr(self.message_router, 'register_agent'):
            self.message_router.register_agent(
                agent_id=self.orchestrator.agent_id,
                agent_info={
                    "agent_name": "Central Orchestrator",
                    "agent_type": "ORCHESTRATOR",
                    "capabilities": [cap.name for cap in self.orchestrator.capabilities]
                },
                message_callback=self.orchestrator.process_message
            )
        
        self.initialization_status["components_initialized"].append("orchestrator")
        logger.info("Orchestrator initialized")
    
    async def _setup_training_system(self):
        """Set up agent training system"""
        logger.info("Setting up training system...")
        
        # Create training orchestrator
        self.training_orchestrator = AgentTrainingOrchestrator()
        
        # Register agents for training based on configuration
        for role_id in self.config.get("agent_configurations", {}):
            if self._is_agent_enabled(role_id):
                await self._register_agent_for_training(role_id)
        
        self.initialization_status["components_initialized"].append("training_system")
        logger.info("Training system setup completed")
    
    async def _register_agent_for_training(self, role_id: str):
        """Register agent for training"""
        try:
            # Create agent config for training
            agent_config = self._create_training_config(role_id)
            
            # Note: This would need actual agent implementation classes
            # For now, we register the configuration for future training
            logger.info(f"Training configuration prepared for {role_id}")
            
        except Exception as e:
            logger.error(f"Failed to register {role_id} for training: {e}")
    
    def _create_training_config(self, role_id: str) -> AgentConfig:
        """Create training configuration for agent role"""
        role_config = self.config.get("agent_configurations", {}).get(role_id, {})
        specific_settings = role_config.get("specific_settings", {})
        
        return AgentConfig(
            role=AgentRole(role_id),
            name=f"{role_id.replace('_', ' ').title()} Agent",
            min_accuracy=specific_settings.get("min_accuracy", 0.85),
            max_response_time_ms=specific_settings.get("max_response_time_ms", 200),
            training_epochs=specific_settings.get("epochs", 100),
            batch_size=specific_settings.get("batch_size", 32),
            learning_rate=specific_settings.get("learning_rate", 0.001)
        )
    
    async def _deploy_initial_agents(self):
        """Deploy initial set of agents"""
        logger.info("Deploying initial agents...")
        
        deployed_agents = []
        
        # Get enabled agents sorted by priority
        enabled_agents = [
            (role_id, config) 
            for role_id, config in self.config.get("agent_configurations", {}).items()
            if config.get("enabled", False)
        ]
        enabled_agents.sort(key=lambda x: x[1].get("priority", 999))
        
        # Deploy agents
        for role_id, config in enabled_agents:
            try:
                agent = await self.agent_factory.create_agent(
                    template_name=role_id,
                    configuration=config.get("specific_settings", {})
                )
                
                if agent:
                    deployed_agents.append(agent.agent_id)
                    logger.info(f"Deployed agent: {agent.agent_id}")
                else:
                    logger.warning(f"Failed to deploy agent for {role_id}")
                    
            except Exception as e:
                logger.error(f"Error deploying {role_id}: {e}")
        
        self.initialization_status["agents_deployed"] = deployed_agents
        logger.info(f"Deployed {len(deployed_agents)} agents")
    
    async def _validate_system_readiness(self):
        """Validate that the system is ready for operation"""
        logger.info("Validating system readiness...")
        
        validation_results = {
            "message_router_active": self.message_router is not None,
            "agent_factory_ready": self.agent_factory is not None,
            "orchestrator_active": self.orchestrator is not None,
            "agents_deployed": len(self.initialization_status["agents_deployed"]) > 0
        }
        
        # Check component health
        if self.agent_factory:
            factory_status = self.agent_factory.get_factory_status()
            validation_results["factory_health"] = factory_status["active_agents"] > 0
        
        if self.orchestrator:
            orchestrator_status = self.orchestrator.get_orchestrator_status()
            validation_results["orchestrator_health"] = orchestrator_status["registered_agents"] > 0
        
        # Overall system readiness
        all_checks_passed = all(validation_results.values())
        
        self.initialization_status["validation_results"] = validation_results
        self.initialization_status["system_ready"] = all_checks_passed
        
        if all_checks_passed:
            logger.info("System validation passed - AI agent system is ready")
        else:
            logger.warning(f"System validation issues detected: {validation_results}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "initialization_status": self.initialization_status,
            "components": {}
        }
        
        if self.message_router:
            status["components"]["message_router"] = "active"
        
        if self.agent_factory:
            status["components"]["agent_factory"] = self.agent_factory.get_factory_status()
        
        if self.orchestrator:
            status["components"]["orchestrator"] = self.orchestrator.get_orchestrator_status()
        
        return status
    
    async def shutdown_system(self):
        """Gracefully shutdown the AI agent system"""
        logger.info("Shutting down AI agent system...")
        
        try:
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
            
            # Stop all agents
            if self.agent_factory:
                for agent_id in list(self.agent_factory.active_agents.keys()):
                    await self.agent_factory.destroy_agent(agent_id)
            
            # Stop message router
            if self.message_router:
                await self.message_router.stop()
            
            logger.info("AI agent system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions for easy initialization
async def initialize_ai_agents(config_path: Optional[str] = None, mode: str = "full") -> Dict[str, Any]:
    """Initialize AI agent system with default settings"""
    initializer = AgentSystemInitializer(config_path)
    return await initializer.initialize_system(mode)


async def quick_start_agents() -> AgentSystemInitializer:
    """Quick start AI agents for development/testing"""
    initializer = AgentSystemInitializer()
    await initializer.initialize_system(mode="minimal")
    return initializer


if __name__ == "__main__":
    async def main():
        """Main initialization routine"""
        print("Beverly Knits ERP - AI Agent System Initializer")
        print("=" * 50)
        
        # Initialize system
        initializer = AgentSystemInitializer()
        status = await initializer.initialize_system()
        
        # Print results
        print(f"\nInitialization Status: {'SUCCESS' if status['is_ready'] else 'FAILED'}")
        print(f"Components Initialized: {len(status['components_initialized'])}")
        print(f"Agents Deployed: {len(status['agents_deployed'])}")
        
        if status['errors']:
            print(f"Errors: {len(status['errors'])}")
            for error in status['errors']:
                print(f"  - {error}")
        
        # Keep system running
        if status['is_ready']:
            print("\nAI Agent System is ready and running...")
            print("Press Ctrl+C to shutdown")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                await initializer.shutdown_system()
    
    asyncio.run(main())
