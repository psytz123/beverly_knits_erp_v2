#!/usr/bin/env python3
"""
eFab AI Agent Architecture
Multi-agent orchestration system for intelligent ERP implementation and management
"""

from .core.orchestrator import CentralOrchestrator
from .core.agent_base import BaseAgent, AgentMessage, AgentCapability
from .core.state_manager import SystemState, CustomerProfile
from .implementation.project_manager_agent import ImplementationProjectManagerAgent
from .implementation.data_migration_agent import DataMigrationIntelligenceAgent
from .implementation.configuration_agent import ConfigurationGenerationAgent
from .industry.furniture_agent import FurnitureManufacturingAgent
from .industry.injection_molding_agent import InjectionMoldingAgent
from .industry.electrical_equipment_agent import ElectricalEquipmentAgent
from .optimization.performance_agent import PerformanceOptimizationAgent
from .learning.knowledge_manager_agent import LearningKnowledgeManagerAgent
from .communication.message_router import MessageRouter
from .deployment.agent_factory import AgentFactory

__version__ = "1.0.0"
__author__ = "eFab AI Team"

# Export main components
__all__ = [
    "CentralOrchestrator",
    "BaseAgent", 
    "AgentMessage",
    "AgentCapability",
    "SystemState",
    "CustomerProfile",
    "ImplementationProjectManagerAgent",
    "DataMigrationIntelligenceAgent", 
    "ConfigurationGenerationAgent",
    "FurnitureManufacturingAgent",
    "InjectionMoldingAgent",
    "ElectricalEquipmentAgent",
    "PerformanceOptimizationAgent",
    "LearningKnowledgeManagerAgent",
    "MessageRouter",
    "AgentFactory"
]