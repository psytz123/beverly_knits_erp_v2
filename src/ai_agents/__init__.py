"""
eFab AI Agent System
====================

Multi-Agent ERP Implementation and Customer Service Platform

This module provides the core AI agent infrastructure for automated ERP 
implementation, customer service, and manufacturing optimization.

Architecture:
- Training System: Comprehensive 12-week agent training framework
- Core Agents: Lead, Customer Manager, Implementation specialists  
- Specialized Agents: Industry-specific domain experts
- Simulation Environment: Training and validation scenarios
- Assessment Framework: Competency validation and certification
"""

__version__ = "1.0.0"
__author__ = "eFab AI Systems"

from .core.base_agent import BaseAgent
from .core.agent_orchestrator import AgentOrchestrator
from .training.training_orchestrator import TrainingOrchestrator

__all__ = [
    'BaseAgent',
    'AgentOrchestrator', 
    'TrainingOrchestrator'
]