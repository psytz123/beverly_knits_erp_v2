#!/usr/bin/env python3
"""
AI Supply Chain Optimization Framework
Core framework for intelligent manufacturing ERP systems

This framework provides a complete platform that can be deployed across different 
manufacturing industries with AI-powered optimization and automated implementation.
"""

__version__ = "1.0.0"
__author__ = "Supply Chain Framework Team"
__description__ = "AI-Powered Supply Chain Optimization Framework for Manufacturing Industries"

# Export key framework components
from .core.abstract_manufacturing import ManufacturingFramework
from .core.legacy_integration import LegacySystemConnector
from .core.template_engine import IndustryTemplateEngine
from .orchestration.customer_onboarding import CustomerImplementationOrchestrator
from .agents.framework_agent_factory import FrameworkAgentFactory

__all__ = [
    "ManufacturingFramework",
    "LegacySystemConnector", 
    "IndustryTemplateEngine",
    "CustomerImplementationOrchestrator",
    "FrameworkAgentFactory"
]