#!/usr/bin/env python3
"""
Integration module for Beverly Knits AI Agents
Provides ERP integration and external system connectivity
"""

from .erp_bridge import ERPIntegrationBridge, APICallResult, APICallStatus, erp_bridge

__all__ = ["ERPIntegrationBridge", "APICallResult", "APICallStatus", "erp_bridge"]