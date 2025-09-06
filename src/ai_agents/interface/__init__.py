#!/usr/bin/env python3
"""
Interface layer for eFab AI Agent System
Customer interaction and management interfaces
"""

from .lead_agent import LeadAgent, ConversationState, MessageIntent, ResponseType
from .customer_manager_agent import CustomerManagerAgent, DocumentType, AgentAssignmentType, DocumentUpload, AgentTaskAssignment

__all__ = [
    "LeadAgent",
    "ConversationState",
    "MessageIntent", 
    "ResponseType",
    "CustomerManagerAgent",
    "DocumentType",
    "AgentAssignmentType",
    "DocumentUpload",
    "AgentTaskAssignment"
]