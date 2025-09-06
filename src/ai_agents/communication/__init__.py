#!/usr/bin/env python3
"""
Communication layer for eFab AI Agent System
Message routing and inter-agent communication
"""

from .message_router import MessageRouter, RoutingStrategy, RoutingRule, MessageEnvelope

__all__ = [
    "MessageRouter",
    "RoutingStrategy", 
    "RoutingRule",
    "MessageEnvelope"
]