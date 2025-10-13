#!/usr/bin/env python3
"""
Communication protocol for inter-agent messaging.

Defines standardized message format and types for multi-agent communication.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class MessageType(Enum):
    """Types of messages agents can exchange."""

    REQUEST = "request"        # Agent requesting another agent's service
    RESPONSE = "response"      # Reply to a request
    EVENT = "event"           # Notification of state change
    HANDOFF = "handoff"       # Transfer of responsibility
    HEARTBEAT = "heartbeat"   # Agent health check
    ERROR = "error"           # Error notification


@dataclass
class AgentMessage:
    """
    Standardized message format for inter-agent communication.

    Attributes:
        from_agent: Sender agent ID
        to_agent: Recipient agent ID
        message_type: Type of message (REQUEST, RESPONSE, etc.)
        payload: Message content as dictionary
        correlation_id: For matching requests with responses
        timestamp: When message was created
        priority: 0 (low) to 10 (urgent)
        ttl: Time-to-live in seconds
        retry_count: Number of retries attempted
    """

    from_agent: str
    to_agent: str
    message_type: MessageType
    payload: Dict[str, Any]
    correlation_id: str
    timestamp: datetime
    priority: int = 5
    ttl: int = 300
    retry_count: int = 0

    def __post_init__(self) -> None:
        """Validate message fields."""
        if not self.from_agent:
            raise ValueError("from_agent cannot be empty")
        if not self.to_agent:
            raise ValueError("to_agent cannot be empty")
        if not (0 <= self.priority <= 10):
            raise ValueError("priority must be between 0 and 10")
        if self.ttl <= 0:
            raise ValueError("ttl must be positive")

    def serialize(self) -> bytes:
        """
        Serialize message to JSON bytes for transmission.

        Returns:
            JSON-encoded message as bytes
        """
        data = asdict(self)
        # Convert enum to string
        data["message_type"] = self.message_type.value
        # Convert datetime to ISO format
        data["timestamp"] = self.timestamp.isoformat()
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "AgentMessage":
        """
        Deserialize message from JSON bytes.

        Args:
            data: JSON-encoded message bytes

        Returns:
            AgentMessage instance

        Raises:
            ValueError: If data is invalid
        """
        try:
            parsed = json.loads(data.decode("utf-8"))
            # Convert string to enum
            parsed["message_type"] = MessageType(parsed["message_type"])
            # Convert ISO format to datetime
            parsed["timestamp"] = datetime.fromisoformat(parsed["timestamp"])
            return cls(**parsed)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid message data: {e}")

    @classmethod
    def create_request(
        cls,
        from_agent: str,
        to_agent: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 5
    ) -> "AgentMessage":
        """
        Create a REQUEST message.

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            action: Action to perform
            params: Action parameters
            priority: Message priority (0-10)

        Returns:
            AgentMessage with REQUEST type
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.REQUEST,
            payload={"action": action, "params": params or {}},
            correlation_id=uuid4().hex,
            timestamp=datetime.now(),
            priority=priority
        )

    @classmethod
    def create_response(
        cls,
        from_agent: str,
        to_agent: str,
        correlation_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> "AgentMessage":
        """
        Create a RESPONSE message.

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            correlation_id: ID from original request
            status: "success" or "failure"
            result: Result data if successful
            error: Error message if failed

        Returns:
            AgentMessage with RESPONSE type
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.RESPONSE,
            payload={
                "status": status,
                "result": result or {},
                "error": error
            },
            correlation_id=correlation_id,
            timestamp=datetime.now()
        )

    @classmethod
    def create_event(
        cls,
        from_agent: str,
        to_agent: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None
    ) -> "AgentMessage":
        """
        Create an EVENT message.

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID (or "broadcast")
            event_type: Type of event
            data: Event data

        Returns:
            AgentMessage with EVENT type
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.EVENT,
            payload={"event_type": event_type, "data": data or {}},
            correlation_id=uuid4().hex,
            timestamp=datetime.now()
        )

    @classmethod
    def create_heartbeat(cls, from_agent: str) -> "AgentMessage":
        """
        Create a HEARTBEAT message.

        Args:
            from_agent: Agent sending heartbeat

        Returns:
            AgentMessage with HEARTBEAT type
        """
        return cls(
            from_agent=from_agent,
            to_agent="health-monitor",
            message_type=MessageType.HEARTBEAT,
            payload={"status": "alive", "timestamp": datetime.now().isoformat()},
            correlation_id=uuid4().hex,
            timestamp=datetime.now(),
            ttl=60  # Heartbeats expire quickly
        )

    def is_expired(self) -> bool:
        """
        Check if message has exceeded its time-to-live.

        Returns:
            True if message is expired
        """
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AgentMessage(type={self.message_type.value}, "
            f"from={self.from_agent}, to={self.to_agent}, "
            f"correlation_id={self.correlation_id[:8]}...)"
        )
