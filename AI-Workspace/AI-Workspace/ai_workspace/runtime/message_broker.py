#!/usr/bin/env python3
"""
In-memory async message broker for inter-agent communication.

Provides reliable message passing between agents using async queues.
Supports priority queuing, message persistence, and dead-letter handling.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from .communication_protocol import AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class MessageStats:
    """Statistics for message broker monitoring."""

    total_sent: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    total_expired: int = 0
    queue_depths: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.queue_depths is None:
            self.queue_depths = {}


class InMemoryMessageBroker:
    """
    Async message queue for inter-agent communication.

    Features:
    - Priority queues (0=low, 10=urgent)
    - Message TTL (time-to-live)
    - Dead letter queue for failed messages
    - Message persistence (in-memory)
    - Broadcast support

    Example:
        >>> broker = InMemoryMessageBroker()
        >>> await broker.send(message)
        >>> response = await broker.receive("agent-id", timeout=30)
    """

    def __init__(self):
        """Initialize message broker."""
        self.queues: Dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue()
        )
        self.dead_letter: asyncio.Queue = asyncio.Queue()
        self.message_history: List[AgentMessage] = []
        self.stats = MessageStats()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._counter = 0  # Monotonic counter for queue ordering

        logger.info("InMemoryMessageBroker initialized")

    async def start(self) -> None:
        """
        Start the message broker.

        Initiates background cleanup task for expired messages.
        """
        if self._running:
            logger.warning("Message broker already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        logger.info("Message broker started")

    async def stop(self) -> None:
        """Stop the message broker and cleanup."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Message broker stopped")

    async def send(self, message: AgentMessage) -> None:
        """
        Send message to target agent's queue.

        Args:
            message: Message to send

        Raises:
            ValueError: If message is invalid
        """
        if message.to_agent == "":
            raise ValueError("Message must have a recipient (to_agent)")

        # Check if message already expired
        if message.is_expired():
            logger.warning(
                f"Message {message.correlation_id} already expired, "
                f"sending to dead letter queue"
            )
            await self.dead_letter.put(message)
            self.stats.total_expired += 1
            return

        # Add to appropriate queue
        queue = self.queues[message.to_agent]

        # Priority queue: lower number = higher priority
        # So we invert: 10-priority to make priority=10 highest
        priority_value = 10 - message.priority

        # Use monotonic counter as tiebreaker to avoid comparing AgentMessage objects
        self._counter += 1
        await queue.put((priority_value, self._counter, message))

        # Update stats
        self.stats.total_sent += 1
        self.stats.queue_depths[message.to_agent] = queue.qsize()

        # Store in history
        self.message_history.append(message)

        logger.debug(
            f"Message sent to {message.to_agent} "
            f"(type={message.message_type.value}, priority={message.priority})"
        )

    async def receive(
        self,
        agent_id: str,
        timeout: int = 30
    ) -> AgentMessage:
        """
        Receive next message from agent's queue.

        Args:
            agent_id: Agent ID to receive messages for
            timeout: Maximum time to wait (seconds)

        Returns:
            Next message from queue

        Raises:
            asyncio.TimeoutError: If no message within timeout
        """
        queue = self.queues[agent_id]

        try:
            # Wait for message with timeout
            priority_value, counter, message = await asyncio.wait_for(
                queue.get(),
                timeout=timeout
            )

            # Check if expired while waiting
            if message.is_expired():
                logger.warning(
                    f"Message {message.correlation_id} expired while in queue"
                )
                await self.dead_letter.put(message)
                self.stats.total_expired += 1

                # Try to get next message (recursive)
                return await self.receive(agent_id, timeout=timeout)

            # Update stats
            self.stats.total_delivered += 1
            self.stats.queue_depths[agent_id] = queue.qsize()

            logger.debug(
                f"Message received by {agent_id} "
                f"(type={message.message_type.value})"
            )

            return message

        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"No message received for {agent_id} within {timeout}s"
            )

    async def broadcast(
        self,
        message: AgentMessage,
        agents: List[str]
    ) -> None:
        """
        Send message to multiple agents.

        Args:
            message: Message to broadcast
            agents: List of agent IDs to send to
        """
        logger.info(f"Broadcasting to {len(agents)} agents")

        for agent_id in agents:
            # Create copy with updated recipient
            broadcast_msg = AgentMessage(
                from_agent=message.from_agent,
                to_agent=agent_id,
                message_type=message.message_type,
                payload=message.payload.copy(),
                correlation_id=message.correlation_id,
                timestamp=message.timestamp,
                priority=message.priority,
                ttl=message.ttl
            )

            await self.send(broadcast_msg)

    async def get_queue_depth(self, agent_id: str) -> int:
        """
        Get number of pending messages for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of messages in queue
        """
        queue = self.queues.get(agent_id)
        return queue.qsize() if queue else 0

    async def get_stats(self) -> MessageStats:
        """
        Get broker statistics.

        Returns:
            MessageStats with current stats
        """
        # Update queue depths
        for agent_id, queue in self.queues.items():
            self.stats.queue_depths[agent_id] = queue.qsize()

        return self.stats

    async def get_message_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Get recent message history.

        Args:
            agent_id: Filter by agent (optional)
            limit: Maximum messages to return

        Returns:
            List of recent messages
        """
        if agent_id:
            filtered = [
                msg for msg in self.message_history
                if msg.from_agent == agent_id or msg.to_agent == agent_id
            ]
            return filtered[-limit:]
        else:
            return self.message_history[-limit:]

    async def _cleanup_expired_messages(self) -> None:
        """
        Background task to cleanup expired messages.

        Runs continuously, checking queues every 60 seconds for expired
        messages and moving them to dead letter queue.
        """
        logger.info("Started expired message cleanup task")

        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                for agent_id, queue in list(self.queues.items()):
                    # Check messages in queue
                    messages_to_requeue = []
                    expired_count = 0

                    # Drain queue
                    while not queue.empty():
                        try:
                            priority, counter, message = queue.get_nowait()

                            if message.is_expired():
                                await self.dead_letter.put(message)
                                expired_count += 1
                            else:
                                messages_to_requeue.append((priority, counter, message))

                        except asyncio.QueueEmpty:
                            break

                    # Re-queue non-expired messages
                    for priority, counter, message in messages_to_requeue:
                        await queue.put((priority, counter, message))

                    if expired_count > 0:
                        logger.info(
                            f"Cleaned up {expired_count} expired messages "
                            f"for {agent_id}"
                        )
                        self.stats.total_expired += expired_count

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(5)

        logger.info("Expired message cleanup task stopped")

    async def drain_queue(self, agent_id: str) -> List[AgentMessage]:
        """
        Remove all messages from agent's queue.

        Useful for cleanup or emergency stops.

        Args:
            agent_id: Agent ID

        Returns:
            List of messages that were in queue
        """
        queue = self.queues.get(agent_id)
        if not queue:
            return []

        messages = []
        while not queue.empty():
            try:
                _, _, message = queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break

        logger.info(f"Drained {len(messages)} messages from {agent_id} queue")
        return messages

    def __repr__(self) -> str:
        """String representation for debugging."""
        active_queues = len([q for q in self.queues.values() if not q.empty()])
        return (
            f"InMemoryMessageBroker(active_queues={active_queues}, "
            f"total_sent={self.stats.total_sent}, "
            f"total_delivered={self.stats.total_delivered})"
        )
