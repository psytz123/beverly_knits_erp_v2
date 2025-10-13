#!/usr/bin/env python3
"""
Structured IPC protocol for agent worker communication.

Provides type-safe, bi-directional message passing between main process and worker
processes with support for commands, results, logs, metrics, artifacts, and errors.

Message Types:
    - command: Send execution commands to worker
    - result: Receive execution results from worker
    - log: Receive log messages from worker
    - metric: Receive performance metrics from worker
    - artifact: Receive generated artifacts from worker
    - error: Receive error information from worker

Usage:
    channel = IPCChannel(work_queue, result_queue)
    await channel.send_command("execute", {"code": "print('hello')"})
    message = await channel.recv_message(timeout=5.0)
    all_results = await channel.collect_all(timeout=10.0)
"""

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Message type constants
MSG_TYPE_COMMAND = "command"
MSG_TYPE_RESULT = "result"
MSG_TYPE_LOG = "log"
MSG_TYPE_METRIC = "metric"
MSG_TYPE_ARTIFACT = "artifact"
MSG_TYPE_ERROR = "error"

MSG_TYPE_BATCH = "batch"
VALID_MESSAGE_TYPES = {
    MSG_TYPE_COMMAND,
    MSG_TYPE_RESULT,
    MSG_TYPE_LOG,
    MSG_TYPE_METRIC,
    MSG_TYPE_ARTIFACT,
    MSG_TYPE_ERROR,
    MSG_TYPE_BATCH,
}


@dataclass
class IPCMessage:
    """
    Structured message for inter-process communication.

    Attributes:
        type: Message type (command, result, log, metric, artifact, error)
        payload: Message payload data (must be JSON-serializable)
        timestamp: Unix timestamp when message was created
        worker_pid: Process ID of the worker that sent/receives the message
    """

    type: str
    payload: Any
    timestamp: float
    worker_pid: int

    def __post_init__(self) -> None:
        """Validate message type after initialization."""
        if self.type not in VALID_MESSAGE_TYPES:
            raise ValueError(
                f"Invalid message type: {self.type}. "
                f"Must be one of {VALID_MESSAGE_TYPES}"
            )

    def to_json(self) -> str:
        """
        Serialize message to JSON string.

        Returns:
            JSON string representation of the message

        Raises:
            TypeError: If payload is not JSON-serializable
        """
        try:
            message_dict = asdict(self)
            return json.dumps(message_dict)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Message payload is not JSON-serializable: {e}") from e

    @classmethod
    def from_json(cls, data: str) -> "IPCMessage":
        """
        Deserialize message from JSON string.

        Args:
            data: JSON string containing message data

        Returns:
            IPCMessage instance

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        try:
            message_dict = json.loads(data)
            return cls(
                type=message_dict["type"],
                payload=message_dict["payload"],
                timestamp=message_dict["timestamp"],
                worker_pid=message_dict["worker_pid"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid message JSON: {e}") from e

    @classmethod
    def create_command(
        cls, command: str, params: Dict[str, Any], worker_pid: int
    ) -> "IPCMessage":
        """
        Create a command message.

        Args:
            command: Command name
            params: Command parameters
            worker_pid: Target worker process ID

        Returns:
            IPCMessage of type command
        """
        return cls(
            type=MSG_TYPE_COMMAND,
            payload={"command": command, "params": params},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

    @classmethod
    def create_result(
        cls, result: Any, success: bool, worker_pid: int
    ) -> "IPCMessage":
        """
        Create a result message.

        Args:
            result: Execution result data
            success: Whether execution was successful
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type result
        """
        return cls(
            type=MSG_TYPE_RESULT,
            payload={"result": result, "success": success},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

    @classmethod
    def create_log(cls, level: str, message: str, worker_pid: int) -> "IPCMessage":
        """
        Create a log message.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message text
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type log
        """
        return cls(
            type=MSG_TYPE_LOG,
            payload={"level": level, "message": message},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

    @classmethod
    def create_metric(
        cls, metric_name: str, value: float, unit: str, worker_pid: int
    ) -> "IPCMessage":
        """
        Create a metric message.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type metric
        """
        return cls(
            type=MSG_TYPE_METRIC,
            payload={"metric": metric_name, "value": value, "unit": unit},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

    @classmethod
    def create_artifact(
        cls, artifact_type: str, content: Any, metadata: Dict[str, Any], worker_pid: int
    ) -> "IPCMessage":
        """
        Create an artifact message.

        Args:
            artifact_type: Type of artifact (file, data, report, etc.)
            content: Artifact content
            metadata: Additional metadata about the artifact
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type artifact
        """
        return cls(
            type=MSG_TYPE_ARTIFACT,
            payload={
                "artifact_type": artifact_type,
                "content": content,
                "metadata": metadata,
            },
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

    @classmethod
    def create_error(
        cls, error_type: str, message: str, traceback: Optional[str], worker_pid: int
    ) -> "IPCMessage":
        """
        Create an error message.

        Args:
            error_type: Type/class of the error
            message: Error message
            traceback: Optional stack trace
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type error
        """
        return cls(
            type=MSG_TYPE_ERROR,
            payload={"error_type": error_type, "message": message, "traceback": traceback},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )



    @classmethod
    def create_batch(
        cls, messages: List["IPCMessage"], worker_pid: int
    ) -> "IPCMessage":
        """
        Create a batch message containing multiple messages.

        Args:
            messages: List of IPCMessage objects to batch
            worker_pid: Worker process ID

        Returns:
            IPCMessage of type batch with messages in payload

        Example:
            >>> logs = [IPCMessage.create_log("INFO", "msg1", 123)]
            >>> batch = IPCMessage.create_batch(logs, 123)
        """
        # Serialize individual messages
        message_dicts = [
            {
                "type": msg.type,
                "payload": msg.payload,
                "timestamp": msg.timestamp,
                "worker_pid": msg.worker_pid,
            }
            for msg in messages
        ]

        return cls(
            type=MSG_TYPE_BATCH,
            payload={"messages": message_dicts, "count": len(messages)},
            timestamp=time.time(),
            worker_pid=worker_pid,
        )

