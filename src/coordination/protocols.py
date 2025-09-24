"""Inter-agent communication protocols."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..logging import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AgentMessage(BaseModel):
    """Inter-agent message format."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None  # For request-response correlation
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL

    # Source and destination
    sender_id: str
    recipient_id: str
    recipient_type: str = "agent"  # agent, broadcast, system

    # Content
    subject: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    timeout_seconds: int = 30

    # Delivery tracking
    delivery_attempts: int = 0
    max_delivery_attempts: int = 3
    delivered_at: Optional[datetime] = None


class MessageResponse(BaseModel):
    """Response to an agent message."""

    original_message_id: str
    success: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    response_time_seconds: float


class MessageProtocol:
    """Handles inter-agent message passing and routing."""

    def __init__(self, agent_id: str, timeout_seconds: int = 30):
        """Initialize message protocol.

        Args:
            agent_id: ID of the agent using this protocol
            timeout_seconds: Default timeout for message delivery
        """
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds

        # Message storage
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._message_handlers: Dict[str, callable] = {}
        self._incoming_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self._messages_sent = 0
        self._messages_received = 0
        self._messages_failed = 0

        logger.info(f"Message protocol initialized for agent {agent_id}")

    def register_handler(self, subject: str, handler: callable) -> None:
        """Register a message handler for a specific subject.

        Args:
            subject: Message subject pattern to handle
            handler: Async function to handle messages
        """
        self._message_handlers[subject] = handler
        logger.debug(f"Handler registered for subject '{subject}'", agent_id=self.agent_id)

    async def send_request(
        self,
        recipient_id: str,
        subject: str,
        payload: Dict[str, Any] = None,
        timeout_seconds: Optional[int] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> MessageResponse:
        """Send a request message and wait for response.

        Args:
            recipient_id: Target agent ID
            subject: Message subject
            payload: Message payload data
            timeout_seconds: Timeout for response (uses default if None)
            priority: Message priority

        Returns:
            MessageResponse: Response from the recipient

        Raises:
            TimeoutError: If response not received within timeout
            RuntimeError: If delivery fails
        """
        start_time = datetime.now()
        timeout = timeout_seconds or self.timeout_seconds

        # Create request message
        message = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload or {},
            timeout_seconds=timeout,
            priority=priority,
        )

        # Set up response future
        response_future = asyncio.Future()
        self._pending_requests[message.id] = response_future

        try:
            # Send message (this would integrate with actual message transport)
            await self._deliver_message(message)
            self._messages_sent += 1

            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(
                f"Request completed",
                agent_id=self.agent_id,
                recipient=recipient_id,
                subject=subject,
                duration=execution_time,
            )

            return response

        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout",
                agent_id=self.agent_id,
                recipient=recipient_id,
                subject=subject,
                timeout=timeout,
            )
            raise TimeoutError(f"No response from {recipient_id} within {timeout}s")

        except Exception as e:
            logger.error(
                f"Request failed",
                agent_id=self.agent_id,
                recipient=recipient_id,
                subject=subject,
                error=str(e),
            )
            self._messages_failed += 1
            raise RuntimeError(f"Failed to send request: {str(e)}")

        finally:
            # Clean up pending request
            self._pending_requests.pop(message.id, None)

    async def send_notification(
        self,
        recipient_id: str,
        subject: str,
        payload: Dict[str, Any] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Send a notification message (fire-and-forget).

        Args:
            recipient_id: Target agent ID
            subject: Message subject
            payload: Message payload data
            priority: Message priority
        """
        message = AgentMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload or {},
            priority=priority,
        )

        try:
            await self._deliver_message(message)
            self._messages_sent += 1

            logger.debug(
                f"Notification sent",
                agent_id=self.agent_id,
                recipient=recipient_id,
                subject=subject,
            )

        except Exception as e:
            logger.error(
                f"Notification failed",
                agent_id=self.agent_id,
                recipient=recipient_id,
                subject=subject,
                error=str(e),
            )
            self._messages_failed += 1
            raise

    async def broadcast(
        self,
        subject: str,
        payload: Dict[str, Any] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Broadcast a message to all agents.

        Args:
            subject: Message subject
            payload: Message payload data
            priority: Message priority
        """
        message = AgentMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=self.agent_id,
            recipient_id="*",
            recipient_type="broadcast",
            subject=subject,
            payload=payload or {},
            priority=priority,
        )

        try:
            await self._deliver_message(message)
            self._messages_sent += 1

            logger.debug(
                f"Broadcast sent",
                agent_id=self.agent_id,
                subject=subject,
            )

        except Exception as e:
            logger.error(
                f"Broadcast failed",
                agent_id=self.agent_id,
                subject=subject,
                error=str(e),
            )
            self._messages_failed += 1
            raise

    async def handle_incoming_message(self, message: AgentMessage) -> None:
        """Handle an incoming message.

        Args:
            message: Incoming message to process
        """
        await self._incoming_queue.put(message)

    async def start_message_processing(self) -> None:
        """Start processing incoming messages."""
        logger.info(f"Starting message processing for agent {self.agent_id}")

        while True:
            try:
                # Get next message from queue
                message = await self._incoming_queue.get()
                self._messages_received += 1

                # Process message based on type
                if message.message_type == MessageType.RESPONSE:
                    await self._handle_response(message)
                elif message.message_type in [MessageType.REQUEST, MessageType.NOTIFICATION]:
                    await self._handle_request_or_notification(message)
                else:
                    logger.warning(
                        f"Unknown message type: {message.message_type}",
                        agent_id=self.agent_id,
                    )

            except asyncio.CancelledError:
                logger.info(f"Message processing cancelled for agent {self.agent_id}")
                break
            except Exception as e:
                logger.error(
                    f"Error processing message",
                    agent_id=self.agent_id,
                    error=str(e),
                )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get message protocol statistics."""
        return {
            "agent_id": self.agent_id,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "messages_failed": self._messages_failed,
            "pending_requests": len(self._pending_requests),
            "registered_handlers": len(self._message_handlers),
        }

    # Private methods

    async def _deliver_message(self, message: AgentMessage) -> None:
        """Deliver a message to its recipient.

        This is a placeholder for actual message transport mechanism.
        In a real implementation, this would integrate with:
        - Redis pub/sub
        - RabbitMQ/Apache Pulsar
        - gRPC streaming
        - WebSocket connections
        """
        # For now, we'll simulate message delivery
        # In the actual implementation, this would handle routing and delivery
        logger.debug(
            f"Message delivery",
            sender=message.sender_id,
            recipient=message.recipient_id,
            subject=message.subject,
            type=message.message_type.value,
        )

        # Simulate delivery delay
        await asyncio.sleep(0.001)

        # Mark message as delivered
        message.delivered_at = datetime.now()

    async def _handle_response(self, message: AgentMessage) -> None:
        """Handle response messages."""
        # Find the corresponding request
        correlation_id = message.correlation_id
        if not correlation_id or correlation_id not in self._pending_requests:
            logger.warning(
                f"Received response without matching request",
                agent_id=self.agent_id,
                correlation_id=correlation_id,
            )
            return

        # Create response object
        response = MessageResponse(
            original_message_id=correlation_id,
            success=message.message_type != MessageType.ERROR,
            result=message.payload,
            error_message=message.payload.get("error") if message.message_type == MessageType.ERROR else None,
            response_time_seconds=0.0,  # Would be calculated from timing
        )

        # Complete the future
        future = self._pending_requests.get(correlation_id)
        if future and not future.done():
            future.set_result(response)

    async def _handle_request_or_notification(self, message: AgentMessage) -> None:
        """Handle request or notification messages."""
        # Find handler for this subject
        handler = self._message_handlers.get(message.subject)
        if not handler:
            logger.warning(
                f"No handler for message subject '{message.subject}'",
                agent_id=self.agent_id,
                sender=message.sender_id,
            )

            # Send error response for requests
            if message.message_type == MessageType.REQUEST:
                await self._send_error_response(
                    message, f"No handler for subject '{message.subject}'"
                )
            return

        try:
            # Call the handler
            result = await handler(message)

            # Send response for requests
            if message.message_type == MessageType.REQUEST:
                await self._send_success_response(message, result or {})

        except Exception as e:
            logger.error(
                f"Handler error",
                agent_id=self.agent_id,
                subject=message.subject,
                error=str(e),
            )

            # Send error response for requests
            if message.message_type == MessageType.REQUEST:
                await self._send_error_response(message, str(e))

    async def _send_success_response(self, original_message: AgentMessage, result: Dict[str, Any]) -> None:
        """Send a success response to a request."""
        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            correlation_id=original_message.id,
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            subject=f"response:{original_message.subject}",
            payload=result,
        )

        await self._deliver_message(response)

    async def _send_error_response(self, original_message: AgentMessage, error: str) -> None:
        """Send an error response to a request."""
        response = AgentMessage(
            message_type=MessageType.ERROR,
            correlation_id=original_message.id,
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            subject=f"error:{original_message.subject}",
            payload={"error": error},
        )

        await self._deliver_message(response)