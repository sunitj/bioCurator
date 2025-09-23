"""Safety event bus for structured safety event handling."""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ..logging import get_logger

logger = get_logger(__name__)


class SafetyEventType(str, Enum):
    """Types of safety events."""

    CIRCUIT_TRIP = "circuit_trip"
    CIRCUIT_PROBE = "circuit_probe"
    CIRCUIT_RECOVERY = "circuit_recovery"
    RATE_LIMIT_BLOCK = "rate_limit_block"
    COST_BUDGET_WARNING = "cost_budget_warning"
    COST_BUDGET_VIOLATION = "cost_budget_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    CLOUD_MODEL_BLOCKED = "cloud_model_blocked"


class SafetyEvent(BaseModel):
    """Structured safety event."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: SafetyEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: str = Field(description="Component that triggered the event")
    agent_id: Optional[str] = Field(default=None, description="Agent involved in the event")
    message: str = Field(description="Human-readable event description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    severity: str = Field(default="INFO", description="Event severity level")

    def to_audit_log(self) -> str:
        """Convert event to structured audit log format."""
        audit_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "agent_id": self.agent_id,
            "message": self.message,
            "metadata": self.metadata,
            "severity": self.severity,
        }
        return json.dumps(audit_data, separators=(",", ":"))


class SafetyEventBus:
    """Central safety event bus for handling and routing safety events."""

    def __init__(self, enable_audit_log: bool = True):
        """Initialize the safety event bus."""
        self._handlers: Dict[SafetyEventType, List[Callable[[SafetyEvent], None]]] = {}
        self._enable_audit_log = enable_audit_log
        self._audit_logger = get_logger("safety.audit")

    def subscribe(
        self, event_type: SafetyEventType, handler: Callable[[SafetyEvent], None]
    ) -> None:
        """Subscribe a handler to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(
        self, event_type: SafetyEventType, handler: Callable[[SafetyEvent], None]
    ) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def emit(self, event: SafetyEvent) -> None:
        """Emit a safety event to all registered handlers."""
        # Log to audit log
        if self._enable_audit_log:
            self._audit_logger.info(
                "Safety event emitted",
                event_type=event.event_type.value,
                component=event.component,
                agent_id=event.agent_id,
                message=event.message,
                metadata=event.metadata,
                severity=event.severity,
                audit_log=event.to_audit_log(),
            )

        # Notify handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "Error in safety event handler",
                    handler=handler.__name__,
                    event_type=event.event_type.value,
                    error=str(e),
                )

    def create_event(
        self,
        event_type: SafetyEventType,
        component: str,
        message: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> SafetyEvent:
        """Create and emit a safety event."""
        event = SafetyEvent(
            event_type=event_type,
            component=component,
            message=message,
            agent_id=agent_id,
            metadata=metadata or {},
            severity=severity,
        )
        self.emit(event)
        return event


# Global safety event bus instance
_global_event_bus: Optional[SafetyEventBus] = None


def get_safety_event_bus() -> SafetyEventBus:
    """Get the global safety event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = SafetyEventBus()
    return _global_event_bus


def emit_safety_event(
    event_type: SafetyEventType,
    component: str,
    message: str,
    agent_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "INFO",
) -> SafetyEvent:
    """Convenience function to emit a safety event."""
    return get_safety_event_bus().create_event(
        event_type=event_type,
        component=component,
        message=message,
        agent_id=agent_id,
        metadata=metadata,
        severity=severity,
    )