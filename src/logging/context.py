"""Logging context management for correlation and request tracking."""

import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


# Context variables for correlation tracking
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


@dataclass
class LogContext:
    """Logging context for structured logging."""
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-generate IDs if not provided."""
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        context = {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
        }
        
        if self.user_id:
            context["user_id"] = self.user_id
        if self.session_id:
            context["session_id"] = self.session_id
        
        # Add extra fields
        context.update(self.extra)
        
        return context
    
    def set_context(self) -> None:
        """Set this context as the current context."""
        if self.correlation_id:
            _correlation_id.set(self.correlation_id)
        if self.request_id:
            _request_id.set(self.request_id)
        if self.user_id:
            _user_id.set(self.user_id)
        if self.session_id:
            _session_id.set(self.session_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current correlation ID."""
    _correlation_id.set(correlation_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    """Set the current request ID."""
    _request_id.set(request_id)


def get_user_id() -> Optional[str]:
    """Get the current user ID."""
    return _user_id.get()


def set_user_id(user_id: str) -> None:
    """Set the current user ID."""
    _user_id.set(user_id)


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _session_id.get()


def set_session_id(session_id: str) -> None:
    """Set the current session ID."""
    _session_id.set(session_id)


def get_current_context() -> LogContext:
    """Get the current logging context."""
    return LogContext(
        correlation_id=get_correlation_id(),
        request_id=get_request_id(),
        user_id=get_user_id(),
        session_id=get_session_id(),
    )


def clear_context() -> None:
    """Clear all context variables."""
    _correlation_id.set(None)
    _request_id.set(None)
    _user_id.set(None)
    _session_id.set(None)


class ContextMiddleware:
    """Middleware to inject logging context into requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware to set logging context."""
        if scope["type"] == "http":
            # Create new context for this request
            context = LogContext()
            
            # Extract existing correlation ID from headers if present
            headers = dict(scope.get("headers", []))
            correlation_header = headers.get(b"x-correlation-id")
            if correlation_header:
                context.correlation_id = correlation_header.decode()
            
            # Set context
            context.set_context()
            
            # Add correlation ID to response headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    message.setdefault("headers", [])
                    message["headers"].append(
                        [b"x-correlation-id", context.correlation_id.encode()]
                    )
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)