"""No-op tracing hooks for future vendor integration."""

import functools
from contextlib import contextmanager
from typing import Any, Callable


class TracingContext:
    """Placeholder tracing context (no-op implementation)."""

    def __init__(self, name: str, **attributes: Any):
        """Initialize tracing context."""
        self.name = name
        self.attributes = attributes

    def add_attribute(self, key: str, value: Any) -> None:
        """Add attribute to trace context (no-op)."""
        pass

    def set_status(self, status: str) -> None:
        """Set trace status (no-op)."""
        pass


@contextmanager
def trace_span(name: str, **attributes: Any):
    """
    Context manager for tracing spans (no-op).

    Future implementation will integrate with chosen vendor.
    """
    context = TracingContext(name, **attributes)
    try:
        yield context
    finally:
        pass  # Future: send span data to tracing backend


def trace_function(name: str | None = None):
    """
    Decorator for function tracing (no-op).

    Future implementation will automatically trace function calls.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            with trace_span(span_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator