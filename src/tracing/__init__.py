"""Lightweight tracing hooks placeholder (no vendor lock)."""

from .hooks import TracingContext, trace_span

__all__ = ["TracingContext", "trace_span"]