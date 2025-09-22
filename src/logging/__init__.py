"""Structured logging for BioCurator."""

from .context import LogContext, get_correlation_id, set_correlation_id
from .structured import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
    "LogContext",
    "get_correlation_id",
    "set_correlation_id",
]
