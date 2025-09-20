"""Metrics and monitoring for BioCurator."""

from .prometheus import create_metrics_router, metrics_registry, REQUEST_COUNT, REQUEST_DURATION

__all__ = [
    "create_metrics_router",
    "metrics_registry",
    "REQUEST_COUNT", 
    "REQUEST_DURATION",
]