"""Prometheus metrics implementation."""

from fastapi import APIRouter, Response
from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from ..logging import get_logger
from ..config import settings

logger = get_logger(__name__)

# Create custom registry for BioCurator metrics
metrics_registry = CollectorRegistry()

# Core application metrics
REQUEST_COUNT = Counter(
    "biocurator_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
    registry=metrics_registry
)

REQUEST_DURATION = Histogram(
    "biocurator_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=metrics_registry
)

ACTIVE_CONNECTIONS = Gauge(
    "biocurator_active_connections",
    "Number of active connections",
    registry=metrics_registry
)

# System health metrics
HEALTH_CHECK_COUNT = Counter(
    "biocurator_health_checks_total",
    "Total number of health checks",
    ["component", "status"],
    registry=metrics_registry
)

HEALTH_CHECK_DURATION = Histogram(
    "biocurator_health_check_duration_seconds",
    "Health check duration in seconds",
    ["component"],
    registry=metrics_registry
)

# Safety metrics (will be expanded in PR #1.5)
CIRCUIT_BREAKER_TRIPS = Counter(
    "biocurator_circuit_breaker_trips_total",
    "Total number of circuit breaker trips",
    ["component"],
    registry=metrics_registry
)

RATE_LIMIT_BLOCKS = Counter(
    "biocurator_rate_limit_blocks_total",
    "Total number of rate limit blocks",
    ["component"],
    registry=metrics_registry
)

COST_BUDGET_WARNINGS = Counter(
    "biocurator_cost_budget_warnings_total",
    "Total number of cost budget warnings",
    registry=metrics_registry
)

ANOMALY_EVENTS = Counter(
    "biocurator_anomaly_events_total",
    "Total number of anomaly events detected",
    ["component", "type"],
    registry=metrics_registry
)

# Agent metrics (for future PRs)
AGENT_TASKS = Counter(
    "biocurator_agent_tasks_total",
    "Total number of agent tasks",
    ["agent", "task_type", "status"],
    registry=metrics_registry
)

AGENT_RESPONSE_TIME = Histogram(
    "biocurator_agent_response_time_seconds",
    "Agent response time in seconds",
    ["agent", "task_type"],
    registry=metrics_registry
)

# Model metrics (for future PRs)
MODEL_REQUESTS = Counter(
    "biocurator_model_requests_total",
    "Total number of model requests",
    ["provider", "model", "status"],
    registry=metrics_registry
)

MODEL_TOKENS = Counter(
    "biocurator_model_tokens_total",
    "Total number of tokens processed",
    ["provider", "model", "type"],  # type: input/output
    registry=metrics_registry
)

MODEL_COST = Counter(
    "biocurator_model_cost_total",
    "Total model costs in USD",
    ["provider", "model"],
    registry=metrics_registry
)

# Cache metrics (for future PRs)
CACHE_HITS = Counter(
    "biocurator_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
    registry=metrics_registry
)

CACHE_MISSES = Counter(
    "biocurator_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
    registry=metrics_registry
)

CACHE_HIT_RATIO = Gauge(
    "biocurator_cache_hit_ratio",
    "Cache hit ratio",
    ["cache_type"],
    registry=metrics_registry
)


def create_metrics_router() -> APIRouter:
    """Create metrics endpoints router."""
    router = APIRouter()
    
    @router.get("/")
    async def metrics_endpoint():
        """
        Prometheus metrics endpoint.
        
        Returns metrics in Prometheus text format.
        """
        if not settings.monitoring.prometheus_enabled:
            return Response(
                content="Metrics disabled",
                status_code=404,
                media_type="text/plain"
            )
        
        try:
            # Generate metrics
            metrics_data = generate_latest(metrics_registry)
            
            return Response(
                content=metrics_data,
                media_type=CONTENT_TYPE_LATEST
            )
            
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return Response(
                content=f"Error generating metrics: {e}",
                status_code=500,
                media_type="text/plain"
            )
    
    @router.get("/health")
    async def metrics_health():
        """Check metrics system health."""
        return {
            "status": "healthy" if settings.monitoring.prometheus_enabled else "disabled",
            "prometheus_enabled": settings.monitoring.prometheus_enabled,
            "registry_collectors": len(metrics_registry._collector_to_names)
        }
    
    return router


def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def record_health_check_metrics(component: str, status: str, duration: float):
    """Record health check metrics."""
    HEALTH_CHECK_COUNT.labels(component=component, status=status).inc()
    HEALTH_CHECK_DURATION.labels(component=component).observe(duration)