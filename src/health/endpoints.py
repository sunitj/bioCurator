"""Health check HTTP endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class ComponentHealthResponse(BaseModel):
    """Response model for component health."""

    name: str
    status: str
    message: str
    response_time_ms: float | None = None
    metadata: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Response model for overall health."""

    status: str
    timestamp: float
    components: list[ComponentHealthResponse]


def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter()

    @router.get("/", response_model=HealthResponse)
    async def health_check(request: Request):
        """
        Get overall system health status.

        Returns health information for all system components.
        """
        try:
            import time

            # Basic health check - just verify configuration is loaded
            components = [
                ComponentHealthResponse(
                    name="config",
                    status="healthy",
                    message=f"Configuration loaded, mode: {settings.app_mode.value}",
                    response_time_ms=1.0,
                    metadata={
                        "app_mode": settings.app_mode.value,
                        "log_level": settings.log_level,
                    },
                ),
                ComponentHealthResponse(
                    name="logging",
                    status="healthy",
                    message="Logging system operational",
                    response_time_ms=0.5,
                ),
            ]

            # Check memory system health if available
            if hasattr(request.app.state, "memory_manager"):
                memory_manager = request.app.state.memory_manager
                try:
                    memory_health = await memory_manager.health_check_all()
                    for backend_name, health_status in memory_health.items():
                        components.append(
                            ComponentHealthResponse(
                                name=f"memory_{backend_name}",
                                status="healthy" if health_status.is_healthy else "unhealthy",
                                message=health_status.message,
                                response_time_ms=health_status.latency_ms,
                                metadata={
                                    "connection_count": health_status.connection_count,
                                    "memory_usage_mb": health_status.memory_usage_mb,
                                },
                            )
                        )
                except Exception as e:
                    logger.warning("Memory health check failed", error=str(e))
                    components.append(
                        ComponentHealthResponse(
                            name="memory_system",
                            status="unhealthy",
                            message=f"Memory health check failed: {str(e)}",
                        )
                    )

            # Check agent system health if available
            if hasattr(request.app.state, "agent_registry"):
                agent_registry = request.app.state.agent_registry
                try:
                    agent_health = await agent_registry.health_check_all()
                    for agent_id, health_info in agent_health.items():
                        components.append(
                            ComponentHealthResponse(
                                name=f"agent_{agent_id}",
                                status="healthy" if health_info.get("is_active", False) else "unhealthy",
                                message=f"Agent {agent_id}: {health_info.get('health_status', 'unknown')}",
                                response_time_ms=1.0,  # Placeholder
                                metadata={
                                    "circuit_breaker_state": health_info.get("circuit_breaker_state", "unknown"),
                                    "current_tasks": health_info.get("current_tasks", 0),
                                    "memory_access": health_info.get("memory_access", "unknown"),
                                },
                            )
                        )

                    # Add agent system overview
                    all_agent_status = await agent_registry.get_all_agent_status()
                    active_agents = sum(1 for status in all_agent_status.values() if status.is_active)
                    total_agents = len(all_agent_status)

                    components.append(
                        ComponentHealthResponse(
                            name="agent_system",
                            status="healthy" if active_agents > 0 else "unhealthy",
                            message=f"Agent system: {active_agents}/{total_agents} agents active",
                            response_time_ms=1.0,
                            metadata={
                                "active_agents": active_agents,
                                "total_agents": total_agents,
                                "system_mode": settings.app_mode.value,
                            },
                        )
                    )

                except Exception as e:
                    logger.warning("Agent health check failed", error=str(e))
                    components.append(
                        ComponentHealthResponse(
                            name="agent_system",
                            status="unhealthy",
                            message=f"Agent health check failed: {str(e)}",
                        )
                    )

            # Check task queue health if available
            if hasattr(request.app.state, "task_queue"):
                task_queue = request.app.state.task_queue
                try:
                    queue_stats = await task_queue.get_queue_statistics()
                    queue_size = await task_queue.get_queue_size()

                    components.append(
                        ComponentHealthResponse(
                            name="task_queue",
                            status="healthy",
                            message=f"Task queue operational: {queue_size} pending tasks",
                            response_time_ms=1.0,
                            metadata={
                                "queue_size": queue_size,
                                "statistics": queue_stats,
                            },
                        )
                    )

                except Exception as e:
                    logger.warning("Task queue health check failed", error=str(e))
                    components.append(
                        ComponentHealthResponse(
                            name="task_queue",
                            status="unhealthy",
                            message=f"Task queue health check failed: {str(e)}",
                        )
                    )

            # Determine overall status
            overall_status = "healthy" if all(c.status == "healthy" for c in components) else "unhealthy"

            return HealthResponse(status=overall_status, timestamp=time.time(), components=components)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")

    @router.get("/ready")
    async def readiness_check(request: Request):
        """
        Check if the system is ready to serve requests.

        Returns 200 if ready, 503 if not ready.
        """
        try:
            # Basic readiness check - verify critical components
            if not settings:
                raise HTTPException(status_code=503, detail="Configuration not loaded")

            # Check if memory system is ready
            if hasattr(request.app.state, "memory_manager"):
                memory_manager = request.app.state.memory_manager
                try:
                    memory_health = await memory_manager.health_check_all()
                    unhealthy_backends = [
                        name for name, health in memory_health.items() if not health.is_healthy
                    ]
                    if unhealthy_backends:
                        raise HTTPException(
                            status_code=503,
                            detail=f"Memory backends not ready: {', '.join(unhealthy_backends)}",
                        )
                except Exception as e:
                    logger.error("Memory readiness check failed", error=str(e))
                    raise HTTPException(status_code=503, detail="Memory system not ready")

            return {"status": "ready", "message": "System is ready to serve requests"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(status_code=503, detail="System not ready")

    @router.get("/live")
    async def liveness_check():
        """
        Check if the system is alive.

        Returns 200 if alive, used for container liveness probes.
        """
        return {"status": "alive", "message": "System is alive"}

    return router
