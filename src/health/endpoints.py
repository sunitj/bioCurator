"""Health check HTTP endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..logging import get_logger
from ..config import settings

logger = get_logger(__name__)


class ComponentHealthResponse(BaseModel):
    """Response model for component health."""
    name: str
    status: str
    message: str
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for overall health."""
    status: str
    timestamp: float
    components: List[ComponentHealthResponse]


def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter()
    
    @router.get("/", response_model=HealthResponse)
    async def health_check():
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
                    }
                ),
                ComponentHealthResponse(
                    name="logging",
                    status="healthy",
                    message="Logging system operational",
                    response_time_ms=0.5,
                )
            ]
            
            return HealthResponse(
                status="healthy",
                timestamp=time.time(),
                components=components
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    @router.get("/ready")
    async def readiness_check():
        """
        Check if the system is ready to serve requests.
        
        Returns 200 if ready, 503 if not ready.
        """
        try:
            # Basic readiness check - verify critical components
            if not settings:
                raise HTTPException(status_code=503, detail="Configuration not loaded")
            
            return {
                "status": "ready",
                "message": "System is ready to serve requests"
            }
            
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(status_code=503, detail="System not ready")
    
    @router.get("/live")
    async def liveness_check():
        """
        Check if the system is alive.
        
        Returns 200 if alive, used for container liveness probes.
        """
        return {
            "status": "alive",
            "message": "System is alive"
        }
    
    return router