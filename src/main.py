"""BioCurator main application entry point."""

import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .health.endpoints import create_health_router
from .logging import get_logger
from .metrics.prometheus import create_metrics_router

# Configure logging
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="BioCurator",
        description="Memory-augmented multi-agent system for scientific literature analysis",
        version="0.1.0",
        debug=settings.monitoring.enable_debug,
    )

    # Add middleware (ContextMiddleware will be added in PR #1.5)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routers
    app.include_router(create_health_router(), prefix="/health", tags=["health"])
    app.include_router(create_metrics_router(), prefix="/metrics", tags=["metrics"])

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic system information."""
        return {
            "name": "BioCurator",
            "version": "0.1.0",
            "mode": settings.app_mode.value,
            "status": "running",
        }

    return app


def main():
    """Main entry point."""
    logger.info(f"Starting BioCurator in {settings.app_mode.value} mode")

    # Create application
    app = create_app()

    # Configure server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.monitoring.enable_debug,
        access_log=True,
    )

    # Start server
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down BioCurator")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
