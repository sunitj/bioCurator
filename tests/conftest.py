"""Test configuration and fixtures for BioCurator."""

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set test environment before importing application modules
os.environ["APP_MODE"] = "development"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["MAX_COST_BUDGET"] = "0.0"
os.environ["MODEL_PROVIDER"] = "ollama"

from src.config import ConfigSchema, settings
from src.logging import get_logger
from src.main import create_app

# Test logger
logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> ConfigSchema:
    """Provide test configuration settings."""
    return settings


@pytest.fixture
def test_app():
    """Create test application instance."""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client for the application."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
async def async_test_client(test_app):
    """Create async test client for the application."""
    from httpx import AsyncClient

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.set.return_value = True
    mock_client.get.return_value = b"test_value"
    mock_client.delete.return_value = 1
    mock_client.close.return_value = None

    return mock_client


@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    from unittest.mock import AsyncMock

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "deepseek-r1:32b"}, {"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]
    }

    return mock_response


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for testing external APIs."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()
    return mock_client


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for each test."""
    # Reset environment variables
    original_env = os.environ.copy()

    # Set test-specific environment
    os.environ.update(
        {
            "APP_MODE": "development",
            "LOG_LEVEL": "DEBUG",
            "MAX_COST_BUDGET": "0.0",
            "MODEL_PROVIDER": "ollama",
            "REDIS_URL": "redis://localhost:6379/15",  # Use test database
            "PROMETHEUS_ENABLED": "true",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_health_response():
    """Sample health check response for testing."""
    return {
        "status": "healthy",
        "timestamp": 1234567890.0,
        "components": [
            {
                "name": "config",
                "status": "healthy",
                "message": "Configuration loaded, mode: development",
                "response_time_ms": 1.0,
                "metadata": {"app_mode": "development", "log_level": "DEBUG"},
            },
            {
                "name": "logging",
                "status": "healthy",
                "message": "Logging system operational",
                "response_time_ms": 0.5,
            },
        ],
    }


@pytest.fixture
def sample_metrics_response():
    """Sample Prometheus metrics response for testing."""
    return """# HELP biocurator_requests_total Total number of HTTP requests
# TYPE biocurator_requests_total counter
biocurator_requests_total{method="GET",endpoint="/health",status="200"} 1.0
# HELP biocurator_request_duration_seconds HTTP request duration in seconds
# TYPE biocurator_request_duration_seconds histogram
biocurator_request_duration_seconds_bucket{method="GET",endpoint="/health",le="0.005"} 1.0
biocurator_request_duration_seconds_bucket{method="GET",endpoint="/health",le="+Inf"} 1.0
biocurator_request_duration_seconds_count{method="GET",endpoint="/health"} 1.0
biocurator_request_duration_seconds_sum{method="GET",endpoint="/health"} 0.001
"""


# Pytest plugins for additional functionality
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov",
]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "safety: Safety system tests")

    logger.info("Pytest configuration completed")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests based on path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for test session."""
    from src.logging import setup_logging

    setup_logging(
        log_level="DEBUG",
        log_path=None,  # Don't write log files during tests
        enable_json=False,  # Use simple format for tests
        enable_console=False,  # Reduce test output noise
    )

    logger.info("Test session started")

    yield

    logger.info("Test session completed")
