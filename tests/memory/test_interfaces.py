"""Test memory interfaces and abstract classes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.memory.interfaces import HealthStatus, MemoryBackend


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_healthy_status(self):
        """Test healthy status creation."""
        status = HealthStatus(
            is_healthy=True,
            message="All good",
            latency_ms=1.5,
            connection_count=10,
            memory_usage_mb=128.0
        )

        assert status.is_healthy is True
        assert status.message == "All good"
        assert status.latency_ms == 1.5
        assert status.connection_count == 10
        assert status.memory_usage_mb == 128.0

    def test_unhealthy_status(self):
        """Test unhealthy status creation."""
        status = HealthStatus(
            is_healthy=False,
            message="Connection failed"
        )

        assert status.is_healthy is False
        assert status.message == "Connection failed"
        assert status.latency_ms is None
        assert status.connection_count is None
        assert status.memory_usage_mb is None


class TestMemoryBackend:
    """Test MemoryBackend abstract class."""

    def test_initialization(self):
        """Test backend initialization."""
        config = {"url": "test://localhost", "timeout": 30}

        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            MemoryBackend(config)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        # Create a concrete implementation for testing
        class TestBackend(MemoryBackend):
            async def connect(self):
                self._connected = True

            async def disconnect(self):
                self._connected = False

            async def health_check(self):
                return HealthStatus(is_healthy=True, message="Test OK")

            async def ping(self):
                return 1.0

        backend = TestBackend({"test": "config"})

        # Test context manager
        async with backend as b:
            assert b.is_connected is True
            assert b is backend

        assert backend.is_connected is False