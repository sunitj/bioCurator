"""Test memory manager implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.schemas import DatabaseConfig
from src.memory.manager import DefaultMemoryManager
from src.memory.interfaces import HealthStatus
from src.safety.event_bus import EventBus


class TestDefaultMemoryManager:
    """Test DefaultMemoryManager implementation."""

    @pytest.fixture
    def config(self):
        """Database configuration for testing."""
        return DatabaseConfig(
            redis_url="redis://localhost:6379/0",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            postgres_url="postgresql://test:test@localhost:5432/test",
            qdrant_url="http://localhost:6333",
            influxdb_url="http://localhost:8086"
        )

    @pytest.fixture
    def event_bus(self):
        """Mock event bus for testing."""
        return MagicMock(spec=EventBus)

    def test_initialization(self, config, event_bus):
        """Test manager initialization."""
        manager = DefaultMemoryManager(config, event_bus)

        assert manager.config == config
        assert manager.event_bus == event_bus
        assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, config, event_bus):
        """Test successful initialization."""
        manager = DefaultMemoryManager(config, event_bus)

        # Mock all backend clients
        with patch.multiple(
            'src.memory.manager',
            Neo4jClient=MagicMock(),
            QdrantClient=MagicMock(),
            PostgresClient=MagicMock(),
            RedisClient=MagicMock(),
            InfluxClient=MagicMock()
        ) as mocks:
            # Setup mock backends
            for mock_class in mocks.values():
                mock_instance = AsyncMock()
                mock_instance.health_check.return_value = HealthStatus(
                    is_healthy=True,
                    message="Healthy"
                )
                mock_class.return_value = mock_instance

            await manager.initialize()

            assert manager.is_initialized is True

            # Verify all backends were connected
            for mock_class in mocks.values():
                mock_instance = mock_class.return_value
                mock_instance.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, config, event_bus):
        """Test initialization failure."""
        manager = DefaultMemoryManager(config, event_bus)

        # Mock failing backend
        with patch.multiple(
            'src.memory.manager',
            Neo4jClient=MagicMock(),
            QdrantClient=MagicMock(),
            PostgresClient=MagicMock(),
            RedisClient=MagicMock(),
            InfluxClient=MagicMock()
        ) as mocks:
            # Make Redis client fail health check
            redis_mock = AsyncMock()
            redis_mock.health_check.return_value = HealthStatus(
                is_healthy=False,
                message="Connection failed"
            )
            mocks['RedisClient'].return_value = redis_mock

            # Other backends are healthy
            for name, mock_class in mocks.items():
                if name != 'RedisClient':
                    mock_instance = AsyncMock()
                    mock_instance.health_check.return_value = HealthStatus(
                        is_healthy=True,
                        message="Healthy"
                    )
                    mock_class.return_value = mock_instance

            with pytest.raises(RuntimeError, match="Failed to initialize backends"):
                await manager.initialize()

            assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_health_check_all(self, config, event_bus):
        """Test health check for all backends."""
        manager = DefaultMemoryManager(config, event_bus)

        # Mock backends
        manager._knowledge_graph = AsyncMock()
        manager._vector_store = AsyncMock()
        manager._episodic_memory = AsyncMock()
        manager._working_memory = AsyncMock()
        manager._time_series = AsyncMock()

        # Setup health check responses
        manager._knowledge_graph.health_check.return_value = HealthStatus(
            is_healthy=True, message="Neo4j healthy"
        )
        manager._vector_store.health_check.return_value = HealthStatus(
            is_healthy=True, message="Qdrant healthy"
        )
        manager._episodic_memory.health_check.return_value = HealthStatus(
            is_healthy=False, message="Postgres connection failed"
        )
        manager._working_memory.health_check.return_value = HealthStatus(
            is_healthy=True, message="Redis healthy"
        )
        manager._time_series.health_check.return_value = HealthStatus(
            is_healthy=True, message="InfluxDB healthy"
        )

        # Mock event bus
        event_bus.emit = AsyncMock()

        results = await manager.health_check_all()

        # Verify results
        assert len(results) == 5
        assert results["knowledge_graph"].is_healthy is True
        assert results["vector_store"].is_healthy is True
        assert results["episodic_memory"].is_healthy is False
        assert results["working_memory"].is_healthy is True
        assert results["time_series"].is_healthy is True

        # Verify safety event was emitted for unhealthy backend
        event_bus.emit.assert_called_once()
        call_args = event_bus.emit.call_args[0][0]
        assert "episodic_memory" in call_args.message

    @pytest.mark.asyncio
    async def test_shutdown(self, config, event_bus):
        """Test manager shutdown."""
        manager = DefaultMemoryManager(config, event_bus)
        manager._initialized = True

        # Mock backends
        manager._knowledge_graph = AsyncMock()
        manager._vector_store = AsyncMock()
        manager._episodic_memory = AsyncMock()
        manager._working_memory = AsyncMock()
        manager._time_series = AsyncMock()

        await manager.shutdown()

        # Verify all backends were disconnected
        manager._knowledge_graph.disconnect.assert_called_once()
        manager._vector_store.disconnect.assert_called_once()
        manager._episodic_memory.disconnect.assert_called_once()
        manager._working_memory.disconnect.assert_called_once()
        manager._time_series.disconnect.assert_called_once()

        assert manager.is_initialized is False

    def test_get_backends_not_initialized(self, config, event_bus):
        """Test getting backends when not initialized."""
        manager = DefaultMemoryManager(config, event_bus)

        with pytest.raises(RuntimeError, match="Knowledge graph backend not initialized"):
            manager.get_knowledge_graph()

        with pytest.raises(RuntimeError, match="Vector store backend not initialized"):
            manager.get_vector_store()

        with pytest.raises(RuntimeError, match="Episodic memory backend not initialized"):
            manager.get_episodic_memory()

        with pytest.raises(RuntimeError, match="Working memory backend not initialized"):
            manager.get_working_memory()

        with pytest.raises(RuntimeError, match="Time series backend not initialized"):
            manager.get_time_series()

    def test_get_backends_initialized(self, config, event_bus):
        """Test getting backends when initialized."""
        manager = DefaultMemoryManager(config, event_bus)

        # Mock backends
        manager._knowledge_graph = MagicMock()
        manager._vector_store = MagicMock()
        manager._episodic_memory = MagicMock()
        manager._working_memory = MagicMock()
        manager._time_series = MagicMock()

        assert manager.get_knowledge_graph() == manager._knowledge_graph
        assert manager.get_vector_store() == manager._vector_store
        assert manager.get_episodic_memory() == manager._episodic_memory
        assert manager.get_working_memory() == manager._working_memory
        assert manager.get_time_series() == manager._time_series

    @pytest.mark.asyncio
    async def test_context_manager(self, config, event_bus):
        """Test async context manager."""
        manager = DefaultMemoryManager(config, event_bus)

        with patch.object(manager, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(manager, 'shutdown', new_callable=AsyncMock) as mock_shutdown:
                async with manager as m:
                    assert m is manager
                    mock_init.assert_called_once()

                mock_shutdown.assert_called_once()