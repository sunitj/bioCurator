"""Memory manager implementation for coordinating multiple backends."""

import asyncio

from ..config.schemas import DatabaseConfig
from ..logging import get_logger
from ..safety.circuit_breaker import CircuitBreaker
from ..safety.event_bus import SafetyEventBus, SafetyEvent, SafetyEventType
from .interfaces import (
    EpisodicBackend,
    HealthStatus,
    KnowledgeGraphBackend,
    MemoryManager,
    TimeSeriesBackend,
    VectorBackend,
    WorkingMemoryBackend,
)

logger = get_logger(__name__)


class DefaultMemoryManager(MemoryManager):
    """Default implementation of memory manager with safety integration."""

    def __init__(
        self,
        config: DatabaseConfig,
        event_bus: SafetyEventBus | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """Initialize memory manager.

        Args:
            config: Database configuration
            event_bus: Event bus for safety events
            circuit_breaker: Circuit breaker for fault tolerance
        """
        self.config = config
        self.event_bus = event_bus or SafetyEventBus()
        self.circuit_breaker = circuit_breaker

        # Backend instances
        self._knowledge_graph: KnowledgeGraphBackend | None = None
        self._vector_store: VectorBackend | None = None
        self._episodic_memory: EpisodicBackend | None = None
        self._working_memory: WorkingMemoryBackend | None = None
        self._time_series: TimeSeriesBackend | None = None

        # Initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory backends."""
        if self._initialized:
            logger.warning("Memory manager already initialized")
            return

        logger.info("Initializing memory backends")

        try:
            # Initialize backends in dependency order
            await self._initialize_backends()

            # Verify all connections
            health_results = await self.health_check_all()

            # Backends considered optional for startup gating
            optional_backends = {"time_series"}

            failed_backends = [
                name for name, status in health_results.items() if not status.is_healthy
            ]
            gating_failures = [name for name in failed_backends if name not in optional_backends]

            if gating_failures:
                error_msg = f"Failed to initialize backends: {gating_failures}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Log optional backend issues without failing startup
            optional_issues = [name for name in failed_backends if name in optional_backends]
            if optional_issues:
                logger.warning(
                    "Optional backends unavailable; continuing", backends=optional_issues
                )

            self._initialized = True
            logger.info("Memory manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize memory manager", error=str(e))
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Shutdown all memory backends."""
        if not self._initialized:
            return

        logger.info("Shutting down memory backends")

        # Shutdown in reverse order
        backends = [
            ("time_series", self._time_series),
            ("working_memory", self._working_memory),
            ("episodic_memory", self._episodic_memory),
            ("vector_store", self._vector_store),
            ("knowledge_graph", self._knowledge_graph),
        ]

        for name, backend in backends:
            if backend:
                try:
                    await backend.disconnect()
                    logger.info(f"Disconnected {name} backend")
                except Exception as e:
                    logger.error(f"Error disconnecting {name}", error=str(e))

        self._initialized = False
        logger.info("Memory manager shutdown complete")

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Check health of all backends."""
        results = {}

        backends = [
            ("knowledge_graph", self._knowledge_graph),
            ("vector_store", self._vector_store),
            ("episodic_memory", self._episodic_memory),
            ("working_memory", self._working_memory),
            ("time_series", self._time_series),
        ]

        # Run health checks concurrently
        async def check_backend(name: str, backend: object | None) -> tuple[str, HealthStatus]:
            if not backend:
                return name, HealthStatus(is_healthy=False, message="Backend not initialized")

            try:
                if self.circuit_breaker and name in self.circuit_breaker.breakers:
                    breaker = self.circuit_breaker.breakers[name]
                    if breaker.is_open:
                        return name, HealthStatus(
                            is_healthy=False, message="Circuit breaker is open"
                        )

                return name, await backend.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}", error=str(e))
                return name, HealthStatus(is_healthy=False, message=f"Health check error: {str(e)}")

        health_tasks = [check_backend(name, backend) for name, backend in backends]
        health_results = await asyncio.gather(*health_tasks)

        results = dict(health_results)

        # Emit safety events for unhealthy backends
        for name, status in results.items():
            if not status.is_healthy:
                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.MEMORY_BACKEND_FAILURE,
                        component=name,
                        message=f"Memory backend {name} is unhealthy: {status.message}",
                        metadata={"backend": name, "status": status.model_dump()},
                    )
                )

        return results

    def get_knowledge_graph(self) -> KnowledgeGraphBackend:
        """Get the knowledge graph backend."""
        if not self._knowledge_graph:
            raise RuntimeError("Knowledge graph backend not initialized")
        return self._knowledge_graph

    def get_vector_store(self) -> VectorBackend:
        """Get the vector storage backend."""
        if not self._vector_store:
            raise RuntimeError("Vector store backend not initialized")
        return self._vector_store

    def get_episodic_memory(self) -> EpisodicBackend:
        """Get the episodic memory backend."""
        if not self._episodic_memory:
            raise RuntimeError("Episodic memory backend not initialized")
        return self._episodic_memory

    def get_working_memory(self) -> WorkingMemoryBackend:
        """Get the working memory backend."""
        if not self._working_memory:
            raise RuntimeError("Working memory backend not initialized")
        return self._working_memory

    def get_time_series(self) -> TimeSeriesBackend | None:
        """Get the time-series backend."""
        if not self._time_series:
            logger.warning("Time series backend not available")
            return None
        return self._time_series

    async def _initialize_backends(self) -> None:
        """Initialize individual backends."""
        # Import here to avoid circular dependencies
        from .neo4j_client import Neo4jClient
        from .postgres_client import PostgresClient
        from .qdrant_client import QdrantClient
        from .redis_client import RedisClient

        # Initialize Redis (working memory) first - it's needed by others
        self._working_memory = RedisClient(self.config.model_dump())
        await self._working_memory.connect()

        # Initialize other backends
        self._knowledge_graph = Neo4jClient(self.config.model_dump())
        await self._knowledge_graph.connect()

        self._vector_store = QdrantClient(self.config.model_dump())
        await self._vector_store.connect()

        self._episodic_memory = PostgresClient(self.config.model_dump())
        await self._episodic_memory.connect()

        # Initialize InfluxDB (time series) - optional backend
        try:
            from .influx_client import InfluxClient

            self._time_series = InfluxClient(self.config.model_dump())
            await self._time_series.connect()
            logger.info("InfluxDB time-series backend initialized successfully")
        except ImportError as e:
            logger.warning(
                "InfluxDB client not available, time-series backend disabled", error=str(e)
            )
            self._time_series = None
        except Exception as e:
            logger.warning("Failed to initialize InfluxDB backend", error=str(e))
            self._time_series = None

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
