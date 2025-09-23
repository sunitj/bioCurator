"""Abstract interfaces for memory backends."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Health status of a memory backend."""

    is_healthy: bool
    message: str
    latency_ms: float | None = None
    connection_count: int | None = None
    memory_usage_mb: float | None = None


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the memory backend.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the memory backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the memory backend."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check the health status of the backend."""
        pass

    @abstractmethod
    async def ping(self) -> float:
        """Ping the backend and return latency in milliseconds."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class KnowledgeGraphBackend(MemoryBackend):
    """Interface for knowledge graph operations (Neo4j)."""

    @abstractmethod
    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
        node_id: str | None = None
    ) -> str:
        """Create a node in the knowledge graph."""
        pass

    @abstractmethod
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """Create a relationship between nodes."""
        pass

    @abstractmethod
    async def find_nodes(
        self,
        label: str | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find nodes matching criteria."""
        pass

    @abstractmethod
    async def find_relationships(
        self,
        from_node_id: str | None = None,
        to_node_id: str | None = None,
        relationship_type: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find relationships matching criteria."""
        pass

    @abstractmethod
    async def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a custom Cypher query."""
        pass


class VectorBackend(MemoryBackend):
    """Interface for vector operations (Qdrant)."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine"
    ) -> bool:
        """Create a new vector collection."""
        pass

    @abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None
    ) -> list[str]:
        """Insert or update vectors in collection."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        vector_ids: list[str]
    ) -> bool:
        """Delete vectors from collection."""
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection."""
        pass


class EpisodicBackend(MemoryBackend):
    """Interface for episodic memory operations (PostgreSQL)."""

    @abstractmethod
    async def store_episode(
        self,
        episode_id: str,
        agent_id: str,
        action: str,
        context: dict[str, Any],
        outcome: dict[str, Any],
        timestamp: float | None = None
    ) -> bool:
        """Store an agent episode."""
        pass

    @abstractmethod
    async def get_episodes(
        self,
        agent_id: str | None = None,
        action_type: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Retrieve episodes matching criteria."""
        pass

    @abstractmethod
    async def get_agent_history(
        self,
        agent_id: str,
        hours_back: int = 24,
        action_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get recent history for an agent."""
        pass

    @abstractmethod
    async def update_episode_outcome(
        self,
        episode_id: str,
        outcome: dict[str, Any]
    ) -> bool:
        """Update the outcome of an existing episode."""
        pass


class WorkingMemoryBackend(MemoryBackend):
    """Interface for working memory operations (Redis)."""

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: int | None = None
    ) -> bool:
        """Set a key-value pair with optional expiration."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on existing key."""
        pass

    @abstractmethod
    async def get_all_keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern."""
        pass


class TimeSeriesBackend(MemoryBackend):
    """Interface for time-series operations (InfluxDB)."""

    @abstractmethod
    async def write_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: int | None = None
    ) -> bool:
        """Write a single data point."""
        pass

    @abstractmethod
    async def write_points(
        self,
        points: list[dict[str, Any]]
    ) -> bool:
        """Write multiple data points."""
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        start_time: str | None = None,
        stop_time: str | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Flux query."""
        pass

    @abstractmethod
    async def get_agent_metrics(
        self,
        agent_id: str,
        metric_name: str,
        hours_back: int = 24
    ) -> list[dict[str, Any]]:
        """Get metrics for a specific agent."""
        pass


class MemoryManager(ABC):
    """Interface for coordinating multiple memory backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all memory backends."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown all memory backends."""
        pass

    @abstractmethod
    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Check health of all backends."""
        pass

    @abstractmethod
    def get_knowledge_graph(self) -> KnowledgeGraphBackend:
        """Get the knowledge graph backend."""
        pass

    @abstractmethod
    def get_vector_store(self) -> VectorBackend:
        """Get the vector storage backend."""
        pass

    @abstractmethod
    def get_episodic_memory(self) -> EpisodicBackend:
        """Get the episodic memory backend."""
        pass

    @abstractmethod
    def get_working_memory(self) -> WorkingMemoryBackend:
        """Get the working memory backend."""
        pass

    @abstractmethod
    def get_time_series(self) -> TimeSeriesBackend:
        """Get the time-series backend."""
        pass
