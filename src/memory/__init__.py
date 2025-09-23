"""Memory system module for BioCurator.

Multi-modal memory architecture supporting:
- Neo4j: Knowledge graph for papers, concepts, relationships
- Qdrant: Vector embeddings for semantic search
- PostgreSQL: Episodic memory for agent interactions
- Redis: Working memory and caching
- InfluxDB: Time-series agent performance metrics
"""

from .interfaces import MemoryBackend, MemoryManager
from .manager import DefaultMemoryManager

__all__ = [
    "MemoryBackend",
    "MemoryManager",
    "DefaultMemoryManager",
]
