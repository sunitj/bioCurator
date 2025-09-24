"""Qdrant client for vector operations."""

import asyncio
import time
from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from ..logging import get_logger
from .interfaces import HealthStatus, VectorBackend

logger = get_logger(__name__)


class QdrantClient(VectorBackend):
    """Qdrant client for vector operations with safety integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Qdrant client.

        Args:
            config: Database configuration containing Qdrant settings
        """
        super().__init__(config)
        self.url = config.get("qdrant_url", "http://localhost:6333")
        self.grpc_port = config.get("qdrant_grpc_port", 6334)
        self.timeout = config.get("qdrant_timeout", 60)
        self.prefer_grpc = config.get("qdrant_prefer_grpc", True)

        self._client: AsyncQdrantClient | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self._connected:
            logger.warning("Qdrant client already connected")
            return

        logger.info("Connecting to Qdrant", url=self.url)

        # Initialize client
        self._client = AsyncQdrantClient(
            url=self.url,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
        )

        # Retry connectivity with exponential backoff using a simple list call
        max_attempts = 8
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                collections = await self._client.get_collections()
                self._connected = True
                logger.info(
                    "Qdrant connection established successfully",
                    collections=len(getattr(collections, "collections", []) or []),
                )
                return
            except qdrant_exceptions.UnexpectedResponse as e:
                last_error = e
            except Exception as e:
                last_error = e

            if attempt == max_attempts:
                break

            logger.warning(
                "Qdrant connection attempt failed; retrying",
                error=str(last_error),
                attempt=attempt,
                next_delay_ms=int(delay * 1000),
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

        if last_error:
            logger.error("Qdrant connection failed", error=str(last_error))
            raise last_error

    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self._client and self._connected:
            try:
                await self._client.close()
                logger.info("Qdrant connection closed")
            except Exception as e:
                logger.error("Error closing Qdrant connection", error=str(e))
            finally:
                self._connected = False

    async def health_check(self) -> HealthStatus:
        """Check the health status of Qdrant."""
        if not self._connected or not self._client:
            return HealthStatus(is_healthy=False, message="Not connected to Qdrant")

        try:
            start_time = time.time()

            # Check health by listing collections
            collections = await self._client.get_collections()

            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                is_healthy=True,
                message="Qdrant is healthy",
                latency_ms=latency_ms,
                connection_count=len(collections.collections),
            )

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.warning("Qdrant health check failed", error=str(e))
            return HealthStatus(is_healthy=False, message=f"Qdrant unavailable: {str(e)}")
        except Exception as e:
            logger.error("Qdrant health check error", error=str(e))
            return HealthStatus(is_healthy=False, message=f"Health check error: {str(e)}")

    async def ping(self) -> float:
        """Ping Qdrant and return latency in milliseconds."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        start_time = time.time()

        try:
            await self._client.get_collections()
            return (time.time() - start_time) * 1000
        except Exception as e:
            logger.error("Qdrant ping failed", error=str(e))
            raise

    async def create_collection(
        self, collection_name: str, vector_size: int, distance_metric: str = "cosine"
    ) -> bool:
        """Create a new vector collection."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        try:
            # Map distance metric to Qdrant enum
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclidean": models.Distance.EUCLID,
                "dot": models.Distance.DOT,
                "manhattan": models.Distance.MANHATTAN,
            }

            if distance_metric not in distance_map:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")

            # Check if collection already exists
            try:
                await self._client.get_collection(collection_name)
                logger.warning("Collection already exists", collection=collection_name)
                return True
            except qdrant_exceptions.UnexpectedResponse:
                # Collection doesn't exist, create it
                pass

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_map[distance_metric],
                ),
                optimizers_config=models.OptimizersConfig(
                    default_segment_number=2,
                    max_optimization_threads=2,
                ),
            )

            logger.info(
                "Created Qdrant collection",
                collection=collection_name,
                vector_size=vector_size,
                distance=distance_metric,
            )
            return True

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.error("Qdrant error creating collection", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error creating collection", error=str(e))
            raise

    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert or update vectors in collection."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        if not vectors:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        if len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")

        if payloads and len(payloads) != len(vectors):
            raise ValueError("Number of payloads must match number of vectors")

        try:
            # Prepare points for upsert
            points = []
            for i, vector in enumerate(vectors):
                point = models.PointStruct(
                    id=ids[i], vector=vector, payload=payloads[i] if payloads else {}
                )
                points.append(point)

            # Upsert points
            await self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,  # Wait for indexing to complete
            )

            logger.debug("Upserted vectors", collection=collection_name, count=len(vectors))
            return ids

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.error("Qdrant error upserting vectors", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error upserting vectors", error=str(e))
            raise

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        try:
            # Build search request
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vectors": True,
            }

            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold

            if filter_conditions:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(key=key, match=models.MatchValue(value=value))
                        for key, value in filter_conditions.items()
                    ]
                )

            # Execute search
            search_results = await self._client.search(**search_params)

            # Format results
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "vector": hit.vector,
                    "payload": hit.payload or {},
                }
                results.append(result)

            logger.debug(
                "Vector search completed",
                collection=collection_name,
                results_count=len(results),
                limit=limit,
            )
            return results

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.error("Qdrant error searching vectors", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error searching vectors", error=str(e))
            raise

    async def delete_vectors(self, collection_name: str, vector_ids: list[str]) -> bool:
        """Delete vectors from collection."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        if not vector_ids:
            return True

        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=vector_ids),
                wait=True,
            )

            logger.debug("Deleted vectors", collection=collection_name, count=len(vector_ids))
            return True

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.error("Qdrant error deleting vectors", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error deleting vectors", error=str(e))
            raise

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Qdrant")

        try:
            collection_info = await self._client.get_collection(collection_name)

            info = {
                "name": collection_name,
                "status": collection_info.status.value,
                "vector_count": collection_info.points_count,
                "indexed_vector_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value,
                },
                "optimizer_status": collection_info.optimizer_status,
            }

            logger.debug("Retrieved collection info", collection=collection_name)
            return info

        except qdrant_exceptions.UnexpectedResponse as e:
            logger.error("Qdrant error getting collection info", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting collection info", error=str(e))
            raise
