"""Neo4j client for knowledge graph operations."""

import asyncio
import time
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    Neo4jError,
    ServiceUnavailable,
    TransientError,
)

from ..logging import get_logger
from .interfaces import HealthStatus, KnowledgeGraphBackend

logger = get_logger(__name__)


class Neo4jClient(KnowledgeGraphBackend):
    """Neo4j client for knowledge graph operations with safety integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Neo4j client.

        Args:
            config: Database configuration containing Neo4j settings
        """
        super().__init__(config)
        self.uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.user = config.get("neo4j_user", "neo4j")
        self.password = config.get("neo4j_password", "")

        # Connection pool settings
        self.max_connection_lifetime = config.get("neo4j_max_connection_lifetime", 3600)
        self.max_connection_pool_size = config.get("neo4j_max_connection_pool_size", 50)
        self.connection_timeout = config.get("neo4j_connection_timeout", 30)

        self._driver: AsyncDriver | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._connected:
            logger.warning("Neo4j client already connected")
            return

        # Retry connection with exponential backoff to tolerate startup races
        max_attempts = 8
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Connecting to Neo4j", uri=self.uri, attempt=attempt)

                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=self.max_connection_lifetime,
                    max_connection_pool_size=self.max_connection_pool_size,
                    connection_timeout=self.connection_timeout,
                    encrypted=False,  # Development mode
                )

                # Verify connectivity
                await self._driver.verify_connectivity()
                self._connected = True

                logger.info("Neo4j connection established successfully")
                return

            except (AuthError, ConfigurationError) as e:
                logger.error("Neo4j authentication/configuration error", error=str(e))
                # Auth/config errors won't be fixed by retrying
                raise
            except (ServiceUnavailable, Exception) as e:
                last_error = e
                # Close driver if partially opened
                try:
                    if self._driver:
                        await self._driver.close()
                except Exception:
                    pass
                self._driver = None

                if attempt == max_attempts:
                    break

                # Jittered exponential backoff
                sleep_for = delay
                logger.warning(
                    "Neo4j connection attempt failed; retrying",
                    error=str(e),
                    attempt=attempt,
                    next_delay_ms=int(sleep_for * 1000),
                )
                await asyncio.sleep(sleep_for)
                delay = min(delay * 2, 10.0)

        # If we get here, all attempts failed
        if last_error:
            if isinstance(last_error, ServiceUnavailable):
                logger.error("Neo4j service unavailable", error=str(last_error))
            else:
                logger.error("Unexpected error connecting to Neo4j", error=str(last_error))
            raise last_error

    async def disconnect(self) -> None:
        """Close connection to Neo4j."""
        if self._driver and self._connected:
            try:
                await self._driver.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error("Error closing Neo4j connection", error=str(e))
            finally:
                self._connected = False

    async def health_check(self) -> HealthStatus:
        """Check the health status of Neo4j."""
        if not self._connected or not self._driver:
            return HealthStatus(is_healthy=False, message="Not connected to Neo4j")

        try:
            start_time = time.time()

            # Simple health check query
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 as health_check")
                record = await result.single()

                if record and record["health_check"] == 1:
                    latency_ms = (time.time() - start_time) * 1000

                    # Get server info (async) for optional diagnostics
                    pool_info = await self._driver.get_server_info()

                    return HealthStatus(
                        is_healthy=True,
                        message="Neo4j is healthy",
                        latency_ms=latency_ms,
                        connection_count=getattr(pool_info, "connection_count", None),
                    )
                else:
                    return HealthStatus(is_healthy=False, message="Neo4j health check query failed")

        except (ServiceUnavailable, TransientError) as e:
            logger.warning("Neo4j health check failed", error=str(e))
            return HealthStatus(is_healthy=False, message=f"Neo4j unavailable: {str(e)}")
        except Exception as e:
            logger.error("Neo4j health check error", error=str(e))
            return HealthStatus(is_healthy=False, message=f"Health check error: {str(e)}")

    async def ping(self) -> float:
        """Ping Neo4j and return latency in milliseconds."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        start_time = time.time()

        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return (time.time() - start_time) * 1000
        except Exception as e:
            logger.error("Neo4j ping failed", error=str(e))
            raise

    async def create_node(
        self, label: str, properties: dict[str, Any], node_id: str | None = None
    ) -> str:
        """Create a node in the knowledge graph."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        node_id = node_id or str(uuid4())
        properties["id"] = node_id
        properties["created_at"] = time.time()

        try:
            async with self._driver.session() as session:
                query = f"""
                CREATE (n:{label} $properties)
                RETURN n.id as node_id
                """
                result = await session.run(query, properties=properties)
                record = await result.single()

                if record:
                    logger.debug("Created node", label=label, node_id=record["node_id"])
                    return record["node_id"]
                else:
                    raise RuntimeError("Failed to create node")

        except Neo4jError as e:
            logger.error("Neo4j error creating node", error=str(e), label=label)
            raise
        except Exception as e:
            logger.error("Unexpected error creating node", error=str(e), label=label)
            raise

    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a relationship between nodes."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        properties = properties or {}
        relationship_id = str(uuid4())
        properties["id"] = relationship_id
        properties["created_at"] = time.time()

        try:
            async with self._driver.session() as session:
                query = f"""
                MATCH (from_node), (to_node)
                WHERE from_node.id = $from_id AND to_node.id = $to_id
                CREATE (from_node)-[r:{relationship_type} $properties]->(to_node)
                RETURN r.id as relationship_id
                """
                result = await session.run(
                    query, from_id=from_node_id, to_id=to_node_id, properties=properties
                )
                record = await result.single()

                if record:
                    logger.debug(
                        "Created relationship",
                        type=relationship_type,
                        from_node=from_node_id,
                        to_node=to_node_id,
                        relationship_id=record["relationship_id"],
                    )
                    return record["relationship_id"]
                else:
                    raise RuntimeError("Failed to create relationship - nodes may not exist")

        except Neo4jError as e:
            logger.error("Neo4j error creating relationship", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error creating relationship", error=str(e))
            raise

    async def find_nodes(
        self, label: str | None = None, properties: dict[str, Any] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find nodes matching criteria."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        try:
            async with self._driver.session() as session:
                # Build query dynamically
                query = f"MATCH (n:{label})" if label else "MATCH (n)"

                where_clauses = []
                parameters = {"limit": limit}

                if properties:
                    for key, value in properties.items():
                        param_name = f"prop_{key}"
                        where_clauses.append(f"n.{key} = ${param_name}")
                        parameters[param_name] = value

                if where_clauses:
                    query += f" WHERE {' AND '.join(where_clauses)}"

                query += " RETURN n LIMIT $limit"

                result = await session.run(query, **parameters)

                nodes = []
                async for record in result:
                    node_data = dict(record["n"])
                    nodes.append(node_data)

                logger.debug("Found nodes", count=len(nodes), label=label)
                return nodes

        except Neo4jError as e:
            logger.error("Neo4j error finding nodes", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error finding nodes", error=str(e))
            raise

    async def find_relationships(
        self,
        from_node_id: str | None = None,
        to_node_id: str | None = None,
        relationship_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find relationships matching criteria."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        try:
            async with self._driver.session() as session:
                # Build query dynamically
                if relationship_type:
                    query = f"MATCH (from_node)-[r:{relationship_type}]->(to_node)"
                else:
                    query = "MATCH (from_node)-[r]->(to_node)"

                where_clauses = []
                parameters = {"limit": limit}

                if from_node_id:
                    where_clauses.append("from_node.id = $from_id")
                    parameters["from_id"] = from_node_id

                if to_node_id:
                    where_clauses.append("to_node.id = $to_id")
                    parameters["to_id"] = to_node_id

                if where_clauses:
                    query += f" WHERE {' AND '.join(where_clauses)}"

                query += " RETURN from_node, r, to_node LIMIT $limit"

                result = await session.run(query, **parameters)

                relationships = []
                async for record in result:
                    rel_data = {
                        "from_node": dict(record["from_node"]),
                        "relationship": dict(record["r"]),
                        "to_node": dict(record["to_node"]),
                    }
                    relationships.append(rel_data)

                logger.debug("Found relationships", count=len(relationships))
                return relationships

        except Neo4jError as e:
            logger.error("Neo4j error finding relationships", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error finding relationships", error=str(e))
            raise

    async def run_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a custom Cypher query."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        parameters = parameters or {}

        try:
            async with self._driver.session() as session:
                result = await session.run(query, **parameters)

                records = []
                async for record in result:
                    records.append(dict(record))

                logger.debug("Executed custom query", records_count=len(records))
                return records

        except Neo4jError as e:
            logger.error("Neo4j error executing query", error=str(e), query=query)
            raise
        except Exception as e:
            logger.error("Unexpected error executing query", error=str(e), query=query)
            raise
