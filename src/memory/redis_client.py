"""Redis client for working memory operations."""

import json
import time
from typing import Any

import redis.asyncio as redis

from ..logging import get_logger
from .interfaces import HealthStatus, WorkingMemoryBackend

logger = get_logger(__name__)


class RedisClient(WorkingMemoryBackend):
    """Redis client for working memory with safety integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Redis client.

        Args:
            config: Database configuration containing Redis settings
        """
        super().__init__(config)
        self.url = config.get("redis_url", "redis://localhost:6379/0")
        self.pool_size = config.get("redis_pool_size", 10)
        self.socket_timeout = config.get("redis_socket_timeout", 5)

        self._client: redis.Redis | None = None
        self._pool: redis.ConnectionPool | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._connected:
            logger.warning("Redis client already connected")
            return

        try:
            logger.info("Connecting to Redis", url=self.url)

            # Create connection pool
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=self.pool_size,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Create Redis client
            self._client = redis.Redis(
                connection_pool=self._pool,
                decode_responses=True
            )

            # Test connection
            await self._client.ping()
            self._connected = True

            logger.info("Redis connection established successfully")

        except redis.ConnectionError as e:
            logger.error("Redis connection failed", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error connecting to Redis", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection to Redis."""
        if self._client and self._connected:
            try:
                await self._client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error("Error closing Redis connection", error=str(e))
            finally:
                self._connected = False

    async def health_check(self) -> HealthStatus:
        """Check the health status of Redis."""
        if not self._connected or not self._client:
            return HealthStatus(
                is_healthy=False,
                message="Not connected to Redis"
            )

        try:
            start_time = time.time()

            # Ping Redis
            pong = await self._client.ping()
            if pong:
                latency_ms = (time.time() - start_time) * 1000

                # Get Redis info
                info = await self._client.info()
                memory_usage_mb = info.get("used_memory", 0) / (1024 * 1024)
                connected_clients = info.get("connected_clients", 0)

                return HealthStatus(
                    is_healthy=True,
                    message="Redis is healthy",
                    latency_ms=latency_ms,
                    connection_count=connected_clients,
                    memory_usage_mb=memory_usage_mb
                )
            else:
                return HealthStatus(
                    is_healthy=False,
                    message="Redis ping failed"
                )

        except redis.ConnectionError as e:
            logger.warning("Redis health check failed", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"Redis unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error("Redis health check error", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"Health check error: {str(e)}"
            )

    async def ping(self) -> float:
        """Ping Redis and return latency in milliseconds."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        start_time = time.time()

        try:
            await self._client.ping()
            return (time.time() - start_time) * 1000
        except Exception as e:
            logger.error("Redis ping failed", error=str(e))
            raise

    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: int | None = None
    ) -> bool:
        """Set a key-value pair with optional expiration."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            # Serialize complex objects to JSON
            serialized_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

            result = await self._client.set(
                key,
                serialized_value,
                ex=expire_seconds
            )

            logger.debug("Set Redis key", key=key, expires_in=expire_seconds)
            return bool(result)

        except redis.RedisError as e:
            logger.error("Redis error setting key", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error setting key", key=key, error=str(e))
            raise

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            value = await self._client.get(key)
            if value is None:
                return None

            # Try to deserialize JSON, fall back to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except redis.RedisError as e:
            logger.error("Redis error getting key", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting key", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            result = await self._client.delete(key)
            logger.debug("Deleted Redis key", key=key, existed=bool(result))
            return bool(result)

        except redis.RedisError as e:
            logger.error("Redis error deleting key", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error deleting key", key=key, error=str(e))
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            result = await self._client.exists(key)
            return bool(result)

        except redis.RedisError as e:
            logger.error("Redis error checking key existence", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error checking key existence", key=key, error=str(e))
            raise

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on existing key."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            result = await self._client.expire(key, seconds)
            logger.debug("Set key expiration", key=key, seconds=seconds, success=bool(result))
            return bool(result)

        except redis.RedisError as e:
            logger.error("Redis error setting expiration", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error setting expiration", key=key, error=str(e))
            raise

    async def get_all_keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            keys = await self._client.keys(pattern)
            logger.debug("Retrieved keys", pattern=pattern, count=len(keys))
            return keys

        except redis.RedisError as e:
            logger.error("Redis error getting keys", pattern=pattern, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting keys", pattern=pattern, error=str(e))
            raise

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            result = await self._client.incrby(key, amount)
            logger.debug("Incremented key", key=key, amount=amount, new_value=result)
            return result

        except redis.RedisError as e:
            logger.error("Redis error incrementing key", key=key, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error incrementing key", key=key, error=str(e))
            raise

    async def set_hash(self, key: str, field: str, value: Any) -> bool:
        """Set a hash field."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            # Serialize complex objects to JSON
            serialized_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

            result = await self._client.hset(key, field, serialized_value)
            logger.debug("Set hash field", key=key, field=field)
            return bool(result)

        except redis.RedisError as e:
            logger.error("Redis error setting hash field", key=key, field=field, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error setting hash field", key=key, field=field, error=str(e))
            raise

    async def get_hash(self, key: str, field: str) -> Any | None:
        """Get a hash field value."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            value = await self._client.hget(key, field)
            if value is None:
                return None

            # Try to deserialize JSON, fall back to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except redis.RedisError as e:
            logger.error("Redis error getting hash field", key=key, field=field, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting hash field", key=key, field=field, error=str(e))
            raise
