"""Test Redis client implementation."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.redis_client import RedisClient
from src.memory.interfaces import HealthStatus


class TestRedisClient:
    """Test RedisClient implementation."""

    @pytest.fixture
    def config(self):
        """Redis configuration for testing."""
        return {
            "redis_url": "redis://localhost:6379/0",
            "redis_pool_size": 10,
            "redis_socket_timeout": 5
        }

    def test_initialization(self, config):
        """Test client initialization."""
        client = RedisClient(config)

        assert client.url == "redis://localhost:6379/0"
        assert client.pool_size == 10
        assert client.socket_timeout == 5
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, config):
        """Test successful connection."""
        client = RedisClient(config)

        with patch('redis.asyncio.ConnectionPool') as mock_pool_class:
            with patch('redis.asyncio.Redis') as mock_redis_class:
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_redis = AsyncMock()
                mock_redis.ping.return_value = True
                mock_redis_class.return_value = mock_redis

                await client.connect()

                assert client.is_connected is True
                mock_pool_class.from_url.assert_called_once_with(
                    "redis://localhost:6379/0",
                    max_connections=10,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )

    @pytest.mark.asyncio
    async def test_connect_failure(self, config):
        """Test connection failure."""
        client = RedisClient(config)

        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis

            with pytest.raises(Exception, match="Connection failed"):
                await client.connect()

            assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config):
        """Test health check when healthy."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            "used_memory": 1024 * 1024,  # 1MB
            "connected_clients": 5
        }
        client._client = mock_redis

        status = await client.health_check()

        assert status.is_healthy is True
        assert status.message == "Redis is healthy"
        assert status.latency_ms is not None
        assert status.connection_count == 5
        assert status.memory_usage_mb == 1.0

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, config):
        """Test health check when not connected."""
        client = RedisClient(config)

        status = await client.health_check()

        assert status.is_healthy is False
        assert status.message == "Not connected to Redis"

    @pytest.mark.asyncio
    async def test_set_and_get_simple_value(self, config):
        """Test setting and getting simple values."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.get.return_value = "test_value"
        client._client = mock_redis

        # Test set
        result = await client.set("test_key", "test_value", expire_seconds=60)
        assert result is True
        mock_redis.set.assert_called_once_with("test_key", "test_value", ex=60)

        # Test get
        value = await client.get("test_key")
        assert value == "test_value"
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_and_get_json_value(self, config):
        """Test setting and getting JSON values."""
        client = RedisClient(config)
        client._connected = True

        test_data = {"key": "value", "number": 123}
        json_data = json.dumps(test_data)

        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.get.return_value = json_data
        client._client = mock_redis

        # Test set with dict
        result = await client.set("test_key", test_data)
        assert result is True
        mock_redis.set.assert_called_once_with("test_key", json_data, ex=None)

        # Test get returns parsed JSON
        value = await client.get("test_key")
        assert value == test_data

    @pytest.mark.asyncio
    async def test_delete(self, config):
        """Test deleting keys."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1  # Key existed
        client._client = mock_redis

        result = await client.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists(self, config):
        """Test checking key existence."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        client._client = mock_redis

        result = await client.exists("test_key")
        assert result is True
        mock_redis.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_expire(self, config):
        """Test setting key expiration."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.expire.return_value = 1  # Key exists
        client._client = mock_redis

        result = await client.expire("test_key", 300)
        assert result is True
        mock_redis.expire.assert_called_once_with("test_key", 300)

    @pytest.mark.asyncio
    async def test_get_all_keys(self, config):
        """Test getting all keys with pattern."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.keys.return_value = ["key1", "key2", "key3"]
        client._client = mock_redis

        keys = await client.get_all_keys("key*")
        assert keys == ["key1", "key2", "key3"]
        mock_redis.keys.assert_called_once_with("key*")

    @pytest.mark.asyncio
    async def test_increment(self, config):
        """Test incrementing numeric values."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.incrby.return_value = 5
        client._client = mock_redis

        result = await client.increment("counter", 2)
        assert result == 5
        mock_redis.incrby.assert_called_once_with("counter", 2)

    @pytest.mark.asyncio
    async def test_hash_operations(self, config):
        """Test hash field operations."""
        client = RedisClient(config)
        client._connected = True

        mock_redis = AsyncMock()
        mock_redis.hset.return_value = 1
        mock_redis.hget.return_value = '{"nested": "value"}'
        client._client = mock_redis

        # Test set hash field
        test_data = {"nested": "value"}
        result = await client.set_hash("hash_key", "field", test_data)
        assert result is True

        # Test get hash field
        value = await client.get_hash("hash_key", "field")
        assert value == test_data

    @pytest.mark.asyncio
    async def test_not_connected_errors(self, config):
        """Test operations when not connected."""
        client = RedisClient(config)

        with pytest.raises(RuntimeError, match="Not connected to Redis"):
            await client.set("key", "value")

        with pytest.raises(RuntimeError, match="Not connected to Redis"):
            await client.get("key")

        with pytest.raises(RuntimeError, match="Not connected to Redis"):
            await client.delete("key")

        with pytest.raises(RuntimeError, match="Not connected to Redis"):
            await client.ping()