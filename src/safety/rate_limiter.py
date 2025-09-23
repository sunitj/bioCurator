"""Token bucket rate limiter for safety controls."""

import time
from threading import Lock
from typing import Optional, Dict, Any

from pydantic import BaseModel

from ..logging import get_logger
from .event_bus import SafetyEventType, emit_safety_event

logger = get_logger(__name__)


class RateLimiterConfig(BaseModel):
    """Rate limiter configuration."""

    capacity: int = 60              # Maximum tokens in bucket
    refill_rate: float = 1.0        # Tokens per second
    per_minute_limit: int = 60      # Human-readable limit
    burst_capacity: int = 10        # Extra tokens for bursts


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, limiter_name: str, retry_after: float):
        self.limiter_name = limiter_name
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{limiter_name}'. Retry after {retry_after:.1f} seconds"
        )


class TokenBucket:
    """Thread-safe token bucket implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket."""
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self._lock:
            now = time.time()

            # Add tokens based on time elapsed
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def tokens_available(self) -> float:
        """Get current number of tokens available."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            return min(self.capacity, self._tokens + elapsed * self.refill_rate)

    def time_until_tokens(self, tokens: int = 1) -> float:
        """Get time in seconds until specified tokens are available."""
        with self._lock:
            available = self.tokens_available()
            if available >= tokens:
                return 0.0

            needed = tokens - available
            return needed / self.refill_rate


class RateLimiter:
    """Rate limiter with per-agent and global quotas."""

    def __init__(self, name: str, config: RateLimiterConfig):
        """Initialize rate limiter."""
        self.name = name
        self.config = config

        # Global bucket
        self._global_bucket = TokenBucket(
            capacity=config.capacity + config.burst_capacity,
            refill_rate=config.refill_rate
        )

        # Per-agent buckets
        self._agent_buckets: Dict[str, TokenBucket] = {}
        self._agent_lock = Lock()

        # Stats
        self._total_requests = 0
        self._blocked_requests = 0
        self._lock = Lock()

    def _get_agent_bucket(self, agent_id: str) -> TokenBucket:
        """Get or create bucket for specific agent."""
        with self._agent_lock:
            if agent_id not in self._agent_buckets:
                self._agent_buckets[agent_id] = TokenBucket(
                    capacity=self.config.capacity,
                    refill_rate=self.config.refill_rate
                )
            return self._agent_buckets[agent_id]

    def allow_request(self, agent_id: Optional[str] = None, tokens: int = 1) -> bool:
        """Check if request is allowed and consume tokens if so."""
        with self._lock:
            self._total_requests += 1

        # Check global limit first
        if not self._global_bucket.consume(tokens):
            retry_after = self._global_bucket.time_until_tokens(tokens)
            self._handle_rate_limit_exceeded("global", retry_after, agent_id)
            return False

        # Check per-agent limit if agent specified
        if agent_id:
            agent_bucket = self._get_agent_bucket(agent_id)
            if not agent_bucket.consume(tokens):
                retry_after = agent_bucket.time_until_tokens(tokens)
                self._handle_rate_limit_exceeded(f"agent_{agent_id}", retry_after, agent_id)
                return False

        return True

    def _handle_rate_limit_exceeded(
        self, quota_type: str, retry_after: float, agent_id: Optional[str]
    ) -> None:
        """Handle rate limit exceeded event."""
        with self._lock:
            self._blocked_requests += 1

        emit_safety_event(
            event_type=SafetyEventType.RATE_LIMIT_BLOCK,
            component=f"rate_limiter.{self.name}",
            message=f"Rate limit exceeded for {quota_type}",
            agent_id=agent_id,
            metadata={
                "quota_type": quota_type,
                "retry_after": retry_after,
                "limiter_config": self.config.model_dump(),
            },
            severity="WARNING"
        )

        logger.warning(
            "Rate limit exceeded",
            limiter=self.name,
            quota_type=quota_type,
            agent_id=agent_id,
            retry_after=retry_after
        )

    def check_request(self, agent_id: Optional[str] = None, tokens: int = 1) -> None:
        """Check if request is allowed, raise exception if not."""
        if not self.allow_request(agent_id, tokens):
            # Calculate retry time
            global_retry = self._global_bucket.time_until_tokens(tokens)
            retry_after = global_retry

            if agent_id:
                agent_bucket = self._get_agent_bucket(agent_id)
                agent_retry = agent_bucket.time_until_tokens(tokens)
                retry_after = max(global_retry, agent_retry)

            raise RateLimitExceededError(self.name, retry_after)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            global_tokens = self._global_bucket.tokens_available()

            agent_stats = {}
            with self._agent_lock:
                for agent_id, bucket in self._agent_buckets.items():
                    agent_stats[agent_id] = {
                        "tokens_available": bucket.tokens_available(),
                        "capacity": bucket.capacity,
                    }

            block_rate = (
                self._blocked_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            )

            return {
                "name": self.name,
                "global_tokens_available": global_tokens,
                "global_capacity": self._global_bucket.capacity,
                "total_requests": self._total_requests,
                "blocked_requests": self._blocked_requests,
                "block_rate": block_rate,
                "agent_buckets": agent_stats,
                "config": self.config.model_dump(),
            }

    def reset(self) -> None:
        """Reset rate limiter (useful for testing)."""
        with self._lock:
            self._global_bucket = TokenBucket(
                capacity=self.config.capacity + self.config.burst_capacity,
                refill_rate=self.config.refill_rate
            )
            with self._agent_lock:
                self._agent_buckets.clear()
            self._total_requests = 0
            self._blocked_requests = 0