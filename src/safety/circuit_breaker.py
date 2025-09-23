"""Circuit breaker implementation for safety controls."""

import time
from enum import Enum
from threading import Lock
from typing import Callable, Optional, TypeVar, Any
from collections import deque
from datetime import datetime, timedelta

from pydantic import BaseModel

from ..logging import get_logger
from .event_bus import SafetyEventType, emit_safety_event

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking all requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    error_threshold: float = 0.5        # Error rate threshold (0.0-1.0)
    window_duration: int = 300          # Rolling window in seconds
    min_volume: int = 10                # Minimum requests before evaluation
    recovery_timeout: int = 60          # Seconds to wait before half-open
    max_probes: int = 3                 # Max probe attempts in half-open
    probe_interval: int = 10            # Seconds between probes


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitBreakerState):
        self.name = name
        self.state = state
        super().__init__(f"Circuit breaker '{name}' is {state.value}")


class CircuitBreaker:
    """Thread-safe circuit breaker implementation."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._probe_count = 0
        self._last_probe_time: Optional[float] = None
        self._lock = Lock()

        # Rolling window for tracking requests
        self._request_times: deque = deque()
        self._failure_times: deque = deque()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    @property
    def failure_rate(self) -> float:
        """Get current failure rate in the rolling window."""
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)

            if len(self._request_times) < self.config.min_volume:
                return 0.0

            return len(self._failure_times) / len(self._request_times)

    def _cleanup_old_requests(self, now: float) -> None:
        """Remove requests outside the rolling window."""
        cutoff = now - self.config.window_duration

        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        now = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            return True

        elif self._state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if (self._last_failure_time and
                now - self._last_failure_time >= self.config.recovery_timeout):
                self._transition_to_half_open()
                return True
            return False

        else:  # HALF_OPEN
            # Allow probes with interval
            if (self._last_probe_time is None or
                now - self._last_probe_time >= self.config.probe_interval):
                if self._probe_count < self.config.max_probes:
                    self._last_probe_time = now
                    return True
            return False

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._probe_count = 0
        self._last_probe_time = None

        emit_safety_event(
            event_type=SafetyEventType.CIRCUIT_PROBE,
            component=f"circuit_breaker.{self.name}",
            message=f"Circuit breaker '{self.name}' transitioning to half-open for probing",
            metadata={
                "state": self._state.value,
                "failure_rate": self.failure_rate,
                "config": self.config.model_dump(),
            }
        )

        logger.info(
            "Circuit breaker transitioning to half-open",
            name=self.name,
            failure_rate=self.failure_rate
        )

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitBreakerState.OPEN
        self._last_failure_time = time.time()

        emit_safety_event(
            event_type=SafetyEventType.CIRCUIT_TRIP,
            component=f"circuit_breaker.{self.name}",
            message=f"Circuit breaker '{self.name}' tripped due to high error rate",
            metadata={
                "state": self._state.value,
                "failure_rate": self.failure_rate,
                "config": self.config.model_dump(),
            },
            severity="WARNING"
        )

        logger.warning(
            "Circuit breaker tripped",
            name=self.name,
            failure_rate=self.failure_rate,
            threshold=self.config.error_threshold
        )

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitBreakerState.CLOSED
        self._probe_count = 0
        self._last_probe_time = None

        emit_safety_event(
            event_type=SafetyEventType.CIRCUIT_RECOVERY,
            component=f"circuit_breaker.{self.name}",
            message=f"Circuit breaker '{self.name}' recovered and closed",
            metadata={
                "state": self._state.value,
                "failure_rate": self.failure_rate,
            }
        )

        logger.info(
            "Circuit breaker recovered",
            name=self.name,
            failure_rate=self.failure_rate
        )

    def _record_success(self) -> None:
        """Record a successful request."""
        now = time.time()

        with self._lock:
            self._request_times.append(now)
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._probe_count += 1
                # If we've had enough successful probes, recover
                if self._probe_count >= self.config.max_probes:
                    self._transition_to_closed()

    def _record_failure(self) -> None:
        """Record a failed request."""
        now = time.time()

        with self._lock:
            self._request_times.append(now)
            self._failure_times.append(now)
            self._failure_count += 1
            self._cleanup_old_requests(now)

            # Check if we should trip the circuit breaker
            if (self._state == CircuitBreakerState.CLOSED and
                len(self._request_times) >= self.config.min_volume and
                self.failure_rate >= self.config.error_threshold):
                self._transition_to_open()
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Failure during probing - back to open
                self._transition_to_open()

    def call(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if not self._should_allow_request():
                raise CircuitBreakerError(self.name, self._state)

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker protection."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)

            return {
                "name": self.name,
                "state": self._state.value,
                "failure_rate": self.failure_rate,
                "total_requests": len(self._request_times),
                "total_failures": len(self._failure_times),
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "probe_count": self._probe_count,
                "config": self.config.model_dump(),
            }