"""Cost tracking and budget enforcement for safety controls."""

import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel

from ..logging import get_logger
from .event_bus import SafetyEventType, emit_safety_event

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    """AI model providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class CostViolationError(Exception):
    """Raised when cost budget is exceeded."""

    def __init__(self, current_cost: float, budget: float):
        self.current_cost = current_cost
        self.budget = budget
        super().__init__(f"Cost budget exceeded: ${current_cost:.4f} > ${budget:.4f}")


class PriceCatalog(ABC):
    """Abstract base class for price catalogs."""

    @abstractmethod
    def get_price(self, provider: ModelProvider, model: str, tokens: int) -> float:
        """Get price for model usage."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get catalog name."""
        pass


class MockLocalPriceCatalog(PriceCatalog):
    """Mock pricing for development mode (always returns 0.0)."""

    def get_price(self, provider: ModelProvider, model: str, tokens: int) -> float:
        """Return zero cost for all local models."""
        return 0.0

    def get_name(self) -> str:
        """Get catalog name."""
        return "mock_local"


class CloudReferencePriceCatalog(PriceCatalog):
    """Reference pricing for cloud providers."""

    def __init__(self):
        """Initialize with reference prices per 1K tokens."""
        self._prices = {
            ModelProvider.OPENAI: {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            },
            ModelProvider.ANTHROPIC: {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
        }

    def get_price(self, provider: ModelProvider, model: str, tokens: int) -> float:
        """Get price for model usage."""
        if provider not in self._prices:
            logger.warning(f"No pricing available for provider: {provider}")
            return 0.0

        provider_prices = self._prices[provider]
        if model not in provider_prices:
            logger.warning(f"No pricing available for model: {model}")
            return 0.0

        # Estimate input/output split (rough approximation)
        input_tokens = int(tokens * 0.7)  # Assume 70% input
        output_tokens = tokens - input_tokens

        model_prices = provider_prices[model]
        cost = (
            (input_tokens / 1000) * model_prices["input"] +
            (output_tokens / 1000) * model_prices["output"]
        )

        return cost

    def get_name(self) -> str:
        """Get catalog name."""
        return "cloud_reference"


class CostUsage(BaseModel):
    """Cost usage record."""

    timestamp: float
    provider: str
    model: str
    tokens: int
    cost: float
    agent_id: Optional[str] = None
    session_id: Optional[str] = None


class CostTracker:
    """Cost tracking with budget enforcement."""

    def __init__(
        self,
        budget: float,
        price_catalog: PriceCatalog,
        warning_threshold: float = 0.8
    ):
        """Initialize cost tracker."""
        self.budget = budget
        self.price_catalog = price_catalog
        self.warning_threshold = warning_threshold

        # Usage tracking
        self._total_cost = 0.0
        self._usage_records: list[CostUsage] = []

        # Per-agent tracking
        self._agent_costs: Dict[str, float] = {}

        # Per-session tracking
        self._session_costs: Dict[str, float] = {}

        # Warning state
        self._warning_sent = False

        # Thread safety
        self._lock = Lock()

    def record_usage(
        self,
        provider: ModelProvider,
        model: str,
        tokens: int,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> float:
        """Record model usage and return cost."""
        cost = self.price_catalog.get_price(provider, model, tokens)

        with self._lock:
            # Create usage record
            usage = CostUsage(
                timestamp=time.time(),
                provider=provider.value,
                model=model,
                tokens=tokens,
                cost=cost,
                agent_id=agent_id,
                session_id=session_id
            )
            self._usage_records.append(usage)

            # Update totals
            self._total_cost += cost

            if agent_id:
                self._agent_costs[agent_id] = self._agent_costs.get(agent_id, 0.0) + cost

            if session_id:
                self._session_costs[session_id] = self._session_costs.get(session_id, 0.0) + cost

            # Check thresholds
            self._check_thresholds()

            logger.info(
                "Model usage recorded",
                provider=provider.value,
                model=model,
                tokens=tokens,
                cost=cost,
                total_cost=self._total_cost,
                budget=self.budget,
                agent_id=agent_id,
                session_id=session_id
            )

            return cost

    def _check_thresholds(self) -> None:
        """Check cost thresholds and emit events."""
        budget_utilization = self._total_cost / self.budget if self.budget > 0 else 0.0

        # Check warning threshold
        if (budget_utilization >= self.warning_threshold and
            not self._warning_sent and
            self.budget > 0):

            self._warning_sent = True
            emit_safety_event(
                event_type=SafetyEventType.COST_BUDGET_WARNING,
                component="cost_tracker",
                message=f"Cost budget warning: ${self._total_cost:.4f} / ${self.budget:.4f} ({budget_utilization:.1%})",
                metadata={
                    "total_cost": self._total_cost,
                    "budget": self.budget,
                    "utilization": budget_utilization,
                    "price_catalog": self.price_catalog.get_name(),
                },
                severity="WARNING"
            )

        # Check budget violation
        if self._total_cost > self.budget and self.budget > 0:
            emit_safety_event(
                event_type=SafetyEventType.COST_BUDGET_VIOLATION,
                component="cost_tracker",
                message=f"Cost budget exceeded: ${self._total_cost:.4f} > ${self.budget:.4f}",
                metadata={
                    "total_cost": self._total_cost,
                    "budget": self.budget,
                    "overage": self._total_cost - self.budget,
                    "price_catalog": self.price_catalog.get_name(),
                },
                severity="ERROR"
            )

    def check_budget(self, estimated_cost: float = 0.0) -> None:
        """Check if operation would exceed budget."""
        with self._lock:
            projected_cost = self._total_cost + estimated_cost
            if projected_cost > self.budget and self.budget > 0:
                raise CostViolationError(projected_cost, self.budget)

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        with self._lock:
            if self.budget == 0:
                return 0.0
            return max(0.0, self.budget - self._total_cost)

    def get_total_cost(self) -> float:
        """Get total cost so far."""
        with self._lock:
            return self._total_cost

    def get_agent_cost(self, agent_id: str) -> float:
        """Get cost for specific agent."""
        with self._lock:
            return self._agent_costs.get(agent_id, 0.0)

    def get_session_cost(self, session_id: str) -> float:
        """Get cost for specific session."""
        with self._lock:
            return self._session_costs.get(session_id, 0.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get cost tracking statistics."""
        with self._lock:
            budget_utilization = (
                self._total_cost / self.budget if self.budget > 0 else 0.0
            )

            return {
                "total_cost": self._total_cost,
                "budget": self.budget,
                "remaining_budget": self.get_remaining_budget(),
                "budget_utilization": budget_utilization,
                "usage_count": len(self._usage_records),
                "agent_costs": dict(self._agent_costs),
                "session_costs": dict(self._session_costs),
                "price_catalog": self.price_catalog.get_name(),
                "warning_sent": self._warning_sent,
            }