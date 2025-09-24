"""Base agent class with integrated safety controls."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ..config.schemas import AppMode, ConfigSchema
from ..logging import get_logger
from ..memory.interfaces import MemoryManager
from ..memory.manager import DefaultMemoryManager
from ..safety.behavior_monitor import BehaviorMonitor
from ..safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..safety.cost_tracker import CostTracker
from ..safety.event_bus import SafetyEvent, SafetyEventBus, SafetyEventType
from ..safety.rate_limiter import RateLimiter, RateLimiterConfig
from .config import AgentConfig, AgentSafetyConfig, MemoryPermission

logger = get_logger(__name__)


class AgentTask(BaseModel):
    """Represents a task for an agent to execute."""

    id: str
    agent_id: str
    task_type: str
    input_data: Dict[str, Any]
    created_at: datetime
    priority: int = 0
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Represents an agent's response to a task."""

    task_id: str
    agent_id: str
    success: bool
    output_data: Dict[str, Any] = {}
    error_message: Optional[str] = None
    execution_time_seconds: float
    metadata: Dict[str, Any] = {}


class AgentStatus(BaseModel):
    """Represents current agent status."""

    agent_id: str
    is_active: bool
    current_tasks: int
    max_concurrent_tasks: int
    total_tasks_completed: int
    total_tasks_failed: int
    last_activity: datetime
    health_status: str
    memory_usage_mb: Optional[float] = None


class BaseAgent(ABC):
    """Base class for all agents with integrated safety controls."""

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        system_config: ConfigSchema,
        memory_manager: MemoryManager,
        event_bus: Optional[SafetyEventBus] = None,
    ):
        """Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
            config: Agent-specific configuration
            system_config: System-wide configuration
            memory_manager: Memory system interface
            event_bus: Event bus for safety events
        """
        self.agent_id = agent_id
        self.config = config
        self.system_config = system_config
        self.memory_manager = memory_manager
        self.event_bus = event_bus or SafetyEventBus()

        # Initialize safety controls
        self._init_safety_controls()

        # Agent state
        self._is_active = False
        self._current_tasks: Dict[str, AgentTask] = {}
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._total_tasks_completed = 0
        self._total_tasks_failed = 0
        self._last_activity = datetime.now()

        logger.info(
            f"Initialized agent {agent_id}",
            agent_id=agent_id,
            role=config.role,
            max_tasks=config.max_concurrent_tasks,
        )

    def _init_safety_controls(self) -> None:
        """Initialize safety control systems."""
        # Circuit breaker configuration
        cb_config = CircuitBreakerConfig(
            error_threshold=self.config.safety.circuit_breaker_threshold,
            window_duration=self.config.safety.circuit_breaker_window,
            min_volume=self.config.safety.circuit_breaker_min_volume,
        )
        self.circuit_breaker = CircuitBreaker(f"agent_{self.agent_id}", cb_config)

        # Rate limiter configuration
        rl_config = RateLimiterConfig(
            max_requests=self.config.safety.max_requests_per_minute,
            time_window=60,  # 1 minute
            burst_capacity=self.config.safety.burst_capacity,
            refill_rate=self.config.safety.refill_rate,
        )
        self.rate_limiter = RateLimiter(f"agent_{self.agent_id}", rl_config)

        # Cost tracker
        self.cost_tracker = CostTracker(
            max_budget=self.config.safety.max_cost_budget_per_hour, time_window_hours=1
        )

        # Behavior monitor
        self.behavior_monitor = BehaviorMonitor(f"agent_{self.agent_id}")

        logger.info(f"Safety controls initialized for agent {self.agent_id}")

    async def start(self) -> None:
        """Start the agent."""
        if self._is_active:
            logger.warning(f"Agent {self.agent_id} is already active")
            return

        logger.info(f"Starting agent {self.agent_id}")

        try:
            # Verify memory system access
            await self._verify_memory_access()

            # Perform agent-specific startup
            await self._startup()

            self._is_active = True
            self._last_activity = datetime.now()

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.AGENT_STARTED,
                    component=self.agent_id,
                    message=f"Agent {self.agent_id} started successfully",
                    metadata={"agent_id": self.agent_id, "role": self.config.role},
                )
            )

            logger.info(f"Agent {self.agent_id} started successfully")

        except Exception as e:
            logger.error(f"Failed to start agent {self.agent_id}", error=str(e))
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.AGENT_FAILURE,
                    component=self.agent_id,
                    message=f"Failed to start agent {self.agent_id}: {str(e)}",
                    metadata={"agent_id": self.agent_id, "error": str(e)},
                )
            )
            raise

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        if not self._is_active:
            logger.warning(f"Agent {self.agent_id} is not active")
            return

        logger.info(f"Stopping agent {self.agent_id}")

        try:
            # Wait for current tasks to complete (with timeout)
            if self._current_tasks:
                logger.info(
                    f"Waiting for {len(self._current_tasks)} tasks to complete",
                    agent_id=self.agent_id,
                )
                await asyncio.wait_for(
                    self._wait_for_tasks_completion(), timeout=30.0  # 30 second timeout
                )

            # Perform agent-specific shutdown
            await self._shutdown()

            self._is_active = False

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.AGENT_STOPPED,
                    component=self.agent_id,
                    message=f"Agent {self.agent_id} stopped successfully",
                    metadata={"agent_id": self.agent_id},
                )
            )

            logger.info(f"Agent {self.agent_id} stopped successfully")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks to complete for agent {self.agent_id}")
            # Force stop
            self._current_tasks.clear()
            self._is_active = False
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}", error=str(e))
            self._is_active = False
            raise

    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a task with safety controls.

        Args:
            task: Task to execute

        Returns:
            AgentResponse: Task execution result

        Raises:
            RuntimeError: If agent is not active or safety controls prevent execution
        """
        if not self._is_active:
            raise RuntimeError(f"Agent {self.agent_id} is not active")

        start_time = datetime.now()

        # Check safety controls
        await self._check_safety_controls(task)

        # Acquire task semaphore (limits concurrent tasks)
        async with self._task_semaphore:
            try:
                # Track current task
                self._current_tasks[task.id] = task
                self._last_activity = datetime.now()

                # Execute the task with circuit breaker protection
                async with self.circuit_breaker:
                    result = await self._execute_task_impl(task)

                # Update success metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self._total_tasks_completed += 1

                # Record successful execution for behavior monitoring
                self.behavior_monitor.record_request(
                    task.task_type, execution_time, success=True
                )

                return AgentResponse(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    success=True,
                    output_data=result,
                    execution_time_seconds=execution_time,
                )

            except Exception as e:
                # Update failure metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self._total_tasks_failed += 1

                # Record failed execution for behavior monitoring
                self.behavior_monitor.record_request(
                    task.task_type, execution_time, success=False
                )

                # Emit safety event for task failure
                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.AGENT_TASK_FAILURE,
                        component=self.agent_id,
                        message=f"Task {task.id} failed: {str(e)}",
                        metadata={
                            "agent_id": self.agent_id,
                            "task_id": task.id,
                            "task_type": task.task_type,
                            "error": str(e),
                        },
                    )
                )

                return AgentResponse(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    success=False,
                    error_message=str(e),
                    execution_time_seconds=execution_time,
                )

            finally:
                # Remove task from current tasks
                self._current_tasks.pop(task.id, None)

    async def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return AgentStatus(
            agent_id=self.agent_id,
            is_active=self._is_active,
            current_tasks=len(self._current_tasks),
            max_concurrent_tasks=self.config.max_concurrent_tasks,
            total_tasks_completed=self._total_tasks_completed,
            total_tasks_failed=self._total_tasks_failed,
            last_activity=self._last_activity,
            health_status="healthy" if self._is_active else "stopped",
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health_info = {
            "agent_id": self.agent_id,
            "is_active": self._is_active,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "current_tasks": len(self._current_tasks),
            "rate_limit_available": self.rate_limiter.get_available_tokens(),
            "memory_access": "unknown",
        }

        # Check memory system access
        if self.memory_manager:
            try:
                memory_health = await self.memory_manager.health_check_all()
                all_healthy = all(status.is_healthy for status in memory_health.values())
                health_info["memory_access"] = "healthy" if all_healthy else "degraded"
            except Exception as e:
                logger.error(f"Memory health check failed for agent {self.agent_id}", error=str(e))
                health_info["memory_access"] = "failed"

        return health_info

    # Abstract methods for subclasses
    @abstractmethod
    async def _startup(self) -> None:
        """Agent-specific startup logic."""
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """Agent-specific shutdown logic."""
        pass

    @abstractmethod
    async def _execute_task_impl(self, task: AgentTask) -> Dict[str, Any]:
        """Agent-specific task execution logic.

        Args:
            task: Task to execute

        Returns:
            Dict[str, Any]: Task execution result
        """
        pass

    # Private helper methods
    async def _verify_memory_access(self) -> None:
        """Verify agent has appropriate memory access."""
        if not self.memory_manager:
            raise RuntimeError("Memory manager not available")

        # Check if memory system is initialized
        if not hasattr(self.memory_manager, "is_initialized") or not self.memory_manager.is_initialized:
            raise RuntimeError("Memory system not initialized")

        # Verify access to required memory backends based on permissions
        permissions = self.config.memory_permissions
        if not permissions:
            logger.warning(f"No memory permissions configured for agent {self.agent_id}")
            return

        # Test access to each configured backend
        try:
            if "knowledge_graph" in permissions:
                self.memory_manager.get_knowledge_graph()

            if "vector_store" in permissions:
                self.memory_manager.get_vector_store()

            if "episodic_memory" in permissions:
                self.memory_manager.get_episodic_memory()

            if "working_memory" in permissions:
                self.memory_manager.get_working_memory()

            if "time_series" in permissions:
                # Time series is optional
                self.memory_manager.get_time_series()

        except Exception as e:
            logger.error(
                f"Memory access verification failed for agent {self.agent_id}", error=str(e)
            )
            raise RuntimeError(f"Memory access verification failed: {str(e)}")

    async def _check_safety_controls(self, task: AgentTask) -> None:
        """Check safety controls before task execution."""
        # Check circuit breaker
        if self.circuit_breaker.is_open:
            raise RuntimeError(f"Circuit breaker is open for agent {self.agent_id}")

        # Check rate limiting
        if not await self.rate_limiter.acquire():
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.RATE_LIMIT_BLOCK,
                    component=self.agent_id,
                    message=f"Rate limit exceeded for agent {self.agent_id}",
                    metadata={"agent_id": self.agent_id, "task_id": task.id},
                )
            )
            raise RuntimeError(f"Rate limit exceeded for agent {self.agent_id}")

        # Check cost budget (in development mode, should always be 0)
        if self.system_config.app_mode == AppMode.DEVELOPMENT:
            if self.config.safety.max_cost_budget_per_hour > 0:
                raise RuntimeError("Development mode must have zero cost budget")

        # Check behavior monitoring for anomalies
        if self.behavior_monitor.is_anomalous():
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.ANOMALY_DETECTED,
                    component=self.agent_id,
                    message=f"Anomalous behavior detected for agent {self.agent_id}",
                    metadata={"agent_id": self.agent_id, "task_id": task.id},
                )
            )
            # Note: We don't block on anomalies, just log them

    async def _wait_for_tasks_completion(self) -> None:
        """Wait for all current tasks to complete."""
        while self._current_tasks:
            await asyncio.sleep(0.1)