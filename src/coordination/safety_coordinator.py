"""Safety-aware coordination for multi-agent systems."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..logging import get_logger
from ..safety.circuit_breaker import CircuitBreakerState
from ..safety.event_bus import SafetyEvent, SafetyEventBus, SafetyEventType
from .task_queue import TaskQueue, TaskStatus, TaskDefinition

logger = get_logger(__name__)


class SafetyCoordinator:
    """Coordinates multi-agent activities with safety oversight."""

    def __init__(
        self,
        task_queue: TaskQueue,
        event_bus: Optional[SafetyEventBus] = None,
        max_concurrent_agents: int = 3,
    ):
        """Initialize safety coordinator.

        Args:
            task_queue: Task queue for coordination
            event_bus: Event bus for safety events
            max_concurrent_agents: Maximum number of agents that can run concurrently
        """
        self.task_queue = task_queue
        self.event_bus = event_bus or SafetyEventBus()
        self.max_concurrent_agents = max_concurrent_agents

        # Agent tracking
        self._active_agents: Dict[str, Dict[str, Any]] = {}
        self._agent_performance: Dict[str, Dict[str, float]] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Safety thresholds
        self._error_rate_threshold = 0.5
        self._response_time_threshold = 30.0  # seconds
        self._resource_usage_threshold = 0.8

        # Coordination state
        self._coordination_active = False
        self._safety_monitoring_task: Optional[asyncio.Task] = None

        logger.info("Safety coordinator initialized", max_concurrent=max_concurrent_agents)

    async def start(self) -> None:
        """Start safety coordination."""
        if self._coordination_active:
            logger.warning("Safety coordinator already active")
            return

        logger.info("Starting safety coordinator")

        try:
            # Start safety monitoring
            self._safety_monitoring_task = asyncio.create_task(self._safety_monitoring_loop())

            self._coordination_active = True

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.COORDINATION_STARTED,
                    component="safety_coordinator",
                    message="Safety coordinator started",
                    metadata={"max_concurrent_agents": self.max_concurrent_agents},
                )
            )

            logger.info("Safety coordinator started successfully")

        except Exception as e:
            logger.error("Failed to start safety coordinator", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop safety coordination."""
        if not self._coordination_active:
            logger.warning("Safety coordinator not active")
            return

        logger.info("Stopping safety coordinator")

        try:
            # Stop monitoring
            if self._safety_monitoring_task and not self._safety_monitoring_task.done():
                self._safety_monitoring_task.cancel()
                try:
                    await self._safety_monitoring_task
                except asyncio.CancelledError:
                    pass

            self._coordination_active = False

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.COORDINATION_STOPPED,
                    component="safety_coordinator",
                    message="Safety coordinator stopped",
                    metadata={},
                )
            )

            logger.info("Safety coordinator stopped")

        except Exception as e:
            logger.error("Error stopping safety coordinator", error=str(e))

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        safety_config: Dict[str, Any]
    ) -> None:
        """Register an agent for coordination.

        Args:
            agent_id: Agent identifier
            capabilities: List of agent capabilities
            safety_config: Agent safety configuration
        """
        logger.info(f"Registering agent for coordination", agent_id=agent_id, capabilities=capabilities)

        self._active_agents[agent_id] = {
            "capabilities": capabilities,
            "safety_config": safety_config,
            "registered_at": datetime.now(),
            "status": "active",
            "current_tasks": 0,
            "total_tasks": 0,
            "total_errors": 0,
            "last_activity": datetime.now(),
        }

        # Initialize performance tracking
        self._agent_performance[agent_id] = {
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "success_rate": 1.0,
            "tasks_per_minute": 0.0,
        }

        # Initialize circuit breaker state
        self._circuit_breakers[agent_id] = CircuitBreakerState.CLOSED

        self.event_bus.emit(
            SafetyEvent(
                event_type=SafetyEventType.AGENT_REGISTERED,
                component="safety_coordinator",
                message=f"Agent {agent_id} registered for coordination",
                metadata={"agent_id": agent_id, "capabilities": capabilities},
            )
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from coordination.

        Args:
            agent_id: Agent identifier
        """
        logger.info(f"Unregistering agent from coordination", agent_id=agent_id)

        if agent_id in self._active_agents:
            del self._active_agents[agent_id]

        if agent_id in self._agent_performance:
            del self._agent_performance[agent_id]

        if agent_id in self._circuit_breakers:
            del self._circuit_breakers[agent_id]

        self.event_bus.emit(
            SafetyEvent(
                event_type=SafetyEventType.AGENT_UNREGISTERED,
                component="safety_coordinator",
                message=f"Agent {agent_id} unregistered from coordination",
                metadata={"agent_id": agent_id},
            )
        )

    async def coordinate_task_execution(
        self,
        task: TaskDefinition,
        preferred_agent: Optional[str] = None
    ) -> str:
        """Coordinate task execution with safety oversight.

        Args:
            task: Task to execute
            preferred_agent: Preferred agent for execution

        Returns:
            str: Assigned agent ID

        Raises:
            RuntimeError: If no suitable agent available or safety constraints violated
        """
        logger.info(f"Coordinating task execution", task_id=task.id, task_type=task.task_type)

        # Check global safety constraints
        await self._check_global_safety_constraints()

        # Find suitable agent
        agent_id = await self._select_agent_for_task(task, preferred_agent)

        if not agent_id:
            raise RuntimeError("No suitable agent available for task")

        # Check agent-specific safety constraints
        await self._check_agent_safety_constraints(agent_id, task)

        # Submit task to queue
        try:
            task_id = await self.task_queue.submit_task(task)

            # Update agent tracking
            if agent_id in self._active_agents:
                self._active_agents[agent_id]["current_tasks"] += 1
                self._active_agents[agent_id]["total_tasks"] += 1
                self._active_agents[agent_id]["last_activity"] = datetime.now()

            logger.info(f"Task coordinated successfully", task_id=task_id, agent_id=agent_id)
            return agent_id

        except Exception as e:
            logger.error(f"Failed to coordinate task", task_id=task.id, error=str(e))
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.COORDINATION_FAILURE,
                    component="safety_coordinator",
                    message=f"Failed to coordinate task {task.id}: {str(e)}",
                    metadata={"task_id": task.id, "error": str(e)},
                )
            )
            raise

    async def report_task_completion(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        execution_time: float,
        error: Optional[str] = None
    ) -> None:
        """Report task completion for safety tracking.

        Args:
            task_id: Task identifier
            agent_id: Agent that executed the task
            success: Whether task was successful
            execution_time: Task execution time in seconds
            error: Error message if task failed
        """
        logger.info(
            f"Task completion reported",
            task_id=task_id,
            agent_id=agent_id,
            success=success,
            execution_time=execution_time,
        )

        # Update agent tracking
        if agent_id in self._active_agents:
            agent_info = self._active_agents[agent_id]
            agent_info["current_tasks"] = max(0, agent_info["current_tasks"] - 1)
            agent_info["last_activity"] = datetime.now()

            if not success:
                agent_info["total_errors"] += 1

        # Update performance metrics
        await self._update_agent_performance(agent_id, success, execution_time)

        # Check for safety issues
        await self._check_agent_safety_after_completion(agent_id, success, execution_time, error)

    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        active_agents = len(self._active_agents)
        total_current_tasks = sum(agent["current_tasks"] for agent in self._active_agents.values())

        # Get queue statistics
        queue_stats = await self.task_queue.get_queue_statistics()

        status = {
            "coordination_active": self._coordination_active,
            "active_agents": active_agents,
            "max_concurrent_agents": self.max_concurrent_agents,
            "current_tasks": total_current_tasks,
            "queue_statistics": queue_stats,
            "agent_status": {},
            "safety_alerts": [],
        }

        # Agent status
        for agent_id, agent_info in self._active_agents.items():
            performance = self._agent_performance.get(agent_id, {})
            circuit_breaker_state = self._circuit_breakers.get(agent_id, CircuitBreakerState.CLOSED)

            status["agent_status"][agent_id] = {
                "status": agent_info["status"],
                "current_tasks": agent_info["current_tasks"],
                "total_tasks": agent_info["total_tasks"],
                "total_errors": agent_info["total_errors"],
                "error_rate": performance.get("error_rate", 0.0),
                "average_response_time": performance.get("average_response_time", 0.0),
                "circuit_breaker_state": circuit_breaker_state.value,
                "last_activity": agent_info["last_activity"].isoformat(),
            }

            # Check for safety alerts
            if performance.get("error_rate", 0.0) > self._error_rate_threshold:
                status["safety_alerts"].append({
                    "type": "high_error_rate",
                    "agent_id": agent_id,
                    "error_rate": performance["error_rate"],
                })

            if performance.get("average_response_time", 0.0) > self._response_time_threshold:
                status["safety_alerts"].append({
                    "type": "slow_response",
                    "agent_id": agent_id,
                    "response_time": performance["average_response_time"],
                })

        return status

    # Private methods

    async def _safety_monitoring_loop(self) -> None:
        """Background safety monitoring loop."""
        logger.info("Starting safety monitoring loop")

        try:
            while self._coordination_active:
                try:
                    # Perform safety checks
                    await self._perform_safety_checks()

                    # Update performance metrics
                    await self._update_performance_metrics()

                    # Check for inactive agents
                    await self._check_agent_health()

                    # Wait before next iteration
                    await asyncio.sleep(30)  # Check every 30 seconds

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in safety monitoring loop", error=str(e))
                    await asyncio.sleep(5)  # Short delay before retry

        except asyncio.CancelledError:
            logger.info("Safety monitoring loop cancelled")

    async def _check_global_safety_constraints(self) -> None:
        """Check global safety constraints."""
        active_tasks = sum(agent["current_tasks"] for agent in self._active_agents.values())

        if active_tasks >= self.max_concurrent_agents * 3:  # Arbitrary threshold
            raise RuntimeError("Too many concurrent tasks")

        # Check system resource usage (placeholder)
        # In real implementation, would check memory, CPU, etc.

    async def _check_agent_safety_constraints(self, agent_id: str, task: TaskDefinition) -> None:
        """Check agent-specific safety constraints.

        Args:
            agent_id: Agent identifier
            task: Task to be executed

        Raises:
            RuntimeError: If safety constraints are violated
        """
        if agent_id not in self._active_agents:
            raise RuntimeError(f"Agent {agent_id} not registered")

        agent_info = self._active_agents[agent_id]

        # Check circuit breaker
        if self._circuit_breakers.get(agent_id) == CircuitBreakerState.OPEN:
            raise RuntimeError(f"Circuit breaker is open for agent {agent_id}")

        # Check agent error rate
        performance = self._agent_performance.get(agent_id, {})
        if performance.get("error_rate", 0.0) > self._error_rate_threshold:
            logger.warning(
                f"High error rate for agent {agent_id}",
                error_rate=performance["error_rate"],
            )

        # Check task load
        max_tasks = agent_info.get("safety_config", {}).get("max_concurrent_tasks", 3)
        if agent_info["current_tasks"] >= max_tasks:
            raise RuntimeError(f"Agent {agent_id} at maximum task capacity")

    async def _select_agent_for_task(
        self,
        task: TaskDefinition,
        preferred_agent: Optional[str] = None
    ) -> Optional[str]:
        """Select the best agent for a task.

        Args:
            task: Task to execute
            preferred_agent: Preferred agent if specified

        Returns:
            str or None: Selected agent ID
        """
        # If preferred agent specified, check if available
        if preferred_agent and preferred_agent in self._active_agents:
            agent_info = self._active_agents[preferred_agent]
            if (agent_info["status"] == "active" and
                self._circuit_breakers.get(preferred_agent) == CircuitBreakerState.CLOSED):
                return preferred_agent

        # Find suitable agents based on capabilities
        suitable_agents = []

        for agent_id, agent_info in self._active_agents.items():
            if (agent_info["status"] == "active" and
                self._circuit_breakers.get(agent_id) == CircuitBreakerState.CLOSED):

                # Check if agent has required capabilities (simplified)
                # In real implementation, would match task requirements to capabilities
                suitable_agents.append((agent_id, agent_info))

        if not suitable_agents:
            return None

        # Simple load balancing: pick agent with fewest current tasks
        best_agent = min(suitable_agents, key=lambda x: x[1]["current_tasks"])
        return best_agent[0]

    async def _update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        execution_time: float
    ) -> None:
        """Update agent performance metrics.

        Args:
            agent_id: Agent identifier
            success: Whether task was successful
            execution_time: Task execution time in seconds
        """
        if agent_id not in self._agent_performance:
            return

        performance = self._agent_performance[agent_id]

        # Update response time (exponential moving average)
        alpha = 0.2  # Smoothing factor
        performance["average_response_time"] = (
            alpha * execution_time + (1 - alpha) * performance["average_response_time"]
        )

        # Update success/error rates
        agent_info = self._active_agents.get(agent_id, {})
        total_tasks = agent_info.get("total_tasks", 1)
        total_errors = agent_info.get("total_errors", 0)

        if total_tasks > 0:
            performance["error_rate"] = total_errors / total_tasks
            performance["success_rate"] = 1.0 - performance["error_rate"]

    async def _check_agent_safety_after_completion(
        self,
        agent_id: str,
        success: bool,
        execution_time: float,
        error: Optional[str] = None
    ) -> None:
        """Check agent safety after task completion.

        Args:
            agent_id: Agent identifier
            success: Whether task was successful
            execution_time: Task execution time in seconds
            error: Error message if task failed
        """
        performance = self._agent_performance.get(agent_id, {})

        # Check for high error rate
        if performance.get("error_rate", 0.0) > self._error_rate_threshold:
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.HIGH_ERROR_RATE,
                    component="safety_coordinator",
                    message=f"High error rate detected for agent {agent_id}",
                    metadata={
                        "agent_id": agent_id,
                        "error_rate": performance["error_rate"],
                        "threshold": self._error_rate_threshold,
                    },
                )
            )

        # Check for slow response
        if execution_time > self._response_time_threshold:
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.SLOW_RESPONSE,
                    component="safety_coordinator",
                    message=f"Slow response detected for agent {agent_id}",
                    metadata={
                        "agent_id": agent_id,
                        "execution_time": execution_time,
                        "threshold": self._response_time_threshold,
                    },
                )
            )

        # Update circuit breaker state based on performance
        if performance.get("error_rate", 0.0) > 0.8:  # Very high error rate
            self._circuit_breakers[agent_id] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened for agent {agent_id}")

    async def _perform_safety_checks(self) -> None:
        """Perform periodic safety checks."""
        current_time = datetime.now()

        for agent_id, agent_info in self._active_agents.items():
            # Check for stuck agents (no activity for too long)
            time_since_activity = current_time - agent_info["last_activity"]
            if time_since_activity > timedelta(minutes=10):  # 10 minutes threshold
                logger.warning(f"Agent {agent_id} appears inactive", inactive_duration=str(time_since_activity))

                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.AGENT_INACTIVE,
                        component="safety_coordinator",
                        message=f"Agent {agent_id} has been inactive for {time_since_activity}",
                        metadata={"agent_id": agent_id, "inactive_duration": str(time_since_activity)},
                    )
                )

    async def _update_performance_metrics(self) -> None:
        """Update overall performance metrics."""
        # Calculate system-wide metrics
        total_tasks = sum(agent["total_tasks"] for agent in self._active_agents.values())
        total_errors = sum(agent["total_errors"] for agent in self._active_agents.values())

        if total_tasks > 0:
            system_error_rate = total_errors / total_tasks

            if system_error_rate > self._error_rate_threshold:
                logger.warning(f"High system error rate", error_rate=system_error_rate)

    async def _check_agent_health(self) -> None:
        """Check health of all registered agents."""
        current_time = datetime.now()

        for agent_id, agent_info in list(self._active_agents.items()):
            # Check if agent has been responsive
            time_since_activity = current_time - agent_info["last_activity"]

            if time_since_activity > timedelta(minutes=15):  # 15 minutes threshold
                # Mark agent as potentially unhealthy
                agent_info["status"] = "unhealthy"

                logger.warning(f"Agent {agent_id} marked as unhealthy", inactive_duration=str(time_since_activity))

                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.AGENT_HEALTH_DEGRADED,
                        component="safety_coordinator",
                        message=f"Agent {agent_id} health degraded due to inactivity",
                        metadata={"agent_id": agent_id, "inactive_duration": str(time_since_activity)},
                    )
                )