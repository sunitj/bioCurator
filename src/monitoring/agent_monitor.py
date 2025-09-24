"""Agent behavior monitoring and performance tracking."""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Deque

from pydantic import BaseModel

from ..logging import get_logger
from ..safety.event_bus import SafetyEvent, SafetyEventBus, SafetyEventType

logger = get_logger(__name__)


class AgentMetrics(BaseModel):
    """Metrics for an individual agent."""

    agent_id: str

    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_tasks: int = 0

    # Performance metrics
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0

    # Resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Activity metrics
    tasks_per_minute: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0

    # Quality metrics
    average_quality_score: float = 0.0
    consistency_score: float = 0.0

    # Safety metrics
    anomaly_count: int = 0
    circuit_breaker_trips: int = 0
    rate_limit_blocks: int = 0


class TaskRecord(BaseModel):
    """Record of a completed task."""

    task_id: str
    agent_id: str
    task_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


class AgentMonitor:
    """Monitors agent behavior and performance."""

    def __init__(
        self,
        event_bus: Optional[SafetyEventBus] = None,
        metrics_retention_hours: int = 24,
        monitoring_interval_seconds: int = 30,
    ):
        """Initialize agent monitor.

        Args:
            event_bus: Event bus for safety events
            metrics_retention_hours: How long to retain detailed metrics
            monitoring_interval_seconds: Interval for periodic monitoring
        """
        self.event_bus = event_bus or SafetyEventBus()
        self.metrics_retention_hours = metrics_retention_hours
        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Metrics storage
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._task_history: Dict[str, Deque[TaskRecord]] = defaultdict(lambda: deque(maxlen=1000))
        self._performance_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=100))

        # Agent registration
        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        self._agent_start_times: Dict[str, datetime] = {}

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None

        logger.info("Agent monitor initialized", retention_hours=metrics_retention_hours)

    async def start_monitoring(self) -> None:
        """Start agent monitoring."""
        if self._monitoring_active:
            logger.warning("Agent monitoring already active")
            return

        logger.info("Starting agent monitoring")

        try:
            # Start monitoring loop
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._monitoring_active = True

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.MONITORING_STARTED,
                    component="agent_monitor",
                    message="Agent monitoring started",
                    metadata={"monitoring_interval": self.monitoring_interval_seconds},
                )
            )

            logger.info("Agent monitoring started successfully")

        except Exception as e:
            logger.error("Failed to start agent monitoring", error=str(e))
            raise

    async def stop_monitoring(self) -> None:
        """Stop agent monitoring."""
        if not self._monitoring_active:
            logger.warning("Agent monitoring not active")
            return

        logger.info("Stopping agent monitoring")

        try:
            # Stop monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            self._monitoring_active = False

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.MONITORING_STOPPED,
                    component="agent_monitor",
                    message="Agent monitoring stopped",
                    metadata={},
                )
            )

            logger.info("Agent monitoring stopped")

        except Exception as e:
            logger.error("Error stopping agent monitoring", error=str(e))

    async def register_agent(
        self,
        agent_id: str,
        agent_info: Dict[str, Any]
    ) -> None:
        """Register an agent for monitoring.

        Args:
            agent_id: Agent identifier
            agent_info: Agent information and configuration
        """
        logger.info(f"Registering agent for monitoring", agent_id=agent_id)

        self._registered_agents[agent_id] = agent_info.copy()
        self._agent_start_times[agent_id] = datetime.now()

        # Initialize metrics
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)

        self.event_bus.emit(
            SafetyEvent(
                event_type=SafetyEventType.AGENT_MONITORING_STARTED,
                component="agent_monitor",
                message=f"Agent {agent_id} registered for monitoring",
                metadata={"agent_id": agent_id},
            )
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring.

        Args:
            agent_id: Agent identifier
        """
        logger.info(f"Unregistering agent from monitoring", agent_id=agent_id)

        # Keep historical data but stop active monitoring
        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]

        if agent_id in self._agent_start_times:
            del self._agent_start_times[agent_id]

        self.event_bus.emit(
            SafetyEvent(
                event_type=SafetyEventType.AGENT_MONITORING_STOPPED,
                component="agent_monitor",
                message=f"Agent {agent_id} unregistered from monitoring",
                metadata={"agent_id": agent_id},
            )
        )

    async def record_task_start(
        self,
        task_id: str,
        agent_id: str,
        task_type: str
    ) -> None:
        """Record when an agent starts a task.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            task_type: Type of task
        """
        logger.debug(f"Recording task start", task_id=task_id, agent_id=agent_id, task_type=task_type)

        if agent_id in self._agent_metrics:
            metrics = self._agent_metrics[agent_id]
            metrics.current_tasks += 1
            metrics.total_tasks += 1
            metrics.last_activity = datetime.now()

    async def record_task_completion(
        self,
        task_id: str,
        agent_id: str,
        task_type: str,
        duration_seconds: float,
        success: bool,
        error_message: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> None:
        """Record when an agent completes a task.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            task_type: Type of task
            duration_seconds: Task execution duration
            success: Whether task was successful
            error_message: Error message if task failed
            quality_score: Quality score if available
        """
        logger.debug(
            f"Recording task completion",
            task_id=task_id,
            agent_id=agent_id,
            success=success,
            duration=duration_seconds,
        )

        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration_seconds)

        # Create task record
        task_record = TaskRecord(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
            quality_score=quality_score,
        )

        # Store task record
        self._task_history[agent_id].append(task_record)

        # Update metrics
        if agent_id in self._agent_metrics:
            await self._update_agent_metrics(agent_id, task_record)

        # Check for anomalies
        await self._check_for_anomalies(agent_id, task_record)

    async def record_agent_status(
        self,
        agent_id: str,
        status_data: Dict[str, Any]
    ) -> None:
        """Record agent status information.

        Args:
            agent_id: Agent identifier
            status_data: Status information
        """
        logger.debug(f"Recording agent status", agent_id=agent_id)

        if agent_id in self._agent_metrics:
            metrics = self._agent_metrics[agent_id]

            # Update resource metrics if provided
            if "memory_usage_mb" in status_data:
                metrics.memory_usage_mb = status_data["memory_usage_mb"]

            if "cpu_usage_percent" in status_data:
                metrics.cpu_usage_percent = status_data["cpu_usage_percent"]

            if "current_tasks" in status_data:
                metrics.current_tasks = status_data["current_tasks"]

            # Update uptime
            if agent_id in self._agent_start_times:
                uptime = datetime.now() - self._agent_start_times[agent_id]
                metrics.uptime_seconds = uptime.total_seconds()

            # Store status history
            status_record = {
                "timestamp": datetime.now(),
                "status": status_data,
            }
            self._performance_history[agent_id].append(status_record)

    async def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get current metrics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentMetrics or None if agent not found
        """
        return self._agent_metrics.get(agent_id)

    async def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all monitored agents."""
        return self._agent_metrics.copy()

    async def get_agent_task_history(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[TaskRecord]:
        """Get task history for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of records to return

        Returns:
            List of task records
        """
        history = list(self._task_history.get(agent_id, []))

        if limit:
            history = history[-limit:]

        return history

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        total_agents = len(self._registered_agents)
        active_agents = sum(1 for metrics in self._agent_metrics.values() if metrics.current_tasks > 0)

        # Aggregate metrics
        total_tasks = sum(metrics.total_tasks for metrics in self._agent_metrics.values())
        total_completed = sum(metrics.completed_tasks for metrics in self._agent_metrics.values())
        total_failed = sum(metrics.failed_tasks for metrics in self._agent_metrics.values())
        total_current = sum(metrics.current_tasks for metrics in self._agent_metrics.values())

        # Calculate averages
        avg_response_time = 0.0
        avg_success_rate = 0.0
        if self._agent_metrics:
            avg_response_time = sum(m.average_response_time for m in self._agent_metrics.values()) / len(self._agent_metrics)
            avg_success_rate = sum(m.success_rate for m in self._agent_metrics.values()) / len(self._agent_metrics)

        return {
            "system_overview": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "monitoring_active": self._monitoring_active,
            },
            "task_metrics": {
                "total_tasks": total_tasks,
                "completed_tasks": total_completed,
                "failed_tasks": total_failed,
                "current_tasks": total_current,
                "success_rate": total_completed / max(total_tasks, 1),
            },
            "performance_metrics": {
                "average_response_time": avg_response_time,
                "average_success_rate": avg_success_rate,
            },
            "agent_metrics": {
                agent_id: metrics.model_dump()
                for agent_id, metrics in self._agent_metrics.items()
            },
        }

    async def get_monitoring_health(self) -> Dict[str, Any]:
        """Get health status of the monitoring system."""
        return {
            "monitoring_active": self._monitoring_active,
            "registered_agents": len(self._registered_agents),
            "monitoring_interval": self.monitoring_interval_seconds,
            "metrics_retention_hours": self.metrics_retention_hours,
            "task_history_size": sum(len(history) for history in self._task_history.values()),
            "performance_history_size": sum(len(history) for history in self._performance_history.values()),
        }

    # Private methods

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")

        try:
            while self._monitoring_active:
                try:
                    # Update derived metrics
                    await self._update_derived_metrics()

                    # Cleanup old data
                    await self._cleanup_old_data()

                    # Check for issues
                    await self._check_system_health()

                    # Wait for next iteration
                    await asyncio.sleep(self.monitoring_interval_seconds)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in monitoring loop", error=str(e))
                    await asyncio.sleep(5)  # Short delay before retry

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")

    async def _update_agent_metrics(self, agent_id: str, task_record: TaskRecord) -> None:
        """Update agent metrics based on task completion."""
        metrics = self._agent_metrics[agent_id]

        # Update task counts
        metrics.current_tasks = max(0, metrics.current_tasks - 1)
        if task_record.success:
            metrics.completed_tasks += 1
        else:
            metrics.failed_tasks += 1

        # Update response time metrics
        duration = task_record.duration_seconds
        if metrics.total_tasks == 1:
            # First task
            metrics.average_response_time = duration
            metrics.min_response_time = duration
            metrics.max_response_time = duration
        else:
            # Update exponential moving average
            alpha = 0.2
            metrics.average_response_time = alpha * duration + (1 - alpha) * metrics.average_response_time
            metrics.min_response_time = min(metrics.min_response_time, duration)
            metrics.max_response_time = max(metrics.max_response_time, duration)

        # Update success/error rates
        completed_tasks = metrics.completed_tasks + metrics.failed_tasks
        if completed_tasks > 0:
            metrics.success_rate = metrics.completed_tasks / completed_tasks
            metrics.error_rate = metrics.failed_tasks / completed_tasks

        # Update quality score if provided
        if task_record.quality_score is not None:
            if metrics.average_quality_score == 0.0:
                metrics.average_quality_score = task_record.quality_score
            else:
                alpha = 0.2
                metrics.average_quality_score = (
                    alpha * task_record.quality_score + (1 - alpha) * metrics.average_quality_score
                )

        metrics.last_activity = task_record.end_time

    async def _update_derived_metrics(self) -> None:
        """Update derived metrics like tasks per minute."""
        current_time = datetime.now()

        for agent_id, metrics in self._agent_metrics.items():
            # Calculate tasks per minute
            recent_tasks = [
                record for record in self._task_history[agent_id]
                if current_time - record.end_time <= timedelta(minutes=1)
            ]
            metrics.tasks_per_minute = len(recent_tasks)

            # Update consistency score (placeholder)
            # In real implementation, would analyze task completion patterns
            metrics.consistency_score = 0.8 if metrics.success_rate > 0.8 else 0.5

    async def _cleanup_old_data(self) -> None:
        """Clean up old data to manage memory usage."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)

        for agent_id in list(self._task_history.keys()):
            # Remove old task records
            history = self._task_history[agent_id]
            while history and history[0].end_time < cutoff_time:
                history.popleft()

            # Remove old performance records
            perf_history = self._performance_history[agent_id]
            while perf_history and perf_history[0]["timestamp"] < cutoff_time:
                perf_history.popleft()

    async def _check_system_health(self) -> None:
        """Check overall system health."""
        total_error_rate = 0.0
        unhealthy_agents = []

        for agent_id, metrics in self._agent_metrics.items():
            # Check agent health
            if metrics.error_rate > 0.5:  # High error rate
                unhealthy_agents.append(agent_id)

            if metrics.last_activity:
                time_since_activity = datetime.now() - metrics.last_activity
                if time_since_activity > timedelta(minutes=10):  # Long inactivity
                    unhealthy_agents.append(agent_id)

            total_error_rate += metrics.error_rate

        # Calculate system-wide error rate
        if self._agent_metrics:
            system_error_rate = total_error_rate / len(self._agent_metrics)

            if system_error_rate > 0.3:  # High system error rate
                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.HIGH_ERROR_RATE,
                        component="agent_monitor",
                        message=f"High system error rate detected: {system_error_rate:.2f}",
                        metadata={"system_error_rate": system_error_rate},
                    )
                )

        # Report unhealthy agents
        for agent_id in unhealthy_agents:
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.AGENT_HEALTH_DEGRADED,
                    component="agent_monitor",
                    message=f"Agent {agent_id} health degraded",
                    metadata={"agent_id": agent_id},
                )
            )

    async def _check_for_anomalies(self, agent_id: str, task_record: TaskRecord) -> None:
        """Check for anomalous behavior in agent performance."""
        metrics = self._agent_metrics[agent_id]

        # Check for unusually long response time
        if task_record.duration_seconds > metrics.average_response_time * 3:  # 3x average
            metrics.anomaly_count += 1

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.ANOMALY_DETECTED,
                    component="agent_monitor",
                    message=f"Unusually long response time for agent {agent_id}",
                    metadata={
                        "agent_id": agent_id,
                        "task_id": task_record.task_id,
                        "duration": task_record.duration_seconds,
                        "average": metrics.average_response_time,
                    },
                )
            )

        # Check for error patterns
        if not task_record.success:
            recent_failures = sum(
                1 for record in list(self._task_history[agent_id])[-5:]  # Last 5 tasks
                if not record.success
            )

            if recent_failures >= 3:  # 3 or more failures in last 5 tasks
                metrics.anomaly_count += 1

                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.ANOMALY_DETECTED,
                        component="agent_monitor",
                        message=f"High failure rate pattern for agent {agent_id}",
                        metadata={
                            "agent_id": agent_id,
                            "recent_failures": recent_failures,
                            "pattern": "consecutive_failures",
                        },
                    )
                )