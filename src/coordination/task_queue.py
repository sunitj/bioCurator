"""Persistent task queue system for agent coordination."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    create_engine,
    MetaData,
    Table,
    select,
    insert,
    update,
    delete,
    func,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from ..logging import get_logger
from ..safety.event_bus import SafetyEvent, SafetyEventBus, SafetyEventType

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskDefinition(BaseModel):
    """Task definition for the queue."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    agent_id: Optional[str] = None  # Target agent, None for any available
    priority: TaskPriority = TaskPriority.NORMAL

    # Task data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None  # Future scheduling
    timeout_seconds: int = 300  # 5 minutes default

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    retry_backoff_multiplier: float = 2.0

    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Task IDs this task depends on


class TaskExecution(BaseModel):
    """Task execution record."""

    task_id: str
    agent_id: Optional[str] = None
    status: TaskStatus

    # Execution details
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

    # Retry tracking
    attempt_number: int = 0
    next_retry_at: Optional[datetime] = None

    # Performance metrics
    execution_time_seconds: Optional[float] = None
    queue_wait_seconds: Optional[float] = None


class TaskQueue:
    """Persistent task queue with PostgreSQL backend."""

    def __init__(
        self,
        database_url: str,
        event_bus: Optional[SafetyEventBus] = None,
        max_queue_size: int = 1000,
    ):
        """Initialize task queue.

        Args:
            database_url: PostgreSQL connection URL
            event_bus: Event bus for safety events
            max_queue_size: Maximum number of pending tasks
        """
        self.database_url = database_url
        self.event_bus = event_bus or SafetyEventBus()
        self.max_queue_size = max_queue_size

        # Database setup
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session_maker = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Database schema
        self.metadata = MetaData()
        self._create_tables()

        # Task processing
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("Task queue initialized", max_size=max_queue_size)

    def _create_tables(self) -> None:
        """Create database tables for task storage."""
        self.tasks_table = Table(
            "agent_tasks",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("task_type", String(100), nullable=False),
            Column("agent_id", String(100), nullable=True),
            Column("priority", Integer, default=2),
            Column("status", String(20), default="pending"),

            # Task data (JSON)
            Column("input_data", Text),
            Column("metadata", Text),
            Column("output_data", Text),
            Column("error_message", Text, nullable=True),

            # Timing
            Column("created_at", DateTime, default=datetime.now),
            Column("scheduled_for", DateTime, nullable=True),
            Column("assigned_at", DateTime, nullable=True),
            Column("started_at", DateTime, nullable=True),
            Column("completed_at", DateTime, nullable=True),
            Column("timeout_seconds", Integer, default=300),

            # Retry configuration
            Column("max_retries", Integer, default=3),
            Column("retry_delay_seconds", Integer, default=60),
            Column("retry_backoff_multiplier", Float, default=2.0),
            Column("attempt_number", Integer, default=0),
            Column("next_retry_at", DateTime, nullable=True),

            # Dependencies
            Column("depends_on", Text),  # JSON array of task IDs

            # Performance metrics
            Column("execution_time_seconds", Float, nullable=True),
            Column("queue_wait_seconds", Float, nullable=True),
        )

    async def initialize(self) -> None:
        """Initialize the task queue database."""
        logger.info("Initializing task queue database")

        try:
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)

            # Start task processing
            self._processing_task = asyncio.create_task(self._process_tasks_loop())

            logger.info("Task queue initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize task queue", error=str(e))
            raise

    async def shutdown(self) -> None:
        """Shutdown the task queue."""
        logger.info("Shutting down task queue")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Close database connections
        await self.engine.dispose()

        logger.info("Task queue shutdown complete")

    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task to the queue.

        Args:
            task: Task definition

        Returns:
            str: Task ID

        Raises:
            RuntimeError: If queue is full or task cannot be submitted
        """
        # Check queue size
        current_size = await self.get_queue_size()
        if current_size >= self.max_queue_size:
            raise RuntimeError(f"Queue is full ({current_size}/{self.max_queue_size})")

        # Validate dependencies
        if task.depends_on:
            await self._validate_dependencies(task.depends_on)

        try:
            async with self.async_session_maker() as session:
                # Insert task
                stmt = insert(self.tasks_table).values(
                    id=task.id,
                    task_type=task.task_type,
                    agent_id=task.agent_id,
                    priority=task.priority.value,
                    input_data=json.dumps(task.input_data),
                    metadata=json.dumps(task.metadata),
                    created_at=task.created_at,
                    scheduled_for=task.scheduled_for,
                    timeout_seconds=task.timeout_seconds,
                    max_retries=task.max_retries,
                    retry_delay_seconds=task.retry_delay_seconds,
                    retry_backoff_multiplier=task.retry_backoff_multiplier,
                    depends_on=json.dumps(task.depends_on),
                )

                await session.execute(stmt)
                await session.commit()

            logger.info(
                f"Task submitted to queue",
                task_id=task.id,
                task_type=task.task_type,
                agent_id=task.agent_id,
                priority=task.priority.value,
            )

            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.TASK_QUEUED,
                    component="task_queue",
                    message=f"Task {task.id} queued for execution",
                    metadata={"task_id": task.id, "task_type": task.task_type},
                )
            )

            return task.id

        except Exception as e:
            logger.error(f"Failed to submit task {task.id}", error=str(e))
            raise RuntimeError(f"Failed to submit task: {str(e)}")

    async def get_next_task(self, agent_id: str, task_types: Optional[List[str]] = None) -> Optional[TaskDefinition]:
        """Get the next available task for an agent.

        Args:
            agent_id: Agent requesting a task
            task_types: Optional list of task types the agent can handle

        Returns:
            TaskDefinition or None if no tasks available
        """
        try:
            async with self.async_session_maker() as session:
                # Build query for available tasks
                query = select(self.tasks_table).where(
                    self.tasks_table.c.status == TaskStatus.PENDING.value,
                    # Task is not scheduled for the future
                    (self.tasks_table.c.scheduled_for.is_(None)) |
                    (self.tasks_table.c.scheduled_for <= datetime.now()),
                )

                # Filter by agent if specified
                if task_types:
                    query = query.where(self.tasks_table.c.task_type.in_(task_types))

                # Filter by target agent
                query = query.where(
                    (self.tasks_table.c.agent_id.is_(None)) |
                    (self.tasks_table.c.agent_id == agent_id)
                )

                # Order by priority (highest first) and creation time
                query = query.order_by(
                    self.tasks_table.c.priority.desc(),
                    self.tasks_table.c.created_at.asc()
                )

                # Get first available task
                result = await session.execute(query.limit(1))
                row = result.fetchone()

                if not row:
                    return None

                # Check dependencies
                depends_on = json.loads(row.depends_on) if row.depends_on else []
                if depends_on and not await self._dependencies_completed(depends_on):
                    return None

                # Mark task as assigned
                await session.execute(
                    update(self.tasks_table)
                    .where(self.tasks_table.c.id == row.id)
                    .values(
                        status=TaskStatus.ASSIGNED.value,
                        agent_id=agent_id,
                        assigned_at=datetime.now(),
                    )
                )
                await session.commit()

                # Create task definition
                task = TaskDefinition(
                    id=row.id,
                    task_type=row.task_type,
                    agent_id=agent_id,
                    priority=TaskPriority(row.priority),
                    input_data=json.loads(row.input_data),
                    metadata=json.loads(row.metadata),
                    created_at=row.created_at,
                    scheduled_for=row.scheduled_for,
                    timeout_seconds=row.timeout_seconds,
                    max_retries=row.max_retries,
                    retry_delay_seconds=row.retry_delay_seconds,
                    retry_backoff_multiplier=row.retry_backoff_multiplier,
                    depends_on=depends_on,
                )

                logger.info(
                    f"Task assigned to agent",
                    task_id=task.id,
                    agent_id=agent_id,
                    task_type=task.task_type,
                )

                return task

        except Exception as e:
            logger.error(f"Failed to get next task for agent {agent_id}", error=str(e))
            return None

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update task execution status.

        Args:
            task_id: Task identifier
            status: New task status
            output_data: Task output data (for completed tasks)
            error_message: Error message (for failed tasks)
        """
        try:
            async with self.async_session_maker() as session:
                update_values = {"status": status.value}

                # Set timestamps based on status
                if status == TaskStatus.RUNNING:
                    update_values["started_at"] = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    update_values["completed_at"] = datetime.now()

                    # Calculate execution time if we have start time
                    result = await session.execute(
                        select(self.tasks_table.c.started_at, self.tasks_table.c.assigned_at)
                        .where(self.tasks_table.c.id == task_id)
                    )
                    row = result.fetchone()
                    if row and row.started_at:
                        execution_time = (datetime.now() - row.started_at).total_seconds()
                        update_values["execution_time_seconds"] = execution_time

                        if row.assigned_at:
                            queue_wait = (row.started_at - row.assigned_at).total_seconds()
                            update_values["queue_wait_seconds"] = queue_wait

                # Set output data and error message
                if output_data is not None:
                    update_values["output_data"] = json.dumps(output_data)

                if error_message:
                    update_values["error_message"] = error_message

                # Handle retry logic for failed tasks
                if status == TaskStatus.FAILED:
                    await self._handle_task_retry(session, task_id, update_values)

                # Update task
                await session.execute(
                    update(self.tasks_table)
                    .where(self.tasks_table.c.id == task_id)
                    .values(**update_values)
                )
                await session.commit()

            logger.info(f"Task status updated", task_id=task_id, status=status.value)

            # Emit safety event
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.TASK_STATUS_CHANGED,
                    component="task_queue",
                    message=f"Task {task_id} status changed to {status.value}",
                    metadata={"task_id": task_id, "status": status.value},
                )
            )

        except Exception as e:
            logger.error(f"Failed to update task status", task_id=task_id, error=str(e))
            raise

    async def get_task_status(self, task_id: str) -> Optional[TaskExecution]:
        """Get current task execution status.

        Args:
            task_id: Task identifier

        Returns:
            TaskExecution or None if not found
        """
        try:
            async with self.async_session_maker() as session:
                result = await session.execute(
                    select(self.tasks_table).where(self.tasks_table.c.id == task_id)
                )
                row = result.fetchone()

                if not row:
                    return None

                return TaskExecution(
                    task_id=row.id,
                    agent_id=row.agent_id,
                    status=TaskStatus(row.status),
                    assigned_at=row.assigned_at,
                    started_at=row.started_at,
                    completed_at=row.completed_at,
                    output_data=json.loads(row.output_data) if row.output_data else {},
                    error_message=row.error_message,
                    attempt_number=row.attempt_number,
                    next_retry_at=row.next_retry_at,
                    execution_time_seconds=row.execution_time_seconds,
                    queue_wait_seconds=row.queue_wait_seconds,
                )

        except Exception as e:
            logger.error(f"Failed to get task status", task_id=task_id, error=str(e))
            return None

    async def get_queue_size(self) -> int:
        """Get current queue size (pending tasks)."""
        try:
            async with self.async_session_maker() as session:
                result = await session.execute(
                    select([func.count()]).select_from(self.tasks_table)
                    .where(self.tasks_table.c.status == TaskStatus.PENDING.value)
                )
                return result.scalar()

        except Exception as e:
            logger.error("Failed to get queue size", error=str(e))
            return 0

    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            async with self.async_session_maker() as session:
                # Count tasks by status
                result = await session.execute(
                    select([
                        self.tasks_table.c.status,
                        func.count().label("count")
                    ])
                    .group_by(self.tasks_table.c.status)
                )

                status_counts = {row.status: row.count for row in result}

                # Get average execution times
                result = await session.execute(
                    select([
                        func.avg(self.tasks_table.c.execution_time_seconds).label("avg_execution_time"),
                        func.avg(self.tasks_table.c.queue_wait_seconds).label("avg_queue_wait"),
                    ])
                    .where(self.tasks_table.c.status == TaskStatus.COMPLETED.value)
                )

                timing_stats = result.fetchone()

                return {
                    "status_counts": status_counts,
                    "average_execution_time_seconds": timing_stats.avg_execution_time or 0.0,
                    "average_queue_wait_seconds": timing_stats.avg_queue_wait or 0.0,
                    "total_tasks": sum(status_counts.values()),
                }

        except Exception as e:
            logger.error("Failed to get queue statistics", error=str(e))
            return {}

    # Private methods

    async def _validate_dependencies(self, depends_on: List[str]) -> None:
        """Validate task dependencies exist."""
        if not depends_on:
            return

        async with self.async_session_maker() as session:
            result = await session.execute(
                select([func.count()])
                .select_from(self.tasks_table)
                .where(self.tasks_table.c.id.in_(depends_on))
            )

            found_count = result.scalar()
            if found_count != len(depends_on):
                missing = set(depends_on) - set(found_count)
                raise ValueError(f"Missing dependency tasks: {missing}")

    async def _dependencies_completed(self, depends_on: List[str]) -> bool:
        """Check if all dependencies are completed."""
        if not depends_on:
            return True

        async with self.async_session_maker() as session:
            result = await session.execute(
                select([func.count()])
                .select_from(self.tasks_table)
                .where(
                    self.tasks_table.c.id.in_(depends_on),
                    self.tasks_table.c.status == TaskStatus.COMPLETED.value
                )
            )

            completed_count = result.scalar()
            return completed_count == len(depends_on)

    async def _handle_task_retry(self, session: AsyncSession, task_id: str, update_values: Dict) -> None:
        """Handle retry logic for failed tasks."""
        # Get current task info
        result = await session.execute(
            select([
                self.tasks_table.c.attempt_number,
                self.tasks_table.c.max_retries,
                self.tasks_table.c.retry_delay_seconds,
                self.tasks_table.c.retry_backoff_multiplier,
            ])
            .where(self.tasks_table.c.id == task_id)
        )

        row = result.fetchone()
        if not row:
            return

        attempt_number = row.attempt_number + 1

        if attempt_number <= row.max_retries:
            # Schedule retry
            delay = row.retry_delay_seconds * (row.retry_backoff_multiplier ** (attempt_number - 1))
            next_retry_at = datetime.now() + timedelta(seconds=delay)

            update_values.update({
                "status": TaskStatus.RETRYING.value,
                "attempt_number": attempt_number,
                "next_retry_at": next_retry_at,
            })

            logger.info(
                f"Task scheduled for retry",
                task_id=task_id,
                attempt=attempt_number,
                delay_seconds=delay,
            )
        else:
            # No more retries
            update_values["status"] = TaskStatus.FAILED.value

            logger.warning(
                f"Task failed after max retries",
                task_id=task_id,
                attempts=attempt_number,
            )

    async def _process_tasks_loop(self) -> None:
        """Background task processing loop for retries and cleanup."""
        logger.info("Starting task processing loop")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Process retries
                    await self._process_retries()

                    # Clean up old completed tasks (optional)
                    await self._cleanup_old_tasks()

                    # Wait before next iteration
                    await asyncio.sleep(30)  # Check every 30 seconds

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in task processing loop", error=str(e))
                    await asyncio.sleep(5)  # Short delay before retry

        except asyncio.CancelledError:
            logger.info("Task processing loop cancelled")

    async def _process_retries(self) -> None:
        """Process tasks scheduled for retry."""
        try:
            async with self.async_session_maker() as session:
                # Find tasks ready for retry
                result = await session.execute(
                    select(self.tasks_table)
                    .where(
                        self.tasks_table.c.status == TaskStatus.RETRYING.value,
                        self.tasks_table.c.next_retry_at <= datetime.now(),
                    )
                )

                for row in result:
                    # Reset task to pending
                    await session.execute(
                        update(self.tasks_table)
                        .where(self.tasks_table.c.id == row.id)
                        .values(
                            status=TaskStatus.PENDING.value,
                            agent_id=None,
                            assigned_at=None,
                            started_at=None,
                            completed_at=None,
                            next_retry_at=None,
                        )
                    )

                    logger.info(f"Task reset for retry", task_id=row.id, attempt=row.attempt_number)

                await session.commit()

        except Exception as e:
            logger.error("Failed to process retries", error=str(e))

    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks (optional)."""
        # This could be configurable - for now, keep all tasks for debugging
        pass