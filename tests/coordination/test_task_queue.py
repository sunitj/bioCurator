"""Tests for task queue functionality."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import tempfile

from src.coordination.task_queue import TaskQueue, TaskDefinition, TaskStatus, TaskPriority
from src.safety.event_bus import SafetyEventBus


@pytest.fixture
def event_bus():
    """Event bus for testing."""
    return SafetyEventBus()


@pytest.fixture
def mock_database_url():
    """Mock database URL for testing."""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture
async def task_queue(mock_database_url, event_bus):
    """Task queue instance for testing."""
    with patch('src.coordination.task_queue.create_async_engine') as mock_engine:
        with patch('src.coordination.task_queue.sessionmaker') as mock_session:
            # Mock the database components
            mock_engine.return_value = AsyncMock()
            mock_session.return_value = AsyncMock()

            queue = TaskQueue(
                database_url=mock_database_url,
                event_bus=event_bus,
                max_queue_size=100,
            )

            # Mock the database operations
            queue.async_session_maker = AsyncMock()
            queue.engine = AsyncMock()

            yield queue

            # Cleanup
            try:
                await queue.shutdown()
            except:
                pass


@pytest.fixture
def sample_task():
    """Sample task definition."""
    return TaskDefinition(
        id="test_task_001",
        task_type="analyze_literature",
        agent_id="research_director",
        priority=TaskPriority.HIGH,
        input_data={
            "query": "protein folding",
            "max_papers": 10,
        },
        metadata={
            "created_by": "user_123",
            "workflow_id": "wf_001",
        },
        timeout_seconds=300,
        max_retries=3,
    )


class TestTaskQueue:
    """Test cases for TaskQueue."""

    def test_task_queue_initialization(self, task_queue, mock_database_url):
        """Test task queue initialization."""
        assert task_queue.database_url == mock_database_url
        assert task_queue.max_queue_size == 100
        assert task_queue.event_bus is not None
        assert task_queue._processing_task is None
        assert not task_queue._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_initialize_success(self, task_queue):
        """Test successful task queue initialization."""
        # Mock database operations
        with patch.object(task_queue.engine, 'begin') as mock_begin:
            mock_conn = AsyncMock()
            mock_begin.return_value.__aenter__.return_value = mock_conn

            await task_queue.initialize()

            # Should create tables and start processing
            assert task_queue._processing_task is not None
            assert not task_queue._processing_task.done()

    @pytest.mark.asyncio
    async def test_initialize_database_error(self, task_queue):
        """Test initialization with database error."""
        with patch.object(task_queue.engine, 'begin', side_effect=Exception("DB Error")):
            with pytest.raises(Exception, match="DB Error"):
                await task_queue.initialize()

    @pytest.mark.asyncio
    async def test_submit_task_success(self, task_queue, sample_task):
        """Test successful task submission."""
        # Mock database operations
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        # Mock queue size check
        with patch.object(task_queue, 'get_queue_size', return_value=0):
            with patch.object(task_queue, '_validate_dependencies'):
                task_id = await task_queue.submit_task(sample_task)

                assert task_id == sample_task.id
                mock_session.execute.assert_called()
                mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_submit_task_queue_full(self, task_queue, sample_task):
        """Test task submission when queue is full."""
        # Mock queue size to be at maximum
        with patch.object(task_queue, 'get_queue_size', return_value=100):
            with pytest.raises(RuntimeError, match="Queue is full"):
                await task_queue.submit_task(sample_task)

    @pytest.mark.asyncio
    async def test_submit_task_with_dependencies(self, task_queue, sample_task):
        """Test task submission with dependencies."""
        sample_task.depends_on = ["dep_task_1", "dep_task_2"]

        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        with patch.object(task_queue, 'get_queue_size', return_value=0):
            with patch.object(task_queue, '_validate_dependencies') as mock_validate:
                await task_queue.submit_task(sample_task)
                mock_validate.assert_called_once_with(["dep_task_1", "dep_task_2"])

    @pytest.mark.asyncio
    async def test_get_next_task_success(self, task_queue):
        """Test getting next available task."""
        # Mock database query result
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_row = Mock()
        mock_row.id = "task_001"
        mock_row.task_type = "analyze_literature"
        mock_row.agent_id = None
        mock_row.priority = 3
        mock_row.input_data = '{"query": "test"}'
        mock_row.metadata = '{}'
        mock_row.created_at = datetime.now()
        mock_row.scheduled_for = None
        mock_row.timeout_seconds = 300
        mock_row.max_retries = 3
        mock_row.retry_delay_seconds = 60
        mock_row.retry_backoff_multiplier = 2.0
        mock_row.depends_on = '[]'

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        with patch.object(task_queue, '_dependencies_completed', return_value=True):
            task = await task_queue.get_next_task("test_agent", ["analyze_literature"])

            assert task is not None
            assert task.id == "task_001"
            assert task.task_type == "analyze_literature"
            assert task.agent_id == "test_agent"  # Should be assigned

    @pytest.mark.asyncio
    async def test_get_next_task_no_tasks(self, task_queue):
        """Test getting next task when none available."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        task = await task_queue.get_next_task("test_agent")
        assert task is None

    @pytest.mark.asyncio
    async def test_get_next_task_dependencies_not_complete(self, task_queue):
        """Test getting next task with incomplete dependencies."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_row = Mock()
        mock_row.depends_on = '["dep1", "dep2"]'
        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        with patch.object(task_queue, '_dependencies_completed', return_value=False):
            task = await task_queue.get_next_task("test_agent")
            assert task is None

    @pytest.mark.asyncio
    async def test_update_task_status_completed(self, task_queue):
        """Test updating task status to completed."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        # Mock query for execution time calculation
        mock_row = Mock()
        mock_row.started_at = datetime.now() - timedelta(seconds=10)
        mock_row.assigned_at = datetime.now() - timedelta(seconds=15)
        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        output_data = {"result": "success"}
        await task_queue.update_task_status(
            "task_001",
            TaskStatus.COMPLETED,
            output_data=output_data
        )

        # Should update status and set completion time
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_task_status_failed_with_retry(self, task_queue):
        """Test updating task status to failed with retry."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        with patch.object(task_queue, '_handle_task_retry') as mock_retry:
            await task_queue.update_task_status(
                "task_001",
                TaskStatus.FAILED,
                error_message="Task failed"
            )

            mock_retry.assert_called_once()
            mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_task_status(self, task_queue):
        """Test getting task status."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_row = Mock()
        mock_row.id = "task_001"
        mock_row.agent_id = "test_agent"
        mock_row.status = "completed"
        mock_row.assigned_at = datetime.now()
        mock_row.started_at = datetime.now()
        mock_row.completed_at = datetime.now()
        mock_row.output_data = '{"result": "success"}'
        mock_row.error_message = None
        mock_row.attempt_number = 1
        mock_row.next_retry_at = None
        mock_row.execution_time_seconds = 10.5
        mock_row.queue_wait_seconds = 2.0

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        status = await task_queue.get_task_status("task_001")

        assert status is not None
        assert status.task_id == "task_001"
        assert status.agent_id == "test_agent"
        assert status.status == TaskStatus.COMPLETED
        assert status.execution_time_seconds == 10.5

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, task_queue):
        """Test getting status for non-existent task."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        status = await task_queue.get_task_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_get_queue_size(self, task_queue):
        """Test getting queue size."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        size = await task_queue.get_queue_size()
        assert size == 5

    @pytest.mark.asyncio
    async def test_get_queue_statistics(self, task_queue):
        """Test getting queue statistics."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        # Mock status counts query
        mock_status_result = Mock()
        mock_status_result.__iter__ = Mock(return_value=iter([
            Mock(status="pending", count=5),
            Mock(status="completed", count=10),
            Mock(status="failed", count=2),
        ]))

        # Mock timing stats query
        mock_timing_result = Mock()
        mock_timing_result.fetchone.return_value = Mock(
            avg_execution_time=15.5,
            avg_queue_wait=3.2,
        )

        mock_session.execute.side_effect = [mock_status_result, mock_timing_result]

        stats = await task_queue.get_queue_statistics()

        assert "status_counts" in stats
        assert "average_execution_time_seconds" in stats
        assert "average_queue_wait_seconds" in stats
        assert "total_tasks" in stats

        assert stats["status_counts"]["pending"] == 5
        assert stats["status_counts"]["completed"] == 10
        assert stats["status_counts"]["failed"] == 2
        assert stats["total_tasks"] == 17

    @pytest.mark.asyncio
    async def test_validate_dependencies_success(self, task_queue):
        """Test successful dependency validation."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.scalar.return_value = 2  # Found 2 dependencies
        mock_session.execute.return_value = mock_result

        # Should not raise exception
        await task_queue._validate_dependencies(["dep1", "dep2"])

    @pytest.mark.asyncio
    async def test_validate_dependencies_missing(self, task_queue):
        """Test dependency validation with missing dependencies."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.scalar.return_value = 1  # Found only 1 of 2 dependencies
        mock_session.execute.return_value = mock_result

        with pytest.raises(ValueError, match="Missing dependency tasks"):
            await task_queue._validate_dependencies(["dep1", "dep2"])

    @pytest.mark.asyncio
    async def test_dependencies_completed_true(self, task_queue):
        """Test checking dependencies when all are completed."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.scalar.return_value = 2  # All 2 dependencies completed
        mock_session.execute.return_value = mock_result

        result = await task_queue._dependencies_completed(["dep1", "dep2"])
        assert result is True

    @pytest.mark.asyncio
    async def test_dependencies_completed_false(self, task_queue):
        """Test checking dependencies when some are not completed."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        mock_result = Mock()
        mock_result.scalar.return_value = 1  # Only 1 of 2 dependencies completed
        mock_session.execute.return_value = mock_result

        result = await task_queue._dependencies_completed(["dep1", "dep2"])
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_task_retry_within_limit(self, task_queue):
        """Test task retry handling when within retry limit."""
        mock_session = AsyncMock()

        mock_row = Mock()
        mock_row.attempt_number = 1
        mock_row.max_retries = 3
        mock_row.retry_delay_seconds = 60
        mock_row.retry_backoff_multiplier = 2.0

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        update_values = {}
        await task_queue._handle_task_retry(mock_session, "task_001", update_values)

        # Should schedule for retry
        assert update_values["status"] == TaskStatus.RETRYING.value
        assert update_values["attempt_number"] == 2
        assert "next_retry_at" in update_values

    @pytest.mark.asyncio
    async def test_handle_task_retry_exceeded_limit(self, task_queue):
        """Test task retry handling when retry limit exceeded."""
        mock_session = AsyncMock()

        mock_row = Mock()
        mock_row.attempt_number = 3
        mock_row.max_retries = 3
        mock_row.retry_delay_seconds = 60
        mock_row.retry_backoff_multiplier = 2.0

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        update_values = {}
        await task_queue._handle_task_retry(mock_session, "task_001", update_values)

        # Should mark as failed
        assert update_values["status"] == TaskStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_process_retries(self, task_queue):
        """Test processing tasks scheduled for retry."""
        mock_session = AsyncMock()
        task_queue.async_session_maker.return_value.__aenter__.return_value = mock_session

        # Mock tasks ready for retry
        mock_row1 = Mock(id="task_001", attempt_number=2)
        mock_row2 = Mock(id="task_002", attempt_number=1)
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_row1, mock_row2]))
        mock_session.execute.return_value = mock_result

        await task_queue._process_retries()

        # Should reset tasks to pending
        assert mock_session.execute.call_count >= 2  # Select + Updates
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown(self, task_queue):
        """Test task queue shutdown."""
        # Mock the processing task
        task_queue._processing_task = AsyncMock()
        task_queue._processing_task.done.return_value = False

        await task_queue.shutdown()

        # Should signal shutdown and cleanup
        assert task_queue._shutdown_event.is_set()
        task_queue._processing_task.cancel.assert_called()
        task_queue.engine.dispose.assert_called()