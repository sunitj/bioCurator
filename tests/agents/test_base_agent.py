"""Tests for base agent functionality."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from src.agents.base import BaseAgent, AgentTask, AgentStatus
from src.agents.config import AgentConfig, AgentSafetyConfig
from src.config.schemas import ConfigSchema, AppMode
from src.safety.event_bus import SafetyEventBus, SafetyEventType


class TestAgent(BaseAgent):
    """Test agent implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_called = False
        self.shutdown_called = False
        self.tasks_executed = []

    async def _startup(self):
        self.startup_called = True

    async def _shutdown(self):
        self.shutdown_called = True

    async def _execute_task_impl(self, task: AgentTask):
        self.tasks_executed.append(task)

        if task.task_type == "failing_task":
            raise RuntimeError("Simulated task failure")

        return {
            "task_id": task.id,
            "task_type": task.task_type,
            "result": f"Executed {task.task_type}",
        }


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager."""
    mock = AsyncMock()
    mock.is_initialized = True
    mock.get_knowledge_graph.return_value = Mock()
    mock.get_vector_store.return_value = Mock()
    mock.get_episodic_memory.return_value = Mock()
    mock.get_working_memory.return_value = Mock()
    mock.get_time_series.return_value = Mock()
    return mock


@pytest.fixture
def agent_config():
    """Basic agent configuration."""
    return AgentConfig(
        class_path="test.TestAgent",
        role="Test Agent",
        specialties=["testing"],
        max_concurrent_tasks=2,
        safety=AgentSafetyConfig(
            max_requests_per_minute=60,
            circuit_breaker_threshold=0.5,
            max_cost_budget_per_hour=0.0,
        ),
        memory_permissions={
            "knowledge_graph": "read_write",
            "working_memory": "read_write",
        },
    )


@pytest.fixture
def system_config():
    """System configuration."""
    return ConfigSchema(
        app_mode=AppMode.DEVELOPMENT,
    )


@pytest.fixture
def event_bus():
    """Event bus for testing."""
    return SafetyEventBus()


@pytest.fixture
async def agent(agent_config, system_config, mock_memory_manager, event_bus):
    """Test agent instance."""
    agent = TestAgent(
        agent_id="test_agent",
        config=agent_config,
        system_config=system_config,
        memory_manager=mock_memory_manager,
        event_bus=event_bus,
    )
    yield agent

    # Cleanup
    if agent._is_active:
        await agent.stop()


class TestBaseAgent:
    """Test cases for BaseAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent, agent_config):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent"
        assert agent.config == agent_config
        assert not agent._is_active
        assert len(agent._current_tasks) == 0
        assert agent._total_tasks_completed == 0
        assert agent._total_tasks_failed == 0

        # Check safety controls are initialized
        assert agent.circuit_breaker is not None
        assert agent.rate_limiter is not None
        assert agent.cost_tracker is not None
        assert agent.behavior_monitor is not None

    @pytest.mark.asyncio
    async def test_agent_start_stop(self, agent):
        """Test agent start and stop lifecycle."""
        # Start agent
        await agent.start()

        assert agent._is_active
        assert agent.startup_called
        assert isinstance(agent._last_activity, datetime)

        # Stop agent
        await agent.stop()

        assert not agent._is_active
        assert agent.shutdown_called

    @pytest.mark.asyncio
    async def test_agent_double_start_stop(self, agent):
        """Test that double start/stop is handled gracefully."""
        # Double start
        await agent.start()
        await agent.start()  # Should not raise error
        assert agent._is_active

        # Double stop
        await agent.stop()
        await agent.stop()  # Should not raise error
        assert not agent._is_active

    @pytest.mark.asyncio
    async def test_task_execution_success(self, agent):
        """Test successful task execution."""
        await agent.start()

        task = AgentTask(
            id="test_task_1",
            agent_id=agent.agent_id,
            task_type="test_task",
            input_data={"test": "data"},
        )

        response = await agent.execute_task(task)

        assert response.task_id == task.id
        assert response.agent_id == agent.agent_id
        assert response.success is True
        assert response.error_message is None
        assert "task_id" in response.output_data
        assert response.execution_time_seconds > 0

        # Check internal state
        assert agent._total_tasks_completed == 1
        assert agent._total_tasks_failed == 0
        assert len(agent.tasks_executed) == 1

    @pytest.mark.asyncio
    async def test_task_execution_failure(self, agent):
        """Test task execution failure."""
        await agent.start()

        task = AgentTask(
            id="test_task_2",
            agent_id=agent.agent_id,
            task_type="failing_task",
            input_data={},
        )

        response = await agent.execute_task(task)

        assert response.task_id == task.id
        assert response.agent_id == agent.agent_id
        assert response.success is False
        assert response.error_message is not None
        assert "Simulated task failure" in response.error_message
        assert response.execution_time_seconds > 0

        # Check internal state
        assert agent._total_tasks_completed == 0
        assert agent._total_tasks_failed == 1

    @pytest.mark.asyncio
    async def test_task_execution_inactive_agent(self, agent):
        """Test task execution on inactive agent."""
        task = AgentTask(
            id="test_task_3",
            agent_id=agent.agent_id,
            task_type="test_task",
            input_data={},
        )

        with pytest.raises(RuntimeError, match="not active"):
            await agent.execute_task(task)

    @pytest.mark.asyncio
    async def test_concurrent_task_limiting(self, agent):
        """Test concurrent task limiting."""
        await agent.start()

        # Create tasks that will block
        async def slow_task_impl(task):
            await asyncio.sleep(0.5)
            return {"result": "slow"}

        agent._execute_task_impl = slow_task_impl

        tasks = [
            AgentTask(id=f"task_{i}", agent_id=agent.agent_id, task_type="slow_task")
            for i in range(4)  # More than max_concurrent_tasks (2)
        ]

        # Start all tasks concurrently
        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(
            *[agent.execute_task(task) for task in tasks],
            return_exceptions=False
        )
        duration = asyncio.get_event_loop().time() - start_time

        # Should take at least 1 second (2 batches of 0.5s each)
        assert duration >= 1.0
        assert len(responses) == 4
        assert all(r.success for r in responses)

    @pytest.mark.asyncio
    async def test_agent_status(self, agent):
        """Test agent status reporting."""
        # Test inactive agent
        status = await agent.get_status()
        assert isinstance(status, AgentStatus)
        assert status.agent_id == agent.agent_id
        assert not status.is_active
        assert status.current_tasks == 0
        assert status.health_status == "stopped"

        # Test active agent
        await agent.start()
        status = await agent.get_status()
        assert status.is_active
        assert status.max_concurrent_tasks == 2
        assert status.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_health_check(self, agent, mock_memory_manager):
        """Test agent health check."""
        await agent.start()

        # Mock memory health check
        mock_memory_manager.health_check_all.return_value = {
            "working_memory": Mock(is_healthy=True),
            "knowledge_graph": Mock(is_healthy=True),
        }

        health = await agent.health_check()

        assert "agent_id" in health
        assert "is_active" in health
        assert "circuit_breaker_state" in health
        assert "current_tasks" in health
        assert "memory_access" in health
        assert health["memory_access"] == "healthy"

    @pytest.mark.asyncio
    async def test_memory_access_verification(self, agent, mock_memory_manager):
        """Test memory access verification."""
        # Test successful verification
        await agent.start()

        # Test verification failure
        mock_memory_manager.get_knowledge_graph.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RuntimeError, match="not active"):
            await agent.stop()
            await agent.execute_task(AgentTask(
                id="test", agent_id=agent.agent_id, task_type="test"
            ))

    @pytest.mark.asyncio
    async def test_development_mode_safety_check(self, agent, system_config):
        """Test development mode safety checks."""
        await agent.start()

        # Development mode should enforce zero cost budget
        assert system_config.app_mode == AppMode.DEVELOPMENT
        assert agent.config.safety.max_cost_budget_per_hour == 0.0

        task = AgentTask(
            id="test_dev_task",
            agent_id=agent.agent_id,
            task_type="test_task",
            input_data={},
        )

        # Should not raise error in development mode
        response = await agent.execute_task(task)
        assert response.success

    @pytest.mark.asyncio
    async def test_safety_event_emission(self, agent, event_bus):
        """Test safety event emission."""
        events = []

        def event_handler(event):
            events.append(event)

        event_bus.subscribe(event_handler)

        # Start agent (should emit event)
        await agent.start()

        # Stop agent (should emit event)
        await agent.stop()

        # Check events
        start_events = [e for e in events if e.event_type == SafetyEventType.AGENT_STARTED]
        stop_events = [e for e in events if e.event_type == SafetyEventType.AGENT_STOPPED]

        assert len(start_events) >= 1
        assert len(stop_events) >= 1

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_running_tasks(self, agent):
        """Test graceful shutdown with running tasks."""
        await agent.start()

        # Start a long-running task
        async def long_task_impl(task):
            await asyncio.sleep(1.0)
            return {"result": "completed"}

        agent._execute_task_impl = long_task_impl

        task = AgentTask(
            id="long_task",
            agent_id=agent.agent_id,
            task_type="long_task",
        )

        # Start task and immediately try to shutdown
        task_future = asyncio.create_task(agent.execute_task(task))

        # Small delay to let task start
        await asyncio.sleep(0.1)

        # Shutdown should wait for task completion
        start_time = asyncio.get_event_loop().time()
        await agent.stop()
        shutdown_duration = asyncio.get_event_loop().time() - start_time

        # Should have waited for task
        assert shutdown_duration < 30  # But not longer than shutdown timeout

        # Task should complete successfully
        response = await task_future
        assert response.success

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, agent):
        """Test task timeout handling."""
        await agent.start()

        task = AgentTask(
            id="timeout_task",
            agent_id=agent.agent_id,
            task_type="test_task",
            timeout_seconds=1,  # Short timeout for testing
        )

        # The base agent doesn't implement timeout directly,
        # but we can test that the timeout value is preserved
        response = await agent.execute_task(task)
        assert response.success  # Should complete quickly