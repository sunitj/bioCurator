"""Tests for agent registry functionality."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import yaml

from src.agents.registry import AgentRegistry
from src.agents.base import BaseAgent, AgentTask
from src.agents.config import AgentSystemConfig, AgentConfig, AgentSafetyConfig
from src.config.schemas import ConfigSchema, AppMode
from src.safety.event_bus import SafetyEventBus


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_called = False
        self.shutdown_called = False

    async def _startup(self):
        self.startup_called = True

    async def _shutdown(self):
        self.shutdown_called = True

    async def _execute_task_impl(self, task):
        return {"result": f"Mock execution of {task.task_type}"}


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
def sample_agent_config():
    """Sample agent system configuration."""
    return AgentSystemConfig(
        version="1.0",
        agents={
            "test_agent": AgentConfig(
                class_path="tests.agents.test_registry.MockAgent",
                role="Test Agent",
                specialties=["testing"],
                enabled=True,
                safety=AgentSafetyConfig(),
                memory_permissions={
                    "knowledge_graph": "read_write",
                    "working_memory": "read",
                },
            ),
            "disabled_agent": AgentConfig(
                class_path="tests.agents.test_registry.MockAgent",
                role="Disabled Agent",
                specialties=["testing"],
                enabled=False,
                safety=AgentSafetyConfig(),
            ),
        },
    )


@pytest.fixture
def config_files(sample_agent_config):
    """Create temporary config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Base config
        base_config_path = Path(temp_dir) / "cagents.yaml"
        with open(base_config_path, "w") as f:
            yaml.safe_dump(sample_agent_config.model_dump(), f)

        # Development config (override)
        dev_config = AgentSystemConfig(
            version="1.0",
            system={"mode": "development"},
            agents={
                "test_agent": AgentConfig(
                    class_path="tests.agents.test_registry.MockAgent",
                    role="Test Agent",
                    specialties=["testing"],
                    enabled=True,
                    safety=AgentSafetyConfig(max_requests_per_minute=30),
                ),
            },
        )

        dev_config_path = Path(temp_dir) / "cagents.development.yaml"
        with open(dev_config_path, "w") as f:
            yaml.safe_dump(dev_config.model_dump(), f)

        yield [str(base_config_path), str(dev_config_path)]


@pytest.fixture
async def registry(system_config, mock_memory_manager, event_bus):
    """Agent registry instance."""
    registry = AgentRegistry(
        system_config=system_config,
        memory_manager=mock_memory_manager,
        event_bus=event_bus,
    )
    yield registry

    # Cleanup
    try:
        await registry.stop_agents()
    except:
        pass


class TestAgentRegistry:
    """Test cases for AgentRegistry."""

    @pytest.mark.asyncio
    async def test_registry_initialization(self, registry, system_config):
        """Test registry initialization."""
        assert registry.system_config == system_config
        assert registry.memory_manager is not None
        assert registry.event_bus is not None
        assert len(registry._agents) == 0
        assert len(registry._agent_configs) == 0

    @pytest.mark.asyncio
    async def test_load_configuration_success(self, registry, config_files):
        """Test successful configuration loading."""
        await registry.load_configuration(config_files)

        assert "main" in registry._agent_configs
        config = registry._agent_configs["main"]

        # Should have merged base and development configs
        assert config.version == "1.0"
        assert "test_agent" in config.agents
        assert "disabled_agent" in config.agents

        # Development config should override base config
        test_agent = config.agents["test_agent"]
        assert test_agent.safety.max_requests_per_minute == 30  # From dev config

    @pytest.mark.asyncio
    async def test_load_configuration_missing_base(self, registry):
        """Test configuration loading with missing base config."""
        with pytest.raises(ValueError, match="Base configuration"):
            await registry.load_configuration(["nonexistent.yaml"])

    @pytest.mark.asyncio
    async def test_load_configuration_invalid_yaml(self, registry):
        """Test configuration loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises((yaml.YAMLError, Exception)):
                await registry.load_configuration([f.name])

    @pytest.mark.asyncio
    async def test_initialize_agents_success(self, registry, config_files):
        """Test successful agent initialization."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            # Mock the module and class
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()

            # Should initialize only enabled agents
            agents = await registry.get_all_agents()
            assert len(agents) == 1
            assert "test_agent" in agents
            assert "disabled_agent" not in agents

            agent = agents["test_agent"]
            assert isinstance(agent, MockAgent)
            assert agent.agent_id == "test_agent"

    @pytest.mark.asyncio
    async def test_initialize_agents_import_error(self, registry, config_files):
        """Test agent initialization with import error."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module", side_effect=ImportError("Module not found")):
            # Should continue with other agents rather than failing completely
            await registry.initialize_agents()

            # No agents should be initialized due to import error
            agents = await registry.get_all_agents()
            assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_start_stop_agents(self, registry, config_files):
        """Test starting and stopping agents."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()
            await registry.start_agents()

            agents = await registry.get_all_agents()
            agent = agents["test_agent"]
            assert agent.startup_called
            assert agent._is_active

            await registry.stop_agents()
            assert agent.shutdown_called
            assert not agent._is_active

    @pytest.mark.asyncio
    async def test_get_agent_by_id(self, registry, config_files):
        """Test getting agent by ID."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()

            # Test existing agent
            agent = await registry.get_agent("test_agent")
            assert agent is not None
            assert agent.agent_id == "test_agent"

            # Test non-existent agent
            agent = await registry.get_agent("nonexistent")
            assert agent is None

    @pytest.mark.asyncio
    async def test_get_agent_status(self, registry, config_files):
        """Test getting agent status."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()
            await registry.start_agents()

            # Test existing agent
            status = await registry.get_agent_status("test_agent")
            assert status is not None
            assert status.agent_id == "test_agent"
            assert status.is_active

            # Test non-existent agent
            status = await registry.get_agent_status("nonexistent")
            assert status is None

    @pytest.mark.asyncio
    async def test_get_all_agent_status(self, registry, config_files):
        """Test getting status for all agents."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()
            await registry.start_agents()

            all_status = await registry.get_all_agent_status()
            assert len(all_status) == 1
            assert "test_agent" in all_status
            assert all_status["test_agent"].is_active

    @pytest.mark.asyncio
    async def test_health_check_all(self, registry, config_files):
        """Test health check for all agents."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()
            await registry.start_agents()

            health_results = await registry.health_check_all()
            assert len(health_results) == 1
            assert "test_agent" in health_results
            assert "is_active" in health_results["test_agent"]

    @pytest.mark.asyncio
    async def test_health_check_with_error(self, registry, config_files):
        """Test health check with agent error."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()

            # Mock health check to raise error
            agents = await registry.get_all_agents()
            agent = agents["test_agent"]
            agent.health_check = AsyncMock(side_effect=RuntimeError("Health check failed"))

            health_results = await registry.health_check_all()
            assert "test_agent" in health_results
            assert health_results["test_agent"]["health_status"] == "error"

    @pytest.mark.asyncio
    async def test_configuration_merging(self, registry):
        """Test configuration merging between base and mode configs."""
        # Create configs with overlapping settings
        base_config = AgentSystemConfig(
            version="1.0",
            agents={
                "agent1": AgentConfig(
                    class_path="test.Agent",
                    role="Base Role",
                    specialties=["base"],
                    enabled=True,
                    safety=AgentSafetyConfig(max_requests_per_minute=60),
                ),
            },
        )

        mode_config = AgentSystemConfig(
            version="1.0",
            system={"mode": "development"},
            agents={
                "agent1": AgentConfig(
                    class_path="test.Agent",
                    role="Override Role",
                    specialties=["override"],
                    enabled=False,
                    safety=AgentSafetyConfig(max_requests_per_minute=30),
                ),
            },
        )

        merged = registry._merge_configurations(base_config, [mode_config])

        # Check merging worked correctly
        assert merged.system.mode == "development"
        agent1 = merged.agents["agent1"]
        assert agent1.role == "Override Role"  # Should be overridden
        assert agent1.specialties == ["override"]  # Should be overridden
        assert not agent1.enabled  # Should be overridden
        assert agent1.safety.max_requests_per_minute == 30  # Should be overridden

    @pytest.mark.asyncio
    async def test_get_configuration(self, registry, config_files):
        """Test getting current configuration."""
        # Before loading
        config = registry.get_configuration()
        assert config is None

        # After loading
        await registry.load_configuration(config_files)
        config = registry.get_configuration()
        assert config is not None
        assert isinstance(config, AgentSystemConfig)
        assert config.version == "1.0"

    @pytest.mark.asyncio
    async def test_health_check_loop(self, registry, config_files):
        """Test background health check loop."""
        await registry.load_configuration(config_files)

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAgent = MockAgent
            mock_import.return_value = mock_module

            await registry.initialize_agents()

            # Start agents (this starts the health check loop)
            await registry.start_agents()

            # Wait a bit for health check loop to run
            await asyncio.sleep(0.1)

            # Stop agents (this should stop the health check loop)
            await registry.stop_agents()

            # Health check task should be done
            assert registry._health_check_task is None or registry._health_check_task.done()

    @pytest.mark.asyncio
    async def test_no_agents_configured(self, registry):
        """Test handling when no agents are configured."""
        # Create empty config
        empty_config = AgentSystemConfig(version="1.0", agents={})

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "empty.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(empty_config.model_dump(), f)

            with pytest.raises(ValueError, match="At least one agent must be configured"):
                await registry.load_configuration([str(config_path)])

    @pytest.mark.asyncio
    async def test_missing_research_director(self, registry):
        """Test handling when research_director is missing."""
        # Config without research_director
        config = AgentSystemConfig(
            version="1.0",
            agents={
                "other_agent": AgentConfig(
                    class_path="test.Agent",
                    role="Other Agent",
                    specialties=["other"],
                ),
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "no_director.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(config.model_dump(), f)

            with pytest.raises(ValueError, match="research_director agent is required"):
                await registry.load_configuration([str(config_path)])