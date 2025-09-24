"""Agent registry for lifecycle management and discovery."""

import asyncio
import importlib
from typing import Dict, List, Optional, Type

import yaml

from ..config.schemas import ConfigSchema
from ..logging import get_logger
from ..memory.interfaces import MemoryManager
from ..safety.event_bus import SafetyEvent, SafetyEventBus, SafetyEventType
from .base import BaseAgent, AgentStatus
from .config import AgentSystemConfig

logger = get_logger(__name__)


class AgentRegistry:
    """Registry for managing agent lifecycle and discovery."""

    def __init__(
        self,
        system_config: ConfigSchema,
        memory_manager: MemoryManager,
        event_bus: Optional[SafetyEventBus] = None,
    ):
        """Initialize agent registry.

        Args:
            system_config: System configuration
            memory_manager: Memory system interface
            event_bus: Event bus for safety events
        """
        self.system_config = system_config
        self.memory_manager = memory_manager
        self.event_bus = event_bus or SafetyEventBus()

        # Agent storage
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_configs: Dict[str, AgentSystemConfig] = {}
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info("Agent registry initialized")

    async def load_configuration(self, config_files: List[str]) -> None:
        """Load agent configuration from YAML files.

        Args:
            config_files: List of configuration file paths to load
        """
        logger.info("Loading agent configuration", files=config_files)

        try:
            # Load base configuration
            base_config = None
            mode_configs = []

            for config_file in config_files:
                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f)

                parsed_config = AgentSystemConfig(**config_data)

                # Determine if this is a base or mode-specific config
                if config_file.endswith("cagents.yaml"):
                    base_config = parsed_config
                else:
                    mode_configs.append(parsed_config)

            if not base_config:
                raise ValueError("Base configuration (cagents.yaml) not found")

            # Merge configurations based on app mode
            final_config = self._merge_configurations(base_config, mode_configs)

            # Store final configuration
            self._agent_configs["main"] = final_config

            logger.info("Agent configuration loaded successfully", agents=list(final_config.agents.keys()))

        except Exception as e:
            logger.error("Failed to load agent configuration", error=str(e))
            raise

    async def initialize_agents(self) -> None:
        """Initialize all configured agents."""
        if "main" not in self._agent_configs:
            raise RuntimeError("Agent configuration not loaded")

        config = self._agent_configs["main"]

        logger.info("Initializing agents", count=len(config.agents))

        for agent_id, agent_config in config.agents.items():
            if not agent_config.enabled:
                logger.info(f"Skipping disabled agent {agent_id}")
                continue

            try:
                agent = await self._create_agent(agent_id, agent_config)
                self._agents[agent_id] = agent

                logger.info(f"Agent {agent_id} initialized", role=agent_config.role)

            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}", error=str(e))
                self.event_bus.emit(
                    SafetyEvent(
                        event_type=SafetyEventType.AGENT_FAILURE,
                        component=agent_id,
                        message=f"Failed to initialize agent {agent_id}: {str(e)}",
                        metadata={"agent_id": agent_id, "error": str(e)},
                    )
                )
                # Continue with other agents rather than failing completely

        if not self._agents:
            raise RuntimeError("No agents were successfully initialized")

        logger.info("Agent initialization completed", active_agents=list(self._agents.keys()))

    async def start_agents(self) -> None:
        """Start all initialized agents."""
        logger.info("Starting agents", count=len(self._agents))

        start_tasks = []
        for agent_id, agent in self._agents.items():
            task = asyncio.create_task(self._start_agent_safe(agent_id, agent))
            start_tasks.append(task)

        # Wait for all agents to start
        results = await asyncio.gather(*start_tasks, return_exceptions=True)

        # Check results
        successful_starts = 0
        for agent_id, result in zip(self._agents.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to start agent {agent_id}", error=str(result))
            else:
                successful_starts += 1

        if successful_starts == 0:
            raise RuntimeError("No agents started successfully")

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Started {successful_starts}/{len(self._agents)} agents successfully")

    async def stop_agents(self) -> None:
        """Stop all agents gracefully."""
        logger.info("Stopping agents", count=len(self._agents))

        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all agents
        stop_tasks = []
        for agent_id, agent in self._agents.items():
            task = asyncio.create_task(self._stop_agent_safe(agent_id, agent))
            stop_tasks.append(task)

        # Wait for all agents to stop
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        logger.info("All agents stopped")

    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            BaseAgent or None if not found
        """
        return self._agents.get(agent_id)

    async def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return self._agents.copy()

    async def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStatus or None if agent not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        return await agent.get_status()

    async def get_all_agent_status(self) -> Dict[str, AgentStatus]:
        """Get status for all agents."""
        status_tasks = []
        for agent_id, agent in self._agents.items():
            task = asyncio.create_task(agent.get_status())
            status_tasks.append((agent_id, task))

        results = {}
        for agent_id, task in status_tasks:
            try:
                status = await task
                results[agent_id] = status
            except Exception as e:
                logger.error(f"Failed to get status for agent {agent_id}", error=str(e))
                # Create error status
                results[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    is_active=False,
                    current_tasks=0,
                    max_concurrent_tasks=0,
                    total_tasks_completed=0,
                    total_tasks_failed=0,
                    last_activity=None,
                    health_status="error",
                )

        return results

    async def health_check_all(self) -> Dict[str, Dict[str, any]]:
        """Perform health check on all agents."""
        health_tasks = []
        for agent_id, agent in self._agents.items():
            task = asyncio.create_task(agent.health_check())
            health_tasks.append((agent_id, task))

        results = {}
        for agent_id, task in health_tasks:
            try:
                health_info = await task
                results[agent_id] = health_info
            except Exception as e:
                logger.error(f"Health check failed for agent {agent_id}", error=str(e))
                results[agent_id] = {
                    "agent_id": agent_id,
                    "is_active": False,
                    "error": str(e),
                    "health_status": "error",
                }

        return results

    def get_configuration(self) -> Optional[AgentSystemConfig]:
        """Get the current agent system configuration."""
        return self._agent_configs.get("main")

    # Private methods

    async def _create_agent(self, agent_id: str, agent_config) -> BaseAgent:
        """Create an agent instance from configuration."""
        try:
            # Import the agent class
            module_path, class_name = agent_config.class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            agent_class: Type[BaseAgent] = getattr(module, class_name)

            # Create agent instance
            agent = agent_class(
                agent_id=agent_id,
                config=agent_config,
                system_config=self.system_config,
                memory_manager=self.memory_manager,
                event_bus=self.event_bus,
            )

            return agent

        except ImportError as e:
            logger.error(f"Failed to import agent class {agent_config.class_path}", error=str(e))
            raise RuntimeError(f"Agent class not found: {agent_config.class_path}") from e
        except AttributeError as e:
            logger.error(f"Agent class not found in module {agent_config.class_path}", error=str(e))
            raise RuntimeError(f"Invalid agent class: {agent_config.class_path}") from e
        except Exception as e:
            logger.error(f"Failed to create agent {agent_id}", error=str(e))
            raise

    async def _start_agent_safe(self, agent_id: str, agent: BaseAgent) -> None:
        """Safely start an agent with error handling."""
        try:
            await agent.start()
            logger.info(f"Agent {agent_id} started successfully")
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}", error=str(e))
            self.event_bus.emit(
                SafetyEvent(
                    event_type=SafetyEventType.AGENT_FAILURE,
                    component=agent_id,
                    message=f"Failed to start agent {agent_id}: {str(e)}",
                    metadata={"agent_id": agent_id, "error": str(e)},
                )
            )
            raise

    async def _stop_agent_safe(self, agent_id: str, agent: BaseAgent) -> None:
        """Safely stop an agent with error handling."""
        try:
            await agent.stop()
            logger.info(f"Agent {agent_id} stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}", error=str(e))

    def _merge_configurations(
        self, base_config: AgentSystemConfig, mode_configs: List[AgentSystemConfig]
    ) -> AgentSystemConfig:
        """Merge base configuration with mode-specific overrides."""
        final_config = base_config.model_copy(deep=True)

        # Find the configuration matching the current app mode
        app_mode = self.system_config.app_mode.value
        mode_config = None

        for config in mode_configs:
            if config.system.mode == app_mode:
                mode_config = config
                break

        if not mode_config:
            logger.warning(f"No mode-specific configuration found for {app_mode}, using base only")
            return final_config

        # Merge system settings
        if mode_config.system:
            for field, value in mode_config.system.model_dump(exclude_unset=True).items():
                setattr(final_config.system, field, value)

        # Merge models configuration
        if mode_config.models:
            final_config.models = mode_config.models

        # Merge safety configuration
        if mode_config.safety:
            final_config.safety = mode_config.safety

        # Merge agent-specific overrides
        for agent_id, mode_agent_config in mode_config.agents.items():
            if agent_id in final_config.agents:
                # Merge agent configuration
                base_agent = final_config.agents[agent_id]

                # Update fields from mode config
                for field, value in mode_agent_config.model_dump(exclude_unset=True).items():
                    setattr(base_agent, field, value)

        # Merge other configurations
        if mode_config.communication:
            final_config.communication = mode_config.communication

        if mode_config.coordination:
            final_config.coordination = mode_config.coordination

        if mode_config.monitoring:
            final_config.monitoring = mode_config.monitoring

        if mode_config.alerts:
            final_config.alerts = mode_config.alerts

        return final_config

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        if "main" not in self._agent_configs:
            return

        config = self._agent_configs["main"]
        interval = config.monitoring.health_check_interval_seconds

        logger.info(f"Starting health check loop with {interval}s interval")

        try:
            while True:
                await asyncio.sleep(interval)

                # Perform health checks
                health_results = await self.health_check_all()

                # Log any unhealthy agents
                unhealthy_agents = [
                    agent_id
                    for agent_id, health in health_results.items()
                    if health.get("is_active", False) is False
                ]

                if unhealthy_agents:
                    logger.warning("Unhealthy agents detected", agents=unhealthy_agents)

                    for agent_id in unhealthy_agents:
                        self.event_bus.emit(
                            SafetyEvent(
                                event_type=SafetyEventType.AGENT_HEALTH_DEGRADED,
                                component=agent_id,
                                message=f"Agent {agent_id} health check failed",
                                metadata={"agent_id": agent_id, "health": health_results[agent_id]},
                            )
                        )

        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error("Health check loop error", error=str(e))