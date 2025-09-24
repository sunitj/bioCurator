"""Agent configuration schemas and models."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryPermission(str, Enum):
    """Memory access permission levels."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ADMIN = "admin"


class TaskAllocationStrategy(str, Enum):
    """Task allocation strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY = "priority"
    SINGLE_AGENT = "single_agent"


class ModelProvider(str, Enum):
    """AI model providers."""

    OLLAMA = "ollama"
    CLOUD = "cloud"


class AgentSafetyConfig(BaseModel):
    """Safety configuration for individual agents."""

    max_requests_per_minute: int = Field(default=30, ge=1, le=1000)
    circuit_breaker_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    circuit_breaker_window: int = Field(default=300, ge=60, le=3600)
    circuit_breaker_min_volume: int = Field(default=10, ge=1, le=1000)
    burst_capacity: int = Field(default=60, ge=1, le=10000)
    refill_rate: float = Field(default=0.5, ge=0.1, le=10.0)
    max_cost_budget_per_hour: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """Model configuration for agents."""

    primary: str = Field(..., description="Primary model identifier")
    fallback: str = Field(..., description="Fallback model identifier")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=100000)
    context_window: int = Field(default=8192, ge=512, le=1000000)

    model_config = ConfigDict(extra="forbid")


class AgentConfig(BaseModel):
    """Configuration for individual agents."""

    class_path: str = Field(..., alias="class", description="Agent class import path")
    role: str = Field(..., description="Agent role description")
    specialties: List[str] = Field(default_factory=list)
    memory_focus: List[str] = Field(default_factory=list)
    enabled: bool = Field(default=True)
    max_concurrent_tasks: int = Field(default=3, ge=1, le=100)

    # Safety configuration
    safety: AgentSafetyConfig = Field(default_factory=AgentSafetyConfig)

    # Memory permissions
    memory_permissions: Dict[str, MemoryPermission] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CommunicationConfig(BaseModel):
    """Communication protocol configuration."""

    protocol: str = Field(default="async_message_passing")
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    max_message_size: int = Field(default=10485760, ge=1024, le=104857600)  # 10MB

    # Task queue configuration
    task_queue: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class CoordinationConfig(BaseModel):
    """Agent coordination configuration."""

    max_parallel_agents: int = Field(default=3, ge=1, le=20)
    task_allocation_strategy: TaskAllocationStrategy = Field(
        default=TaskAllocationStrategy.ROUND_ROBIN
    )
    load_balancing_enabled: bool = Field(default=True)
    auto_scaling_enabled: bool = Field(default=False)

    # Auto-scaling configuration (if enabled)
    auto_scaling: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class MonitoringConfig(BaseModel):
    """Agent monitoring configuration."""

    health_check_interval_seconds: int = Field(default=30, ge=5, le=300)
    performance_metrics_enabled: bool = Field(default=True)
    behavior_monitoring_enabled: bool = Field(default=True)
    audit_logging_enabled: bool = Field(default=True)
    debug_logging_enabled: bool = Field(default=False)
    trace_all_requests: bool = Field(default=False)

    # Metrics configuration
    metrics: Dict[str, bool] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    """System-level configuration."""

    name: str = Field(default="biocurator")
    description: str = Field(default="Multi-agent system")
    mode: Optional[str] = Field(default=None)
    cost_budget_per_hour: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ModelsConfig(BaseModel):
    """Models configuration for all agents."""

    provider: ModelProvider = Field(default=ModelProvider.OLLAMA)
    ollama_host: str = Field(default="http://localhost:11434")

    # Per-agent model configurations
    research_director: Optional[ModelConfig] = None
    literature_scout: Optional[ModelConfig] = None
    deep_reader: Optional[ModelConfig] = None
    domain_specialist: Optional[ModelConfig] = None
    knowledge_weaver: Optional[ModelConfig] = None
    memory_keeper: Optional[ModelConfig] = None

    model_config = ConfigDict(extra="forbid")


class SafetyConfig(BaseModel):
    """Global safety configuration."""

    enforce_zero_cost: bool = Field(default=True)
    cloud_model_guard: bool = Field(default=True)
    max_requests_per_session: int = Field(default=500, ge=1, le=10000)
    session_timeout_hours: int = Field(default=2, ge=1, le=24)
    cost_monitoring_enabled: bool = Field(default=True)
    budget_alerts_enabled: bool = Field(default=True)
    budget_alert_thresholds: List[float] = Field(default_factory=lambda: [0.5, 0.8, 0.95])

    model_config = ConfigDict(extra="forbid")


class AlertsConfig(BaseModel):
    """Alerting configuration."""

    cost_budget_warning: bool = Field(default=True)
    high_error_rate: bool = Field(default=True)
    agent_failures: bool = Field(default=True)
    memory_system_issues: bool = Field(default=True)

    notification_channels: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AgentSystemConfig(BaseModel):
    """Complete agent system configuration."""

    version: str = Field(default="1.0")
    system: SystemConfig = Field(default_factory=SystemConfig)
    models: Optional[ModelsConfig] = None
    safety: Optional[SafetyConfig] = None
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    alerts: Optional[AlertsConfig] = None

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v):
        """Validate agent configurations."""
        if not v:
            raise ValueError("At least one agent must be configured")

        # Check for required research_director agent
        if "research_director" not in v:
            raise ValueError("research_director agent is required")

        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate configuration version."""
        if v not in ["1.0"]:
            raise ValueError(f"Unsupported configuration version: {v}")
        return v

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )