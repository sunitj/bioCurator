"""Configuration schemas for validation and type safety."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppMode(str, Enum):
    """Application operation modes."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    HYBRID = "hybrid"


class ModelProvider(str, Enum):
    """AI model providers."""

    OLLAMA = "ollama"
    CLOUD = "cloud"


class SafetyConfig(BaseModel):
    """Safety controls configuration."""

    max_cost_budget: float = Field(default=0.0, ge=0.0, description="Maximum cost budget in USD")
    rate_limit_per_minute: int = Field(
        default=30, ge=1, le=1000, description="Rate limit per minute"
    )
    circuit_breaker_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Error rate threshold"
    )
    circuit_breaker_window: int = Field(
        default=300, ge=60, le=3600, description="Rolling window in seconds"
    )
    circuit_breaker_min_volume: int = Field(
        default=10, ge=1, le=1000, description="Min requests before evaluation"
    )
    burst_capacity: int = Field(default=60, ge=1, description="Token bucket burst capacity")
    refill_rate: float = Field(
        default=0.5, ge=0.1, le=10.0, description="Token bucket refill rate per second"
    )


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    # Redis - Working memory and caching
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_pool_size: int = Field(default=10, ge=1, le=100)
    redis_socket_timeout: int = Field(default=5, ge=1, le=60)

    # Neo4j - Knowledge graph
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_max_connection_lifetime: int = Field(default=3600, ge=300, le=7200)
    neo4j_max_connection_pool_size: int = Field(default=50, ge=1, le=200)
    neo4j_connection_timeout: int = Field(default=30, ge=5, le=120)

    # PostgreSQL - Episodic memory
    postgres_url: str = Field(default="postgresql://user:password@localhost:5432/biocurator")
    postgres_pool_size: int = Field(default=20, ge=1, le=100)
    postgres_max_overflow: int = Field(default=30, ge=0, le=100)
    postgres_pool_timeout: int = Field(default=30, ge=5, le=120)
    postgres_pool_recycle: int = Field(default=3600, ge=300, le=7200)

    # Qdrant - Vector embeddings
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_grpc_port: int = Field(default=6334, ge=1024, le=65535)
    qdrant_timeout: int = Field(default=60, ge=5, le=300)
    qdrant_prefer_grpc: bool = Field(default=True)

    # InfluxDB - Time series metrics
    influxdb_url: str = Field(default="http://localhost:8086")
    influxdb_token: str = Field(default="dev_token_12345")
    influxdb_org: str = Field(default="biocurator")
    influxdb_bucket: str = Field(default="agent_metrics")
    influxdb_timeout: int = Field(default=30, ge=5, le=120)


class ModelConfig(BaseModel):
    """AI model configuration."""

    provider: ModelProvider = Field(default=ModelProvider.OLLAMA)
    ollama_host: str = Field(default="http://localhost:11434")
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    huggingface_token: str | None = Field(default=None)


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    prometheus_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    enable_profiling: bool = Field(default=False)
    enable_debug: bool = Field(default=False)


class PathConfig(BaseModel):
    """File system paths configuration."""

    data_path: str = Field(default="/data")
    log_path: str = Field(default="/logs")
    config_path: str = Field(default="/app/configs")


class ConfigSchema(BaseModel):
    """Complete application configuration schema."""

    # Core settings
    app_mode: AppMode = Field(default=AppMode.DEVELOPMENT)
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    port: int = Field(default=8080, ge=1024, le=65535)
    secret_key: str = Field(default="change-me-in-production", min_length=16)

    # Component configurations
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # CORS configuration
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])

    @field_validator("app_mode")
    @classmethod
    def validate_app_mode(cls, v):
        """Validate application mode."""
        if v not in AppMode:
            raise ValueError(f"app_mode must be one of {[mode.value for mode in AppMode]}")
        return v

    @field_validator("safety")
    @classmethod
    def validate_safety_for_mode(cls, v, info):
        """Validate safety configuration based on app mode."""
        app_mode = info.data.get("app_mode") if info.data else None

        if app_mode == AppMode.DEVELOPMENT and v.max_cost_budget != 0.0:
            # Development mode must have zero cost budget
            raise ValueError("Development mode must have max_cost_budget=0.0")

        return v

    @field_validator("models")
    @classmethod
    def validate_models_for_mode(cls, v, info):
        """Validate model configuration based on app mode."""
        app_mode = info.data.get("app_mode") if info.data else None

        if app_mode == AppMode.DEVELOPMENT:
            # Development mode should use Ollama
            if v.provider != ModelProvider.OLLAMA:
                raise ValueError("Development mode should use Ollama provider")

        elif (
            app_mode == AppMode.PRODUCTION
            and v.provider == ModelProvider.CLOUD
            and not v.openai_api_key
            and not v.anthropic_api_key
        ):
            # Production mode requires API keys for cloud providers
            raise ValueError("Cloud provider requires at least one API key")

        return v

    model_config = ConfigDict(
        env_prefix="",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid",  # Reject unknown configuration keys
    )
