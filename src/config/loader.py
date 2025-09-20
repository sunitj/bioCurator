"""Configuration loader with environment and YAML support."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from .schemas import ConfigSchema


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing critical keys."""
    pass


class ConfigLoader:
    """Load and validate configuration from environment and YAML files."""
    
    def __init__(self):
        self._config: Optional[ConfigSchema] = None
        self._config_data: Dict[str, Any] = {}
    
    def load_config(self, config_path: Optional[str] = None) -> ConfigSchema:
        """
        Load configuration from environment variables and YAML files.
        
        Args:
            config_path: Optional path to configuration directory
            
        Returns:
            Validated configuration schema
            
        Raises:
            ConfigurationError: If configuration is invalid or missing critical keys
        """
        try:
            # Start with empty config data
            self._config_data = {}
            
            # Load from YAML files
            self._load_yaml_config(config_path)
            
            # Override with environment variables
            self._load_env_config()
            
            # Validate and create config schema
            self._config = ConfigSchema(**self._config_data)
            
            # Validate critical configuration
            self._validate_critical_config()
            
            return self._config
            
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_yaml_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML files."""
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "configs")
        
        config_dir = Path(config_path)
        if not config_dir.exists():
            # Config directory doesn't exist, skip YAML loading
            return
        
        # Load base configuration
        base_config_file = config_dir / "base.yaml"
        if base_config_file.exists():
            with open(base_config_file, "r") as f:
                base_config = yaml.safe_load(f) or {}
                self._config_data.update(base_config)
        
        # Load mode-specific configuration
        app_mode = os.getenv("APP_MODE", "development")
        mode_config_file = config_dir / f"{app_mode}.yaml"
        if mode_config_file.exists():
            with open(mode_config_file, "r") as f:
                mode_config = yaml.safe_load(f) or {}
                self._merge_config(self._config_data, mode_config)
    
    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            # Core settings
            "APP_MODE": "app_mode",
            "LOG_LEVEL": "log_level",
            "PORT": ("port", int),
            "SECRET_KEY": "secret_key",
            
            # Safety settings
            "MAX_COST_BUDGET": ("safety.max_cost_budget", float),
            "RATE_LIMIT_PER_MINUTE": ("safety.rate_limit_per_minute", int),
            "CIRCUIT_BREAKER_THRESHOLD": ("safety.circuit_breaker_threshold", float),
            "CIRCUIT_BREAKER_WINDOW": ("safety.circuit_breaker_window", int),
            "CIRCUIT_BREAKER_MIN_VOLUME": ("safety.circuit_breaker_min_volume", int),
            "BURST_CAPACITY": ("safety.burst_capacity", int),
            "REFILL_RATE": ("safety.refill_rate", float),
            
            # Database settings
            "REDIS_URL": "database.redis_url",
            "NEO4J_URI": "database.neo4j_uri",
            "NEO4J_USER": "database.neo4j_user",
            "NEO4J_PASSWORD": "database.neo4j_password",
            "POSTGRES_URL": "database.postgres_url",
            "QDRANT_URL": "database.qdrant_url",
            
            # Model settings
            "MODEL_PROVIDER": "models.provider",
            "OLLAMA_HOST": "models.ollama_host",
            "OPENAI_API_KEY": "models.openai_api_key",
            "ANTHROPIC_API_KEY": "models.anthropic_api_key",
            "HUGGINGFACE_TOKEN": "models.huggingface_token",
            
            # Monitoring settings
            "PROMETHEUS_ENABLED": ("monitoring.prometheus_enabled", bool),
            "METRICS_PORT": ("monitoring.metrics_port", int),
            "ENABLE_PROFILING": ("monitoring.enable_profiling", bool),
            "ENABLE_DEBUG": ("monitoring.enable_debug", bool),
            
            # Path settings
            "DATA_PATH": "paths.data_path",
            "LOG_PATH": "paths.log_path",
            "CONFIG_PATH": "paths.config_path",
            
            # CORS settings
            "CORS_ORIGINS": ("cors_origins", "list"),
        }
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_key, tuple):
                    config_path, value_type = config_key
                    env_value = self._convert_env_value(env_value, value_type)
                else:
                    config_path = config_key
                
                self._set_nested_config(self._config_data, config_path, env_value)
    
    def _convert_env_value(self, value: str, value_type: Any) -> Any:
        """Convert environment variable string to appropriate type."""
        if value_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        elif value_type == "list":
            return [item.strip() for item in value.split(",") if item.strip()]
        else:
            return value
    
    def _set_nested_config(self, config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = key_path.split(".")
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _validate_critical_config(self) -> None:
        """Validate that critical configuration keys are present and valid."""
        if not self._config:
            raise ConfigurationError("Configuration not loaded")
        
        critical_checks = [
            (self._config.app_mode, "app_mode must be specified"),
            (self._config.secret_key != "change-me-in-production" or 
             self._config.app_mode.value == "development", 
             "secret_key must be changed in production"),
        ]
        
        # Mode-specific validation
        if self._config.app_mode.value == "production":
            critical_checks.extend([
                (self._config.models.provider == "cloud" and 
                 (self._config.models.openai_api_key or self._config.models.anthropic_api_key),
                 "Production mode with cloud provider requires API keys"),
                (self._config.safety.max_cost_budget > 0,
                 "Production mode should have a positive cost budget"),
            ])
        
        elif self._config.app_mode.value == "development":
            critical_checks.extend([
                (self._config.safety.max_cost_budget == 0.0,
                 "Development mode must have zero cost budget"),
                (self._config.models.provider == "ollama",
                 "Development mode should use Ollama provider"),
            ])
        
        for condition, error_message in critical_checks:
            if not condition:
                raise ConfigurationError(error_message)
    
    @property
    def config(self) -> Optional[ConfigSchema]:
        """Get the loaded configuration."""
        return self._config
    
    def reload_config(self, config_path: Optional[str] = None) -> ConfigSchema:
        """Reload configuration from files and environment."""
        return self.load_config(config_path)


# Global configuration loader instance
_config_loader = ConfigLoader()

# Load configuration on module import
try:
    settings = _config_loader.load_config()
except ConfigurationError as e:
    print(f"Critical configuration error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading configuration: {e}", file=sys.stderr)
    sys.exit(1)


def reload_settings(config_path: Optional[str] = None) -> ConfigSchema:
    """Reload global settings configuration."""
    global settings
    settings = _config_loader.reload_config(config_path)
    return settings