"""Tests for configuration loader."""

import os
from unittest.mock import mock_open, patch

import pytest

from src.config.loader import ConfigLoader, ConfigurationError
from src.config.schemas import AppMode, ConfigSchema


class TestConfigLoader:
    """Test configuration loader functionality."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        loader = ConfigLoader()

        with patch.dict(os.environ, {"APP_MODE": "development"}, clear=True):
            config = loader.load_config()

        assert isinstance(config, ConfigSchema)
        assert config.app_mode == AppMode.DEVELOPMENT
        assert config.safety.max_cost_budget == 0.0

    def test_load_config_with_env_vars(self):
        """Test loading configuration with environment variables."""
        loader = ConfigLoader()

        env_vars = {
            "APP_MODE": "development",
            "LOG_LEVEL": "DEBUG",
            "PORT": "9000",
            "MAX_COST_BUDGET": "0.0",
            "REDIS_URL": "redis://test:6379/0",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = loader.load_config()

        assert config.app_mode == AppMode.DEVELOPMENT
        assert config.log_level == "DEBUG"
        assert config.port == 9000
        assert config.safety.max_cost_budget == 0.0
        assert config.database.redis_url == "redis://test:6379/0"

    def test_load_config_with_yaml_file(self):
        """Test loading configuration from YAML file."""
        loader = ConfigLoader()

        yaml_content = """
app_mode: development
log_level: INFO
safety:
  max_cost_budget: 0.0
  rate_limit_per_minute: 30
"""

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch.dict(os.environ, {"APP_MODE": "development"}, clear=True),
        ):
            config = loader.load_config("test_configs")

        assert config.app_mode == AppMode.DEVELOPMENT
        assert config.log_level == "INFO"
        assert config.safety.max_cost_budget == 0.0
        assert config.safety.rate_limit_per_minute == 30

    def test_env_overrides_yaml(self):
        """Test that environment variables override YAML configuration."""
        loader = ConfigLoader()

        yaml_content = """
app_mode: production
log_level: INFO
port: 8080
"""

        env_vars = {"APP_MODE": "development", "LOG_LEVEL": "DEBUG", "PORT": "9000"}

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch.dict(os.environ, env_vars, clear=True),
        ):
            config = loader.load_config("test_configs")

        # Environment variables should override YAML
        assert config.app_mode == AppMode.DEVELOPMENT
        assert config.log_level == "DEBUG"
        assert config.port == 9000

    def test_validation_error_invalid_mode(self):
        """Test validation error with invalid app mode."""
        loader = ConfigLoader()

        with patch.dict(os.environ, {"APP_MODE": "invalid_mode"}, clear=True):
            with pytest.raises(ConfigurationError, match="Configuration validation failed"):
                loader.load_config()

    def test_critical_config_validation_development(self):
        """Test critical configuration validation for development mode."""
        loader = ConfigLoader()

        # Development mode with non-zero budget should fail
        env_vars = {"APP_MODE": "development", "MAX_COST_BUDGET": "100.0"}

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(
                ConfigurationError, match="Development mode must have max_cost_budget=0.0"
            ):
                loader.load_config()

    def test_critical_config_validation_production(self):
        """Test critical configuration validation for production mode."""
        loader = ConfigLoader()

        # Production mode should allow non-zero budget
        env_vars = {
            "APP_MODE": "production",
            "MAX_COST_BUDGET": "100.0",
            "MODEL_PROVIDER": "cloud",
            "OPENAI_API_KEY": "test_key",
            "SECRET_KEY": "production_secret_key_123456",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = loader.load_config()

        assert config.app_mode == AppMode.PRODUCTION
        assert config.safety.max_cost_budget == 100.0
        assert config.models.openai_api_key == "test_key"

    def test_nested_config_setting(self):
        """Test setting nested configuration values."""
        loader = ConfigLoader()
        loader._config_data = {}

        loader._set_nested_config(loader._config_data, "safety.max_cost_budget", 50.0)
        loader._set_nested_config(loader._config_data, "database.redis_url", "redis://test")

        assert loader._config_data["safety"]["max_cost_budget"] == 50.0
        assert loader._config_data["database"]["redis_url"] == "redis://test"

    def test_config_merge(self):
        """Test configuration dictionary merging."""
        loader = ConfigLoader()

        base = {"app_mode": "development", "safety": {"max_cost_budget": 0.0, "rate_limit": 30}}

        override = {
            "app_mode": "production",
            "safety": {"max_cost_budget": 100.0},
            "new_setting": "value",
        }

        loader._merge_config(base, override)

        assert base["app_mode"] == "production"
        assert base["safety"]["max_cost_budget"] == 100.0
        assert base["safety"]["rate_limit"] == 30  # Preserved
        assert base["new_setting"] == "value"

    def test_env_value_conversion(self):
        """Test environment variable type conversion."""
        loader = ConfigLoader()

        assert loader._convert_env_value("true", bool) is True
        assert loader._convert_env_value("false", bool) is False
        assert loader._convert_env_value("123", int) == 123
        assert loader._convert_env_value("45.67", float) == 45.67
        assert loader._convert_env_value("a,b,c", "list") == ["a", "b", "c"]
        assert loader._convert_env_value("test", str) == "test"
