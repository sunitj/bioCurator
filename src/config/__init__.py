"""Configuration management for BioCurator."""

from .loader import settings
from .schemas import ConfigSchema

__all__ = ["settings", "ConfigSchema"]
