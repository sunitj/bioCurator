"""Ollama client with safety integration."""

import httpx
from typing import Dict, Any, Optional

from ..config import settings
from ..logging import get_logger
from ..safety import emit_safety_event, SafetyEventType

logger = get_logger(__name__)


class CloudModelBlockedError(Exception):
    """Raised when cloud model is blocked in development mode."""

    def __init__(self, model: str, mode: str):
        self.model = model
        self.mode = mode
        super().__init__(f"Cloud model '{model}' blocked in {mode} mode")


class OllamaClient:
    """Safety-aware Ollama client."""

    def __init__(self, base_url: str = None):
        """Initialize Ollama client."""
        self.base_url = base_url or settings.models.ollama_host
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def _check_development_mode_guard(self, model: str) -> None:
        """Hard guard against cloud models in development mode."""
        if settings.app_mode.value == "development":
            # In development mode, only allow local models
            cloud_indicators = ["gpt", "claude", "openai", "anthropic"]
            if any(indicator in model.lower() for indicator in cloud_indicators):
                emit_safety_event(
                    event_type=SafetyEventType.CLOUD_MODEL_BLOCKED,
                    component="ollama_client",
                    message=f"Blocked cloud model '{model}' in development mode",
                    metadata={"model": model, "mode": settings.app_mode.value},
                    severity="ERROR"
                )
                raise CloudModelBlockedError(model, settings.app_mode.value)

    async def generate(
        self,
        model: str,
        prompt: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with safety checks."""
        self._check_development_mode_guard(model)

        try:
            response = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(
                "Ollama API error",
                model=model,
                agent_id=agent_id,
                error=str(e)
            )
            raise

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("Failed to list Ollama models", error=str(e))
            raise

    def close(self):
        """Close the client."""
        self._client.close()