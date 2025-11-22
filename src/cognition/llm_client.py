"""Unified API client for remote LLMs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class LLMClientConfig:
    endpoint: str
    api_key: str
    model: str


class LLMClient:
    """Thin wrapper to keep LLM calls consistent across modules."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def generate(self, prompt: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Placeholder POST request."""
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        body = {"model": self.config.model, "prompt": prompt, "payload": payload or {}}
        _ = (headers, body)  # Avoid unused variable warnings before implementation
        # TODO: wire up actual API call via requests.post
        return {"text": "", "json": {}}


__all__ = ["LLMClient", "LLMClientConfig"]
