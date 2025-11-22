"""Unified API client for remote LLMs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    endpoint: str
    api_key: str
    model: str
    timeout: float = 30.0


class LLMClient:
    """Thin wrapper to keep LLM calls consistent across modules."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def generate(self, prompt: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Call the configured LLM endpoint and normalize the JSON response."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "system",
                "content": "You are a JSON-only assistant. Always reply with strict JSON objects.",
            },
            {"role": "user", "content": prompt},
        ]
        if payload:
            payload_json = json.dumps(payload, default=str)
            messages.append(
                {
                    "role": "user",
                    "content": f"Additional structured context: {payload_json}",
                }
            )
        body = {"model": self.config.model, "messages": messages, "temperature": 0.2}
        try:
            response = self.session.post(
                self.config.endpoint,
                headers=headers,
                json=body,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("LLM API call failed") from exc
        data = response.json()
        text = self._extract_text(data)
        parsed = self._parse_json(text)
        return {"text": text, "json": parsed}

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        """Normalize text content for both Chat Completions and Responses APIs."""
        if "choices" in payload:
            choice = payload["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content or "")
        if "output" in payload:
            parts: list[str] = []
            for item in payload["output"]:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    parts.append(item.get("content", ""))
            if parts:
                return "".join(parts)
        return str(payload.get("text", ""))

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Best-effort JSON parsing with graceful fallback."""
        stripped = (text or "").strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("LLM response is not valid JSON: %s", stripped)
            return {}


__all__ = ["LLMClient", "LLMClientConfig"]
