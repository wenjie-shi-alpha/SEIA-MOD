"""Unified API client for remote LLMs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

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
        headers = self._build_headers()
        messages = self._build_messages(prompt, payload)
        try:
            data = self._dispatch_request(headers, messages)
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", "") if hasattr(exc, "response") else ""
            raise RuntimeError(f"LLM API call failed: {detail}") from exc
        text = self._extract_text(data)
        parsed = self._parse_json(text)
        return {"text": text, "json": parsed}

    def _dispatch_request(self, headers: Dict[str, str], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        endpoint = self.config.endpoint.rstrip("/")
        if endpoint.endswith("responses"):
            body = self._responses_body(messages)
            response = self._post(endpoint, headers, body)
            response.raise_for_status()
            return response.json()
        body = {"model": self.config.model, "messages": messages, "temperature": 0.2}
        response = self._post(endpoint, headers, body, raise_for_status=False)
        if response.status_code == 400:
            # Fallback to responses API for models that no longer support chat completions.
            fallback_endpoint = self._responses_endpoint(endpoint)
            body = self._responses_body(messages)
            fallback = self._post(fallback_endpoint, headers, body)
            fallback.raise_for_status()
            return fallback.json()
        response.raise_for_status()
        return response.json()

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, prompt: str, payload: Dict[str, Any] | None) -> List[Dict[str, Any]]:
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
        return messages

    def _responses_body(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "input": [
                {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "input_text",
                            "text": str(message.get("content", "")),
                        }
                    ],
                }
                for message in messages
            ],
        }

    def _post(
        self,
        endpoint: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        raise_for_status: bool = True,
    ) -> requests.Response:
        response = self.session.post(endpoint, headers=headers, json=body, timeout=self.config.timeout)
        if raise_for_status:
            response.raise_for_status()
        return response

    @staticmethod
    def _responses_endpoint(endpoint: str) -> str:
        base = endpoint.rstrip("/")
        if base.endswith("chat/completions"):
            prefix = base[: -len("chat/completions")].rstrip("/")
            return f"{prefix}/responses"
        return base

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
