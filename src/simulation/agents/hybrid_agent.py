"""Mesa-compatible hybrid agent stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HybridAgentState:
    """Minimal state container until Mesa integration is implemented."""

    agent_id: str
    capacity: float
    inventory: float
    vulnerability: Dict[str, Any] = field(default_factory=dict)


class HybridAgent:
    """Placeholder agent exposing physics + cognition hooks."""

    def __init__(self, state: HybridAgentState) -> None:
        self.state = state

    def physics_step(self, meteo_context: Dict[str, Any]) -> None:
        """Apply physical rules (e.g., sigmoid damage curves)."""
        _ = meteo_context

    def cognition_step(self, llm_client: Any) -> None:
        """Call external LLM for decision making when triggers fire."""
        _ = llm_client


__all__ = ["HybridAgent", "HybridAgentState"]
