"""Mesa-compatible hybrid agent implementation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VulnerabilityCurve:
    """Represents a parametric vulnerability curve."""

    curve_type: str = "sigmoid"
    params: Dict[str, float] = field(default_factory=dict)

    def evaluate(self, hazard_index: float) -> float:
        """Return a normalized damage ratio given a hazard index."""
        if self.curve_type == "linear":
            slope = self.params.get("slope", 1.0)
            intercept = self.params.get("intercept", 0.0)
            return max(0.0, min(1.0, slope * hazard_index + intercept))
        # Default sigmoid
        threshold = self.params.get("threshold", 0.5)
        sharpness = self.params.get("sharpness", 10.0)
        offset = hazard_index - threshold
        return 1.0 / (1.0 + math.exp(-sharpness * offset))


@dataclass
class HybridAgentState:
    """State container synchronized with the graph loader."""

    agent_id: str
    capacity: float
    inventory: float
    vulnerability: Dict[str, Any] = field(default_factory=dict)
    production_rate: float = 1.0
    demand_rate: float = 1.0
    damage_rate: float = 0.0
    last_action: Optional[str] = None
    baseline_capacity: float = 0.0
    baseline_inventory: float = 0.0


class HybridAgent:
    """Agent exposing physics + cognition hooks."""

    def __init__(self, state: HybridAgentState) -> None:
        self.state = state
        vuln = state.vulnerability or {}
        self.curve = VulnerabilityCurve(
            curve_type=vuln.get("curve_type", "sigmoid"),
            params=vuln.get("params", {}),
        )

    def physics_step(self, meteo_context: Dict[str, Any]) -> None:
        """Apply physical rules (damage curves + conservation)."""
        hazard = float(meteo_context.get("hazard_index", 0.0))
        self.state.damage_rate = self.curve.evaluate(hazard)
        self.state.capacity = max(0.0, self.state.capacity * (1.0 - self.state.damage_rate))
        demand_shock = float(meteo_context.get("demand_multiplier", 1.0))
        production = self.state.production_rate * (1.0 - self.state.damage_rate)
        self.state.inventory = max(
            0.0,
            self.state.inventory + production - self.state.demand_rate * demand_shock,
        )

    def should_trigger_cognition(self) -> bool:
        """Return True when the agent must call an external policy model."""
        capacity_floor = self.state.vulnerability.get("capacity_floor", 0.4)
        inventory_floor = self.state.vulnerability.get("inventory_floor", 0.3)
        capacity_base = self.state.baseline_capacity or self.state.capacity or 1.0
        inventory_base = self.state.baseline_inventory or self.state.inventory or 1.0
        capacity_ratio = self.state.capacity / capacity_base
        inventory_ratio = self.state.inventory / inventory_base
        return capacity_ratio < capacity_floor or inventory_ratio < inventory_floor

    def cognition_step(self, llm_client: Any) -> None:
        """Call external LLM for decision making when triggers fire."""
        if not self.should_trigger_cognition():
            return
        prompt = (
            "You control an industrial agent under climate stress. "
            f"Capacity={self.state.capacity:.2f}, Inventory={self.state.inventory:.2f}. "
            "Return JSON with an 'action' field."
        )
        response = llm_client.generate(prompt, payload={"agent_id": self.state.agent_id})
        action = response.get("json", {}).get("action") or "hold"
        self.state.last_action = action
        if action == "panic_buy":
            self.state.inventory += self.state.demand_rate
        elif action == "reduce_output":
            self.state.production_rate *= 0.8


__all__ = ["HybridAgent", "HybridAgentState", "VulnerabilityCurve"]
