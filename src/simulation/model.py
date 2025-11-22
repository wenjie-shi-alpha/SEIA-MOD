"""High-level Mesa-like hybrid simulation scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .agents.hybrid_agent import HybridAgent, HybridAgentState


@dataclass
class SimulationConfig:
    """Runtime configuration for the hybrid simulation."""

    tick_duration_minutes: int = 15
    llm_model: str = "qwen-plus"
    cognition_enabled: bool = True


class HybridSimulationModel:
    """Coordinates physics + cognition steps for each tick."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.agents: List[HybridAgent] = []
        self.tick = 0
        self.history: List[Dict[str, float]] = []

    def load_subgraph(self, subgraph: Dict[str, Any]) -> None:
        """Initialize agents from mapper payload."""
        nodes = subgraph.get("nodes", [])
        self.agents = []
        for idx, node in enumerate(nodes):
            capacity = float(node.get("capacity", 100.0))
            inventory = float(node.get("inventory", 100.0))
            state = HybridAgentState(
                agent_id=node.get("_key", f"agent_{idx}"),
                capacity=capacity,
                inventory=inventory,
                vulnerability=node.get("vulnerability", {}),
                production_rate=node.get("production_rate", 1.0),
                demand_rate=node.get("demand_rate", 1.0),
                baseline_capacity=capacity,
                baseline_inventory=inventory,
            )
            self.agents.append(HybridAgent(state))
        self.history.clear()

    def step(self, meteo_context: Dict[str, Any], llm_client: Any) -> Dict[str, float]:
        """Execute one hybrid simulation tick and return aggregated metrics."""
        self.tick += 1
        for agent in self.agents:
            agent.physics_step(meteo_context)
            if self.config.cognition_enabled:
                agent.cognition_step(llm_client)
        metrics = self.aggregate_metrics()
        self.history.append(metrics)
        return metrics

    def aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate macro indicators for downstream collectors."""
        if not self.agents:
            return {"total_capacity": 0.0, "total_inventory": 0.0, "avg_damage": 0.0}
        total_capacity = sum(agent.state.capacity for agent in self.agents)
        total_inventory = sum(agent.state.inventory for agent in self.agents)
        avg_damage = sum(agent.state.damage_rate for agent in self.agents) / len(self.agents)
        return {
            "total_capacity": total_capacity,
            "total_inventory": total_inventory,
            "avg_damage": avg_damage,
        }

    def run(self, steps: int, meteo_contexts: Iterable[Dict[str, Any]], llm_client: Any) -> List[Dict[str, float]]:
        """Convenience method to advance multiple ticks."""
        contexts = list(meteo_contexts)
        for idx in range(steps):
            context = contexts[idx] if idx < len(contexts) else contexts[-1]
            self.step(context, llm_client)
        return self.history


__all__ = ["HybridSimulationModel", "SimulationConfig"]
