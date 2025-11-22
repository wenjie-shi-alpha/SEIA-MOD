"""High-level Mesa model scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .agents.hybrid_agent import HybridAgent, HybridAgentState


@dataclass
class SimulationConfig:
    tick_duration_minutes: int = 15
    llm_model: str = "qwen-plus"


class HybridSimulationModel:
    """Coordinates physics + cognition steps for each tick."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.agents: List[HybridAgent] = []
        self.tick = 0

    def load_subgraph(self, subgraph: Dict[str, Any]) -> None:
        """Initialize agents from mapper payload."""
        self.agents = [
            HybridAgent(
                HybridAgentState(
                    agent_id=node.get("_key", f"agent_{idx}"),
                    capacity=node.get("capacity", 100.0),
                    inventory=node.get("inventory", 100.0),
                    vulnerability=node.get("vulnerability", {}),
                )
            )
            for idx, node in enumerate(subgraph.get("nodes", []))
        ]

    def step(self, meteo_context: Dict[str, Any], llm_client: Any) -> None:
        """Execute one hybrid simulation tick."""
        self.tick += 1
        for agent in self.agents:
            agent.physics_step(meteo_context)
            agent.cognition_step(llm_client)


__all__ = ["HybridSimulationModel", "SimulationConfig"]
