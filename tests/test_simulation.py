"""Tests for the L3 simulation kernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from src.cognition.llm_client import LLMClient, LLMClientConfig
from src.simulation.meteo_cognition import MeteoCognitionConfig, MeteoCognitionPipeline
from src.simulation.mapper import DynamicMapper, MapperConfig, GraphBackendProtocol
from src.simulation.model import HybridSimulationModel, SimulationConfig


class StubLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(LLMClientConfig(endpoint="", api_key="", model=""))
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompt: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "payload": payload})
        return {
            "json": {
                "type": "typhoon",
                "h3_coverage": ["892830828cbffff"],
                "intensity_index": 0.87,
                "action": "panic_buy",
            }
        }


def test_meteo_cognition_pipeline_emits_event():
    downscaler = lambda tensor: {"hi_res": tensor}  # noqa: E731
    segmenter = lambda tensor: {  # noqa: E731
        "stats": {"max_intensity": 0.9, "mean_intensity": 0.6},
        "h3_coverage": ["892830828cbffff"],
    }
    pipeline = MeteoCognitionPipeline(
        MeteoCognitionConfig(
            downscaler=downscaler,
            segmenter=segmenter,
            llm_client=StubLLMClient(),
        )
    )

    event = pipeline.run({"coarse": "tensor"})
    assert event.type == "typhoon"
    assert event.h3_coverage == ["892830828cbffff"]
    assert event.intensity_index > 0


class FakeBackend(GraphBackendProtocol):
    def __init__(
        self,
        assets: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        vulnerabilities: Dict[str, Dict[str, Any]],
    ) -> None:
        self.assets = assets
        self.edges = edges
        self.vulnerabilities = vulnerabilities

    def assets_by_h3(self, h3_ids):
        return [asset for asset in self.assets.values() if asset["h3_index"] in h3_ids]

    def vulnerability_context(self, asset_keys):
        return {key: self.vulnerabilities.get(key, {}) for key in asset_keys}

    def subgraph_from_assets(self, asset_keys, hops):
        nodes = [self.assets[key] for key in asset_keys]
        return {"nodes": nodes, "edges": self.edges}


def test_dynamic_mapper_builds_subgraph():
    assets = {
        "A": {"_key": "A", "h3_index": "89283", "vulnerability": {}},
        "B": {"_key": "B", "h3_index": "89284", "vulnerability": {}},
    }
    vulnerabilities = {"A": {"curve_type": "linear", "params": {"slope": 0.8}}}
    edges = [{"_from": "Assets/A", "_to": "Assets/B", "type": "logistics"}]
    backend = FakeBackend(assets=assets, edges=edges, vulnerabilities=vulnerabilities)
    mapper = DynamicMapper(
        MapperConfig(
            arangodb_url="http://localhost:8529",
            database="test",
            username="root",
            password="pw",
        ),
        backend=backend,
    )

    subgraph = mapper.run({"h3_coverage": ["89283"]})
    assert len(subgraph["nodes"]) == 1
    node = subgraph["nodes"][0]
    assert node["vulnerability"]["curve_type"] == "linear"
    assert subgraph["edges"] == edges


def test_hybrid_simulation_model_triggers_cognition():
    subgraph = {
        "nodes": [
            {
                "_key": "A",
                "capacity": 100,
                "inventory": 20,
                "vulnerability": {
                    "curve_type": "sigmoid",
                    "threshold": 0.4,
                    "capacity_floor": 0.95,
                    "inventory_floor": 0.95,
                },
                "production_rate": 2,
                "demand_rate": 5,
            }
        ]
    }
    model = HybridSimulationModel(SimulationConfig())
    model.load_subgraph(subgraph)
    llm_client = StubLLMClient()
    metrics = model.step({"hazard_index": 0.8, "demand_multiplier": 2.0}, llm_client)

    assert metrics["avg_damage"] > 0
    assert model.agents[0].state.last_action == "panic_buy"
    assert len(llm_client.calls) == 1
