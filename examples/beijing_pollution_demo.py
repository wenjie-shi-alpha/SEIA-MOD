"""Run a Beijing heavy air-pollution scenario end-to-end."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import geopandas as gpd
import h3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognition.llm_client import LLMClient, LLMClientConfig
from src.server.socket_server import encode_agent_frame
from src.simulation.meteo_cognition import MeteoCognitionConfig, MeteoCognitionPipeline
from src.simulation.mapper import DynamicMapper, GraphBackendProtocol, MapperConfig
from src.simulation.model import HybridSimulationModel, SimulationConfig


DEFAULT_H3 = ["8831aa18bdfffff", "8831aa723dfffff", "883181a0c7fffff"]
ACTIVE_H3_COVERAGE: List[str] = list(DEFAULT_H3)
COARSE_EVENT_INPUT = {
    "avg_pm25": 260,
    "max_pm25": 320,
    "pm10": 420,
    "wind_speed": 1.3,
    "duration_hours": 42,
}
VECTOR_PARQUET = Path("data/vectorized/beijing_assets.parquet")
L4_FRAME_PATH = Path("data/outputs/beijing_agents.frame")
MAX_SIM_NODES = 4


def load_env(path: str = ".env") -> None:
    """Populate os.environ with values from the local .env file if present."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_llm_client() -> LLMClient:
    """Instantiate an LLM client using OPENAI_* environment variables."""
    endpoint = (
        os.environ.get("OPENAI_ENDPOINT")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1/chat/completions"
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_MODEL must be set in the environment or .env file.")
    return LLMClient(LLMClientConfig(endpoint=endpoint, api_key=api_key, model=model))


class BeijingDemoBackend(GraphBackendProtocol):
    """Fallback in-memory graph backend for the scenario."""

    def __init__(self, h3_cells: Sequence[str] | None = None) -> None:
        cells = list(h3_cells) if h3_cells else list(DEFAULT_H3)
        if len(cells) < 3:
            cells = list(DEFAULT_H3)
        self.assets: Dict[str, Dict[str, Any]] = {
            "BJ_LOGISTICS": {
                "_key": "BJ_LOGISTICS",
                "name": "Beijing Eastern Logistics Hub",
                "district": "Tongzhou",
                "sector": "logistics",
                "capacity": 180.0,
                "inventory": 120.0,
                "production_rate": 6.0,
                "demand_rate": 5.0,
                "latitude": 39.9096,
                "longitude": 116.6781,
                "h3_index": cells[1],
            },
            "BJ_CONSUMER": {
                "_key": "BJ_CONSUMER",
                "name": "Capital Consumer Electronics Plant",
                "district": "Shunyi",
                "sector": "electronics",
                "capacity": 220.0,
                "inventory": 80.0,
                "production_rate": 8.0,
                "demand_rate": 9.5,
                "latitude": 40.128,
                "longitude": 116.648,
                "h3_index": cells[0],
            },
            "BJ_ENERGY": {
                "_key": "BJ_ENERGY",
                "name": "South Beijing Gas Peaker",
                "district": "Daxing",
                "sector": "energy",
                "capacity": 260.0,
                "inventory": 60.0,
                "production_rate": 10.0,
                "demand_rate": 4.0,
                "latitude": 39.5095,
                "longitude": 116.4101,
                "h3_index": cells[2],
            },
        }
        self.edges: List[Dict[str, Any]] = [
            {"_from": "Assets/BJ_ENERGY", "_to": "Assets/BJ_CONSUMER", "type": "energy_feed"},
            {"_from": "Assets/BJ_LOGISTICS", "_to": "Assets/BJ_CONSUMER", "type": "supply"},
        ]
        self.vulnerability_curves: Dict[str, Dict[str, Any]] = {
            "BJ_LOGISTICS": {
                "curve_type": "sigmoid",
                "params": {"threshold": 0.35, "sharpness": 7.0},
                "capacity_floor": 0.75,
                "inventory_floor": 0.6,
            },
            "BJ_CONSUMER": {
                "curve_type": "sigmoid",
                "params": {"threshold": 0.4, "sharpness": 9.0},
                "capacity_floor": 0.6,
                "inventory_floor": 0.5,
            },
            "BJ_ENERGY": {
                "curve_type": "sigmoid",
                "params": {"threshold": 0.3, "sharpness": 6.0},
                "capacity_floor": 0.7,
                "inventory_floor": 0.65,
            },
        }

    def assets_by_h3(self, h3_ids: Sequence[str]) -> List[Dict[str, Any]]:
        lookup = set(h3_ids)
        return [asset.copy() for asset in self.assets.values() if asset["h3_index"] in lookup]

    def vulnerability_context(self, asset_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        return {key: self.vulnerability_curves.get(key, {}) for key in asset_keys}

    def subgraph_from_assets(self, asset_keys: Sequence[str], hops: int) -> Dict[str, Any]:
        keys = set(asset_keys)
        nodes = [self.assets[key].copy() for key in keys if key in self.assets]
        edges = [edge for edge in self.edges if self._edge_impacts(edge, keys)]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _edge_impacts(edge: Dict[str, Any], keys: Iterable[str]) -> bool:
        from_key = edge.get("_from", "").split("/")[-1]
        to_key = edge.get("_to", "").split("/")[-1]
        keys_set = set(keys)
        return from_key in keys_set or to_key in keys_set


class GeoParquetDemoBackend(GraphBackendProtocol):
    """Graph backend that materializes nodes directly from GeoParquet assets."""

    def __init__(self, gdf: gpd.GeoDataFrame, resolution: int = 8) -> None:
        if gdf.empty:
            raise ValueError("GeoParquet dataframe is empty; cannot build backend.")
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.vulnerabilities: Dict[str, Dict[str, Any]] = {}
        frame = gdf.reset_index(drop=True)
        if "latitude" not in frame.columns:
            frame["latitude"] = frame.geometry.y
            frame["longitude"] = frame.geometry.x
        for idx, row in frame.iterrows():
            key = self._make_key(row.get("name"), idx)
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
            h3_index = h3.latlng_to_cell(lat, lon, resolution)
            pop = float(row.get("pop_max") or row.get("pop_min") or 100000)
            capacity = max(70.0, pop / 1200.0)
            inventory = capacity * 0.55
            production_rate = capacity / 35.0
            demand_rate = capacity / 30.0
            sector = self._sector_from_adm(row.get("adm1name"))
            doc = {
                "_key": key,
                "name": row.get("name") or key,
                "district": row.get("adm1name") or "Beijing",
                "sector": sector,
                "capacity": capacity,
                "inventory": inventory,
                "production_rate": production_rate,
                "demand_rate": demand_rate,
                "latitude": lat,
                "longitude": lon,
                "h3_index": h3_index,
            }
            self.assets[key] = doc
            self.vulnerabilities[key] = self._build_vulnerability(pop)
        self._build_edges()

    def assets_by_h3(self, h3_ids: Sequence[str]) -> List[Dict[str, Any]]:
        lookup = set(h3_ids)
        return [asset.copy() for asset in self.assets.values() if asset["h3_index"] in lookup]

    def vulnerability_context(self, asset_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        return {key: self.vulnerabilities.get(key, {}) for key in asset_keys}

    def subgraph_from_assets(self, asset_keys: Sequence[str], hops: int) -> Dict[str, Any]:
        keys = set(asset_keys)
        nodes = [self.assets[key].copy() for key in keys if key in self.assets]
        edges = [edge for edge in self.edges if self._edge_impacts(edge, keys)]
        return {"nodes": nodes, "edges": edges}

    def _build_edges(self) -> None:
        central_candidates = [key for key in self.assets if key.startswith("BEIJING")]
        central = central_candidates[0] if central_candidates else next(iter(self.assets))
        for key in self.assets:
            if key == central:
                continue
            self.edges.append(
                {
                    "_from": f"Assets/{key}",
                    "_to": f"Assets/{central}",
                    "type": "logistics",
                }
            )

    @staticmethod
    def _make_key(name: str | None, idx: int) -> str:
        base = (name or "asset").upper().replace(" ", "_")
        return f"{base}_{idx}"

    @staticmethod
    def _sector_from_adm(adm_name: str | None) -> str:
        if not adm_name:
            return "manufacturing"
        if adm_name in {"Tianjin", "Beijing"}:
            return "electronics"
        if adm_name in {"Hebei"}:
            return "logistics"
        return "manufacturing"

    @staticmethod
    def _build_vulnerability(pop: float) -> Dict[str, Any]:
        threshold = 0.28 + min(0.4, pop / 2.5e7)
        sharpness = 5.0 + min(4.0, pop / 5e6)
        return {
            "curve_type": "sigmoid",
            "params": {"threshold": threshold, "sharpness": sharpness},
            "capacity_floor": 0.6,
            "inventory_floor": 0.55,
        }

    @staticmethod
    def _edge_impacts(edge: Dict[str, Any], keys: Iterable[str]) -> bool:
        from_key = edge.get("_from", "").split("/")[-1]
        to_key = edge.get("_to", "").split("/")[-1]
        keys_set = set(keys)
        return from_key in keys_set or to_key in keys_set


def smog_downscaler(coarse_tensor: Dict[str, Any]) -> Dict[str, Any]:
    """Simple heuristic downscaler for the demo scenario."""
    upscale_factor = 1.15
    hi_res = dict(coarse_tensor)
    hi_res["max_pm25"] = coarse_tensor["max_pm25"] * upscale_factor
    hi_res["avg_pm25"] = coarse_tensor["avg_pm25"] * upscale_factor
    hi_res["wind_speed"] = max(0.4, coarse_tensor["wind_speed"] - 0.3)
    return hi_res


def smog_segmenter(hi_res_tensor: Dict[str, Any]) -> Dict[str, Any]:
    """Return segmentation stats mimicking a persistent smog plume."""
    max_pm25 = hi_res_tensor["max_pm25"]
    aqi = min(500.0, max_pm25 * 1.5)
    stats = {
        "max_pm25": max_pm25,
        "avg_pm25": hi_res_tensor["avg_pm25"],
        "aqi": aqi,
        "duration_hours": COARSE_EVENT_INPUT["duration_hours"],
        "population_exposed": 2.2e7,
        "max_intensity": min(1.0, aqi / 500.0),
    }
    return {"stats": stats, "h3_coverage": ACTIVE_H3_COVERAGE}


def build_pipeline(llm_client: LLMClient) -> MeteoCognitionPipeline:
    prompt = (
        "你是一名北京空气重污染事件的分析师。请严格返回 JSON 对象，"
        "包含字段 type(固定为 air_pollution)、h3_coverage(直接使用输入的列表)、"
        "intensity_index(0-1 浮点)、action(一句建议) 与 metadata(包含 aqilevel、duration_hours、population_exposed)。"
    )
    return MeteoCognitionPipeline(
        MeteoCognitionConfig(
            downscaler=smog_downscaler,
            segmenter=smog_segmenter,
            llm_client=llm_client,
            llm_prompt=prompt,
        )
    )


def run_simulation() -> None:
    load_env()
    llm_client = build_llm_client()
    gdf = load_vectorized_assets(VECTOR_PARQUET)
    h3_cells = compute_h3_coverage(gdf) if gdf is not None else list(DEFAULT_H3)
    set_active_h3(h3_cells)
    backend: GraphBackendProtocol = GeoParquetDemoBackend(gdf) if gdf is not None else BeijingDemoBackend(h3_cells)
    pipeline = build_pipeline(llm_client)
    event = pipeline.run(COARSE_EVENT_INPUT)

    mapper = DynamicMapper(
        MapperConfig(
            arangodb_url="http://localhost:8529",
            database="demo",
            username="root",
            password="",
            neighborhood_hops=1,
        ),
        backend=backend,
    )
    subgraph = mapper.run({"h3_coverage": event.h3_coverage})
    nodes = subgraph.get("nodes", [])
    if not nodes:
        raise RuntimeError("No Beijing assets were matched for the generated event coverage.")

    simulation = HybridSimulationModel(SimulationConfig())
    simulation.load_subgraph(subgraph)
    meteo_contexts = build_meteo_timeline(event.intensity_index)
    history = simulation.run(len(meteo_contexts), meteo_contexts, llm_client)
    frame_path = export_l4_frame(nodes, simulation.agents)

    print_event_summary(event)
    print_assets(nodes)
    print_metrics(history)
    print(f"L4 binary frame written to {frame_path}")


def load_vectorized_assets(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    gdf = gpd.read_parquet(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    if "latitude" not in gdf.columns:
        gdf["latitude"] = gdf.geometry.y
        gdf["longitude"] = gdf.geometry.x
    if MAX_SIM_NODES and len(gdf) > MAX_SIM_NODES and "pop_max" in gdf.columns:
        gdf = gdf.nlargest(MAX_SIM_NODES, "pop_max")
    return gdf


def compute_h3_coverage(gdf: gpd.GeoDataFrame, resolution: int = 8) -> List[str]:
    cells = {
        h3.latlng_to_cell(float(row.latitude), float(row.longitude), resolution)
        for _, row in gdf.iterrows()
    }
    return sorted(cells)


def set_active_h3(cells: Sequence[str]) -> None:
    global ACTIVE_H3_COVERAGE
    ACTIVE_H3_COVERAGE = list(cells) if cells else list(DEFAULT_H3)


def export_l4_frame(nodes: List[Dict[str, Any]], agents: List[Any]) -> Path:
    """Serialize the latest agent states into the SEIA binary frame."""
    positions = []
    damage = []
    ids = []
    for agent, node in zip(agents, nodes):
        lat = float(node.get("latitude", 0.0))
        lon = float(node.get("longitude", 0.0))
        positions.append([lon, lat])
        damage.append(agent.state.damage_rate)
        ids.append(agent.state.agent_id)
    payload = encode_agent_frame(
        positions=np.array(positions, dtype="<f4"),
        damage_rates=np.array(damage, dtype="<f4"),
        agent_ids=ids,
        timestamp=time.time(),
        components=3,
    )
    L4_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    L4_FRAME_PATH.write_bytes(payload)
    return L4_FRAME_PATH


def build_meteo_timeline(base_intensity: float) -> List[Dict[str, Any]]:
    """Construct a three-step progression for the smog episode."""
    base = min(1.0, max(0.1, base_intensity))
    return [
        {"hazard_index": base * 0.9, "demand_multiplier": 1.05},
        {"hazard_index": base, "demand_multiplier": 1.2},
        {"hazard_index": base * 0.7, "demand_multiplier": 1.1},
    ]


def print_event_summary(event) -> None:
    print("=== LLM-generated air pollution event ===")
    payload = {
        "type": event.type,
        "h3_coverage": event.h3_coverage,
        "intensity_index": event.intensity_index,
        "metadata": event.metadata,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_assets(nodes: List[Dict[str, Any]]) -> None:
    print("\n=== Impacted Beijing assets ===")
    for node in nodes:
        print(
            f"- {node.get('name')} ({node.get('sector')}, {node.get('district')}) | "
            f"capacity={node.get('capacity'):.1f} inventory={node.get('inventory'):.1f}"
        )


def print_metrics(history: List[Dict[str, Any]]) -> None:
    print("\n=== Simulation macro metrics ===")
    for idx, tick in enumerate(history, start=1):
        print(
            f"Tick {idx}: total_capacity={tick['total_capacity']:.2f} "
            f"total_inventory={tick['total_inventory']:.2f} avg_damage={tick['avg_damage']:.3f}"
        )


if __name__ == "__main__":
    run_simulation()
