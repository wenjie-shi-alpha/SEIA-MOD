#!/usr/bin/env python
"""Simulate a Beijing heatwave event against the JJJ power exposure layer."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd

from src.simulation.agents.hybrid_agent import VulnerabilityCurve

ASSET_PATH = Path("data/vectorized/jjj_power_assets.parquet")
VULN_PATH = Path("resources/vulnerability/jjj_power_vulnerabilities.json")


@dataclass
class HeatwaveEvent:
    name: str
    peak_hazard_index: float
    center_lat: float
    center_lon: float
    sigma_km: float = 120.0
    floor_hazard_index: float = 0.25

    def hazard_at(self, lat: float, lon: float) -> float:
        dist = haversine_km(self.center_lat, self.center_lon, lat, lon)
        attenuation = math.exp(-((dist ** 2) / (2 * self.sigma_km**2)))
        value = self.floor_hazard_index + (self.peak_hazard_index - self.floor_hazard_index) * attenuation
        return max(0.0, min(1.0, value))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_assets() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(ASSET_PATH)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf


def load_vulnerabilities() -> Dict[str, VulnerabilityCurve]:
    data = json.loads(VULN_PATH.read_text())
    curves: Dict[str, VulnerabilityCurve] = {}
    for item in data:
        curves[item["_key"]] = VulnerabilityCurve(
            curve_type=item.get("curve_type", "sigmoid"),
            params=item.get("params", {}),
        )
    return curves


def evaluate_damage(gdf: gpd.GeoDataFrame, event: HeatwaveEvent) -> Tuple[pd.DataFrame, Dict[str, float]]:
    vulnerabilities = load_vulnerabilities()
    records: List[Dict[str, object]] = []

    for _, row in gdf.iterrows():
        hazard = event.hazard_at(row["lat"], row["lon"])
        vuln_curve = vulnerabilities.get(row["vulnerability_type"])
        if vuln_curve is None:
            damage = 0.0
        else:
            damage = vuln_curve.evaluate(hazard)
        record = {
            "_key": row["_key"],
            "asset_name": row["asset_name"],
            "asset_type": row["asset_type"],
            "province": row["province"],
            "capacity_mw": row["capacity_mw"],
            "hazard_index": hazard,
            "damage_rate": damage,
            "vulnerability_type": row["vulnerability_type"],
            "grid_node_key": row.get("grid_node_key"),
            "supplier_ids": row.get("supplier_ids", ""),
        }
        records.append(record)

    impact_df = pd.DataFrame(records)
    summary = {
        "total_assets": len(impact_df),
        "affected_assets": int((impact_df["damage_rate"] > 0.2).sum()),
        "weighted_damage": float(
            (impact_df["capacity_mw"] * impact_df["damage_rate"]).sum()
            / impact_df["capacity_mw"].sum()
        ),
    }
    return impact_df, summary


def grid_node_analysis(impact_df: pd.DataFrame) -> pd.DataFrame:
    plants = impact_df[impact_df["asset_type"] == "power_plant"].copy()
    grids = impact_df[impact_df["asset_type"] == "grid_node"].copy()
    capacity_map = plants.set_index("_key")["capacity_mw"].to_dict()
    damage_map = plants.set_index("_key")["damage_rate"].to_dict()

    rows = []
    for _, node in grids.iterrows():
        suppliers = [s for s in node["supplier_ids"].split(",") if s]
        if not suppliers:
            continue
        nominal = sum(capacity_map.get(s, 0.0) for s in suppliers)
        surviving = sum(capacity_map.get(s, 0.0) * (1.0 - damage_map.get(s, 0.0)) for s in suppliers)
        supply_loss = 1.0 - (surviving / nominal) if nominal else 0.0
        combined_damage = 1.0 - (1.0 - node["damage_rate"]) * (1.0 - supply_loss)
        rows.append(
            {
                "grid_node": node["asset_name"],
                "province": node["province"],
                "hazard_index": node["hazard_index"],
                "direct_damage": node["damage_rate"],
                "supply_loss": supply_loss,
                "combined_damage": combined_damage,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    gdf = load_assets()
    event = HeatwaveEvent(
        name="2024-07-15 North China Heat Dome",
        peak_hazard_index=0.9,
        center_lat=39.9042,
        center_lon=116.4074,
    )
    impact_df, summary = evaluate_damage(gdf, event)
    grid_summary = grid_node_analysis(impact_df)

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    impact_path = out_dir / "heatwave_impact.parquet"
    impact_df.to_parquet(impact_path, index=False)
    grid_summary.to_parquet(out_dir / "heatwave_grid_summary.parquet", index=False)
    (out_dir / "heatwave_event.json").write_text(
        json.dumps(
            {
                "name": event.name,
                "peak_hazard_index": event.peak_hazard_index,
                "center": [event.center_lat, event.center_lon],
                "sigma_km": event.sigma_km,
            },
            indent=2,
        )
    )

    print("Event:", event.name)
    print("Summary:", summary)
    print("Top impacted assets:")
    print(
        impact_df[impact_df["asset_type"] == "power_plant"]
        .nlargest(5, "damage_rate")
        [["asset_name", "province", "capacity_mw", "hazard_index", "damage_rate", "vulnerability_type"]]
        .to_string(index=False)
    )
    print("\nGrid impact:")
    print(grid_summary.sort_values("combined_damage", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
