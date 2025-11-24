#!/usr/bin/env python
"""Build the Jing-Jin-Ji power generation exposure layer."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

try:
    import h3
except ImportError as exc:  # pragma: no cover - h3 is required at runtime
    raise SystemExit("h3 python bindings are required for this script") from exc

DATA_ROOT = Path("data")
EXTERNAL_DIR = DATA_ROOT / "external"
VECTOR_DIR = DATA_ROOT / "vector"
VECTORIZED_DIR = DATA_ROOT / "vectorized"

GPPD_PATH = EXTERNAL_DIR / "global_power_plant_database.csv"
ADMIN1_PATH = (
    EXTERNAL_DIR
    / "ne_10m_admin_1_states_provinces"
    / "ne_10m_admin_1_states_provinces.shp"
)

H3_RESOLUTION = 9
TARGET_PROVINCES = {"Beijing", "Tianjin", "Hebei"}

FUEL_TECH_MAP = {
    "Coal": "thermal_coal",
    "Gas": "thermal_gas",
    "Oil": "thermal_oil",
    "Biomass": "bioenergy",
    "Wind": "wind",
    "Solar": "photovoltaic",
    "Hydro": "hydro",
    "Nuclear": "nuclear",
    "Waste": "waste_to_energy",
    "Petcoke": "thermal_coal",
}

VULNERABILITY_MAPPING = {
    "storm_surge_chp": "vul_chp_storm",
    "urban_heat_chp": "vul_chp_heat",
    "upland_wind": "vul_windstorm",
    "plateau_pv": "vul_pv_heat",
    "plain_load": "vul_generic",
    "load_cluster": "vul_grid_cluster",
}

GRID_METADATA: Dict[str, Dict[str, object]] = {
    "GRID_BEIJING_CORE": {
        "name": "State Grid Beijing Core Load Center",
        "province": "Beijing",
        "latitude": 39.9042,
        "longitude": 116.4074,
        "load_mw": 13000,
        "segment": "urban_load",
    },
    "GRID_TIANJIN_URBAN": {
        "name": "State Grid Tianjin Urban Ring",
        "province": "Tianjin",
        "latitude": 39.122,
        "longitude": 117.200,
        "load_mw": 9500,
        "segment": "urban_load",
    },
    "GRID_TIANJIN_PORT": {
        "name": "Tianjin Port Petrochemical Hub",
        "province": "Tianjin",
        "latitude": 39.0,
        "longitude": 117.714,
        "load_mw": 8000,
        "segment": "industrial_port",
    },
    "GRID_HEBEI_NORTH": {
        "name": "Hebei North Heavy Industry Ring",
        "province": "Hebei",
        "latitude": 40.6,
        "longitude": 118.2,
        "load_mw": 17500,
        "segment": "steel_corridor",
    },
    "GRID_HEBEI_COASTAL": {
        "name": "Caofeidian Coastal Grid",
        "province": "Hebei",
        "latitude": 39.2,
        "longitude": 118.7,
        "load_mw": 12000,
        "segment": "coastal_industrial",
    },
    "GRID_HEBEI_SOUTH": {
        "name": "Hebei Southern Load Pocket",
        "province": "Hebei",
        "latitude": 38.0,
        "longitude": 114.5,
        "load_mw": 16000,
        "segment": "manufacturing_cluster",
    },
}


@dataclass
class LayerArtifacts:
    """Container for build outputs."""

    frame: gpd.GeoDataFrame
    summary: Dict[str, object]


def slugify(value: str) -> str:
    """Generate a stable uppercase slug for asset keys."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token.upper()


def compute_h3(lat: float, lon: float, resolution: int = H3_RESOLUTION) -> str:
    """Convert a WGS84 coordinate to an H3 index."""
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)  # pragma: no cover


def assign_grid_node(province: str, lat: float, lon: float) -> str:
    """Assign each plant to a logical load pocket."""
    if province == "Beijing":
        return "GRID_BEIJING_CORE"
    if province == "Tianjin":
        return "GRID_TIANJIN_PORT" if lon >= 117.5 else "GRID_TIANJIN_URBAN"
    # Hebei sub-pockets
    if lon >= 118.3:
        return "GRID_HEBEI_COASTAL"
    if lat >= 39.8:
        return "GRID_HEBEI_NORTH"
    return "GRID_HEBEI_SOUTH"


def classify_segment(fuel_group: str) -> str:
    """Collapse technology types into thermal vs renewable clusters."""
    if fuel_group in {"thermal_coal", "thermal_gas", "thermal_oil"}:
        return "thermal"
    if fuel_group in {"bioenergy", "waste_to_energy"}:
        return "cogeneration"
    return "renewable"


def resilience_tier(segment: str, criticality: int) -> str:
    """Derive a qualitative resilience tier used by the loader."""
    if criticality >= 5 or (segment == "thermal" and criticality >= 4):
        return "A"
    if criticality >= 3:
        return "B"
    return "C"


def exposure_profile(row: pd.Series) -> str:
    """Tag the hazard context for downstream vulnerability binding."""
    if row["province"] == "Tianjin" and row["segment"] == "thermal":
        return "storm_surge_chp"
    if row["province"] == "Beijing" and row["segment"] == "thermal":
        return "urban_heat_chp"
    if row["fuel_group"] == "wind":
        return "upland_wind"
    if row["fuel_group"] == "photovoltaic":
        return "plateau_pv"
    return "plain_load"


def load_power_plants() -> gpd.GeoDataFrame:
    """Load and spatially filter the GPPD dataset."""
    if not GPPD_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {GPPD_PATH}")
    if not ADMIN1_PATH.exists():
        raise FileNotFoundError(f"Missing province shapefile: {ADMIN1_PATH}")

    gppd = pd.read_csv(GPPD_PATH, low_memory=False)
    china = gppd[gppd["country_long"] == "China"].dropna(subset=["latitude", "longitude"])
    china = china.rename(columns={"name": "plant_name"})
    china_gdf = gpd.GeoDataFrame(
        china,
        geometry=gpd.points_from_xy(china["longitude"], china["latitude"]),
        crs="EPSG:4326",
    )

    admin = gpd.read_file(ADMIN1_PATH)
    provinces = (
        admin[(admin["admin"] == "China") & (admin["name"].isin(TARGET_PROVINCES))]
        .rename(columns={"name": "province"})
        .loc[:, ["province", "geometry"]]
    )

    return gpd.sjoin(china_gdf, provinces, how="inner", predicate="within")


def enrich_plants(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add derived exposure attributes to the filtered plant layer."""
    plants = raw.copy()
    plants["_key"] = plants["gppd_idnr"].astype(str).apply(lambda v: f"PLANT_{slugify(v)}")
    plants["asset_name"] = plants["plant_name"]
    plants["asset_type"] = "power_plant"
    plants["industry"] = "power_generation"
    plants["fuel_group"] = plants["primary_fuel"].map(FUEL_TECH_MAP).fillna("other")
    plants["segment"] = plants["fuel_group"].apply(classify_segment)
    plants["grid_node_key"] = plants.apply(
        lambda row: assign_grid_node(row["province"], row["latitude"], row["longitude"]),
        axis=1,
    )
    percentile = plants["capacity_mw"].rank(pct=True)
    plants["criticality_score"] = (
        (percentile * 4).apply(math.floor).clip(lower=0) + 1
    ).astype(int)
    plants["impact_buffer_km"] = plants["criticality_score"] * 2 + plants["capacity_mw"].clip(
        upper=1500
    ) / 500
    generation_cols = [
        "generation_gwh_2017",
        "generation_gwh_2018",
        "generation_gwh_2019",
    ]
    plants["generation_gwh_mean"] = plants[generation_cols].mean(axis=1, skipna=True).round(2)
    plants["resilience_tier"] = plants.apply(
        lambda row: resilience_tier(row["segment"], row["criticality_score"]), axis=1
    )
    plants["exposure_profile"] = plants.apply(exposure_profile, axis=1)
    plants["vulnerability_type"] = plants["exposure_profile"].map(
        VULNERABILITY_MAPPING
    )
    plants["status"] = "active"
    plants["data_source"] = "WRI Global Power Plant Database v1.3 (CC-BY 4.0)"
    plants["supplier_ids"] = ""  # upstream fuel logistics not modeled at this stage
    plants["h3_index"] = plants.apply(
        lambda row: compute_h3(row["latitude"], row["longitude"]), axis=1
    )

    keep_columns = [
        "_key",
        "asset_name",
        "asset_type",
        "industry",
        "province",
        "primary_fuel",
        "fuel_group",
        "segment",
        "capacity_mw",
        "commissioning_year",
        "owner",
        "status",
        "grid_node_key",
        "criticality_score",
        "impact_buffer_km",
        "generation_gwh_mean",
        "resilience_tier",
        "exposure_profile",
        "vulnerability_type",
        "supplier_ids",
        "data_source",
        "url",
        "h3_index",
        "geometry",
    ]
    plants = plants[keep_columns]
    plants["commissioning_year"] = plants["commissioning_year"].astype("Int64")
    return plants


def build_grid_nodes(plants: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create synthetic load-center nodes so the SupplyChain edges form a DAG."""
    records: List[Dict[str, object]] = []
    for grid_key, meta in GRID_METADATA.items():
        supplier_keys = sorted(plants.loc[plants["grid_node_key"] == grid_key, "_key"].tolist())
        geometry = Point(float(meta["longitude"]), float(meta["latitude"]))
        record = {
            "_key": grid_key,
            "asset_name": meta["name"],
            "asset_type": "grid_node",
            "industry": "power_transmission",
            "province": meta["province"],
            "primary_fuel": "load",
            "fuel_group": meta["segment"],
            "segment": meta["segment"],
            "capacity_mw": meta["load_mw"],
            "commissioning_year": 2020,
            "owner": "State Grid North China",
            "status": "demand_node",
            "grid_node_key": grid_key,
            "criticality_score": 5 if supplier_keys else 3,
            "impact_buffer_km": 35.0,
            "generation_gwh_mean": 0.0,
            "resilience_tier": "A" if supplier_keys else "B",
            "exposure_profile": "load_cluster",
            "vulnerability_type": VULNERABILITY_MAPPING["load_cluster"],
            "supplier_ids": ",".join(supplier_keys),
            "data_source": "Synthetic load center derived from SGCC public planning",
            "url": "",
            "h3_index": compute_h3(meta["latitude"], meta["longitude"]),
            "geometry": geometry,
        }
        records.append(record)
    grid_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    grid_gdf = grid_gdf.reindex(columns=plants.columns, fill_value=pd.NA)
    return grid_gdf


def build_layer() -> LayerArtifacts:
    """Run the end-to-end build and return the GeoDataFrame + summary."""
    raw = load_power_plants()
    plants = enrich_plants(raw)
    grids = build_grid_nodes(plants)
    combined = pd.concat([plants, grids], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    summary = {
        "total_assets": len(combined),
        "plants": len(plants),
        "grid_nodes": len(grids),
        "capacity_mw": round(plants["capacity_mw"].sum(), 2),
        "fuel_mix": plants.groupby("primary_fuel").size().to_dict(),
        "province_counts": plants.groupby("province").size().to_dict(),
    }
    return LayerArtifacts(combined, summary)


def write_outputs(artifacts: LayerArtifacts) -> None:
    """Persist GeoJSON + GeoParquet artifacts for downstream loaders."""
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)

    geojson_path = VECTOR_DIR / "jjj_power_assets.geojson"
    parquet_path = VECTORIZED_DIR / "jjj_power_assets.parquet"

    artifacts.frame.to_file(geojson_path, driver="GeoJSON")
    artifacts.frame.to_parquet(parquet_path, index=False)

    print(f"Wrote {len(artifacts.frame)} features -> {geojson_path}")
    print(f"Wrote {len(artifacts.frame)} features -> {parquet_path}")
    print("Summary:", artifacts.summary)


def main() -> None:
    artifacts = build_layer()
    write_outputs(artifacts)


if __name__ == "__main__":
    main()
