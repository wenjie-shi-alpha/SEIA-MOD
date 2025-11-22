"""Tests for the exposure graph loader."""
from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import Point

pytest.importorskip("geopandas")
pytest.importorskip("pyarrow")

import geopandas as gpd

from src.graph.loader import ExposureGraphLoader, GraphLoaderConfig
from src.graph import loader as loader_module


class FakeCollection:
    def __init__(self, name: str, edge: bool = False):
        self.name = name
        self.edge = edge
        self.docs = []
        self.indexes = []

    def import_bulk(self, docs, on_duplicate: str = "update"):
        self.docs.extend(docs)

    def add_hash_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeDB:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, edge: bool = False):
        collection = FakeCollection(name, edge=edge)
        self.collections[name] = collection
        return collection

    def collection(self, name: str):
        return self.collections[name]


class FakeArangoClient:
    def __init__(self, hosts: str):
        self.hosts = hosts
        self.db_handle = FakeDB()

    def db(self, *args, **kwargs):
        return self.db_handle


def test_graph_loader_imports_assets(tmp_path, monkeypatch):
    geo_df = gpd.GeoDataFrame(
        {
            "_key": ["asset_a", "asset_b"],
            "type": ["factory", "warehouse"],
            "supplier_ids": [[], ["asset_a"]],
            "vulnerability_type": ["factory_vuln", "warehouse_vuln"],
        },
        geometry=[Point(0.0, 0.0), Point(0.1, 0.1)],
        crs="EPSG:4326",
    )

    parquet_path = tmp_path / "assets.parquet"
    geo_df.to_parquet(parquet_path)

    client = FakeArangoClient("http://localhost:8529")

    def fake_client(hosts: str):
        return client

    monkeypatch.setattr(loader_module, "ArangoClient", fake_client)

    config = GraphLoaderConfig(
        geo_parquet_path=parquet_path,
        arangodb_url="http://localhost:8529",
        database="seia",
        username="root",
        password="pw",
    )

    loader = ExposureGraphLoader(config)
    loader.run()

    assets = client.db_handle.collection("Assets").docs
    assert len(assets) == 2
    assert all("geometry_wkt" in doc for doc in assets)
    assert all(doc["h3_index"] for doc in assets)

    edges = client.db_handle.collection("SupplyChain").docs
    assert edges[0]["_from"] == "Assets/asset_a"
    assert edges[0]["_to"] == "Assets/asset_b"

    vuln_edges = client.db_handle.collection("HasVulnerability").docs
    assert {edge["_to"] for edge in vuln_edges} == {
        "Vulnerability/factory_vuln",
        "Vulnerability/warehouse_vuln",
    }

    query = loader.impacted_assets_aql(["h3_a", "h3_b"])
    assert "FILTER asset.h3_index IN" in query
