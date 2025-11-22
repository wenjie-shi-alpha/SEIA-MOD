"""Tests for ingestion pipelines (indexer + vectorizer)."""
from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import pytest

pytest.importorskip("kerchunk")
pytest.importorskip("fsspec")
pytest.importorskip("geopandas")
pytest.importorskip("duckdb")

from shapely.geometry import Point
import geopandas as gpd

if "cfgrib" not in sys.modules:
    sys.modules["cfgrib"] = types.ModuleType("cfgrib")

from src.ingestion.indexer import WeatherCubeIndexer, WeatherCubeIndexerConfig
from src.ingestion import indexer as indexer_module
from src.ingestion.vectorizer import VectorAssetVectorizer, VectorizerConfig
from src.ingestion import vectorizer as vectorizer_module


def test_weather_cube_indexer_runs_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "grib"
    source.mkdir()
    grib_file = source / "sample.grib2"
    grib_file.write_bytes(b"")

    outputs = tmp_path / "indexes"

    config = WeatherCubeIndexerConfig(
        source_glob=str(grib_file),
        output_dir=outputs,
        s3_bucket="seia-bucket",
        object_store_options={"client_kwargs": {"endpoint_url": "http://minio"}},
    )

    scanned: dict[str, Any] = {}

    def fake_scan(path: str, storage_options: dict, inline_threshold: int):
        scanned["path"] = path
        scanned["threshold"] = inline_threshold
        return {"path": path, "chunks": []}

    monkeypatch.setattr(indexer_module, "scan_grib", fake_scan)

    class DummyTranslator:
        def __init__(self, payloads, **kwargs):
            self.payloads = payloads
            self.kwargs = kwargs

        def translate(self):
            return {"combined": len(self.payloads)}

    monkeypatch.setattr(indexer_module, "MultiZarrToZarr", DummyTranslator)

    uploads: list[tuple[str, str]] = []

    class DummyFS:
        def put_file(self, local: str, remote: str):
            uploads.append((local, remote))

    def fake_filesystem(protocol: str, **kwargs):
        assert protocol == "s3"
        return DummyFS()

    monkeypatch.setattr(indexer_module.fsspec, "filesystem", fake_filesystem)

    consolidated = WeatherCubeIndexer(config).run()

    assert consolidated.exists()
    assert uploads, "Expected consolidated manifest to be uploaded to object store"
    assert scanned["path"].endswith("sample.grib2")


def test_vectorizer_filters_and_converts(tmp_path, monkeypatch):
    geo_df = gpd.GeoDataFrame(
        {
            "name": ["Plant A", "Plant B"],
            "region": ["focus", "other"],
            "supplier_ids": [["asset_a"], []],
            "_key": ["asset_a", "asset_b"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )

    def fake_read_file(path: Path):
        frame = geo_df.copy()
        frame["source"] = path.name
        return frame

    monkeypatch.setattr(vectorizer_module.gpd, "read_file", fake_read_file)

    config = VectorizerConfig(
        input_paths=[tmp_path / "assets.geojson"],
        output_dir=tmp_path,
        region_filter_sql="SELECT * FROM vector_assets WHERE region = 'focus'",
    )

    output = VectorAssetVectorizer(config).run()
    assert output.exists()

    filtered = gpd.read_parquet(output)
    assert list(filtered["name"]) == ["Plant A"]
