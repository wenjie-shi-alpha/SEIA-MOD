"""Vector asset conversion pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import duckdb
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely import from_wkb

logger = logging.getLogger(__name__)


@dataclass
class VectorizerConfig:
    """Configuration for GeoParquet conversion."""

    input_paths: Iterable[Path]
    output_dir: Path
    region_filter_sql: str | None = None
    output_filename: str = "assets.parquet"


class VectorAssetVectorizer:
    """Handles Shapefile/GeoJSON -> GeoParquet conversion and filtering."""

    def __init__(self, config: VectorizerConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sources(self) -> GeoDataFrame:
        """Load all vector inputs into a single GeoDataFrame."""
        frames: List[GeoDataFrame] = []
        for path in self.config.input_paths:
            logger.info("Reading vector asset %s", path)
            frames.append(gpd.read_file(path))
        if not frames:
            raise ValueError("Vectorizer received no input paths.")
        combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
        return combined

    def apply_region_filter(self, gdf: GeoDataFrame) -> GeoDataFrame:
        """Filter GeoDataFrame rows via DuckDB SQL (attribute or spatial)."""
        if not self.config.region_filter_sql:
            return gdf
        logger.info("Applying DuckDB filter: %s", self.config.region_filter_sql)
        working = gdf.copy()
        working["geometry_wkb"] = working.geometry.to_wkb()
        con = duckdb.connect(database=":memory:")
        con.register("vector_assets", working.drop(columns="geometry"))
        filtered_df = con.execute(self.config.region_filter_sql).df()
        con.close()
        if "geometry_wkb" not in filtered_df.columns:
            raise ValueError("Region filter SQL must return a geometry_wkb column.")
        filtered_df["geometry"] = filtered_df.pop("geometry_wkb").apply(
            lambda value: from_wkb(self._coerce_wkb(value))
        )
        return gpd.GeoDataFrame(filtered_df, geometry="geometry", crs=gdf.crs)

    def convert(self) -> Path:
        """Run the full conversion pipeline and return the GeoParquet path."""
        combined = self.load_sources()
        filtered = self.apply_region_filter(combined)
        target = self.config.output_dir / self.config.output_filename
        logger.info("Writing %d features to %s", len(filtered), target)
        filtered.to_parquet(target, index=False)
        return target

    def run(self) -> Path:
        """Convenience wrapper for CLI usage."""
        return self.convert()

    @staticmethod
    def _coerce_wkb(value):
        """Normalize DuckDB BLOB outputs into bytes for Shapely."""
        if value is None:
            return value
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if hasattr(value, "tobytes"):
            return value.tobytes()
        if isinstance(value, list):
            return bytes(value)
        return value


__all__ = ["VectorAssetVectorizer", "VectorizerConfig"]
