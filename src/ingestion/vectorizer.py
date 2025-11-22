"""Vector asset conversion pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class VectorizerConfig:
    """Configuration for GeoParquet conversion."""

    input_paths: Iterable[Path]
    output_dir: Path
    region_filter_sql: str


class VectorAssetVectorizer:
    """Handles Shapefile/GeoJSON -> GeoParquet conversion and filtering."""

    def __init__(self, config: VectorizerConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sources(self) -> None:
        """Placeholder for geopandas.read_file."""
        # TODO: implement geopandas + DuckDB filtering

    def convert(self) -> None:
        """Placeholder for conversion logic."""

    def run(self) -> None:
        self.load_sources()
        self.convert()


__all__ = ["VectorAssetVectorizer", "VectorizerConfig"]
