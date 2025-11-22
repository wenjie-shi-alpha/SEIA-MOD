"""Kerchunk-based GRIB2 indexer for weather cube virtualization."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
import glob
from pathlib import Path
from typing import Iterable, List, Sequence

import fsspec
from kerchunk.combine import MultiZarrToZarr
from kerchunk.grib2 import scan_grib

logger = logging.getLogger(__name__)


@dataclass
class WeatherCubeIndexerConfig:
    """Runtime configuration for the weather cube indexer."""

    source_glob: str
    output_dir: Path
    s3_bucket: str | None = None
    s3_prefix: str = "weather/indexes"
    storage_options: dict | None = None
    object_store_options: dict | None = None
    inline_threshold: int = 300
    remote_protocol: str = "file"
    concat_dims: Sequence[str] = field(default_factory=lambda: ["time"])
    identical_dims: Sequence[str] | None = None


class WeatherCubeIndexer:
    """Wraps Kerchunk routines to produce consolidated Zarr references."""

    def __init__(self, config: WeatherCubeIndexerConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_sources(self) -> List[Path]:
        """Return a sorted list of GRIB2 files to index."""
        sources = sorted(Path(path) for path in glob.glob(self.config.source_glob))
        if not sources:
            logger.warning("No GRIB2 files matched pattern %s", self.config.source_glob)
        return sources

    def build_single_reference(self, grib_path: Path) -> Path:
        """Run kerchunk.grib2.scan_grib for a single GRIB2 file."""
        logger.info("Scanning %s", grib_path)
        reference = scan_grib(
            grib_path.as_posix(),
            storage_options=self.config.storage_options or {},
            inline_threshold=self.config.inline_threshold,
        )
        target = self.output_dir / f"{grib_path.stem}.json"
        target.write_text(json.dumps(reference), encoding="utf-8")
        return target

    def consolidate_references(self, refs: Iterable[Path]) -> Path:
        """Combine individual Kerchunk manifests via MultiZarrToZarr."""
        refs = list(refs)
        if not refs:
            raise ValueError("No reference manifests were produced; aborting consolidation.")
        logger.info("Consolidating %d reference files", len(refs))
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in refs]
        translator = MultiZarrToZarr(
            payloads,
            remote_protocol=self.config.remote_protocol,
            remote_options=self.config.storage_options or {},
            concat_dims=list(self.config.concat_dims),
            identical_dims=list(self.config.identical_dims) if self.config.identical_dims else None,
        )
        consolidated = translator.translate()
        target = self.output_dir / "consolidated.json"
        target.write_text(json.dumps(consolidated), encoding="utf-8")
        return target

    def push_to_object_store(self, artifact: Path) -> str | None:
        """Upload the consolidated manifest to MinIO/S3 via fsspec."""
        if not self.config.s3_bucket:
            logger.debug("s3_bucket not configured; skipping upload for %s", artifact)
            return None
        fs = fsspec.filesystem("s3", **(self.config.object_store_options or {}))
        key = f"{self.config.s3_prefix.rstrip('/')}/{artifact.name}"
        remote_path = f"{self.config.s3_bucket}/{key}"
        logger.info("Uploading %s to s3://%s", artifact, remote_path)
        fs.put_file(str(artifact), remote_path)
        return f"s3://{remote_path}"

    def run(self) -> Path:
        """Execute the end-to-end indexing pipeline."""
        refs = [self.build_single_reference(path) for path in self.scan_sources()]
        consolidated = self.consolidate_references(refs)
        self.push_to_object_store(consolidated)
        return consolidated


__all__ = ["WeatherCubeIndexer", "WeatherCubeIndexerConfig"]
