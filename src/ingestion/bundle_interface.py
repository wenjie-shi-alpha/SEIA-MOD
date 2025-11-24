"""Interfaces for consuming externally prepared exposure asset bundles."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(slots=True)
class AssetBundleManifest:
    """Metadata describing an externally prepared asset bundle."""

    bundle_id: str
    version: str
    region: str
    asset_types: Sequence[str]
    hazard_types: Sequence[str]
    asset_count: int
    geo_parquet: str
    vulnerability_profiles: str | None = None
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetBundleManifest":
        return cls(
            bundle_id=str(payload["bundle_id"]),
            version=str(payload["version"]),
            region=str(payload.get("region", "")),
            asset_types=tuple(payload.get("asset_types", [])),
            hazard_types=tuple(payload.get("hazard_types", [])),
            asset_count=int(payload.get("asset_count", 0)),
            geo_parquet=str(payload["geo_parquet"]),
            vulnerability_profiles=payload.get("vulnerability_profiles"),
            checksum=payload.get("checksum"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class AssetBundle:
    """Concrete file references ready for ingestion."""

    manifest: AssetBundleManifest
    geo_parquet: Path
    vulnerability_profiles: Path | None = None


class AssetBundleProvider(Protocol):
    """Protocol for pulling externally managed exposure bundles."""

    def list_bundles(self, region: str | None = None) -> Sequence[AssetBundleManifest]:
        """Return manifests that match the optional region filter."""

    def fetch_bundle(self, manifest: AssetBundleManifest, target_dir: Path | None = None) -> AssetBundle:
        """Materialize the manifest into filesystem paths ready for ingestion."""


class LocalBundleProvider:
    """
    Loads bundle manifests from a local JSON file and resolves relative paths.

    This keeps SEIA-Mod decoupled from how bundles are produced; an external
    crawler can export GeoParquet files and update the manifest without
    changing the core project.
    """

    def __init__(self, manifest_path: Path, storage_root: Path | None = None) -> None:
        self.manifest_path = manifest_path
        self.storage_root = storage_root or manifest_path.parent
        self._payload = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_bundles(self, region: str | None = None) -> Sequence[AssetBundleManifest]:
        records = []
        for item in self._payload.get("bundles", []):
            manifest = AssetBundleManifest.from_dict(item)
            if region and manifest.region != region:
                continue
            records.append(manifest)
        return records

    def fetch_bundle(self, manifest: AssetBundleManifest, target_dir: Path | None = None) -> AssetBundle:
        """
        Resolve manifest file references into concrete Paths.

        If target_dir is provided, files are expected to already exist there
        (for example, synced from object storage by an external pipeline).
        """

        def _resolve(uri: str) -> Path:
            path = Path(uri)
            if path.is_absolute():
                return path
            base = target_dir or self.storage_root
            return (base / uri).resolve()

        geo_path = _resolve(manifest.geo_parquet)
        if not geo_path.exists():
            raise FileNotFoundError(f"GeoParquet for bundle {manifest.bundle_id} not found: {geo_path}")

        vuln_path = None
        if manifest.vulnerability_profiles:
            candidate = _resolve(manifest.vulnerability_profiles)
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Vulnerability profiles for bundle {manifest.bundle_id} not found: {candidate}"
                )
            vuln_path = candidate

        return AssetBundle(manifest=manifest, geo_parquet=geo_path, vulnerability_profiles=vuln_path)


__all__ = ["AssetBundle", "AssetBundleManifest", "AssetBundleProvider", "LocalBundleProvider"]
