"""Neuro-symbolic meteorological cognition pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, Sequence

import numpy as np
import requests
import xarray as xr
import h3

from src.cognition.llm_client import LLMClient


logger = logging.getLogger(__name__)


class DownscalerProtocol(Protocol):
    """Protocol describing a Prithvi-like downscaler."""

    def infer(self, coarse_tensor: Any) -> xr.Dataset:  # pragma: no cover - protocol definition
        ...


class SegmenterProtocol(Protocol):
    """Protocol describing a SAM-Geo like segmenter."""

    def segment(self, hi_res_tensor: Any) -> Dict[str, Any]:  # pragma: no cover - protocol definition
        ...


@dataclass
class PrithviWxCConfig:
    """Runtime parameters for the Prithvi WxC downscaler wrapper."""

    endpoint: str | None = None
    api_key: str | None = None
    model: str = "ibm-granite/prithvi-wxc"
    timeout: float = 60.0
    target_resolution_km: float = 2.0
    fallback_interp_method: str = "cubic"
    static_features: Dict[str, Any] = field(default_factory=dict)


class PrithviWxCDownscaler(DownscalerProtocol):
    """Bridges the pipeline with a remote or local Prithvi WxC deployment."""

    DEG_PER_KM = 1.0 / 111.0

    def __init__(self, config: PrithviWxCConfig) -> None:
        self.config = config
        self.session = requests.Session() if config.endpoint else None

    def infer(self, coarse_tensor: Any) -> xr.Dataset:
        dataset = _ensure_dataset(coarse_tensor)
        dataset = self._inject_static_features(dataset)
        if self.config.endpoint:
            try:
                return self._remote_infer(dataset)
            except RuntimeError as exc:  # pragma: no cover - network path
                logger.warning("Remote Prithvi WxC inference failed, falling back to local interp: %s", exc)
        return self._fallback_infer(dataset)

    def _inject_static_features(self, dataset: xr.Dataset) -> xr.Dataset:
        if not self.config.static_features:
            return dataset
        template = next(iter(dataset.data_vars.values()))
        coords = {dim: dataset.coords[dim] for dim in template.dims}
        assigned: Dict[str, xr.DataArray] = {}
        for name, value in self.config.static_features.items():
            if isinstance(value, xr.DataArray):
                data_array = value
                if set(data_array.dims) != set(template.dims):
                    data_array = data_array.interp(coords, kwargs={"fill_value": "extrapolate"})
            elif isinstance(value, xr.Dataset):
                first = next(iter(value.data_vars.values()))
                data_array = first.interp(coords)
            else:
                filled = np.full(template.shape, float(value), dtype=float)
                data_array = xr.DataArray(filled, dims=template.dims, coords=coords)
            assigned[name] = data_array
        return dataset.assign(assigned)

    def _remote_infer(self, dataset: xr.Dataset) -> xr.Dataset:
        if not self.session or not self.config.endpoint:
            raise RuntimeError("Remote endpoint not configured")
        payload = {
            "model": self.config.model,
            "target_resolution_km": self.config.target_resolution_km,
            "tensor": _dataset_to_payload(dataset),
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = self.session.post(
            self.config.endpoint,
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Prithvi WxC endpoint error: {response.text}")
        data = response.json()
        tensor = data.get("tensor")
        if not tensor:
            raise RuntimeError("Remote response missing tensor field")
        return _dataset_from_payload(tensor)

    def _fallback_infer(self, dataset: xr.Dataset) -> xr.Dataset:
        lat_name = _resolve_coord_name(dataset, ("lat", "latitude", "y"))
        lon_name = _resolve_coord_name(dataset, ("lon", "longitude", "x"))
        coords = self._target_coords(
            lat_name,
            dataset.coords[lat_name].values,
            lon_name,
            dataset.coords[lon_name].values,
        )
        return dataset.interp(
            {lat_name: coords[lat_name], lon_name: coords[lon_name]},
            method=self.config.fallback_interp_method,
        )

    def _target_coords(
        self,
        lat_name: str,
        latitudes: np.ndarray,
        lon_name: str,
        longitudes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        step_deg = max(self.config.target_resolution_km * self.DEG_PER_KM, 1e-4)
        lat_min, lat_max = float(np.min(latitudes)), float(np.max(latitudes))
        lon_min, lon_max = float(np.min(longitudes)), float(np.max(longitudes))
        lat_points = max(int(abs(lat_max - lat_min) / step_deg) + 1, len(latitudes))
        lon_points = max(int(abs(lon_max - lon_min) / step_deg) + 1, len(longitudes))
        return {
            lat_name: np.linspace(lat_min, lat_max, lat_points),
            lon_name: np.linspace(lon_min, lon_max, lon_points),
        }


@dataclass
class SAMGeoSegmenterConfig:
    """Configuration for the SAM-Geo inspired segmenter."""

    endpoint: str | None = None
    api_key: str | None = None
    model: str = "sam-geo"
    timeout: float = 45.0
    intensity_field: str | None = None
    threshold: float | None = None
    percentile: float = 85.0
    h3_resolution: int = 8
    min_mask_fraction: float = 0.003
    return_mask: bool = False
    prompts: List[Dict[str, float]] = field(default_factory=list)


class SAMGeoSegmenter(SegmenterProtocol):
    """Approximation of SAM-Geo segmentation with optional remote execution."""

    def __init__(self, config: SAMGeoSegmenterConfig) -> None:
        self.config = config
        self.session = requests.Session() if config.endpoint else None

    def segment(self, hi_res_tensor: Any) -> Dict[str, Any]:
        dataset = _ensure_dataset(hi_res_tensor)
        if self.config.endpoint:
            try:
                return self._remote_segment(dataset)
            except RuntimeError as exc:  # pragma: no cover - network path
                logger.warning("Remote SAM-Geo segmentation failed, falling back to heuristic segmentation: %s", exc)
        return self._local_segment(dataset)

    def _remote_segment(self, dataset: xr.Dataset) -> Dict[str, Any]:
        if not self.session or not self.config.endpoint:
            raise RuntimeError("Remote SAM-Geo endpoint not configured")
        payload = {
            "model": self.config.model,
            "tensor": _dataset_to_payload(dataset),
            "prompts": self.config.prompts,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = self.session.post(
            self.config.endpoint,
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"SAM-Geo endpoint error: {response.text}")
        data = response.json()
        if "stats" not in data:
            raise RuntimeError("Segmenter response missing stats field")
        return data

    def _local_segment(self, dataset: xr.Dataset) -> Dict[str, Any]:
        field = self._select_field(dataset)
        lat_name = _resolve_coord_name(dataset, ("lat", "latitude", "y"))
        lon_name = _resolve_coord_name(dataset, ("lon", "longitude", "x"))
        latitudes = dataset.coords[lat_name].values
        longitudes = dataset.coords[lon_name].values
        arr = np.nan_to_num(field.values.astype(float))
        threshold = self._determine_threshold(arr)
        mask = arr >= threshold
        mask_fraction = float(mask.mean()) if mask.size else 0.0
        if mask_fraction < self.config.min_mask_fraction and mask.size:
            peak_idx = np.unravel_index(np.argmax(arr), arr.shape)
            mask[:] = False
            mask[peak_idx] = True
            mask_fraction = float(mask.mean())
        h3_cells = self._mask_to_h3(mask, latitudes, longitudes)
        stats = {
            "max_intensity": float(np.max(arr)) if arr.size else 0.0,
            "mean_intensity": float(np.mean(arr)) if arr.size else 0.0,
            "threshold": float(threshold),
            "mask_fraction": mask_fraction,
            "coverage_cells": len(h3_cells),
        }
        bounds = self._mask_bounds(mask, latitudes, longitudes)
        payload: Dict[str, Any] = {
            "stats": stats,
            "h3_coverage": sorted(h3_cells),
            "bounds": bounds,
        }
        if self.config.return_mask:
            payload["mask"] = mask.astype(int).tolist()
        return payload

    def _select_field(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.config.intensity_field and self.config.intensity_field in dataset:
            field = dataset[self.config.intensity_field]
        else:
            first = next(iter(dataset.data_vars))
            field = dataset[first]
        dims = [dim for dim in field.dims if dim not in ("lat", "latitude", "lon", "longitude", "x", "y")]
        if dims:
            field = field.isel({dims[0]: 0})
        return field

    def _determine_threshold(self, arr: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if not arr.size:
            return 0.0
        percentile = np.nanpercentile(arr, self.config.percentile)
        return float(percentile)

    def _mask_to_h3(self, mask: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> set[str]:
        if not mask.size:
            return set()
        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        lat_selected = lat_grid[mask]
        lon_selected = lon_grid[mask]
        cells = {
            h3.latlng_to_cell(float(lat), float(lon), self.config.h3_resolution)
            for lat, lon in zip(lat_selected, lon_selected)
        }
        return cells

    def _mask_bounds(self, mask: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> Dict[str, float]:
        if not mask.size or not mask.any():
            return {"lat_min": float(latitudes.min()), "lat_max": float(latitudes.max()), "lon_min": float(longitudes.min()), "lon_max": float(longitudes.max())}
        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        lat_selected = lat_grid[mask]
        lon_selected = lon_grid[mask]
        return {
            "lat_min": float(lat_selected.min()),
            "lat_max": float(lat_selected.max()),
            "lon_min": float(lon_selected.min()),
            "lon_max": float(lon_selected.max()),
        }


@dataclass
class MeteoCognitionConfig:
    """Configuration for the meteorological cognition pipeline."""

    downscaler: DownscalerProtocol | Callable[[Any], Any]
    segmenter: SegmenterProtocol | Callable[[Any], Dict[str, Any]]
    llm_client: LLMClient
    llm_prompt: str = (
        "You are a meteorological event generator. "
        "Given the segmentation statistics, produce a structured JSON event description."
    )
    default_event_type: str = "storm"
    h3_fallback_resolution: int = 7
    metrics_fields: List[str] = field(default_factory=lambda: ["max_intensity", "mean_intensity"])


@dataclass
class EventObservation:
    """Structured event output consumed by downstream mappers."""

    type: str
    h3_coverage: List[str]
    intensity_index: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MeteoCognitionPipeline:
    """Coordinates downscaling, segmentation, and LLM event synthesis."""

    def __init__(self, config: MeteoCognitionConfig) -> None:
        self.config = config

    def downscale(self, coarse_tensor: Any) -> Any:
        """Run the configured downscaler."""
        downscaler = self.config.downscaler
        if hasattr(downscaler, "infer"):
            return downscaler.infer(coarse_tensor)
        return downscaler(coarse_tensor)  # type: ignore[operator]

    def segment(self, hi_res_tensor: Any) -> Dict[str, Any]:
        """Execute SAM-Geo segmentation and extract statistics."""
        segmenter = self.config.segmenter
        if hasattr(segmenter, "segment"):
            result = segmenter.segment(hi_res_tensor)
        else:
            result = segmenter(hi_res_tensor)  # type: ignore[operator]
        result.setdefault("stats", {})
        result.setdefault("h3_coverage", [])
        return result

    def describe_event(self, segmentation: Dict[str, Any]) -> EventObservation:
        """Use the LLM client to describe a meteorological event."""
        stats = segmentation.get("stats", {})
        prompt = f"{self.config.llm_prompt}\nStats: {json.dumps(stats)}"
        llm_payload = self.config.llm_client.generate(prompt, payload=segmentation)
        event_json = llm_payload.get("json") or {}
        event_type = event_json.get("type") or self.config.default_event_type
        h3_coverage = event_json.get("h3_coverage") or segmentation.get("h3_coverage") or []
        intensity = float(event_json.get("intensity_index") or stats.get("max_intensity") or 0.0)
        metadata = event_json.get("metadata") or stats
        return EventObservation(
            type=event_type,
            h3_coverage=h3_coverage,
            intensity_index=intensity,
            metadata=metadata,
        )

    def run(self, coarse_tensor: Any) -> EventObservation:
        """End-to-end execution from downscaling to event emission."""
        hi_res = self.downscale(coarse_tensor)
        segmentation = self.segment(hi_res)
        return self.describe_event(segmentation)


def _ensure_dataset(tensor: Any) -> xr.Dataset:
    """Normalize arbitrary tensor inputs into an xarray.Dataset."""
    if isinstance(tensor, xr.Dataset):
        return tensor
    if isinstance(tensor, xr.DataArray):
        name = tensor.name or "field"
        return tensor.to_dataset(name=name)
    if isinstance(tensor, dict):
        if "tensor" in tensor:
            return _dataset_from_payload(tensor["tensor"])
        if "coords" in tensor and "data_vars" in tensor:
            return _dataset_from_payload(tensor)
        if {"lat", "lon", "values"}.issubset(tensor.keys()):
            lat = np.asarray(tensor["lat"], dtype=float)
            lon = np.asarray(tensor["lon"], dtype=float)
            values = np.asarray(tensor["values"], dtype=float)
            dims = tuple(tensor.get("dims", ("lat", "lon")))
            data_vars = {
                tensor.get("name", "field"): (dims, values),
            }
            coords = {"lat": lat, "lon": lon}
            return xr.Dataset(data_vars=data_vars, coords=coords)
    raise TypeError("Unsupported tensor format for meteo cognition pipeline")


def _resolve_coord_name(dataset: xr.Dataset, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in dataset.coords:
            return candidate
    raise ValueError(f"None of the coordinate candidates {candidates} exist in dataset ({list(dataset.coords)})")


def _dataset_to_payload(dataset: xr.Dataset) -> Dict[str, Any]:
    return {
        "coords": {name: dataset.coords[name].values.tolist() for name in dataset.coords},
        "data_vars": {
            name: {
                "dims": list(array.dims),
                "values": array.values.tolist(),
            }
            for name, array in dataset.data_vars.items()
        },
    }


def _dataset_from_payload(payload: Dict[str, Any]) -> xr.Dataset:
    coords = {name: np.asarray(values) for name, values in payload.get("coords", {}).items()}
    data_vars = {}
    for name, spec in payload.get("data_vars", {}).items():
        dims = tuple(spec.get("dims") or ("lat", "lon"))
        values = np.asarray(spec.get("values") or spec.get("data"))
        data_vars[name] = (dims, values)
    if not data_vars:
        raise ValueError("Payload missing data_vars for dataset reconstruction")
    return xr.Dataset(data_vars=data_vars, coords=coords)


__all__ = [
    "EventObservation",
    "MeteoCognitionConfig",
    "MeteoCognitionPipeline",
    "PrithviWxCConfig",
    "PrithviWxCDownscaler",
    "SAMGeoSegmenterConfig",
    "SAMGeoSegmenter",
]
