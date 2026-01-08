"""Enhanced meteorological cognition pipeline with local AI model support."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from src.cognition.llm_client import LLMClient
from src.models import LocalPrithviWXC, LocalSAMGeo, ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class LocalMeteoCognitionConfig:
    """Configuration for the enhanced local meteorological cognition pipeline."""

    # Model configurations
    prithvi_model: str = "prithvi_wxc_small"
    sam_model: str = "sam_geo_base"
    device: str = "cuda"
    use_local_models: bool = True
    fallback_to_remote: bool = True

    # Processing parameters
    target_resolution_km: float = 2.0
    h3_resolution: int = 8
    intensity_percentile: float = 85.0
    min_mask_fraction: float = 0.003

    # LLM configuration
    llm_client: Optional[LLMClient] = None
    llm_prompt: str = (
        "You are a meteorological event generator. "
        "Given the segmentation statistics, produce a structured JSON event description "
        "with type, h3_coverage, intensity_index, and metadata fields."
    )
    default_event_type: str = "storm"

    # Model-specific settings
    prithvi_static_features: Dict[str, Any] = field(default_factory=dict)
    sam_prompts: List[Dict[str, Any]] = field(default_factory=list)
    sam_threshold: Optional[float] = None


class LocalMeteoCognitionPipeline:
    """Enhanced meteorological cognition pipeline with local AI model support."""

    def __init__(self, config: LocalMeteoCognitionConfig) -> None:
        self.config = config
        self.model_registry = ModelRegistry()

        # Initialize local models
        self.prithvi_model: Optional[LocalPrithviWXC] = None
        self.sam_model: Optional[LocalSAMGeo] = None

        if config.use_local_models:
            self._initialize_local_models()

        # Fallback to original remote pipeline if models fail to load
        self._fallback_to_remote = False

    def _initialize_local_models(self) -> None:
        """Initialize local AI models."""
        try:
            # Initialize Prithvi WXC
            logger.info(f"Initializing Prithvi WXC model: {self.config.prithvi_model}")
            self.prithvi_model = LocalPrithviWXC(
                model_name=self.config.prithvi_model
            )

            # Initialize SAM-Geo
            logger.info(f"Initializing SAM-Geo model: {self.config.sam_model}")
            self.sam_model = LocalSAMGeo(
                model_name=self.config.sam_model
            )

            logger.info("Local AI models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize local models: {e}")
            if self.config.fallback_to_remote:
                logger.warning("Falling back to remote pipeline")
                self._fallback_to_remote = True
                self.prithvi_model = None
                self.sam_model = None
            else:
                raise

    def downscale(self, coarse_data: Any) -> Any:
        """Perform AI-based downscaling."""
        if self.prithvi_model and not self._fallback_to_remote:
            return self._local_downscale(coarse_data)
        else:
            return self._fallback_downscale(coarse_data)

    def _local_downscale(self, coarse_data: Any) -> xr.Dataset:
        """Use local Prithvi WXC model for downscaling."""
        try:
            dataset = self._ensure_dataset(coarse_data)

            # Add static features if provided
            if self.config.prithvi_static_features:
                dataset = self._add_static_features(dataset)

            # Perform downscaling
            high_res_data = self.prithvi_model.downscale(dataset)

            logger.info(f"Local downscaling completed: {dataset.sizes} -> {high_res_data.sizes}")
            return high_res_data

        except Exception as e:
            logger.error(f"Local downscaling failed: {e}")
            return self._fallback_downscale(coarse_data)

    def _fallback_downscale(self, coarse_data: Any) -> xr.Dataset:
        """Fallback to simple interpolation."""
        dataset = self._ensure_dataset(coarse_data)

        # Simple upscaling by interpolation
        scale_factor = 4
        lat_name = self._find_coord_name(dataset, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(dataset, ["lon", "longitude", "x"])

        lat_coords = dataset.coords[lat_name].values
        lon_coords = dataset.coords[lon_name].values

        lat_hr = np.linspace(
            lat_coords.min(), lat_coords.max(),
            len(lat_coords) * scale_factor
        )
        lon_hr = np.linspace(
            lon_coords.min(), lon_coords.max(),
            len(lon_coords) * scale_factor
        )

        # Interpolate data variables
        output_vars = {}
        for name, var in dataset.data_vars.items():
            if var.ndim >= 2:
                interpolated = var.interp(
                    {lat_name: lat_hr, lon_name: lon_hr},
                    method="linear"
                )
                output_vars[name] = interpolated

        result = xr.Dataset(output_vars)
        logger.info(f"Fallback downscaling completed: {dataset.sizes} -> {result.sizes}")
        return result

    def segment(self, hi_res_data: Any) -> Dict[str, Any]:
        """Perform AI-based segmentation."""
        if self.sam_model and not self._fallback_to_remote:
            return self._local_segment(hi_res_data)
        else:
            return self._fallback_segment(hi_res_data)

    def _local_segment(self, hi_res_data: Any) -> Dict[str, Any]:
        """Use local SAM-Geo model for segmentation."""
        try:
            dataset = self._ensure_dataset(hi_res_data)

            # Perform segmentation
            result = self.sam_model.segment_meteorological_features(
                data=dataset,
                intensity_field=None,  # Auto-detect
                threshold=self.config.sam_threshold,
                percentile=self.config.intensity_percentile,
                h3_resolution=self.config.h3_resolution,
                prompts=self.config.sam_prompts if self.config.sam_prompts else None
            )

            logger.info(f"Local segmentation completed: {len(result.get('h3_coverage', []))} H3 cells")
            return result

        except Exception as e:
            logger.error(f"Local segmentation failed: {e}")
            return self._fallback_segment(hi_res_data)

    def _fallback_segment(self, hi_res_data: Any) -> Dict[str, Any]:
        """Fallback to simple thresholding segmentation."""
        dataset = self._ensure_dataset(hi_res_data)

        # Use first data variable
        first_var = list(dataset.data_vars.keys())[0]
        field = dataset[first_var]

        # Handle time dimension
        if field.ndim > 2:
            field = field.isel({dim: 0 for dim in field.dims[:-2]})

        # Threshold-based segmentation
        values = field.values
        threshold = np.nanpercentile(values, self.config.intensity_percentile)
        mask = values >= threshold

        # Convert to H3 cells
        h3_cells = self._mask_to_h3(mask, dataset, self.config.h3_resolution)

        stats = {
            "max_intensity": float(np.nanmax(values)),
            "mean_intensity": float(np.nanmean(values)),
            "threshold": float(threshold),
            "mask_fraction": float(mask.mean()),
            "coverage_cells": len(h3_cells),
        }

        result = {
            "stats": stats,
            "h3_coverage": sorted(h3_cells),
            "method": "threshold_fallback"
        }

        logger.info(f"Fallback segmentation completed: {len(h3_cells)} H3 cells")
        return result

    def describe_event(self, segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate event description using LLM or simple mapping."""
        if self.config.llm_client:
            return self._llm_describe_event(segmentation)
        else:
            return self._simple_describe_event(segmentation)

    def _llm_describe_event(self, segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to describe the meteorological event."""
        try:
            stats = segmentation.get("stats", {})
            prompt = f"{self.config.llm_prompt}\nStats: {json.dumps(stats)}"

            llm_response = self.config.llm_client.generate(prompt, payload=segmentation)
            event_json = llm_response.get("json") or {}

            return {
                "type": event_json.get("type") or self.config.default_event_type,
                "h3_coverage": event_json.get("h3_coverage") or segmentation.get("h3_coverage", []),
                "intensity_index": float(event_json.get("intensity_index") or stats.get("max_intensity", 0.0)),
                "metadata": event_json.get("metadata") or stats,
            }

        except Exception as e:
            logger.error(f"LLM event description failed: {e}")
            return self._simple_describe_event(segmentation)

    def _simple_describe_event(self, segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple event description without LLM."""
        stats = segmentation.get("stats", {})
        max_intensity = stats.get("max_intensity", 0.0)

        # Simple event classification based on intensity
        if max_intensity > 0.8:
            event_type = "severe_storm"
        elif max_intensity > 0.6:
            event_type = "storm"
        elif max_intensity > 0.4:
            event_type = "moderate_weather"
        else:
            event_type = "light_weather"

        return {
            "type": event_type,
            "h3_coverage": segmentation.get("h3_coverage", []),
            "intensity_index": float(max_intensity),
            "metadata": stats,
        }

    def run(self, coarse_data: Any) -> Dict[str, Any]:
        """Run the complete meteorological cognition pipeline."""
        logger.info("Starting meteorological cognition pipeline")

        # Step 1: Downscaling
        hi_res_data = self.downscale(coarse_data)

        # Step 2: Segmentation
        segmentation = self.segment(hi_res_data)

        # Step 3: Event description
        event = self.describe_event(segmentation)

        logger.info(f"Pipeline completed: {event['type']} event with intensity {event['intensity_index']:.2f}")
        return event

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models."""
        status = {
            "local_models_enabled": self.config.use_local_models,
            "fallback_mode": self._fallback_to_remote,
            "prithvi_model": None,
            "sam_model": None,
        }

        if self.prithvi_model:
            try:
                status["prithvi_model"] = self.prithvi_model.get_model_info()
            except Exception as e:
                status["prithvi_model"] = {"error": str(e)}

        if self.sam_model:
            try:
                status["sam_model"] = self.sam_model.get_model_info()
            except Exception as e:
                status["sam_model"] = {"error": str(e)}

        return status

    def _ensure_dataset(self, data: Any) -> xr.Dataset:
        """Convert input to xarray Dataset."""
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            name = data.name or "field"
            return data.to_dataset(name=name)
        elif isinstance(data, dict):
            # Handle various dictionary formats
            if "lat" in data and "lon" in data and "values" in data:
                lat = np.asarray(data["lat"])
                lon = np.asarray(data["lon"])
                values = np.asarray(data["values"])
                dims = data.get("dims", ("lat", "lon"))
                return xr.Dataset(
                    {"field": (dims, values)},
                    coords={"lat": lat, "lon": lon}
                )
            # Add more dictionary format handling as needed
        raise TypeError(f"Unsupported data format: {type(data)}")

    def _add_static_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Add static features to the dataset for Prithvi WXC."""
        for name, feature in self.config.prithvi_static_features.items():
            if isinstance(feature, (int, float)):
                # Add as constant field
                template = next(iter(dataset.data_vars.values()))
                constant_field = xr.full_like(template, float(feature))
                dataset[name] = constant_field
            elif isinstance(feature, np.ndarray):
                # Add as array field
                if feature.shape == template.shape:
                    dataset[name] = (template.dims, feature)
                else:
                    logger.warning(f"Static feature {name} shape mismatch, skipping")

        return dataset

    def _mask_to_h3(self, mask: np.ndarray, dataset: xr.Dataset, h3_resolution: int) -> List[str]:
        """Convert binary mask to H3 cell coverage."""
        lat_name = self._find_coord_name(dataset, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(dataset, ["lon", "longitude", "x"])

        lats = dataset.coords[lat_name].values
        lons = dataset.coords[lon_name].values

        # Sample points from mask
        mask_points = np.where(mask)
        h3_cells = set()

        # Limit sampling for performance
        max_samples = 1000
        step = max(1, len(mask_points[0]) // max_samples)

        for i in range(0, len(mask_points[0]), step):
            lat_idx = mask_points[0][i]
            lon_idx = mask_points[1][i]

            if lat_idx < len(lats) and lon_idx < len(lons):
                lat = float(lats[lat_idx])
                lon = float(lons[lon_idx])
                try:
                    import h3
                    h3_cell = h3.latlng_to_cell(lat, lon, h3_resolution)
                    h3_cells.add(h3_cell)
                except Exception:
                    continue

        return list(h3_cells)

    @staticmethod
    def _find_coord_name(dataset: xr.Dataset, candidates: List[str]) -> str:
        """Find coordinate name from candidates."""
        for candidate in candidates:
            if candidate in dataset.coords:
                return candidate
        raise ValueError(f"None of {candidates} found in dataset coordinates")


# Convenience function for easy pipeline creation
def create_local_pipeline(
    prithvi_model: str = "prithvi_wxc_small",
    sam_model: str = "sam_geo_base",
    device: str = "cuda",
    llm_client: Optional[LLMClient] = None,
    **kwargs
) -> LocalMeteoCognitionPipeline:
    """Create a local meteorological cognition pipeline with default settings."""
    config = LocalMeteoCognitionConfig(
        prithvi_model=prithvi_model,
        sam_model=sam_model,
        device=device,
        llm_client=llm_client,
        **kwargs
    )
    return LocalMeteoCognitionPipeline(config)