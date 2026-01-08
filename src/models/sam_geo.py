"""Local SAM-Geo implementation for geospatial segmentation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import h3
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from segment_anything import sam_model_registry, SamPredictor

from .model_registry import ModelRegistry, ModelConfig

logger = logging.getLogger(__name__)


class LocalSAMGeo:
    """Local implementation of SAM-Geo for geospatial feature segmentation."""

    def __init__(self, config: Optional[ModelConfig] = None, model_name: str = "sam_geo_base"):
        self.model_name = model_name
        self.registry = ModelRegistry()
        self.config = config or self.registry.get_model_config(model_name)

        # Initialize SAM model
        self._setup_sam_model()

    def _setup_sam_model(self) -> None:
        """Initialize the SAM model."""
        try:
            model_path = self.registry.get_model_path(self.model_name)
            logger.info(f"Loading SAM model from {model_path}")

            # Map model names to SAM checkpoint types
            sam_checkpoint_map = {
                "sam_geo_huge": "sam_vit_h_4b8939.pth",
                "sam_geo_large": "sam_vit_l_0b3195.pth",
                "sam_geo_base": "sam_vit_b_01ec64.pth"
            }

            checkpoint_name = sam_checkpoint_map.get(self.model_name, "sam_vit_b_01ec64.pth")
            checkpoint_path = f"{model_path}/{checkpoint_name}"

            # Register and load SAM model
            model_type = self.model_name.split("_")[-1]  # Extract 'base', 'large', or 'huge'
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.config.device)

            self.predictor = SamPredictor(sam)
            logger.info(f"SAM model loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            # Fallback to simple thresholding segmentation
            self._setup_fallback_segmenter()

    def _setup_fallback_segmenter(self) -> None:
        """Setup a fallback segmentation method when SAM fails to load."""
        logger.warning("Using fallback thresholding segmentation")
        self.predictor = None

    def segment_meteorological_features(
        self,
        data: xr.Dataset,
        intensity_field: Optional[str] = None,
        threshold: Optional[float] = None,
        percentile: float = 85.0,
        h3_resolution: int = 8,
        prompts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Segment meteorological features from weather data.

        Args:
            data: Input weather dataset
            intensity_field: Field name to use for segmentation
            threshold: Manual threshold for segmentation
            percentile: Percentile threshold if threshold is None
            h3_resolution: H3 grid resolution
            prompts: List of prompts for SAM (points, boxes, etc.)

        Returns:
            Dictionary with segmentation results and H3 coverage
        """
        try:
            if self.predictor:
                return self._sam_segment(data, intensity_field, threshold, percentile, h3_resolution, prompts)
            else:
                return self._threshold_segment(data, intensity_field, threshold, percentile, h3_resolution)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return self._fallback_segment(data, h3_resolution)

    def _sam_segment(
        self,
        data: xr.Dataset,
        intensity_field: Optional[str],
        threshold: Optional[float],
        percentile: float,
        h3_resolution: int,
        prompts: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Perform SAM-based segmentation."""
        # Extract intensity field
        field = self._extract_intensity_field(data, intensity_field)

        # Convert to image format
        image = self._to_image_format(field)

        # Set image for SAM predictor
        self.predictor.set_image(image)

        # Generate prompts if not provided
        if not prompts:
            prompts = self._generate_auto_prompts(field, percentile)

        # Perform segmentation
        masks, scores, logits = self._run_sam_inference(prompts)

        # Process masks
        combined_mask = self._combine_masks(masks, scores)

        # Convert to H3 cells
        h3_cells = self._mask_to_h3(combined_mask, data, h3_resolution)

        # Calculate statistics
        stats = self._calculate_segmentation_stats(field, combined_mask)

        return {
            "stats": stats,
            "h3_coverage": sorted(h3_cells),
            "masks": [mask.cpu().numpy().tolist() for mask in masks] if masks else [],
            "scores": scores,
            "bounds": self._get_mask_bounds(combined_mask, data),
            "method": "sam_segmentation"
        }

    def _threshold_segment(
        self,
        data: xr.Dataset,
        intensity_field: Optional[str],
        threshold: Optional[float],
        percentile: float,
        h3_resolution: int
    ) -> Dict[str, Any]:
        """Perform threshold-based segmentation."""
        # Extract intensity field
        field = self._extract_intensity_field(data, intensity_field)

        # Determine threshold
        if threshold is None:
            threshold = np.nanpercentile(field.values, percentile)

        # Create binary mask
        mask = field.values >= threshold
        mask = np.nan_to_num(mask, nan=False).astype(bool)

        # Convert to H3 cells
        h3_cells = self._mask_to_h3(mask, data, h3_resolution)

        # Calculate statistics
        stats = self._calculate_segmentation_stats(field, mask)

        return {
            "stats": stats,
            "h3_coverage": sorted(h3_cells),
            "mask": mask.tolist(),
            "bounds": self._get_mask_bounds(mask, data),
            "method": "threshold_segmentation",
            "threshold": float(threshold)
        }

    def _extract_intensity_field(self, data: xr.Dataset, intensity_field: Optional[str]) -> xr.DataArray:
        """Extract the intensity field for segmentation."""
        if intensity_field and intensity_field in data:
            field = data[intensity_field]
        else:
            # Use the first data variable
            first_var = list(data.data_vars.keys())[0]
            field = data[first_var]

        # Handle time dimension if present
        if field.ndim > 2:
            field = field.isel({dim: 0 for dim in field.dims[:-2]})

        return field

    def _to_image_format(self, field: xr.DataArray) -> np.ndarray:
        """Convert field to image format for SAM."""
        values = field.values

        # Normalize to 0-255 range
        values = np.nan_to_num(values, nan=0.0)
        if values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min()) * 255

        # Convert to uint8
        image = values.astype(np.uint8)

        # Convert to RGB if needed (SAM expects 3-channel)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        return image

    def _generate_auto_prompts(self, field: xr.DataArray, percentile: float) -> List[Dict[str, Any]]:
        """Generate automatic prompts based on field intensity."""
        values = field.values
        threshold = np.nanpercentile(values, percentile)

        # Find high-intensity regions
        high_intensity_mask = values >= threshold
        if not high_intensity_mask.any():
            # Fallback to global maximum
            max_idx = np.unravel_index(np.nanargmax(values), values.shape)
            return [{"point": max_idx[::-1], "label": 1}]  # Reverse for (x, y) format

        # Get centroids of high-intensity regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(high_intensity_mask)

        prompts = []
        for i in range(1, min(num_features + 1, 6)):  # Limit to 5 prompts
            centroid = ndimage.center_of_mass(labeled == i)
            if not np.isnan(centroid).any():
                y, x = centroid
                prompts.append({"point": [int(x), int(y)], "label": 1})

        return prompts

    def _run_sam_inference(self, prompts: List[Dict[str, Any]]) -> Tuple:
        """Run SAM inference with prompts."""
        if not prompts:
            return [], [], []

        all_masks = []
        all_scores = []
        all_logits = []

        for prompt in prompts:
            point_coords = None
            point_labels = None
            boxes = None

            if "point" in prompt:
                point_coords = [prompt["point"]]
                point_labels = [prompt.get("label", 1)]

            if "box" in prompt:
                boxes = [prompt["box"]]

            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=boxes,
                multimask_output=True  # Get multiple mask predictions
            )

            all_masks.extend(masks)
            all_scores.extend(scores)
            all_logits.extend(logits)

        return all_masks, all_scores, all_logits

    def _combine_masks(self, masks: List, scores: List[float]) -> np.ndarray:
        """Combine multiple masks into a single mask."""
        if not masks:
            return np.zeros((256, 256), dtype=bool)  # Default size

        # Use the mask with highest score
        best_idx = np.argmax(scores)
        combined_mask = masks[best_idx].cpu().numpy() if hasattr(masks[best_idx], 'cpu') else masks[best_idx]

        return combined_mask.astype(bool)

    def _mask_to_h3(self, mask: np.ndarray, data: xr.Dataset, h3_resolution: int) -> List[str]:
        """Convert binary mask to H3 cell coverage."""
        # Get coordinate information
        lat_name = self._find_coord_name(data, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(data, ["lon", "longitude", "x"])

        lats = data.coords[lat_name].values
        lons = data.coords[lon_name].values

        # Create coordinate grids
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        # Sample points from mask
        mask_points = np.where(mask)

        if len(mask_points[0]) == 0:
            return []

        # Convert mask points to H3 cells
        h3_cells = set()
        for i in range(0, len(mask_points[0]), max(1, len(mask_points[0]) // 1000)):  # Sample points
            lat_idx = mask_points[0][i]
            lon_idx = mask_points[1][i]

            if lat_idx < len(lats) and lon_idx < len(lons):
                lat = float(lat_grid[lat_idx, lon_idx])
                lon = float(lon_grid[lat_idx, lon_idx])
                h3_cell = h3.latlng_to_cell(lat, lon, h3_resolution)
                h3_cells.add(h3_cell)

        return list(h3_cells)

    def _calculate_segmentation_stats(self, field: xr.DataArray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate segmentation statistics."""
        values = field.values
        mask_float = mask.astype(float)

        stats = {
            "max_intensity": float(np.nanmax(values)),
            "mean_intensity": float(np.nanmean(values)),
            "mask_fraction": float(mask_float.mean()),
            "masked_mean": float(np.nanmean(values[mask])) if mask.any() else 0.0,
            "coverage_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
        }

        if mask.any():
            stats["masked_max"] = float(np.nanmax(values[mask]))
            stats["masked_std"] = float(np.nanstd(values[mask]))

        return stats

    def _get_mask_bounds(self, mask: np.ndarray, data: xr.Dataset) -> Dict[str, float]:
        """Get geographical bounds of the mask."""
        lat_name = self._find_coord_name(data, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(data, ["lon", "longitude", "x"])

        lats = data.coords[lat_name].values
        lons = data.coords[lon_name].values

        if not mask.any():
            return {
                "lat_min": float(lats.min()),
                "lat_max": float(lats.max()),
                "lon_min": float(lons.min()),
                "lon_max": float(lons.max()),
            }

        # Get bounding box of mask
        mask_indices = np.where(mask)
        lat_indices = mask_indices[0]
        lon_indices = mask_indices[1]

        return {
            "lat_min": float(lats[lat_indices.min()]),
            "lat_max": float(lats[lat_indices.max()]),
            "lon_min": float(lons[lon_indices.min()]),
            "lon_max": float(lons[lon_indices.max()]),
        }

    def _fallback_segment(self, data: xr.Dataset, h3_resolution: int) -> Dict[str, Any]:
        """Fallback segmentation when all methods fail."""
        field = self._extract_intensity_field(data, None)
        values = field.values

        # Use simple threshold at 75th percentile
        threshold = np.nanpercentile(values, 75)
        mask = values >= threshold

        h3_cells = self._mask_to_h3(mask, data, h3_resolution)
        stats = self._calculate_segmentation_stats(field, mask)

        return {
            "stats": stats,
            "h3_coverage": sorted(h3_cells),
            "method": "fallback_segmentation",
            "threshold": float(threshold)
        }

    @staticmethod
    def _find_coord_name(dataset: xr.Dataset, candidates: list) -> str:
        """Find coordinate name from candidates."""
        for candidate in candidates:
            if candidate in dataset.coords:
                return candidate
        raise ValueError(f"None of {candidates} found in dataset coordinates")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": "sam_geo",
            "device": self.config.device,
            "dtype": self.config.dtype,
            "has_sam": self.predictor is not None,
            "model_path": self.registry.get_model_path(self.model_name),
        }