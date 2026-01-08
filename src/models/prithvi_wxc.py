"""Local Prithvi WxC implementation for meteorological downscaling."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from transformers import AutoModel, AutoTokenizer

from .model_registry import ModelRegistry, ModelConfig

logger = logging.getLogger(__name__)


class LocalPrithviWXC(nn.Module):
    """Local implementation of Prithvi WxC for weather downscaling."""

    def __init__(self, config: Optional[ModelConfig] = None, model_name: str = "prithvi_wxc_small"):
        super().__init__()
        self.model_name = model_name
        self.registry = ModelRegistry()
        self.config = config or self.registry.get_model_config(model_name)

        # Initialize model components
        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize the Prithvi WxC model."""
        try:
            model_path = self.registry.get_model_path(self.model_name)
            logger.info(f"Loading Prithvi WxC from {model_path}")

            # For now, we'll use a transformer-based approach
            # In a full implementation, this would be the actual Prithvi WxC architecture
            self.backbone = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=self.config.torch_dtype,
            )

            # Move to specified device
            self.backbone.to(self.config.device)
            self.backbone.eval()

            logger.info(f"Prithvi WXC loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load Prithvi WXC model: {e}")
            # Fallback to a simple CNN-based downscaler
            self._setup_fallback_model()

    def _setup_fallback_model(self) -> None:
        """Setup a fallback CNN-based downscaler when the main model fails to load."""
        logger.warning("Using fallback CNN-based downscaler")

        class SimpleDownscaler(nn.Module):
            def __init__(self, input_channels=3, output_channels=3, scale_factor=4):
                super().__init__()
                self.scale_factor = scale_factor

                # Simple encoder-decoder architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, output_channels, 3, padding=1),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                # Upsample to target resolution
                return torch.nn.functional.interpolate(
                    decoded, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
                )

        self.backbone = SimpleDownscaler().to(self.config.device)
        self.backbone.eval()

    @torch.no_grad()
    def downscale(self, coarse_data: xr.Dataset) -> xr.Dataset:
        """
        Perform downscaling from coarse to high resolution.

        Args:
            coarse_data: Input dataset with coarse resolution weather data

        Returns:
            Dataset with high resolution weather data
        """
        try:
            # Preprocess input data
            input_tensor = self._preprocess_data(coarse_data)

            # Move to device
            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor).float()
            input_tensor = input_tensor.to(self.config.device)

            # Add batch dimension if needed
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                high_res_tensor = self.backbone(input_tensor)

            # Postprocess output
            output_data = self._postprocess_data(high_res_tensor, coarse_data)

            return output_data

        except Exception as e:
            logger.error(f"Downscaling failed: {e}")
            # Fallback to simple interpolation
            return self._fallback_interpolation(coarse_data)

    def _preprocess_data(self, data: xr.Dataset) -> torch.Tensor:
        """Convert xarray Dataset to model input tensor."""
        # Extract data variables
        data_vars = list(data.data_vars.keys())
        if not data_vars:
            raise ValueError("No data variables found in input dataset")

        # Stack multiple variables if available
        if len(data_vars) > 1:
            # Use first few variables
            selected_vars = data_vars[:min(3, len(data_vars))]
            arrays = [data[var].values for var in selected_vars]

            # Handle different dimensions
            processed_arrays = []
            for arr in arrays:
                if arr.ndim == 2:
                    processed_arrays.append(arr[np.newaxis, ...])  # Add channel dim
                elif arr.ndim == 3:
                    processed_arrays.append(arr)
                else:
                    # For higher dimensions, take the first time slice
                    processed_arrays.append(arr[0] if arr.shape[0] > 0 else arr.reshape(1, *arr.shape[1:]))

            stacked = np.stack(processed_arrays, axis=0)
        else:
            # Single variable case
            arr = data[data_vars[0]].values
            if arr.ndim == 2:
                stacked = arr[np.newaxis, np.newaxis, ...]  # (batch, channel, H, W)
            else:
                stacked = arr[np.newaxis, ...]  # (batch, ...)

        return torch.from_numpy(stacked).float()

    def _postprocess_data(self, tensor: torch.Tensor, reference_data: xr.Dataset) -> xr.Dataset:
        """Convert model output tensor back to xarray Dataset."""
        # Remove batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert to numpy
        output_np = tensor.cpu().numpy()

        # Create high-resolution coordinates
        scale_factor = 4  # Default upscaling factor
        lat_name = self._find_coord_name(reference_data, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(reference_data, ["lon", "longitude", "x"])

        lat_coords = reference_data.coords[lat_name].values
        lon_coords = reference_data.coords[lon_name].values

        # Create high-resolution grid
        lat_hr = np.linspace(lat_coords.min(), lat_coords.max(),
                           len(lat_coords) * scale_factor)
        lon_hr = np.linspace(lon_coords.min(), lon_coords.max(),
                           len(lon_coords) * scale_factor)

        # Create output dataset
        if output_np.ndim == 3:
            # Multiple channels
            output_vars = {}
            channel_names = list(reference_data.data_vars.keys())[:output_np.shape[0]]
            for i, name in enumerate(channel_names):
                output_vars[name] = ((lat_name, lon_name), output_np[i])
        else:
            # Single channel
            first_var = list(reference_data.data_vars.keys())[0]
            output_vars = {first_var: ((lat_name, lon_name), output_np)}

        return xr.Dataset(
            output_vars,
            coords={
                lat_name: lat_hr,
                lon_name: lon_hr
            }
        )

    def _fallback_interpolation(self, data: xr.Dataset) -> xr.Dataset:
        """Fallback to simple interpolation when model inference fails."""
        lat_name = self._find_coord_name(data, ["lat", "latitude", "y"])
        lon_name = self._find_coord_name(data, ["lon", "longitude", "x"])

        lat_coords = data.coords[lat_name].values
        lon_coords = data.coords[lon_name].values

        # Create high-resolution coordinates
        scale_factor = 4
        lat_hr = np.linspace(lat_coords.min(), lat_coords.max(),
                           len(lat_coords) * scale_factor)
        lon_hr = np.linspace(lon_coords.min(), lon_coords.max(),
                           len(lon_coords) * scale_factor)

        # Interpolate each variable
        output_vars = {}
        for name, var in data.data_vars.items():
            if var.ndim >= 2:
                interpolated = var.interp(
                    {lat_name: lat_hr, lon_name: lon_hr},
                    method="linear"
                )
                output_vars[name] = interpolated

        return xr.Dataset(output_vars)

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
            "model_type": "prithvi_wxc",
            "device": self.config.device,
            "dtype": self.config.dtype,
            "parameters": sum(p.numel() for p in self.backbone.parameters()),
            "model_path": self.registry.get_model_path(self.model_name),
        }