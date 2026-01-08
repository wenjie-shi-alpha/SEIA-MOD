"""Tests for local AI model integration."""

import numpy as np
import pytest
import xarray as xr

from src.models import LocalPrithviWXC, LocalSAMGeo, ModelRegistry
from src.simulation.local_meteo_cognition import (
    LocalMeteoCognitionConfig,
    LocalMeteoCognitionPipeline,
    create_local_pipeline,
)


class TestModelRegistry:
    """Test the model registry functionality."""

    def test_list_available_models(self):
        """Test listing available models."""
        registry = ModelRegistry()
        models = registry.list_available_models()

        assert isinstance(models, dict)
        assert len(models) > 0

        # Check that expected models are present
        expected_models = ["prithvi_wxc_small", "sam_geo_base"]
        for model_name in expected_models:
            assert model_name in models

    def test_get_model_config(self):
        """Test getting model configuration."""
        registry = ModelRegistry()
        config = registry.get_model_config("prithvi_wxc_small")

        assert config.name == "prithvi_wxc_small"
        assert config.model_type == "prithvi_wxc"
        assert config.device in ["cuda", "cpu"]


@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not available")
class TestLocalPrithviWXC:
    """Test the local Prithvi WXC implementation."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = LocalPrithviWXC(model_name="prithvi_wxc_small")
        assert model is not None
        assert hasattr(model, 'backbone')

    def test_downscaling_simple(self):
        """Test basic downscaling functionality."""
        model = LocalPrithviWXC(model_name="prithvi_wxc_small")

        # Create synthetic input data
        lat = np.linspace(39.7, 40.1, 8)
        lon = np.linspace(116.1, 116.5, 8)
        precip = np.random.rand(8, 8) * 10

        input_data = xr.Dataset(
            {"precip": (("lat", "lon"), precip)},
            coords={"lat": lat, "lon": lon}
        )

        # Perform downscaling
        result = model.downscale(input_data)

        assert isinstance(result, xr.Dataset)
        assert "precip" in result.data_vars

        # Check that resolution has increased
        assert result.dims["lat"] > input_data.dims["lat"]
        assert result.dims["lon"] > input_data.dims["lon"]

    def test_model_info(self):
        """Test getting model information."""
        model = LocalPrithviWXC(model_name="prithvi_wxc_small")
        info = model.get_model_info()

        assert "model_name" in info
        assert "model_type" in info
        assert info["model_type"] == "prithvi_wxc"


@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not available")
class TestLocalSAMGeo:
    """Test the local SAM-Geo implementation."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = LocalSAMGeo(model_name="sam_geo_base")
        assert model is not None

    def test_segmentation_simple(self):
        """Test basic segmentation functionality."""
        model = LocalSAMGeo(model_name="sam_geo_base")

        # Create synthetic input data
        lat = np.linspace(39.7, 40.1, 16)
        lon = np.linspace(116.1, 116.5, 16)
        intensity = np.random.rand(16, 16)

        input_data = xr.Dataset(
            {"intensity": (("lat", "lon"), intensity)},
            coords={"lat": lat, "lon": lon}
        )

        # Perform segmentation
        result = model.segment_meteorological_features(
            data=input_data,
            percentile=75.0,
            h3_resolution=8
        )

        assert isinstance(result, dict)
        assert "stats" in result
        assert "h3_coverage" in result

        # Check stats
        stats = result["stats"]
        assert "max_intensity" in stats
        assert "mean_intensity" in stats
        assert "mask_fraction" in stats

    def test_model_info(self):
        """Test getting model information."""
        model = LocalSAMGeo(model_name="sam_geo_base")
        info = model.get_model_info()

        assert "model_name" in info
        assert "model_type" in info
        assert info["model_type"] == "sam_geo"


@pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not available")
class TestLocalMeteoCognitionPipeline:
    """Test the local meteorological cognition pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        config = LocalMeteoCognitionConfig(
            use_local_models=True,
            prithvi_model="prithvi_wxc_small",
            sam_model="sam_geo_base"
        )

        pipeline = LocalMeteoCognitionPipeline(config)
        assert pipeline is not None
        assert hasattr(pipeline, 'config')

    def test_convenience_function(self):
        """Test the convenience function for pipeline creation."""
        pipeline = create_local_pipeline(
            prithvi_model="prithvi_wxc_small",
            sam_model="sam_geo_base"
        )
        assert pipeline is not None

    def test_full_pipeline_simple(self):
        """Test the full pipeline with simple data."""
        pipeline = create_local_pipeline(
            prithvi_model="prithvi_wxc_small",
            sam_model="sam_geo_base"
        )

        # Create synthetic coarse weather data
        lat = np.linspace(39.7, 40.1, 4)
        lon = np.linspace(116.1, 116.5, 4)
        precip = np.random.rand(4, 4) * 20

        input_data = {
            "lat": lat,
            "lon": lon,
            "values": precip,
            "dims": ("lat", "lon")
        }

        # Run the full pipeline
        event = pipeline.run(input_data)

        assert isinstance(event, dict)
        assert "type" in event
        assert "h3_coverage" in event
        assert "intensity_index" in event
        assert "metadata" in event

    def test_pipeline_model_status(self):
        """Test getting pipeline model status."""
        pipeline = create_local_pipeline()
        status = pipeline.get_model_status()

        assert isinstance(status, dict)
        assert "local_models_enabled" in status
        assert "prithvi_model" in status
        assert "sam_model" in status

    def test_pipeline_fallback_mode(self):
        """Test pipeline in fallback mode."""
        config = LocalMeteoCognitionConfig(
            use_local_models=False  # Force fallback mode
        )
        pipeline = LocalMeteoCognitionPipeline(config)

        # Create simple input
        input_data = {
            "lat": [39.8, 39.9, 40.0],
            "lon": [116.2, 116.3, 116.4],
            "values": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "dims": ("lat", "lon")
        }

        # Should work even without local models
        event = pipeline.run(input_data)
        assert isinstance(event, dict)
        assert "type" in event


def test_synthetic_data():
    """Test synthetic data creation for testing."""
    # Create a realistic synthetic weather dataset
    lat = np.linspace(39.5, 40.5, 32)  # ~1km resolution
    lon = np.linspace(116.0, 117.0, 32)

    # Create a synthetic storm pattern
    x, y = np.meshgrid(lon, lat)
    storm_center = (116.5, 40.0)
    distance = np.sqrt((x - storm_center[0])**2 + (y - storm_center[1])**2)

    # Intensity decreases with distance from storm center
    intensity = 50 * np.exp(-distance**2 / 0.1) + np.random.normal(0, 2, x.shape)

    dataset = xr.Dataset(
        {"precip": (("lat", "lon"), intensity)},
        coords={"lat": lat, "lon": lon}
    )

    assert isinstance(dataset, xr.Dataset)
    assert "precip" in dataset.data_vars
    assert dataset.precip.max() > 0

    return dataset


if __name__ == "__main__":
    pytest.main([__file__])