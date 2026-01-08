#!/usr/bin/env python3
"""
Basic demonstration of the meteorological cognition pipeline without AI model dependencies.
This shows the core workflow and fallback functionality.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr

from src.simulation.meteo_cognition import MeteoCognitionPipeline, MeteoCognitionConfig


def create_synthetic_weather_scenario():
    """Create a realistic synthetic weather scenario for testing."""
    print("Creating synthetic weather scenario...")

    # Create a grid around Beijing area
    lat = np.linspace(39.5, 40.5, 8)  # Coarse resolution (~12km)
    lon = np.linspace(116.0, 117.0, 8)

    # Create a synthetic precipitation system (typhoon-like structure)
    x, y = np.meshgrid(lon, lat)

    # Define storm center
    storm_center = (116.4, 39.95)
    distance = np.sqrt((x - storm_center[0])**2 + (y - storm_center[1])**2)

    # Create intensity pattern (Gaussian core with asymmetric distribution)
    core_intensity = 80 * np.exp(-distance**2 / 0.02)
    asymmetric_factor = 1 + 0.3 * np.sin(4 * np.arctan2(y - storm_center[1], x - storm_center[0]))
    total_precip = core_intensity * asymmetric_factor

    # Add noise and ensure non-negative
    total_precip = np.maximum(total_precip + np.random.normal(0, 5, x.shape), 0)

    # Create additional weather variables
    wind_speed = 25 + 15 * np.exp(-distance**2 / 0.03) + np.random.normal(0, 3, x.shape)
    wind_speed = np.maximum(wind_speed, 0)

    pressure = 1010 - 12 * np.exp(-distance**2 / 0.015) + np.random.normal(0, 1, x.shape)

    # Create dataset
    dataset = xr.Dataset(
        {
            "precip": (("lat", "lon"), total_precip),
            "wind_speed": (("lat", "lon"), wind_speed),
            "pressure": (("lat", "lon"), pressure),
        },
        coords={"lat": lat, "lon": lon}
    )

    print(f"  Dataset shape: {dataset.sizes}")
    print(f"  Precipitation range: {total_precip.min():.1f} - {total_precip.max():.1f} mm")
    print(f"  Max intensity location: lat={lat[np.unravel_index(total_precip.argmax(), total_precip.shape)[0]]:.2f}, "
          f"lon={lon[np.unravel_index(total_precip.argmax(), total_precip.shape)[1]]:.2f}")

    return dataset


def create_simple_downscaler():
    """Create a simple interpolation-based downscaler for testing."""
    def simple_downscaler(coarse_data):
        """Simple downscaling using bilinear interpolation."""
        dataset = coarse_data
        if isinstance(dataset, dict):
            # Convert dict to xarray
            lat = np.array(dataset["lat"])
            lon = np.array(dataset["lon"])
            values = np.array(dataset["values"])
            dataset = xr.Dataset(
                {"field": (("lat", "lon"), values)},
                coords={"lat": lat, "lon": lon}
            )

        # Upscale by 4x using interpolation
        scale_factor = 4
        lat_name = "lat"
        lon_name = "lon"

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

        # Interpolate all variables
        output_vars = {}
        for name, var in dataset.data_vars.items():
            if var.ndim >= 2:
                interpolated = var.interp(
                    {lat_name: lat_hr, lon_name: lon_hr},
                    method="linear"
                )
                output_vars[name] = interpolated

        return xr.Dataset(output_vars)

    return simple_downscaler


def create_simple_segmenter():
    """Create a simple threshold-based segmenter for testing."""
    def simple_segmenter(hi_res_data):
        """Simple segmentation using percentile thresholding."""
        dataset = hi_res_data

        # Use the first data variable
        first_var = list(dataset.data_vars.keys())[0]
        field = dataset[first_var]

        # Handle time dimension
        if field.ndim > 2:
            field = field.isel({dim: 0 for dim in field.dims[:-2]})

        # Threshold-based segmentation
        values = field.values
        threshold = np.nanpercentile(values, 85.0)
        mask = values >= threshold

        # Convert to H3 cells
        h3_cells = []
        try:
            import h3
            lat_name = "lat"
            lon_name = "lon"
            lats = dataset.coords[lat_name].values
            lons = dataset.coords[lon_name].values

            # Sample points from mask
            mask_points = np.where(mask)
            max_samples = 500  # Limit for performance
            step = max(1, len(mask_points[0]) // max_samples)

            for i in range(0, len(mask_points[0]), step):
                lat_idx = mask_points[0][i]
                lon_idx = mask_points[1][i]

                if lat_idx < len(lats) and lon_idx < len(lons):
                    lat = float(lats[lat_idx])
                    lon = float(lons[lon_idx])
                    try:
                        h3_cell = h3.latlng_to_cell(lat, lon, 8)
                        h3_cells.append(h3_cell)
                    except Exception:
                        continue
        except ImportError:
            print("  Warning: H3 not available, using empty H3 coverage")
            h3_cells = []

        stats = {
            "max_intensity": float(np.nanmax(values)),
            "mean_intensity": float(np.nanmean(values)),
            "threshold": float(threshold),
            "mask_fraction": float(mask.mean()),
            "coverage_cells": len(h3_cells),
        }

        return {
            "stats": stats,
            "h3_coverage": sorted(set(h3_cells)),
            "method": "simple_thresholding"
        }

    return simple_segmenter


def demo_basic_pipeline():
    """Demonstrate the basic meteorological cognition pipeline."""
    print("=" * 60)
    print("SEIA-MOD BASIC METEOROLOGICAL COGNITION DEMO")
    print("=" * 60)

    # Create synthetic weather data
    weather_data = create_synthetic_weather_scenario()

    # Create simple processing functions
    downscaler = create_simple_downscaler()
    segmenter = create_simple_segmenter()

    # Mock LLM client for demonstration
    class MockLLMClient:
        def generate(self, prompt, payload=None):
            # Simple event generation based on stats
            stats = payload.get("stats", {})
            max_intensity = stats.get("max_intensity", 0)
            mask_fraction = stats.get("mask_fraction", 0)

            if max_intensity > 70:
                event_type = "severe_storm"
                intensity_index = min(1.0, max_intensity / 100)
            elif max_intensity > 40:
                event_type = "moderate_storm"
                intensity_index = min(0.8, max_intensity / 80)
            else:
                event_type = "light_precipitation"
                intensity_index = min(0.5, max_intensity / 60)

            return {
                "json": {
                    "type": event_type,
                    "intensity_index": intensity_index,
                    "h3_coverage": payload.get("h3_coverage", []),
                    "metadata": {
                        "max_intensity": max_intensity,
                        "coverage_fraction": mask_fraction,
                        "threshold": stats.get("threshold", 0)
                    }
                }
            }

    mock_llm = MockLLMClient()

    # Create pipeline configuration
    config = MeteoCognitionConfig(
        downscaler=downscaler,
        segmenter=segmenter,
        llm_client=mock_llm,
        llm_prompt="Generate a weather event description based on the provided statistics.",
        default_event_type="storm",
        h3_fallback_resolution=8
    )

    # Create and run pipeline
    print(f"\n1. Pipeline Configuration")
    print("-" * 30)
    print("  Downscaler: Simple bilinear interpolation (4x)")
    print("  Segmenter: Percentile thresholding (85th percentile)")
    print("  LLM Client: Mock event generator")

    print(f"\n2. Running Meteorological Cognition Pipeline")
    print("-" * 55)
    pipeline = MeteoCognitionPipeline(config)

    # Step 1: Downscaling
    print("  Step 1: Downscaling coarse weather data...")
    start_time = time.time()
    hi_res_data = pipeline.downscale(weather_data)
    downscale_time = time.time() - start_time

    # Check downscaling results
    first_var = list(hi_res_data.data_vars.keys())[0]
    input_shape = weather_data[first_var].shape
    output_shape = hi_res_data[first_var].shape
    upscale_factor = output_shape[0] / input_shape[0]

    print(f"    ✅ Completed in {downscale_time:.3f}s")
    print(f"    Input shape: {input_shape}")
    print(f"    Output shape: {output_shape}")
    print(f"    Upscale factor: {upscale_factor:.1f}x")

    # Step 2: Segmentation
    print("  Step 2: Performing segmentation...")
    start_time = time.time()
    segmentation = pipeline.segment(hi_res_data)
    segment_time = time.time() - start_time

    stats = segmentation.get("stats", {})
    h3_cells = len(segmentation.get("h3_coverage", []))

    print(f"    ✅ Completed in {segment_time:.3f}s")
    print(f"    Max intensity: {stats.get('max_intensity', 0):.2f}")
    print(f"    Threshold: {stats.get('threshold', 0):.2f}")
    print(f"    Coverage fraction: {stats.get('mask_fraction', 0):.3f}")
    print(f"    H3 cells: {h3_cells}")

    # Step 3: Event description
    print("  Step 3: Generating event description...")
    start_time = time.time()
    event = pipeline.describe_event(segmentation)
    event_time = time.time() - start_time

    print(f"    ✅ Completed in {event_time:.3f}s")
    print(f"    Event type: {event.type}")
    print(f"    Intensity index: {event.intensity_index:.3f}")
    print(f"    H3 coverage: {len(event.h3_coverage)} cells")

    # Full pipeline test
    print(f"\n3. Full Pipeline Test")
    print("-" * 25)
    start_time = time.time()
    full_event = pipeline.run(weather_data)
    total_time = time.time() - start_time

    print(f"  ✅ Full pipeline completed in {total_time:.3f}s")
    print(f"  Event type: {full_event.type}")
    print(f"  Intensity: {full_event.intensity_index:.3f}")
    print(f"  H3 cells: {len(full_event.h3_coverage)}")

    # Show metadata
    metadata = full_event.metadata
    print(f"\n4. Event Metadata")
    print("-" * 18)
    for key, value in metadata.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Performance summary
    print(f"\n5. Performance Summary")
    print("-" * 22)
    print(f"  Downscaling: {downscale_time:.3f}s")
    print(f"  Segmentation: {segment_time:.3f}s")
    print(f"  Event generation: {event_time:.3f}s")
    print(f"  Total pipeline: {total_time:.3f}s")

    print(f"\n6. Comparison Table")
    print("-" * 21)
    print(f"{'Stage':<15} {'Time (s)':<10} {'Relative':<10}")
    print("-" * 35)
    print(f"{'Downscaling':<15} {downscale_time:<10.3f} {downscale_time/total_time:<10.1%}")
    print(f"{'Segmentation':<15} {segment_time:<10.3f} {segment_time/total_time:<10.1%}")
    print(f"{'Event Gen':<15} {event_time:<10.3f} {event_time/total_time:<10.1%}")
    print(f"{'Total':<15} {total_time:<10.3f} {100.0:<10.1%}")

    print(f"\n{'='*60}")
    print("DEMO COMPLETED SUCCESSFULLY")
    print("This demonstrates the core pipeline functionality without AI models.")
    print("Local AI models will enhance each stage with more sophisticated processing.")
    print("="*60)


if __name__ == "__main__":
    demo_basic_pipeline()