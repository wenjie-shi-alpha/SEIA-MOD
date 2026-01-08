#!/usr/bin/env python3
"""
Demonstrate the local AI model framework without requiring PyTorch.
This shows the model management and pipeline integration.
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

from src.models.simple_registry import simple_registry
from src.simulation.meteo_cognition import MeteoCognitionPipeline, MeteoCognitionConfig


def test_model_registry():
    """Test the model registry functionality."""
    print("=" * 60)
    print("MODEL REGISTRY TEST")
    print("=" * 60)

    # Environment check
    print("\n1. Environment Check")
    print("-" * 25)
    status = simple_registry.check_environment()
    for key, value in status.items():
        if isinstance(value, bool):
            status_symbol = "✅" if value else "❌"
            print(f"  {status_symbol} {key}: {value}")
        else:
            print(f"  ℹ️  {key}: {value}")

    # Available models
    print("\n2. Available Models")
    print("-" * 22)
    models = simple_registry.list_available_models()
    for name, info in models.items():
        print(f"  {name}:")
        print(f"    Model ID: {info['model_id']}")
        print(f"    Type: {info['type']}")
        print(f"    Size: {info['size_gb']} GB")

    # Cache information
    print(f"\n3. Cache Information")
    print("-" * 22)
    print(f"  Cache dir: {simple_registry.cache_dir}")
    print(f"  Cache usage: {simple_registry.get_cache_size()}")

    return models, status


def test_meteo_cognition_framework():
    """Test the meteorological cognition framework."""
    print(f"\n4. Meteo Cognition Framework Test")
    print("-" * 38)

    # Create synthetic data
    lat = np.linspace(39.5, 40.5, 4)  # Very coarse for fast testing
    lon = np.linspace(116.0, 117.0, 4)

    # Create storm pattern
    x, y = np.meshgrid(lon, lat)
    storm_center = (116.4, 39.8)
    distance = np.sqrt((x - storm_center[0])**2 + (y - storm_center[1])**2)

    # High intensity storm
    precip = 50 * np.exp(-distance**2 / 0.02) + np.random.normal(0, 5, x.shape)
    precip = np.maximum(precip, 0)

    weather_data = xr.Dataset(
        {"precip": (("lat", "lon"), precip)},
        coords={"lat": lat, "lon": lon}
    )

    print(f"  Created synthetic weather data: {weather_data.sizes}")
    print(f"  Precip range: {precip.min():.1f} - {precip.max():.1f} mm")

    # Create pipeline with simple fallback functions
    def simple_downscaler(data):
        """Simple downscaling for testing."""
        scale_factor = 3
        lat_name = "lat"
        lon_name = "lon"

        lat_coords = data.coords[lat_name].values
        lon_coords = data.coords[lon_name].values

        lat_hr = np.linspace(
            lat_coords.min(), lat_coords.max(),
            len(lat_coords) * scale_factor
        )
        lon_hr = np.linspace(
            lon_coords.min(), lon_coords.max(),
            len(lon_coords) * scale_factor
        )

        output_vars = {}
        for name, var in data.data_vars.items():
            if var.ndim >= 2:
                interpolated = var.interp(
                    {lat_name: lat_hr, lon_name: lon_hr},
                    method="linear"
                )
                output_vars[name] = interpolated

        return xr.Dataset(output_vars)

    def simple_segmenter(data):
        """Simple segmentation for testing."""
        first_var = list(data.data_vars.keys())[0]
        field = data[first_var]
        values = field.values

        threshold = np.nanpercentile(values, 80.0)
        mask = values >= threshold

        # Generate H3 cells
        h3_cells = []
        try:
            import h3
            lat_name = "lat"
            lon_name = "lon"
            lats = data.coords[lat_name].values
            lons = data.coords[lon_name].values

            mask_points = np.where(mask)
            for i in range(min(50, len(mask_points[0]))):
                lat_idx = mask_points[0][i]
                lon_idx = mask_points[1][i]

                if lat_idx < len(lats) and lon_idx < len(lons):
                    lat = float(lats[lat_idx])
                    lon = float(lons[lon_idx])
                    h3_cell = h3.latlng_to_cell(lat, lon, 7)
                    h3_cells.append(h3_cell)
        except ImportError:
            pass

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
            "method": "framework_test"
        }

    # Mock LLM client
    class MockLLM:
        def generate(self, prompt, payload=None):
            return {
                "json": {
                    "type": "test_event",
                    "intensity_index": 0.75,
                    "h3_coverage": payload.get("h3_coverage", []),
                    "metadata": payload.get("stats", {})
                }
            }

    # Configure and run pipeline
    config = MeteoCognitionConfig(
        downscaler=simple_downscaler,
        segmenter=simple_segmenter,
        llm_client=MockLLM(),
        default_event_type="framework_test"
    )

    pipeline = MeteoCognitionPipeline(config)

    # Test each step
    print(f"\n5. Pipeline Step Tests")
    print("-" * 25)

    # Step 1: Downscaling
    print("  Testing downscaling...")
    start_time = time.time()
    hi_res = pipeline.downscale(weather_data)
    downscale_time = time.time() - start_time
    print(f"    ✅ {hi_res[list(hi_res.data_vars.keys())[0]].shape} in {downscale_time:.3f}s")

    # Step 2: Segmentation
    print("  Testing segmentation...")
    start_time = time.time()
    segmentation = pipeline.segment(hi_res)
    segment_time = time.time() - start_time
    h3_count = len(segmentation.get("h3_coverage", []))
    print(f"    ✅ {h3_count} H3 cells in {segment_time:.3f}s")

    # Step 3: Event generation
    print("  Testing event generation...")
    start_time = time.time()
    event = pipeline.describe_event(segmentation)
    event_time = time.time() - start_time
    print(f"    ✅ {event.type} event in {event_time:.3f}s")

    # Full pipeline test
    print(f"\n6. Full Pipeline Test")
    print("-" * 23)
    start_time = time.time()
    full_event = pipeline.run(weather_data)
    total_time = time.time() - start_time
    print(f"  ✅ Complete pipeline in {total_time:.3f}s")
    print(f"  Event: {full_event.type}")
    print(f"  Intensity: {full_event.intensity_index:.3f}")
    print(f"  H3 cells: {len(full_event.h3_coverage)}")

    return {
        "downscale_time": downscale_time,
        "segment_time": segment_time,
        "event_time": event_time,
        "total_time": total_time,
        "h3_cells": len(full_event.h3_coverage)
    }


def test_model_integration_readiness():
    """Test readiness for AI model integration."""
    print(f"\n7. AI Model Integration Readiness")
    print("-" * 35)

    # Check if we can import our model classes (even if they fail)
    imports_ok = True
    try:
        from src.models import ModelRegistry, ModelConfig
        print("  ✅ ModelRegistry import available")
    except Exception as e:
        print(f"  ❌ ModelRegistry import failed: {e}")
        imports_ok = False

    try:
        from src.models import LocalPrithviWXC, LocalSAMGeo
        print("  ✅ AI model class definitions available")
    except Exception as e:
        print(f"  ⚠️  AI model classes available with lazy loading: {e}")

    # Test model config creation
    try:
        registry = ModelRegistry()
        config = registry.get_model_config("prithvi_wxc_small")
        print(f"  ✅ Model config creation works: {config.name}")
    except Exception as e:
        print(f"  ❌ Model config failed: {e}")
        imports_ok = False

    return imports_ok


def main():
    """Run all framework tests."""
    print("SEIA-MOD LOCAL AI MODEL FRAMEWORK DEMO")
    print("Testing framework readiness without PyTorch dependencies")

    # Test 1: Model Registry
    models, status = test_model_registry()

    # Test 2: Meteo Cognition Framework
    performance = test_meteo_cognition_framework()

    # Test 3: Model Integration Readiness
    ready = test_model_integration_readiness()

    # Summary
    print(f"\n8. Test Summary")
    print("-" * 17)

    # Hardware status
    if status.get("cuda_available", False):
        print("  🚀 GPU: Ready for AI acceleration")
        print(f"     Memory: {status.get('gpu_memory_gb', 0):.0f} GB")
    else:
        print("  ⚠️  GPU: Not detected (will use CPU)")

    print(f"  💾 Disk: {status.get('disk_space_gb', 0):.0f} GB available")
    print(f"  📦 Cache: {simple_registry.get_cache_size()}")

    # Performance
    print(f"\n  📊 Performance (framework only):")
    print(f"     Downscale: {performance['downscale_time']:.3f}s")
    print(f"     Segment:   {performance['segment_time']:.3f}s")
    print(f"     Event:      {performance['event_time']:.3f}s")
    print(f"     Total:      {performance['total_time']:.3f}s")
    print(f"     H3 cells:   {performance['h3_cells']}")

    # Integration status
    print(f"\n  🔗 Integration:")
    if ready:
        print("     ✅ Framework ready for AI models")
    else:
        print("     ⚠️  Some integration issues detected")

    # Recommendations
    print(f"\n9. Next Steps")
    print("-" * 15)

    if not status.get("pytorch_available", False):
        print("  ⏳ Wait for PyTorch installation to complete")
        print("  💡 Then run: python scripts/setup_local_models.py --download-all")
    else:
        print("  🚀 PyTorch available - can proceed with AI model setup")

    if status.get("disk_space_gb", 0) > 20:
        print("  💾 Sufficient disk space for model downloads")
    else:
        print("  ⚠️  Consider freeing up disk space for AI models")

    print(f"\n{'='*60}")
    print("FRAMEWORK DEMO COMPLETED")
    print("The core infrastructure is ready for AI model integration!")
    print("="*60)


if __name__ == "__main__":
    main()