#!/usr/bin/env python3
"""
Complete demonstration of local AI model integration for meteorological cognition.
This script showcases the full pipeline with real AI models (when available).
"""

import sys
import time
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr


def check_environment():
    """Check if all required dependencies are available."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    deps_status = {}

    # Check PyTorch
    try:
        import torch
        deps_status["pytorch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: ✅ Available ({torch.cuda.device_count()} GPUs)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print(f"   CUDA: ❌ Not available")
    except ImportError:
        deps_status["pytorch"] = {"available": False}
        print(f"❌ PyTorch: Not available")

    # Check other dependencies
    other_deps = {
        "transformers": "transformers",
        "segment_anything": "segment_anything",
        "h3": "h3",
        "xarray": "xarray",
        "opencv": "cv2"
    }

    for name, module in other_deps.items():
        try:
            __import__(module)
            deps_status[name] = {"available": True}
            print(f"✅ {name}: Available")
        except ImportError:
            deps_status[name] = {"available": False}
            print(f"❌ {name}: Not available")

    return deps_status


def create_realistic_weather_scenario():
    """Create a realistic weather scenario for testing."""
    print(f"\n🌤️  Creating Weather Scenario")
    print("-" * 30)

    # Create high-resolution grid around Beijing
    lat = np.linspace(39.6, 40.4, 16)  # ~5km resolution
    lon = np.linspace(116.0, 117.0, 16)

    # Create complex weather system (typhoon-like structure)
    x, y = np.meshgrid(lon, lat)

    # Primary storm center
    storm1_center = (116.3, 39.9)
    distance1 = np.sqrt((x - storm1_center[0])**2 + (y - storm1_center[1])**2)

    # Secondary system
    storm2_center = (116.6, 40.1)
    distance2 = np.sqrt((x - storm2_center[0])**2 + (y - storm2_center[1])**2)

    # Create realistic precipitation pattern
    # Storm 1: Core with spiral structure
    angle = np.arctan2(y - storm1_center[1], x - storm1_center[0])
    spiral_factor = 1 + 0.2 * np.sin(3 * angle + distance1 * 10)
    precip1 = 60 * np.exp(-distance1**2 / 0.02) * spiral_factor

    # Storm 2: Secondary system
    precip2 = 25 * np.exp(-distance2**2 / 0.03)

    # Combine systems with noise
    total_precip = precip1 + precip2 + np.random.normal(0, 3, x.shape)
    total_precip = np.maximum(total_precip, 0)

    # Additional weather variables
    wind_speed = 20 + 15 * np.exp(-distance1**2 / 0.025) + np.random.normal(0, 2, x.shape)
    wind_speed = np.maximum(wind_speed, 0)

    pressure = 1012 - 10 * np.exp(-distance1**2 / 0.015) + np.random.normal(0, 0.5, x.shape)

    # Temperature perturbation
    temperature = 25 - 5 * np.exp(-distance1**2 / 0.02) + np.random.normal(0, 1, x.shape)

    dataset = xr.Dataset(
        {
            "precip": (("lat", "lon"), total_precip),
            "wind_speed": (("lat", "lon"), wind_speed),
            "pressure": (("lat", "lon"), pressure),
            "temperature": (("lat", "lon"), temperature),
        },
        coords={"lat": lat, "lon": lon}
    )

    print(f"  Dataset: {dataset.sizes}")
    print(f"  Precipitation: {total_precip.min():.1f} - {total_precip.max():.1f} mm")
    print(f"  Wind speed: {wind_speed.min():.1f} - {wind_speed.max():.1f} m/s")
    print(f"  Max intensity location: lat={lat[np.unravel_index(total_precip.argmax(), total_precip.shape)[0]]:.2f}")

    return dataset


def test_local_ai_pipeline(weather_data, deps_status):
    """Test the complete local AI pipeline."""
    print(f"\n🤖 Testing Local AI Pipeline")
    print("-" * 35)

    # Choose pipeline based on available dependencies
    if deps_status.get("pytorch", {}).get("available"):
        return test_enhanced_pipeline(weather_data)
    else:
        return test_basic_pipeline(weather_data)


def test_enhanced_pipeline(weather_data):
    """Test pipeline with local AI models (PyTorch available)."""
    print(f"  Using enhanced pipeline with local AI models...")

    try:
        from src.simulation.local_meteo_cognition import create_local_pipeline

        # Create LLM client if available
        llm_client = None
        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"):
            try:
                from src.cognition.llm_client import LLMClient, LLMClientConfig
                llm_client = LLMClient(LLMClientConfig(
                    endpoint=os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=os.environ.get("OPENAI_MODEL"),
                ))
                print(f"    ✅ LLM client configured")
            except Exception as e:
                print(f"    ⚠️  LLM client setup failed: {e}")

        # Create pipeline with local models
        pipeline = create_local_pipeline(
            prithvi_model="prithvi_wxc_small",
            sam_model="sam_geo_base",
            device="cuda",
            llm_client=llm_client
        )

        # Test pipeline
        print(f"    Running meteorological cognition with AI models...")
        start_time = time.time()

        try:
            event = pipeline.run(weather_data)
            total_time = time.time() - start_time

            # Get model status
            status = pipeline.get_model_status()

            results = {
                "success": True,
                "time": total_time,
                "event": {
                    "type": event.type,
                    "intensity": event.intensity_index,
                    "h3_cells": len(event.h3_coverage)
                },
                "models": status,
                "pipeline_type": "enhanced"
            }

            print(f"    ✅ Completed in {total_time:.3f}s")
            print(f"    Event: {event.type} (intensity: {event.intensity_index:.3f})")
            print(f"    H3 cells: {len(event.h3_coverage)}")

            if status.get("prithvi_model"):
                print(f"    Prithvi WXC: ✅")
            if status.get("sam_model"):
                print(f"    SAM-Geo: ✅")

            return results

        except Exception as e:
            print(f"    ❌ Enhanced pipeline failed: {e}")
            print(f"    Falling back to basic pipeline...")
            return test_basic_pipeline(weather_data)

    except Exception as e:
        print(f"    ❌ Could not create enhanced pipeline: {e}")
        return test_basic_pipeline(weather_data)


def test_basic_pipeline(weather_data):
    """Test basic pipeline without AI models."""
    print(f"  Using basic pipeline (fallback mode)...")

    from src.simulation.meteo_cognition import MeteoCognitionPipeline, MeteoCognitionConfig

    def simple_downscaler(data):
        scale_factor = 4
        lat_name = "lat"
        lon_name = "lon"

        lat_coords = data.coords[lat_name].values
        lon_coords = data.coords[lon_name].values

        lat_hr = np.linspace(lat_coords.min(), lat_coords.max(), len(lat_coords) * scale_factor)
        lon_hr = np.linspace(lon_coords.min(), lon_coords.max(), len(lon_coords) * scale_factor)

        output_vars = {}
        for name, var in data.data_vars.items():
            if var.ndim >= 2:
                interpolated = var.interp({lat_name: lat_hr, lon_name: lon_hr}, method="linear")
                output_vars[name] = interpolated

        return xr.Dataset(output_vars)

    def simple_segmenter(data):
        first_var = list(data.data_vars.keys())[0]
        field = data[first_var]
        values = field.values
        threshold = np.nanpercentile(values, 85.0)
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
            max_samples = 200
            for i in range(0, len(mask_points[0]), max(1, len(mask_points[0]) // max_samples)):
                lat_idx = mask_points[0][i]
                lon_idx = mask_points[1][i]

                if lat_idx < len(lats) and lon_idx < len(lons):
                    lat = float(lats[lat_idx])
                    lon = float(lons[lon_idx])
                    h3_cell = h3.latlng_to_cell(lat, lon, 8)
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
            "method": "basic_segmentation"
        }

    class MockLLM:
        def generate(self, prompt, payload=None):
            stats = payload.get("stats", {})
            max_intensity = stats.get("max_intensity", 0)

            if max_intensity > 50:
                event_type = "severe_storm"
                intensity_index = min(1.0, max_intensity / 80)
            elif max_intensity > 25:
                event_type = "moderate_storm"
                intensity_index = min(0.8, max_intensity / 50)
            else:
                event_type = "light_precipitation"
                intensity_index = min(0.5, max_intensity / 30)

            return {
                "json": {
                    "type": event_type,
                    "intensity_index": intensity_index,
                    "h3_coverage": payload.get("h3_coverage", []),
                    "metadata": stats
                }
            }

    config = MeteoCognitionConfig(
        downscaler=simple_downscaler,
        segmenter=simple_segmenter,
        llm_client=MockLLM(),
        default_event_type="test_event"
    )

    pipeline = MeteoCognitionPipeline(config)

    print(f"    Running meteorological cognition (basic mode)...")
    start_time = time.time()
    event = pipeline.run(weather_data)
    total_time = time.time() - start_time

    results = {
        "success": True,
        "time": total_time,
        "event": {
            "type": event.type,
            "intensity": event.intensity_index,
            "h3_cells": len(event.h3_coverage)
        },
        "pipeline_type": "basic"
    }

    print(f"    ✅ Completed in {total_time:.3f}s")
    print(f"    Event: {event.type} (intensity: {event.intensity_index:.3f})")
    print(f"    H3 cells: {len(event.h3_coverage)}")

    return results


def test_model_downloads(deps_status):
    """Test model download and management."""
    print(f"\n📦 Testing Model Management")
    print("-" * 32)

    if not deps_status.get("pytorch", {}).get("available"):
        print(f"  ⏳ Skipping model tests (PyTorch not available)")
        return

    try:
        from src.models.simple_registry import simple_registry

        print(f"  Available models:")
        models = simple_registry.list_available_models()
        for name, info in models.items():
            size = info.get("size_gb", "Unknown")
            print(f"    {name}: {info['type']} (~{size} GB)")

        print(f"  Cache directory: {simple_registry.cache_dir}")
        print(f"  Current cache usage: {simple_registry.get_cache_size()}")

        # Test environment check
        status = simple_registry.check_environment()
        if status.get("cuda_available", False):
            print(f"  ✅ GPU acceleration ready")
        else:
            print(f"  ⚠️  Will use CPU for model inference")

    except Exception as e:
        print(f"  ❌ Model management test failed: {e}")


def main():
    """Run the complete local AI demonstration."""
    print("SEIA-MOD COMPLETE LOCAL AI DEMONSTRATION")
    print("Testing local AI model integration for meteorological cognition")

    # Step 1: Environment check
    deps_status = check_environment()

    # Step 2: Create weather scenario
    weather_data = create_realistic_weather_scenario()

    # Step 3: Test AI pipeline
    pipeline_results = test_local_ai_pipeline(weather_data, deps_status)

    # Step 4: Test model management
    test_model_downloads(deps_status)

    # Step 5: Summary and recommendations
    print(f"\n📋 DEMONSTRATION SUMMARY")
    print("-" * 30)

    if pipeline_results["success"]:
        print(f"  ✅ Pipeline: {'Enhanced (AI)' if pipeline_results['pipeline_type'] == 'enhanced' else 'Basic (fallback)'}")
        print(f"  ⏱️  Processing time: {pipeline_results['time']:.3f}s")
        print(f"  🌪️  Event type: {pipeline_results['event']['type']}")
        print(f"  💪 Intensity: {pipeline_results['event']['intensity']:.3f}")
        print(f"  🗺️  H3 cells: {pipeline_results['event']['h3_cells']}")

    print(f"\n🔧 SETUP STATUS")
    print("-" * 18)

    pytorch_status = deps_status.get("pytorch", {})
    if pytorch_status.get("available"):
        print(f"  ✅ PyTorch: {pytorch_status['version']}")
        if pytorch_status.get("cuda"):
            print(f"  ✅ CUDA: {pytorch_status['gpu_count']} GPUs detected")
        else:
            print(f"  ⚠️  CUDA: Not available (CPU mode)")
    else:
        print(f"  ❌ PyTorch: Not installed yet")

    print(f"\n💡 NEXT STEPS")
    print("-" * 15)

    if not deps_status.get("pytorch", {}).get("available"):
        print(f"  ⏳ Wait for PyTorch installation to complete")
        print(f"  🔄 Then re-run this demo")
    else:
        print(f"  🚀 Ready to download AI models!")
        print(f"  📦 Run: python scripts/setup_local_models.py --download-all")
        print(f"  🧪 Run: python examples/demo_local_ai_models.py --individual")

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETED")
    print("Your SEIA-MOD system is ready for local AI model integration!")
    print("="*60)


if __name__ == "__main__":
    main()