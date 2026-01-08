#!/usr/bin/env python3
"""
Demo script showcasing local AI model integration for meteorological cognition.
This script demonstrates the enhanced capabilities with local Prithvi WxC and SAM-Geo models.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr

from src.cognition.llm_client import LLMClient, LLMClientConfig
from src.simulation.local_meteo_cognition import create_local_pipeline
from src.models.model_registry import model_registry


def create_synthetic_weather_data():
    """Create synthetic weather data for demonstration."""
    print("Creating synthetic weather data...")

    # Create a grid around Beijing area
    lat = np.linspace(39.5, 40.5, 16)  # Coarse resolution (~6km)
    lon = np.linspace(116.0, 117.0, 16)

    # Create a synthetic storm system
    x, y = np.meshgrid(lon, lat)

    # Storm 1: Intense precipitation system
    storm1_center = (116.3, 39.9)
    distance1 = np.sqrt((x - storm1_center[0])**2 + (y - storm1_center[1])**2)
    precip1 = 30 * np.exp(-distance1**2 / 0.05)

    # Storm 2: Secondary system
    storm2_center = (116.6, 40.2)
    distance2 = np.sqrt((x - storm2_center[0])**2 + (y - storm2_center[1])**2)
    precip2 = 20 * np.exp(-distance2**2 / 0.08)

    # Combine systems with some noise
    total_precip = precip1 + precip2 + np.random.normal(0, 2, x.shape)
    total_precip = np.maximum(total_precip, 0)  # Ensure non-negative

    # Create additional weather variables
    wind_speed = 15 + 10 * np.exp(-distance1**2 / 0.1) + np.random.normal(0, 3, x.shape)
    wind_speed = np.maximum(wind_speed, 0)

    pressure = 1013 - 8 * np.exp(-distance1**2 / 0.06) + np.random.normal(0, 1, x.shape)

    # Create dataset
    dataset = xr.Dataset(
        {
            "precip": (("lat", "lon"), total_precip),
            "wind_speed": (("lat", "lon"), wind_speed),
            "pressure": (("lat", "lon"), pressure),
        },
        coords={"lat": lat, "lon": lon}
    )

    print(f"  Created dataset: {dataset.sizes}")
    print(f"  Precip range: {total_precip.min():.1f} - {total_precip.max():.1f} mm")
    print(f"  Wind range: {wind_speed.min():.1f} - {wind_speed.max():.1f} m/s")
    print(f"  Pressure range: {pressure.min():.1f} - {pressure.max():.1f} hPa")

    return dataset


def demo_local_models():
    """Demonstrate local AI model capabilities."""
    print("\n" + "="*60)
    print("SEIA-MOD LOCAL AI MODELS DEMONSTRATION")
    print("="*60)

    # Check environment
    print("\n1. Environment Check")
    print("-" * 30)
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("  ❌ PyTorch not available")

    # Show available models
    print("\n2. Available Models")
    print("-" * 30)
    models = model_registry.list_available_models()
    for name, info in models.items():
        print(f"  {name}: {info['type']} ({info['model_id']})")

    # Create LLM client if available
    llm_client = None
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"):
        try:
            llm_client = LLMClient(LLMClientConfig(
                endpoint=os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=os.environ.get("OPENAI_MODEL"),
            ))
            print("  ✅ LLM client configured")
        except Exception as e:
            print(f"  ⚠️  LLM client setup failed: {e}")
    else:
        print("  ℹ️  No LLM client configured (set OPENAI_API_KEY and OPENAI_MODEL)")

    # Create synthetic data
    weather_data = create_synthetic_weather_data()

    # Test different pipeline configurations
    configs_to_test = [
        {
            "name": "Full Local AI",
            "use_local_models": True,
            "prithvi_model": "prithvi_wxc_small",
            "sam_model": "sam_geo_base",
            "llm_client": llm_client,
        },
        {
            "name": "Fallback Mode",
            "use_local_models": False,
            "llm_client": llm_client,
        },
        {
            "name": "Local AI Only (No LLM)",
            "use_local_models": True,
            "prithvi_model": "prithvi_wxc_small",
            "sam_model": "sam_geo_base",
            "llm_client": None,
        },
    ]

    results = []

    for config in configs_to_test:
        print(f"\n3. Testing: {config['name']}")
        print("-" * 40)

        try:
            # Create pipeline
            start_time = time.time()
            pipeline = create_local_pipeline(**{k: v for k, v in config.items() if k != 'name'})
            init_time = time.time() - start_time

            # Get model status
            status = pipeline.get_model_status()
            print(f"  Models loaded: {status['local_models_enabled']}")
            print(f"  Fallback mode: {status['fallback_mode']}")
            if status.get('prithvi_model'):
                print(f"  Prithvi: ✅")
            if status.get('sam_model'):
                print(f"  SAM-Geo: ✅")

            # Run pipeline
            print("  Running meteorological cognition...")
            start_time = time.time()
            event = pipeline.run(weather_data)
            process_time = time.time() - start_time

            # Store results
            results.append({
                'config': config['name'],
                'init_time': init_time,
                'process_time': process_time,
                'event': event,
                'success': True,
                'h3_cells': len(event.get('h3_coverage', [])),
                'intensity': event.get('intensity_index', 0),
            })

            # Display results
            print(f"  ✅ Completed in {process_time:.2f}s")
            print(f"  Event type: {event['type']}")
            print(f"  Intensity: {event['intensity_index']:.3f}")
            print(f"  H3 coverage: {len(event.get('h3_coverage', []))} cells")

            # Show metadata
            metadata = event.get('metadata', {})
            if 'max_intensity' in metadata:
                print(f"  Max intensity: {metadata['max_intensity']:.2f}")
            if 'mask_fraction' in metadata:
                print(f"  Coverage fraction: {metadata['mask_fraction']:.3f}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'config': config['name'],
                'init_time': 0,
                'process_time': 0,
                'event': None,
                'success': False,
                'error': str(e),
            })

    # Performance comparison
    print(f"\n4. Performance Comparison")
    print("-" * 30)
    successful_results = [r for r in results if r['success']]
    if len(successful_results) > 1:
        print(f"{'Configuration':<25} {'Init Time':<12} {'Process Time':<14} {'H3 Cells':<10} {'Intensity':<10}")
        print("-" * 75)
        for result in successful_results:
            print(f"{result['config']:<25} {result['init_time']:<12.3f} {result['process_time']:<14.3f} "
                  f"{result['h3_cells']:<10} {result['intensity']:<10.3f}")

    # Summary
    print(f"\n5. Summary")
    print("-" * 15)
    successful = len(successful_results)
    total = len(results)
    print(f"Successful runs: {successful}/{total}")

    if successful > 0:
        avg_time = np.mean([r['process_time'] for r in successful_results])
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Local AI models working: {any(r['config'].startswith('Full Local') or r['config'].startswith('Local AI') for r in successful_results)}")

    # Cache information
    cache_size = model_registry.get_cache_size()
    print(f"Model cache size: {cache_size}")

    print(f"\n{'='*60}")
    print("DEMO COMPLETED")
    print("="*60)


def demo_individual_models():
    """Demonstrate individual model capabilities."""
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL DEMONSTRATION")
    print("="*60)

    # Test Prithvi WXC
    print("\n1. Prithvi WXC Downscaling Test")
    print("-" * 40)
    try:
        from src.models import LocalPrithviWXC
        model = LocalPrithviWXC(model_name="prithvi_wxc_small")

        # Create coarse input
        lat = np.linspace(39.8, 40.0, 8)
        lon = np.linspace(116.2, 116.4, 8)
        precip = np.random.rand(8, 8) * 25

        input_data = xr.Dataset(
            {"precip": (("lat", "lon"), precip)},
            coords={"lat": lat, "lon": lon}
        )

        print(f"  Input shape: {input_data.precip.shape}")

        # Perform downscaling
        start_time = time.time()
        output = model.downscale(input_data)
        downscale_time = time.time() - start_time

        print(f"  Output shape: {output.precip.shape}")
        print(f"  Upscaling factor: {output.precip.shape[1] / input_data.precip.shape[1]:.1f}x")
        print(f"  Processing time: {downscale_time:.3f}s")
        print("  ✅ Prithvi WXC working correctly")

    except Exception as e:
        print(f"  ❌ Prithvi WXC test failed: {e}")

    # Test SAM-Geo
    print("\n2. SAM-Geo Segmentation Test")
    print("-" * 40)
    try:
        from src.models import LocalSAMGeo
        model = LocalSAMGeo(model_name="sam_geo_base")

        # Create high-resolution input
        lat = np.linspace(39.8, 40.0, 32)
        lon = np.linspace(116.2, 116.4, 32)
        intensity = np.random.rand(32, 32) * 50

        input_data = xr.Dataset(
            {"intensity": (("lat", "lon"), intensity)},
            coords={"lat": lat, "lon": lon}
        )

        print(f"  Input shape: {input_data.intensity.shape}")

        # Perform segmentation
        start_time = time.time()
        result = model.segment_meteorological_features(input_data, percentile=80.0)
        segment_time = time.time() - start_time

        h3_cells = len(result.get('h3_coverage', []))
        stats = result.get('stats', {})

        print(f"  H3 cells identified: {h3_cells}")
        print(f"  Max intensity: {stats.get('max_intensity', 0):.2f}")
        print(f"  Coverage fraction: {stats.get('mask_fraction', 0):.3f}")
        print(f"  Processing time: {segment_time:.3f}s")
        print("  ✅ SAM-Geo working correctly")

    except Exception as e:
        print(f"  ❌ SAM-Geo test failed: {e}")


if __name__ == "__main__":
    # Check if we should run individual model tests
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        demo_individual_models()
    else:
        demo_local_models()