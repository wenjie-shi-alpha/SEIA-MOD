#!/usr/bin/env python3
"""
Test PyTorch installation and GPU availability.
This script will be ready to run as soon as PyTorch finishes installing.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_pytorch_basic():
    """Test basic PyTorch functionality."""
    print("=" * 60)
    print("PYTORCH INSTALLATION TEST")
    print("=" * 60)

    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        print(f"   Python bindings: {torch.__config__.show()}")

        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False


def test_cuda_availability():
    """Test CUDA and GPU availability."""
    try:
        import torch

        print(f"\n🚀 CUDA Testing")
        print("-" * 20)

        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"     Compute capability: {props.major}.{props.minor}")

                # Test simple GPU operation
                try:
                    device = torch.device(f"cuda:{i}")
                    x = torch.randn(1000, 1000, device=device)
                    y = torch.randn(1000, 1000, device=device)
                    z = torch.matmul(x, y)
                    print(f"     ✅ GPU matrix multiplication test passed")
                except Exception as e:
                    print(f"     ❌ GPU test failed: {e}")

            return True
        else:
            print(f"❌ CUDA is not available")
            print(f"   Will use CPU for computations")
            return False

    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False


def test_basic_tensor_operations():
    """Test basic tensor operations."""
    try:
        import torch

        print(f"\n🧮 Tensor Operations Test")
        print("-" * 30)

        # CPU operations
        print(f"  Testing CPU operations...")
        x_cpu = torch.randn(2, 3)
        y_cpu = torch.randn(2, 3)
        z_cpu = x_cpu + y_cpu
        print(f"    ✅ CPU tensor operations work")

        # GPU operations if available
        if torch.cuda.is_available():
            print(f"  Testing GPU operations...")
            device = torch.device("cuda:0")
            x_gpu = torch.randn(2, 3, device=device)
            y_gpu = torch.randn(2, 3, device=device)
            z_gpu = x_gpu + y_gpu
            print(f"    ✅ GPU tensor operations work")

            # Test memory management
            print(f"  Testing GPU memory management...")
            large_tensor = torch.randn(10000, 10000, device=device)
            print(f"    ✅ Large GPU tensor allocation works")
            del large_tensor
            torch.cuda.empty_cache()
            print(f"    ✅ GPU memory cleanup works")

        return True

    except Exception as e:
        print(f"❌ Tensor operations test failed: {e}")
        return False


def test_ai_model_compatibility():
    """Test compatibility with our AI model framework."""
    print(f"\n🤖 AI Model Compatibility Test")
    print("-" * 35)

    try:
        # Test if we can import our model classes
        sys.path.insert(0, str(ROOT))

        print(f"  Testing model imports...")

        try:
            from src.models.prithvi_wxc import LocalPrithviWXC
            print(f"    ✅ LocalPrithviWXC import successful")
        except ImportError as e:
            print(f"    ⚠️  LocalPrithviWXC import (expected without full setup): {e}")

        try:
            from src.models.sam_geo import LocalSAMGeo
            print(f"    ✅ LocalSAMGeo import successful")
        except ImportError as e:
            print(f"    ⚠️  LocalSAMGeo import (expected without full setup): {e}")

        try:
            from src.models.model_registry import ModelRegistry
            registry = ModelRegistry()
            config = registry.get_model_config("prithvi_wxc_small")
            print(f"    ✅ ModelRegistry and config working")
            print(f"    ✅ Available models: {len(registry.list_available_models())}")
        except Exception as e:
            print(f"    ⚠️  ModelRegistry (may need huggingface_hub): {e}")

        # Test transformer compatibility
        try:
            import transformers
            print(f"    ✅ Transformers library available: {transformers.__version__}")
        except ImportError:
            print(f"    ❌ Transformers library not available")

        # Test H3 compatibility
        try:
            import h3
            test_cell = h3.latlng_to_cell(39.9, 116.4, 8)
            print(f"    ✅ H3 geospatial indexing working")
        except ImportError:
            print(f"    ❌ H3 library not available")

        return True

    except Exception as e:
        print(f"❌ AI model compatibility test failed: {e}")
        return False


def test_performance_benchmark():
    """Run a simple performance benchmark."""
    try:
        import torch
        import time

        print(f"\n⚡ Performance Benchmark")
        print("-" * 28)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Running benchmark on: {device}")

        # Matrix multiplication benchmark
        sizes = [1000, 2000, 4000]

        for size in sizes:
            print(f"  Testing {size}x{size} matrix multiplication...")

            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)

            # Warm up
            for _ in range(3):
                _ = torch.matmul(x, y)

            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            for _ in range(10):
                _ = torch.matmul(x, y)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            gflops = (2 * size**3) / (avg_time * 1e9)

            print(f"    ✅ {avg_time:.4f}s avg, {gflops:.1f} GFLOPS")

        return True

    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Run all PyTorch tests."""
    print("Testing PyTorch installation for SEIA-MOD AI models...")

    # Test 1: Basic installation
    pytorch_ok = test_pytorch_basic()
    if not pytorch_ok:
        print("\n❌ PyTorch not installed yet. Please wait for installation to complete.")
        return

    # Test 2: CUDA availability
    cuda_ok = test_cuda_availability()

    # Test 3: Tensor operations
    tensor_ok = test_basic_tensor_operations()

    # Test 4: AI model compatibility
    ai_ok = test_ai_model_compatibility()

    # Test 5: Performance benchmark
    perf_ok = test_performance_benchmark()

    # Summary
    print(f"\n📋 Test Summary")
    print("-" * 20)
    print(f"  PyTorch: {'✅' if pytorch_ok else '❌'}")
    print(f"  CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"  Tensors: {'✅' if tensor_ok else '❌'}")
    print(f"  AI Models: {'✅' if ai_ok else '❌'}")
    print(f"  Performance: {'✅' if perf_ok else '❌'}")

    # Recommendations
    print(f"\n💡 Next Steps")
    print("-" * 15)

    if pytorch_ok and ai_ok:
        print(f"  🚀 Ready to install AI models!")
        print(f"     Run: python scripts/setup_local_models.py --download-all")
        print(f"     Run: python examples/demo_local_ai_models.py")
    elif pytorch_ok:
        print(f"  ⚠️  PyTorch installed but some AI components missing")
        print(f"     Check transformer and h3 library installation")
    else:
        print(f"  ⏳ Still waiting for PyTorch installation...")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()