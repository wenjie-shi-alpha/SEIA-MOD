#!/usr/bin/env python3
"""
Setup script for downloading and configuring local AI models.
This script helps set up the local Prithvi WxC and SAM-Geo models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_registry import model_registry


def download_model(model_name: str, force: bool = False) -> None:
    """Download a specific model."""
    print(f"Downloading model: {model_name}")

    try:
        config = model_registry.get_model_config(model_name)
        print(f"  Model ID: {config.model_id}")
        print(f"  Cache dir: {config.cache_dir}")

        if force:
            # Force re-download by removing existing cache
            import shutil
            cache_path = Path(config.cache_dir) / config.model_id.replace("/", "--")
            if cache_path.exists():
                print(f"  Removing existing cache: {cache_path}")
                shutil.rmtree(cache_path)

        local_path = model_registry.download_model(model_name)
        print(f"  ✅ Downloaded to: {local_path}")

    except Exception as e:
        print(f"  ❌ Failed to download {model_name}: {e}")
        return

    # Verify model size and contents
    try:
        cache_path = Path(model_registry.get_model_path(model_name))
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"  📊 Model size: {size_gb:.2f} GB")
            print(f"  📁 Files: {len(list(cache_path.rglob('*')))}")
    except Exception as e:
        print(f"  ⚠️  Could not verify model: {e}")


def setup_environment() -> None:
    """Setup the environment for local model usage."""
    print("Setting up environment for local AI models...")

    # Check CUDA availability
    try:
        import torch
        print(f"  🔥 PyTorch version: {torch.__version__}")
        print(f"  🚀 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  📱 CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    Device {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    except ImportError:
        print("  ❌ PyTorch not installed. Please install with: pip install torch torchvision")

    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    print(f"  💾 Available disk space: {free_gb} GB")

    if free_gb < 10:
        print("  ⚠️  Warning: Less than 10GB free space available for model storage")


def install_dependencies() -> None:
    """Install required dependencies for local models."""
    print("Installing dependencies for local AI models...")

    requirements_path = ROOT / "requirements-ml.txt"
    if requirements_path.exists():
        print(f"  📦 Installing from: {requirements_path}")
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ], check=True, capture_output=True, text=True)
            print("  ✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install dependencies: {e}")
            print(f"  stderr: {e.stderr}")
    else:
        print("  ⚠️  requirements-ml.txt not found")


def list_models() -> None:
    """List all available models."""
    print("Available models:")
    models = model_registry.list_available_models()

    for name, info in models.items():
        print(f"  {name}:")
        print(f"    Type: {info['type']}")
        print(f"    Model ID: {info['model_id']}")
        print(f"    Device: {info['device']}")
        print(f"    Dtype: {info['dtype']}")
        print()

    # Show cache usage
    cache_size = model_registry.get_cache_size()
    print(f"Cache usage: {cache_size}")


def main():
    parser = argparse.ArgumentParser(description="Setup local AI models for SEIA-MOD")
    parser.add_argument("--setup-env", action="store_true", help="Setup environment")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--download", type=str, help="Download specific model")
    parser.add_argument("--download-all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.setup_env:
        setup_environment()

    if args.install_deps:
        install_dependencies()

    if args.list:
        list_models()

    if args.download:
        download_model(args.download, args.force)

    if args.download_all:
        # Download a representative set of models
        models_to_download = ["prithvi_wxc_small", "sam_geo_base"]
        for model_name in models_to_download:
            download_model(model_name, args.force)

    if not any([args.setup_env, args.install_deps, args.download, args.download_all, args.list]):
        print("SEIA-MOD Local Model Setup")
        print("Usage: python setup_local_models.py [options]")
        print("Options:")
        print("  --setup-env      Check environment setup")
        print("  --install-deps   Install required dependencies")
        print("  --list           List available models")
        print("  --download MODEL Download specific model")
        print("  --download-all   Download essential models")
        print("  --force          Force re-download")
        print()
        print("Recommended workflow:")
        print("1. python setup_local_models.py --setup-env")
        print("2. python setup_local_models.py --install-deps")
        print("3. python setup_local_models.py --list")
        print("4. python setup_local_models.py --download-all")


if __name__ == "__main__":
    main()