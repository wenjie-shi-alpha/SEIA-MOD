"""Simplified model registry that doesn't require PyTorch for basic operations."""

import os
from pathlib import Path


class SimpleModelRegistry:
    """Simplified model registry for testing without PyTorch dependencies."""

    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "seia-mod" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure Hugging Face cache
        os.environ["HF_HOME"] = str(self.cache_dir / "huggingface")

    def list_available_models(self):
        """List all available models with their info."""
        models = {
            "prithvi_wxc_small": {
                "model_id": "ibm-granite/prithvi.wxc.small",
                "type": "prithvi_wxc",
                "device": "cuda",
                "dtype": "bfloat16",
                "description": "Small Prithvi WxC model for weather downscaling",
                "size_gb": "2-4",
            },
            "prithvi_wxc_base": {
                "model_id": "ibm-granite/prithvi.wxc.base",
                "type": "prithvi_wxc",
                "device": "cuda",
                "dtype": "bfloat16",
                "description": "Base Prithvi WxC model for weather downscaling",
                "size_gb": "4-8",
            },
            "sam_geo_base": {
                "model_id": "facebook/sam-vit-base",
                "type": "sam_geo",
                "device": "cuda",
                "dtype": "float16",
                "description": "Base SAM model for geospatial segmentation",
                "size_gb": "0.4-1",
            },
            "sam_geo_large": {
                "model_id": "facebook/sam-vit-large",
                "type": "sam_geo",
                "device": "cuda",
                "dtype": "float16",
                "description": "Large SAM model for geospatial segmentation",
                "size_gb": "1-2",
            },
            "sam_geo_huge": {
                "model_id": "facebook/sam-vit-huge",
                "type": "sam_geo",
                "device": "cuda",
                "dtype": "float16",
                "description": "Huge SAM model for geospatial segmentation",
                "size_gb": "2-4",
            }
        }
        return models

    def get_cache_size(self):
        """Get the total size of cached models."""
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"

    def check_environment(self):
        """Check if the environment supports local models."""
        status = {
            "python_available": True,
            "cache_dir_exists": self.cache_dir.exists(),
            "cache_dir_writable": False,
            "disk_space_gb": 0,
            "pytorch_available": False,
            "cuda_available": False,
            "gpu_memory_gb": 0,
        }

        # Check cache directory writable
        try:
            test_file = self.cache_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            status["cache_dir_writable"] = True
        except Exception:
            pass

        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        status["disk_space_gb"] = free // (1024**3)

        # Check PyTorch
        try:
            import torch
            status["pytorch_available"] = True
            status["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                status["gpu_memory_gb"] = props.total_memory // (1024**3)
        except ImportError:
            pass

        return status


# Global simple registry instance
simple_registry = SimpleModelRegistry()