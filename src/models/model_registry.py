"""Model registry for managing local AI models."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import huggingface_hub
import torch


@dataclass
class ModelConfig:
    """Configuration for a local AI model."""

    name: str
    model_id: str  # Hugging Face model ID or local path
    model_type: str  # "prithvi_wxc", "sam_geo", etc.
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        if self.torch_dtype is None:
            self.torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "seia-mod" / "models")


class ModelRegistry:
    """Registry for managing local AI model downloads and loading."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "seia-mod" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, any] = {}

        # Configure Hugging Face cache
        os.environ["HF_HOME"] = str(self.cache_dir / "huggingface")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        configs = {
            # Prithvi WxC configurations
            "prithvi_wxc_small": ModelConfig(
                name="prithvi_wxc_small",
                model_id="ibm-granite/prithvi.wxc.small",
                model_type="prithvi_wxc",
                device="cuda",
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            ),
            "prithvi_wxc_base": ModelConfig(
                name="prithvi_wxc_base",
                model_id="ibm-granite/prithvi.wxc.base",
                model_type="prithvi_wxc",
                device="cuda",
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            ),

            # SAM-Geo configurations
            "sam_geo_huge": ModelConfig(
                name="sam_geo_huge",
                model_id="facebook/sam-vit-huge",  # Using official SAM as base
                model_type="sam_geo",
                device="cuda",
                dtype="float16",
            ),
            "sam_geo_large": ModelConfig(
                name="sam_geo_large",
                model_id="facebook/sam-vit-large",
                model_type="sam_geo",
                device="cuda",
                dtype="float16",
            ),
            "sam_geo_base": ModelConfig(
                name="sam_geo_base",
                model_id="facebook/sam-vit-base",
                model_type="sam_geo",
                device="cuda",
                dtype="float16",
            ),
        }

        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(configs.keys())}")

        return configs[model_name]

    def download_model(self, model_name: str) -> str:
        """Download a model and return the local path."""
        config = self.get_model_config(model_name)

        print(f"Downloading model {model_name} ({config.model_id})...")
        print(f"Cache directory: {config.cache_dir}")

        try:
            # Use huggingface_hub to download the model
            local_path = huggingface_hub.snapshot_download(
                repo_id=config.model_id,
                cache_dir=config.cache_dir,
                trust_remote_code=config.trust_remote_code,
            )
            print(f"Model downloaded to: {local_path}")
            return local_path
        except Exception as e:
            print(f"Failed to download model {model_name}: {e}")
            raise

    def get_model_path(self, model_name: str) -> str:
        """Get the local path for a model, downloading if necessary."""
        config = self.get_model_config(model_name)
        local_path = huggingface_hub.snapshot_download(
            repo_id=config.model_id,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        return local_path

    def list_available_models(self) -> Dict[str, Dict[str, str]]:
        """List all available models with their info."""
        models = {}
        for model_name in ["prithvi_wxc_small", "prithvi_wxc_base",
                          "sam_geo_base", "sam_geo_large", "sam_geo_huge"]:
            try:
                config = self.get_model_config(model_name)
                models[model_name] = {
                    "model_id": config.model_id,
                    "type": config.model_type,
                    "device": config.device,
                    "dtype": config.dtype,
                }
            except ValueError:
                continue
        return models

    def get_cache_size(self) -> str:
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


# Global registry instance
model_registry = ModelRegistry()