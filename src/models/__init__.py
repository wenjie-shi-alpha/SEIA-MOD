"""Local AI model implementations for meteorological cognition."""

# Lazy imports to avoid dependency issues
def _lazy_import(module_name, class_name):
    """Lazy import function to avoid dependency errors."""
    def wrapper(*args, **kwargs):
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)(*args, **kwargs)
    return wrapper

try:
    from .prithvi_wxc import LocalPrithviWXC
except ImportError:
    LocalPrithviWXC = _lazy_import('prithvi_wxc', 'LocalPrithviWXC')

try:
    from .sam_geo import LocalSAMGeo
except ImportError:
    LocalSAMGeo = _lazy_import('sam_geo', 'LocalSAMGeo')

try:
    from .model_registry import ModelRegistry, ModelConfig
except ImportError:
    ModelRegistry = _lazy_import('model_registry', 'ModelRegistry')
    ModelConfig = _lazy_import('model_registry', 'ModelConfig')

__all__ = ["LocalPrithviWXC", "LocalSAMGeo", "ModelRegistry", "ModelConfig"]