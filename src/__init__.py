# Package initializer for src
# Provide lazy subpackage imports to avoid circular import problems.

__all__ = [
    # new subpackages
    'demosaic', 'io', 'process', 'visual',
    # legacy module names kept for compatibility
    'debayer', 'enhance', 'quality_and_align', 'visualize_results', 'stacking',
]

# Lazy import mechanism (PEP 562 style)
import importlib
import types

_lazy_submodules = {
    'demosaic': 'src.demosaic',
    'io': 'src.io',
    'process': 'src.process',
    'visual': 'src.visual',
    # legacy modules mapping
    'debayer': 'src.demosaic.debayer',
    'debayer_methods': 'src.demosaic.debayer_methods',
    'enhance': 'src.process.enhance',
    'quality_and_align': 'src.process.quality_and_align',
    'stacking': 'src.process.stacking',
    'visualize_results': 'src.visual.visualize_results',
}


def __getattr__(name: str):
    if name in _lazy_submodules:
        module_name = _lazy_submodules[name]
        module = importlib.import_module(module_name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_submodules.keys()))
