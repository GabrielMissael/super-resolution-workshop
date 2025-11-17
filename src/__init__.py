# Package initializer for src
# Re-organized to expose logical subpackages while keeping backwards compatibility.

# Re-export top-level modules for backwards compatibility
from . import debayer, enhance, quality_and_align, visualize_results, stacking  # noqa: F401

# New organized subpackages
from . import demosaic, io, process, visual  # noqa: F401

__all__ = [
    # new subpackages
    'demosaic', 'io', 'process', 'visual',
]
