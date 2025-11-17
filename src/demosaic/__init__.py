"""Demosaic helpers subpackage

Re-exports existing debayer modules for a cleaner API:

Usage:
    from src.demosaic import debayer, debayer_methods
    debayer.debayer_image(...)
"""
from __future__ import annotations

# Re-export the sibling modules from the top-level package so we don't have to move files.
from . import debayer as debayer  # type: ignore
from . import debayer_methods as debayer_methods  # type: ignore

__all__ = ["debayer", "debayer_methods"]

