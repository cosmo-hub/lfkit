"""Top-level package for LFKit."""

from __future__ import annotations

from lfkit.api.corrections import Corrections
from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.api.conditional_luminosity_function import ConditionalLuminosityFunction

try:
    from lfkit._version import version as __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "Corrections",
    "LuminosityFunction",
    "ConditionalLuminosityFunction",
]

__author__ = """Niko Sarcevic and collaborators."""
