"""Shared luminosity function discovery helpers."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

import lfkit.luminosity_functions.models as models_pkg


_SKIP_MODULES = {
    "modifiers",
}


def iter_model_functions() -> dict[str, Callable]:
    """Return public callable luminosity function models discovered from ``models/``.

    Returns:
        Dictionary mapping exported function names to callable model functions.
    """
    functions: dict[str, Callable] = {}

    for module_info in pkgutil.iter_modules(models_pkg.__path__):
        if module_info.name.startswith("_"):
            continue

        if module_info.name in _SKIP_MODULES:
            continue

        module = importlib.import_module(f"{models_pkg.__name__}.{module_info.name}")

        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)

            if callable(obj):
                functions[name] = obj

    return functions
