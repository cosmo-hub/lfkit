"""Luminosity-function model registries used by the public API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from lfkit.photometry.conditional_lf_models import (
    conditional_schechter,
    conditional_double_schechter,
    conditional_evolving_schechter,
    lognormal_conditional_lf,
    modified_schechter_conditional_lf,
    two_component_conditional_lf,
)
from lfkit.photometry.luminosity_function import (
    schechter,
    double_schechter,
    double_schechter_from_m,
    evolving_schechter,
    evolving_schechter_from_m,
    schechter_from_m,
)


class LFModelSpec(TypedDict):
    """Description of a luminosity-function model exposed by the API."""

    function: Callable[..., Any]
    requires_z: bool


LF_MODELS: dict[str, LFModelSpec] = {
    "schechter": {
        "function": schechter,
        "requires_z": False,
    },
    "evolving_schechter": {
        "function": evolving_schechter,
        "requires_z": True,
    },
    "double_schechter": {
        "function": double_schechter,
        "requires_z": False,
    },
    "conditional_schechter": {
        "function": conditional_schechter,
        "requires_z": True,
    },
    "conditional_evolving_schechter": {
        "function": conditional_evolving_schechter,
        "requires_z": True,
    },
    "conditional_double_schechter": {
        "function": conditional_double_schechter,
        "requires_z": True,
    },
    "lognormal_conditional_lf": {
        "function": lognormal_conditional_lf,
        "requires_z": True,
    },
    "modified_schechter_conditional_lf": {
        "function": modified_schechter_conditional_lf,
        "requires_z": True,
    },
    "two_component_conditional_lf": {
        "function": two_component_conditional_lf,
        "requires_z": True,
    },
}


LF_FROM_M_MODELS: dict[str, Callable[..., Any]] = {
    "schechter": schechter_from_m,
    "evolving_schechter": evolving_schechter_from_m,
    "double_schechter": double_schechter_from_m,
}
