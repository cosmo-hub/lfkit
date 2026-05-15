"""Luminosity-function model registries used by the public API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from lfkit.photometry.conditional_lf_models import (
    conditional_schechter,
    conditional_schechter_double,
    conditional_schechter_evolving,
    lognormal_conditional_lf,
    modified_schechter_conditional_lf,
    two_component_conditional_lf,
)
from lfkit.photometry.luminosity_function import (
    schechter,
    schechter_double,
    schechter_double_from_m,
    schechter_evolving,
    schechter_evolving_from_m,
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
        "function": schechter_evolving,
        "requires_z": True,
    },
    "double_schechter": {
        "function": schechter_double,
        "requires_z": False,
    },
    "conditional_schechter": {
        "function": conditional_schechter,
        "requires_z": True,
    },
    "conditional_evolving_schechter": {
        "function": conditional_schechter_evolving,
        "requires_z": True,
    },
    "conditional_double_schechter": {
        "function": conditional_schechter_double,
        "requires_z": True,
    },
    "central_lognormal_conditional": {
        "function": lognormal_conditional_lf,
        "requires_z": True,
    },
    "satellite_modified_schechter_conditional": {
        "function": modified_schechter_conditional_lf,
        "requires_z": True,
    },
    "central_satellite_conditional": {
        "function": two_component_conditional_lf,
        "requires_z": True,
    },
}


LF_FROM_M_MODELS: dict[str, Callable[..., Any]] = {
    "schechter": schechter_from_m,
    "evolving_schechter": schechter_evolving_from_m,
    "double_schechter": schechter_double_from_m,
}
