"""Conditional luminosity function model utilities.

This module provides generic conditional wrappers around LFKit luminosity
function models.

A conditional luminosity function has the form ``Phi(M | x)``, where ``M`` is
absolute magnitude and ``x`` is an external conditioning variable.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from lfkit.luminosity_functions.models.schechter import (
    double_schechter,
    lognormal_lf,
    modified_schechter,
    schechter,
    two_component_lf,
)
from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "conditionalize_lf_model",
    "conditional_schechter",
    "conditional_double_schechter",
    "conditional_lognormal_lf",
    "conditional_modified_schechter",
    "conditional_two_component_lf",
]


def conditionalize_lf_model(
    lf_model: Callable[..., FloatArray],
) -> Callable[..., FloatArray]:
    """Return a conditional version of an LF model.

    Keyword arguments that are callable are evaluated as functions of
    ``condition``. Non-callable keyword arguments are passed through unchanged.
    """

    @wraps(lf_model)
    def conditional_model(
        absolute_mag: FloatInput,
        condition: FloatInput,
        **kwargs: Any,
    ) -> FloatArray:
        condition_arr = validate_array(condition, name="condition")

        evaluated_kwargs: dict[str, Any] = {}
        for name, value in kwargs.items():
            if callable(value):
                evaluated_kwargs[name] = validate_array(
                    value(condition_arr),
                    name=name,
                )
            else:
                evaluated_kwargs[name] = value

        phi = lf_model(absolute_mag, **evaluated_kwargs)
        return _validate_lf_output(phi, name=lf_model.__name__)

    return conditional_model


conditional_schechter = conditionalize_lf_model(schechter)
conditional_double_schechter = conditionalize_lf_model(double_schechter)
conditional_lognormal_lf = conditionalize_lf_model(lognormal_lf)
conditional_modified_schechter = conditionalize_lf_model(modified_schechter)
conditional_two_component_lf = conditionalize_lf_model(two_component_lf)


def _validate_lf_output(
    phi: FloatInput,
    *,
    name: str,
) -> FloatArray:
    """Validate luminosity function model output."""
    phi_arr = validate_array(phi, name=name)

    if np.any(phi_arr < 0.0):
        raise ValueError(f"{name} returned negative values, which are not allowed.")

    return np.asarray(phi_arr, dtype=np.float64)
