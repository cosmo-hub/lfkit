"""Conditional luminosity function model utilities.

This module provides generic conditional wrappers around LFKit luminosity
function models.

A conditional luminosity function has the form ``Phi(M | x)``, where ``M`` is
absolute magnitude and ``x`` is an external conditioning variable. Callable
model parameters are evaluated at ``x`` before the wrapped luminosity function
is evaluated.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from lfkit.luminosity_functions._discovery import iter_model_functions
from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array


def conditionalize_lf_model(
    lf_model: Callable[..., FloatArray],
) -> Callable[..., FloatArray]:
    """Return a conditional version of a luminosity function model.

    Callable keyword arguments are interpreted as parameter models and evaluated as
    functions of ``condition``. Non-callable keyword arguments are passed through
    unchanged.

    Args:
        lf_model: Luminosity function model to wrap.

    Returns:
        Conditional luminosity function model with signature
        ``conditional_model(absolute_mag, condition, **kwargs)``.
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


def _conditional_model_name(name: str) -> str:
    """Return the generated conditional wrapper name for a luminosity function model.

    Args:
        name: Base luminosity function model name.

    Returns:
        Conditional model name prefixed with ``"conditional_"``.
    """
    if name.endswith("_lf"):
        name = name.removesuffix("_lf")

    return f"conditional_{name}"


def _validate_lf_output(
    phi: FloatInput,
    *,
    name: str,
) -> FloatArray:
    """Validate luminosity function model output.

    Args:
        phi: Luminosity function values returned by a model.
        name: Model name used in validation errors.

    Returns:
        Validated luminosity function values as a float array.

    Raises:
        ValueError: If any luminosity function value is negative.
    """
    phi_arr = validate_array(phi, name=name)

    if np.any(phi_arr < 0.0):
        raise ValueError(f"{name} returned negative values, which are not allowed.")

    return np.asarray(phi_arr, dtype=np.float64)


__all__ = ["conditionalize_lf_model"]

for _name, _function in iter_model_functions().items():
    if _name.endswith("_from_m"):
        continue

    _conditional_name = _conditional_model_name(_name)
    globals()[_conditional_name] = conditionalize_lf_model(_function)
    __all__.append(_conditional_name)
