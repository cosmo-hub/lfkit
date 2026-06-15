"""Conditional luminosity function model utilities.

This module provides generic conditional wrappers around LFKit luminosity
function models.

A conditional luminosity function has the form ``Phi(M | x_1, x_2, ...)``,
where ``M`` is absolute magnitude and the ``x_i`` are external conditioning
variables. Callable model parameters are evaluated at the supplied conditioning
variables before the wrapped luminosity function is evaluated.
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

    Callable keyword arguments are interpreted as parameter models and evaluated
    as functions of the supplied conditioning variables. Non-callable keyword
    arguments are passed through unchanged.

    Args:
        lf_model: Luminosity function model to wrap.

    Returns:
        Conditional luminosity function model with signature
        ``conditional_model(absolute_mag, *conditions, **kwargs)``.
    """

    @wraps(lf_model)
    def conditional_model(
        absolute_mag: FloatInput,
        *conditions: FloatInput,
        **kwargs: Any,
    ) -> FloatArray:
        absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
        condition_arrays = _validate_conditions(conditions)

        evaluated_kwargs: dict[str, Any] = {}
        for name, value in kwargs.items():
            if callable(value):
                evaluated_kwargs[name] = validate_array(
                    value(*condition_arrays),
                    name=name,
                )
            else:
                evaluated_kwargs[name] = value

        phi = lf_model(absolute_mag_arr, **evaluated_kwargs)
        return _validate_lf_output(phi, name=lf_model.__name__)

    return conditional_model


def _validate_conditions(conditions: tuple[FloatInput, ...]) -> tuple[FloatArray, ...]:
    """Return validated conditioning variable arrays.

    Args:
        conditions: Conditioning variable values.

    Returns:
        Validated conditioning variable arrays.

    Raises:
        ValueError: If no conditioning variables are supplied.
    """
    if not conditions:
        raise ValueError("At least one conditioning variable is required.")

    return tuple(
        validate_array(condition, name=f"condition_{i}")
        for i, condition in enumerate(conditions)
    )


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
