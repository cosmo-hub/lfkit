"""Conditional luminosity function integration utilities.

This module provides small numerical helpers for conditional luminosity
functions of the form ``Phi(M | x)``, where ``M`` is absolute magnitude and
``x`` is an external conditioning variable.

The conditioning variable is intentionally generic. It may represent halo mass,
environment, galaxy type, richness, stellar mass, or any other quantity. This
module does not implement HOD or halo-model machinery.

The goal is to support conditional luminosity function evaluation and
integration while keeping halo-model calculations outside LFKit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "evaluate_conditional_lf",
    "integrate_conditional_lf",
    "integrate_weighted_conditional_lf",
]


def evaluate_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    conditional_lf: Callable[[FloatArray, FloatArray], FloatArray],
) -> FloatArray:
    """Evaluate a conditional luminosity function.

    Args:
        absolute_mag: Absolute magnitude values.
        condition: Values of the conditioning variable.
        conditional_lf: Callable returning ``Phi(M | x)``. The callable must
            accept absolute magnitude and condition arrays.

    Returns:
        Conditional luminosity function evaluated at the requested absolute
        magnitude and condition values.

    Raises:
        ValueError: If the inputs or evaluated conditional luminosity function
            contain non-finite values, or if the evaluated conditional
            luminosity function contains negative values.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    condition_arr = validate_array(condition, name="condition")

    phi = np.asarray(conditional_lf(absolute_mag_arr, condition_arr), dtype=float)

    if not np.all(np.isfinite(phi)):
        raise ValueError("conditional_lf returned NaN or infinite values.")

    if np.any(phi < 0.0):
        raise ValueError(
            "conditional_lf returned negative values, which are not allowed."
        )

    return np.asarray(phi, dtype=np.float64)


def integrate_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    conditional_lf: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    axis: int = -1,
) -> FloatArray:
    """Integrate a conditional luminosity function over absolute magnitude.

    This computes ``integral Phi(M | x) dM``, where ``x`` is the conditioning
    variable.

    Args:
        absolute_mag: Absolute magnitude grid.
        condition: Values of the conditioning variable.
        conditional_lf: Callable returning ``Phi(M | x)``.
        axis: Axis corresponding to the absolute magnitude grid.

    Returns:
        Conditional luminosity function integrated over absolute magnitude.

    Raises:
        ValueError: If the inputs or evaluated conditional luminosity function
            contain invalid values.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")

    phi = evaluate_conditional_lf(
        absolute_mag=absolute_mag_arr,
        condition=condition,
        conditional_lf=conditional_lf,
    )

    return np.asarray(
        np.trapezoid(phi, x=absolute_mag_arr, axis=axis),
        dtype=np.float64,
    )


def integrate_weighted_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    conditional_lf: Callable[[FloatArray, FloatArray], FloatArray],
    weight: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    axis: int = -1,
) -> FloatArray:
    """Integrate a weighted conditional luminosity function.

    This computes ``integral w(M, x) Phi(M | x) dM``, where ``x`` is the
    conditioning variable.

    Args:
        absolute_mag: Absolute magnitude grid.
        condition: Values of the conditioning variable.
        conditional_lf: Callable returning ``Phi(M | x)``.
        weight: Callable returning weights ``w(M, x)``.
        axis: Axis corresponding to the absolute magnitude grid.

    Returns:
        Weighted conditional luminosity function integrated over absolute
        magnitude.

    Raises:
        ValueError: If the inputs, evaluated conditional luminosity function,
            or weights contain invalid values.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    condition_arr = validate_array(condition, name="condition")

    phi = evaluate_conditional_lf(
        absolute_mag=absolute_mag_arr,
        condition=condition_arr,
        conditional_lf=conditional_lf,
    )

    weight_arr = np.asarray(weight(absolute_mag_arr, condition_arr), dtype=float)

    if not np.all(np.isfinite(weight_arr)):
        raise ValueError("weight returned NaN or infinite values.")

    weighted_phi = np.asarray(weight_arr * phi, dtype=float)

    return np.asarray(
        np.trapezoid(weighted_phi, x=absolute_mag_arr, axis=axis),
        dtype=np.float64,
    )
