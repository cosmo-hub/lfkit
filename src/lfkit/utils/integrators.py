"""Numerical integration utilities.

This module provides small reusable helpers for integrating tabulated values
over fixed or variable finite bounds. These helpers do not encode any
luminosity function, photometry, or cosmology assumptions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array


__all__ = [
    "integrate_between_variable_bounds",
    "safe_divide",
]


def integrate_between_variable_bounds(
    y: FloatInput,
    *,
    lower: FloatInput,
    upper: FloatInput,
    integrand_fn: Callable[[FloatArray, FloatArray], FloatArray],
    n_grid: int = 512,
    y_name: str = "y",
    lower_name: str = "lower",
    upper_name: str = "upper",
    n_grid_name: str = "n_grid",
) -> FloatArray:
    """Integrate a callable between finite bounds that may vary with ``y``.

    Args:
        y: Coordinate values at which to evaluate the integral.
        lower: Lower integration bound. May be scalar or array-like.
        upper: Upper integration bound. May be scalar or array-like.
        integrand_fn: Callable evaluated as ``integrand_fn(x_grid, y_grid)``.
        n_grid: Number of integration points.
        y_name: Name used for ``y`` in error messages.
        lower_name: Name used for ``lower`` in error messages.
        upper_name: Name used for ``upper`` in error messages.
        n_grid_name: Name used for ``n_grid`` in error messages.

    Returns:
        Integral values with the broadcast shape of ``y``, ``lower``, and
        ``upper``.
    """
    if n_grid < 2:
        raise ValueError(f"{n_grid_name} must be at least 2")

    y_arr = validate_array(y, name=y_name)

    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    if np.any(~np.isfinite(lower_arr)):
        raise ValueError(_bound_finite_error_message(lower_name))

    if np.any(~np.isfinite(upper_arr)):
        raise ValueError(_bound_finite_error_message(upper_name))

    lower_arr = validate_array(lower_arr, name=lower_name)
    upper_arr = validate_array(upper_arr, name=upper_name)

    y_arr, lower_arr, upper_arr = np.broadcast_arrays(y_arr, lower_arr, upper_arr)

    width = upper_arr - lower_arr
    empty = width <= 0.0

    t_grid = np.linspace(0.0, 1.0, n_grid, dtype=float)
    grid_shape = (n_grid,) + (1,) * lower_arr.ndim

    x_grid = lower_arr[None, ...] + t_grid.reshape(grid_shape) * width[None, ...]
    y_grid = np.broadcast_to(y_arr[None, ...], x_grid.shape)

    values = np.asarray(integrand_fn(x_grid, y_grid), dtype=float)

    try:
        values = np.broadcast_to(values, x_grid.shape)
    except ValueError as exc:
        raise ValueError(
            "integrand_fn(x, y) must return values broadcastable "
            "to the integration grid shape"
        ) from exc

    if np.any(~np.isfinite(values)):
        raise ValueError("integrand_fn(x, y) returned non-finite values")

    result = np.trapezoid(values, x=x_grid, axis=0)
    return np.where(empty, 0.0, result)


def safe_divide(
    numerator: FloatInput,
    denominator: FloatInput,
    *,
    fill_value: float = 0.0,
) -> FloatArray:
    """Divide arrays safely, replacing invalid or zero-denominator results.

    Args:
        numerator: Values in the numerator.
        denominator: Values in the denominator.
        fill_value: Value returned where the denominator is zero or where
            the division would produce a non-finite result.

    Returns:
        Broadcasted division result with invalid entries replaced by
        ``fill_value``.
    """
    numerator_arr, denominator_arr = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )

    result = np.full(numerator_arr.shape, fill_value, dtype=float)

    valid = denominator_arr > 0.0
    np.divide(
        numerator_arr,
        denominator_arr,
        out=result,
        where=valid,
    )

    result = np.where(np.isfinite(result), result, fill_value)
    return result


def _variable_bounds_grid(
    *,
    lower: FloatArray,
    upper: FloatArray,
    n_grid: int,
) -> FloatArray:
    """Return a column-wise grid between variable lower and upper bounds."""
    t = np.linspace(0.0, 1.0, int(n_grid), dtype=float)

    return np.asarray(
        lower[None, :] + t[:, None] * (upper[None, :] - lower[None, :]),
        dtype=float,
    )


def _validate_variable_bound_inputs(
    *,
    lower: FloatArray,
    upper: FloatArray,
    n_grid: int,
) -> None:
    """Validate finite variable-bound integration inputs."""
    if np.any(~np.isfinite(lower)):
        raise ValueError("lower must contain only finite values.")

    if np.any(~np.isfinite(upper)):
        raise ValueError("upper must contain only finite values.")

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")


def _bound_finite_error_message(name: str) -> str:
    """Return the finite-bound validation message for a named bound."""
    if name in {"lower", "upper"}:
        return f"{name} must contain only finite values."

    return f"{name} contains NaN or infinite values."
