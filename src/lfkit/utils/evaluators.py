"""Callable evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.utils.types import FloatArray, LuminosityFunction


__all__ = [
    "evaluate_non_negative_redshift_callable",
    "evaluate_optional_redshift_callable",
    "evaluate_positive_redshift_callable",
    "evaluate_weight_on_grid",
    "evaluate_lf_on_grid",
]


def evaluate_optional_redshift_callable(
    fn: Callable[[FloatArray], FloatArray] | None,
    z: FloatArray,
    *,
    name: str,
) -> FloatArray | None:
    """Evaluate an optional redshift-dependent callable.

    Args:
        fn: Callable to evaluate. If None, None is returned.
        z: Redshift array passed to the callable.
        name: Name used in error messages.

    Returns:
        Callable values with the same shape as ``z``, or None.
    """
    if fn is None:
        return None

    return _evaluate_redshift_callable(fn, z, name=name)


def evaluate_positive_redshift_callable(
    fn: Callable[[FloatArray], FloatArray],
    z: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Evaluate a redshift-dependent callable that must be positive.

    Args:
        fn: Callable to evaluate.
        z: Redshift array passed to the callable.
        name: Name used in error messages.

    Returns:
        Positive callable values with the same shape as ``z``.
    """
    values = _evaluate_redshift_callable(fn, z, name=name)

    if np.any(values <= 0.0):
        raise ValueError(f"{name}(z) must return positive values.")

    return values


def evaluate_non_negative_redshift_callable(
    fn: Callable[[FloatArray], FloatArray],
    z: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Evaluate a redshift-dependent callable that must be non-negative.

    Args:
        fn: Callable to evaluate.
        z: Redshift array passed to the callable.
        name: Name used in error messages.

    Returns:
        Non-negative callable values with the same shape as ``z``.
    """
    values = _evaluate_redshift_callable(fn, z, name=name)

    if np.any(values < 0.0):
        raise ValueError(f"{name}(z) must return non-negative values.")

    return values


def evaluate_weight_on_grid(
    weight_fn: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    m_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    r"""Return finite non-negative weight values on a magnitude-redshift grid."""
    weight = np.asarray(weight_fn(m_grid, z_grid), dtype=float)

    if weight.shape != m_grid.shape:
        try:
            weight = np.broadcast_to(weight, m_grid.shape)
        except ValueError as exc:
            raise ValueError(
                "weight_fn(M, z) must return values broadcastable to the shape "
                "of the magnitude-redshift integration grid."
            ) from exc

    if np.any(~np.isfinite(weight)):
        raise ValueError("weight_fn(M, z) returned non-finite values.")

    if np.any(weight < 0.0):
        raise ValueError("weight_fn(M, z) must be non-negative.")

    return np.asarray(weight, dtype=float)


def evaluate_lf_on_grid(
    lf: LuminosityFunction,
    *,
    m_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    r"""Return LF values evaluated on a magnitude-redshift grid."""
    phi = np.asarray(lf(m_grid, z_grid), dtype=float)

    if phi.shape != m_grid.shape:
        try:
            phi = np.broadcast_to(phi, m_grid.shape)
        except ValueError as exc:
            raise ValueError(
                "lf(M, z) must return values broadcastable to the shape "
                "of the magnitude-redshift integration grid."
            ) from exc

    if np.any(~np.isfinite(phi)):
        raise ValueError("lf(M, z) returned non-finite values.")

    if np.any(phi < 0.0):
        raise ValueError("lf(M, z) must be non-negative.")

    return np.asarray(phi, dtype=float)


def _evaluate_redshift_callable(
    fn: Callable[[FloatArray], FloatArray],
    z: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Evaluate a redshift callable and validate shape and finite values."""
    values = np.asarray(fn(z), dtype=float)

    if values.shape != z.shape:
        raise ValueError(f"{name}(z) must return an array with the same shape as z.")

    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name}(z) returned non-finite values.")

    return np.asarray(values, dtype=float)
