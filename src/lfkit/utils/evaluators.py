"""Callable evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.utils.types import FloatArray


__all__ = [
    "evaluate_non_negative_redshift_callable",
    "evaluate_optional_redshift_callable",
    "evaluate_positive_redshift_callable",
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
