"""Validation utilities."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput

__all__ = [
    "validate_array",
    "validate_luminosity_distance",
    "validate_magnitude_range",
]


def validate_array(
    x: FloatInput,
    *,
    name: str,
    allow_negative: bool = True,
) -> FloatArray:
    """Return a finite float array."""
    arr = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not allow_negative and np.any(arr < 0):
        raise ValueError(f"{name} contains negative values, which are not allowed.")

    return np.asarray(arr, dtype=np.float64)


def validate_luminosity_distance(
    luminosity_distance_mpc: FloatInput,
) -> FloatArray:
    """Return finite positive luminosity distances in Mpc."""
    distance = validate_array(
        luminosity_distance_mpc,
        name="luminosity_distance_mpc",
        allow_negative=False,
    )

    if np.any(distance <= 0.0):
        raise ValueError("luminosity_distance_mpc must contain positive values.")

    return distance


def validate_magnitude_range(
    *,
    m_bright: float,
    m_faint: float,
) -> None:
    """Validate bright and faint magnitude bounds."""
    if not np.isfinite(m_bright):
        raise ValueError("m_bright must be finite.")

    if not np.isfinite(m_faint):
        raise ValueError("m_faint must be finite.")

    if m_faint <= m_bright:
        raise ValueError("m_faint must be larger than m_bright.")
