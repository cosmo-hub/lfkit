"""Validation utilities."""

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias

FloatArray: TypeAlias = NDArray[np.float64]
ParameterValue: TypeAlias = float | FloatArray

__all__ = [
    "validate_array",
]


def validate_array(
    x: ParameterValue,
    *,
    name: str,
    allow_negative: bool = True,
) -> NDArray[np.float64]:
    """Basic validation for numeric arrays."""
    arr = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not allow_negative and np.any(arr < 0):
        raise ValueError(f"{name} contains negative values, which are not allowed.")

    return np.asarray(arr, dtype=np.float64)
