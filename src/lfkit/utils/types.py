"""Shared type aliases for LFKit."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
FloatInput: TypeAlias = float | Sequence[float] | FloatArray

ParameterValue: TypeAlias = FloatInput
ParameterModel: TypeAlias = Callable[..., FloatArray]

Cosmology: TypeAlias = Any
LuminosityFunction: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]

__all__ = [
    "Cosmology",
    "FloatArray",
    "FloatInput",
    "LuminosityFunction",
    "ParameterModel",
    "ParameterValue",
]
