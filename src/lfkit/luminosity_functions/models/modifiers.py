"""Generic luminosity function modifiers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.photometry.luminosities import luminosity_ratio
from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "apply_luminosity_cutoff",
]


def apply_luminosity_cutoff(
    absolute_mag: FloatInput,
    *,
    base_lf: Callable[..., FloatArray],
    m_star: ParameterValue,
    cutoff_power: ParameterValue = 2.0,
    cutoff_amplitude: ParameterValue = 1.0,
    **base_lf_parameters: ParameterValue,
) -> FloatArray:
    """Return a luminosity function multiplied by a luminosity-ratio cutoff."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    cutoff_power_arr = validate_array(cutoff_power, name="cutoff_power")
    cutoff_amplitude_arr = validate_array(
        cutoff_amplitude,
        name="cutoff_amplitude",
    )

    if np.any(cutoff_power_arr <= 0.0):
        raise ValueError("cutoff_power must be positive.")

    if np.any(cutoff_amplitude_arr < 0.0):
        raise ValueError("cutoff_amplitude must be non-negative.")

    x = luminosity_ratio(absolute_mag_arr, m_star)

    base_phi = base_lf(
        absolute_mag_arr,
        m_star=m_star,
        **base_lf_parameters,
    )

    modifier = np.exp(-cutoff_amplitude_arr * x**cutoff_power_arr)

    return np.asarray(base_phi * modifier, dtype=float)
