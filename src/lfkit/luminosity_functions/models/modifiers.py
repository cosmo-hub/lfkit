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
    """Return a luminosity function multiplied by a luminosity ratio cutoff.

    The modifier has the form ``exp(-A x**p)``, where ``x`` is the luminosity ratio
    relative to ``m_star``, ``A`` is ``cutoff_amplitude``, and ``p`` is
    ``cutoff_power``.

    Args:
        absolute_mag: Absolute magnitude value or array.
        base_lf: Base luminosity function model to modify.
        m_star: Characteristic magnitude used to compute the luminosity ratio.
        cutoff_power: Positive power applied to the luminosity ratio.
        cutoff_amplitude: Non-negative amplitude multiplying the cutoff term.
        **base_lf_parameters: Parameters passed to ``base_lf``.

    Returns:
        Modified luminosity function evaluated at ``absolute_mag``.

    Raises:
        ValueError: If ``cutoff_power`` is not positive or if ``cutoff_amplitude``
            is negative.
    """
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
