"""Gaussian-like luminosity function models."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "gaussian_lf",
    "lognormal_lf",
]


def gaussian_lf(
    absolute_mag: FloatInput,
    *,
    mean_absolute_mag: ParameterValue,
    sigma_absolute_mag: ParameterValue,
    amplitude: ParameterValue = 1.0,
) -> FloatArray:
    """Return a Gaussian luminosity function in magnitude space."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    mean_absolute_mag_arr = validate_array(
        mean_absolute_mag,
        name="mean_absolute_mag",
    )
    sigma_absolute_mag_arr = validate_array(
        sigma_absolute_mag,
        name="sigma_absolute_mag",
    )
    amplitude_arr = validate_array(amplitude, name="amplitude")

    if np.any(sigma_absolute_mag_arr <= 0.0):
        raise ValueError("sigma_absolute_mag must be positive.")

    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude must be non-negative.")

    phi = (
        amplitude_arr
        / (np.sqrt(2.0 * np.pi) * sigma_absolute_mag_arr)
        * np.exp(
            -0.5
            * ((absolute_mag_arr - mean_absolute_mag_arr) / sigma_absolute_mag_arr)
            ** 2.0
        )
    )

    return np.asarray(phi, dtype=float)


def lognormal_lf(
    absolute_mag: FloatInput,
    *,
    mean_absolute_mag: ParameterValue,
    sigma_log_luminosity: ParameterValue,
    amplitude: ParameterValue = 1.0,
) -> FloatArray:
    """Return a lognormal luminosity function in magnitude space."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    mean_absolute_mag_arr = validate_array(
        mean_absolute_mag,
        name="mean_absolute_mag",
    )
    sigma_log_luminosity_arr = validate_array(
        sigma_log_luminosity,
        name="sigma_log_luminosity",
    )
    amplitude_arr = validate_array(amplitude, name="amplitude")

    if np.any(sigma_log_luminosity_arr <= 0.0):
        raise ValueError("sigma_log_luminosity must be positive.")

    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude must be non-negative.")

    delta_log_luminosity = -0.4 * (absolute_mag_arr - mean_absolute_mag_arr)

    phi = (
        amplitude_arr
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity_arr)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity_arr) ** 2.0)
    )

    return np.asarray(phi, dtype=float)
