"""Power-law luminosity function models."""

from __future__ import annotations

import numpy as np

from lfkit.photometry.luminosities import luminosity_ratio
from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "power_law_lf",
    "double_power_law_lf",
    "broken_power_law_lf",
    "log_power_law_lf",
]


def power_law_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
) -> FloatArray:
    """Return a single power-law luminosity function."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    x = luminosity_ratio(absolute_mag_arr, m_star)

    phi = 0.4 * np.log(10.0) * phi_star_arr * x ** (alpha_arr + 1.0)

    return np.asarray(phi, dtype=float)


def double_power_law_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    beta: ParameterValue,
) -> FloatArray:
    """Return a double power-law luminosity function."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")
    beta_arr = validate_array(beta, name="beta")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    x = luminosity_ratio(absolute_mag_arr, m_star)

    phi = (
        0.4
        * np.log(10.0)
        * phi_star_arr
        / (x ** (-(alpha_arr + 1.0)) + x ** (-(beta_arr + 1.0)))
    )

    return np.asarray(phi, dtype=float)


def broken_power_law_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha_faint: ParameterValue,
    alpha_bright: ParameterValue,
) -> FloatArray:
    """Return a sharply broken power-law luminosity function."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_faint_arr = validate_array(alpha_faint, name="alpha_faint")
    alpha_bright_arr = validate_array(alpha_bright, name="alpha_bright")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    x = luminosity_ratio(absolute_mag_arr, m_star)

    phi = np.where(
        x < 1.0,
        phi_star_arr * x ** (alpha_faint_arr + 1.0),
        phi_star_arr * x ** (alpha_bright_arr + 1.0),
    )
    phi = 0.4 * np.log(10.0) * phi

    return np.asarray(phi, dtype=float)


def log_power_law_lf(
    absolute_mag: FloatInput,
    *,
    log_phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
) -> FloatArray:
    """Return a power-law luminosity function using log10 normalization."""
    phi_star = 10.0 ** validate_array(log_phi_star, name="log_phi_star")

    return power_law_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )
