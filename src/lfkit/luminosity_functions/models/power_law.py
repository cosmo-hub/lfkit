"""Power law luminosity function models."""

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
    r"""Return a single power law luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1},

    where

    .. math::

        x = 10^{-0.4(M - M_\star)} = L/L_\star.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Non-negative normalization.
        m_star: Characteristic magnitude used to define ``x``.
        alpha: power law slope in luminosity space.

    Returns:
        NumPy array containing the power law luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative.
    """
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
    r"""Return a double-power law luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        \frac{0.4 \ln(10) \, \phi_\star}
        {x^{-(\alpha + 1)} + x^{-(\beta + 1)}},

    where

    .. math::

        x = 10^{-0.4(M - M_\star)} = L/L_\star.

    The two slopes control the asymptotic behaviour on either side of
    ``m_star``. This form is useful when a luminosity function behaves like
    one power law at the faint end and another at the bright end, with a smooth
    turnover between them.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Non-negative normalization.
        m_star: Characteristic magnitude where the transition occurs.
        alpha: Faint-end power law slope.
        beta: Bright-end power law slope.

    Returns:
        NumPy array containing the double-power law luminosity function
        evaluated at ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative.
    """
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
    r"""Return a sharply broken power law luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        0.4 \ln(10) \, \phi_\star
        \begin{cases}
            x^{\alpha_{\mathrm{faint}} + 1}, & x < 1, \\
            x^{\alpha_{\mathrm{bright}} + 1}, & x \ge 1,
        \end{cases}

    where

    .. math::

        x = 10^{-0.4(M - M_\star)} = L/L_\star.

    Since smaller luminosity means larger magnitude, the ``x < 1`` branch
    corresponds to the faint side of the break and the ``x >= 1`` branch
    corresponds to the bright side.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Non-negative normalization.
        m_star: Break magnitude used to define ``x``.
        alpha_faint: power law slope used for the faint side.
        alpha_bright: power law slope used for the bright side.

    Returns:
        NumPy array containing the broken-power law luminosity function
        evaluated at ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative.
    """
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
    r"""Return a power law luminosity function using log10 normalization.

    This is a convenience wrapper around :func:`power_law_lf` with

    .. math::

        \phi_\star = 10^{\log_{10}\phi_\star}.

    The evaluated luminosity function is therefore

    .. math::

        \phi(M) =
        0.4 \ln(10) \, 10^{\log_{10}\phi_\star}
        x^{\alpha + 1},

    where

    .. math::

        x = 10^{-0.4(M - M_\star)} = L/L_\star.

    Args:
        absolute_mag: Absolute magnitude value(s).
        log_phi_star: Base-10 logarithm of the normalization.
        m_star: Characteristic magnitude used to define ``x``.
        alpha: power law slope in luminosity space.

    Returns:
        NumPy array containing the power law luminosity function evaluated at
        ``absolute_mag``.
    """
    phi_star = 10.0 ** validate_array(log_phi_star, name="log_phi_star")

    return power_law_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )
