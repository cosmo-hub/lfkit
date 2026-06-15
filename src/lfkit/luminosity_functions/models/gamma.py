r"""Gamma-family luminosity function models.

This module provides standalone functions for evaluating gamma-like luminosity
function models in rest frame absolute magnitude space.

Implemented models:
    - Gamma luminosity function
    - Generalized gamma luminosity function
"""

from __future__ import annotations

import numpy as np

from lfkit.utils.integrators import safe_power10
from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "gamma_lf",
    "generalized_gamma_lf",
]


def gamma_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
) -> FloatArray:
    r"""Return a gamma luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        0.4 \ln(10) \, \phi_\star \,
        x^{\alpha + 1} \exp(-x),

    where

    .. math::

        x = 10^{-0.4(M - M_\star)}.

    This is mathematically equivalent to the standard Schechter luminosity
    function, but is provided as a gamma-family model for users who want to
    describe the luminosity function as a gamma distribution in luminosity
    space.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Gamma luminosity function normalization.
        m_star: Characteristic magnitude.
        alpha: Low-luminosity power-law slope.

    Returns:
        NumPy array containing the gamma luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    x = safe_power10(-0.4 * (absolute_mag_arr - m_star))

    phi = 0.4 * np.log(10.0) * phi_star_arr * x ** (alpha_arr + 1.0) * np.exp(-x)

    return np.asarray(phi, dtype=float)


def generalized_gamma_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    beta: ParameterValue,
) -> FloatArray:
    r"""Return a generalized gamma luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        0.4 \ln(10) \, \beta \, \phi_\star \,
        x^{\alpha + 1} \exp(-x^\beta),

    where

    .. math::

        x = 10^{-0.4(M - M_\star)}.

    The gamma or Schechter form is recovered when ``beta = 1``. Values
    ``beta < 1`` produce a shallower bright-end cutoff, while values
    ``beta > 1`` produce a sharper bright-end cutoff.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Generalized gamma luminosity function normalization.
        m_star: Characteristic magnitude.
        alpha: Low-luminosity power-law slope.
        beta: Positive bright-end cutoff shape parameter.

    Returns:
        NumPy array containing the generalized gamma luminosity function
        evaluated at ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative or if ``beta`` is not positive.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")
    beta_arr = validate_array(beta, name="beta")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be positive.")

    x = safe_power10(-0.4 * (absolute_mag_arr - m_star))

    phi = (
        0.4
        * np.log(10.0)
        * beta_arr
        * phi_star_arr
        * x ** (alpha_arr + 1.0)
        * np.exp(-(x**beta_arr))
    )

    return np.asarray(phi, dtype=float)
