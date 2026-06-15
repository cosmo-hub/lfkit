"""Saunders luminosity function models."""

from __future__ import annotations

import numpy as np

from lfkit.photometry.luminosities import luminosity_ratio
from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "saunders_lf",
    "evolving_saunders_lf",
    "double_saunders_lf",
    "generalized_saunders_lf",
]


def saunders_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    sigma: ParameterValue,
) -> FloatArray:
    r"""Return a Saunders luminosity function in magnitude space."""
    return generalized_saunders_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        sigma=sigma,
        beta=2.0,
    )


def generalized_saunders_lf(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    sigma: ParameterValue,
    beta: ParameterValue,
) -> FloatArray:
    r"""Return a generalized Saunders luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        0.4 \ln(10) \, \phi_\star \,
        x^{\alpha}
        \exp\left[
            -\left(
            \frac{\log_{10}(1 + x)}
            {\sqrt{2}\sigma}
            \right)^\beta
        \right],

    where

    .. math::

        x = 10^{-0.4(M - M_\star)} = L/L_\star.

    For ``beta = 2``, this reduces to the standard Saunders form.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Non-negative normalization.
        m_star: Characteristic magnitude used to define ``x``.
        alpha: Faint-end luminosity slope parameter.
        sigma: Positive width of the bright-end logarithmic cutoff.
        beta: Positive exponent controlling the bright-end cutoff shape.

    Returns:
        NumPy array containing the generalized Saunders luminosity function
        evaluated at ``absolute_mag``.

    Raises:
        ValueError: If ``phi_star`` is negative, ``sigma`` is not positive,
            or ``beta`` is not positive.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")
    sigma_arr = validate_array(sigma, name="sigma")
    beta_arr = validate_array(beta, name="beta")

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    if np.any(sigma_arr <= 0.0):
        raise ValueError("sigma must be positive.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be positive.")

    x = luminosity_ratio(absolute_mag_arr, m_star)

    cutoff_argument = np.log10(1.0 + x) / (np.sqrt(2.0) * sigma_arr)
    cutoff = np.exp(-(cutoff_argument**beta_arr))
    phi = 0.4 * np.log(10.0) * phi_star_arr * x**alpha_arr * cutoff

    return np.asarray(phi, dtype=float)


def evolving_saunders_lf(
    absolute_mag: FloatInput,
    redshift: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    sigma: ParameterValue,
    p: ParameterValue = 0.0,
    q: ParameterValue = 0.0,
) -> FloatArray:
    r"""Return an evolving Saunders luminosity function.

    This computes a Saunders luminosity function with redshift evolution in
    normalization and characteristic magnitude,

    .. math::

        \phi_\star(z) = \phi_\star (1 + z)^p,

    and

    .. math::

        M_\star(z) = M_\star - qz.

    Args:
        absolute_mag: Absolute magnitude value(s).
        redshift: Redshift value(s).
        phi_star: Non-negative normalization at ``z = 0``.
        m_star: Characteristic magnitude at ``z = 0``.
        alpha: Faint-end luminosity slope parameter.
        sigma: Positive width of the bright-end logarithmic cutoff.
        p: Density evolution exponent.
        q: Linear magnitude evolution coefficient.

    Returns:
        NumPy array containing the evolving Saunders luminosity function
        evaluated at ``absolute_mag`` and ``redshift``.

    Raises:
        ValueError: If ``redshift`` is negative.
    """
    redshift_arr = validate_array(redshift, name="redshift")
    p_arr = validate_array(p, name="p")
    q_arr = validate_array(q, name="q")

    if np.any(redshift_arr < 0.0):
        raise ValueError("redshift must be non-negative.")

    phi_star_z = (
        validate_array(phi_star, name="phi_star") * (1.0 + redshift_arr) ** p_arr
    )
    m_star_z = validate_array(m_star, name="m_star") - q_arr * redshift_arr

    return saunders_lf(
        absolute_mag,
        phi_star=phi_star_z,
        m_star=m_star_z,
        alpha=alpha,
        sigma=sigma,
    )


def double_saunders_lf(
    absolute_mag: FloatInput,
    *,
    phi_star_1: ParameterValue,
    m_star_1: ParameterValue,
    alpha_1: ParameterValue,
    sigma_1: ParameterValue,
    phi_star_2: ParameterValue,
    m_star_2: ParameterValue,
    alpha_2: ParameterValue,
    sigma_2: ParameterValue,
) -> FloatArray:
    r"""Return a two-component Saunders luminosity function.

    This computes

    .. math::

        \phi(M) = \phi_1(M) + \phi_2(M),

    where each component is a Saunders luminosity function with its own
    normalization, characteristic magnitude, faint-end slope, and bright-end
    cutoff width.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star_1: Non-negative normalization of the first component.
        m_star_1: Characteristic magnitude of the first component.
        alpha_1: Faint-end luminosity slope parameter of the first component.
        sigma_1: Positive bright-end cutoff width of the first component.
        phi_star_2: Non-negative normalization of the second component.
        m_star_2: Characteristic magnitude of the second component.
        alpha_2: Faint-end luminosity slope parameter of the second component.
        sigma_2: Positive bright-end cutoff width of the second component.

    Returns:
        NumPy array containing the summed Saunders luminosity function
        evaluated at ``absolute_mag``.
    """
    phi_1 = saunders_lf(
        absolute_mag,
        phi_star=phi_star_1,
        m_star=m_star_1,
        alpha=alpha_1,
        sigma=sigma_1,
    )
    phi_2 = saunders_lf(
        absolute_mag,
        phi_star=phi_star_2,
        m_star=m_star_2,
        alpha=alpha_2,
        sigma=sigma_2,
    )

    return np.asarray(phi_1 + phi_2, dtype=float)
