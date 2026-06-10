r"""Luminosity function utilities for LFKit.

This module provides simple standalone functions for evaluating
common galaxy luminosity function parameterization.

All luminosity functions in this module are defined in rest-frame absolute
magnitude space. Functions with names ending in ``_from_m`` are convenience
wrappers that accept apparent magnitudes, convert them to absolute
magnitudes, and then evaluate the luminosity function in absolute magnitude
space.

Implemented models:
    - Standard Schechter LF in rest-frame absolute magnitudes
    - Schechter LF with configurable redshift evolution of parameters
    - Double-power-law Schechter LF in rest-frame absolute magnitudes

The magnitude-space Schechter function is

.. math::

    \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1} \exp(-x),

where

.. math::

    x = 10^{-0.4 (M - M_\star)}.

Redshift evolution is handled by separate helper functions for
``phi_star(z)``, ``M_star(z)``, and ``alpha(z)``. Built-in options include
constant evolution and common linearized forms such as

.. math::

    M_\star(z) = M_{0,\star} - q (z - z_{\mathrm{ref}})

.. math::

    \phi_\star(z) = \phi_{0,\star} \, 10^{0.4 p z}.

For more information see The CNOC2 Field Galaxy Luminosity Function I:
A Description of Luminosity Function Evolution by Lin et al. 1999
(arXiv:9902249) or GAMA: ugriz galaxy luminosity functions by Loveday et al.
(arXiv:1111.0166).

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from scipy.special import gammaincc, gamma

from lfkit.luminosity_functions.parameter_models import evaluate_lf_parameters
from lfkit.photometry.luminosities import luminosity_ratio
from lfkit.photometry.magnitudes import absolute_magnitude
from lfkit.utils.types import Cosmology, FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "schechter",
    "evolving_schechter",
    "double_schechter",
    "schechter_from_m",
    "evolving_schechter_from_m",
    "double_schechter_from_m",
    "schechter_cumulative",
    "schechter_cumulative_evolving",
]


def schechter(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
) -> FloatArray:
    r"""Return the standard Schechter luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1} \exp(-x),

    where

    .. math::

        x = 10^{-0.4 (M - M_\star)}.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Schechter normalization.
        m_star: Characteristic magnitude.
        alpha: Faint-end slope. Can be a scalar or array-like.

    Returns:
        NumPy array of luminosity function values evaluated at
        ``absolute_mag``.
    """
    x = luminosity_ratio(absolute_mag, m_star)
    # avoid x=0 issues
    x = np.clip(x, 1e-300, None)

    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha_arr = validate_array(alpha, name="alpha")

    if np.any(phi_star_arr < 0):
        raise ValueError("phi_star must be non-negative.")

    if np.any(phi_star_arr == 0):
        warnings.warn("phi_star is zero; LF will be identically zero.", stacklevel=2)

    prefactor = 0.4 * np.log(10.0) * phi_star_arr

    phi = prefactor * x ** (alpha_arr + 1.0) * np.exp(-x)

    return np.asarray(phi, dtype=float)


def evolving_schechter(
    absolute_mag: FloatInput,
    z: FloatInput,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
) -> FloatArray:
    r"""Return an evolving Schechter luminosity function in magnitude space.

    This evaluates

    .. math::

        \phi(M, z) = 0.4 \ln(10) \, \phi_\star(z) \, x(M, z)^{\alpha(z) + 1} \exp(-x(M, z)),

    where

    .. math::

        x(M, z) = 10^{-0.4 (M - M_\star(z))}.

    Args:
        absolute_mag: Absolute magnitude value(s).
        z: Redshift value or array-like of redshift values.
        phi_model: Evolution model for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected
            ``phi_star`` evolution model.
        m_star_model: Evolution model for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected
            ``M_star`` evolution model.
        alpha_model: Evolution model for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected
            ``alpha`` evolution model.

    Returns:
        NumPy array of luminosity function values evaluated at
        ``absolute_mag`` and ``z``.

    Raises:
        ValueError: If an unsupported evolution model is requested.
    """
    phi_star, m_star, alpha = evaluate_lf_parameters(
        z,
        phi_model=phi_model,
        phi_kwargs=phi_kwargs,
        m_star_model=m_star_model,
        m_star_kwargs=m_star_kwargs,
        alpha_model=alpha_model,
        alpha_kwargs=alpha_kwargs,
    )

    return schechter(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )


def double_schechter(
    absolute_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: float,
    beta: float,
    m_transition: ParameterValue,
) -> FloatArray:
    r"""Return a double-power-law Schechter luminosity function in magnitude space.

    This implements the Loveday/GAMA-style faint-end extension

    .. math::

        \phi(L) = \phi_\star (L/L_\star)^\alpha \exp(-L/L_\star)
                 \times \left[1 + (L/L_t)^\beta\right]

    converted into a per-magnitude luminosity function.

    Since

    .. math::

        \phi(M) \, dM = \phi(L) \, dL

    and

    .. math::

        \frac{dL}{dM} = -0.4 \ln(10) \, L,

    the magnitude-space form becomes

    .. math::

        \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1} \exp(-x)
                 \times \left[1 + (x / x_t)^\beta\right]

    with :math:`x = L/L_\star` and :math:`x_t = L_t/L_\star`.

    Args:
        absolute_mag: Absolute magnitude value(s).
        phi_star: Overall normalization.
        m_star: Characteristic magnitude M_star.
        alpha: Bright/intermediate faint-end slope parameter.
        beta: Additional faint-end slope modifier.
        m_transition: Transition magnitude M_t corresponding to L_t.

    Returns:
        NumPy array of luminosity function values evaluated at
        ``absolute_mag``.
    """
    absolute_mag = validate_array(absolute_mag, name="absolute_mag")
    phi_star_arr = validate_array(phi_star, name="phi_star")
    alpha = float(alpha)
    beta = float(beta)

    if np.any(phi_star_arr == 0):
        warnings.warn(
            "phi_star is zero; LF will be identically zero.",
            stacklevel=2,
        )

    if np.any(phi_star_arr < 0):
        raise ValueError("phi_star must be non-negative.")

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")

    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")

    x = luminosity_ratio(absolute_mag, m_star)
    x_t = luminosity_ratio(m_transition, m_star)

    # prevent division blow-up
    x_t = np.clip(x_t, 1e-300, None)

    prefactor = 0.4 * np.log(10.0) * phi_star_arr
    modifier = 1.0 + (x / x_t) ** beta

    return np.asarray(
        prefactor * x ** (alpha + 1.0) * np.exp(-x) * modifier,
        dtype=float,
    )


def schechter_from_m(
    cosmo_obj: Cosmology,
    z: FloatInput,
    apparent_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: ParameterValue,
    h: float | None = None,
    k_correction: ParameterValue | None = None,
    e_correction: ParameterValue | None = None,
) -> FloatArray:
    r"""Return the standard Schechter luminosity function from apparent magnitudes.

    This uses

    .. math::

        M = m - \mu - K + E,

    followed by

    .. math::

        \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1} \exp(-x),

    where

    .. math::

        x = 10^{-0.4 (M - M_\star)}.

    This is a convenience wrapper; the luminosity function itself is still
    defined and evaluated in rest-frame absolute magnitude space.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        apparent_mag: Observed apparent magnitude value(s), which are
            converted to rest-frame absolute magnitudes before LF evaluation.
        phi_star: Schechter normalization.
        m_star: Characteristic magnitude.
        alpha: Faint-end slope.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of luminosity function values corresponding to the
        supplied apparent magnitudes.
    """
    z = validate_array(z, name="z")

    if np.any(z < 0):
        raise ValueError("Redshift z must be >= 0.")

    abs_mag = absolute_magnitude(
        cosmo_obj,
        z,
        apparent_mag,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    abs_mag = validate_array(abs_mag, name="abs_mag")

    return schechter(
        abs_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )


def evolving_schechter_from_m(
    cosmo_obj: Cosmology,
    z: FloatInput,
    apparent_mag: FloatInput,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
    h: float | None = None,
    k_correction: ParameterValue | None = None,
    e_correction: ParameterValue | None = None,
) -> FloatArray:
    r"""Return an evolving Schechter luminosity function from apparent magnitudes.

    This uses

    .. math::

        M = m - \mu - K + E,

    followed by

    .. math::

        \phi(M, z) = 0.4 \ln(10) \, \phi_\star(z) \, x(M, z)^{\alpha(z) + 1} \exp(-x(M, z)),

    where

    .. math::

        x(M, z) = 10^{-0.4 (M - M_\star(z))}.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        apparent_mag: Apparent magnitude value(s).
        phi_model: Evolution model for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected
            ``phi_star`` evolution model.
        m_star_model: Evolution model for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected
            ``M_star`` evolution model.
        alpha_model: Evolution model for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected
            ``alpha`` evolution model.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of luminosity function values corresponding to the
        supplied apparent magnitudes.
    """
    z = validate_array(z, name="z")
    m_star_kwargs_dict: dict[str, ParameterValue] = (
        {} if m_star_kwargs is None else dict(m_star_kwargs)
    )

    if np.any(z < 0):
        raise ValueError("Redshift z must be >= 0.")

    q_value = m_star_kwargs_dict.get("q", 0.0)
    has_nonzero_q = bool(np.any(np.asarray(q_value) != 0.0))

    if (m_star_model == "linear_q") and has_nonzero_q and (e_correction is not None):
        warnings.warn(
            "You are using an evolving LF with m_star_model='linear_q' "
            "and also providing an e_correction. This may double-count "
            "luminosity evolution.",
            UserWarning,
            stacklevel=2,
        )

    abs_mag = absolute_magnitude(
        cosmo_obj,
        z,
        apparent_mag,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    abs_mag = validate_array(abs_mag, name="abs_mag")

    return evolving_schechter(
        abs_mag,
        z,
        phi_model=phi_model,
        phi_kwargs=phi_kwargs,
        m_star_model=m_star_model,
        m_star_kwargs=m_star_kwargs_dict,
        alpha_model=alpha_model,
        alpha_kwargs=alpha_kwargs,
    )


def double_schechter_from_m(
    cosmo_obj: Cosmology,
    z: FloatInput,
    apparent_mag: FloatInput,
    *,
    phi_star: ParameterValue,
    m_star: ParameterValue,
    alpha: float,
    beta: float,
    m_transition: ParameterValue,
    h: float | None = None,
    k_correction: ParameterValue | None = None,
    e_correction: ParameterValue | None = None,
) -> FloatArray:
    r"""Return a double-power-law Schechter luminosity function from apparent magnitudes.

    This uses

    .. math::

        M = m - \mu - K + E,

    followed by

    .. math::

        \phi(M) = 0.4 \ln(10) \, \phi_\star \, x^{\alpha + 1} \exp(-x)
                 \times \left[1 + (x / x_t)^\beta\right],

    where

    .. math::

        x = 10^{-0.4 (M - M_\star)}

    .. math::

        x_t = 10^{-0.4 (M_t - M_\star)}.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        apparent_mag: Apparent magnitude value(s).
        phi_star: Overall normalization.
        m_star: Characteristic magnitude M_star.
        alpha: Bright/intermediate faint-end slope parameter.
        beta: Additional faint-end slope modifier.
        m_transition: Transition magnitude M_t.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of luminosity function values corresponding to the
        supplied apparent magnitudes.
    """
    z = validate_array(z, name="z")

    if np.any(z < 0):
        raise ValueError("Redshift z must be >= 0.")

    abs_mag = absolute_magnitude(
        cosmo_obj,
        z,
        apparent_mag,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    abs_mag = validate_array(abs_mag, name="abs_mag")

    return double_schechter(
        abs_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        beta=beta,
        m_transition=m_transition,
    )


def schechter_cumulative(
    magnitude_limit: FloatInput,
    *,
    phi_star: float,
    m_star: float,
    alpha: float,
    brighter_than: bool = True,
) -> FloatArray:
    r"""Return the cumulative number density for a standard Schechter LF.

    For a magnitude threshold :math:`M_{\mathrm{lim}}`, define

    .. math::

        x_{\mathrm{lim}} = 10^{-0.4 (M_{\mathrm{lim}} - M_\star)}.

    Then the cumulative number density of galaxies brighter than
    the threshold is

    .. math::

        n(M < M_{\mathrm{lim}}) = \phi_\star \, \Gamma(\alpha + 1, x_{\mathrm{lim}}),

    Then the cumulative number density of galaxies fainter than
    the threshold is

    .. math::

        n(M > M_{\mathrm{lim}}) = \phi_\star \, \Gamma(\alpha + 1) - n(M < M_{\mathrm{lim}}).

    Note:
        This analytic helper is provided only for the standard
        Schechter form. The double-power-law case is not included
        here because it is usually cleaner to integrate numerically.

    Args:
        magnitude_limit: Magnitude threshold(s).
        phi_star: Schechter normalization.
        m_star: Characteristic magnitude.
        alpha: Faint-end slope.
        brighter_than: If True, return number density brighter than
            the threshold. If False, return number density fainter
            than the threshold.

    Returns:
        NumPy array of cumulative number densities for the supplied
        magnitude threshold(s).
    """
    m_lim = validate_array(magnitude_limit, name="m_lim")
    x_lim = luminosity_ratio(m_lim, m_star)
    x_lim = np.clip(x_lim, 1e-300, 1e300)

    alpha = float(alpha)
    s = alpha + 1.0

    if s <= 0:
        raise ValueError(
            "Cumulative Schechter LF is undefined for alpha <= -1 "
            "because the integral diverges."
        )

    if not np.isfinite(phi_star):
        raise ValueError("phi_star must be finite.")

    if phi_star < 0:
        raise ValueError("phi_star must be non-negative.")

    total_gamma = gamma(s)
    n_brighter = phi_star * total_gamma * gammaincc(s, x_lim)

    if brighter_than:
        return np.asarray(n_brighter, dtype=float)

    n_total = phi_star * total_gamma
    return np.asarray(n_total - n_brighter, dtype=float)


def schechter_cumulative_evolving(
    magnitude_limit: FloatInput,
    z: FloatInput,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
    brighter_than: bool = True,
) -> FloatArray:
    r"""Return the cumulative number density for an evolving Schechter LF.

    For a magnitude threshold :math:`M_{\mathrm{lim}}`, define

    .. math::

        x_{\mathrm{lim}} = 10^{-0.4 (M_{\mathrm{lim}} - M_\star(z))}.

    Then the cumulative number density of galaxies brighter than the
    threshold is

    .. math::

        n(< M_{\mathrm{lim}}, z) = \phi_\star(z) \, \Gamma(\alpha(z) + 1, x_{\mathrm{lim}}),

    Then the cumulative number density of galaxies fainter than
    the threshold is

    .. math::

        n(M > M_{\mathrm{lim}}, z) = \phi_\star(z) \, \Gamma(\alpha(z) + 1) - n(M < M_{\mathrm{lim}}, z).

    Args:
        magnitude_limit: Magnitude threshold(s).
        z: Redshift value or array-like of redshift values.
        phi_model: Evolution model for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected
            ``phi_star`` evolution model.
        m_star_model: Evolution model for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected
            ``M_star`` evolution model.
        alpha_model: Evolution model for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected
            ``alpha`` evolution model.
        brighter_than: If True, return number density brighter than
            the threshold. If False, return number density fainter
            than the threshold.

    Returns:
        NumPy array of cumulative number densities evaluated at the
        supplied magnitude threshold(s) and redshift(s).

    Raises:
        ValueError: If an unsupported evolution model is requested.
    """
    m_lim = validate_array(magnitude_limit, name="m_lim")

    phi_star, m_star, alpha = evaluate_lf_parameters(
        z,
        phi_model=phi_model,
        phi_kwargs=phi_kwargs,
        m_star_model=m_star_model,
        m_star_kwargs=m_star_kwargs,
        alpha_model=alpha_model,
        alpha_kwargs=alpha_kwargs,
    )

    if np.any(phi_star < 0):
        raise ValueError("phi_star must be non-negative.")

    x_lim = luminosity_ratio(m_lim, m_star)
    x_lim = np.clip(x_lim, 1e-300, 1e300)

    s = np.asarray(alpha + 1.0, dtype=float)

    if np.any(s <= 0):
        raise ValueError(
            "Cumulative Schechter LF is undefined where alpha <= -1 "
            "because the integral diverges."
        )

    total_gamma = gamma(s)
    n_brighter = np.asarray(
        phi_star * total_gamma * gammaincc(s, x_lim),
        dtype=float,
    )

    if brighter_than:
        return n_brighter

    n_total = np.asarray(phi_star * total_gamma, dtype=float)
    return np.asarray(n_total - n_brighter, dtype=float)
