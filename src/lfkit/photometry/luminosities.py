r"""Luminosity and magnitude conversion utilities for LFKit.

This module provides lightweight helpers for converting between
magnitude differences and luminosity ratios, as well as common
Schechter-function quantities expressed in luminosity space.

The core convention is

.. math::

    L_1 / L_2 = 10^{-0.4 (M_1 - M_2)},

where ``M1`` and ``M2`` may be absolute or apparent magnitudes,
as long as they are defined in the same photometric system.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma, gammaincc

from lfkit.utils.validators import validate_array

__all__ = (
    "luminosity_ratio",
    "luminosity_ratio_from_magnitudes",
    "magnitude_difference_from_luminosity_ratio",
    "luminosity_weight_from_magnitude",
    "luminosity_from_magnitude",
    "schechter_cumulative_number_density_luminosity",
    "schechter_luminosity_density",
    "schechter_mean_luminosity",
    "sample_schechter_luminosity",
    "schechter_selection_function",
)


def luminosity_ratio(
    absolute_mag: ArrayLike,
    m_star: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the luminosity ratio relative to the characteristic luminosity.

    For magnitudes,

    .. math::

        \frac{L}{L_\star} = 10^{-0.4 (M - M_\star)}.

    Args:
        absolute_mag: Absolute magnitude value(s).
        m_star: Characteristic magnitude value(s).

    Returns:
        NumPy array of luminosity ratios ``L / L_star``.
    """
    m_arr = validate_array(absolute_mag, name="absolute magnitude")
    m_star_arr = validate_array(m_star, name="m_star")
    x = 10.0 ** (-0.4 * (m_arr - m_star_arr))

    # Clip extreme values to avoid overflow in exp later.
    x = np.clip(x, 1e-300, 1e300)

    return np.asarray(x, dtype=float)


def luminosity_ratio_from_magnitudes(
    magnitude: ArrayLike,
    ref_magnitude: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return luminosity relative to a reference magnitude.

    This uses :math:`L / L_{\mathrm{ref}} = 10^{-0.4 (m - m_{\mathrm{ref}})}`.

    Args:
        magnitude: Magnitude value(s).
        ref_magnitude: Reference magnitude value(s).

    Returns:
        NumPy array of luminosity ratios :math:`L / L_{\mathrm{ref}}`.
    """
    mag_1 = validate_array(magnitude, name="magnitude")
    mag_2 = validate_array(ref_magnitude, name="ref_magnitude")

    ratio = 10.0 ** (-0.4 * (mag_1 - mag_2))
    ratio = np.clip(ratio, 1e-300, 1e300)

    return np.asarray(ratio, dtype=float)


def magnitude_difference_from_luminosity_ratio(
    luminosity_ratio: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return the magnitude difference corresponding to a luminosity ratio.

    This uses

    .. math::

        m_1 - m_2 = -2.5 \log_{10}(L_1 / L_2).

    Args:
        luminosity_ratio: Luminosity ratio value(s) ``L1 / L2``.

    Returns:
        NumPy array of magnitude differences ``m1 - m2``.

    Raises:
        ValueError: If any luminosity ratio is not strictly positive.
    """
    ratio = validate_array(luminosity_ratio, name="luminosity_ratio")

    if np.any(ratio <= 0):
        raise ValueError("luminosity_ratio must be strictly positive.")

    return np.asarray(-2.5 * np.log10(ratio), dtype=float)


def luminosity_weight_from_magnitude(
    magnitude: ArrayLike,
    reference_magnitude: float = 0.0,
) -> NDArray[np.float64]:
    r"""Return an unnormalized luminosity weight from a magnitude.

    This uses

    .. math::

        L \propto 10^{-0.4 M}.

    Args:
        magnitude: Magnitude value(s).
        reference_magnitude: Reference zero-point magnitude for the
            proportionality constant.

    Returns:
        NumPy array proportional to luminosity.
    """
    mag = validate_array(magnitude, name="magnitude")
    weight = 10.0 ** (-0.4 * (mag - reference_magnitude))
    weight = np.clip(weight, 1e-300, 1e300)
    return np.asarray(weight, dtype=float)


def luminosity_from_magnitude(
    magnitude: ArrayLike,
    *,
    reference_magnitude: float = 0.0,
    reference_luminosity: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return luminosity corresponding to a magnitude relative to a reference.

    This uses

    .. math::

        L = L_{\mathrm{ref}} \, 10^{-0.4 (m - m_{\mathrm{ref}})}.

    Args:
        magnitude: Magnitude value(s).
        reference_magnitude: Reference magnitude ``m_ref``.
        reference_luminosity: Reference luminosity ``L_ref``.

    Returns:
        NumPy array of luminosities in the same units as
        ``reference_luminosity``.

    Raises:
        ValueError: If ``reference_luminosity`` is not strictly positive.
    """
    if reference_luminosity <= 0:
        raise ValueError("reference_luminosity must be strictly positive.")

    ratio = luminosity_ratio_from_magnitudes(
        magnitude,
        reference_magnitude,
    )
    return np.asarray(reference_luminosity * ratio, dtype=float)


def schechter_cumulative_number_density_luminosity(
    luminosity_min: ArrayLike,
    *,
    phi_star: float,
    l_star: float,
    alpha: float,
) -> NDArray[np.float64]:
    r"""Return cumulative Schechter number density above a luminosity threshold.

    This evaluates

    .. math::

        n(L > L_{\min}) = \phi_* \, \Gamma(\alpha + 1, L_{\min} / L_*),

    where ``Gamma`` is the upper incomplete gamma function.

    Args:
        luminosity_min: Lower luminosity threshold(s).
        phi_star: Schechter normalization.
        l_star: Characteristic luminosity ``L_star``.
        alpha: Faint-end slope.

    Returns:
        NumPy array of cumulative number densities above ``luminosity_min``.

    Raises:
        ValueError: If ``phi_star`` is negative, ``l_star`` is not strictly
            positive, or ``alpha <= -1``.
    """
    l_min = validate_array(luminosity_min, name="luminosity_min")

    if np.any(l_min < 0):
        raise ValueError("luminosity_min must be non-negative.")

    if phi_star < 0:
        raise ValueError("phi_star must be non-negative.")

    if l_star <= 0:
        raise ValueError("l_star must be strictly positive.")

    s = alpha + 1.0
    if s <= 0:
        raise ValueError(
            "Cumulative number density is undefined for alpha <= -1 "
            "because the integral diverges."
        )

    x_min = np.clip(l_min / l_star, 0.0, 1e300)
    n_gt = phi_star * gamma(s) * gammaincc(s, x_min)
    return np.asarray(n_gt, dtype=float)


def schechter_luminosity_density(
    *,
    phi_star: float,
    l_star: float,
    alpha: float,
) -> float:
    r"""Return total luminosity density for a Schechter luminosity function.

    This evaluates

    .. math::

        \rho_L = \phi_* \, L_* \, \Gamma(\alpha + 2).

    Args:
        phi_star: Schechter normalization.
        l_star: Characteristic luminosity ``L_star``.
        alpha: Faint-end slope.

    Returns:
        Total luminosity density.

    Raises:
        ValueError: If ``phi_star`` is negative, ``l_star`` is not strictly
            positive, or ``alpha <= -2``.
    """
    if phi_star < 0:
        raise ValueError("phi_star must be non-negative.")

    if l_star <= 0:
        raise ValueError("l_star must be strictly positive.")

    s = alpha + 2.0
    if s <= 0:
        raise ValueError(
            "Luminosity density is undefined for alpha <= -2 "
            "because the integral diverges."
        )

    return float(phi_star * l_star * gamma(s))


def schechter_mean_luminosity(
    *,
    l_star: float,
    alpha: float,
) -> float:
    r"""Return the mean luminosity of a normalized Schechter distribution.

    This evaluates

    .. math::

        \langle L \rangle = L_* \, \Gamma(\alpha + 2) / \Gamma(\alpha + 1).

    For finite values this simplifies to

    .. math::

        \langle L \rangle = L_* (\alpha + 1),

    provided ``alpha > -1``.

    Args:
        l_star: Characteristic luminosity ``L_star``.
        alpha: Faint-end slope.

    Returns:
        Mean luminosity.

    Raises:
        ValueError: If ``l_star`` is not strictly positive or ``alpha <= -1``.
    """
    if l_star <= 0:
        raise ValueError("l_star must be strictly positive.")

    s = alpha + 1.0
    if s <= 0:
        raise ValueError(
            "Mean luminosity is undefined for alpha <= -1 "
            "because the number-density integral diverges."
        )

    return float(l_star * gamma(alpha + 2.0) / gamma(alpha + 1.0))


def sample_schechter_luminosity(
    size: int | tuple[int, ...],
    *,
    l_star: float,
    alpha: float,
    rng: Generator | None = None,
) -> NDArray[np.float64]:
    r"""Sample luminosities from a normalized Schechter distribution.

    This samples from

    .. math::

        p(L) \propto (L / L_*)^{\alpha} \exp(-L / L_*),

    which is equivalent to a Gamma distribution in

    .. math::

        x = L / L_*,

    with shape parameter

    .. math::

        k = \alpha + 1.

    Args:
        size: Number or shape of samples to draw.
        l_star: Characteristic luminosity ``L_star``.
        alpha: Faint-end slope.
        rng: Optional NumPy random number generator.

    Returns:
        NumPy array of sampled luminosities.

    Raises:
        ValueError: If ``l_star`` is not strictly positive or ``alpha <= -1``.
    """
    if l_star <= 0:
        raise ValueError("l_star must be strictly positive.")

    shape = alpha + 1.0
    if shape <= 0:
        raise ValueError(
            "Sampling from the normalized Schechter distribution requires alpha > -1."
        )

    generator = default_rng() if rng is None else rng
    samples = generator.gamma(shape=shape, scale=l_star, size=size)
    return np.asarray(samples, dtype=float)


def schechter_selection_function(
    luminosity_min: ArrayLike,
    *,
    l_star: float,
    alpha: float,
) -> NDArray[np.float64]:
    r"""Return the Schechter selection fraction above a luminosity threshold.

    This evaluates

    .. math::

        S(L_{\min}) = n(L > L_{\min}) / n_{\mathrm{tot}}
                     = \Gamma(\alpha + 1, L_{\min} / L_*) / \Gamma(\alpha + 1),

    which is equivalently

    .. math::

        S(L_{\min}) = \mathrm{gammaincc}(\alpha + 1, L_{\min} / L_*).

    Args:
        luminosity_min: Lower luminosity threshold(s).
        l_star: Characteristic luminosity ``L_star``.
        alpha: Faint-end slope.

    Returns:
        NumPy array of selection fractions between 0 and 1.

    Raises:
        ValueError: If ``l_star`` is not strictly positive, any luminosity
            threshold is negative, or ``alpha <= -1``.
    """
    l_min = validate_array(luminosity_min, name="luminosity_min")

    if np.any(l_min < 0):
        raise ValueError("luminosity_min must be non-negative.")

    if l_star <= 0:
        raise ValueError("l_star must be strictly positive.")

    s = alpha + 1.0
    if s <= 0:
        raise ValueError(
            "Selection function is undefined for alpha <= -1 "
            "because the total number-density integral diverges."
        )

    x_min = np.clip(l_min / l_star, 0.0, 1e300)
    return np.asarray(gammaincc(s, x_min), dtype=float)
