r"""Luminosity function population fractions.

This module provides helpers for computing population fractions from
luminosity functions. These functions are thin wrappers around the generic
luminosity function integration utilities.
"""

from __future__ import annotations

import numpy as np

from lfkit.luminosity_functions.integrals import integrated_number_density
from lfkit.utils.integrators import safe_divide
from lfkit.utils.types import FloatArray, FloatInput, LuminosityFunction
from lfkit.utils.validators import validate_magnitude_range

__all__ = [
    "fraction_from_luminosity_functions",
    "complement_fraction_from_luminosity_functions",
    "red_fraction_from_luminosity_functions",
    "blue_fraction_from_luminosity_functions",
    "red_blue_fractions_from_luminosity_functions",
    "population_densities_from_luminosity_functions",
]

__api_aliases__ = {
    "fraction_from_luminosity_functions": "fraction",
    "complement_fraction_from_luminosity_functions": "complement_fraction",
    "red_fraction_from_luminosity_functions": "red_fraction",
    "blue_fraction_from_luminosity_functions": "blue_fraction",
    "red_blue_fractions_from_luminosity_functions": "red_blue",
    "population_densities_from_luminosity_functions": "population_densities",
}


def fraction_from_luminosity_functions(
    z: FloatInput,
    numerator_lf: LuminosityFunction,
    denominator_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return the LF number density fraction between two luminosity functions.

    This computes

    .. math::

        f(z) =
        \frac{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{num}}(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{den}}(M,z)\,dM
        }.

    Args:
        z: Redshift value or array-like of redshift values.
        numerator_lf: Numerator luminosity function callable with signature
            ``lf(M, z)``.
        denominator_lf: Denominator luminosity function callable with signature
            ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array of LF number density fractions. Entries are zero where the
        denominator number density is zero.

    Raises:
        ValueError: If redshift values are negative, if magnitude bounds are
            invalid, or if either luminosity function returns invalid values.
    """
    validate_magnitude_range(m_bright=m_bright, m_faint=m_faint)

    numerator_density = integrated_number_density(
        z,
        numerator_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )
    denominator_density = integrated_number_density(
        z,
        denominator_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )

    return np.asarray(
        safe_divide(numerator_density, denominator_density),
        dtype=float,
    )


def complement_fraction_from_luminosity_functions(
    z: FloatInput,
    numerator_lf: LuminosityFunction,
    denominator_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return one minus the LF number density fraction.

    This computes

    .. math::

        f_{\mathrm{comp}}(z) = 1 - f(z),

    where

    .. math::

        f(z) =
        \frac{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{num}}(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{den}}(M,z)\,dM
        }.

    Args:
        z: Redshift value or array-like of redshift values.
        numerator_lf: Numerator luminosity function callable with signature
            ``lf(M, z)``.
        denominator_lf: Denominator luminosity function callable with signature
            ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array containing ``1 - numerator / denominator``.
    """
    fraction = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )

    return np.asarray(1.0 - fraction, dtype=float)


def red_fraction_from_luminosity_functions(
    z: FloatInput,
    red_lf: LuminosityFunction,
    total_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return the red fraction from red and total luminosity functions.

    This computes

    .. math::

        f_{\mathrm{red}}(z) =
        \frac{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{red}}(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{tot}}(M,z)\,dM
        }.

    Args:
        z: Redshift value or array-like of redshift values.
        red_lf: Red galaxy luminosity function callable with signature
            ``lf(M, z)``.
        total_lf: Total luminosity function callable with signature
            ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array of red fractions. Entries are zero where the total number
        density is zero.
    """
    return fraction_from_luminosity_functions(
        z,
        red_lf,
        total_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )


def blue_fraction_from_luminosity_functions(
    z: FloatInput,
    blue_lf: LuminosityFunction,
    total_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return the blue fraction from blue and total luminosity functions.

    This computes

    .. math::

        f_{\mathrm{blue}}(z) =
        \frac{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{blue}}(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi_{\mathrm{tot}}(M,z)\,dM
        }.

    Args:
        z: Redshift value or array-like of redshift values.
        blue_lf: Blue galaxy luminosity function callable with signature
            ``lf(M, z)``.
        total_lf: Total luminosity function callable with signature
            ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array of blue fractions. Entries are zero where the total number
        density is zero.
    """
    return fraction_from_luminosity_functions(
        z,
        blue_lf,
        total_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )


def red_blue_fractions_from_luminosity_functions(
    z: FloatInput,
    red_lf: LuminosityFunction,
    blue_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> tuple[FloatArray, FloatArray]:
    r"""Return red and blue fractions from red and blue luminosity functions.

    This computes

    .. math::

        f_{\mathrm{red}}(z) =
        \frac{n_{\mathrm{red}}(z)}
             {n_{\mathrm{red}}(z) + n_{\mathrm{blue}}(z)},

    and

    .. math::

        f_{\mathrm{blue}}(z) =
        \frac{n_{\mathrm{blue}}(z)}
             {n_{\mathrm{red}}(z) + n_{\mathrm{blue}}(z)},

    where

    .. math::

        n_{\mathrm{red}}(z) =
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
        \phi_{\mathrm{red}}(M,z)\,dM,

    and similarly for the blue luminosity function.

    Args:
        z: Redshift value or array-like of redshift values.
        red_lf: Red galaxy luminosity function callable with signature
            ``lf(M, z)``.
        blue_lf: Blue galaxy luminosity function callable with signature
            ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        Tuple containing ``(red_fraction, blue_fraction)``. Entries are zero
        where the combined red plus blue number density is zero.

    Raises:
        ValueError: If redshift values are negative, if magnitude bounds are
            invalid, or if either luminosity function returns invalid values.
    """
    red_density, blue_density, total_density = (
        population_densities_from_luminosity_functions(
            z,
            red_lf,
            blue_lf,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
        )
    )

    red_fraction = safe_divide(red_density, total_density)
    blue_fraction = safe_divide(blue_density, total_density)

    return (
        np.asarray(red_fraction, dtype=float),
        np.asarray(blue_fraction, dtype=float),
    )


def population_densities_from_luminosity_functions(
    z: FloatInput,
    first_lf: LuminosityFunction,
    second_lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Return two population densities and their sum.

    This computes

    .. math::

        n_1(z) =
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
        \phi_1(M,z)\,dM,

    .. math::

        n_2(z) =
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
        \phi_2(M,z)\,dM,

    and

    .. math::

        n_{\mathrm{tot}}(z) = n_1(z) + n_2(z).

    Args:
        z: Redshift value or array-like of redshift values.
        first_lf: First population luminosity function callable with signature
            ``lf(M, z)``.
        second_lf: Second population luminosity function callable with
            signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        Tuple containing ``(first_density, second_density, total_density)``.
    """
    validate_magnitude_range(m_bright=m_bright, m_faint=m_faint)

    first_density = integrated_number_density(
        z,
        first_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )
    second_density = integrated_number_density(
        z,
        second_lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )
    total_density = first_density + second_density

    return (
        np.asarray(first_density, dtype=float),
        np.asarray(second_density, dtype=float),
        np.asarray(total_density, dtype=float),
    )
