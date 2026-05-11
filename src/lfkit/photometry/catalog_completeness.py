r"""Catalog completeness utilities for luminosity-function models.

This module provides helpers for estimating the observed and missing galaxy
population implied by a magnitude-limited catalog. These functions are useful
for applications that need an out-of-catalog correction, such as galaxy-catalog
priors for gravitational-wave cosmology.

The utilities convert an apparent magnitude limit into an absolute-magnitude
limit, integrate a luminosity function over finite absolute-magnitude ranges,
and return number densities or fractions.

The core API accepts a luminosity-function callable with signature

    lf(absolute_mag, z)

where ``absolute_mag`` and ``z`` are NumPy arrays that can be broadcast
together. This keeps the completeness machinery independent of any specific
luminosity-function parameterization.
"""

from __future__ import annotations

import numpy as np

from lfkit.photometry.magnitudes import absolute_magnitude
from lfkit.utils.types import (
    Cosmology,
    FloatArray,
    FloatInput,
    LuminosityFunction,
)
from lfkit.utils.validators import validate_array, validate_magnitude_range


__all__ = [
    "absolute_magnitude_limit",
    "integrated_number_density",
    "observed_number_density",
    "missing_number_density",
    "catalog_completeness_fraction",
    "out_of_catalog_fraction",
]


def absolute_magnitude_limit(
    cosmo_obj: Cosmology,
    z: FloatInput,
    *,
    m_lim: float,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return the absolute-magnitude limit of an apparent-magnitude catalog cut.

    This converts an apparent magnitude limit into the corresponding limiting
    absolute magnitude at each redshift,

    .. math::

        M_{\mathrm{lim}}(z) = m_{\mathrm{lim}} - \mu(z) - K(z) + E(z).

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        m_lim: Apparent magnitude limit of the catalog.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of limiting absolute magnitudes.
    """
    z_arr = validate_array(z, name="z")

    if np.any(z_arr < 0):
        raise ValueError("Redshift z must be >= 0.")

    if not np.isfinite(m_lim):
        raise ValueError("m_lim must be finite.")

    return absolute_magnitude(
        cosmo_obj,
        z_arr,
        m_lim,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )


def integrated_number_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return finite-range number density from a luminosity function.

    This computes

    .. math::

        n(z) = \int_{M_{\mathrm{bright}}(z)}^{M_{\mathrm{faint}}(z)}
               \phi(M, z) \, dM.

    Magnitudes are ordered so that more negative values are brighter.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of number densities evaluated at ``z``.
    """
    return _integrated_number_density_between_bounds(
        z,
        lf,
        m_lower=m_bright,
        m_upper=m_faint,
        n_m=n_m,
    )


def observed_number_density(
    cosmo_obj: Cosmology,
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    m_faint: float,
    n_m: int = 512,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return number density observable in a magnitude-limited catalog.

    This integrates the luminosity function over galaxies brighter than the
    catalog limit,

    .. math::

        n_{\mathrm{obs}}(z) =
        \int_{M_{\mathrm{bright}}}^{\min[M_{\lim}(z), M_{\mathrm{faint}}]}
        \phi(M, z) \, dM.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude bound of the LF model.
        m_faint: Faint absolute-magnitude bound of the LF model.
        n_m: Number of magnitude-grid points used for the integral.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of observed number densities.
    """
    validate_magnitude_range(m_bright=m_bright, m_faint=m_faint)

    z_arr = validate_array(z, name="z")
    m_abs_lim = absolute_magnitude_limit(
        cosmo_obj,
        z_arr,
        m_lim=m_lim,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    observed_upper = np.minimum(m_abs_lim, m_faint)

    return _integrated_number_density_between_bounds(
        z_arr,
        lf,
        m_lower=m_bright,
        m_upper=observed_upper,
        n_m=n_m,
    )


def missing_number_density(
    cosmo_obj: Cosmology,
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    m_faint: float,
    n_m: int = 512,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return number density missing from a magnitude-limited catalog.

    This integrates the luminosity function over galaxies fainter than the
    catalog limit,

    .. math::

        n_{\mathrm{miss}}(z) =
        \int_{\max[M_{\lim}(z), M_{\mathrm{bright}}]}^{M_{\mathrm{faint}}}
        \phi(M, z) \, dM.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude bound of the LF model.
        m_faint: Faint absolute-magnitude bound of the LF model.
        n_m: Number of magnitude-grid points used for the integral.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of missing number densities.
    """
    validate_magnitude_range(m_bright=m_bright, m_faint=m_faint)

    z_arr = validate_array(z, name="z")
    m_abs_lim = absolute_magnitude_limit(
        cosmo_obj,
        z_arr,
        m_lim=m_lim,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    missing_lower = np.maximum(m_abs_lim, m_bright)

    return _integrated_number_density_between_bounds(
        z_arr,
        lf,
        m_lower=missing_lower,
        m_upper=m_faint,
        n_m=n_m,
    )


def catalog_completeness_fraction(
    cosmo_obj: Cosmology,
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    m_faint: float,
    n_m: int = 512,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return the LF fraction observable in a magnitude-limited catalog.

    This returns

    .. math::

        f_{\mathrm{obs}}(z) =
        \frac{n_{\mathrm{obs}}(z)}
             {n_{\mathrm{obs}}(z) + n_{\mathrm{miss}}(z)}.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude bound of the LF model.
        m_faint: Faint absolute-magnitude bound of the LF model.
        n_m: Number of magnitude-grid points used for the integral.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of catalog completeness fractions.
    """
    observed = observed_number_density(
        cosmo_obj,
        z,
        lf,
        m_lim=m_lim,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    missing = missing_number_density(
        cosmo_obj,
        z,
        lf,
        m_lim=m_lim,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return _fraction(observed, observed + missing)


def out_of_catalog_fraction(
    cosmo_obj: Cosmology,
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    m_faint: float,
    n_m: int = 512,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return the LF fraction missing from a magnitude-limited catalog.

    This returns

    .. math::

        f_{\mathrm{miss}}(z) = 1 - f_{\mathrm{obs}}(z).

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude bound of the LF model.
        m_faint: Faint absolute-magnitude bound of the LF model.
        n_m: Number of magnitude-grid points used for the integral.
        h: Optional dimensionless Hubble parameter used in the
            distance-modulus convention.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of out-of-catalog fractions.
    """
    completeness = catalog_completeness_fraction(
        cosmo_obj,
        z,
        lf,
        m_lim=m_lim,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
        h=h,
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return np.asarray(1.0 - completeness, dtype=float)


def _integrated_number_density_between_bounds(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lower: FloatInput,
    m_upper: FloatInput,
    n_m: int,
) -> FloatArray:
    r"""Return finite-range number density between magnitude bounds."""
    z_arr = validate_array(z, name="z")
    m_lower_arr = validate_array(m_lower, name="m_lower")
    m_upper_arr = validate_array(m_upper, name="m_upper")

    _validate_integration_inputs(
        z=z_arr,
        m_lower=m_lower_arr,
        m_upper=m_upper_arr,
        n_m=n_m,
    )

    z_b, m_lower_b, m_upper_b = np.broadcast_arrays(
        z_arr,
        m_lower_arr,
        m_upper_arr,
    )

    density = np.zeros_like(z_b, dtype=float)
    valid = m_upper_b > m_lower_b

    if not np.any(valid):
        return density

    m_grid = _magnitude_grid(
        m_lower=m_lower_b[valid],
        m_upper=m_upper_b[valid],
        n_m=n_m,
    )
    z_grid = np.broadcast_to(z_b[valid][None, :], m_grid.shape)
    phi = _evaluate_lf_on_grid(lf, m_grid=m_grid, z_grid=z_grid)

    density[valid] = np.trapezoid(phi, x=m_grid, axis=0)

    return np.asarray(density, dtype=float)


def _magnitude_grid(
    *,
    m_lower: FloatArray,
    m_upper: FloatArray,
    n_m: int,
) -> FloatArray:
    r"""Return a magnitude grid for column-wise finite-range integration."""
    t = np.linspace(0.0, 1.0, int(n_m), dtype=float)
    return np.asarray(
        m_lower[None, :] + t[:, None] * (m_upper[None, :] - m_lower[None, :]),
        dtype=float,
    )


def _evaluate_lf_on_grid(
    lf: LuminosityFunction,
    *,
    m_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    r"""Return LF values evaluated on a magnitude-redshift grid."""
    phi = np.asarray(lf(m_grid, z_grid), dtype=float)

    if phi.shape != m_grid.shape:
        try:
            phi = np.broadcast_to(phi, m_grid.shape)
        except ValueError as exc:
            raise ValueError(
                "lf(M, z) must return values broadcastable to the shape "
                "of the magnitude-redshift integration grid."
            ) from exc

    if np.any(~np.isfinite(phi)):
        raise ValueError("lf(M, z) returned non-finite values.")

    if np.any(phi < 0):
        raise ValueError("lf(M, z) must be non-negative.")

    return np.asarray(phi, dtype=float)


def _validate_integration_inputs(
    *,
    z: FloatArray,
    m_lower: FloatArray,
    m_upper: FloatArray,
    n_m: int,
) -> None:
    r"""Validate inputs for finite-range LF integration."""
    if np.any(z < 0):
        raise ValueError("Redshift z must be >= 0.")

    if np.any(~np.isfinite(m_lower)):
        raise ValueError("m_lower must contain only finite values.")

    if np.any(~np.isfinite(m_upper)):
        raise ValueError("m_upper must contain only finite values.")

    if n_m < 2:
        raise ValueError("n_m must be at least 2.")


def _fraction(
    numerator: FloatArray,
    denominator: FloatArray,
) -> FloatArray:
    r"""Return a clipped fraction with safe zero-denominator handling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0.0,
        )

    return np.asarray(np.clip(result, 0.0, 1.0), dtype=float)
