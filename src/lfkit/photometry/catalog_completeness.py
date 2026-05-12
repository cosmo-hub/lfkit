r"""Catalog completeness utilities for luminosity-function models.

This module provides helpers for estimating the observed and missing galaxy
population implied by a magnitude-limited catalog. These functions are useful
for applications that need an out-of-catalog correction, such as galaxy-catalog
priors for gravitational-wave cosmology.

The utilities convert an apparent magnitude limit into an absolute-magnitude
limit and call the generic LF integration helpers to return number densities
or fractions.

The core API accepts a luminosity-function callable with signature

    lf(absolute_mag, z)

where ``absolute_mag`` and ``z`` are NumPy arrays that can be broadcast
together. This keeps the completeness machinery independent of any specific
luminosity-function parameterization.
"""

from __future__ import annotations

import numpy as np

from lfkit.photometry.lf_integrals import (
    integrated_number_density as _integrated_number_density,
)
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
            distance-modulus convention. If not provided, this is read from
            ``cosmo_obj["h"]``.
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

    h_resolved = _resolve_h(cosmo_obj, h)

    return absolute_magnitude(
        cosmo_obj,
        z_arr,
        m_lim,
        h=h_resolved,
        k_correction=k_correction,
        e_correction=e_correction,
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

    return _integrated_number_density(
        z_arr,
        lf,
        m_bright=m_bright,
        m_faint=observed_upper,
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

    return _integrated_number_density(
        z_arr,
        lf,
        m_bright=missing_lower,
        m_faint=m_faint,
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


def _resolve_h(
    cosmo_obj: Cosmology,
    h: float | None,
) -> float:
    """Return explicit h or read it from a PyCCL-style cosmology object."""
    if h is not None:
        if not np.isfinite(h):
            raise ValueError("h must be finite.")
        return float(h)

    try:
        h_from_cosmo = cosmo_obj["h"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "h was not provided and could not be read from cosmo_obj['h']."
        ) from exc

    if not np.isfinite(h_from_cosmo):
        raise ValueError("cosmo_obj['h'] must be finite.")

    return float(h_from_cosmo)


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
