r"""Luminosity-function integration utilities.

This module provides generic numerical integrals of luminosity function
callables over finite absolute magnitude ranges.

The core API accepts a luminosity function callable with signature

    lf(absolute_mag, z)

where ``absolute_mag`` and ``z`` are NumPy arrays that can be broadcast
together. This keeps the integration machinery independent of any specific
luminosity function parameterization, catalog selection, or cosmology backend.

These helpers are intentionally generic. Catalog completeness, LF-dependent
redshift densities, luminosity-density calculations, and selection-weighted
integrals can all call this module instead of duplicating magnitude-grid logic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput, LuminosityFunction
from lfkit.utils.integrators import integrate_between_variable_bounds, safe_divide
from lfkit.utils.evaluators import (
    evaluate_optional_redshift_callable,
    evaluate_positive_redshift_callable,
    evaluate_lf_on_grid,
    evaluate_weight_on_grid,
)
from lfkit.utils.validators import validate_array


__all__ = [
    "integrated_number_density",
    "lf_weighted_integral",
    "selection_weighted_number_density",
    "integrated_luminosity_density",
    "mean_luminosity",
    "cumulative_number_density",
    "magnitude_window_number_density",
]


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
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of number densities evaluated at ``z``.
    """
    return _integrated_lf_between_bounds(
        z,
        lf,
        m_lower=m_bright,
        m_upper=m_faint,
        n_m=n_m,
    )


def lf_weighted_integral(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    weight_fn: Callable[[FloatArray, FloatArray], FloatArray],
    n_m: int = 512,
) -> FloatArray:
    r"""Return a weighted luminosity function integral.

    This computes

    .. math::

        I(z) = \int_{M_{\mathrm{bright}}(z)}^{M_{\mathrm{faint}}(z)}
               w(M, z)\,\phi(M, z)\,dM.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        weight_fn: Weight callable with signature ``weight_fn(M, z)``. Its
            return values must be broadcastable to the magnitude-redshift grid.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of weighted LF integrals evaluated at ``z``.
    """
    return _integrated_lf_between_bounds(
        z,
        lf,
        m_lower=m_bright,
        m_upper=m_faint,
        n_m=n_m,
        weight_fn=weight_fn,
    )


def selection_weighted_number_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    selection_fn: Callable[[FloatArray, FloatArray], FloatArray],
    n_m: int = 512,
) -> FloatArray:
    r"""Return number density weighted by a selection function.

    This computes

    .. math::

        n_{\mathrm{sel}}(z) =
        \int_{M_{\mathrm{bright}}(z)}^{M_{\mathrm{faint}}(z)}
        S(M, z)\,\phi(M, z)\,dM.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        selection_fn: Selection callable with signature ``selection_fn(M, z)``.
            Values should usually lie between 0 and 1, although this function
            only requires finite non-negative values.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of selection-weighted number densities evaluated at ``z``.
    """
    return lf_weighted_integral(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        weight_fn=selection_fn,
        n_m=n_m,
    )


def integrated_luminosity_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    m_reference: float = 0.0,
    n_m: int = 512,
) -> FloatArray:
    r"""Return luminosity density from a luminosity function.

    This computes

    .. math::

        \rho_L(z) =
        \int_{M_{\mathrm{bright}}(z)}^{M_{\mathrm{faint}}(z)}
        L(M)\,\phi(M, z)\,dM,

    using relative luminosities

    .. math::

        L(M) / L_{\mathrm{ref}} =
        10^{-0.4(M - M_{\mathrm{ref}})}.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        m_reference: Reference absolute magnitude defining the luminosity unit.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of luminosity densities in units of the reference
        luminosity.
    """
    if not np.isfinite(m_reference):
        raise ValueError("m_reference must be finite.")

    def luminosity_weight(
        absolute_mag: FloatArray,
        _redshift: FloatArray,
    ) -> FloatArray:
        """Return relative luminosity weights for absolute magnitudes."""
        return np.asarray(10.0 ** (-0.4 * (absolute_mag - m_reference)), dtype=float)

    return lf_weighted_integral(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        weight_fn=luminosity_weight,
        n_m=n_m,
    )


def mean_luminosity(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput,
    m_faint: FloatInput,
    m_reference: float = 0.0,
    n_m: int = 512,
) -> FloatArray:
    r"""Return mean luminosity over a finite magnitude range.

    This computes

    .. math::

        \langle L \rangle(z) =
        \frac{\int L(M)\,\phi(M, z)\,dM}
             {\int \phi(M, z)\,dM}.

    The luminosity is returned in units of the reference luminosity defined by
    ``m_reference``.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        m_reference: Reference absolute magnitude defining the luminosity unit.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of mean luminosities. Entries are zero where the integrated
        number density is zero.
    """
    luminosity_density = integrated_luminosity_density(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        m_reference=m_reference,
        n_m=n_m,
    )
    number_density = integrated_number_density(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )

    return safe_divide(luminosity_density, number_density)


def cumulative_number_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_threshold: FloatInput,
    m_bright: FloatInput,
    m_faint: FloatInput,
    brighter_than: bool = True,
    n_m: int = 512,
) -> FloatArray:
    r"""Return cumulative LF number density around a magnitude threshold.

    If ``brighter_than`` is True, this computes

    .. math::

        n(< M_{\mathrm{thr}}, z) =
        \int_{M_{\mathrm{bright}}}^{\min(M_{\mathrm{thr}}, M_{\mathrm{faint}})}
        \phi(M, z)\,dM.

    If ``brighter_than`` is False, this computes

    .. math::

        n(> M_{\mathrm{thr}}, z) =
        \int_{\max(M_{\mathrm{thr}}, M_{\mathrm{bright}})}^{M_{\mathrm{faint}}}
        \phi(M, z)\,dM.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_threshold: Absolute magnitude threshold. May be scalar or array-like.
        m_bright: Bright absolute magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute magnitude bound. May be scalar or array-like.
        brighter_than: If True, integrate galaxies brighter than the threshold.
            If False, integrate galaxies fainter than the threshold.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of cumulative number densities evaluated at ``z``.
    """
    z_arr = validate_array(z, name="z")
    m_threshold_arr = validate_array(m_threshold, name="m_threshold")
    m_bright_arr = validate_array(m_bright, name="m_bright")
    m_faint_arr = validate_array(m_faint, name="m_faint")

    z_b, threshold_b, bright_b, faint_b = np.broadcast_arrays(
        z_arr,
        m_threshold_arr,
        m_bright_arr,
        m_faint_arr,
    )

    if brighter_than:
        m_lower = bright_b
        m_upper = np.minimum(threshold_b, faint_b)
    else:
        m_lower = np.maximum(threshold_b, bright_b)
        m_upper = faint_b

    return _integrated_lf_between_bounds(
        z_b,
        lf,
        m_lower=m_lower,
        m_upper=m_upper,
        n_m=n_m,
    )


def magnitude_window_number_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_bright: FloatInput | None = None,
    m_faint: FloatInput | None = None,
    apparent_m_bright: FloatInput | None = None,
    apparent_m_faint: FloatInput | None = None,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray] | None = None,
    k_correction_fn: Callable[[FloatArray], FloatArray] | None = None,
    e_correction_fn: Callable[[FloatArray], FloatArray] | None = None,
    n_m: int = 512,
) -> FloatArray:
    r"""Return LF number density inside a magnitude-selection window.

    This integrates a luminosity function over a finite absolute magnitude
    range. The bright and faint limits may be supplied directly as absolute
    magnitudes, converted from apparent magnitudes, or supplied as a mixture of
    both.

    Magnitudes are ordered so that more negative values are brighter. Apparent
    magnitude limits are converted to absolute magnitude limits at each
    redshift before integration.

    This helper is science-use-case agnostic. It only defines the LF integral
    over a magnitude window; interpretation of that selected population belongs
    to the calling analysis.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_bright: Bright absolute magnitude bound.
        m_faint: Faint absolute magnitude bound.
        apparent_m_bright: Bright apparent magnitude bound.
        apparent_m_faint: Faint apparent magnitude bound.
        luminosity_distance_mpc_fn: Callable returning luminosity distance in
            Mpc. Required when either apparent magnitude bound is supplied.
        k_correction_fn: Optional K-correction callable evaluated at ``z``.
        e_correction_fn: Optional E-correction callable evaluated at ``z``.
        n_m: Number of magnitude-grid points used for the integral.

    Returns:
        NumPy array of number densities evaluated at ``z``.

    Raises:
        ValueError: If a bright or faint bound is missing, if both absolute and
            apparent values are supplied for the same bound, or if an apparent
            bound is supplied without a luminosity-distance callable.
    """
    z_arr = validate_array(z, name="z")

    m_bright_resolved = _resolve_magnitude_window_bound(
        z_arr,
        absolute_mag=m_bright,
        apparent_mag=apparent_m_bright,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction_fn=k_correction_fn,
        e_correction_fn=e_correction_fn,
        bound_name="bright",
    )
    m_faint_resolved = _resolve_magnitude_window_bound(
        z_arr,
        absolute_mag=m_faint,
        apparent_mag=apparent_m_faint,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction_fn=k_correction_fn,
        e_correction_fn=e_correction_fn,
        bound_name="faint",
    )

    return integrated_number_density(
        z_arr,
        lf,
        m_bright=m_bright_resolved,
        m_faint=m_faint_resolved,
        n_m=n_m,
    )


def _resolve_magnitude_window_bound(
    z: FloatArray,
    *,
    absolute_mag: FloatInput | None,
    apparent_mag: FloatInput | None,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray] | None,
    k_correction_fn: Callable[[FloatArray], FloatArray] | None,
    e_correction_fn: Callable[[FloatArray], FloatArray] | None,
    bound_name: str,
) -> FloatArray:
    r"""Return an absolute magnitude bound for a magnitude window."""
    if absolute_mag is None and apparent_mag is None:
        raise ValueError(
            f"Must provide either m_{bound_name} or apparent_m_{bound_name}."
        )

    if absolute_mag is not None and apparent_mag is not None:
        raise ValueError(
            f"Provide only one of m_{bound_name} or apparent_m_{bound_name}."
        )

    if absolute_mag is not None:
        return validate_array(absolute_mag, name=f"m_{bound_name}")

    if luminosity_distance_mpc_fn is None:
        raise ValueError(
            "luminosity_distance_mpc_fn is required when apparent magnitude "
            f"bounds are supplied. Missing conversion for apparent_m_{bound_name}."
        )

    apparent_mag_arr = validate_array(
        apparent_mag,
        name=f"apparent_m_{bound_name}",
    )
    luminosity_distance_mpc = evaluate_positive_redshift_callable(
        luminosity_distance_mpc_fn,
        z,
        name="luminosity_distance_mpc_fn",
    )

    k_correction = evaluate_optional_redshift_callable(
        k_correction_fn,
        z,
        name="k_correction_fn",
    )
    e_correction = evaluate_optional_redshift_callable(
        e_correction_fn,
        z,
        name="e_correction_fn",
    )

    if k_correction is None:
        k_correction = np.zeros_like(z, dtype=float)

    if e_correction is None:
        e_correction = np.zeros_like(z, dtype=float)

    distance_modulus = 5.0 * np.log10(luminosity_distance_mpc) + 25.0

    return np.asarray(
        apparent_mag_arr - distance_modulus - k_correction + e_correction,
        dtype=float,
    )


def _integrated_lf_between_bounds(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lower: FloatInput,
    m_upper: FloatInput,
    n_m: int,
    weight_fn: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
) -> FloatArray:
    r"""Return finite-range LF integral between magnitude bounds."""
    z_arr = validate_array(z, name="z")

    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0.")

    def integrand(
        absolute_mag: FloatArray,
        redshift: FloatArray,
    ) -> FloatArray:
        phi = evaluate_lf_on_grid(
            lf,
            m_grid=absolute_mag,
            z_grid=redshift,
        )

        if weight_fn is None:
            return phi

        weight = evaluate_weight_on_grid(
            weight_fn,
            m_grid=absolute_mag,
            z_grid=redshift,
        )

        return np.asarray(phi * weight, dtype=float)

    return integrate_between_variable_bounds(
        z_arr,
        lower=m_lower,
        upper=m_upper,
        integrand_fn=integrand,
        n_grid=n_m,
        y_name="z",
        lower_name="m_lower",
        upper_name="m_upper",
        n_grid_name="n_m",
    )
