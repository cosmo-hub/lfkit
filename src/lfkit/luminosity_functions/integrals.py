"""Luminosity-function integration utilities.

This module provides generic numerical integrals of luminosity function
callables over finite absolute magnitude ranges.

The core API accepts a luminosity function callable with signature

    lf(absolute_mag, z)

where ``absolute_mag`` and ``z`` are NumPy arrays that can be broadcast
together. This keeps the integration machinery independent of any specific
luminosity function parameterization, catalog selection, or cosmology backend.

The helper ``_bind_lf`` converts model functions with fixed parameters into
this common callable form. Static luminosity functions that do not depend on
redshift can be wrapped with ``_bind_static_lf``.

These helpers are intentionally generic. Catalog completeness, LF-dependent
redshift densities, luminosity-density calculations, selection fractions, and
selection-weighted integrals can all call this module instead of duplicating
magnitude-grid logic.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

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
    "selection_fraction",
    "cumulative_selection_function",
    "luminosity_weight",
]

__api_aliases__ = {
    "integrated_number_density": "number_density",
    "lf_weighted_integral": "weighted",
    "selection_weighted_number_density": "selection_weighted_number_density",
    "integrated_luminosity_density": "luminosity_density",
    "mean_luminosity": "mean_luminosity",
    "cumulative_number_density": "cumulative_number_density",
    "magnitude_window_number_density": "magnitude_window_number_density",
    "selection_fraction": "selection_fraction",
    "cumulative_selection_function": "selection_function",
    "luminosity_weight": "luminosity_weight",
}


def _bind_lf(
    model_fn: Callable[..., FloatArray],
    /,
    **params: Any,
) -> LuminosityFunction:
    """Bind parameters to a redshift-dependent LF model.

    This converts a model function with signature approximately

    ``model_fn(absolute_mag, z, **params)``

    into the common integration signature

    ``lf(absolute_mag, z)``.

    Args:
        model_fn: Luminosity-function model callable.
        **params: Parameters passed to ``model_fn`` every time it is evaluated.

    Returns:
        Callable with signature ``lf(absolute_mag, z)``.

    Raises:
        TypeError: If ``model_fn`` cannot be called with the supplied
            parameters.

    Examples:
        >>> lf = _bind_lf(
        ...     evolving_schechter,
        ...     phi_model="linear_p",
        ...     phi_kwargs={"phi0": 1e-3, "p": 1.0},
        ...     m_star_model="linear_q",
        ...     m_star_kwargs={"m0": -21.0, "q": 1.0},
        ...     alpha_model="constant",
        ...     alpha_kwargs={"alpha0": -0.9},
        ... )
    """

    @wraps(model_fn)
    def lf(absolute_mag: FloatArray, z: FloatArray) -> FloatArray:
        return np.asarray(model_fn(absolute_mag, z, **params), dtype=float)

    return lf


def _bind_static_lf(
    model_fn: Callable[..., FloatArray],
    /,
    **params: Any,
) -> LuminosityFunction:
    """Bind parameters to a redshift-independent LF model.

    This converts a static model function with signature approximately

    ``model_fn(absolute_mag, **params)``

    into the common integration signature

    ``lf(absolute_mag, z)``.

    The redshift argument is accepted but ignored. This lets static and
    evolving luminosity functions use the same integration API.

    Args:
        model_fn: Redshift-independent luminosity function model callable.
        **params: Parameters passed to ``model_fn`` every time it is evaluated.

    Returns:
        Callable with signature ``lf(absolute_mag, z)``.

    Raises:
        TypeError: If ``model_fn`` cannot be called with the supplied
            parameters.

    Examples:
        >>> lf = _bind_static_lf(
        ...     schechter,
        ...     phi_star=1e-3,
        ...     m_star=-21.0,
        ...     alpha=-0.9,
        ... )
    """

    @wraps(model_fn)
    def lf(absolute_mag: FloatArray, _z: FloatArray) -> FloatArray:
        return np.asarray(model_fn(absolute_mag, **params), dtype=float)

    return lf


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

    Raises:
        ValueError: If redshift values are negative or if the magnitude
            bounds are invalid.
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

    Raises:
        ValueError: If redshift values are negative, if the magnitude
            bounds are invalid, or if ``weight_fn`` returns invalid values.
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

    Raises:
        ValueError: If redshift values are negative, if the magnitude
            bounds are invalid, or if ``selection_fn`` returns invalid values.
    """
    return lf_weighted_integral(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        weight_fn=selection_fn,
        n_m=n_m,
    )


def luminosity_weight(
    absolute_mag: FloatInput,
    _z: FloatInput | None = None,
    *,
    m_reference: float = 0.0,
) -> FloatArray:
    r"""Return relative luminosity weights for absolute magnitudes.

    This evaluates

    .. math::

        L(M) / L_{\mathrm{ref}} =
        10^{-0.4(M - M_{\mathrm{ref}})}.

    Args:
        absolute_mag: Absolute magnitude value(s).
        _z: Optional redshift argument, accepted for compatibility with
            weight callables of signature ``weight_fn(M, z)``.
        m_reference: Reference absolute magnitude defining the luminosity unit.

    Returns:
        NumPy array of relative luminosity weights.

    Raises:
        ValueError: If ``m_reference`` is not finite.
    """
    if not np.isfinite(m_reference):
        raise ValueError("m_reference must be finite.")

    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")

    exponent = -0.4 * (absolute_mag_arr - m_reference)
    exponent = np.clip(exponent, -300.0, 300.0)

    weight = 10.0**exponent
    return np.asarray(weight, dtype=float)


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

    Raises:
        ValueError: If ``m_reference`` is not finite, if redshift values are
            negative, or if the magnitude bounds are invalid.
    """

    def weight_fn(
        absolute_mag: FloatArray,
        redshift: FloatArray,
    ) -> FloatArray:
        return luminosity_weight(
            absolute_mag,
            redshift,
            m_reference=m_reference,
        )

    return lf_weighted_integral(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        weight_fn=weight_fn,
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

    Raises:
        ValueError: If ``m_reference`` is not finite, if redshift values are
            negative, or if the magnitude bounds are invalid.
    """
    luminosity_density_arr = integrated_luminosity_density(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        m_reference=m_reference,
        n_m=n_m,
    )
    number_density_arr = integrated_number_density(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )

    return safe_divide(luminosity_density_arr, number_density_arr)


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

    Raises:
        ValueError: If redshift values are negative or if the magnitude
            bounds are invalid.
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


def selection_fraction(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_selected_bright: FloatInput,
    m_selected_faint: FloatInput,
    m_total_bright: FloatInput,
    m_total_faint: FloatInput,
    n_m: int = 512,
) -> FloatArray:
    r"""Return the fraction of LF number density inside a selected window.

    This computes

    .. math::

        f_{\mathrm{sel}}(z) =
        \frac{
            \int_{M_{\mathrm{sel,bright}}}^{M_{\mathrm{sel,faint}}}
            \phi(M,z)\,dM
        }{
            \int_{M_{\mathrm{tot,bright}}}^{M_{\mathrm{tot,faint}}}
            \phi(M,z)\,dM
        }.

    This is the generic numerical analogue of model-specific analytic
    selection functions. It works for any luminosity function callable with
    signature ``lf(M, z)``.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_selected_bright: Bright bound of the selected magnitude window.
        m_selected_faint: Faint bound of the selected magnitude window.
        m_total_bright: Bright bound of the reference total window.
        m_total_faint: Faint bound of the reference total window.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array of selected fractions. Entries are zero where the total
        number density is zero.

    Raises:
        ValueError: If redshift values are negative or if any magnitude
            window is invalid.
    """
    selected = integrated_number_density(
        z,
        lf,
        m_bright=m_selected_bright,
        m_faint=m_selected_faint,
        n_m=n_m,
    )
    total = integrated_number_density(
        z,
        lf,
        m_bright=m_total_bright,
        m_faint=m_total_faint,
        n_m=n_m,
    )

    return safe_divide(selected, total)


def cumulative_selection_function(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_threshold: FloatInput,
    m_bright: FloatInput,
    m_faint: FloatInput,
    brighter_than: bool = True,
    n_m: int = 512,
) -> FloatArray:
    r"""Return the cumulative LF selection fraction around a threshold.

    This computes the cumulative number density brighter or fainter than a
    threshold divided by the total number density in the supplied reference
    magnitude range.

    If ``brighter_than`` is True,

    .. math::

        S(z) =
        \frac{
            \int_{M_{\mathrm{bright}}}^{\min(M_{\mathrm{thr}},M_{\mathrm{faint}})}
            \phi(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi(M,z)\,dM
        }.

    If ``brighter_than`` is False,

    .. math::

        S(z) =
        \frac{
            \int_{\max(M_{\mathrm{thr}},M_{\mathrm{bright}})}^{M_{\mathrm{faint}}}
            \phi(M,z)\,dM
        }{
            \int_{M_{\mathrm{bright}}}^{M_{\mathrm{faint}}}
            \phi(M,z)\,dM
        }.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_threshold: Absolute magnitude threshold.
        m_bright: Bright absolute magnitude bound of the reference window.
        m_faint: Faint absolute magnitude bound of the reference window.
        brighter_than: If True, return the brighter-than-threshold fraction.
            If False, return the fainter-than-threshold fraction.
        n_m: Number of magnitude-grid points used for each integral.

    Returns:
        NumPy array of cumulative selection fractions. Entries are zero where
        the total number density is zero.

    Raises:
        ValueError: If redshift values are negative or if the supplied
            magnitude limits are invalid.
    """
    selected = cumulative_number_density(
        z,
        lf,
        m_threshold=m_threshold,
        m_bright=m_bright,
        m_faint=m_faint,
        brighter_than=brighter_than,
        n_m=n_m,
    )
    total = integrated_number_density(
        z,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )

    return safe_divide(selected, total)


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
    r"""Return an absolute magnitude bound for a magnitude window.

    Args:
        z: Redshift values.
        absolute_mag: Absolute magnitude bound.
        apparent_mag: Apparent magnitude bound.
        luminosity_distance_mpc_fn: Callable returning luminosity distance in Mpc.
        k_correction_fn: Optional K-correction callable.
        e_correction_fn: Optional E-correction callable.
        bound_name: Human-readable bound name used in error messages.

    Returns:
        Absolute magnitude bound evaluated at ``z``.

    Raises:
        ValueError: If neither or both of ``absolute_mag`` and
            ``apparent_mag`` are supplied, or if an apparent magnitude bound is
            supplied without ``luminosity_distance_mpc_fn``.
    """
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
    r"""Return a finite-range luminosity function integral.

    Args:
        z: Redshift value or array.
        lf: Luminosity function callable with signature ``lf(M, z)``.
        m_lower: Lower magnitude bound.
        m_upper: Upper magnitude bound.
        n_m: Number of magnitude-grid points.
        weight_fn: Optional weighting function.

    Returns:
        Luminosity function integral evaluated at ``z``.

    Raises:
        ValueError: If redshift values are negative or if the integration
            bounds are invalid.
    """
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
