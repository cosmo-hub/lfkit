r"""Luminosity-function integration utilities.

This module provides generic numerical integrals of luminosity function
callables over finite absolute-magnitude ranges.

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
from lfkit.utils.validators import validate_array


__all__ = [
    "integrated_number_density",
    "lf_weighted_integral",
    "selection_weighted_number_density",
    "integrated_luminosity_density",
    "mean_luminosity",
    "cumulative_number_density",
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
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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

    return _safe_divide(luminosity_density, number_density)


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
        m_threshold: Absolute-magnitude threshold. May be scalar or array-like.
        m_bright: Bright absolute-magnitude bound. May be scalar or array-like.
        m_faint: Faint absolute-magnitude bound. May be scalar or array-like.
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

    integral = np.zeros_like(z_b, dtype=float)
    valid = m_upper_b > m_lower_b

    if not np.any(valid):
        return np.asarray(integral, dtype=float)

    m_grid = _magnitude_grid(
        m_lower=m_lower_b[valid],
        m_upper=m_upper_b[valid],
        n_m=n_m,
    )
    z_grid = np.broadcast_to(z_b[valid][None, :], m_grid.shape)

    phi = _evaluate_lf_on_grid(lf, m_grid=m_grid, z_grid=z_grid)

    if weight_fn is not None:
        weight = _evaluate_weight_on_grid(
            weight_fn,
            m_grid=m_grid,
            z_grid=z_grid,
        )
        phi = phi * weight

    integral[valid] = np.trapezoid(phi, x=m_grid, axis=0)

    return np.asarray(integral, dtype=float)


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

    if np.any(phi < 0.0):
        raise ValueError("lf(M, z) must be non-negative.")

    return np.asarray(phi, dtype=float)


def _evaluate_weight_on_grid(
    weight_fn: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    m_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    r"""Return finite non-negative weight values on a magnitude-redshift grid."""
    weight = np.asarray(weight_fn(m_grid, z_grid), dtype=float)

    if weight.shape != m_grid.shape:
        try:
            weight = np.broadcast_to(weight, m_grid.shape)
        except ValueError as exc:
            raise ValueError(
                "weight_fn(M, z) must return values broadcastable to the shape "
                "of the magnitude-redshift integration grid."
            ) from exc

    if np.any(~np.isfinite(weight)):
        raise ValueError("weight_fn(M, z) returned non-finite values.")

    if np.any(weight < 0.0):
        raise ValueError("weight_fn(M, z) must be non-negative.")

    return np.asarray(weight, dtype=float)


def _validate_integration_inputs(
    *,
    z: FloatArray,
    m_lower: FloatArray,
    m_upper: FloatArray,
    n_m: int,
) -> None:
    r"""Validate inputs for finite-range LF integration."""
    if np.any(z < 0.0):
        raise ValueError("Redshift z must be >= 0.")

    if np.any(~np.isfinite(m_lower)):
        raise ValueError("m_lower must contain only finite values.")

    if np.any(~np.isfinite(m_upper)):
        raise ValueError("m_upper must contain only finite values.")

    if n_m < 2:
        raise ValueError("n_m must be at least 2.")


def _safe_divide(
    numerator: FloatArray,
    denominator: FloatArray,
) -> FloatArray:
    r"""Return numerator / denominator with zero output for zero denominator."""
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            numerator_arr,
            denominator_arr,
            out=np.zeros_like(numerator_arr, dtype=float),
            where=denominator_arr > 0.0,
        )

    return np.asarray(result, dtype=float)
