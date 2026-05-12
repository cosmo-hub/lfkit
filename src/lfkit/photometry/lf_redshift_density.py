r"""Luminosity-function redshift-density utilities.

This module provides helpers for converting a luminosity-function callable into
an LF-integrated redshift-density curve.

The core operation is

    n_lf(z) = int phi(M, z) dM

over the observable absolute-magnitude range implied by an apparent-magnitude
limit. A second helper multiplies this LF-integrated density by a user-supplied
redshift or volume weight.

The cosmology-dependent distance and volume pieces are supplied as callables.
This keeps the interface independent of CCL, Astropy, or any other cosmology
backend, which is useful for downstream packages such as Binny.

Magnitude corrections are supplied as scalar or array-like values evaluated on
the same redshift grid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.photometry.lf_integrals import integrated_number_density
from lfkit.photometry.magnitudes import absolute_magnitude_from_luminosity_distance
from lfkit.utils.evaluators import (
    evaluate_non_negative_redshift_callable,
    evaluate_positive_redshift_callable,
)
from lfkit.utils.types import FloatArray, FloatInput, LuminosityFunction
from lfkit.utils.validators import validate_array


__all__ = [
    "lf_integrated_number_density",
    "lf_weighted_redshift_density",
]


def lf_integrated_number_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    n_m: int = 512,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray],
    k_correction: FloatInput | None = None,
    evolution_correction: FloatInput | None = None,
) -> FloatArray:
    r"""Return LF-integrated number density as a function of redshift.

    This computes

    .. math::

        n_{\mathrm{LF}}(z) =
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{lim}}(z)}
        \phi(M, z)\,dM,

    where ``M_lim(z)`` is the absolute-magnitude limit implied by the apparent
    magnitude cut ``m_lim``.

    The magnitude conversion follows

    .. math::

        M_{\mathrm{lim}}(z) =
        m_{\mathrm{lim}} - \mu(z) - K(z) + E(z),

    where ``mu`` is the distance modulus, ``K`` is the k-correction, and ``E``
    is the evolution correction.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude integration bound.
        n_m: Number of magnitude-grid points used for the integral.
        luminosity_distance_mpc_fn: Callable returning luminosity distance in
            Mpc as a function of redshift.
        k_correction: Optional scalar or array-like k-correction values.
        evolution_correction: Optional scalar or array-like evolution-correction
            values.

    Returns:
        NumPy array of LF-integrated number densities evaluated at ``z``.
    """
    z_arr = validate_array(z, name="z")

    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0.")

    if not np.isfinite(m_lim):
        raise ValueError("m_lim must be finite.")

    if not np.isfinite(m_bright):
        raise ValueError("m_bright must be finite.")

    luminosity_distance = evaluate_positive_redshift_callable(
        luminosity_distance_mpc_fn,
        z_arr,
        name="luminosity_distance_mpc_fn",
    )

    k_correction_arr = _optional_correction_array(
        k_correction,
        z_arr,
        name="k_correction",
    )

    evolution_correction_arr = _optional_correction_array(
        evolution_correction,
        z_arr,
        name="evolution_correction",
    )

    m_faint = absolute_magnitude_from_luminosity_distance(
        m_lim,
        luminosity_distance,
        k_correction=k_correction_arr,
        e_correction=evolution_correction_arr,
    )

    return integrated_number_density(
        z_arr,
        lf,
        m_bright=m_bright,
        m_faint=m_faint,
        n_m=n_m,
    )


def lf_weighted_redshift_density(
    z: FloatInput,
    lf: LuminosityFunction,
    *,
    m_lim: float,
    m_bright: float,
    n_m: int = 512,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray],
    volume_weight_fn: Callable[[FloatArray], FloatArray],
    k_correction: FloatInput | None = None,
    evolution_correction: FloatInput | None = None,
    normalize: bool = True,
) -> FloatArray:
    r"""Return an LF-weighted redshift-density curve.

    This computes

    .. math::

        n(z) \propto W(z)
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{lim}}(z)}
        \phi(M, z)\,dM,

    where ``W(z)`` is supplied by ``volume_weight_fn``.

    Args:
        z: Redshift value or array-like of redshift values.
        lf: Luminosity-function callable with signature ``lf(M, z)``.
        m_lim: Apparent magnitude limit of the catalog.
        m_bright: Bright absolute-magnitude integration bound.
        n_m: Number of magnitude-grid points used for the integral.
        luminosity_distance_mpc_fn: Callable returning luminosity distance in
            Mpc as a function of redshift.
        volume_weight_fn: Callable returning the redshift or volume weight.
        k_correction: Optional scalar or array-like k-correction values.
        evolution_correction: Optional scalar or array-like evolution-correction
            values.
        normalize: If True, normalize the returned curve to integrate to one
            over ``z``.

    Returns:
        NumPy array of LF-weighted redshift-density values.
    """
    z_arr = validate_array(z, name="z")

    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0.")

    n_lf = lf_integrated_number_density(
        z_arr,
        lf,
        m_lim=m_lim,
        m_bright=m_bright,
        n_m=n_m,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction=k_correction,
        evolution_correction=evolution_correction,
    )

    volume_weight = evaluate_non_negative_redshift_callable(
        volume_weight_fn,
        z_arr,
        name="volume_weight_fn",
    )

    nz = n_lf * volume_weight

    if normalize:
        norm = np.trapezoid(nz, x=z_arr)

        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                "Cannot normalize LF-weighted redshift density with "
                "non-positive integral."
            )

        nz = nz / norm

    return np.asarray(nz, dtype=float)


def _optional_correction_array(
    correction: FloatInput | None,
    z: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Return an optional scalar or array correction broadcast to redshift."""
    if correction is None:
        return np.zeros_like(z, dtype=float)

    correction_arr = validate_array(correction, name=name)

    try:
        correction_b = np.broadcast_to(correction_arr, z.shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to the shape of z."
        ) from exc

    return np.asarray(correction_b, dtype=float)
