"""Magnitude conversion utilities for LFKit.

This module provides helpers to translate between apparent ``m`` and
absolute magnitudes ``M`` using the cosmology utilities defined in LFKit.

The conversions follow the convention

    M = m - mu - K + E
    m = M + mu + K - E

where ``mu`` is the distance modulus, ``K`` is the k-correction,
and ``E`` is the evolution correction.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

import numpy as np

from lfkit.cosmo.cosmology import distance_modulus
from lfkit.utils.types import Cosmology, FloatArray, FloatInput
from lfkit.utils.validators import validate_luminosity_distance

__all__ = [
    "total_magnitude_correction",
    "absolute_magnitude",
    "absolute_magnitude_from_luminosity_distance",
    "apparent_magnitude",
    "apparent_magnitude_from_luminosity_distance",
]

__api_aliases__ = {
    "total_magnitude_correction": "correction",
    "absolute_magnitude": "absolute",
    "absolute_magnitude_from_luminosity_distance": "absolute_from_luminosity_distance",
    "apparent_magnitude": "apparent",
    "apparent_magnitude_from_luminosity_distance": "apparent_from_luminosity_distance",
}


def total_magnitude_correction(
    *,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    """Return the net correction added to apparent absolute conversion.

    This combines optional k-correction and evolution-correction terms as

    ``correction = K - E``

    so that

    ``M = m - mu - correction``.

    Args:
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of net correction values.
    """
    correction = np.asarray(0.0, dtype=float)

    if k_correction is not None:
        correction = correction + np.asarray(k_correction, dtype=float)

    if e_correction is not None:
        correction = correction - np.asarray(e_correction, dtype=float)

    return np.asarray(correction, dtype=np.float64)


def absolute_magnitude(
    cosmo_obj: Cosmology,
    z: FloatInput,
    apparent_mag: FloatInput,
    *,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    """Convert apparent magnitude m to absolute magnitude M.

    The convention used is

    ``M = m - mu - K + E``.

    Args:
        cosmo_obj: Cosmology object passed to LFKit distance utilities.
        z: Redshift value or array-like of redshift values.
        apparent_mag: Apparent magnitude value(s).
        h: Optional dimensionless Hubble parameter. If provided,
            the distance modulus uses ``mu = 5 log10(d_L * h) + 25``.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of absolute magnitudes.
    """
    m = np.asarray(apparent_mag, dtype=float)
    mu = distance_modulus(cosmo_obj, z, h=h)
    correction = total_magnitude_correction(
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return np.asarray(m - mu - correction, dtype=np.float64)


def apparent_magnitude(
    cosmo_obj: Cosmology,
    z: FloatInput,
    absolute_mag: FloatInput,
    *,
    h: float | None = None,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    """Convert absolute magnitude M to apparent magnitude m.

    The convention used is

    ``m = M + mu + K - E``.

    Args:
        cosmo_obj: Cosmology object passed to LFKit distance utilities.
        z: Redshift value or array-like of redshift values.
        absolute_mag: Absolute magnitude value(s).
        h: Optional dimensionless Hubble parameter. If provided,
            the distance modulus uses ``mu = 5 log10(d_L * h) + 25``.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of apparent magnitudes.
    """
    M = np.asarray(absolute_mag, dtype=float)
    mu = distance_modulus(cosmo_obj, z, h=h)
    correction = total_magnitude_correction(
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return np.asarray(M + mu + correction, dtype=np.float64)


def absolute_magnitude_from_luminosity_distance(
    apparent_mag: FloatInput,
    luminosity_distance_mpc: FloatInput,
    *,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    """Convert apparent magnitude to absolute magnitude from luminosity distance.

    The convention used is

    ``M = m - mu - K + E``,

    with

    ``mu = 5 log10(d_L / Mpc) + 25``.

    Args:
        apparent_mag: Apparent magnitude value(s).
        luminosity_distance_mpc: Luminosity distance value(s) in Mpc.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of absolute magnitudes.
    """
    m = np.asarray(apparent_mag, dtype=float)
    d_l = validate_luminosity_distance(luminosity_distance_mpc)

    mu = 5.0 * np.log10(d_l) + 25.0
    correction = total_magnitude_correction(
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return np.asarray(m - mu - correction, dtype=np.float64)


def apparent_magnitude_from_luminosity_distance(
    absolute_mag: FloatInput,
    luminosity_distance_mpc: FloatInput,
    *,
    k_correction: FloatInput | None = None,
    e_correction: FloatInput | None = None,
) -> FloatArray:
    """Convert absolute magnitude to apparent magnitude from luminosity distance.

    The convention used is

    ``m = M + mu + K - E``,

    with

    ``mu = 5 log10(d_L / Mpc) + 25``.

    Args:
        absolute_mag: Absolute magnitude value(s).
        luminosity_distance_mpc: Luminosity distance value(s) in Mpc.
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of apparent magnitudes.
    """
    M = np.asarray(absolute_mag, dtype=float)
    d_l = validate_luminosity_distance(luminosity_distance_mpc)

    mu = 5.0 * np.log10(d_l) + 25.0
    correction = total_magnitude_correction(
        k_correction=k_correction,
        e_correction=e_correction,
    )

    return np.asarray(M + mu + correction, dtype=np.float64)
