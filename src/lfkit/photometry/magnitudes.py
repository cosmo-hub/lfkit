"""Magnitude conversion utilities for LFKit.

This module provides helpers to translate between apparent and
absolute magnitudes using the cosmology utilities defined in LFKit.

The conversions follow the convention

    M = m - mu - K - E
    m = M + mu + K + E

where ``mu`` is the distance modulus, ``K`` is the k-correction,
and ``E`` is the evolution correction.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lfkit.cosmo.cosmology import distance_modulus

if TYPE_CHECKING:
    import pyccl as ccl

    Cosmology = ccl.Cosmology
else:
    Cosmology = object


__all__ = (
    "total_magnitude_correction",
    "absolute_magnitude",
    "apparent_magnitude",
)


def total_magnitude_correction(
    *,
    k_correction: ArrayLike | float | None = None,
    e_correction: ArrayLike | float | None = None,
) -> NDArray[np.float64]:
    """Return the total additive magnitude correction.

    This combines optional k-correction and evolution-correction
    terms into a single array:

    ``correction = K + E``

    Args:
        k_correction: Optional k-correction term(s).
        e_correction: Optional evolution-correction term(s).

    Returns:
        NumPy array of total correction values.
    """
    correction = 0.0

    if k_correction is not None:
        correction = correction + np.asarray(k_correction, dtype=float)

    if e_correction is not None:
        correction = correction + np.asarray(e_correction, dtype=float)

    return np.asarray(correction, dtype=float)


def absolute_magnitude(
    cosmo_obj: Cosmology,
    z: ArrayLike,
    apparent_mag: ArrayLike,
    *,
    h: float | None = None,
    k_correction: ArrayLike | float | None = None,
    e_correction: ArrayLike | float | None = None,
) -> NDArray[np.float64]:
    """Convert apparent magnitude to absolute magnitude.

    The convention used is

    ``M = m - mu - K - E``

    where ``mu`` is the distance modulus, ``K`` is the k-correction,
    and ``E`` is the evolution correction.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        apparent_mag: Apparent magnitude value(s).
        h: Optional dimensionless Hubble parameter. If provided,
            the distance modulus uses the convention
            ``mu = 5 log10(d_L * h) + 25``.
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
    return np.asarray(m - mu - correction, dtype=float)


def apparent_magnitude(
    cosmo_obj: Cosmology,
    z: ArrayLike,
    absolute_mag: ArrayLike,
    *,
    h: float | None = None,
    k_correction: ArrayLike | float | None = None,
    e_correction: ArrayLike | float | None = None,
) -> NDArray[np.float64]:
    """Convert absolute magnitude to apparent magnitude.

    The convention used is

    ``m = M + mu + K + E``

    where ``mu`` is the distance modulus, ``K`` is the k-correction,
    and ``E`` is the evolution correction.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        absolute_mag: Absolute magnitude value(s).
        h: Optional dimensionless Hubble parameter. If provided,
            the distance modulus uses the convention
            ``mu = 5 log10(d_L * h) + 25``.
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
    return np.asarray(M + mu + correction, dtype=float)
