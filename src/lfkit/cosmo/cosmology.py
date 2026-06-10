"""Cosmology utilities for LFKit.

This module provides wrappers around PyCCL cosmology
objects and background calculations.

It standardizes how cosmology instances are created and how
lookback time is computed, ensuring consistent behavior across LFKit.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyccl as ccl

if TYPE_CHECKING:
    Cosmology = ccl.Cosmology
else:
    Cosmology = object


__all__ = (
    "C_KM_S",
    "cosmo_object",
    "lookback_time_gyr",
    "luminosity_distance_mpc",
    "comoving_distance_mpc",
    "distance_modulus",
    "differential_comoving_volume",
)

C_KM_S = 299792.458  # speed of light in vacuum in km/s


def cosmo_object(
    *,
    instance: Cosmology | None = None,
    **params: Any,
) -> Cosmology:
    """Return a PyCCL cosmology object.

    This function provides a standardized way to obtain a
    ``pyccl.Cosmology`` instance within LFKit.

    Behavior:
        1. If ``instance`` is provided, it is returned unchanged.
        2. Else if cosmological parameters are provided, a new
           ``ccl.Cosmology`` is constructed using those parameters.
        3. Else, a default ``ccl.CosmologyVanillaLCDM`` is returned.

    Args:
        instance: Pre-built ``ccl.Cosmology`` object.
        **params: Cosmological parameters passed directly to
            ``ccl.Cosmology(**params)``.

    Returns:
        A PyCCL cosmology object.

    Raises:
        ValueError: If both ``instance`` and cosmological parameters
            are provided.
    """
    if instance is not None:
        if params:
            raise ValueError("Pass instance OR parameters, not both.")
        return instance

    if params:
        return ccl.Cosmology(**params)

    return ccl.CosmologyVanillaLCDM()


def lookback_time_gyr(
    cosmo_obj: Cosmology,
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Compute lookback time in gigayears.

    This function evaluates the cosmological lookback time using
    PyCCL background calculations.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.

    Returns:
        NumPy array of lookback time values in gigayears.
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    return np.asarray(ccl.background.lookback_time(cosmo_obj, a), dtype=float)


def luminosity_distance_mpc(
    cosmo_obj: Cosmology,
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Compute luminosity distance in megaparsecs.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.

    Returns:
        NumPy array of luminosity distances in Mpc.
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    return np.asarray(ccl.luminosity_distance(cosmo_obj, a), dtype=float)


def comoving_distance_mpc(
    cosmo_obj: Cosmology,
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Compute comoving radial distance in megaparsecs.

    This follows the same explicit ``c / H(z)`` integral structure
    used in the old code, rather than relying on a compact shortcut.

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.

    Returns:
        NumPy array of comoving radial distances in Mpc.
    """
    z = np.asarray(z, dtype=float)

    h = float(cosmo_obj["h"])
    omega_m = float(cosmo_obj["Omega_c"] + cosmo_obj["Omega_b"])
    omega_l = 1.0 - omega_m

    h0 = 100.0 * h  # km/s/Mpc

    hubble_distance = C_KM_S / h0
    ez = np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)

    distance = np.zeros_like(z, dtype=float)
    integrand = hubble_distance / ez

    for i in range(len(z)):
        distance[i] = np.trapezoid(integrand[: i + 1], z[: i + 1])

    return distance


def distance_modulus(
    cosmo_obj: Cosmology,
    z: ArrayLike,
    h: float | None = None,
) -> NDArray[np.float64]:
    """Compute distance modulus.

    If ``h`` is provided, this uses the same convention as the old code:

    ``mu = 5 log10(d_L * h) + 25``

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        h: Optional dimensionless Hubble parameter used to rescale the
            luminosity distance before computing the modulus.

    Returns:
        NumPy array of distance modulus values.
    """
    d_l = luminosity_distance_mpc(cosmo_obj, z)
    if h is not None:
        d_l = d_l * h
    return np.asarray(5.0 * np.log10(d_l) + 25.0, dtype=float)


def differential_comoving_volume(
    cosmo_obj: Cosmology,
    z: ArrayLike,
    frac_sky: float = 1.0,
) -> NDArray[np.float64]:
    """Compute differential comoving volume per unit redshift.

    This follows the same structure as the old code:

    ``dV/dz = 4 pi f_sky chi(z)^2 c / H(z)``

    Args:
        cosmo_obj: A PyCCL cosmology object.
        z: Redshift value or array-like of redshift values.
        frac_sky: Fraction of sky covered by the survey.

    Returns:
        NumPy array of differential comoving volume values in
        Mpc^3 per unit redshift.
    """
    z = np.asarray(z, dtype=float)

    h = float(cosmo_obj["h"])
    omega_m = float(cosmo_obj["Omega_c"] + cosmo_obj["Omega_b"])
    omega_l = 1.0 - omega_m

    h0 = 100.0 * h  # km/s/Mpc

    chi = comoving_distance_mpc(cosmo_obj, z)
    hz = h0 * np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)

    prefactor = 4.0 * np.pi * frac_sky
    return np.asarray(prefactor * C_KM_S * chi**2 / hz, dtype=float)
