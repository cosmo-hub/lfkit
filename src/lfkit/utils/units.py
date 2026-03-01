"""Unit conversion utilities used across LFKit.

This module provides small, self-contained helpers for converting between
astronomical and cosmological unit systems commonly needed in luminosity
function and correction calculations.

Currently supported conversions include:

- Megaparsec to kilometers.
- Gigayear to seconds.
- Hubble constant from km/s/Mpc to 1/Gyr.

These helpers are intentionally lightweight and avoid external dependencies.
They are designed for internal use by cosmology and correction modules.
"""

from __future__ import annotations

import numpy as np


__all__ = (
    "km_per_mpc",
    "sec_per_gyr",
    "h0_km_s_mpc_to_gyr_inv",
    "mag_to_maggies",
    "maggies_to_mag",
    "magerr_to_ivar_maggies",
)

def km_per_mpc() -> float:
    """Return the number of kilometers in one megaparsec.

    Returns:
        Conversion factor from megaparsecs (Mpc) to kilometers (km).
    """
    # IAU exact definition of astronomical unit (meters).
    au_m = 149_597_870_700.0

    # 1 arcsec in radians.
    arcsec_rad = np.pi / (180.0 * 3600.0)

    # Parsec in meters, then megaparsec in kilometers.
    pc_m = au_m / arcsec_rad
    mpc_km = (1e6 * pc_m) / 1000.0
    return mpc_km


def sec_per_gyr() -> float:
    """Return the number of seconds in one gigayear.

    Returns:
        Conversion factor from gigayears (Gyr) to seconds (s).
    """
    seconds_per_day = 86400.0
    days_per_year = 365.25
    years_per_gyr = 1e9
    return seconds_per_day * days_per_year * years_per_gyr


def h0_km_s_mpc_to_gyr_inv(h0_km_s_mpc: float) -> float:
    """Convert a Hubble constant from km/s/Mpc to 1/Gyr.

    Args:
        h0_km_s_mpc: Hubble constant in units of km/s/Mpc.

    Returns:
        The Hubble constant expressed in inverse gigayears (1/Gyr).
    """
    h0_s_inv = float(h0_km_s_mpc) / km_per_mpc()
    return h0_s_inv * sec_per_gyr()


def mag_to_maggies(m_ab: np.ndarray) -> np.ndarray:
    """Converts AB magnitudes to maggies (linear flux units).

    In the AB system:

        m_AB = -2.5 * log10(f_nu) - 48.6

    In kcorrect-style workflows, fluxes are expressed in *maggies*:

        f_maggies = 10**(-0.4 * m_AB)

    where 1 maggie corresponds to m_AB = 0.

    Args:
        m_ab: AB magnitudes. Can be scalar-like or array-like.

    Returns:
        Fluxes in maggies with the same shape as the input.

    Notes:
        This is a purely algebraic transformation with no cosmology
        dependence. The returned values are linear flux densities in
        arbitrary units normalized such that m_AB = 0 corresponds to
        f = 1 maggie.
    """
    m = np.asarray(m_ab, dtype=float)
    return 10.0 ** (-0.4 * m)


def maggies_to_mag(maggies: np.ndarray, *, floor: float = 1e-300) -> np.ndarray:
    """Convert maggies to magnitudes with a safety floor.

    Args:
        maggies: Fluxes in maggies.
        floor: Minimum value to avoid log(0).

    Returns:
        Magnitudes in AB system.
    """
    m = np.asarray(maggies, dtype=float)
    m = np.maximum(m, float(floor))
    return -2.5 * np.log10(m)


def magerr_to_ivar_maggies(m_ab: np.ndarray, sigma_m: np.ndarray) -> np.ndarray:
    """Converts magnitude uncertainties to inverse variance in maggies.

    Flux uncertainties are computed using first-order error propagation.
    Given:

        f = 10**(-0.4 * m)

    the derivative is:

        df/dm = -0.4 * ln(10) * f

    so the propagated flux uncertainty is approximated as:

        sigma_f = (0.4 * ln(10)) * f * sigma_m

    and the inverse variance is:

        ivar = 1 / sigma_f**2

    Args:
        m_ab: AB magnitudes.
        sigma_m: 1σ magnitude uncertainties corresponding to `m_ab`.

    Returns:
        Inverse variance of the fluxes in maggies (1 / sigma_f**2).
        Entries are set to zero where the propagated uncertainty is
        non-finite or non-positive.

    Notes:
        Uses linear (first-order) error propagation and assumes small
        magnitude errors. No covariance between bands is included.
        Non-finite or zero uncertainties yield zero inverse variance.
    """
    m = np.asarray(m_ab, dtype=float)
    sm = np.asarray(sigma_m, dtype=float)
    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sm
    ivar = np.zeros_like(sigma_f)
    ok = np.isfinite(sigma_f) & (sigma_f > 0)
    ivar[ok] = 1.0 / (sigma_f[ok] ** 2)
    return ivar
