"""Unit tests for the `lfkit.utils.units` module."""

from __future__ import annotations

import numpy as np

from lfkit.utils.units import (
    km_per_mpc,
    sec_per_gyr,
    h0_km_s_mpc_to_gyr_inv,
    mag_to_maggies,
    maggies_to_mag,
    magerr_to_ivar_maggies,
)


def test_km_per_mpc_is_positive_and_reasonable():
    """Tests that km_per_mpc returns a positive value consistent with the IAU AU + parsec definition."""
    val = km_per_mpc()
    assert np.isfinite(val)
    assert val > 0.0

    # Use the same defining relation:
    # 1 pc = 1 AU / tan(1 arcsec) ≈ 1 AU / (1 arcsec in radians)
    # with 1 arcsec = pi / 648000 rad, so pc = AU * 648000 / pi exactly under this convention.
    au_m = 149_597_870_700.0  # IAU exact AU in meters
    arcsec_rad = np.pi / (180.0 * 3600.0)  # pi / 648000
    pc_m = au_m / arcsec_rad
    expected = (1e6 * pc_m) / 1000.0  # Mpc in km

    assert np.isclose(val, expected, rtol=0.0, atol=0.0)


def test_sec_per_gyr_is_positive_and_reasonable():
    """Tests that sec_per_gyr returns a positive value equal to 86400*365.25*1e9."""
    val = sec_per_gyr()
    assert np.isfinite(val)
    assert val > 0.0
    assert np.isclose(val, 86400.0 * 365.25 * 1e9, rtol=0.0, atol=0.0)


def test_h0_conversion_matches_manual_formula():
    """Tests that h0_km_s_mpc_to_gyr_inv matches h0/km_per_mpc*sec_per_gyr."""
    h0 = 70.0
    expected = (h0 / km_per_mpc()) * sec_per_gyr()
    got = h0_km_s_mpc_to_gyr_inv(h0)
    assert np.isfinite(got)
    assert np.isclose(got, expected, rtol=0.0, atol=0.0)


def test_mag_to_maggies_known_points_and_roundtrip():
    """Tests that mag_to_maggies maps m=0 to 1 and roundtrips with maggies_to_mag."""
    m = np.array([0.0, 2.5, 10.0])
    f = mag_to_maggies(m)
    assert np.all(np.isfinite(f))
    assert np.isclose(f[0], 1.0, rtol=0.0, atol=0.0)
    assert np.isclose(f[1], 0.1, rtol=0.0, atol=1e-15)  # 10**(-1)

    m_back = maggies_to_mag(f)
    assert np.all(np.isfinite(m_back))
    assert np.allclose(m_back, m, rtol=0.0, atol=1e-12)


def test_maggies_to_mag_floor_prevents_infs():
    """Tests that maggies_to_mag applies a floor so zero/negative fluxes yield finite magnitudes."""
    f = np.array([0.0, -1.0, 1e-320, 1.0])
    m = maggies_to_mag(f, floor=1e-300)
    assert np.all(np.isfinite(m))
    # Everything <= floor maps to the same magnitude
    assert np.isclose(m[0], m[2], rtol=0.0, atol=0.0)
    assert np.isclose(m[1], m[2], rtol=0.0, atol=0.0)


def test_magerr_to_ivar_maggies_matches_propagation_and_masks_bad():
    """Tests that magerr_to_ivar_maggies matches first-order propagation and returns 0 for non-finite/invalid errors."""
    m = np.array([20.0, 21.0, 22.0])
    sm = np.array([0.1, np.nan, 0.0])

    ivar = magerr_to_ivar_maggies(m, sm)

    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sm
    expected = np.zeros_like(sigma_f)
    ok = np.isfinite(sigma_f) & (sigma_f > 0)
    expected[ok] = 1.0 / (sigma_f[ok] ** 2)

    assert np.all(np.isfinite(ivar))
    assert np.allclose(ivar, expected, rtol=0.0, atol=0.0)
    assert ivar[1] == 0.0  # nan sigma_m -> 0 ivar
    assert ivar[2] == 0.0  # zero sigma_m -> 0 ivar
