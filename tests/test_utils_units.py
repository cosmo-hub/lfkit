"""Unit tests for the ``lfkit.utils.units``."""

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


def test_h0_conversion_accepts_scalar_like_inputs() -> None:
    """Tests that H0 conversion accepts NumPy scalar inputs."""
    h0 = np.float64(100.0)
    got = h0_km_s_mpc_to_gyr_inv(h0)
    expected = (100.0 / km_per_mpc()) * sec_per_gyr()
    assert np.isclose(got, expected, rtol=0.0, atol=0.0)


def test_h0_conversion_preserves_sign() -> None:
    """Tests that H0 conversion is algebraic and preserves sign."""
    assert h0_km_s_mpc_to_gyr_inv(0.0) == 0.0
    assert h0_km_s_mpc_to_gyr_inv(-70.0) < 0.0


def test_mag_to_maggies_accepts_scalar_input() -> None:
    """Tests that scalar magnitude input returns scalar-shaped maggies."""
    got = mag_to_maggies(0.0)
    assert np.shape(got) == ()
    assert np.isclose(got, 1.0, rtol=0.0, atol=0.0)


def test_maggies_to_mag_accepts_scalar_input() -> None:
    """Tests that scalar maggie input returns scalar-shaped magnitude."""
    got = maggies_to_mag(1.0)
    assert np.shape(got) == ()
    assert np.isclose(got, 0.0, rtol=0.0, atol=0.0)


def test_mag_to_maggies_preserves_array_shape() -> None:
    """Tests that magnitude-to-maggies conversion preserves input shape."""
    m = np.array([[0.0, 2.5], [5.0, 7.5]])
    f = mag_to_maggies(m)

    assert f.shape == m.shape
    np.testing.assert_allclose(f, 10.0 ** (-0.4 * m))


def test_maggies_to_mag_preserves_array_shape() -> None:
    """Tests that maggies-to-magnitude conversion preserves input shape."""
    f = np.array([[1.0, 0.1], [0.01, 0.001]])
    m = maggies_to_mag(f)

    assert m.shape == f.shape
    np.testing.assert_allclose(m, -2.5 * np.log10(f))


def test_magnitude_difference_maps_to_flux_ratio() -> None:
    """Tests that a 2.5 mag increase lowers maggies by a factor of 10."""
    f0 = mag_to_maggies(20.0)
    f1 = mag_to_maggies(22.5)

    np.testing.assert_allclose(f1 / f0, 0.1, rtol=1e-14)


def test_maggies_to_mag_custom_floor() -> None:
    """Tests that a custom floor controls the maximum returned magnitude."""
    m = maggies_to_mag(np.array([0.0, 1e-20]), floor=1e-10)

    np.testing.assert_allclose(m, np.array([25.0, 25.0]))


def test_magerr_to_ivar_maggies_broadcasts_inputs() -> None:
    """Tests that magerr_to_ivar_maggies supports NumPy broadcasting."""
    m = np.array([20.0, 21.0, 22.0])
    sigma_m = 0.1

    ivar = magerr_to_ivar_maggies(m, sigma_m)

    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sigma_m
    expected = 1.0 / sigma_f**2

    assert ivar.shape == m.shape
    np.testing.assert_allclose(ivar, expected)


def test_magerr_to_ivar_maggies_negative_sigma_is_masked() -> None:
    """Tests that negative magnitude uncertainties produce zero inverse variance."""
    ivar = magerr_to_ivar_maggies(
        np.array([20.0, 21.0]),
        np.array([-0.1, 0.1]),
    )

    assert ivar[0] == 0.0
    assert ivar[1] > 0.0


def test_magerr_to_ivar_maggies_infinite_sigma_is_masked() -> None:
    """Tests that infinite propagated uncertainty produces zero inverse variance."""
    ivar = magerr_to_ivar_maggies(
        np.array([20.0, 21.0]),
        np.array([np.inf, 0.1]),
    )

    assert ivar[0] == 0.0
    assert ivar[1] > 0.0


def test_magerr_to_ivar_maggies_preserves_shape() -> None:
    """Tests that inverse-variance conversion preserves broadcasted array shape."""
    m = np.array([[20.0, 21.0], [22.0, 23.0]])
    sigma_m = np.full_like(m, 0.1)

    ivar = magerr_to_ivar_maggies(m, sigma_m)

    assert ivar.shape == m.shape
    assert np.all(ivar > 0.0)
