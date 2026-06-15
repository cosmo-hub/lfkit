"""Unit tests for ``lfkit.cosmo.cosmology``."""

from __future__ import annotations

import numpy as np
import pyccl as ccl
import pytest

from lfkit.cosmo.cosmology import (
    C_KM_S,
    comoving_distance_mpc,
    cosmo_object,
    differential_comoving_volume,
    distance_modulus,
    lookback_time_gyr,
    luminosity_distance_mpc,
)


def test_cosmo_object_returns_instance_unchanged():
    """Tests that cosmo_object returns the provided instance unchanged."""
    inst = ccl.CosmologyVanillaLCDM()
    out = cosmo_object(instance=inst)
    assert out is inst


def test_cosmo_object_raises_if_instance_and_params_given():
    """Tests that cosmo_object raises ValueError when instance and params are given."""
    inst = ccl.CosmologyVanillaLCDM()
    with pytest.raises(ValueError):
        cosmo_object(instance=inst, Omega_c=0.25)


def test_cosmo_object_builds_from_params():
    """Tests that cosmo_object constructs a ccl.Cosmology from parameters."""
    cosmo = cosmo_object(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
    )
    assert isinstance(cosmo, ccl.Cosmology)


def test_cosmo_object_defaults_to_vanilla_lcdm():
    """Tests that cosmo_object returns a default ccl.Cosmology."""
    cosmo = cosmo_object()
    assert isinstance(cosmo, ccl.Cosmology)


def test_lookback_time_gyr_shape_and_dtype_scalar_and_array():
    """Tests that lookback_time_gyr returns float arrays matching input shapes."""
    cosmo = ccl.CosmologyVanillaLCDM()

    t0 = lookback_time_gyr(cosmo, 0.0)
    assert isinstance(t0, np.ndarray)
    assert t0.shape == ()
    assert t0.dtype == float
    assert np.isfinite(t0)

    z = np.array([0.0, 0.5, 1.0])
    t = lookback_time_gyr(cosmo, z)
    assert isinstance(t, np.ndarray)
    assert t.shape == z.shape
    assert t.dtype == float
    assert np.all(np.isfinite(t))


def test_lookback_time_gyr_monotonic_in_redshift():
    """Tests that lookback_time_gyr is non-decreasing with redshift."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    t = lookback_time_gyr(cosmo, z)
    assert np.all(t[1:] >= t[:-1])


def test_lookback_time_gyr_zero_at_z0_with_tolerance():
    """Tests that lookback_time_gyr is approximately zero at z=0."""
    cosmo = ccl.CosmologyVanillaLCDM()
    t0 = float(lookback_time_gyr(cosmo, 0.0))
    assert abs(t0) < 1e-10


def test_speed_of_light_constant():
    """Tests that C_KM_S stores the expected speed of light in km/s."""
    assert C_KM_S == pytest.approx(299792.458)


def test_luminosity_distance_shape_dtype_and_z0():
    """Tests that luminosity_distance_mpc returns floats and zero distance at z=0."""
    cosmo = ccl.CosmologyVanillaLCDM()

    d0 = luminosity_distance_mpc(cosmo, 0.0)
    assert isinstance(d0, np.ndarray)
    assert d0.shape == ()
    assert d0.dtype == float
    assert d0 == pytest.approx(0.0)

    z = np.array([0.1, 0.5, 1.0])
    d = luminosity_distance_mpc(cosmo, z)
    assert d.shape == z.shape
    assert d.dtype == float
    assert np.all(np.isfinite(d))
    assert np.all(d > 0.0)
    assert np.all(d[1:] > d[:-1])


def test_comoving_distance_shape_dtype_z0_and_monotonic():
    """Tests that comoving_distance_mpc returns floats starting at zero."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.0, 0.1, 0.5, 1.0])

    chi = comoving_distance_mpc(cosmo, z)

    assert isinstance(chi, np.ndarray)
    assert chi.shape == z.shape
    assert chi.dtype == float
    assert chi[0] == pytest.approx(0.0)
    assert np.all(np.isfinite(chi))
    assert np.all(chi[1:] > chi[:-1])


def test_comoving_distance_matches_ccl_comoving_radial_distance_approximately():
    """Tests that comoving_distance_mpc approximately matches PyCCL distances."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0.0, 2.0, 512)
    a = 1.0 / (1.0 + z)

    chi = comoving_distance_mpc(cosmo, z)
    expected = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float)

    assert np.allclose(chi, expected, rtol=5e-3, atol=1e-6)


def test_distance_modulus_matches_luminosity_distance_formula():
    """Tests that distance_modulus applies the luminosity-distance formula."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.1, 0.5, 1.0])

    d_l = luminosity_distance_mpc(cosmo, z)
    mu = distance_modulus(cosmo, z)

    assert np.allclose(mu, 5.0 * np.log10(d_l) + 25.0)


def test_distance_modulus_with_h_rescaling():
    """Tests that distance_modulus applies the optional h rescaling convention."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.1, 0.5, 1.0])
    h = 0.7

    d_l = luminosity_distance_mpc(cosmo, z)
    mu = distance_modulus(cosmo, z, h=h)

    assert np.allclose(mu, 5.0 * np.log10(d_l * h) + 25.0)


def test_differential_comoving_volume_shape_dtype_and_nonnegative():
    """Tests that differential_comoving_volume returns finite non-negative floats."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.0, 0.1, 0.5, 1.0])

    dv_dz = differential_comoving_volume(cosmo, z)

    assert isinstance(dv_dz, np.ndarray)
    assert dv_dz.shape == z.shape
    assert dv_dz.dtype == float
    assert np.all(np.isfinite(dv_dz))
    assert dv_dz[0] == pytest.approx(0.0)
    assert np.all(dv_dz >= 0.0)


def test_differential_comoving_volume_scales_with_frac_sky():
    """Tests that differential_comoving_volume scales linearly with sky fraction."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.1, 0.5, 1.0])

    full_sky = differential_comoving_volume(cosmo, z, frac_sky=1.0)
    half_sky = differential_comoving_volume(cosmo, z, frac_sky=0.5)

    assert np.allclose(half_sky, 0.5 * full_sky)
