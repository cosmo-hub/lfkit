"""Unit tests for `lfkit.cosmo.cosmology` module."""

from __future__ import annotations

import numpy as np
import pytest
import pyccl as ccl

from lfkit.cosmo.cosmology import cosmo_object, lookback_time_gyr


def test_cosmo_object_returns_instance_unchanged():
    """Tests that cosmo_object returns the provided instance unchanged when instance is given."""
    inst = ccl.CosmologyVanillaLCDM()
    out = cosmo_object(instance=inst)
    assert out is inst


def test_cosmo_object_raises_if_instance_and_params_given():
    """Tests that cosmo_object raises ValueError when both instance and params are provided."""
    inst = ccl.CosmologyVanillaLCDM()
    with pytest.raises(ValueError):
        cosmo_object(instance=inst, Omega_c=0.25)  # any param triggers the error


def test_cosmo_object_builds_from_params():
    """Tests that cosmo_object constructs a ccl.Cosmology when parameters are provided."""
    cosmo = cosmo_object(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
    )
    assert isinstance(cosmo, ccl.Cosmology)


def test_cosmo_object_defaults_to_vanilla_lcdm():
    """Tests that cosmo_object returns CosmologyVanillaLCDM when no instance or params are provided."""
    cosmo = cosmo_object()
    assert isinstance(cosmo, ccl.Cosmology)


def test_lookback_time_gyr_shape_and_dtype_scalar_and_array():
    """Tests that lookback_time_gyr returns float arrays with shapes matching scalar and vector z inputs."""
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
    """Tests that lookback_time_gyr is non-decreasing with increasing redshift."""
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    t = lookback_time_gyr(cosmo, z)
    assert np.all(t[1:] >= t[:-1])


def test_lookback_time_gyr_zero_at_z0_with_tolerance():
    """Tests that lookback_time_gyr at z=0 is approximately zero within numerical tolerance."""
    cosmo = ccl.CosmologyVanillaLCDM()
    t0 = float(lookback_time_gyr(cosmo, 0.0))
    assert abs(t0) < 1e-10
