"""Tests for LF completeness utilities using the packaged fake catalog."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pyccl as ccl

from lfkit.luminosity_functions.completeness import (
    catalog_fraction,
    missing_number_density,
    observed_number_density,
    out_of_catalog_fraction,
)


def toy_luminosity_function(
    absolute_mag: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Return a smooth non-negative toy luminosity function."""
    phi0 = 1.0e-3
    m_star = -20.5 - 0.3 * z
    width = 1.2

    return phi0 * np.exp(-0.5 * ((absolute_mag - m_star) / width) ** 2)


def make_cosmology() -> ccl.Cosmology:
    """Return a small test cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    )


def load_fake_catalog() -> np.ndarray:
    """Return the packaged fake magnitude-limited catalog."""
    path = (
        files("lfkit")
        / "data"
        / "demo_catalogs"
        / "fake_magnitude_limited_catalog.csv"
    )
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def test_fake_catalog_is_packaged() -> None:
    """Tests that the fake catalog is available as packaged data."""
    catalog = load_fake_catalog()

    assert len(catalog) == 200
    assert set(catalog.dtype.names) == {
        "galaxy_id",
        "ra_deg",
        "dec_deg",
        "z",
        "m_app",
        "band",
    }


def test_completeness_fraction_on_fake_catalog_redshifts() -> None:
    """Tests that completeness fractions are valid on fake catalog redshifts."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    completeness = catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=256,
    )

    assert completeness.shape == z.shape
    assert np.all(np.isfinite(completeness))
    assert np.all(completeness >= 0.0)
    assert np.all(completeness <= 1.0)


def test_observed_and_missing_fractions_sum_to_one() -> None:
    """Tests that observed and out-of-catalog fractions are complementary."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    completeness = catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=256,
    )
    missing_fraction = out_of_catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=256,
    )

    np.testing.assert_allclose(
        completeness + missing_fraction,
        np.ones_like(z),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_deeper_catalog_limit_increases_completeness() -> None:
    """Tests that a fainter apparent magnitude limit increases completeness."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    shallow = catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=22.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )
    deep = catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )

    np.testing.assert_array_less(shallow, deep + 1.0e-8)
    assert np.mean(deep) > np.mean(shallow)


def test_deeper_catalog_limit_decreases_missing_density() -> None:
    """Tests that a fainter apparent magnitude limit misses no more galaxies."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    shallow = missing_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=22.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )
    deep = missing_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )

    np.testing.assert_array_less(deep, shallow + 1.0e-8)


def test_deeper_catalog_limit_increases_observed_density() -> None:
    """Tests that a fainter apparent magnitude limit observes no fewer galaxies."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    shallow = observed_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=22.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )
    deep = observed_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )

    np.testing.assert_array_less(shallow, deep + 1.0e-8)
    assert np.mean(deep) > np.mean(shallow)


def test_observed_and_missing_number_densities_sum_to_total_density() -> None:
    """Tests that observed and missing densities recover the total LF density."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()

    z = np.asarray(catalog["z"], dtype=float)

    total = observed_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=1.0e6,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )
    observed = observed_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )
    missing = missing_number_density(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=1024,
    )

    np.testing.assert_allclose(
        observed + missing,
        total,
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test_fake_catalog_redshifts_are_physical() -> None:
    """Tests that fake catalog redshifts are finite and non-negative."""
    catalog = load_fake_catalog()
    z = np.asarray(catalog["z"], dtype=float)

    assert np.all(np.isfinite(z))
    assert np.all(z >= 0.0)


def test_fake_catalog_apparent_magnitudes_are_finite() -> None:
    """Tests that fake catalog apparent magnitudes are finite."""
    catalog = load_fake_catalog()
    m_app = np.asarray(catalog["m_app"], dtype=float)

    assert np.all(np.isfinite(m_app))


def test_catalog_fraction_accepts_k_and_e_corrections() -> None:
    """Tests fake-catalog completeness with k/e corrections."""
    catalog = load_fake_catalog()
    cosmo = make_cosmology()
    z = np.asarray(catalog["z"], dtype=float)

    completeness = catalog_fraction(
        cosmo,
        z,
        toy_luminosity_function,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-14.0,
        n_m=256,
        h=0.7,
        k_correction=0.1 * z,
        e_correction=0.05 * z,
    )

    assert completeness.shape == z.shape
    assert np.all(np.isfinite(completeness))
    assert np.all((completeness >= 0.0) & (completeness <= 1.0))


