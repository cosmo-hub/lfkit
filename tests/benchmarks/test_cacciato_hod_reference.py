"""Reference benchmark for Cacciato-style HOD occupation integrals.

This benchmark compares the original luminosity-space Cacciato conditional
luminosity functions against LFKit's magnitude-space conditional luminosity
function helpers. The goal is to check that integrating the central and
satellite CLFs over a luminosity window gives the same halo occupation numbers
when expressed through LFKit.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.photometry.conditional_lf_models import (
    lognormal_conditional_lf,
    modified_schechter_conditional_lf,
    two_component_conditional_lf,
)
from lfkit.photometry.luminosities import (
    magnitude_difference_from_luminosity_ratio,
)


pytestmark = pytest.mark.benchmark


LOGE = 0.4342944819


def cacciato_lc(
    halo_mass: np.ndarray,
    *,
    log_l0: float = 9.95,
    log_m1: float = 11.24,
    gamma1: float = 3.18,
    gamma2: float = 0.245,
) -> np.ndarray:
    """Return the Cacciato central luminosity relation."""

    l0 = 10.0**log_l0
    m1 = 10.0**log_m1

    return l0 * (halo_mass / m1) ** gamma1 / (
        1.0 + (halo_mass / m1) ** (gamma1 - gamma2)
    )


def cacciato_phi_star_satellite(
    halo_mass: np.ndarray,
    *,
    b0: float = -1.17,
    b1: float = 1.53 * 0.739,
    b2: float = -0.217 * 0.739**2,
) -> np.ndarray:
    """Return the Cacciato satellite normalization."""

    log_m12 = np.log10(halo_mass / 1.0e12)

    return 10.0 ** (b0 + b1 * log_m12 + b2 * log_m12**2.0)


def cacciato_lstar_satellite(
    halo_mass: np.ndarray,
    *,
    luminosity_fraction: float = 0.562,
) -> np.ndarray:
    """Return the Cacciato satellite characteristic luminosity."""

    return luminosity_fraction * cacciato_lc(halo_mass)


def magnitude_from_luminosity_ratio(
    luminosity: np.ndarray,
) -> np.ndarray:
    """Return magnitude for luminosity relative to unit reference luminosity."""

    return magnitude_difference_from_luminosity_ratio(luminosity)


def central_luminosity_clf_reference(
    luminosity: np.ndarray,
    halo_mass: np.ndarray,
    *,
    sigma_c: float = 0.157,
) -> np.ndarray:
    """Return the Cacciato central CLF in luminosity units."""

    luminosity_grid = np.atleast_1d(luminosity).reshape(1, -1)
    halo_mass_grid = np.atleast_1d(halo_mass).reshape(-1, 1)

    lc = cacciato_lc(halo_mass_grid)

    return (
        LOGE
        / (np.sqrt(2.0 * np.pi) * sigma_c)
        * np.exp(
            -(
                np.log10(luminosity_grid)
                - np.log10(lc)
            )
            ** 2.0
            / (2.0 * sigma_c**2.0)
        )
        / luminosity_grid
    )


def satellite_luminosity_clf_reference(
    luminosity: np.ndarray,
    halo_mass: np.ndarray,
    *,
    alpha_s: float = -1.18,
) -> np.ndarray:
    """Return the Cacciato satellite CLF in luminosity units."""

    luminosity_grid = np.atleast_1d(luminosity).reshape(1, -1)
    halo_mass_grid = np.atleast_1d(halo_mass).reshape(-1, 1)

    l_star = cacciato_lstar_satellite(halo_mass_grid)
    phi_star = cacciato_phi_star_satellite(halo_mass_grid)

    return (
        phi_star
        / luminosity_grid
        * (luminosity_grid / l_star) ** (alpha_s + 1.0)
        * np.exp(-((luminosity_grid / l_star) ** 2.0))
    )


def test_lfkit_cacciato_central_occupation_matches_luminosity_reference() -> None:
    """Compare central occupation integrals from luminosity and magnitude CLFs."""

    halo_mass = np.geomspace(1.0e8, 1.0e16, 128)
    luminosity = np.geomspace(1.5e7, 5.6e9, 2048)

    reference_clf = central_luminosity_clf_reference(
        luminosity=luminosity,
        halo_mass=halo_mass,
    )
    reference_nc = np.trapezoid(reference_clf, x=luminosity, axis=1)

    magnitude = magnitude_from_luminosity_ratio(luminosity)
    magnitude_grid = magnitude[::-1]

    central_mag = magnitude_from_luminosity_ratio(cacciato_lc(halo_mass))

    lfkit_clf = lognormal_conditional_lf(
        absolute_mag=magnitude_grid.reshape(1, -1),
        condition=np.log10(halo_mass).reshape(-1, 1),
        mean_absolute_mag=central_mag.reshape(-1, 1),
        sigma_log_luminosity=0.157,
        amplitude=1.0,
    )
    lfkit_nc = np.trapezoid(lfkit_clf, x=magnitude_grid, axis=1)

    np.testing.assert_allclose(
        lfkit_nc,
        reference_nc,
        rtol=5.0e-4,
        atol=1.0e-8,
    )


def test_lfkit_cacciato_satellite_occupation_matches_luminosity_reference() -> None:
    """Compare satellite occupation integrals from luminosity and magnitude CLFs."""

    halo_mass = np.geomspace(1.0e8, 1.0e16, 128)
    luminosity = np.geomspace(1.5e7, 5.6e9, 2048)

    reference_clf = satellite_luminosity_clf_reference(
        luminosity=luminosity,
        halo_mass=halo_mass,
    )
    reference_ns = np.trapezoid(reference_clf, x=luminosity, axis=1)

    magnitude = magnitude_from_luminosity_ratio(luminosity)
    magnitude_grid = magnitude[::-1]

    central_mag = magnitude_from_luminosity_ratio(cacciato_lc(halo_mass))
    satellite_m_star = central_mag + magnitude_difference_from_luminosity_ratio(0.562)
    phi_star = cacciato_phi_star_satellite(halo_mass)

    lfkit_clf = modified_schechter_conditional_lf(
        absolute_mag=magnitude_grid.reshape(1, -1),
        condition=np.log10(halo_mass).reshape(-1, 1),
        phi_star=phi_star.reshape(-1, 1),
        m_star=satellite_m_star.reshape(-1, 1),
        alpha=-1.18,
    )
    lfkit_ns = np.trapezoid(lfkit_clf, x=magnitude_grid, axis=1)

    np.testing.assert_allclose(
        lfkit_ns,
        reference_ns,
        rtol=5.0e-4,
        atol=1.0e-8,
    )


def test_lfkit_cacciato_total_occupation_matches_luminosity_reference() -> None:
    """Compare total occupation integrals from luminosity and magnitude CLFs."""

    halo_mass = np.geomspace(1.0e8, 1.0e16, 128)
    luminosity = np.geomspace(1.5e7, 5.6e9, 2048)

    reference_central = central_luminosity_clf_reference(
        luminosity=luminosity,
        halo_mass=halo_mass,
    )
    reference_satellite = satellite_luminosity_clf_reference(
        luminosity=luminosity,
        halo_mass=halo_mass,
    )
    reference_total = np.trapezoid(
        reference_central + reference_satellite,
        x=luminosity,
        axis=1,
    )

    magnitude = magnitude_from_luminosity_ratio(luminosity)
    magnitude_grid = magnitude[::-1]

    central_mag = magnitude_from_luminosity_ratio(cacciato_lc(halo_mass))
    phi_star = cacciato_phi_star_satellite(halo_mass)

    lfkit_clf = two_component_conditional_lf(
        absolute_mag=magnitude_grid.reshape(1, -1),
        condition=np.log10(halo_mass).reshape(-1, 1),
        lognormal_mean_absolute_mag=central_mag.reshape(-1, 1),
        lognormal_sigma_log_luminosity=0.157,
        modified_phi_star=phi_star.reshape(-1, 1),
        modified_alpha=-1.18,
        lognormal_amplitude=1.0,
        modified_m_star=None,
        modified_luminosity_fraction=0.562,
    )
    lfkit_total = np.trapezoid(lfkit_clf, x=magnitude_grid, axis=1)

    np.testing.assert_allclose(
        lfkit_total,
        reference_total,
        rtol=5.0e-4,
        atol=1.0e-8,
    )
