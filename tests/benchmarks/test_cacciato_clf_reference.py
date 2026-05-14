"""Reference benchmarks for Cacciato-style conditional luminosity functions.

These tests compare LFKit's generic conditional luminosity function helpers
against explicit reference formulae from the Cacciato et al. conditional
luminosity function parametrization.

They are intentionally written as reference/benchmark tests rather than normal
unit tests. They check that LFKit preserves the expected central lognormal,
satellite modified-Schechter, and central-plus-satellite behaviour.
"""

import numpy as np
import pytest

from lfkit.photometry.conditional_lf_models import (
    lognormal_conditional_lf,
    modified_schechter_conditional_lf,
    two_component_conditional_lf,
)
from lfkit.photometry.luminosities import magnitude_difference_from_luminosity_ratio


pytestmark = pytest.mark.benchmark


def _central_magnitude_from_halo_mass(
    log_halo_mass: np.ndarray,
    *,
    log_m1: float,
    log_l0: float,
    gamma1: float,
    gamma2: float,
    m_reference: float = 0.0,
) -> np.ndarray:
    """Return central absolute magnitude from the Cacciato Lc(M) relation."""
    halo_mass = 10.0**log_halo_mass
    m1 = 10.0**log_m1

    luminosity_c = 10.0**log_l0 * (halo_mass / m1) ** gamma1
    luminosity_c /= (1.0 + halo_mass / m1) ** (gamma1 - gamma2)

    return m_reference - 2.5 * np.log10(luminosity_c)


def _satellite_phi_star_from_halo_mass(
    log_halo_mass: np.ndarray,
    *,
    b0: float,
    b1: float,
    b2: float,
) -> np.ndarray:
    """Return satellite normalization from the Cacciato quadratic relation."""
    log_m12 = log_halo_mass - 12.0

    return 10.0 ** (b0 + b1 * log_m12 + b2 * log_m12**2.0)


def _reference_central_lognormal_magnitude_clf(
    absolute_mag: np.ndarray,
    mean_absolute_mag: np.ndarray,
    sigma_log_luminosity: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return the reference central lognormal CLF in magnitude units."""
    delta_log_luminosity = -0.4 * (absolute_mag - mean_absolute_mag)

    return (
        amplitude
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity) ** 2.0)
    )


def _reference_satellite_modified_schechter_magnitude_clf(
    absolute_mag: np.ndarray,
    m_star: np.ndarray,
    phi_star: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the reference satellite modified-Schechter CLF in magnitudes."""
    luminosity_ratio = 10.0 ** (-0.4 * (absolute_mag - m_star))

    return (
        0.4
        * np.log(10.0)
        * phi_star
        * luminosity_ratio ** (alpha + 1.0)
        * np.exp(-(luminosity_ratio**2.0))
    )


def test_cacciato_central_lognormal_matches_reference_formula() -> None:
    """Tests central Cacciato-style CLF against the explicit formula."""

    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])
    log_halo_mass = np.array([11.5, 12.0, 12.5, 13.0])

    mean_absolute_mag = _central_magnitude_from_halo_mass(
        log_halo_mass,
        log_m1=11.24,
        log_l0=9.95,
        gamma1=3.18,
        gamma2=0.245,
    )

    result = lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=log_halo_mass,
        mean_absolute_mag=mean_absolute_mag,
        sigma_log_luminosity=0.157,
        amplitude=1.0,
    )

    expected = _reference_central_lognormal_magnitude_clf(
        absolute_mag=absolute_mag,
        mean_absolute_mag=mean_absolute_mag,
        sigma_log_luminosity=0.157,
        amplitude=1.0,
    )

    np.testing.assert_allclose(result, expected, rtol=1.0e-12, atol=0.0)


def test_cacciato_satellite_modified_schechter_matches_reference_formula() -> None:
    """Tests satellite Cacciato-style CLF against the explicit formula."""

    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])
    log_halo_mass = np.array([11.5, 12.0, 12.5, 13.0])

    central_mag = _central_magnitude_from_halo_mass(
        log_halo_mass,
        log_m1=11.24,
        log_l0=9.95,
        gamma1=3.18,
        gamma2=0.245,
    )
    satellite_m_star = central_mag + magnitude_difference_from_luminosity_ratio(0.562)

    phi_star = _satellite_phi_star_from_halo_mass(
        log_halo_mass,
        b0=-1.17,
        b1=1.53,
        b2=-0.217,
    )

    result = modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=log_halo_mass,
        phi_star=phi_star,
        m_star=satellite_m_star,
        alpha=-1.18,
    )

    expected = _reference_satellite_modified_schechter_magnitude_clf(
        absolute_mag=absolute_mag,
        m_star=satellite_m_star,
        phi_star=phi_star,
        alpha=-1.18,
    )

    np.testing.assert_allclose(result, expected, rtol=1.0e-12, atol=0.0)


def test_cacciato_central_satellite_matches_sum_of_components() -> None:
    """Tests that the Cacciato-style total CLF equals central plus satellite."""

    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])
    log_halo_mass = np.array([11.5, 12.0, 12.5, 13.0])

    central_mag = _central_magnitude_from_halo_mass(
        log_halo_mass,
        log_m1=11.24,
        log_l0=9.95,
        gamma1=3.18,
        gamma2=0.245,
    )
    phi_star = _satellite_phi_star_from_halo_mass(
        log_halo_mass,
        b0=-1.17,
        b1=1.53,
        b2=-0.217,
    )

    result = two_component_conditional_lf(
        absolute_mag=absolute_mag,
        condition=log_halo_mass,
        lognormal_mean_absolute_mag=central_mag,
        lognormal_sigma_log_luminosity=0.157,
        modified_phi_star=phi_star,
        modified_alpha=-1.18,
        lognormal_amplitude=1.0,
        modified_m_star=None,
        modified_luminosity_fraction=0.562,
    )

    satellite_m_star = central_mag + magnitude_difference_from_luminosity_ratio(0.562)

    central = _reference_central_lognormal_magnitude_clf(
        absolute_mag=absolute_mag,
        mean_absolute_mag=central_mag,
        sigma_log_luminosity=0.157,
        amplitude=1.0,
    )
    satellite = _reference_satellite_modified_schechter_magnitude_clf(
        absolute_mag=absolute_mag,
        m_star=satellite_m_star,
        phi_star=phi_star,
        alpha=-1.18,
    )

    expected = central + satellite

    np.testing.assert_allclose(result, expected, rtol=1.0e-12, atol=0.0)


def test_cacciato_satellite_m_star_is_fainter_than_central_for_fraction_below_one() -> None:
    """Tests the Cacciato convention Ls_star = 0.562 Lc in magnitude space."""

    central_mag = np.array([-22.0, -21.0, -20.0])

    satellite_m_star = central_mag + magnitude_difference_from_luminosity_ratio(0.562)

    assert np.all(satellite_m_star > central_mag)
    np.testing.assert_allclose(
        satellite_m_star - central_mag,
        magnitude_difference_from_luminosity_ratio(0.562),
    )


def test_cacciato_satellite_phi_star_peaks_near_reference_mass_range() -> None:
    """Tests that the quadratic satellite normalization remains positive."""

    log_halo_mass = np.linspace(11.0, 14.5, 32)

    phi_star = _satellite_phi_star_from_halo_mass(
        log_halo_mass,
        b0=-1.17,
        b1=1.53,
        b2=-0.217,
    )

    assert np.all(np.isfinite(phi_star))
    assert np.all(phi_star > 0.0)
