"""Unit tests for ``lfkit.photometry.conditional_lf_models.py``."""

import numpy as np
import pytest

from lfkit.photometry.conditional_lf_models import (
    central_lognormal_conditional_lf,
    central_satellite_conditional_lf,
    conditional_schechter,
    conditional_schechter_double,
    conditional_schechter_evolving,
    satellite_modified_schechter_conditional_lf,
)
from lfkit.photometry.luminosities import (
    luminosity_ratio,
    magnitude_difference_from_luminosity_ratio,
)
from lfkit.photometry.luminosity_function import schechter, schechter_double


def test_conditional_schechter_matches_schechter_for_scalar_parameters() -> None:
    """Tests that the conditional Schechter wrapper matches Schechter."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.1, 0.2, 0.3])

    result = conditional_schechter(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.1,
    )

    expected = schechter(
        absolute_mag,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.1,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_schechter_accepts_callable_parameters() -> None:
    """Tests that conditional Schechter parameters can be callables."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_schechter(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=lambda x: 1.0e-3 * (1.0 + x),
        m_star=lambda x: -21.0 - 0.1 * x,
        alpha=lambda x: -1.0 - 0.05 * x,
    )

    expected = schechter(
        absolute_mag,
        phi_star=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        m_star=np.array([-21.0, -21.1, -21.2]),
        alpha=np.array([-1.0, -1.05, -1.1]),
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_schechter_rejects_non_finite_condition() -> None:
    """Tests that non-finite condition values are rejected."""

    with pytest.raises(ValueError, match="condition contains NaN or infinite values."):
        conditional_schechter(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, np.nan, 1.0],
            phi_star=1.0e-3,
            m_star=-21.0,
            alpha=-1.1,
        )


def test_conditional_schechter_rejects_non_finite_callable_parameter() -> None:
    """Tests that non-finite callable parameter values are rejected."""

    with pytest.raises(ValueError, match="phi_star contains NaN or infinite values."):
        conditional_schechter(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            phi_star=lambda x: np.array([1.0e-3, np.nan, 2.0e-3]),
            m_star=-21.0,
            alpha=-1.1,
        )


def test_conditional_schechter_evolving_matches_explicit_parameter_models() -> None:
    """Tests the conditional evolving Schechter wrapper with simple models."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_schechter_evolving(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_model="constant",
        phi_kwargs={"phi_star": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -21.0},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.1},
    )

    expected = schechter(
        absolute_mag,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.1,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_schechter_evolving_rejects_unknown_model() -> None:
    """Tests that unknown LF parameter models are rejected."""

    with pytest.raises(ValueError):
        conditional_schechter_evolving(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            phi_model="not_a_model",
            phi_kwargs={"value": 1.0e-3},
            m_star_model="constant",
            m_star_kwargs={"value": -21.0},
            alpha_model="constant",
            alpha_kwargs={"value": -1.1},
        )


def test_conditional_schechter_double_matches_double_schechter() -> None:
    """Tests that the conditional double-Schechter wrapper matches the model."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_schechter_double(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.0,
        beta=-0.5,
        m_transition=-19.5,
    )

    expected = schechter_double(
        absolute_mag,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.0,
        beta=-0.5,
        m_transition=-19.5,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_schechter_double_accepts_callable_parameters() -> None:
    """Tests callable parameters for the conditional double-Schechter model."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_schechter_double(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=lambda x: 1.0e-3 * (1.0 + x),
        m_star=lambda x: -21.0 - 0.1 * x,
        alpha=-1.0,
        beta=-0.5,
        m_transition=lambda x: -19.5 - 0.2 * x,
    )

    expected = schechter_double(
        absolute_mag,
        phi_star=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        m_star=np.array([-21.0, -21.1, -21.2]),
        alpha=-1.0,
        beta=-0.5,
        m_transition=np.array([-19.5, -19.7, -19.9]),
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_central_lognormal_conditional_lf_matches_expected_formula() -> None:
    """Tests the central lognormal conditional LF formula."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])
    mean_absolute_mag = np.array([-21.0, -21.0, -21.0])
    sigma_log_luminosity = np.array([0.2, 0.2, 0.2])
    amplitude = np.array([1.0, 2.0, 3.0])

    result = central_lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        mean_absolute_mag=mean_absolute_mag,
        sigma_log_luminosity=sigma_log_luminosity,
        amplitude=amplitude,
    )

    delta_log_luminosity = -0.4 * (absolute_mag - mean_absolute_mag)
    expected = (
        amplitude
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity) ** 2.0)
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_central_lognormal_conditional_lf_accepts_callable_parameters() -> None:
    """Tests callable parameters for the central lognormal conditional LF."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = central_lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        mean_absolute_mag=lambda x: -21.0 - 0.1 * x,
        sigma_log_luminosity=lambda x: 0.2 + 0.01 * x,
        amplitude=lambda x: 1.0 + x,
    )

    mean_absolute_mag = np.array([-21.0, -21.1, -21.2])
    sigma_log_luminosity = np.array([0.2, 0.21, 0.22])
    amplitude = np.array([1.0, 2.0, 3.0])

    delta_log_luminosity = -0.4 * (absolute_mag - mean_absolute_mag)
    expected = (
        amplitude
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity) ** 2.0)
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_central_lognormal_conditional_lf_rejects_zero_sigma() -> None:
    """Tests that zero central scatter is rejected."""

    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive."):
        central_lognormal_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.0,
            amplitude=1.0,
        )


def test_central_lognormal_conditional_lf_rejects_negative_sigma() -> None:
    """Tests that negative central scatter is rejected."""

    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive."):
        central_lognormal_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=-0.2,
            amplitude=1.0,
        )


def test_central_lognormal_conditional_lf_rejects_negative_amplitude() -> None:
    """Tests that negative central amplitude is rejected."""

    with pytest.raises(ValueError, match="amplitude must be non-negative."):
        central_lognormal_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.2,
            amplitude=-1.0,
        )


def test_satellite_modified_schechter_conditional_lf_matches_expected_formula() -> None:
    """Tests the satellite modified-Schechter conditional LF formula."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])
    phi_star = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    m_star = np.array([-21.0, -21.1, -21.2])
    alpha = np.array([-1.0, -1.1, -1.2])

    result = satellite_modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    expected = 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-(x**2.0))

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_satellite_modified_schechter_conditional_lf_accepts_callable_parameters() -> None:
    """Tests callable parameters for the satellite modified-Schechter LF."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = satellite_modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=lambda x: 1.0e-3 * (1.0 + x),
        m_star=lambda x: -21.0 - 0.1 * x,
        alpha=lambda x: -1.0 - 0.05 * x,
    )

    phi_star = np.array([1.0e-3, 2.0e-3, 3.0e-3])
    m_star = np.array([-21.0, -21.1, -21.2])
    alpha = np.array([-1.0, -1.05, -1.1])

    x = luminosity_ratio(absolute_mag, m_star)
    expected = 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-(x**2.0))

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_satellite_modified_schechter_conditional_lf_rejects_negative_phi_star() -> None:
    """Tests that negative satellite normalization is rejected."""

    with pytest.raises(ValueError, match="phi_star must be non-negative."):
        satellite_modified_schechter_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            phi_star=-1.0e-3,
            m_star=-21.0,
            alpha=-1.1,
        )


def test_central_satellite_conditional_lf_equals_sum_with_explicit_satellite_m_star() -> None:
    """Tests that the total conditional LF equals central plus satellite parts."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = central_satellite_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        central_mean_absolute_mag=-21.0,
        central_sigma_log_luminosity=0.2,
        satellite_phi_star=1.0e-3,
        satellite_alpha=-1.1,
        central_amplitude=1.0,
        satellite_m_star=-20.5,
    )

    central = central_lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.2,
        amplitude=1.0,
    )
    satellite = satellite_modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )
    expected = central + satellite

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_central_satellite_conditional_lf_derives_satellite_m_star() -> None:
    """Tests that satellite M-star is derived from the luminosity fraction."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])
    central_mean_absolute_mag = np.array([-21.0, -21.1, -21.2])
    satellite_luminosity_fraction = np.array([0.5, 0.6, 0.7])

    result = central_satellite_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        central_mean_absolute_mag=central_mean_absolute_mag,
        central_sigma_log_luminosity=0.2,
        satellite_phi_star=1.0e-3,
        satellite_alpha=-1.1,
        central_amplitude=1.0,
        satellite_m_star=None,
        satellite_luminosity_fraction=satellite_luminosity_fraction,
    )

    satellite_m_star = central_mean_absolute_mag + (
        magnitude_difference_from_luminosity_ratio(satellite_luminosity_fraction)
    )

    central = central_lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        mean_absolute_mag=central_mean_absolute_mag,
        sigma_log_luminosity=0.2,
        amplitude=1.0,
    )
    satellite = satellite_modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=1.0e-3,
        m_star=satellite_m_star,
        alpha=-1.1,
    )
    expected = central + satellite

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_central_satellite_conditional_lf_rejects_zero_luminosity_fraction() -> None:
    """Tests that zero satellite luminosity fraction is rejected."""

    with pytest.raises(
        ValueError,
        match="satellite_luminosity_fraction must be positive.",
    ):
        central_satellite_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            central_mean_absolute_mag=-21.0,
            central_sigma_log_luminosity=0.2,
            satellite_phi_star=1.0e-3,
            satellite_alpha=-1.1,
            central_amplitude=1.0,
            satellite_m_star=None,
            satellite_luminosity_fraction=0.0,
        )


def test_central_satellite_conditional_lf_rejects_negative_luminosity_fraction() -> None:
    """Tests that negative satellite luminosity fraction is rejected."""

    with pytest.raises(
        ValueError,
        match="satellite_luminosity_fraction must be positive.",
    ):
        central_satellite_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            central_mean_absolute_mag=-21.0,
            central_sigma_log_luminosity=0.2,
            satellite_phi_star=1.0e-3,
            satellite_alpha=-1.1,
            central_amplitude=1.0,
            satellite_m_star=None,
            satellite_luminosity_fraction=-0.5,
        )


def test_central_satellite_conditional_lf_propagates_invalid_central_component() -> None:
    """Tests that invalid central-component parameters are propagated."""

    with pytest.raises(ValueError, match="central_sigma_log_luminosity must be positive.|sigma_log_luminosity must be positive."):
        central_satellite_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            central_mean_absolute_mag=-21.0,
            central_sigma_log_luminosity=0.0,
            satellite_phi_star=1.0e-3,
            satellite_alpha=-1.1,
            central_amplitude=1.0,
            satellite_m_star=-20.5,
        )


def test_central_satellite_conditional_lf_propagates_invalid_satellite_component() -> None:
    """Tests that invalid satellite-component parameters are propagated."""

    with pytest.raises(ValueError, match="phi_star must be non-negative."):
        central_satellite_conditional_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            central_mean_absolute_mag=-21.0,
            central_sigma_log_luminosity=0.2,
            satellite_phi_star=-1.0e-3,
            satellite_alpha=-1.1,
            central_amplitude=1.0,
            satellite_m_star=-20.5,
        )
