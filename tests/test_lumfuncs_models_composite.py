"""Unit tests for ``lfkit.luminosity_functions.models.composite``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.composite import (
    additive_lf,
    two_component_lf,
)
from lfkit.luminosity_functions.models.gaussian import lognormal_lf
from lfkit.luminosity_functions.models.modifiers import apply_luminosity_cutoff
from lfkit.luminosity_functions.models.schechter import schechter
from lfkit.photometry.luminosities import magnitude_difference_from_luminosity_ratio


def test_additive_lf_sums_single_component() -> None:
    """Tests that additive_lf returns a single component unchanged."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])

    result = additive_lf(
        absolute_mag,
        lambda mag: np.ones_like(mag),
    )

    np.testing.assert_allclose(result, np.ones_like(absolute_mag))


def test_additive_lf_sums_multiple_components() -> None:
    """Tests that additive_lf sums several component functions."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])

    result = additive_lf(
        absolute_mag,
        lambda mag: np.ones_like(mag),
        lambda mag: 2.0 * np.ones_like(mag),
        lambda mag: mag + 25.0,
    )

    expected = 3.0 + absolute_mag + 25.0
    np.testing.assert_allclose(result, expected)


def test_additive_lf_accepts_scalar_input() -> None:
    """Tests that additive_lf accepts scalar magnitude input."""
    result = additive_lf(
        -21.0,
        lambda mag: np.ones_like(mag),
        lambda mag: 2.0 * np.ones_like(mag),
    )

    assert np.shape(result) == ()
    np.testing.assert_allclose(result, np.array(3.0))


def test_additive_lf_preserves_array_shape() -> None:
    """Tests that additive_lf preserves the input magnitude shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = additive_lf(
        absolute_mag,
        lambda mag: np.ones_like(mag),
        lambda mag: mag + 25.0,
    )

    assert result.shape == absolute_mag.shape


def test_additive_lf_returns_float_array() -> None:
    """Tests that additive_lf returns a floating-point array."""
    result = additive_lf(
        np.array([-22, -21, -20]),
        lambda mag: np.ones_like(mag, dtype=int),
    )

    assert result.dtype.kind == "f"


def test_additive_lf_rejects_missing_components() -> None:
    """Tests that additive_lf requires at least one component."""
    with pytest.raises(
        ValueError,
        match="At least one luminosity function component is required",
    ):
        additive_lf(np.array([-22.0, -21.0]))


def test_additive_lf_rejects_nonfinite_magnitudes() -> None:
    """Tests that additive_lf rejects non-finite magnitude values."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        additive_lf(
            np.array([-22.0, np.nan]),
            lambda mag: np.ones_like(mag),
        )


def test_additive_lf_supports_broadcastable_scalar_component() -> None:
    """Tests that additive_lf supports scalar component outputs."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])

    result = additive_lf(
        absolute_mag,
        lambda mag: 1.0,
        lambda mag: np.ones_like(mag),
    )

    np.testing.assert_allclose(result, 2.0 * np.ones_like(absolute_mag))


def test_two_component_lf_matches_manual_component_sum_with_explicit_m_star() -> None:
    """Tests that two_component_lf matches explicit lognormal plus modified Schechter sum."""
    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])

    params = {
        "lognormal_mean_absolute_mag": -21.5,
        "lognormal_sigma_log_luminosity": 0.25,
        "lognormal_amplitude": 1.7,
        "modified_phi_star": 0.004,
        "modified_m_star": -20.8,
        "modified_alpha": -1.1,
    }

    result = two_component_lf(absolute_mag, **params)

    expected_lognormal = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=params["lognormal_mean_absolute_mag"],
        sigma_log_luminosity=params["lognormal_sigma_log_luminosity"],
        amplitude=params["lognormal_amplitude"],
    )
    expected_modified = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=schechter,
        phi_star=params["modified_phi_star"],
        m_star=params["modified_m_star"],
        alpha=params["modified_alpha"],
    )
    expected = expected_lognormal + expected_modified

    np.testing.assert_allclose(result, expected)


def test_two_component_lf_infers_modified_m_star_from_luminosity_fraction() -> None:
    """Tests that two_component_lf infers modified_m_star from the luminosity fraction."""
    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])
    mean_mag = -21.5
    luminosity_fraction = 0.562

    result = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=mean_mag,
        lognormal_sigma_log_luminosity=0.25,
        lognormal_amplitude=1.7,
        modified_phi_star=0.004,
        modified_alpha=-1.1,
        modified_luminosity_fraction=luminosity_fraction,
    )

    inferred_m_star = mean_mag + magnitude_difference_from_luminosity_ratio(
        luminosity_fraction,
    )
    expected = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=mean_mag,
        sigma_log_luminosity=0.25,
        amplitude=1.7,
    ) + apply_luminosity_cutoff(
        absolute_mag,
        base_lf=schechter,
        phi_star=0.004,
        m_star=inferred_m_star,
        alpha=-1.1,
    )

    np.testing.assert_allclose(result, expected)


def test_two_component_lf_accepts_scalar_magnitude_input() -> None:
    """Tests that two_component_lf accepts scalar absolute magnitude input."""
    result = two_component_lf(
        -21.0,
        lognormal_mean_absolute_mag=-21.5,
        lognormal_sigma_log_luminosity=0.25,
        modified_phi_star=0.004,
        modified_alpha=-1.1,
    )

    assert np.shape(result) == ()
    assert np.isfinite(result)
    assert result >= 0.0


def test_two_component_lf_preserves_array_shape() -> None:
    """Tests that two_component_lf preserves the absolute magnitude shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=-21.5,
        lognormal_sigma_log_luminosity=0.25,
        modified_phi_star=0.004,
        modified_alpha=-1.1,
    )

    assert result.shape == absolute_mag.shape


def test_two_component_lf_rejects_nonfinite_absolute_magnitude() -> None:
    """Tests that two_component_lf rejects non-finite absolute magnitudes."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        two_component_lf(
            np.array([-22.0, np.nan]),
            lognormal_mean_absolute_mag=-21.5,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_alpha=-1.1,
        )


def test_two_component_lf_rejects_nonfinite_lognormal_mean() -> None:
    """Tests that two_component_lf rejects non-finite lognormal mean magnitudes."""
    with pytest.raises(
        ValueError,
        match="lognormal_mean_absolute_mag contains NaN or infinite values",
    ):
        two_component_lf(
            np.array([-22.0, -21.0]),
            lognormal_mean_absolute_mag=np.nan,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_alpha=-1.1,
        )


def test_two_component_lf_rejects_zero_luminosity_fraction() -> None:
    """Tests that zero modified_luminosity_fraction is rejected."""
    with pytest.raises(ValueError, match="modified_luminosity_fraction must be positive"):
        two_component_lf(
            np.array([-22.0, -21.0]),
            lognormal_mean_absolute_mag=-21.5,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_alpha=-1.1,
            modified_luminosity_fraction=0.0,
        )


def test_two_component_lf_rejects_negative_luminosity_fraction() -> None:
    """Tests that negative modified_luminosity_fraction is rejected."""
    with pytest.raises(ValueError, match="modified_luminosity_fraction must be positive"):
        two_component_lf(
            np.array([-22.0, -21.0]),
            lognormal_mean_absolute_mag=-21.5,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_alpha=-1.1,
            modified_luminosity_fraction=-0.5,
        )


def test_two_component_lf_rejects_nonfinite_luminosity_fraction() -> None:
    """Tests that non-finite modified_luminosity_fraction is rejected."""
    with pytest.raises(
        ValueError,
        match="modified_luminosity_fraction contains NaN or infinite values",
    ):
        two_component_lf(
            np.array([-22.0, -21.0]),
            lognormal_mean_absolute_mag=-21.5,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_alpha=-1.1,
            modified_luminosity_fraction=np.nan,
        )


def test_two_component_lf_rejects_nonfinite_explicit_m_star() -> None:
    """Tests that non-finite explicit modified_m_star values are rejected."""
    with pytest.raises(
        ValueError,
        match="modified_m_star contains NaN or infinite values",
    ):
        two_component_lf(
            np.array([-22.0, -21.0]),
            lognormal_mean_absolute_mag=-21.5,
            lognormal_sigma_log_luminosity=0.25,
            modified_phi_star=0.004,
            modified_m_star=np.inf,
            modified_alpha=-1.1,
        )


def test_two_component_lf_explicit_m_star_overrides_bad_luminosity_fraction() -> None:
    """Tests that modified_luminosity_fraction is ignored when modified_m_star is explicit."""
    result = two_component_lf(
        np.array([-22.0, -21.0]),
        lognormal_mean_absolute_mag=-21.5,
        lognormal_sigma_log_luminosity=0.25,
        modified_phi_star=0.004,
        modified_m_star=-20.8,
        modified_alpha=-1.1,
        modified_luminosity_fraction=np.nan,
    )

    assert np.all(np.isfinite(result))


def test_two_component_lf_allows_array_lognormal_mean() -> None:
    """Tests that array-valued lognormal means broadcast through the model."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    mean_mag = np.array([-21.5, -21.0, -20.5])

    result = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=mean_mag,
        lognormal_sigma_log_luminosity=0.25,
        modified_phi_star=0.004,
        modified_alpha=-1.1,
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


def test_two_component_lf_allows_array_luminosity_fraction() -> None:
    """Tests that array-valued luminosity fractions broadcast through the model."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    luminosity_fraction = np.array([0.5, 0.6, 0.7])

    result = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=-21.5,
        lognormal_sigma_log_luminosity=0.25,
        modified_phi_star=0.004,
        modified_alpha=-1.1,
        modified_luminosity_fraction=luminosity_fraction,
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


def test_two_component_lf_zero_lognormal_amplitude_matches_modified_component() -> None:
    """Tests that zero lognormal amplitude leaves only the modified Schechter component."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    modified_m_star = -20.8

    result = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=-21.5,
        lognormal_sigma_log_luminosity=0.25,
        lognormal_amplitude=0.0,
        modified_phi_star=0.004,
        modified_m_star=modified_m_star,
        modified_alpha=-1.1,
    )

    expected = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=schechter,
        phi_star=0.004,
        m_star=modified_m_star,
        alpha=-1.1,
    )

    np.testing.assert_allclose(result, expected)
