"""Unit tests for ``lfkit.photometry.conditional_lf_models``."""

import numpy as np
import pytest


from lfkit.luminosity_functions.conditional_models import (
    __all__,
    conditionalize_lf_model,
)
from lfkit.luminosity_functions.models.composite import two_component_lf
from lfkit.luminosity_functions.models.schechter import double_schechter, schechter
from lfkit.luminosity_functions.registry import get_conditional_lf_model


conditional_schechter = get_conditional_lf_model("schechter").function
conditional_double_schechter = get_conditional_lf_model("double_schechter").function
conditional_lognormal_lf = get_conditional_lf_model("lognormal").function
conditional_two_component_lf = get_conditional_lf_model("two_component").function


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


def test_conditional_double_schechter_matches_double_schechter() -> None:
    """Tests that the conditional double-Schechter wrapper matches the model."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_double_schechter(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.0,
        beta=-0.5,
        m_transition=-19.5,
    )

    expected = double_schechter(
        absolute_mag,
        phi_star=1.0e-3,
        m_star=-21.0,
        alpha=-1.0,
        beta=-0.5,
        m_transition=-19.5,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_double_schechter_accepts_callable_parameters() -> None:
    """Tests callable parameters for the conditional double-Schechter model."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_double_schechter(
        absolute_mag=absolute_mag,
        condition=condition,
        phi_star=lambda x: 1.0e-3 * (1.0 + x),
        m_star=lambda x: -21.0 - 0.1 * x,
        alpha=-1.0,
        beta=-0.5,
        m_transition=lambda x: -19.5 - 0.2 * x,
    )

    expected = double_schechter(
        absolute_mag,
        phi_star=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        m_star=np.array([-21.0, -21.1, -21.2]),
        alpha=-1.0,
        beta=-0.5,
        m_transition=np.array([-19.5, -19.7, -19.9]),
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_lognormal_lf_matches_expected_formula() -> None:
    """Tests the lognormal conditional LF formula."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])
    mean_absolute_mag = np.array([-21.0, -21.0, -21.0])
    sigma_log_luminosity = np.array([0.2, 0.2, 0.2])
    amplitude = np.array([1.0, 2.0, 3.0])

    result = conditional_lognormal_lf(
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


def test_conditional_lognormal_lf_accepts_callable_parameters() -> None:
    """Tests callable parameters for the lognormal conditional LF."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_lognormal_lf(
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


def test_conditional_lognormal_lf_rejects_zero_sigma() -> None:
    """Tests that zero lognormal scatter is rejected."""

    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive."):
        conditional_lognormal_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.0,
            amplitude=1.0,
        )


def test_conditional_lognormal_lf_rejects_negative_sigma() -> None:
    """Tests that negative lognormal scatter is rejected."""

    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive."):
        conditional_lognormal_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=-0.2,
            amplitude=1.0,
        )


def test_conditional_lognormal_lf_rejects_negative_amplitude() -> None:
    """Tests that negative lognormal amplitude is rejected."""

    with pytest.raises(ValueError, match="amplitude must be non-negative."):
        conditional_lognormal_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.2,
            amplitude=-1.0,
        )


def test_conditional_two_component_lf_equals_sum_with_explicit_modified_m_star() -> None:
    """Tests that the two-component LF equals lognormal plus modified parts."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])

    result = conditional_two_component_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        lognormal_mean_absolute_mag=-21.0,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
        lognormal_amplitude=1.0,
        modified_m_star=-20.5,
    )

    expected = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=-21.0,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
        lognormal_amplitude=1.0,
        modified_m_star=-20.5,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_two_component_lf_derives_modified_m_star() -> None:
    """Tests that modified M-star is derived from the luminosity fraction."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.0, 1.0, 2.0])
    lognormal_mean_absolute_mag = np.array([-21.0, -21.1, -21.2])
    modified_luminosity_fraction = np.array([0.5, 0.6, 0.7])

    result = conditional_two_component_lf(
        absolute_mag=absolute_mag,
        condition=condition,
        lognormal_mean_absolute_mag=lognormal_mean_absolute_mag,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
        lognormal_amplitude=1.0,
        modified_m_star=None,
        modified_luminosity_fraction=modified_luminosity_fraction,
    )

    expected = two_component_lf(
        absolute_mag,
        lognormal_mean_absolute_mag=lognormal_mean_absolute_mag,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
        lognormal_amplitude=1.0,
        modified_m_star=None,
        modified_luminosity_fraction=modified_luminosity_fraction,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_conditional_two_component_lf_rejects_zero_luminosity_fraction() -> None:
    """Tests that zero modified luminosity fraction is rejected."""

    with pytest.raises(
        ValueError,
        match="modified_luminosity_fraction must be positive.",
    ):
        conditional_two_component_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            lognormal_mean_absolute_mag=-21.0,
            lognormal_sigma_log_luminosity=0.2,
            modified_phi_star=1.0e-3,
            modified_alpha=-1.1,
            lognormal_amplitude=1.0,
            modified_m_star=None,
            modified_luminosity_fraction=0.0,
        )


def test_conditional_two_component_lf_rejects_negative_luminosity_fraction() -> None:
    """Tests that negative modified luminosity fraction is rejected."""

    with pytest.raises(
        ValueError,
        match="modified_luminosity_fraction must be positive.",
    ):
        conditional_two_component_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            lognormal_mean_absolute_mag=-21.0,
            lognormal_sigma_log_luminosity=0.2,
            modified_phi_star=1.0e-3,
            modified_alpha=-1.1,
            lognormal_amplitude=1.0,
            modified_m_star=None,
            modified_luminosity_fraction=-0.5,
        )


def test_conditional_two_component_lf_propagates_invalid_lognormal_component() -> None:
    """Tests that invalid lognormal-component parameters are propagated."""

    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive."):
        conditional_two_component_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            lognormal_mean_absolute_mag=-21.0,
            lognormal_sigma_log_luminosity=0.0,
            modified_phi_star=1.0e-3,
            modified_alpha=-1.1,
            lognormal_amplitude=1.0,
            modified_m_star=-20.5,
        )


def test_conditional_two_component_lf_propagates_invalid_modified_component() -> None:
    """Tests that invalid modified-component parameters are propagated."""

    with pytest.raises(ValueError, match="phi_star must be non-negative."):
        conditional_two_component_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            lognormal_mean_absolute_mag=-21.0,
            lognormal_sigma_log_luminosity=0.2,
            modified_phi_star=-1.0e-3,
            modified_alpha=-1.1,
            lognormal_amplitude=1.0,
            modified_m_star=-20.5,
        )


def test_conditionalize_lf_model_preserves_wrapped_model_name() -> None:
    """Tests that conditional wrappers preserve model metadata."""

    def toy_lf(absolute_mag, amplitude):
        return amplitude * np.ones_like(absolute_mag, dtype=float)

    conditional_toy_lf = conditionalize_lf_model(toy_lf)

    assert conditional_toy_lf.__name__ == "toy_lf"


def test_conditionalize_lf_model_passes_non_callable_kwargs_unchanged() -> None:
    """Tests that non-callable keyword arguments pass through unchanged."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])

    def toy_lf(absolute_mag, amplitude, offset):
        return amplitude * np.ones_like(absolute_mag, dtype=float) + offset

    conditional_toy_lf = conditionalize_lf_model(toy_lf)

    result = conditional_toy_lf(
        absolute_mag=absolute_mag,
        condition=np.array([0.0, 1.0, 2.0]),
        amplitude=2.0,
        offset=3.0,
    )

    np.testing.assert_allclose(result, np.array([5.0, 5.0, 5.0]))
    assert result.dtype == np.float64


def test_conditionalize_lf_model_rejects_negative_wrapped_output() -> None:
    """Tests that negative wrapped LF outputs are rejected."""

    def toy_lf(absolute_mag, amplitude):
        return np.array([1.0, -1.0, 2.0])

    conditional_toy_lf = conditionalize_lf_model(toy_lf)

    with pytest.raises(
        ValueError,
        match="toy_lf returned negative values, which are not allowed.",
    ):
        conditional_toy_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            amplitude=1.0,
        )


def test_conditionalize_lf_model_rejects_non_finite_wrapped_output() -> None:
    """Tests that non-finite wrapped LF outputs are rejected."""

    def toy_lf(absolute_mag, amplitude):
        return np.array([1.0, np.inf, 2.0])

    conditional_toy_lf = conditionalize_lf_model(toy_lf)

    with pytest.raises(ValueError, match="toy_lf contains NaN or infinite values."):
        conditional_toy_lf(
            absolute_mag=[-22.0, -21.0, -20.0],
            condition=[0.0, 1.0, 2.0],
            amplitude=1.0,
        )


def test_conditional_model_registry_exports_generated_names() -> None:
    """Tests that generated conditional model names are public exports."""

    assert "conditional_schechter" in __all__
    assert "conditional_double_schechter" in __all__
    assert "conditional_lognormal_lf" in __all__
    assert "conditional_two_component_lf" in __all__


def test_conditional_schechter_accepts_scalar_condition_with_callable_parameter() -> None:
    """Tests callable parameter evaluation for scalar condition input."""

    result = conditional_schechter(
        absolute_mag=np.array([-22.0, -21.0, -20.0]),
        condition=2.0,
        phi_star=lambda x: 1.0e-3 * (1.0 + x),
        m_star=-21.0,
        alpha=-1.1,
    )

    expected = schechter(
        np.array([-22.0, -21.0, -20.0]),
        phi_star=3.0e-3,
        m_star=-21.0,
        alpha=-1.1,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64
