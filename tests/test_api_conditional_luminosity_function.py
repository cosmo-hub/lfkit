"""Unit tests for ``api.conditional_luminosity_function``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.api.conditional_luminosity_function import ConditionalLuminosityFunction
from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.luminosity_functions.registry import CONDITIONAL_LF_MODELS


def test_conditional_schechter_constructor_stores_parameters() -> None:
    """Tests that the conditional Schechter constructor stores parameters."""
    lf = ConditionalLuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
    }
    assert lf.meta == {"source": "test"}


def test_conditional_double_schechter_constructor_stores_parameters() -> None:
    """Tests that the conditional double Schechter constructor stores parameters."""
    lf = ConditionalLuminosityFunction.double_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        beta=-1.5,
        m_transition=-19.5,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "double_schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
        "beta": -1.5,
        "m_transition": -19.5,
    }
    assert lf.meta == {"source": "test"}


def test_lognormal_constructor_stores_parameters() -> None:
    """Tests that the conditional lognormal constructor stores parameters."""
    lf = ConditionalLuminosityFunction.lognormal(
        mean_absolute_mag=-20.5,
        sigma_log_luminosity=0.2,
        amplitude=2.0,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "lognormal"
    assert lf.parameters_dict == {
        "mean_absolute_mag": -20.5,
        "sigma_log_luminosity": 0.2,
        "amplitude": 2.0,
    }
    assert lf.meta == {"source": "test"}


def test_lognormal_constructor_uses_default_amplitude() -> None:
    """Tests that the conditional lognormal constructor uses default amplitude."""
    lf = ConditionalLuminosityFunction.lognormal(
        mean_absolute_mag=-20.5,
        sigma_log_luminosity=0.2,
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "lognormal"
    assert lf.parameters_dict == {
        "mean_absolute_mag": -20.5,
        "sigma_log_luminosity": 0.2,
        "amplitude": 1.0,
    }
    assert lf.meta == {}


def test_luminosity_cutoff_modifier_is_not_registered() -> None:
    """Tests that luminosity cutoff modifiers are not registered as conditional models."""
    assert "apply_luminosity_cutoff" not in CONDITIONAL_LF_MODELS


def test_two_component_constructor_stores_parameters() -> None:
    """Tests that the conditional two-component constructor stores parameters."""
    lf = ConditionalLuminosityFunction.two_component(
        lognormal_mean_absolute_mag=-21.0,
        lognormal_sigma_log_luminosity=0.2,
        lognormal_amplitude=2.0,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
        modified_m_star=-20.0,
        modified_luminosity_fraction=0.6,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "two_component"
    assert lf.parameters_dict == {
        "lognormal_mean_absolute_mag": -21.0,
        "lognormal_sigma_log_luminosity": 0.2,
        "lognormal_amplitude": 2.0,
        "modified_phi_star": 1.0e-3,
        "modified_alpha": -1.1,
        "modified_m_star": -20.0,
        "modified_luminosity_fraction": 0.6,
    }
    assert lf.meta == {"source": "test"}


def test_two_component_constructor_uses_default_optional_parameters() -> None:
    """Tests that the two-component constructor uses optional defaults."""
    lf = ConditionalLuminosityFunction.two_component(
        lognormal_mean_absolute_mag=-21.0,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "two_component"
    assert lf.parameters_dict == {
        "lognormal_mean_absolute_mag": -21.0,
        "lognormal_sigma_log_luminosity": 0.2,
        "lognormal_amplitude": 1.0,
        "modified_phi_star": 1.0e-3,
        "modified_alpha": -1.1,
        "modified_m_star": None,
        "modified_luminosity_fraction": 0.562,
    }
    assert lf.meta == {}


def test_available_models_includes_expected_models() -> None:
    """Tests that available models includes expected conditional models."""
    models = ConditionalLuminosityFunction.available_models()

    assert "schechter" in models
    assert "double_schechter" in models
    assert "lognormal" in models
    assert "two_component" in models
    assert models == sorted(models)


@pytest.mark.parametrize(
    "name",
    [
        "saunders",
        "evolving_saunders",
        "double_saunders",
        "generalized_saunders",
        "gamma",
        "generalized_gamma",
        "tabulated",
        "binned",
        "redshift_tabulated",
        "redshift_binned",
        "distance_tabulated",
        "distance_binned",
    ],
)
def test_available_models_includes_extended_model_families(name: str) -> None:
    """Tests that extended model families are available as conditional models."""
    assert name in ConditionalLuminosityFunction.available_models()


def test_constructor_rejects_unexpected_parameter() -> None:
    """Tests that constructors reject unexpected parameters."""
    with pytest.raises(TypeError, match="Unexpected parameter"):
        ConditionalLuminosityFunction.schechter(
            phi_star=1.0e-3,
            m_star=-20.5,
            alpha=-1.1,
            bad_parameter=123,
        )


CONDITIONAL_MODEL_TEST_PARAMETERS = {
    "schechter": {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
    },
    "double_schechter": {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
        "beta": -1.5,
        "m_transition": -19.5,
    },
    "lognormal": {
        "mean_absolute_mag": -20.5,
        "sigma_log_luminosity": 0.2,
        "amplitude": 1.0,
    },
    "two_component": {
        "lognormal_mean_absolute_mag": -21.0,
        "lognormal_sigma_log_luminosity": 0.2,
        "lognormal_amplitude": 1.0,
        "modified_phi_star": 1.0e-3,
        "modified_alpha": -1.1,
        "modified_m_star": -20.0,
        "modified_luminosity_fraction": 0.562,
    },
}


@pytest.fixture(params=CONDITIONAL_MODEL_TEST_PARAMETERS)
def conditional_lf(request: pytest.FixtureRequest) -> LuminosityFunction:
    """Return a conditional luminosity function for API tests."""
    name = str(request.param)
    constructor = getattr(ConditionalLuminosityFunction, name)
    return constructor(**CONDITIONAL_MODEL_TEST_PARAMETERS[name])


def test_phi_requires_condition(conditional_lf: LuminosityFunction) -> None:
    """Tests that conditional phi requires a conditioning variable."""
    with pytest.raises(
        ValueError,
        match="At least one conditioning variable is required",
    ):
        conditional_lf.phi(-20.0)


def test_phi_accepts_scalar_condition(conditional_lf: LuminosityFunction) -> None:
    """Tests that conditional phi accepts scalar conditioning variables."""
    phi = conditional_lf.phi(-20.0, 0.5)

    assert np.asarray(phi).shape == ()
    assert np.isfinite(phi)


def test_phi_accepts_array_condition(conditional_lf: LuminosityFunction) -> None:
    """Tests that conditional phi accepts array conditioning variables."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.1, 0.5, 1.0])

    phi = conditional_lf.phi(absolute_mag, condition)

    assert phi.shape == absolute_mag.shape
    assert np.all(np.isfinite(phi))


def test_phi_accepts_two_conditions(
    conditional_lf: LuminosityFunction,
) -> None:
    """Tests that conditional phi accepts two conditioning variables."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    first_condition = np.array([0.1, 0.5, 1.0])
    second_condition = np.array([1.0, 2.0, 3.0])

    phi = conditional_lf.phi(
        absolute_mag,
        first_condition,
        second_condition,
    )

    assert phi.shape == absolute_mag.shape
    assert np.all(np.isfinite(phi))


def test_phi_accepts_three_conditions(
    conditional_lf: LuminosityFunction,
) -> None:
    """Tests that conditional phi accepts three conditioning variables."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    first_condition = np.array([0.1, 0.5, 1.0])
    second_condition = np.array([1.0, 2.0, 3.0])
    third_condition = np.array([10.0, 20.0, 30.0])

    phi = conditional_lf.phi(
        absolute_mag,
        first_condition,
        second_condition,
        third_condition,
    )

    assert phi.shape == absolute_mag.shape
    assert np.all(np.isfinite(phi))


def test_phi_rejects_nonfinite_condition(
    conditional_lf: LuminosityFunction,
) -> None:
    """Tests that conditional phi rejects non-finite conditioning variables."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        conditional_lf.phi(-20.0, np.nan)


def test_phi_rejects_nonfinite_later_condition(
    conditional_lf: LuminosityFunction,
) -> None:
    """Tests that conditional phi rejects non-finite later conditioning variables."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        conditional_lf.phi(-20.0, 0.5, np.inf)
