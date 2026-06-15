"""Unit tests for ``api.conditional_luminosity_function``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.api.conditional_luminosity_function import ConditionalLuminosityFunction
from lfkit.api.luminosity_function import LuminosityFunction


def test_conditional_schechter_constructor_stores_parameters():
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


def test_conditional_double_schechter_constructor_stores_parameters():
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


def test_lognormal_constructor_stores_parameters():
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


def test_lognormal_constructor_uses_default_amplitude():
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


def test_modified_schechter_constructor_is_not_registered() -> None:
    """Tests that modified Schechter is not a standalone conditional model."""
    assert not hasattr(ConditionalLuminosityFunction, "modified_schechter")


def test_two_component_constructor_stores_parameters():
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


def test_two_component_constructor_uses_default_optional_parameters():
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


def test_available_models_includes_expected_models():
    """Tests that available models includes expected conditional models."""
    models = ConditionalLuminosityFunction.available_models()

    assert "schechter" in models
    assert "double_schechter" in models
    assert "lognormal" in models
    assert "two_component" in models
    assert models == sorted(models)


def test_phi_requires_condition():
    """Tests that conditional phi requires a condition."""
    lf = ConditionalLuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    with pytest.raises(ValueError, match="condition is required"):
        lf.phi(-20.0)


def test_phi_accepts_scalar_condition():
    """Tests that conditional phi accepts scalar conditions."""
    lf = ConditionalLuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    phi = lf.phi(-20.0, condition=0.5)

    assert np.asarray(phi).shape == ()
    assert np.isfinite(phi)


def test_phi_accepts_array_condition():
    """Tests that conditional phi accepts array conditions."""
    lf = ConditionalLuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([0.1, 0.5, 1.0])

    phi = lf.phi(absolute_mag, condition=condition)

    assert phi.shape == absolute_mag.shape
    assert np.all(np.isfinite(phi))


def test_constructor_rejects_unexpected_parameter():
    """Tests that constructors reject unexpected parameters."""
    with pytest.raises(TypeError, match="Unexpected parameter"):
        ConditionalLuminosityFunction.schechter(
            phi_star=1.0e-3,
            m_star=-20.5,
            alpha=-1.1,
            bad_parameter=123,
        )
