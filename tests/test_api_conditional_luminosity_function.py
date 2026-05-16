"""Smoke tests for conditional luminosity function API constructors.

These tests check that the public conditional LF factory methods create
LuminosityFunction objects with the expected model names and parameter payloads.
They intentionally avoid testing conditional LF physics, which is covered by
the lower-level photometry tests.
"""

from __future__ import annotations

from lfkit.api.conditional_luminosity_function import ConditionalLuminosityFunction
from lfkit.api.luminosity_function import LuminosityFunction


def test_conditional_schechter_constructor_delegates_to_luminosity_function():
    lf = ConditionalLuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "conditional_schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
    }
    assert lf.meta == {"source": "test"}


def test_conditional_evolving_schechter_constructor_delegates_to_luminosity_function():
    lf = ConditionalLuminosityFunction.evolving_schechter(
        phi_model="linear_p",
        phi_kwargs={"phi_star0": 1.0e-3, "p": 0.2},
        m_star_model="linear_q",
        m_star_kwargs={"m_star0": -20.5, "q": 0.5},
        alpha_model="constant",
        alpha_kwargs={"value": -1.1},
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "conditional_evolving_schechter"
    assert lf.parameters_dict == {
        "phi_model": "linear_p",
        "phi_kwargs": {"phi_star0": 1.0e-3, "p": 0.2},
        "m_star_model": "linear_q",
        "m_star_kwargs": {"m_star0": -20.5, "q": 0.5},
        "alpha_model": "constant",
        "alpha_kwargs": {"value": -1.1},
    }
    assert lf.meta == {"source": "test"}


def test_conditional_evolving_schechter_constructor_uses_empty_default_kwargs():
    lf = ConditionalLuminosityFunction.evolving_schechter()

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "conditional_evolving_schechter"
    assert lf.parameters_dict == {
        "phi_model": "linear_p",
        "phi_kwargs": {},
        "m_star_model": "linear_q",
        "m_star_kwargs": {},
        "alpha_model": "constant",
        "alpha_kwargs": {},
    }
    assert lf.meta == {}


def test_conditional_double_schechter_constructor_delegates_to_luminosity_function():
    lf = ConditionalLuminosityFunction.double_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        beta=-1.5,
        m_transition=-19.5,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "conditional_double_schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
        "beta": -1.5,
        "m_transition": -19.5,
    }
    assert lf.meta == {"source": "test"}


def test_lognormal_constructor_delegates_to_luminosity_function():
    lf = ConditionalLuminosityFunction.lognormal(
        mean_absolute_mag=-20.5,
        sigma_log_luminosity=0.2,
        amplitude=2.0,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "lognormal_conditional_lf"
    assert lf.parameters_dict == {
        "mean_absolute_mag": -20.5,
        "sigma_log_luminosity": 0.2,
        "amplitude": 2.0,
    }
    assert lf.meta == {"source": "test"}


def test_lognormal_constructor_uses_default_amplitude():
    lf = ConditionalLuminosityFunction.lognormal(
        mean_absolute_mag=-20.5,
        sigma_log_luminosity=0.2,
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "lognormal_conditional_lf"
    assert lf.parameters_dict == {
        "mean_absolute_mag": -20.5,
        "sigma_log_luminosity": 0.2,
        "amplitude": 1.0,
    }
    assert lf.meta == {}


def test_modified_schechter_constructor_delegates_to_luminosity_function():
    lf = ConditionalLuminosityFunction.modified_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        meta={"source": "test"},
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "modified_schechter_conditional_lf"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
    }
    assert lf.meta == {"source": "test"}


def test_two_component_constructor_delegates_to_luminosity_function():
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
    assert lf.model == "two_component_conditional_lf"
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
    lf = ConditionalLuminosityFunction.two_component(
        lognormal_mean_absolute_mag=-21.0,
        lognormal_sigma_log_luminosity=0.2,
        modified_phi_star=1.0e-3,
        modified_alpha=-1.1,
    )

    assert isinstance(lf, LuminosityFunction)
    assert lf.model == "two_component_conditional_lf"
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
