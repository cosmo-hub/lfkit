"""Smoke tests for user-facing API delegation.

These tests check that the public API namespaces are wired to the expected
low-level functions. They intentionally avoid testing luminosity-function
physics, which is covered by the lower-level photometry tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyccl as ccl

from lfkit.api._expose import expose_lf_function
from lfkit.api.luminosity_function import LuminosityFunction


def make_test_cosmology():
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    )


def test_luminosity_function_initializes_grouped_api_namespaces():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    assert hasattr(lf, "integrals")
    assert hasattr(lf, "redshift_density")
    assert hasattr(lf, "completeness")
    assert hasattr(lf, "luminosities")
    assert hasattr(lf, "magnitudes")


def test_luminosity_function_does_not_initialize_conditional_models_namespace():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    assert not hasattr(lf, "conditional_models")


def test_expose_lf_function_injects_lf_callable_by_position():
    calls = {}

    def low_level(x, lf_callable, z, *, scale=1.0):
        calls["x"] = x
        calls["z"] = z
        calls["scale"] = scale
        calls["lf_value"] = lf_callable(x, z)
        return scale * calls["lf_value"]

    class Parent:
        def _as_callable(self):
            return lambda absolute_mag, redshift: absolute_mag + redshift

    class API:
        def __init__(self):
            self.lf = Parent()

    API.method = expose_lf_function(low_level, lf_arg_position=1)

    api = API()
    result = api.method(2.0, 3.0, scale=4.0)

    assert result == 20.0
    assert calls["x"] == 2.0
    assert calls["z"] == 3.0
    assert calls["scale"] == 4.0
    assert calls["lf_value"] == 5.0


def test_expose_lf_function_injects_lf_callable_by_keyword():
    calls = {}

    def low_level(x, z, *, lf_callable):
        calls["lf_value"] = lf_callable(x, z)
        return calls["lf_value"]

    class Parent:
        def _as_callable(self):
            return lambda absolute_mag, redshift: absolute_mag * redshift

    class API:
        def __init__(self):
            self.lf = Parent()

    API.method = expose_lf_function(
        low_level,
        lf_arg_position=None,
        lf_arg_name="lf_callable",
    )

    api = API()
    result = api.method(2.0, 3.0)

    assert result == 6.0
    assert calls["lf_value"] == 6.0


def test_phi_evaluates_schechter_model():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.phi(-20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_evolving_schechter_constructor_stores_model_and_parameter():
    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"phi_star": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.5},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.1},
        meta={"source": "test"},
    )

    assert lf.model == "evolving_schechter"
    assert lf.parameters_dict == {
        "phi_model": "constant",
        "phi_kwargs": {"phi_star": 1.0e-3},
        "m_star_model": "constant",
        "m_star_kwargs": {"m_star": -20.5},
        "alpha_model": "constant",
        "alpha_kwargs": {"alpha": -1.1},
    }
    assert lf.meta == {"source": "test"}


def test_phi_evaluates_double_schechter_model():
    lf = LuminosityFunction.double_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        beta=-1.5,
        m_transition=-19.5,
    )

    result = lf.phi(-20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_phi_from_m_evaluates_supported_model():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )
    cosmo = make_test_cosmology()

    result = lf.phi_from_m(
        cosmo,
        0.5,
        24.0,
        h=0.7,
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_phi_requires_redshift_for_evolving_model():
    lf = LuminosityFunction.evolving_schechter()

    with pytest.raises(ValueError, match="z is required"):
        lf.phi(-20.0)


def test_parameters_raises_for_non_evolving_model():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    with pytest.raises(ValueError, match="only defined for evolving_schechter"):
        lf.parameters(0.5)


def test_integrals_namespace_delegates_to_bound_lf_callable():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.integrals.number_density(
        0.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=32,
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_completeness_namespace_delegates_to_bound_lf_callable():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )
    cosmo = make_test_cosmology()

    result = lf.completeness.catalog_fraction(
        cosmo,
        0.5,
        m_lim=24.0,
        m_bright=-24.0,
        m_faint=-16.0,
        n_m=32,
        h=0.7,
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)
    assert 0.0 <= float(result) <= 1.0


def test_completeness_absolute_magnitude_limit_is_static_helper():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )
    cosmo = make_test_cosmology()

    result = lf.completeness.absolute_magnitude_limit(
        cosmo,
        0.5,
        m_lim=24.0,
        h=0.7,
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_magnitude_namespace_static_helpers_are_available():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.magnitudes.absolute_from_luminosity_distance(
        24.0,
        1000.0,
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_luminosity_namespace_static_helpers_are_available():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.luminosities.ratio_from_magnitudes(-21.0, -20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_unsupported_model_raises_clear_error():
    lf = LuminosityFunction(
        model="not_a_model",
        parameters={},
    )

    with pytest.raises(ValueError, match="Unsupported luminosity function model"):
        lf.phi(-20.0, 0.5)


def test_unsupported_phi_from_m_model_raises_clear_error():
    lf = LuminosityFunction(
        model="not_a_model",
        parameters={},
    )
    cosmo = make_test_cosmology()

    with pytest.raises(ValueError, match="phi_from_m is not defined"):
        lf.phi_from_m(cosmo, 0.5, 24.0, h=0.7)


def test_available_model_helpers_return_public_model_names():
    assert "schechter" in LuminosityFunction.available_models()
    assert "evolving_schechter" in LuminosityFunction.available_models()
    assert "schechter" in LuminosityFunction.available_from_m_models()


def test_available_parameter_models_returns_grouped_registry_names():
    models = LuminosityFunction.available_parameter_models()

    assert "phi_star" in models
    assert "m_star" in models
    assert "alpha" in models


def test_integrals_namespace_exposes_expected_methods():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected = [
        "number_density",
        "weighted",
        "selection_weighted_number_density",
        "luminosity_density",
        "mean_luminosity",
        "cumulative_number_density",
        "magnitude_window_number_density",
    ]

    for name in expected:
        assert callable(getattr(lf.integrals, name))


def test_redshift_density_namespace_exposes_expected_methods():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected = [
        "integrated_number_density",
        "weighted",
    ]

    for name in expected:
        assert callable(getattr(lf.redshift_density, name))


def test_completeness_namespace_exposes_expected_methods():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected = [
        "observed_number_density",
        "missing_number_density",
        "catalog_fraction",
        "out_of_catalog_fraction",
        "absolute_magnitude_limit",
    ]

    for name in expected:
        assert callable(getattr(lf.completeness, name))


def test_magnitudes_namespace_exposes_expected_methods():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected = [
        "correction",
        "absolute",
        "absolute_from_luminosity_distance",
        "apparent",
        "apparent_from_luminosity_distance",
    ]

    for name in expected:
        assert callable(getattr(lf.magnitudes, name))


def test_luminosities_namespace_exposes_expected_methods():
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected = [
        "ratio",
        "ratio_from_magnitudes",
        "magnitude_difference_from_ratio",
        "weight_from_magnitude",
        "from_magnitude",
        "schechter_cumulative_number_density",
        "schechter_luminosity_density",
        "schechter_mean_luminosity",
        "sample_schechter",
        "schechter_selection",
    ]

    for name in expected:
        assert callable(getattr(lf.luminosities, name))
