"""Smoke tests for the user-facing luminosity function API."""

from __future__ import annotations

import inspect

import numpy as np
import pyccl as ccl
import pytest

from lfkit.api._namespaces import expose_lf_function
from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.luminosity_functions.registry import LF_MODELS


class DummyCorrections:
    """Simple correction object for testing correction dispatch."""

    def k(self, z):
        """Return a simple K-correction."""
        return np.zeros_like(np.asarray(z, dtype=float)) + 0.1

    def e(self, z):
        """Return a simple evolution correction."""
        return np.zeros_like(np.asarray(z, dtype=float)) + 0.2


def make_test_cosmology():
    """Return a small CCL cosmology for API tests."""
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
    """Tests that luminosity functions initialize grouped API namespaces."""
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
    """Tests that base luminosity functions do not expose conditional models."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    assert not hasattr(lf, "conditional_models")


def test_constructor_stores_model_parameters_and_meta():
    """Tests that constructors store model parameters and metadata."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        meta={"source": "test"},
    )

    assert lf.model == "schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.1,
    }
    assert lf.meta == {"source": "test"}


def test_constructor_uses_empty_meta_by_default():
    """Tests that constructors use empty metadata by default."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    assert lf.meta == {}


def test_constructor_preserves_arbitrary_parameters():
    """Tests that generic constructors preserve arbitrary parameters."""
    lf = LuminosityFunction(
        model="custom",
        parameters={"a": 1.0, "b": "x"},
        meta={"source": "test"},
    )

    assert lf.model == "custom"
    assert lf.parameters_dict == {"a": 1.0, "b": "x"}
    assert lf.meta == {"source": "test"}


def test_constructor_cleans_none_kwargs_parameters():
    """Tests that constructor kwargs ending in kwargs convert None to dictionaries."""
    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs=None,
        m_star_model="constant",
        m_star_kwargs=None,
        alpha_model="constant",
        alpha_kwargs=None,
    )

    assert lf.parameters_dict["phi_kwargs"] == {}
    assert lf.parameters_dict["m_star_kwargs"] == {}
    assert lf.parameters_dict["alpha_kwargs"] == {}


def test_expose_lf_function_injects_lf_callable_by_position():
    """Tests that namespace wrappers inject LF callables by position."""
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
    """Tests that namespace wrappers inject LF callables by keyword."""
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


def test_as_callable_returns_bound_phi_function():
    """Tests that luminosity functions expose a bound callable."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf._as_callable()(-20.0, 0.5)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_phi_evaluates_schechter_model():
    """Tests that phi evaluates a Schechter model."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.phi(-20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_phi_accepts_array_absolute_magnitudes():
    """Tests that phi accepts array absolute magnitudes."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    result = lf.phi(absolute_mag)

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


def test_phi_evaluates_double_schechter_model():
    """Tests that phi evaluates a double Schechter model."""
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


def test_evolving_schechter_constructor_stores_model_and_parameters():
    """Tests that evolving Schechter constructors store model parameters."""
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


def test_phi_requires_redshift_for_evolving_model():
    """Tests that evolving models require redshift in phi."""
    lf = LuminosityFunction.evolving_schechter()

    with pytest.raises(ValueError, match="z is required"):
        lf.phi(-20.0)


def test_phi_evaluates_evolving_model_with_redshift():
    """Tests that evolving models evaluate when redshift is supplied."""
    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"phi_star": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.5},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.1},
    )

    result = lf.phi(-20.0, z=0.5)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_parameters_raises_for_non_evolving_model():
    """Tests that parameters raises for non-evolving models."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    with pytest.raises(ValueError, match="only defined for evolving_schechter"):
        lf.parameters(0.5)


def test_parameters_evaluates_evolving_schechter_parameters():
    """Tests that parameters evaluates evolving Schechter parameter values."""
    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"phi_star": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.5},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.1},
    )

    phi_star, m_star, alpha = lf.parameters(0.5)

    assert np.asarray(phi_star).shape == ()
    assert np.asarray(m_star).shape == ()
    assert np.asarray(alpha).shape == ()
    assert np.isfinite(phi_star)
    assert np.isfinite(m_star)
    assert np.isfinite(alpha)


def test_phi_from_m_evaluates_supported_model():
    """Tests that phi_from_m evaluates a supported model."""
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


def test_phi_from_m_accepts_corrections_object():
    """Tests that phi_from_m accepts correction objects."""
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
        corrections=DummyCorrections(),
    )

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_correction_values_return_none_without_corrections():
    """Tests that correction values return None without corrections."""
    k_corr, e_corr = LuminosityFunction._correction_values(None, 0.5)

    assert k_corr is None
    assert e_corr is None


def test_correction_values_evaluate_corrections_object():
    """Tests that correction values evaluate correction objects."""
    k_corr, e_corr = LuminosityFunction._correction_values(
        DummyCorrections(),
        np.array([0.3, 0.5]),
    )

    assert np.allclose(k_corr, [0.1, 0.1])
    assert np.allclose(e_corr, [0.2, 0.2])


def test_with_luminosity_cutoff_preserves_model_and_merges_meta():
    """Tests that luminosity cutoffs preserve model names and merge metadata."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
        meta={"source": "base"},
    )

    modified = lf.with_luminosity_cutoff(meta={"cutoff": True})

    assert modified.model == "schechter"
    assert modified.parameters_dict == lf.parameters_dict
    assert modified.meta == {"source": "base", "cutoff": True}


def test_with_luminosity_cutoff_suppresses_phi_values():
    """Tests that luminosity cutoffs suppress phi values."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )
    modified = lf.with_luminosity_cutoff(
        cutoff_power=2.0,
        cutoff_amplitude=1.0,
    )

    base_phi = lf.phi(-22.0)
    modified_phi = modified.phi(-22.0)

    assert np.asarray(modified_phi).shape == ()
    assert np.isfinite(modified_phi)
    assert modified_phi < base_phi


def test_with_luminosity_cutoff_accepts_explicit_m_star():
    """Tests that luminosity cutoffs accept explicit m_star values."""
    lf = LuminosityFunction.gaussian(
        mean_absolute_mag=-20.5,
        sigma_absolute_mag=0.5,
        amplitude=1.0,
    )

    modified = lf.with_luminosity_cutoff(m_star=-20.5)

    result = modified.phi(-20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_with_luminosity_cutoff_requires_m_star_when_missing():
    """Tests that luminosity cutoffs require m_star when unavailable."""
    lf = LuminosityFunction.gaussian(
        mean_absolute_mag=-20.5,
        sigma_absolute_mag=0.5,
        amplitude=1.0,
    )

    with pytest.raises(ValueError, match="m_star must be supplied"):
        lf.with_luminosity_cutoff()


def test_integrals_namespace_delegates_to_bound_lf_callable():
    """Tests that integrals delegate to the bound LF callable."""
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
    """Tests that completeness delegates to the bound LF callable."""
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
    """Tests that completeness exposes absolute magnitude limits."""
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
    """Tests that magnitude namespace static helpers are available."""
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
    """Tests that luminosity namespace static helpers are available."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    result = lf.luminosities.ratio_from_magnitudes(-21.0, -20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


def test_unsupported_model_raises_clear_error():
    """Tests that unknown LF models raise clear errors."""
    lf = LuminosityFunction(
        model="not_a_model",
        parameters={},
    )

    with pytest.raises(ValueError, match="Unknown luminosity function model"):
        lf.phi(-20.0, 0.5)


def test_unsupported_phi_from_m_model_raises_clear_error():
    """Tests that unsupported phi_from_m models raise clear errors."""
    lf = LuminosityFunction(
        model="not_a_model",
        parameters={},
    )
    cosmo = make_test_cosmology()

    with pytest.raises(ValueError, match="phi_from_m is not defined"):
        lf.phi_from_m(cosmo, 0.5, 24.0, h=0.7)


def test_available_model_helpers_return_public_model_names():
    """Tests that available model helpers return public model names."""
    assert "schechter" in LuminosityFunction.available_models()
    assert "evolving_schechter" in LuminosityFunction.available_models()
    assert "schechter" in LuminosityFunction.available_from_m_models()


def test_available_model_helpers_return_sorted_names():
    """Tests that available model helpers return sorted names."""
    assert LuminosityFunction.available_models() == sorted(
        LuminosityFunction.available_models()
    )
    assert LuminosityFunction.available_from_m_models() == sorted(
        LuminosityFunction.available_from_m_models()
    )


def test_available_parameter_models_returns_grouped_registry_names():
    """Tests that parameter model helpers return grouped registry names."""
    models = LuminosityFunction.available_parameter_models()

    assert "phi_star" in models
    assert "m_star" in models
    assert "alpha" in models


def test_integrals_namespace_exposes_expected_methods():
    """Tests that integrals namespace exposes expected methods."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected_methods = [
        "number_density",
        "weighted",
        "selection_weighted_number_density",
        "luminosity_density",
        "mean_luminosity",
        "cumulative_number_density",
        "magnitude_window_number_density",
        "selection_fraction",
        "selection_function",
        "luminosity_weight",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.integrals, name))


def test_redshift_density_namespace_exposes_expected_methods():
    """Tests that redshift density namespace exposes expected methods."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected_methods = [
        "integrated_number_density",
        "weighted",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.redshift_density, name))


def test_completeness_namespace_exposes_expected_methods():
    """Tests that completeness namespace exposes expected methods."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected_methods = [
        "observed_number_density",
        "missing_number_density",
        "catalog_fraction",
        "out_of_catalog_fraction",
        "absolute_magnitude_limit",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.completeness, name))


def test_magnitudes_namespace_exposes_expected_methods():
    """Tests that magnitudes namespace exposes expected methods."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected_methods = [
        "correction",
        "absolute",
        "absolute_from_luminosity_distance",
        "apparent",
        "apparent_from_luminosity_distance",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.magnitudes, name))


def test_luminosities_namespace_exposes_expected_methods():
    """Tests that luminosities namespace exposes expected methods."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.1,
    )

    expected_methods = [
        "ratio",
        "ratio_from_magnitudes",
        "magnitude_difference_from_ratio",
        "weight_from_magnitude",
        "from_magnitude",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.luminosities, name))


def _minimal_parameter_payload(model_name, function):
    """Return a minimal parameter payload for a registered model."""
    if model_name == "evolving_schechter":
        return {
            "phi_model": "constant",
            "phi_kwargs": {"phi_star": 1.0e-3},
            "m_star_model": "constant",
            "m_star_kwargs": {"m_star": -20.5},
            "alpha_model": "constant",
            "alpha_kwargs": {"alpha": -1.1},
        }

    if model_name == "tabulated":
        return {
            "magnitude_grid": np.array([-24.0, -22.0, -20.0, -18.0]),
            "phi_grid": np.array([1.0e-5, 5.0e-4, 1.0e-3, 2.0e-4]),
        }

    if model_name == "binned":
        return {
            "magnitude_bin_edges": np.array([-24.0, -22.0, -20.0, -18.0]),
            "phi_bin_values": np.array([1.0e-5, 5.0e-4, 1.0e-3]),
        }

    if model_name == "redshift_tabulated":
        return {
            "magnitude_grid": np.array([-24.0, -22.0, -20.0, -18.0]),
            "redshift_grid": np.array([0.1, 0.5, 1.0]),
            "phi_grid": np.array(
                [
                    [1.0e-5, 5.0e-4, 1.0e-3, 2.0e-4],
                    [2.0e-5, 6.0e-4, 1.1e-3, 3.0e-4],
                    [3.0e-5, 7.0e-4, 1.2e-3, 4.0e-4],
                ]
            ),
        }

    if model_name == "redshift_binned":
        return {
            "magnitude_bin_edges": np.array([-24.0, -22.0, -20.0, -18.0]),
            "redshift_bin_edges": np.array([0.1, 0.5, 1.0, 1.5]),
            "phi_bin_values": np.array(
                [
                    [1.0e-5, 5.0e-4, 1.0e-3],
                    [2.0e-5, 6.0e-4, 1.1e-3],
                    [3.0e-5, 7.0e-4, 1.2e-3],
                ]
            ),
        }

    if model_name == "distance_tabulated":
        return {
            "magnitude_grid": np.array([-24.0, -22.0, -20.0, -18.0]),
            "distance_grid": np.array([100.0, 500.0, 1000.0]),
            "comoving_distance": 500.0,
            "phi_grid": np.array(
                [
                    [1.0e-5, 5.0e-4, 1.0e-3, 2.0e-4],
                    [2.0e-5, 6.0e-4, 1.1e-3, 3.0e-4],
                    [3.0e-5, 7.0e-4, 1.2e-3, 4.0e-4],
                ]
            ),
        }

    if model_name == "distance_binned":
        return {
            "magnitude_bin_edges": np.array([-24.0, -22.0, -20.0, -18.0]),
            "distance_bin_edges": np.array([100.0, 500.0, 1000.0, 1500.0]),
            "comoving_distance": 500.0,
            "phi_bin_values": np.array(
                [
                    [1.0e-5, 5.0e-4, 1.0e-3],
                    [2.0e-5, 6.0e-4, 1.1e-3],
                    [3.0e-5, 7.0e-4, 1.2e-3],
                ]
            ),
        }

    payload = {}

    for name, parameter in inspect.signature(function).parameters.items():
        if name in {"absolute_mag", "z", "redshift"}:
            continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        if name in {"phi_star", "modified_phi_star", "phi_star_1", "phi_star_2"}:
            payload[name] = 1.0e-3
        elif name == "log_phi_star":
            payload[name] = -3.0
        elif name in {
            "m_star",
            "modified_m_star",
            "mean_absolute_mag",
            "lognormal_mean_absolute_mag",
            "m_transition",
            "m_star_1",
            "m_star_2",
        }:
            payload[name] = -20.5
        elif name in {
            "alpha",
            "alpha_faint",
            "alpha_bright",
            "modified_alpha",
            "alpha_1",
            "alpha_2",
        }:
            payload[name] = -1.1
        elif name == "beta":
            payload[name] = 1.0
        elif name == "phi_stars":
            payload[name] = np.array([1.0e-3, 5.0e-4])
        elif name == "m_stars":
            payload[name] = np.array([-20.5, -19.5])
        elif name == "alphas":
            payload[name] = np.array([-1.1, -0.5])
        elif name in {
            "sigma_absolute_mag",
            "sigma_log_luminosity",
            "lognormal_sigma_log_luminosity",
            "sigma",
            "sigma_1",
            "sigma_2",
        }:
            payload[name] = 0.7
        elif name in {"amplitude", "lognormal_amplitude"}:
            payload[name] = 1.0
        elif name in {"fraction", "modified_luminosity_fraction"}:
            payload[name] = 0.5
        else:
            pytest.fail(f"No test default for required parameter {name!r}")

    return payload


@pytest.mark.parametrize("model_name", sorted(LF_MODELS))
def test_registered_luminosity_function_models_expose_constructors(model_name):
    """Tests that registered models expose public API constructors."""
    assert callable(getattr(LuminosityFunction, model_name))


@pytest.mark.parametrize("model_name", sorted(LF_MODELS))
def test_registered_luminosity_function_models_evaluate_phi(model_name):
    """Tests that registered models evaluate phi through the API."""
    model_spec = LF_MODELS[model_name]
    lf = getattr(LuminosityFunction, model_name)(
        **_minimal_parameter_payload(model_name, model_spec.function)
    )

    if model_spec.requires_z:
        result = lf.phi(-20.0, z=0.5)
    else:
        result = lf.phi(-20.0)

    assert np.asarray(result).shape == ()
    assert np.isfinite(result)


@pytest.mark.parametrize("model_name", sorted(LF_MODELS))
def test_registered_luminosity_function_models_expose_generic_integrals(model_name):
    """Tests that registered models expose generic integral methods."""
    model_spec = LF_MODELS[model_name]
    lf = getattr(LuminosityFunction, model_name)(
        **_minimal_parameter_payload(model_name, model_spec.function)
    )

    expected_methods = [
        "number_density",
        "weighted",
        "selection_weighted_number_density",
        "luminosity_density",
        "mean_luminosity",
        "cumulative_number_density",
        "magnitude_window_number_density",
        "selection_fraction",
        "selection_function",
        "luminosity_weight",
    ]

    for name in expected_methods:
        assert callable(getattr(lf.integrals, name))


def test_available_models_includes_extended_model_families() -> None:
    """Tests that available models includes extended model families."""
    models = LuminosityFunction.available_models()

    expected = [
        "gamma",
        "generalized_gamma",
        "saunders",
        "evolving_saunders",
        "double_saunders",
        "generalized_saunders",
        "tabulated",
        "binned",
        "redshift_tabulated",
        "redshift_binned",
        "distance_tabulated",
        "distance_binned",
    ]

    for name in expected:
        assert name in models
