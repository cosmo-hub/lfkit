"""Unit tests for ``api._namespaces``."""

from __future__ import annotations

import types

import numpy as np

from lfkit.api._namespaces import (
    LFCompletenessAPI,
    LFIntegralsAPI,
    LFLuminositiesAPI,
    LFMagnitudesAPI,
    LFRedshiftDensityAPI,
    _method_name,
    _public_functions,
    expose_lf_function,
)


class DummyLF:
    """Minimal luminosity function object for namespace tests."""

    def _as_callable(self):
        """Return a simple luminosity function callable."""
        return lambda absolute_mag, redshift=None: np.asarray(absolute_mag) + 1.0


class DummyBoundAPI:
    """Minimal bound API object for injection tests."""

    def __init__(self):
        """Create a dummy API object."""
        self.lf = DummyLF()


def test_integrals_namespace_stores_parent_lf():
    """Tests that integrals namespace stores the parent LF."""
    lf = DummyLF()
    api = LFIntegralsAPI(lf)

    assert api.lf is lf


def test_completeness_namespace_stores_parent_lf():
    """Tests that completeness namespace stores the parent LF."""
    lf = DummyLF()
    api = LFCompletenessAPI(lf)

    assert api.lf is lf


def test_redshift_density_namespace_stores_parent_lf():
    """Tests that redshift density namespace stores the parent LF."""
    lf = DummyLF()
    api = LFRedshiftDensityAPI(lf)

    assert api.lf is lf


def test_magnitude_namespace_can_be_instantiated():
    """Tests that magnitude namespace can be instantiated."""
    api = LFMagnitudesAPI()

    assert isinstance(api, LFMagnitudesAPI)


def test_luminosity_namespace_can_be_instantiated():
    """Tests that luminosity namespace can be instantiated."""
    api = LFLuminositiesAPI()

    assert isinstance(api, LFLuminositiesAPI)


def test_expose_lf_function_injects_lf_callable_by_position():
    """Tests that exposed functions inject LF callables by position."""
    calls = {}

    def low_level(x, lf_callable, z, *, scale=1.0):
        calls["x"] = x
        calls["z"] = z
        calls["scale"] = scale
        calls["lf_value"] = lf_callable(x, z)
        return scale * calls["lf_value"]

    method = expose_lf_function(low_level, lf_arg_position=1)
    api = DummyBoundAPI()

    result = method(api, 2.0, 0.5, scale=3.0)

    assert result == 9.0
    assert calls["x"] == 2.0
    assert calls["z"] == 0.5
    assert calls["scale"] == 3.0
    assert calls["lf_value"] == 3.0


def test_expose_lf_function_injects_lf_callable_by_keyword():
    """Tests that exposed functions inject LF callables by keyword."""
    calls = {}

    def low_level(x, z, *, lf):
        calls["lf_value"] = lf(x, z)
        return calls["lf_value"]

    method = expose_lf_function(
        low_level,
        lf_arg_position=None,
        lf_arg_name="lf",
    )
    api = DummyBoundAPI()

    result = method(api, 2.0, 0.5)

    assert result == 3.0
    assert calls["lf_value"] == 3.0


def test_expose_lf_function_uses_keyword_over_position():
    """Tests that keyword LF injection takes priority over position."""
    calls = {}

    def low_level(x, z, *, lf):
        calls["x"] = x
        calls["z"] = z
        calls["lf_value"] = lf(x, z)
        return calls["lf_value"]

    method = expose_lf_function(
        low_level,
        lf_arg_position=1,
        lf_arg_name="lf",
    )
    api = DummyBoundAPI()

    result = method(api, 2.0, 0.5)

    assert result == 3.0
    assert calls["x"] == 2.0
    assert calls["z"] == 0.5
    assert calls["lf_value"] == 3.0


def test_expose_lf_function_without_lf_argument_delegates_directly():
    """Tests that exposed functions can delegate without LF injection."""

    def low_level(x, *, scale=1.0):
        return scale * x

    method = expose_lf_function(low_level, lf_arg_position=None)
    api = DummyBoundAPI()

    result = method(api, 2.0, scale=4.0)

    assert result == 8.0


def test_expose_lf_function_preserves_function_metadata():
    """Tests that exposed methods preserve wrapped function metadata."""

    def low_level(x):
        """Low-level test function."""
        return x

    method = expose_lf_function(low_level, lf_arg_position=None)

    assert method.__name__ == "low_level"
    assert method.__doc__ == "Low-level test function."


def test_public_functions_returns_only_callables_from_all():
    """Tests that public function discovery returns callable __all__ members."""

    def public_function():
        return 1

    module = types.SimpleNamespace(
        __all__=["public_function", "public_value"],
        public_function=public_function,
        public_value=3,
    )

    functions = _public_functions(module)

    assert functions == {"public_function": public_function}


def test_public_functions_returns_empty_dict_without_all():
    """Tests that public function discovery returns empty output without __all__."""
    module = types.SimpleNamespace(public_function=lambda: 1)

    functions = _public_functions(module)

    assert functions == {}


def test_method_name_uses_alias_when_available():
    """Tests that method names use module API aliases when available."""
    module = types.SimpleNamespace(
        __api_aliases__={"low_level_name": "public_name"},
    )

    assert _method_name(module, "low_level_name") == "public_name"


def test_method_name_falls_back_to_function_name_without_alias():
    """Tests that method names fall back to the low-level function name."""
    module = types.SimpleNamespace(__api_aliases__={})

    assert _method_name(module, "low_level_name") == "low_level_name"


def test_integrals_namespace_has_bound_methods():
    """Tests that integrals namespace exposes bound methods."""
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
        assert callable(getattr(LFIntegralsAPI, name))


def test_completeness_namespace_has_expected_methods():
    """Tests that completeness namespace exposes expected methods."""
    expected_methods = [
        "observed_number_density",
        "missing_number_density",
        "catalog_fraction",
        "out_of_catalog_fraction",
        "absolute_magnitude_limit",
    ]

    for name in expected_methods:
        assert callable(getattr(LFCompletenessAPI, name))


def test_redshift_density_namespace_has_expected_methods():
    """Tests that redshift density namespace exposes expected methods."""
    expected_methods = [
        "integrated_number_density",
        "weighted",
    ]

    for name in expected_methods:
        assert callable(getattr(LFRedshiftDensityAPI, name))


def test_magnitude_namespace_has_expected_methods():
    """Tests that magnitude namespace exposes expected methods."""
    expected_methods = [
        "correction",
        "absolute",
        "absolute_from_luminosity_distance",
        "apparent",
        "apparent_from_luminosity_distance",
    ]

    for name in expected_methods:
        assert callable(getattr(LFMagnitudesAPI, name))


def test_luminosity_namespace_has_expected_methods():
    """Tests that luminosity namespace exposes expected methods."""
    expected_methods = [
        "ratio",
        "ratio_from_magnitudes",
        "magnitude_difference_from_ratio",
        "weight_from_magnitude",
        "from_magnitude",
    ]

    for name in expected_methods:
        assert callable(getattr(LFLuminositiesAPI, name))
