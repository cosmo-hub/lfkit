"""Unit tests for ``lfkit.utils.types``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, get_args, get_origin

import numpy as np

import lfkit.utils.types as lf_types


def test_types_exports_expected_public_names() -> None:
    """Tests that types exposes the expected public API names."""
    expected = {
        "Cosmology",
        "FloatArray",
        "FloatInput",
        "LuminosityFunction",
        "ParameterModel",
        "ParameterValue",
        "ConditionalParameter",
    }

    assert set(lf_types.__all__) == expected


def test_float_array_alias_points_to_numpy_ndarray() -> None:
    """Tests that FloatArray aliases a NumPy ndarray type."""
    assert get_origin(lf_types.FloatArray) is np.ndarray
    assert "float64" in str(lf_types.FloatArray)


def test_float_input_includes_expected_user_input_forms() -> None:
    """Tests that FloatInput includes scalar, sequence, and array input forms."""
    args = get_args(lf_types.FloatInput)

    assert float in args
    assert Sequence[float] in args
    assert lf_types.FloatArray in args


def test_parameter_value_matches_float_input() -> None:
    """Tests that ParameterValue is the FloatInput alias."""
    assert lf_types.ParameterValue == lf_types.FloatInput


def test_parameter_model_is_callable_alias() -> None:
    """Tests that ParameterModel is a callable alias."""
    assert get_origin(lf_types.ParameterModel) is Callable
    assert lf_types.FloatArray in get_args(lf_types.ParameterModel)


def test_cosmology_alias_is_any() -> None:
    """Tests that Cosmology is the Any alias."""
    assert lf_types.Cosmology is Any


def test_luminosity_function_is_callable_alias() -> None:
    """Tests that LuminosityFunction is a callable alias."""
    assert get_origin(lf_types.LuminosityFunction) is Callable
    assert lf_types.FloatArray in get_args(lf_types.LuminosityFunction)


def test_conditional_parameter_includes_parameter_value_text() -> None:
    """Tests that ConditionalParameter includes the parameter-value alias."""
    assert str(lf_types.ParameterValue) in str(lf_types.ConditionalParameter)


def test_conditional_parameter_includes_callable_text() -> None:
    """Tests that ConditionalParameter includes callable parameter models."""
    assert "Callable" in str(lf_types.ConditionalParameter)
    assert "FloatArray" not in str(lf_types.ConditionalParameter) or "ndarray" in str(
        lf_types.ConditionalParameter
    )
