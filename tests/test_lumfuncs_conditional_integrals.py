"""Unit tests for ``lfkit.photometry.conditional_lf_integrals``."""

import numpy as np
import pytest

from lfkit.luminosity_functions.conditional_integrals import (
    evaluate_conditional_luminosity_function,
    integrate_conditional_luminosity_function,
    integrate_weighted_conditional_luminosity_function,
)


def test_evaluate_conditional_luminosity_function_accepts_scalar_inputs() -> None:
    """Tests that scalar inputs are evaluated as float arrays."""

    def conditional_lf(absolute_mag, condition):
        return condition * np.exp(-0.1 * absolute_mag)

    result = evaluate_conditional_luminosity_function(
        -20.0,
        2.0,
        conditional_lf=conditional_lf,
    )

    expected = 2.0 * np.exp(2.0)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()
    assert result == pytest.approx(expected)


def test_evaluate_conditional_luminosity_function_accepts_array_inputs() -> None:
    """Tests that array inputs are evaluated element-wise."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([1.0, 2.0, 3.0])

    def conditional_lf(absolute_mag, condition):
        return condition * (absolute_mag + 23.0)

    result = evaluate_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
    )

    expected = np.array([1.0, 4.0, 9.0])

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_evaluate_conditional_luminosity_function_accepts_broadcastable_inputs() -> None:
    """Tests that broadcastable magnitude and condition arrays are supported."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([[1.0], [2.0]])

    def conditional_lf(absolute_mag, condition):
        return condition * (absolute_mag + 23.0)

    result = evaluate_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
    )

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_evaluate_conditional_luminosity_function_rejects_non_finite_absolute_mag() -> None:
    """Tests that non-finite absolute magnitudes are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values."):
        evaluate_conditional_luminosity_function(
            [-22.0, np.nan, -20.0],
            1.0,
            conditional_lf=conditional_lf,
        )


def test_evaluate_conditional_luminosity_function_rejects_non_finite_condition() -> None:
    """Tests that non-finite condition values are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    with pytest.raises(ValueError, match="condition_0 contains NaN or infinite values."):
        evaluate_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            np.inf,
            conditional_lf=conditional_lf,
        )


def test_evaluate_conditional_luminosity_function_rejects_non_finite_result() -> None:
    """Tests that non-finite conditional luminosity values are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.array([1.0, np.nan, 3.0])

    with pytest.raises(
        ValueError,
        match="conditional_lf returned NaN or infinite values.",
    ):
        evaluate_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
        )


def test_evaluate_conditional_luminosity_function_rejects_negative_result() -> None:
    """Tests that negative conditional luminosity values are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.array([1.0, -2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="conditional_lf returned negative values, which are not allowed.",
    ):
        evaluate_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
        )


def test_integrate_conditional_lf_integrates_over_last_axis() -> None:
    """Tests conditional luminosity function integration over magnitude."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([[1.0], [2.0]])

    def conditional_lf(absolute_mag, condition):
        return condition * (absolute_mag + 23.0)

    result = integrate_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
    )

    expected = np.trapezoid(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
            ]
        ),
        x=absolute_mag,
        axis=-1,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_integrate_conditional_lf_integrates_over_requested_axis() -> None:
    """Tests conditional luminosity function integration over a custom axis."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([[1.0, 2.0]])

    def conditional_lf(absolute_mag, condition):
        return (absolute_mag[:, None] + 23.0) * condition

    result = integrate_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
        axis=0,
    )

    phi = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    expected = np.trapezoid(phi, x=absolute_mag, axis=0)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_integrate_conditional_lf_rejects_invalid_conditional_lf() -> None:
    """Tests that integration propagates invalid conditional LF errors."""

    def conditional_lf(absolute_mag, condition):
        return np.array([1.0, -1.0, 3.0])

    with pytest.raises(
        ValueError,
        match="conditional_lf returned negative values, which are not allowed.",
    ):
        integrate_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
        )


def test_integrate_weighted_conditional_lf_integrates_weighted_values() -> None:
    """Tests weighted conditional luminosity function integration."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([[1.0], [2.0]])

    def conditional_lf(absolute_mag, condition):
        return condition * (absolute_mag + 23.0)

    def weight(absolute_mag, condition):
        return absolute_mag + 24.0

    result = integrate_weighted_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
        weight=weight,
    )

    phi = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )
    weight_arr = np.array([2.0, 3.0, 4.0])
    expected = np.trapezoid(phi * weight_arr, x=absolute_mag, axis=-1)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_integrate_weighted_conditional_lf_integrates_over_requested_axis() -> None:
    """Tests weighted conditional LF integration over a custom axis."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])
    condition = np.array([[1.0, 2.0]])

    def conditional_lf(absolute_mag, condition):
        return (absolute_mag[:, None] + 23.0) * condition

    def weight(absolute_mag, condition):
        return absolute_mag[:, None] + 24.0

    result = integrate_weighted_conditional_luminosity_function(
        absolute_mag,
        condition,
        conditional_lf=conditional_lf,
        weight=weight,
        axis=0,
    )

    phi = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    weight_arr = np.array(
        [
            [2.0],
            [3.0],
            [4.0],
        ]
    )
    expected = np.trapezoid(phi * weight_arr, x=absolute_mag, axis=0)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_integrate_weighted_conditional_lf_rejects_non_finite_weight() -> None:
    """Tests that non-finite weights are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    def weight(absolute_mag, condition):
        return np.array([1.0, np.inf, 3.0])

    with pytest.raises(ValueError, match="weight returned NaN or infinite values."):
        integrate_weighted_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
            weight=weight,
        )


def test_integrate_weighted_conditional_lf_allows_negative_weight() -> None:
    """Tests that negative weights are allowed."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    def weight(absolute_mag, condition):
        return np.array([1.0, -2.0, 3.0])

    result = integrate_weighted_conditional_luminosity_function(
        absolute_mag,
        1.0,
        conditional_lf=conditional_lf,
        weight=weight,
    )

    expected = np.trapezoid(np.array([1.0, -2.0, 3.0]), x=absolute_mag)

    assert result == pytest.approx(expected)
    assert result.dtype == np.float64


def test_integrate_weighted_conditional_lf_rejects_invalid_conditional_lf() -> None:
    """Tests that weighted integration propagates invalid conditional LF errors."""

    def conditional_lf(absolute_mag, condition):
        return np.array([1.0, -1.0, 3.0])

    def weight(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    with pytest.raises(
        ValueError,
        match="conditional_lf returned negative values, which are not allowed.",
    ):
        integrate_weighted_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
            weight=weight,
        )


def test_evaluate_conditional_lf_allows_zero_values() -> None:
    """Tests that zero conditional LF values are allowed."""

    def conditional_lf(absolute_mag, condition):
        return np.zeros_like(absolute_mag, dtype=float)

    result = evaluate_conditional_luminosity_function(
        [-22.0, -21.0, -20.0],
        1.0,
        conditional_lf=conditional_lf,
    )

    np.testing.assert_allclose(result, np.zeros(3))


def test_evaluate_conditional_lf_rejects_non_finite_broadcasted_condition() -> None:
    """Tests that non-finite broadcasted condition arrays are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.ones((2, 3))

    with pytest.raises(ValueError, match="condition_0 contains NaN or infinite values."):
        evaluate_conditional_luminosity_function(
            np.array([-22.0, -21.0, -20.0]),
            np.array([[1.0], [np.nan]]),
            conditional_lf=conditional_lf,
        )


def test_integrate_conditional_lf_matches_constant_function_width() -> None:
    """Tests integration of a constant conditional LF over magnitude width."""

    absolute_mag = np.array([-23.0, -22.0, -21.0, -20.0])

    def conditional_lf(absolute_mag, condition):
        return 2.0 * np.ones_like(absolute_mag)

    result = integrate_conditional_luminosity_function(
        absolute_mag,
        1.0,
        conditional_lf=conditional_lf,
    )

    assert result == pytest.approx(6.0)


def test_integrate_weighted_conditional_lf_accepts_scalar_weight() -> None:
    """Tests that scalar weights broadcast over the conditional LF."""

    absolute_mag = np.array([-22.0, -21.0, -20.0])

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    def weight(absolute_mag, condition):
        return 2.0

    result = integrate_weighted_conditional_luminosity_function(
        absolute_mag,
        1.0,
        conditional_lf=conditional_lf,
        weight=weight,
    )

    expected = np.trapezoid(2.0 * np.ones_like(absolute_mag), x=absolute_mag)

    assert result == pytest.approx(expected)


def test_integrate_weighted_conditional_lf_rejects_nan_weight() -> None:
    """Tests that NaN weights are rejected."""

    def conditional_lf(absolute_mag, condition):
        return np.ones_like(absolute_mag)

    def weight(absolute_mag, condition):
        return np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="weight returned NaN or infinite values."):
        integrate_weighted_conditional_luminosity_function(
            [-22.0, -21.0, -20.0],
            1.0,
            conditional_lf=conditional_lf,
            weight=weight,
        )
