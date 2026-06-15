"""Unit tests for ``lfkit.utils.integrators``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.integrators import (
    integrate_between_variable_bounds,
    safe_divide,
    safe_power10,
)


def constant_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a constant integrand."""
    return np.ones_like(np.broadcast_arrays(x, y)[0], dtype=float)


def double_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a constant integrand with amplitude two."""
    return 2.0 * np.ones_like(np.broadcast_arrays(x, y)[0], dtype=float)


def linear_x_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return an integrand that varies linearly with the integration coordinate."""
    _ = y
    return np.asarray(x, dtype=float)


def linear_y_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return an integrand that varies linearly with the external coordinate."""
    _ = x
    return np.asarray(y, dtype=float)


def test_integrate_between_variable_bounds_integrates_constant_function() -> None:
    """Tests that variable-bound integration returns the expected interval width."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=-24.0,
        upper=-18.0,
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_integrate_between_variable_bounds_preserves_integrand_amplitude() -> None:
    """Tests that variable-bound integration preserves integrand amplitude."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=-24.0,
        upper=-18.0,
        integrand_fn=double_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([12.0, 12.0]))


def test_integrate_between_variable_bounds_supports_array_bounds() -> None:
    """Tests that variable-bound integration supports coordinate-dependent bounds."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2, 0.3],
        lower=np.array([-24.0, -23.0, -22.0]),
        upper=np.array([-18.0, -18.0, -18.0]),
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 5.0, 4.0]))


def test_integrate_between_variable_bounds_supports_scalar_y() -> None:
    """Tests that variable-bound integration supports scalar coordinate input."""
    result = integrate_between_variable_bounds(
        0.1,
        lower=-24.0,
        upper=-18.0,
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    assert result.shape == ()
    assert result == pytest.approx(6.0)


def test_integrate_between_variable_bounds_returns_zero_for_empty_ranges() -> None:
    """Tests that integration returns zero where the upper bound is not larger."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2, 0.3],
        lower=np.array([-18.0, -20.0, -19.0]),
        upper=np.array([-20.0, -20.0, -21.0]),
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0]))


def test_integrate_between_variable_bounds_handles_mixed_valid_and_empty_ranges() -> None:
    """Tests that only valid intervals contribute to the integral."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2, 0.3],
        lower=np.array([-24.0, -18.0, -22.0]),
        upper=np.array([-18.0, -20.0, -18.0]),
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 0.0, 4.0]))


def test_integrate_between_variable_bounds_integrates_linear_x_function() -> None:
    """Tests integration of a linear function over the integration coordinate."""
    result = integrate_between_variable_bounds(
        0.1,
        lower=0.0,
        upper=2.0,
        integrand_fn=linear_x_integrand,
        n_grid=128,
    )

    assert result == pytest.approx(2.0)


def test_integrate_between_variable_bounds_integrates_y_dependent_function() -> None:
    """Tests integration of an external-coordinate-dependent function."""
    result = integrate_between_variable_bounds(
        [1.0, 2.0, 3.0],
        lower=0.0,
        upper=2.0,
        integrand_fn=linear_y_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


def test_integrate_between_variable_bounds_accepts_scalar_integrand_output() -> None:
    """Tests that scalar integrand output is broadcast to the integration grid."""

    def scalar_integrand(x: np.ndarray, y: np.ndarray) -> float:
        """Return a scalar integrand value."""
        _ = x
        _ = y
        return 3.0

    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=0.0,
        upper=2.0,
        integrand_fn=scalar_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_integrate_between_variable_bounds_accepts_broadcastable_integrand_output() -> None:
    """Tests that broadcastable integrand output is expanded to the grid shape."""

    def broadcastable_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return one integrand value per coordinate value."""
        _ = x
        _ = y
        return np.array([[1.0, 2.0]], dtype=float)

    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=0.0,
        upper=2.0,
        integrand_fn=broadcastable_integrand,
        n_grid=64,
    )

    np.testing.assert_allclose(result, np.array([2.0, 4.0]))


def test_integrate_between_variable_bounds_preserves_broadcasted_shape() -> None:
    """Tests that broadcasted coordinate and bound inputs define the output shape."""
    result = integrate_between_variable_bounds(
        np.array([[0.1], [0.2]], dtype=float),
        lower=np.array([0.0, 1.0, 2.0], dtype=float),
        upper=np.array([1.0, 3.0, 5.0], dtype=float),
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected)


def test_integrate_between_variable_bounds_rejects_small_grid() -> None:
    """Tests that variable-bound integration requires at least two grid points."""
    with pytest.raises(ValueError, match="n_grid must be at least 2"):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=0.0,
            upper=1.0,
            integrand_fn=constant_integrand,
            n_grid=1,
        )


def test_integrate_between_variable_bounds_rejects_nonfinite_lower_bound() -> None:
    """Tests that lower integration bounds must be finite."""
    with pytest.raises(ValueError, match="lower must contain only finite values"):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=np.nan,
            upper=1.0,
            integrand_fn=constant_integrand,
            n_grid=64,
        )


def test_integrate_between_variable_bounds_rejects_nonfinite_upper_bound() -> None:
    """Tests that upper integration bounds must be finite."""
    with pytest.raises(ValueError, match="upper must contain only finite values"):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=0.0,
            upper=np.inf,
            integrand_fn=constant_integrand,
            n_grid=64,
        )


def test_integrate_between_variable_bounds_rejects_unbroadcastable_bounds() -> None:
    """Tests that coordinate and bound inputs must be broadcastable together."""
    with pytest.raises(ValueError):
        integrate_between_variable_bounds(
            np.ones((2, 3), dtype=float),
            lower=np.ones((4,), dtype=float),
            upper=2.0,
            integrand_fn=constant_integrand,
            n_grid=64,
        )


def test_integrate_between_variable_bounds_rejects_unbroadcastable_integrand() -> None:
    """Tests that integrand output must be broadcastable to the integration grid."""

    def bad_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return integrand values with an invalid shape."""
        _ = x
        _ = y
        return np.ones((4, 4), dtype=float)

    with pytest.raises(
        ValueError,
        match="integrand_fn\\(x, y\\) must return values broadcastable",
    ):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=0.0,
            upper=1.0,
            integrand_fn=bad_integrand,
            n_grid=64,
        )


def test_integrate_between_variable_bounds_rejects_nonfinite_integrand_values() -> None:
    """Tests that integrand values must be finite."""

    def bad_integrand(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return non-finite integrand values."""
        _ = y
        return np.full_like(x, np.nan, dtype=float)

    with pytest.raises(
        ValueError,
        match="integrand_fn\\(x, y\\) returned non-finite values",
    ):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=0.0,
            upper=1.0,
            integrand_fn=bad_integrand,
            n_grid=64,
        )


def test_safe_divide_returns_ratio_for_positive_denominator() -> None:
    """Tests that safe division returns the ordinary ratio for positive denominators."""
    result = safe_divide(
        np.array([2.0, 6.0, 12.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


def test_safe_divide_returns_zero_for_zero_denominator() -> None:
    """Tests that safe division returns zero for zero denominators."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 2.0]),
    )

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 1.5]))


def test_safe_divide_returns_zero_for_negative_denominator() -> None:
    """Tests that safe division returns zero for negative denominators."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, -1.0, 2.0]),
    )

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 1.5]))


def test_safe_divide_supports_scalar_inputs() -> None:
    """Tests that safe division supports scalar inputs."""
    result = safe_divide(4.0, 2.0)

    assert result.shape == ()
    assert result == pytest.approx(2.0)


def test_safe_divide_returns_zero_for_scalar_zero_denominator() -> None:
    """Tests that safe division returns zero for scalar zero denominator."""
    result = safe_divide(4.0, 0.0)

    assert result.shape == ()
    assert result == pytest.approx(0.0)


def test_safe_divide_supports_broadcastable_inputs() -> None:
    """Tests that safe division supports broadcastable numerator and denominator."""
    result = safe_divide(
        np.array([[2.0], [4.0]], dtype=float),
        np.array([1.0, 2.0, 0.0], dtype=float),
    )

    expected = np.array(
        [
            [2.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)


def test_safe_divide_rejects_unbroadcastable_inputs() -> None:
    """Tests that safe division requires broadcastable inputs."""
    with pytest.raises(ValueError):
        safe_divide(
            np.ones((2, 3), dtype=float),
            np.ones((4,), dtype=float),
        )


def test_integrators_exports_expected_public_names() -> None:
    """Tests that integrators exposes the expected public API names."""
    import lfkit.utils.integrators as integrators

    expected = {
        "integrate_between_variable_bounds",
        "safe_divide",
        "safe_power10",
    }

    assert set(integrators.__all__) == expected


def test_integrate_between_variable_bounds_returns_float64_array() -> None:
    """Tests that variable-bound integration returns a float64 array."""
    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=0,
        upper=2,
        integrand_fn=constant_integrand,
        n_grid=64,
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_integrate_between_variable_bounds_uses_custom_error_names() -> None:
    """Tests that variable-bound integration uses custom names in validation errors."""
    with pytest.raises(ValueError, match="m_min contains NaN or infinite values"):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=np.nan,
            upper=1.0,
            integrand_fn=constant_integrand,
            lower_name="m_min",
        )


def test_integrate_between_variable_bounds_uses_custom_grid_name() -> None:
    """Tests that variable-bound integration uses the custom grid name in errors."""
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        integrate_between_variable_bounds(
            [0.1, 0.2],
            lower=0.0,
            upper=1.0,
            integrand_fn=constant_integrand,
            n_grid=1,
            n_grid_name="n_points",
        )


def test_integrate_between_variable_bounds_passes_expected_grid_shapes() -> None:
    """Tests that variable-bound integration passes expected grids to integrand."""
    y = np.array([1.0, 2.0], dtype=float)
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([2.0, 4.0], dtype=float)

    def integrand_fn(x: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """Check integration-grid inputs and return constant values."""
        assert x.shape == (5, 2)
        assert y_grid.shape == (5, 2)
        np.testing.assert_allclose(x[0], lower)
        np.testing.assert_allclose(x[-1], upper)
        np.testing.assert_allclose(y_grid[0], y)
        np.testing.assert_allclose(y_grid[-1], y)
        return np.ones_like(x, dtype=float)

    result = integrate_between_variable_bounds(
        y,
        lower=lower,
        upper=upper,
        integrand_fn=integrand_fn,
        n_grid=5,
    )

    np.testing.assert_allclose(result, np.array([2.0, 3.0]))


def test_integrate_between_variable_bounds_converts_list_integrand_output() -> None:
    """Tests that variable-bound integration converts list output to a float array."""

    def integrand_fn(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
        """Return list values broadcastable to the integration grid."""
        _ = x
        _ = y
        return [[1.0, 2.0]]

    result = integrate_between_variable_bounds(
        [0.1, 0.2],
        lower=0.0,
        upper=2.0,
        integrand_fn=integrand_fn,
        n_grid=4,
    )

    np.testing.assert_allclose(result, np.array([2.0, 4.0]))


def test_safe_divide_uses_custom_fill_value() -> None:
    """Tests that safe division uses the requested fill value."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, -1.0]),
        fill_value=-99.0,
    )

    np.testing.assert_allclose(result, np.array([1.0, -99.0, -99.0]))


def test_safe_divide_replaces_nonfinite_results() -> None:
    """Tests that safe division replaces non-finite results with fill value."""
    result = safe_divide(
        np.array([np.inf, 2.0, np.nan]),
        np.array([1.0, 1.0, 1.0]),
        fill_value=-1.0,
    )

    np.testing.assert_allclose(result, np.array([-1.0, 2.0, -1.0]))


def test_safe_divide_returns_float64_array() -> None:
    """Tests that safe division returns a float64 array."""
    result = safe_divide(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_safe_power10_matches_base_ten_power() -> None:
    """Tests that safe_power10 matches ordinary base-ten powers."""
    result = safe_power10(np.array([-1.0, 0.0, 2.0]))

    np.testing.assert_allclose(result, np.array([0.1, 1.0, 100.0]))


def test_safe_power10_clips_large_positive_exponents() -> None:
    """Tests that safe_power10 clips large positive exponents."""
    result = safe_power10(
        np.array([1.0, 5.0]),
        min_exponent=-2.0,
        max_exponent=2.0,
    )

    np.testing.assert_allclose(result, np.array([10.0, 100.0]))


def test_safe_power10_clips_large_negative_exponents() -> None:
    """Tests that safe_power10 clips large negative exponents."""
    result = safe_power10(
        np.array([-5.0, -1.0]),
        min_exponent=-2.0,
        max_exponent=2.0,
    )

    np.testing.assert_allclose(result, np.array([0.01, 0.1]))


def test_safe_power10_supports_scalar_input() -> None:
    """Tests that safe_power10 supports scalar input."""
    result = safe_power10(2.0)

    assert result.shape == ()
    assert result == pytest.approx(100.0)


def test_safe_power10_returns_float64_array() -> None:
    """Tests that safe_power10 returns a float64 array."""
    result = safe_power10(np.array([0, 1, 2]))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_safe_power10_rejects_nan_exponent() -> None:
    """Tests that safe_power10 rejects NaN exponents."""
    with pytest.raises(ValueError, match="exponent contains NaN"):
        safe_power10(np.nan)


def test_safe_power10_rejects_infinite_exponent() -> None:
    """Tests that safe_power10 rejects infinite exponents."""
    with pytest.raises(ValueError, match="exponent contains NaN or infinite values"):
        safe_power10(np.inf)
