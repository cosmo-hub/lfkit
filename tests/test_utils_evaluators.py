"""Unit tests for ``lfkit.utils.evaluators.py``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.evaluators import (
    evaluate_non_negative_redshift_callable,
    evaluate_optional_redshift_callable,
    evaluate_positive_redshift_callable,
)


def test_evaluate_optional_redshift_callable_returns_none() -> None:
    """Tests that an optional callable returns None when the callable is None."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    result = evaluate_optional_redshift_callable(
        None,
        z,
        name="test_fn",
    )

    assert result is None


def test_evaluate_optional_redshift_callable_evaluates_callable() -> None:
    """Tests that an optional callable is evaluated when provided."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return a simple redshift-dependent array."""
        return 1.0 + z_arr

    result = evaluate_optional_redshift_callable(
        fn,
        z,
        name="test_fn",
    )

    expected = np.array([1.0, 1.5, 2.0])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_evaluate_optional_redshift_callable_converts_list_output() -> None:
    """Tests that callable list output is converted to a float array."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> list[float]:
        """Return a list with the same length as the redshift array."""
        _ = z_arr
        return [1.0, 2.0, 3.0]

    result = evaluate_optional_redshift_callable(
        fn,
        z,
        name="test_fn",
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_evaluate_optional_redshift_callable_rejects_wrong_shape() -> None:
    """Tests that optional callable output must match the redshift shape."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return an array with the wrong shape."""
        _ = z_arr
        return np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="test_fn\\(z\\) must return an array"):
        evaluate_optional_redshift_callable(
            fn,
            z,
            name="test_fn",
        )


def test_evaluate_optional_redshift_callable_rejects_scalar_output() -> None:
    """Tests that scalar callable output is rejected for array redshift input."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> float:
        """Return a scalar instead of an array."""
        _ = z_arr
        return 1.0

    with pytest.raises(ValueError, match="test_fn\\(z\\) must return an array"):
        evaluate_optional_redshift_callable(
            fn,
            z,
            name="test_fn",
        )


def test_evaluate_optional_redshift_callable_accepts_scalar_redshift_shape() -> None:
    """Tests that scalar-shaped redshift input accepts scalar-shaped output."""
    z = np.asarray(0.5, dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return a scalar-shaped array."""
        return np.asarray(1.0 + z_arr, dtype=float)

    result = evaluate_optional_redshift_callable(
        fn,
        z,
        name="test_fn",
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()
    assert result == pytest.approx(1.5)


def test_evaluate_optional_redshift_callable_rejects_nan() -> None:
    """Tests that optional callable output cannot contain NaN values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing NaN."""
        _ = z_arr
        return np.array([1.0, np.nan, 2.0], dtype=float)

    with pytest.raises(ValueError, match="test_fn\\(z\\) returned non-finite values"):
        evaluate_optional_redshift_callable(
            fn,
            z,
            name="test_fn",
        )


def test_evaluate_optional_redshift_callable_rejects_infinite_value() -> None:
    """Tests that optional callable output cannot contain infinite values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing infinity."""
        _ = z_arr
        return np.array([1.0, np.inf, 2.0], dtype=float)

    with pytest.raises(ValueError, match="test_fn\\(z\\) returned non-finite values"):
        evaluate_optional_redshift_callable(
            fn,
            z,
            name="test_fn",
        )


def test_evaluate_positive_redshift_callable_accepts_positive_values() -> None:
    """Tests that positive callable output is accepted."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return positive redshift-dependent values."""
        return 1.0 + z_arr

    result = evaluate_positive_redshift_callable(
        fn,
        z,
        name="distance",
    )

    expected = np.array([1.0, 1.5, 2.0])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_evaluate_positive_redshift_callable_rejects_zero_value() -> None:
    """Tests that positive callable output cannot contain zero values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing zero."""
        _ = z_arr
        return np.array([1.0, 0.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="distance\\(z\\) must return positive values"):
        evaluate_positive_redshift_callable(
            fn,
            z,
            name="distance",
        )


def test_evaluate_positive_redshift_callable_rejects_negative_value() -> None:
    """Tests that positive callable output cannot contain negative values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing a negative value."""
        _ = z_arr
        return np.array([1.0, -0.5, 2.0], dtype=float)

    with pytest.raises(ValueError, match="distance\\(z\\) must return positive values"):
        evaluate_positive_redshift_callable(
            fn,
            z,
            name="distance",
        )


def test_evaluate_positive_redshift_callable_rejects_wrong_shape() -> None:
    """Tests that positive callable output must match the redshift shape."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return an array with the wrong shape."""
        _ = z_arr
        return np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="distance\\(z\\) must return an array"):
        evaluate_positive_redshift_callable(
            fn,
            z,
            name="distance",
        )


def test_evaluate_positive_redshift_callable_rejects_nonfinite_value() -> None:
    """Tests that positive callable output must be finite."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing infinity."""
        _ = z_arr
        return np.array([1.0, np.inf, 2.0], dtype=float)

    with pytest.raises(ValueError, match="distance\\(z\\) returned non-finite values"):
        evaluate_positive_redshift_callable(
            fn,
            z,
            name="distance",
        )


def test_evaluate_non_negative_redshift_callable_accepts_positive_values() -> None:
    """Tests that non-negative callable output accepts positive values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return positive redshift-dependent values."""
        return 1.0 + z_arr

    result = evaluate_non_negative_redshift_callable(
        fn,
        z,
        name="weight",
    )

    expected = np.array([1.0, 1.5, 2.0])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_evaluate_non_negative_redshift_callable_accepts_zero_value() -> None:
    """Tests that non-negative callable output accepts zero values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing zero."""
        _ = z_arr
        return np.array([0.0, 1.0, 2.0], dtype=float)

    result = evaluate_non_negative_redshift_callable(
        fn,
        z,
        name="weight",
    )

    np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0]))


def test_evaluate_non_negative_redshift_callable_rejects_negative_value() -> None:
    """Tests that non-negative callable output cannot contain negative values."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing a negative value."""
        _ = z_arr
        return np.array([1.0, -0.5, 2.0], dtype=float)

    with pytest.raises(
        ValueError,
        match="weight\\(z\\) must return non-negative values",
    ):
        evaluate_non_negative_redshift_callable(
            fn,
            z,
            name="weight",
        )


def test_evaluate_non_negative_redshift_callable_rejects_wrong_shape() -> None:
    """Tests that non-negative callable output must match the redshift shape."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return an array with the wrong shape."""
        _ = z_arr
        return np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="weight\\(z\\) must return an array"):
        evaluate_non_negative_redshift_callable(
            fn,
            z,
            name="weight",
        )


def test_evaluate_non_negative_redshift_callable_rejects_nonfinite_value() -> None:
    """Tests that non-negative callable output must be finite."""
    z = np.array([0.0, 0.5, 1.0], dtype=float)

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return output containing NaN."""
        _ = z_arr
        return np.array([1.0, np.nan, 2.0], dtype=float)

    with pytest.raises(ValueError, match="weight\\(z\\) returned non-finite values"):
        evaluate_non_negative_redshift_callable(
            fn,
            z,
            name="weight",
        )


def test_evaluate_non_negative_redshift_callable_preserves_input_shape() -> None:
    """Tests that callable output preserves multi-dimensional redshift shape."""
    z = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
        ],
        dtype=float,
    )

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return values with the same two-dimensional shape as redshift."""
        return 1.0 + z_arr

    result = evaluate_non_negative_redshift_callable(
        fn,
        z,
        name="weight",
    )

    expected = np.array(
        [
            [1.0, 1.5],
            [2.0, 2.5],
        ],
        dtype=float,
    )
    assert result.shape == z.shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_evaluate_positive_redshift_callable_preserves_input_shape() -> None:
    """Tests that positive callable output preserves multi-dimensional shape."""
    z = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
        ],
        dtype=float,
    )

    def fn(z_arr: np.ndarray) -> np.ndarray:
        """Return positive values with the same shape as redshift."""
        return 2.0 + z_arr

    result = evaluate_positive_redshift_callable(
        fn,
        z,
        name="distance",
    )

    expected = np.array(
        [
            [2.0, 2.5],
            [3.0, 3.5],
        ],
        dtype=float,
    )
    assert result.shape == z.shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)
