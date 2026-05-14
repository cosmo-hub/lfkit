"""Unit tests for ``lfkit.utils.evaluators.py``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.evaluators import (
    evaluate_lf_on_grid,
    evaluate_non_negative_redshift_callable,
    evaluate_optional_redshift_callable,
    evaluate_positive_redshift_callable,
    evaluate_weight_on_grid,
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


def test_evaluate_lf_on_grid_accepts_matching_shape() -> None:
    """Tests that LF grid evaluation accepts values with matching shape."""
    m_grid = np.array(
        [
            [-24.0, -23.0],
            [-22.0, -21.0],
        ],
        dtype=float,
    )
    z_grid = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2],
        ],
        dtype=float,
    )

    def lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return LF values with the same shape as the input grid."""
        return np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)

    result = evaluate_lf_on_grid(
        lf,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    np.testing.assert_allclose(result, np.ones_like(m_grid))


def test_evaluate_lf_on_grid_broadcasts_scalar_output() -> None:
    """Tests that scalar LF output is broadcast to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def lf(m_abs: np.ndarray, z: np.ndarray) -> float:
        """Return a scalar LF value."""
        _ = m_abs
        _ = z
        return 2.0

    result = evaluate_lf_on_grid(
        lf,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    np.testing.assert_allclose(result, 2.0 * np.ones_like(m_grid))


def test_evaluate_lf_on_grid_broadcasts_column_output() -> None:
    """Tests that broadcastable LF output is expanded to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return one LF value per redshift column."""
        _ = m_abs
        _ = z
        return np.array([[1.0, 2.0]], dtype=float)

    result = evaluate_lf_on_grid(
        lf,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    expected = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)


def test_evaluate_lf_on_grid_rejects_unbroadcastable_output() -> None:
    """Tests that LF grid output must be broadcastable to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return LF values with an invalid shape."""
        _ = m_abs
        _ = z
        return np.ones((4, 4), dtype=float)

    with pytest.raises(
        ValueError,
        match="lf\\(M, z\\) must return values broadcastable",
    ):
        evaluate_lf_on_grid(
            lf,
            m_grid=m_grid,
            z_grid=z_grid,
        )


def test_evaluate_lf_on_grid_rejects_nonfinite_values() -> None:
    """Tests that LF grid values must be finite."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return LF values containing NaN."""
        _ = z
        return np.full_like(m_abs, np.nan, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) returned non-finite values"):
        evaluate_lf_on_grid(
            lf,
            m_grid=m_grid,
            z_grid=z_grid,
        )


def test_evaluate_lf_on_grid_rejects_negative_values() -> None:
    """Tests that LF grid values must be non-negative."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return negative LF values."""
        _ = z
        return -np.ones_like(m_abs, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) must be non-negative"):
        evaluate_lf_on_grid(
            lf,
            m_grid=m_grid,
            z_grid=z_grid,
        )


def test_evaluate_weight_on_grid_accepts_matching_shape() -> None:
    """Tests that weight grid evaluation accepts values with matching shape."""
    m_grid = np.array(
        [
            [-24.0, -23.0],
            [-22.0, -21.0],
        ],
        dtype=float,
    )
    z_grid = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2],
        ],
        dtype=float,
    )

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return weights with the same shape as the input grid."""
        return np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)

    result = evaluate_weight_on_grid(
        weight_fn,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    np.testing.assert_allclose(result, np.ones_like(m_grid))


def test_evaluate_weight_on_grid_broadcasts_scalar_output() -> None:
    """Tests that scalar weight output is broadcast to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> float:
        """Return a scalar weight value."""
        _ = m_abs
        _ = z
        return 0.5

    result = evaluate_weight_on_grid(
        weight_fn,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    np.testing.assert_allclose(result, 0.5 * np.ones_like(m_grid))


def test_evaluate_weight_on_grid_broadcasts_column_output() -> None:
    """Tests that broadcastable weight output is expanded to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return one weight value per redshift column."""
        _ = m_abs
        _ = z
        return np.array([[0.25, 0.75]], dtype=float)

    result = evaluate_weight_on_grid(
        weight_fn,
        m_grid=m_grid,
        z_grid=z_grid,
    )

    expected = np.array(
        [
            [0.25, 0.75],
            [0.25, 0.75],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)


def test_evaluate_weight_on_grid_rejects_unbroadcastable_output() -> None:
    """Tests that weight output must be broadcastable to the grid shape."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return weights with an invalid shape."""
        _ = m_abs
        _ = z
        return np.ones((4, 4), dtype=float)

    with pytest.raises(
        ValueError,
        match="weight_fn\\(M, z\\) must return values broadcastable",
    ):
        evaluate_weight_on_grid(
            weight_fn,
            m_grid=m_grid,
            z_grid=z_grid,
        )


def test_evaluate_weight_on_grid_rejects_nonfinite_values() -> None:
    """Tests that weight grid values must be finite."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return weights containing infinity."""
        _ = z
        return np.full_like(m_abs, np.inf, dtype=float)

    with pytest.raises(
        ValueError,
        match="weight_fn\\(M, z\\) returned non-finite values",
    ):
        evaluate_weight_on_grid(
            weight_fn,
            m_grid=m_grid,
            z_grid=z_grid,
        )


def test_evaluate_weight_on_grid_rejects_negative_values() -> None:
    """Tests that weight grid values must be non-negative."""
    m_grid = np.ones((3, 2), dtype=float)
    z_grid = np.ones((3, 2), dtype=float)

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return negative weights."""
        _ = z
        return -np.ones_like(m_abs, dtype=float)

    with pytest.raises(ValueError, match="weight_fn\\(M, z\\) must be non-negative"):
        evaluate_weight_on_grid(
            weight_fn,
            m_grid=m_grid,
            z_grid=z_grid,
        )
