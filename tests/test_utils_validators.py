"""Unit tests for the ``lfkit.utils.validators``."""

import numpy as np
import pytest

from lfkit.utils.validators import (
    validate_array,
    validate_luminosity_distance,
    validate_magnitude_range,
    validate_2d_binned_grid,
    validate_2d_tabulated_grid,
    validate_binned_grid,
    validate_strictly_increasing_1d,
    validate_tabulated_grid,
)


def test_validate_array_accepts_scalar() -> None:
    """Tests that a finite scalar is converted to a float array."""
    result = validate_array(1.5, name="x")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()
    assert result == pytest.approx(1.5)


def test_validate_array_accepts_list() -> None:
    """Tests that a finite list is converted to a float array."""
    result = validate_array([1.0, 2.0, 3.0], name="x")

    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64


def test_validate_array_accepts_numpy_array() -> None:
    """Tests that a finite numpy array is preserved as a float array."""
    x = np.array([1, 2, 3])

    result = validate_array(x, name="x")

    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64


def test_validate_array_accepts_negative_values_by_default() -> None:
    """Tests that negative values are allowed by default."""
    result = validate_array([-1.0, 0.0, 1.0], name="x")

    np.testing.assert_allclose(result, np.array([-1.0, 0.0, 1.0]))


def test_validate_array_rejects_negative_values_when_disallowed() -> None:
    """Tests that negative values are rejected when requested."""
    with pytest.raises(ValueError, match="x contains negative values"):
        validate_array([-1.0, 0.0, 1.0], name="x", allow_negative=False)


def test_validate_array_accepts_zero_when_negative_values_are_disallowed() -> None:
    """Tests that zero is allowed when only negative values are disallowed."""
    result = validate_array([0.0, 1.0, 2.0], name="x", allow_negative=False)

    np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0]))


def test_validate_array_rejects_nan() -> None:
    """Tests that NaN values are rejected."""
    with pytest.raises(ValueError, match="x contains NaN or infinite values"):
        validate_array([1.0, np.nan], name="x")


def test_validate_array_rejects_positive_infinity() -> None:
    """Tests that positive infinite values are rejected."""
    with pytest.raises(ValueError, match="x contains NaN or infinite values"):
        validate_array([1.0, np.inf], name="x")


def test_validate_array_rejects_negative_infinity() -> None:
    """Tests that negative infinite values are rejected."""
    with pytest.raises(ValueError, match="x contains NaN or infinite values"):
        validate_array([1.0, -np.inf], name="x")


def test_validate_array_uses_name_in_error_message() -> None:
    """Tests that the provided parameter name appears in errors."""
    with pytest.raises(ValueError, match="redshift contains NaN or infinite values"):
        validate_array([1.0, np.nan], name="redshift")


def test_validate_magnitude_range_accepts_valid_bounds() -> None:
    """Tests that valid magnitude bounds are accepted."""
    validate_magnitude_range(m_bright=-24.0, m_faint=-18.0)


def test_validate_magnitude_range_rejects_nonfinite_bright_bound() -> None:
    """Tests that non-finite bright magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_bright must be finite"):
        validate_magnitude_range(m_bright=np.nan, m_faint=-18.0)


def test_validate_magnitude_range_rejects_nonfinite_faint_bound() -> None:
    """Tests that non-finite faint magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_faint must be finite"):
        validate_magnitude_range(m_bright=-24.0, m_faint=np.inf)


def test_validate_magnitude_range_rejects_equal_bounds() -> None:
    """Tests that equal magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        validate_magnitude_range(m_bright=-20.0, m_faint=-20.0)


def test_validate_magnitude_range_rejects_reversed_bounds() -> None:
    """Tests that reversed magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        validate_magnitude_range(m_bright=-18.0, m_faint=-24.0)


def test_validate_luminosity_distance_accepts_scalar() -> None:
    """Tests that a finite positive scalar distance is accepted."""
    result = validate_luminosity_distance(100.0)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()
    assert result == pytest.approx(100.0)


def test_validate_luminosity_distance_accepts_list() -> None:
    """Tests that finite positive distance values are accepted."""
    result = validate_luminosity_distance([10.0, 100.0, 1000.0])

    np.testing.assert_allclose(result, np.array([10.0, 100.0, 1000.0]))
    assert result.dtype == np.float64


def test_validate_luminosity_distance_accepts_numpy_array() -> None:
    """Tests that a finite positive numpy array is accepted."""
    distance = np.array([10, 100, 1000])

    result = validate_luminosity_distance(distance)

    np.testing.assert_allclose(result, np.array([10.0, 100.0, 1000.0]))
    assert result.dtype == np.float64


def test_validate_luminosity_distance_rejects_zero() -> None:
    """Tests that zero luminosity distance is rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc must contain positive values",
    ):
        validate_luminosity_distance([0.0, 10.0])


def test_validate_luminosity_distance_rejects_negative_values() -> None:
    """Tests that negative luminosity distances are rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc contains negative values",
    ):
        validate_luminosity_distance([-1.0, 10.0])


def test_validate_luminosity_distance_rejects_nan() -> None:
    """Tests that NaN luminosity distances are rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc contains NaN or infinite values",
    ):
        validate_luminosity_distance([10.0, np.nan])


def test_validate_luminosity_distance_rejects_positive_infinity() -> None:
    """Tests that infinite luminosity distances are rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc contains NaN or infinite values",
    ):
        validate_luminosity_distance([10.0, np.inf])


def test_validate_luminosity_distance_rejects_negative_infinity() -> None:
    """Tests that negative infinite luminosity distances are rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc contains NaN or infinite values",
    ):
        validate_luminosity_distance([10.0, -np.inf])


def test_validate_array_preserves_multidimensional_shape() -> None:
    """Tests that validate_array preserves multidimensional input shape."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = validate_array(x, name="x")

    assert result.shape == x.shape
    np.testing.assert_allclose(result, x)


def test_validate_array_accepts_tuple_input() -> None:
    """Tests that validate_array accepts tuple input."""
    result = validate_array((1.0, 2.0, 3.0), name="x")

    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64


def test_validate_array_casts_float32_to_float64() -> None:
    """Tests that validate_array casts float32 arrays to float64."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = validate_array(x, name="x")

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, x)


def test_validate_array_accepts_empty_array() -> None:
    """Tests that validate_array accepts empty finite arrays."""
    result = validate_array([], name="x")

    assert result.dtype == np.float64
    assert result.shape == (0,)


def test_validate_array_rejects_negative_scalar_when_disallowed() -> None:
    """Tests that a negative scalar is rejected when negatives are disallowed."""
    with pytest.raises(ValueError, match="x contains negative values"):
        validate_array(-1.0, name="x", allow_negative=False)


def test_validate_luminosity_distance_preserves_multidimensional_shape() -> None:
    """Tests that luminosity-distance validation preserves multidimensional shape."""
    distance = np.array([[10.0, 100.0], [1000.0, 2000.0]])

    result = validate_luminosity_distance(distance)

    assert result.shape == distance.shape
    np.testing.assert_allclose(result, distance)


def test_validate_luminosity_distance_rejects_zero_scalar() -> None:
    """Tests that scalar zero luminosity distance is rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc must contain positive values",
    ):
        validate_luminosity_distance(0.0)


def test_validate_luminosity_distance_rejects_negative_scalar() -> None:
    """Tests that scalar negative luminosity distance is rejected."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc contains negative values",
    ):
        validate_luminosity_distance(-1.0)


def test_validate_magnitude_range_accepts_zero_width_sign_crossing_range() -> None:
    """Tests that magnitude ranges may cross zero if faint is larger than bright."""
    validate_magnitude_range(m_bright=-1.0, m_faint=1.0)


def test_validate_magnitude_range_rejects_negative_infinite_bright_bound() -> None:
    """Tests that negative infinite bright bounds are rejected."""
    with pytest.raises(ValueError, match="m_bright must be finite"):
        validate_magnitude_range(m_bright=-np.inf, m_faint=-18.0)


def test_validate_magnitude_range_rejects_negative_infinite_faint_bound() -> None:
    """Tests that negative infinite faint bounds are rejected."""
    with pytest.raises(ValueError, match="m_faint must be finite"):
        validate_magnitude_range(m_bright=-24.0, m_faint=-np.inf)


def test_validate_strictly_increasing_1d_accepts_valid_grid() -> None:
    """Tests that a valid increasing one-dimensional grid is accepted."""
    result = validate_strictly_increasing_1d([0.0, 1.0, 2.0], name="z")

    np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0]))
    assert result.dtype == np.float64


def test_validate_strictly_increasing_1d_rejects_multidimensional_input() -> None:
    """Tests that multidimensional inputs are rejected."""
    with pytest.raises(ValueError, match="z must be one-dimensional"):
        validate_strictly_increasing_1d([[0.0, 1.0], [2.0, 3.0]], name="z")


def test_validate_strictly_increasing_1d_rejects_too_few_values() -> None:
    """Tests that grids with too few values are rejected."""
    with pytest.raises(ValueError, match="z must contain at least 2 values"):
        validate_strictly_increasing_1d([0.0], name="z")


def test_validate_strictly_increasing_1d_respects_custom_min_size() -> None:
    """Tests that the custom minimum size is enforced."""
    with pytest.raises(ValueError, match="z must contain at least 3 values"):
        validate_strictly_increasing_1d([0.0, 1.0], name="z", min_size=3)


def test_validate_strictly_increasing_1d_rejects_equal_values() -> None:
    """Tests that repeated values are rejected."""
    with pytest.raises(ValueError, match="z must be strictly increasing"):
        validate_strictly_increasing_1d([0.0, 1.0, 1.0], name="z")


def test_validate_strictly_increasing_1d_rejects_decreasing_values() -> None:
    """Tests that decreasing grids are rejected."""
    with pytest.raises(ValueError, match="z must be strictly increasing"):
        validate_strictly_increasing_1d([0.0, 2.0, 1.0], name="z")


def test_validate_strictly_increasing_1d_rejects_negative_values_when_disallowed() -> None:
    """Tests that negative grid values are rejected when requested."""
    with pytest.raises(ValueError, match="z contains negative values"):
        validate_strictly_increasing_1d([-1.0, 0.0, 1.0], name="z", allow_negative=False)


def test_validate_tabulated_grid_accepts_valid_nonnegative_values() -> None:
    """Tests that a valid tabulated grid is accepted."""
    x, y = validate_tabulated_grid(
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 4.0],
        coordinate_name="x",
        values_name="y",
    )

    np.testing.assert_allclose(x, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(y, np.array([0.0, 2.0, 4.0]))


def test_validate_tabulated_grid_rejects_multidimensional_values() -> None:
    """Tests that tabulated values must be one-dimensional."""
    with pytest.raises(ValueError, match="y must be one-dimensional"):
        validate_tabulated_grid(
            [0.0, 1.0],
            [[1.0, 2.0]],
            coordinate_name="x",
            values_name="y",
        )


def test_validate_tabulated_grid_rejects_length_mismatch() -> None:
    """Tests that coordinate and value lengths must match."""
    with pytest.raises(ValueError, match="x and y must have the same length"):
        validate_tabulated_grid(
            [0.0, 1.0, 2.0],
            [1.0, 2.0],
            coordinate_name="x",
            values_name="y",
        )


def test_validate_tabulated_grid_rejects_negative_values_by_default() -> None:
    """Tests that negative tabulated values are rejected by default."""
    with pytest.raises(ValueError, match="y must be non-negative"):
        validate_tabulated_grid(
            [0.0, 1.0, 2.0],
            [1.0, -1.0, 2.0],
            coordinate_name="x",
            values_name="y",
        )


def test_validate_tabulated_grid_rejects_zero_when_positive_values_required() -> None:
    """Tests that zero values are rejected when positive values are required."""
    with pytest.raises(ValueError, match="y must be positive"):
        validate_tabulated_grid(
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            coordinate_name="x",
            values_name="y",
            positive_values=True,
        )


def test_validate_tabulated_grid_rejects_negative_coordinate_when_disallowed() -> None:
    """Tests that negative coordinates are rejected when requested."""
    with pytest.raises(ValueError, match="x contains negative values"):
        validate_tabulated_grid(
            [-1.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            coordinate_name="x",
            values_name="y",
            allow_negative_coordinate=False,
        )


def test_validate_binned_grid_accepts_valid_nonnegative_values() -> None:
    """Tests that a valid binned grid is accepted."""
    edges, values = validate_binned_grid(
        [0.0, 1.0, 2.0],
        [0.0, 2.0],
        edges_name="edges",
        values_name="counts",
    )

    np.testing.assert_allclose(edges, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(values, np.array([0.0, 2.0]))


def test_validate_binned_grid_rejects_multidimensional_values() -> None:
    """Tests that binned values must be one-dimensional."""
    with pytest.raises(ValueError, match="counts must be one-dimensional"):
        validate_binned_grid(
            [0.0, 1.0, 2.0],
            [[1.0, 2.0]],
            edges_name="edges",
            values_name="counts",
        )


def test_validate_binned_grid_rejects_wrong_edge_count() -> None:
    """Tests that bin edges must have one more value than bin values."""
    with pytest.raises(
        ValueError,
        match="edges must have one more value than counts",
    ):
        validate_binned_grid(
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            edges_name="edges",
            values_name="counts",
        )


def test_validate_binned_grid_rejects_negative_values_by_default() -> None:
    """Tests that negative binned values are rejected by default."""
    with pytest.raises(ValueError, match="counts must be non-negative"):
        validate_binned_grid(
            [0.0, 1.0, 2.0],
            [1.0, -1.0],
            edges_name="edges",
            values_name="counts",
        )


def test_validate_binned_grid_rejects_zero_when_positive_values_required() -> None:
    """Tests that zero binned values are rejected when positive values are required."""
    with pytest.raises(ValueError, match="counts must be positive"):
        validate_binned_grid(
            [0.0, 1.0, 2.0],
            [1.0, 0.0],
            edges_name="edges",
            values_name="counts",
            positive_values=True,
        )


def test_validate_binned_grid_rejects_negative_edges_when_disallowed() -> None:
    """Tests that negative bin edges are rejected when requested."""
    with pytest.raises(ValueError, match="edges contains negative values"):
        validate_binned_grid(
            [-1.0, 0.0, 1.0],
            [1.0, 2.0],
            edges_name="edges",
            values_name="counts",
            allow_negative_edges=False,
        )


def test_validate_2d_tabulated_grid_accepts_valid_values() -> None:
    """Tests that a valid two-dimensional tabulated grid is accepted."""
    x, y, values = validate_2d_tabulated_grid(
        [0.0, 1.0, 2.0],
        [10.0, 20.0],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        x_name="x",
        y_name="y",
        values_name="phi",
    )

    np.testing.assert_allclose(x, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(y, np.array([10.0, 20.0]))
    np.testing.assert_allclose(values, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_validate_2d_tabulated_grid_rejects_non_2d_values() -> None:
    """Tests that tabulated 2D values must be two-dimensional."""
    with pytest.raises(ValueError, match="phi must be two-dimensional"):
        validate_2d_tabulated_grid(
            [0.0, 1.0],
            [10.0, 20.0],
            [1.0, 2.0],
            x_name="x",
            y_name="y",
            values_name="phi",
        )


def test_validate_2d_tabulated_grid_rejects_wrong_shape() -> None:
    """Tests that tabulated 2D values must match y-by-x shape."""
    with pytest.raises(
        ValueError,
        match=r"phi must have shape \(y.size, x.size\)",
    ):
        validate_2d_tabulated_grid(
            [0.0, 1.0, 2.0],
            [10.0, 20.0],
            [[1.0, 2.0], [3.0, 4.0]],
            x_name="x",
            y_name="y",
            values_name="phi",
        )


def test_validate_2d_tabulated_grid_rejects_negative_values_by_default() -> None:
    """Tests that negative tabulated 2D values are rejected by default."""
    with pytest.raises(ValueError, match="phi must be non-negative"):
        validate_2d_tabulated_grid(
            [0.0, 1.0],
            [10.0, 20.0],
            [[1.0, -1.0], [2.0, 3.0]],
            x_name="x",
            y_name="y",
            values_name="phi",
        )


def test_validate_2d_tabulated_grid_rejects_zero_when_positive_values_required() -> None:
    """Tests that zero tabulated 2D values are rejected when positivity is required."""
    with pytest.raises(ValueError, match="phi must be positive"):
        validate_2d_tabulated_grid(
            [0.0, 1.0],
            [10.0, 20.0],
            [[1.0, 0.0], [2.0, 3.0]],
            x_name="x",
            y_name="y",
            values_name="phi",
            positive_values=True,
        )


def test_validate_2d_tabulated_grid_rejects_negative_x_when_disallowed() -> None:
    """Tests that negative x coordinates are rejected when requested."""
    with pytest.raises(ValueError, match="x contains negative values"):
        validate_2d_tabulated_grid(
            [-1.0, 0.0],
            [10.0, 20.0],
            [[1.0, 2.0], [3.0, 4.0]],
            x_name="x",
            y_name="y",
            values_name="phi",
            allow_negative_x=False,
        )


def test_validate_2d_tabulated_grid_rejects_negative_y_when_disallowed() -> None:
    """Tests that negative y coordinates are rejected when requested."""
    with pytest.raises(ValueError, match="y contains negative values"):
        validate_2d_tabulated_grid(
            [0.0, 1.0],
            [-10.0, 20.0],
            [[1.0, 2.0], [3.0, 4.0]],
            x_name="x",
            y_name="y",
            values_name="phi",
            allow_negative_y=False,
        )


def test_validate_2d_binned_grid_accepts_valid_values() -> None:
    """Tests that a valid two-dimensional binned grid is accepted."""
    x_edges, y_edges, values = validate_2d_binned_grid(
        [0.0, 1.0, 2.0],
        [10.0, 20.0, 30.0],
        [[1.0, 2.0], [3.0, 4.0]],
        x_edges_name="x_edges",
        y_edges_name="y_edges",
        values_name="counts",
    )

    np.testing.assert_allclose(x_edges, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(y_edges, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(values, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_validate_2d_binned_grid_rejects_non_2d_values() -> None:
    """Tests that binned 2D values must be two-dimensional."""
    with pytest.raises(ValueError, match="counts must be two-dimensional"):
        validate_2d_binned_grid(
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
            [1.0, 2.0],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
        )


def test_validate_2d_binned_grid_rejects_wrong_shape() -> None:
    """Tests that binned 2D values must match y-bin by x-bin shape."""
    with pytest.raises(
        ValueError,
        match=r"counts must have shape \(y_edges.size - 1, x_edges.size - 1\)",
    ):
        validate_2d_binned_grid(
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
        )


def test_validate_2d_binned_grid_rejects_negative_values_by_default() -> None:
    """Tests that negative binned 2D values are rejected by default."""
    with pytest.raises(ValueError, match="counts must be non-negative"):
        validate_2d_binned_grid(
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
            [[1.0, -1.0], [2.0, 3.0]],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
        )


def test_validate_2d_binned_grid_rejects_zero_when_positive_values_required() -> None:
    """Tests that zero binned 2D values are rejected when positivity is required."""
    with pytest.raises(ValueError, match="counts must be positive"):
        validate_2d_binned_grid(
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
            [[1.0, 0.0], [2.0, 3.0]],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
            positive_values=True,
        )


def test_validate_2d_binned_grid_rejects_negative_x_edges_when_disallowed() -> None:
    """Tests that negative x bin edges are rejected when requested."""
    with pytest.raises(ValueError, match="x_edges contains negative values"):
        validate_2d_binned_grid(
            [-1.0, 0.0, 1.0],
            [10.0, 20.0, 30.0],
            [[1.0, 2.0], [3.0, 4.0]],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
            allow_negative_x_edges=False,
        )


def test_validate_2d_binned_grid_rejects_negative_y_edges_when_disallowed() -> None:
    """Tests that negative y bin edges are rejected when requested."""
    with pytest.raises(ValueError, match="y_edges contains negative values"):
        validate_2d_binned_grid(
            [0.0, 1.0, 2.0],
            [-10.0, 20.0, 30.0],
            [[1.0, 2.0], [3.0, 4.0]],
            x_edges_name="x_edges",
            y_edges_name="y_edges",
            values_name="counts",
            allow_negative_y_edges=False,
        )


def test_validate_magnitude_range_accepts_array_bounds() -> None:
    """Tests that array magnitude bounds are accepted."""
    validate_magnitude_range(
        m_bright=np.array([-24.0, -23.0, -22.0]),
        m_faint=np.array([-20.0, -19.0, -18.0]),
    )


def test_validate_magnitude_range_accepts_broadcast_bounds() -> None:
    """Tests that scalar and array magnitude bounds can be broadcast."""
    validate_magnitude_range(
        m_bright=-24.0,
        m_faint=np.array([-22.0, -20.0, -18.0]),
    )


def test_validate_magnitude_range_rejects_array_reversed_bounds() -> None:
    """Tests that any invalid array magnitude pair is rejected."""
    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        validate_magnitude_range(
            m_bright=np.array([-24.0, -18.0, -22.0]),
            m_faint=np.array([-20.0, -23.0, -18.0]),
        )


def test_validate_magnitude_range_rejects_array_nonfinite_bounds() -> None:
    """Tests that non finite array magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_bright must be finite"):
        validate_magnitude_range(
            m_bright=np.array([-24.0, np.nan]),
            m_faint=np.array([-20.0, -18.0]),
        )

    with pytest.raises(ValueError, match="m_faint must be finite"):
        validate_magnitude_range(
            m_bright=np.array([-24.0, -23.0]),
            m_faint=np.array([-20.0, np.inf]),
        )
