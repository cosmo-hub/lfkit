"""Unit tests for the ``lfkit.utils.validators``."""

import numpy as np
import pytest

from lfkit.utils.validators import (
    validate_array,
    validate_luminosity_distance,
    validate_magnitude_range,
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
