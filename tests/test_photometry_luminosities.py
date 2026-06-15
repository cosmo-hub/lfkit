"""Unit tests for ``lfkit.photometry.luminosities``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.photometry.luminosities import (
    luminosity_from_magnitude,
    luminosity_ratio,
    luminosity_ratio_from_magnitudes,
    luminosity_weight_from_magnitude,
    magnitude_difference_from_luminosity_ratio,
)


def test_luminosity_ratio_is_unity_for_equal_magnitudes() -> None:
    """Tests that equal magnitudes give unit luminosity ratio."""
    magnitudes = np.array([-21.0, -20.0, -19.0])
    result = luminosity_ratio_from_magnitudes(magnitudes, magnitudes)
    np.testing.assert_allclose(result, np.ones_like(magnitudes))


def test_luminosity_ratio_matches_known_magnitude_difference() -> None:
    """Tests that a 2.5-mag difference gives a factor of 10 ratio."""
    result = luminosity_ratio_from_magnitudes(-22.5, -20.0)
    np.testing.assert_allclose(result, 10.0)


def test_luminosity_ratio_function_matches_ratio_from_magnitudes() -> None:
    """Tests that luminosity_ratio matches luminosity_ratio_from_magnitudes."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    m_star = -21.0

    result = luminosity_ratio(absolute_mag, m_star)
    expected = luminosity_ratio_from_magnitudes(absolute_mag, m_star)

    np.testing.assert_allclose(result, expected)


def test_luminosity_ratio_accepts_broadcastable_inputs() -> None:
    """Tests that luminosity_ratio accepts broadcastable magnitude inputs."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    m_star = np.array([[-21.0], [-20.0]])

    result = luminosity_ratio(absolute_mag, m_star)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(
        result,
        10.0 ** (-0.4 * (absolute_mag - m_star)),
    )


def test_luminosity_ratio_returns_scalar_array_for_scalar_inputs() -> None:
    """Tests that luminosity_ratio returns a scalar NumPy array for scalars."""
    result = luminosity_ratio(-21.0, -21.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == float
    np.testing.assert_allclose(result, 1.0)


def test_luminosity_ratio_clips_extreme_values() -> None:
    """Tests that luminosity_ratio clips extreme values to finite bounds."""
    result = luminosity_ratio(
        absolute_mag=np.array([-2000.0, 2000.0]),
        m_star=0.0,
    )

    np.testing.assert_allclose(result, np.array([1e300, 1e-300]))


def test_luminosity_ratio_from_magnitudes_accepts_broadcastable_inputs() -> None:
    """Tests that luminosity_ratio_from_magnitudes broadcasts inputs."""
    magnitude = np.array([-22.0, -21.0, -20.0])
    ref_magnitude = np.array([[-21.0], [-20.0]])

    result = luminosity_ratio_from_magnitudes(magnitude, ref_magnitude)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(
        result,
        10.0 ** (-0.4 * (magnitude - ref_magnitude)),
    )


def test_luminosity_ratio_from_magnitudes_returns_scalar_array() -> None:
    """Tests that luminosity_ratio_from_magnitudes returns scalar arrays."""
    result = luminosity_ratio_from_magnitudes(-20.0, -20.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == float
    np.testing.assert_allclose(result, 1.0)


def test_luminosity_ratio_from_magnitudes_clips_extreme_values() -> None:
    """Tests that luminosity_ratio_from_magnitudes clips extreme values."""
    result = luminosity_ratio_from_magnitudes(
        magnitude=np.array([-2000.0, 2000.0]),
        ref_magnitude=0.0,
    )

    np.testing.assert_allclose(result, np.array([1e300, 1e-300]))


def test_magnitude_difference_inverts_luminosity_ratio() -> None:
    """Tests that magnitude differences invert luminosity ratios."""
    ratios = np.array([0.1, 1.0, 10.0, 25.0])
    recovered = magnitude_difference_from_luminosity_ratio(ratios)
    reconstructed = luminosity_ratio_from_magnitudes(recovered, 0.0)
    np.testing.assert_allclose(reconstructed, ratios)


def test_magnitude_difference_returns_scalar_array_for_scalar_input() -> None:
    """Tests that magnitude_difference_from_luminosity_ratio returns scalar arrays."""
    result = magnitude_difference_from_luminosity_ratio(10.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == float
    np.testing.assert_allclose(result, -2.5)


def test_magnitude_difference_raises_for_non_positive_ratio() -> None:
    """Tests that non-positive luminosity ratios raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        magnitude_difference_from_luminosity_ratio([1.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="strictly positive"):
        magnitude_difference_from_luminosity_ratio([-1.0, 2.0])


def test_luminosity_weight_matches_expected_scaling() -> None:
    """Tests that luminosity weights follow the expected magnitude scaling."""
    magnitudes = np.array([0.0, 2.5, 5.0])
    result = luminosity_weight_from_magnitude(magnitudes)
    expected = np.array([1.0, 0.1, 0.01])
    np.testing.assert_allclose(result, expected)


def test_luminosity_weight_respects_reference_magnitude() -> None:
    """Tests that the reference magnitude shifts the luminosity zero-point."""
    result = luminosity_weight_from_magnitude(
        magnitude=np.array([1.0, 2.0]),
        reference_magnitude=1.0,
    )
    expected = np.array([1.0, 10.0 ** (-0.4)])
    np.testing.assert_allclose(result, expected)


def test_luminosity_weight_returns_scalar_array_for_scalar_input() -> None:
    """Tests that luminosity_weight_from_magnitude returns scalar arrays."""
    result = luminosity_weight_from_magnitude(0.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == float
    np.testing.assert_allclose(result, 1.0)


def test_luminosity_weight_clips_extreme_values() -> None:
    """Tests that luminosity_weight_from_magnitude clips extreme values."""
    result = luminosity_weight_from_magnitude(np.array([-2000.0, 2000.0]))

    np.testing.assert_allclose(result, np.array([1e300, 1e-300]))


def test_luminosity_from_magnitude_scales_reference_luminosity() -> None:
    """Tests that luminosity scales with the supplied reference luminosity."""
    result = luminosity_from_magnitude(
        magnitude=np.array([-21.0, -20.0]),
        reference_magnitude=-20.0,
        reference_luminosity=3.0,
    )
    expected = 3.0 * np.array([10.0**0.4, 1.0])
    np.testing.assert_allclose(result, expected)


def test_luminosity_from_magnitude_returns_scalar_array_for_scalar_input() -> None:
    """Tests that luminosity_from_magnitude returns scalar arrays."""
    result = luminosity_from_magnitude(
        magnitude=-20.0,
        reference_magnitude=-20.0,
        reference_luminosity=3.0,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == float
    np.testing.assert_allclose(result, 3.0)


def test_luminosity_from_magnitude_uses_reference_magnitude() -> None:
    """Tests that luminosity_from_magnitude uses the reference magnitude."""
    result = luminosity_from_magnitude(
        magnitude=np.array([0.0, 2.5]),
        reference_magnitude=2.5,
        reference_luminosity=4.0,
    )

    expected = np.array([40.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_luminosity_from_magnitude_raises_for_non_positive_reference_luminosity() -> None:
    """Tests that non-positive reference luminosity raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        luminosity_from_magnitude(-20.0, reference_luminosity=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        luminosity_from_magnitude(-20.0, reference_luminosity=-1.0)


def test_magnitude_difference_matches_known_ratios() -> None:
    """Tests that known luminosity ratios map to known magnitude differences."""
    ratios = np.array([0.1, 1.0, 10.0])
    expected = np.array([2.5, -0.0, -2.5])

    result = magnitude_difference_from_luminosity_ratio(ratios)

    np.testing.assert_allclose(result, expected)


def test_luminosity_from_magnitude_matches_weight_times_reference_luminosity() -> None:
    """Tests that luminosity_from_magnitude matches weight scaled by reference luminosity."""
    magnitude = np.array([-22.0, -21.0, -20.0])
    reference_magnitude = -21.0
    reference_luminosity = 5.0

    result = luminosity_from_magnitude(
        magnitude,
        reference_magnitude=reference_magnitude,
        reference_luminosity=reference_luminosity,
    )
    expected = reference_luminosity * luminosity_weight_from_magnitude(
        magnitude,
        reference_magnitude=reference_magnitude,
    )

    np.testing.assert_allclose(result, expected)


def test_luminosity_from_magnitude_accepts_broadcastable_inputs() -> None:
    """Tests that luminosity_from_magnitude accepts broadcastable magnitude inputs."""
    magnitude = np.array([-22.0, -21.0, -20.0])
    reference_magnitude = np.array([[-21.0], [-20.0]])

    result = luminosity_from_magnitude(
        magnitude,
        reference_magnitude=reference_magnitude,
        reference_luminosity=2.0,
    )

    assert result.shape == (2, 3)
    np.testing.assert_allclose(
        result,
        2.0 * 10.0 ** (-0.4 * (magnitude - reference_magnitude)),
    )


def test_luminosity_outputs_are_finite_for_extreme_magnitudes() -> None:
    """Tests that luminosity helpers return finite values for extreme magnitudes."""
    magnitudes = np.array([-2000.0, 0.0, 2000.0])

    ratio = luminosity_ratio_from_magnitudes(magnitudes, 0.0)
    weight = luminosity_weight_from_magnitude(magnitudes)
    luminosity = luminosity_from_magnitude(magnitudes)

    assert np.all(np.isfinite(ratio))
    assert np.all(np.isfinite(weight))
    assert np.all(np.isfinite(luminosity))
