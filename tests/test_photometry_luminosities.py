"""Unit tests for ``lfkit.photometry.luminosity_function.py``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma, gammaincc

from lfkit.photometry.luminosities import (
    luminosity_from_magnitude,
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


def test_magnitude_difference_inverts_luminosity_ratio() -> None:
    """Tests that magnitude differences invert luminosity ratios."""
    ratios = np.array([0.1, 1.0, 10.0, 25.0])
    recovered = magnitude_difference_from_luminosity_ratio(ratios)
    reconstructed = luminosity_ratio_from_magnitudes(recovered, 0.0)
    np.testing.assert_allclose(reconstructed, ratios)


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


def test_luminosity_from_magnitude_scales_reference_luminosity() -> None:
    """Tests that luminosity scales with the supplied reference luminosity."""
    result = luminosity_from_magnitude(
        magnitude=np.array([-21.0, -20.0]),
        reference_magnitude=-20.0,
        reference_luminosity=3.0,
    )
    expected = 3.0 * np.array([10.0 ** 0.4, 1.0])
    np.testing.assert_allclose(result, expected)


def test_luminosity_from_magnitude_raises_for_non_positive_reference_luminosity() -> None:
    """Tests that non-positive reference luminosity raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        luminosity_from_magnitude(-20.0, reference_luminosity=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        luminosity_from_magnitude(-20.0, reference_luminosity=-1.0)
