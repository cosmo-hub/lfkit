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
    sample_schechter_luminosity,
    schechter_cumulative_number_density_luminosity,
    schechter_luminosity_density,
    schechter_mean_luminosity,
    schechter_selection_function,
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


def test_schechter_cumulative_number_density_matches_analytic_expression() -> None:
    """Tests that cumulative number density matches the incomplete-gamma form."""
    luminosity_min = np.array([0.0, 0.5, 1.0, 3.0])
    phi_star = 2.0
    l_star = 4.0
    alpha = -0.2

    s = alpha + 1.0
    x_min = luminosity_min / l_star
    expected = phi_star * gamma(s) * gammaincc(s, x_min)

    result = schechter_cumulative_number_density_luminosity(
        luminosity_min,
        phi_star=phi_star,
        l_star=l_star,
        alpha=alpha,
    )
    np.testing.assert_allclose(result, expected)


def test_schechter_cumulative_number_density_at_zero_threshold_is_total_density() -> None:
    """Tests that zero threshold returns the total number density."""
    phi_star = 1.7
    l_star = 2.3
    alpha = -0.4
    expected = phi_star * gamma(alpha + 1.0)

    result = schechter_cumulative_number_density_luminosity(
        0.0,
        phi_star=phi_star,
        l_star=l_star,
        alpha=alpha,
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("luminosity_min", "phi_star", "l_star", "alpha"),
    [
        (-1.0, 1.0, 2.0, -0.5),
        (0.0, -1.0, 2.0, -0.5),
        (0.0, 1.0, 0.0, -0.5),
        (0.0, 1.0, 2.0, -1.0),
        (0.0, 1.0, 2.0, -1.5),
    ],
)
def test_schechter_cumulative_number_density_rejects_invalid_inputs(
    luminosity_min: float,
    phi_star: float,
    l_star: float,
    alpha: float,
) -> None:
    """Tests that invalid cumulative-density inputs raise ValueError."""
    with pytest.raises(ValueError):
        schechter_cumulative_number_density_luminosity(
            luminosity_min,
            phi_star=phi_star,
            l_star=l_star,
            alpha=alpha,
        )


def test_schechter_luminosity_density_matches_analytic_formula() -> None:
    """Tests that luminosity density matches the Schechter analytic formula."""
    phi_star = 1.5
    l_star = 3.0
    alpha = -0.7
    expected = phi_star * l_star * gamma(alpha + 2.0)

    result = schechter_luminosity_density(
        phi_star=phi_star,
        l_star=l_star,
        alpha=alpha,
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("phi_star", "l_star", "alpha"),
    [
        (-1.0, 2.0, -0.5),
        (1.0, 0.0, -0.5),
        (1.0, 2.0, -2.0),
        (1.0, 2.0, -3.0),
    ],
)
def test_schechter_luminosity_density_rejects_invalid_inputs(
    phi_star: float,
    l_star: float,
    alpha: float,
) -> None:
    """Tests that invalid luminosity-density inputs raise ValueError."""
    with pytest.raises(ValueError):
        schechter_luminosity_density(
            phi_star=phi_star,
            l_star=l_star,
            alpha=alpha,
        )


def test_schechter_mean_luminosity_matches_simplified_expression() -> None:
    """Tests that mean luminosity equals l_star times alpha plus one."""
    l_star = 5.0
    alpha = 0.3
    expected = l_star * (alpha + 1.0)

    result = schechter_mean_luminosity(
        l_star=l_star,
        alpha=alpha,
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("l_star", "alpha"),
    [
        (0.0, -0.5),
        (-1.0, -0.5),
        (1.0, -1.0),
        (1.0, -1.2),
    ],
)
def test_schechter_mean_luminosity_rejects_invalid_inputs(
    l_star: float,
    alpha: float,
) -> None:
    """Tests that invalid mean-luminosity inputs raise ValueError."""
    with pytest.raises(ValueError):
        schechter_mean_luminosity(
            l_star=l_star,
            alpha=alpha,
        )


def test_sample_schechter_luminosity_is_reproducible_with_seeded_rng() -> None:
    """Tests that seeded sampling is reproducible."""
    rng_1 = np.random.default_rng(12345)
    rng_2 = np.random.default_rng(12345)

    sample_1 = sample_schechter_luminosity(
        8,
        l_star=2.0,
        alpha=0.4,
        rng=rng_1,
    )
    sample_2 = sample_schechter_luminosity(
        8,
        l_star=2.0,
        alpha=0.4,
        rng=rng_2,
    )

    np.testing.assert_allclose(sample_1, sample_2)


def test_sample_schechter_luminosity_returns_positive_samples_with_requested_shape() -> None:
    """Tests that sampled luminosities are positive and follow the requested shape."""
    rng = np.random.default_rng(7)
    samples = sample_schechter_luminosity(
        (4, 3),
        l_star=1.5,
        alpha=0.2,
        rng=rng,
    )

    assert samples.shape == (4, 3)
    assert np.all(samples > 0.0)


def test_sample_schechter_luminosity_has_correct_mean_in_large_sample() -> None:
    """Tests that sampled luminosities recover the analytic mean on average."""
    rng = np.random.default_rng(42)
    l_star = 2.5
    alpha = 0.4
    expected_mean = schechter_mean_luminosity(l_star=l_star, alpha=alpha)

    samples = sample_schechter_luminosity(
        200_000,
        l_star=l_star,
        alpha=alpha,
        rng=rng,
    )

    assert np.isclose(np.mean(samples), expected_mean, rtol=2e-2)


@pytest.mark.parametrize(
    ("l_star", "alpha"),
    [
        (0.0, 0.1),
        (-1.0, 0.1),
        (1.0, -1.0),
        (1.0, -1.3),
    ],
)
def test_sample_schechter_luminosity_rejects_invalid_inputs(
    l_star: float,
    alpha: float,
) -> None:
    """Tests that invalid sampling inputs raise ValueError."""
    with pytest.raises(ValueError):
        sample_schechter_luminosity(
            5,
            l_star=l_star,
            alpha=alpha,
        )


def test_schechter_selection_function_matches_regularized_incomplete_gamma() -> None:
    """Tests that the selection function matches the regularized gamma form."""
    luminosity_min = np.array([0.0, 0.5, 1.0, 2.0])
    l_star = 2.0
    alpha = -0.3
    s = alpha + 1.0
    expected = gammaincc(s, luminosity_min / l_star)

    result = schechter_selection_function(
        luminosity_min,
        l_star=l_star,
        alpha=alpha,
    )
    np.testing.assert_allclose(result, expected)


def test_schechter_selection_function_is_one_at_zero_and_decreases_with_threshold() -> None:
    """Tests that the selection function starts at one and decreases monotonically."""
    thresholds = np.array([0.0, 0.3, 1.0, 3.0, 5.0])
    result = schechter_selection_function(
        thresholds,
        l_star=2.0,
        alpha=-0.2,
    )

    np.testing.assert_allclose(result[0], 1.0)
    assert np.all(result[:-1] >= result[1:])
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_selection_function_matches_cumulative_density_ratio() -> None:
    """Tests that selection equals cumulative density divided by total density."""
    luminosity_min = np.array([0.0, 0.7, 1.5, 4.0])
    phi_star = 3.0
    l_star = 2.5
    alpha = -0.1

    cumulative = schechter_cumulative_number_density_luminosity(
        luminosity_min,
        phi_star=phi_star,
        l_star=l_star,
        alpha=alpha,
    )
    total = schechter_cumulative_number_density_luminosity(
        0.0,
        phi_star=phi_star,
        l_star=l_star,
        alpha=alpha,
    )
    selection = schechter_selection_function(
        luminosity_min,
        l_star=l_star,
        alpha=alpha,
    )

    np.testing.assert_allclose(selection, cumulative / total)


@pytest.mark.parametrize(
    ("luminosity_min", "l_star", "alpha"),
    [
        (-1.0, 1.0, -0.5),
        (0.0, 0.0, -0.5),
        (0.0, -1.0, -0.5),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, -2.0),
    ],
)
def test_schechter_selection_function_rejects_invalid_inputs(
    luminosity_min: float,
    l_star: float,
    alpha: float,
) -> None:
    """Tests that invalid selection-function inputs raise ValueError."""
    with pytest.raises(ValueError):
        schechter_selection_function(
            luminosity_min,
            l_star=l_star,
            alpha=alpha,
        )
