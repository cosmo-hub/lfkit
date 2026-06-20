"""Unit tests for the ``lfkit.luminosity_functions.fractions``."""

from __future__ import annotations

import pytest

import numpy as np

from lfkit.luminosity_functions.fractions import (
    blue_fraction_from_luminosity_functions,
    complement_fraction_from_luminosity_functions,
    fraction_from_luminosity_functions,
    population_densities_from_luminosity_functions,
    red_blue_fractions_from_luminosity_functions,
    red_fraction_from_luminosity_functions,
)


def test_fraction_from_luminosity_functions_returns_constant_ratio() -> None:
    """Tests that LF fractions recover a constant normalization ratio."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def red_lf(absolute_mag, z):
        return 0.35 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.1, 0.5, 1.0])

    frac = fraction_from_luminosity_functions(
        z,
        red_lf,
        total_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, 0.35, rtol=1.0e-12, atol=1.0e-12)


def test_red_fraction_from_luminosity_functions_returns_expected_ratio() -> None:
    """Tests that the red fraction helper returns the expected LF ratio."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def red_lf(absolute_mag, z):
        return 0.6 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.2, 0.4, 0.8])

    frac = red_fraction_from_luminosity_functions(
        z,
        red_lf,
        total_lf,
        m_bright=-24.0,
        m_faint=-19.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, 0.6, rtol=1.0e-12, atol=1.0e-12)


def test_blue_fraction_from_luminosity_functions_returns_expected_ratio() -> None:
    """Tests that the blue fraction helper returns the expected LF ratio."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def blue_lf(absolute_mag, z):
        return 0.4 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.2, 0.4, 0.8])

    frac = blue_fraction_from_luminosity_functions(
        z,
        blue_lf,
        total_lf,
        m_bright=-24.0,
        m_faint=-19.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, 0.4, rtol=1.0e-12, atol=1.0e-12)


def test_fraction_from_luminosity_functions_returns_zero_for_zero_denominator() -> None:
    """Tests that zero denominator density gives zero fraction."""

    def numerator_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def denominator_lf(absolute_mag, z):
        return np.zeros_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    z = np.array([0.1, 0.5])

    frac = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, np.zeros_like(z), rtol=0.0, atol=0.0)


def test_red_fraction_from_luminosity_functions_accepts_scalar_redshift() -> None:
    """Tests that scalar redshift input is accepted."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def red_lf(absolute_mag, z):
        return 0.25 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    frac = red_fraction_from_luminosity_functions(
        0.5,
        red_lf,
        total_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, 0.25, rtol=1.0e-12, atol=1.0e-12)


def test_red_blue_fractions_from_luminosity_functions_return_expected_ratios() -> None:
    """Tests that red and blue LF fractions use red plus blue as the denominator."""

    def red_lf(absolute_mag, z):
        return 0.25 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    def blue_lf(absolute_mag, z):
        return 0.75 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.1, 0.5, 1.0])

    red_fraction, blue_fraction = red_blue_fractions_from_luminosity_functions(
        z,
        red_lf,
        blue_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(red_fraction, 0.25, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(blue_fraction, 0.75, rtol=1.0e-12, atol=1.0e-12)


def test_red_blue_fractions_from_luminosity_functions_zero_for_zero_total() -> None:
    """Tests that red and blue fractions are zero when red plus blue is zero."""

    def red_lf(absolute_mag, z):
        return np.zeros_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def blue_lf(absolute_mag, z):
        return np.zeros_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    z = np.array([0.1, 0.5])

    red_fraction, blue_fraction = red_blue_fractions_from_luminosity_functions(
        z,
        red_lf,
        blue_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(red_fraction, np.zeros_like(z), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(blue_fraction, np.zeros_like(z), rtol=0.0, atol=0.0)


def test_complement_fraction_from_luminosity_functions_returns_expected_value() -> None:
    """Tests that complement fractions return one minus the LF ratio."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def red_lf(absolute_mag, z):
        return 0.6 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.2, 0.4, 0.8])

    frac = complement_fraction_from_luminosity_functions(
        z,
        red_lf,
        total_lf,
        m_bright=-24.0,
        m_faint=-19.0,
        n_m=128,
    )

    np.testing.assert_allclose(frac, 0.4, rtol=1.0e-12, atol=1.0e-12)


def test_population_densities_from_luminosity_functions_returns_expected_values() -> None:
    """Tests that population densities return both densities and their sum."""

    def red_lf(absolute_mag, z):
        return 0.25 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    def blue_lf(absolute_mag, z):
        return 0.75 * np.ones_like(
            np.broadcast_arrays(absolute_mag, z)[0],
            dtype=float,
        )

    z = np.array([0.1, 0.5, 1.0])

    red_density, blue_density, total_density = population_densities_from_luminosity_functions(
        z,
        red_lf,
        blue_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    expected_width = 5.0

    np.testing.assert_allclose(red_density, 0.25 * expected_width)
    np.testing.assert_allclose(blue_density, 0.75 * expected_width)
    np.testing.assert_allclose(total_density, expected_width)


def test_fraction_from_luminosity_functions_tracks_redshift_dependence() -> None:
    """Tests that LF fractions can vary with redshift."""

    def numerator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.2 + 0.1 * z) * np.ones_like(absolute_mag, dtype=float)

    def denominator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return np.ones_like(absolute_mag, dtype=float)

    z = np.array([0.0, 0.5, 1.0])

    frac = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    expected = 0.2 + 0.1 * z

    np.testing.assert_allclose(frac, expected, rtol=1.0e-12, atol=1.0e-12)


def test_fraction_from_luminosity_functions_tracks_magnitude_limit_dependence() -> None:
    """Tests that LF fractions respond to the chosen magnitude range."""

    def numerator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return absolute_mag + 24.0

    def denominator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return np.ones_like(absolute_mag, dtype=float)

    z = 0.5

    bright_selection = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=-23.0,
        m_faint=-21.0,
        n_m=512,
    )

    faint_selection = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=-21.0,
        m_faint=-18.0,
        n_m=512,
    )

    assert faint_selection > bright_selection

    np.testing.assert_allclose(
        bright_selection,
        2.0,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        faint_selection,
        4.5,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_complement_fraction_matches_one_minus_fraction() -> None:
    """Tests that complement fractions are the complement of the LF ratio."""

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    def selected_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.3 + 0.2 * z) * np.ones_like(absolute_mag, dtype=float)

    z = np.array([0.0, 0.5, 1.0])

    fraction = fraction_from_luminosity_functions(
        z,
        selected_lf,
        total_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    complement = complement_fraction_from_luminosity_functions(
        z,
        selected_lf,
        total_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(
        complement,
        1.0 - fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_red_blue_fractions_sum_to_one_for_nonzero_total() -> None:
    """Tests that red and blue fractions sum to one when total density is nonzero."""

    def red_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.2 + 0.1 * z) * np.ones_like(absolute_mag, dtype=float)

    def blue_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.8 - 0.1 * z) * np.ones_like(absolute_mag, dtype=float)

    z = np.array([0.0, 0.5, 1.0])

    red_fraction, blue_fraction = red_blue_fractions_from_luminosity_functions(
        z,
        red_lf,
        blue_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    np.testing.assert_allclose(
        red_fraction + blue_fraction,
        np.ones_like(z),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_population_densities_are_consistent_with_fraction() -> None:
    """Tests that density ratios agree with the fraction helper."""

    def red_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.2 + 0.1 * z) * np.ones_like(absolute_mag, dtype=float)

    def total_lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    z = np.array([0.0, 0.5, 1.0])

    fraction = fraction_from_luminosity_functions(
        z,
        red_lf,
        total_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    red_density, total_density, combined_density = (
        population_densities_from_luminosity_functions(
            z,
            red_lf,
            total_lf,
            m_bright=-23.0,
            m_faint=-18.0,
            n_m=128,
        )
    )

    np.testing.assert_allclose(
        red_density / total_density,
        fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        combined_density,
        red_density + total_density,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_fraction_from_luminosity_functions_returns_finite_array() -> None:
    """Tests that array inputs return finite fraction values."""

    def numerator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (0.4 + 0.05 * z) * np.ones_like(absolute_mag, dtype=float)

    def denominator_lf(absolute_mag, z):
        absolute_mag, z = np.broadcast_arrays(absolute_mag, z)
        return (1.0 + 0.1 * z) * np.ones_like(absolute_mag, dtype=float)

    z = np.linspace(0.0, 1.0, 5)

    fraction = fraction_from_luminosity_functions(
        z,
        numerator_lf,
        denominator_lf,
        m_bright=-23.0,
        m_faint=-18.0,
        n_m=128,
    )

    assert fraction.shape == z.shape
    assert np.all(np.isfinite(fraction))


def test_fraction_from_luminosity_functions_rejects_invalid_magnitude_range() -> None:
    """Tests that the bright magnitude bound must be smaller than the faint bound."""

    def lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        fraction_from_luminosity_functions(
            0.5,
            lf,
            lf,
            m_bright=-18.0,
            m_faint=-23.0,
            n_m=128,
        )


def test_fraction_from_luminosity_functions_rejects_equal_magnitude_bounds() -> None:
    """Tests that equal bright and faint magnitude bounds are rejected."""

    def lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        fraction_from_luminosity_functions(
            0.5,
            lf,
            lf,
            m_bright=-20.0,
            m_faint=-20.0,
            n_m=128,
        )


def test_fraction_from_luminosity_functions_rejects_nonfinite_magnitude_bounds() -> None:
    """Tests that non finite magnitude bounds are rejected."""

    def lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    with pytest.raises(ValueError, match="m_bright must be finite"):
        fraction_from_luminosity_functions(
            0.5,
            lf,
            lf,
            m_bright=np.nan,
            m_faint=-18.0,
            n_m=128,
        )

    with pytest.raises(ValueError, match="m_faint must be finite"):
        fraction_from_luminosity_functions(
            0.5,
            lf,
            lf,
            m_bright=-23.0,
            m_faint=np.inf,
            n_m=128,
        )


def test_population_densities_from_luminosity_functions_rejects_invalid_magnitude_range() -> None:
    """Tests that population densities validate magnitude bounds."""

    def lf(absolute_mag, z):
        return np.ones_like(np.broadcast_arrays(absolute_mag, z)[0], dtype=float)

    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        population_densities_from_luminosity_functions(
            0.5,
            lf,
            lf,
            m_bright=-18.0,
            m_faint=-23.0,
            n_m=128,
        )
