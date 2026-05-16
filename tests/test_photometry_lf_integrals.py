"""Unit tests for the ``lfkit.photometry.lf_integrals.py`` module."""

import numpy as np
import pytest

import lfkit.photometry.lf_integrals as li


def constant_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function."""
    return np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def double_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function with amplitude two."""
    return 2.0 * np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def zero_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a zero luminosity function."""
    return np.zeros_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def linear_magnitude_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a positive LF that varies linearly with absolute magnitude."""
    return np.asarray(m_abs + 25.0, dtype=float)


def test_integrated_number_density_integrates_constant_lf() -> None:
    """Tests that finite-range integration returns the expected width."""
    result = li.integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_integrated_number_density_integrates_lf_amplitude() -> None:
    """Tests that finite-range integration preserves LF amplitude."""
    result = li.integrated_number_density(
        [0.1, 0.2],
        double_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([12.0, 12.0]))


def test_integrated_number_density_supports_array_bounds() -> None:
    """Tests that finite-range integration supports redshift-dependent bounds."""
    result = li.integrated_number_density(
        [0.1, 0.2, 0.3],
        constant_lf,
        m_bright=np.array([-24.0, -23.0, -22.0]),
        m_faint=np.array([-18.0, -18.0, -18.0]),
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 5.0, 4.0]))


def test_integrated_number_density_returns_zero_for_empty_ranges() -> None:
    """Tests that empty magnitude ranges return zero density."""
    result = li.integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=np.array([-18.0, -20.0]),
        m_faint=np.array([-20.0, -20.0]),
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_integrated_number_density_accepts_scalar_redshift() -> None:
    """Tests that finite-range integration accepts scalar redshift input."""
    result = li.integrated_number_density(
        0.1,
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    assert result.shape == ()
    assert result == pytest.approx(6.0)


def test_integrated_number_density_accepts_broadcastable_scalar_lf_output() -> None:
    """Tests that scalar luminosity function outputs are broadcast."""

    def scalar_lf(m_abs: np.ndarray, z: np.ndarray) -> float:
        """Return a scalar LF value."""
        return 2.0

    result = li.integrated_number_density(
        [0.1, 0.2],
        scalar_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([12.0, 12.0]))


def test_integrated_number_density_rejects_negative_redshift() -> None:
    """Tests that finite-range integration rejects negative redshifts."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        li.integrated_number_density(
            [-0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_small_magnitude_grid() -> None:
    """Tests that finite-range integration requires at least two grid points."""
    with pytest.raises(ValueError, match="n_m must be at least 2"):
        li.integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            n_m=1,
        )


def test_integrated_number_density_rejects_nonfinite_magnitude_lower_bound() -> None:
    """Tests that non-finite lower magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_lower contains NaN or infinite values"):
        li.integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_bright=np.nan,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_nonfinite_magnitude_upper_bound() -> None:
    """Tests that non-finite upper magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_upper contains NaN or infinite values"):
        li.integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=np.inf,
        )


def test_integrated_number_density_rejects_nonfinite_lf_values() -> None:
    """Tests that non-finite luminosity function values are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return non-finite LF values."""
        return np.full_like(m_abs, np.nan, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) returned non-finite values"):
        li.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_negative_lf_values() -> None:
    """Tests that negative luminosity function values are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return negative LF values."""
        return -np.ones_like(m_abs, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) must be non-negative"):
        li.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_unbroadcastable_lf_values() -> None:
    """Tests that unbroadcastable luminosity function outputs are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return LF values with an invalid shape."""
        return np.ones((3, 3), dtype=float)

    with pytest.raises(
        ValueError,
        match="lf\\(M, z\\) must return values broadcastable",
    ):
        li.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_lf_weighted_integral_applies_constant_weight() -> None:
    """Tests that weighted LF integration applies a constant weight."""

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return constant weights."""
        return 3.0 * np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)

    result = li.lf_weighted_integral(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        weight_fn=weight_fn,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([18.0, 18.0]))


def test_lf_weighted_integral_applies_magnitude_weight() -> None:
    """Tests that weighted LF integration supports magnitude-dependent weights."""

    def weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return a positive magnitude-dependent weight."""
        return np.asarray(m_abs + 25.0, dtype=float)

    result = li.lf_weighted_integral(
        0.1,
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        weight_fn=weight_fn,
        n_m=128,
    )

    assert result == pytest.approx(24.0)


def test_lf_weighted_integral_rejects_nonfinite_weight_values() -> None:
    """Tests that non-finite weight values are rejected."""

    def bad_weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return non-finite weights."""
        return np.full_like(m_abs, np.nan, dtype=float)

    with pytest.raises(ValueError, match="weight_fn\\(M, z\\) returned non-finite values"):
        li.lf_weighted_integral(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            weight_fn=bad_weight_fn,
        )


def test_lf_weighted_integral_rejects_negative_weight_values() -> None:
    """Tests that negative weight values are rejected."""

    def bad_weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return negative weights."""
        return -np.ones_like(m_abs, dtype=float)

    with pytest.raises(ValueError, match="weight_fn\\(M, z\\) must be non-negative"):
        li.lf_weighted_integral(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            weight_fn=bad_weight_fn,
        )


def test_lf_weighted_integral_rejects_unbroadcastable_weight_values() -> None:
    """Tests that unbroadcastable weight outputs are rejected."""

    def bad_weight_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return weights with an invalid shape."""
        return np.ones((3, 3), dtype=float)

    with pytest.raises(
        ValueError,
        match="weight_fn\\(M, z\\) must return values broadcastable",
    ):
        li.lf_weighted_integral(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            weight_fn=bad_weight_fn,
        )


def test_selection_weighted_number_density_matches_weighted_integral() -> None:
    """Tests that selection-weighted density applies the selection function."""

    def selection_fn(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return a constant fifty-percent selection."""
        return 0.5 * np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)

    result = li.selection_weighted_number_density(
        [0.1, 0.2],
        double_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        selection_fn=selection_fn,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_integrated_luminosity_density_uses_reference_magnitude() -> None:
    """Tests luminosity-density integration with relative luminosity weights."""
    result = li.integrated_luminosity_density(
        0.1,
        constant_lf,
        m_bright=0.0,
        m_faint=1.0,
        m_reference=0.0,
        n_m=1024,
    )

    expected = (1.0 - 10.0**-0.4) / (0.4 * np.log(10.0))
    assert result == pytest.approx(expected, rel=1.0e-5)


def test_integrated_luminosity_density_rejects_nonfinite_reference() -> None:
    """Tests that luminosity-density integration rejects non-finite references."""
    with pytest.raises(ValueError, match="m_reference must be finite"):
        li.integrated_luminosity_density(
            0.1,
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            m_reference=np.nan,
        )


def test_mean_luminosity_returns_luminosity_density_over_number_density() -> None:
    """Tests that mean luminosity divides luminosity density by number density."""
    result = li.mean_luminosity(
        0.1,
        constant_lf,
        m_bright=0.0,
        m_faint=1.0,
        m_reference=0.0,
        n_m=1024,
    )

    expected = (1.0 - 10.0**-0.4) / (0.4 * np.log(10.0))
    assert result == pytest.approx(expected, rel=1.0e-5)


def test_mean_luminosity_returns_zero_for_zero_number_density() -> None:
    """Tests that mean luminosity is zero when number density is zero."""
    result = li.mean_luminosity(
        [0.1, 0.2],
        zero_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        m_reference=0.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_cumulative_number_density_brighter_than_threshold() -> None:
    """Tests cumulative number density brighter than a threshold."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-21.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=True,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([3.0, 3.0]))


def test_cumulative_number_density_fainter_than_threshold() -> None:
    """Tests cumulative number density fainter than a threshold."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-21.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=False,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([3.0, 3.0]))


def test_cumulative_number_density_clips_bright_threshold_to_faint_bound() -> None:
    """Tests that brighter-than cumulative density clips to the faint bound."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-17.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=True,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_cumulative_number_density_returns_zero_for_too_bright_threshold() -> None:
    """Tests that brighter-than cumulative density is zero for too-bright cuts."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-25.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=True,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_cumulative_number_density_clips_faint_threshold_to_bright_bound() -> None:
    """Tests that fainter-than cumulative density clips to the bright bound."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-25.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=False,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_cumulative_number_density_returns_zero_for_too_faint_threshold() -> None:
    """Tests that fainter-than cumulative density is zero for too-faint cuts."""
    result = li.cumulative_number_density(
        [0.1, 0.2],
        constant_lf,
        m_threshold=-17.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=False,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_cumulative_number_density_supports_array_thresholds() -> None:
    """Tests that cumulative number density supports redshift-dependent thresholds."""
    result = li.cumulative_number_density(
        [0.1, 0.2, 0.3],
        constant_lf,
        m_threshold=np.array([-22.0, -21.0, -20.0]),
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=True,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


def test_cumulative_number_density_rejects_negative_redshift() -> None:
    """Tests that cumulative number density rejects negative redshifts."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        li.cumulative_number_density(
            [-0.1, 0.2],
            constant_lf,
            m_threshold=-21.0,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_magnitude_window_number_density_uses_absolute_bounds() -> None:
    """Tests magnitude-window density with direct absolute magnitude bounds."""
    result = li.magnitude_window_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_magnitude_window_density_supports_mixed_bounds() -> None:
    """Tests magnitude-window density with one apparent and one absolute bound."""

    def luminosity_distance_mpc_fn(z: np.ndarray) -> np.ndarray:
        """Return a constant luminosity distance."""
        return 10.0 * np.ones_like(z, dtype=float)

    result = li.magnitude_window_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        apparent_m_faint=12.0,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_magnitude_window_number_density_rejects_missing_bright_bound() -> None:
    """Tests that magnitude-window density requires a bright bound."""
    with pytest.raises(
        ValueError,
        match="Must provide either m_bright or apparent_m_bright",
    ):
        li.magnitude_window_number_density(
            0.1,
            constant_lf,
            m_faint=-18.0,
        )


def test_magnitude_window_number_density_rejects_missing_faint_bound() -> None:
    """Tests that magnitude-window density requires a faint bound."""
    with pytest.raises(
        ValueError,
        match="Must provide either m_faint or apparent_m_faint",
    ):
        li.magnitude_window_number_density(
            0.1,
            constant_lf,
            m_bright=-24.0,
        )


def test_magnitude_window_number_density_rejects_duplicate_bright_bounds() -> None:
    """Tests that absolute and apparent bright bounds cannot both be supplied."""
    with pytest.raises(
        ValueError,
        match="Provide only one of m_bright or apparent_m_bright",
    ):
        li.magnitude_window_number_density(
            0.1,
            constant_lf,
            m_bright=-24.0,
            apparent_m_bright=18.0,
            m_faint=-18.0,
        )


def test_magnitude_window_density_requires_distance_for_apparent_bounds() -> None:
    """Tests that apparent magnitude bounds require a distance callable."""
    with pytest.raises(
        ValueError,
        match="luminosity_distance_mpc_fn is required",
    ):
        li.magnitude_window_number_density(
            0.1,
            constant_lf,
            apparent_m_bright=18.0,
            m_faint=-18.0,
        )


def test_magnitude_window_number_density_applies_k_and_e_corrections() -> None:
    """Tests apparent magnitude conversion with K- and E-corrections."""

    def luminosity_distance_mpc_fn(z: np.ndarray) -> np.ndarray:
        """Return a constant luminosity distance."""
        return 10.0 * np.ones_like(z, dtype=float)

    def k_correction_fn(z: np.ndarray) -> np.ndarray:
        """Return a constant K-correction."""
        return 1.0 * np.ones_like(z, dtype=float)

    def e_correction_fn(z: np.ndarray) -> np.ndarray:
        """Return a constant E-correction."""
        return 0.5 * np.ones_like(z, dtype=float)

    result = li.magnitude_window_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        apparent_m_faint=12.0,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction_fn=k_correction_fn,
        e_correction_fn=e_correction_fn,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([5.5, 5.5]))
