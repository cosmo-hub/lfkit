"""Unit tests for ``lfkit.luminosity_functions.models.non_parametric``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.non_parametric import (
    binned_lf,
    distance_binned_lf,
    distance_tabulated_lf,
    redshift_binned_lf,
    redshift_tabulated_lf,
    tabulated_lf,
)


MAG_GRID = np.array([-24.0, -22.0, -20.0, -18.0])
PHI_GRID = np.array([1.0e-5, 5.0e-4, 1.0e-3, 2.0e-4])

MAG_EDGES = np.array([-24.0, -22.0, -20.0, -18.0])
PHI_BINS = np.array([1.0e-5, 5.0e-4, 1.0e-3])

Z_GRID = np.array([0.1, 0.5, 1.0])
Z_EDGES = np.array([0.1, 0.5, 1.0, 1.5])

DIST_GRID = np.array([100.0, 500.0, 1000.0])
DIST_EDGES = np.array([100.0, 500.0, 1000.0, 1500.0])

PHI_2D_GRID = np.array(
    [
        [1.0e-5, 5.0e-4, 1.0e-3, 2.0e-4],
        [2.0e-5, 6.0e-4, 1.1e-3, 3.0e-4],
        [3.0e-5, 7.0e-4, 1.2e-3, 4.0e-4],
    ]
)

PHI_2D_BINS = np.array(
    [
        [1.0e-5, 5.0e-4, 1.0e-3],
        [2.0e-5, 6.0e-4, 1.1e-3],
        [3.0e-5, 7.0e-4, 1.2e-3],
    ]
)


def test_tabulated_lf_interpolates_linearly() -> None:
    """Tests that tabulated_lf linearly interpolates between grid values."""
    result = tabulated_lf(
        -21.0,
        magnitude_grid=MAG_GRID,
        phi_grid=PHI_GRID,
    )

    assert np.asarray(result).shape == ()
    assert result == pytest.approx(7.5e-4)


def test_tabulated_lf_accepts_array_input() -> None:
    """Tests that tabulated_lf preserves array input shape."""
    absolute_mag = np.array([-23.0, -21.0, -19.0])

    result = tabulated_lf(
        absolute_mag,
        magnitude_grid=MAG_GRID,
        phi_grid=PHI_GRID,
    )

    assert result.shape == absolute_mag.shape
    np.testing.assert_allclose(result, np.array([2.55e-4, 7.5e-4, 6.0e-4]))


def test_tabulated_lf_uses_fill_value_outside_grid() -> None:
    """Tests that tabulated_lf returns fill_value outside the magnitude grid."""
    result = tabulated_lf(
        np.array([-25.0, -21.0, -17.0]),
        magnitude_grid=MAG_GRID,
        phi_grid=PHI_GRID,
        fill_value=9.0,
    )

    np.testing.assert_allclose(result, np.array([9.0, 7.5e-4, 9.0]))


def test_tabulated_lf_interpolates_in_log_phi() -> None:
    """Tests that tabulated_lf can interpolate in log10(phi)."""
    result = tabulated_lf(
        -21.0,
        magnitude_grid=np.array([-22.0, -20.0]),
        phi_grid=np.array([1.0e-4, 1.0e-2]),
        log_phi=True,
    )

    assert result == pytest.approx(1.0e-3)


def test_tabulated_lf_log_phi_zero_fill_value_outside_grid() -> None:
    """Tests that log interpolation supports zero fill_value."""
    result = tabulated_lf(
        np.array([-25.0, -21.0]),
        magnitude_grid=np.array([-22.0, -20.0]),
        phi_grid=np.array([1.0e-4, 1.0e-2]),
        fill_value=0.0,
        log_phi=True,
    )

    np.testing.assert_allclose(result, np.array([0.0, 1.0e-3]))


def test_tabulated_lf_rejects_negative_fill_value() -> None:
    """Tests that tabulated_lf rejects negative fill_value."""
    with pytest.raises(ValueError, match="fill_value must be non-negative"):
        tabulated_lf(
            -21.0,
            magnitude_grid=MAG_GRID,
            phi_grid=PHI_GRID,
            fill_value=-1.0,
        )


def test_tabulated_lf_rejects_nonfinite_fill_value() -> None:
    """Tests that tabulated_lf rejects non-finite fill_value."""
    with pytest.raises(ValueError, match="fill_value must be finite"):
        tabulated_lf(
            -21.0,
            magnitude_grid=MAG_GRID,
            phi_grid=PHI_GRID,
            fill_value=np.inf,
        )


def test_tabulated_lf_rejects_nonpositive_values_for_log_phi() -> None:
    """Tests that log interpolation requires positive LF values."""
    with pytest.raises(ValueError, match="phi_grid must be positive"):
        tabulated_lf(
            -21.0,
            magnitude_grid=MAG_GRID,
            phi_grid=np.array([1.0e-5, 0.0, 1.0e-3, 2.0e-4]),
            log_phi=True,
        )


def test_tabulated_lf_rejects_mismatched_grid_lengths() -> None:
    """Tests that tabulated_lf rejects mismatched grid and value lengths."""
    with pytest.raises(ValueError, match="magnitude_grid and phi_grid"):
        tabulated_lf(
            -21.0,
            magnitude_grid=MAG_GRID,
            phi_grid=np.array([1.0, 2.0]),
        )


def test_binned_lf_returns_piecewise_constant_values() -> None:
    """Tests that binned_lf returns the value assigned to each bin."""
    result = binned_lf(
        np.array([-23.0, -21.0, -19.0]),
        magnitude_bin_edges=MAG_EDGES,
        phi_bin_values=PHI_BINS,
    )

    np.testing.assert_allclose(result, PHI_BINS)


def test_binned_lf_uses_fill_value_outside_bins() -> None:
    """Tests that binned_lf returns fill_value outside the bin range."""
    result = binned_lf(
        np.array([-25.0, -23.0, -17.0]),
        magnitude_bin_edges=MAG_EDGES,
        phi_bin_values=PHI_BINS,
        fill_value=7.0,
    )

    np.testing.assert_allclose(result, np.array([7.0, 1.0e-5, 7.0]))


def test_binned_lf_treats_upper_edge_as_outside() -> None:
    """Tests that the final bin edge is treated as outside the binned range."""
    result = binned_lf(
        np.array([-24.0, -18.0]),
        magnitude_bin_edges=MAG_EDGES,
        phi_bin_values=PHI_BINS,
        fill_value=9.0,
    )

    np.testing.assert_allclose(result, np.array([1.0e-5, 9.0]))


def test_binned_lf_rejects_negative_fill_value() -> None:
    """Tests that binned_lf rejects negative fill_value."""
    with pytest.raises(ValueError, match="fill_value must be non-negative"):
        binned_lf(
            -21.0,
            magnitude_bin_edges=MAG_EDGES,
            phi_bin_values=PHI_BINS,
            fill_value=-1.0,
        )


def test_binned_lf_rejects_nonfinite_fill_value() -> None:
    """Tests that binned_lf rejects non-finite fill_value."""
    with pytest.raises(ValueError, match="fill_value must be finite"):
        binned_lf(
            -21.0,
            magnitude_bin_edges=MAG_EDGES,
            phi_bin_values=PHI_BINS,
            fill_value=np.nan,
        )


def test_binned_lf_rejects_wrong_number_of_bin_values() -> None:
    """Tests that binned_lf requires one fewer value than bin edges."""
    with pytest.raises(ValueError, match="magnitude_bin_edges must have one more"):
        binned_lf(
            -21.0,
            magnitude_bin_edges=MAG_EDGES,
            phi_bin_values=np.array([1.0, 2.0]),
        )


def test_redshift_tabulated_lf_interpolates_in_magnitude_and_redshift() -> None:
    """Tests bilinear interpolation for redshift_tabulated_lf."""
    result = redshift_tabulated_lf(
        -21.0,
        0.3,
        magnitude_grid=MAG_GRID,
        redshift_grid=Z_GRID,
        phi_grid=PHI_2D_GRID,
    )

    assert np.asarray(result).shape == ()
    assert result == pytest.approx(8.0e-4)


def test_redshift_tabulated_lf_accepts_broadcastable_inputs() -> None:
    """Tests that redshift_tabulated_lf broadcasts magnitude and redshift."""
    result = redshift_tabulated_lf(
        np.array([-21.0, -21.0]),
        0.3,
        magnitude_grid=MAG_GRID,
        redshift_grid=Z_GRID,
        phi_grid=PHI_2D_GRID,
    )

    assert result.shape == (2,)
    np.testing.assert_allclose(result, np.array([8.0e-4, 8.0e-4]))


def test_redshift_tabulated_lf_uses_fill_value_outside_grid() -> None:
    """Tests that redshift_tabulated_lf fills outside magnitude or redshift grid."""
    result = redshift_tabulated_lf(
        np.array([-25.0, -21.0, -21.0]),
        np.array([0.3, 2.0, 0.3]),
        magnitude_grid=MAG_GRID,
        redshift_grid=Z_GRID,
        phi_grid=PHI_2D_GRID,
        fill_value=4.0,
    )

    np.testing.assert_allclose(result, np.array([4.0, 4.0, 8.0e-4]))


def test_redshift_tabulated_lf_interpolates_in_log_phi() -> None:
    """Tests that redshift_tabulated_lf supports log interpolation."""
    result = redshift_tabulated_lf(
        -21.0,
        0.3,
        magnitude_grid=np.array([-22.0, -20.0]),
        redshift_grid=np.array([0.1, 0.5]),
        phi_grid=np.array([[1.0e-4, 1.0e-2], [1.0e-3, 1.0e-1]]),
        log_phi=True,
    )

    assert result == pytest.approx(np.sqrt(1.0e-3 * 1.0e-2))


def test_redshift_tabulated_lf_rejects_negative_redshift() -> None:
    """Tests that redshift_tabulated_lf rejects negative requested redshift."""
    with pytest.raises(ValueError, match="redshift must be non-negative"):
        redshift_tabulated_lf(
            -21.0,
            -0.1,
            magnitude_grid=MAG_GRID,
            redshift_grid=Z_GRID,
            phi_grid=PHI_2D_GRID,
        )


def test_redshift_tabulated_lf_rejects_negative_redshift_grid() -> None:
    """Tests that redshift_tabulated_lf rejects negative redshift grid values."""
    with pytest.raises(ValueError, match="redshift_grid contains negative values"):
        redshift_tabulated_lf(
            -21.0,
            0.3,
            magnitude_grid=MAG_GRID,
            redshift_grid=np.array([-0.1, 0.5, 1.0]),
            phi_grid=PHI_2D_GRID,
        )


def test_redshift_tabulated_lf_rejects_wrong_phi_grid_shape() -> None:
    """Tests that redshift_tabulated_lf validates phi_grid shape."""
    with pytest.raises(ValueError, match=r"phi_grid must have shape"):
        redshift_tabulated_lf(
            -21.0,
            0.3,
            magnitude_grid=MAG_GRID,
            redshift_grid=Z_GRID,
            phi_grid=np.ones((2, 4)),
        )


def test_redshift_binned_lf_returns_piecewise_constant_values() -> None:
    """Tests that redshift_binned_lf returns values from magnitude-redshift bins."""
    result = redshift_binned_lf(
        np.array([-23.0, -21.0, -19.0]),
        np.array([0.2, 0.7, 1.2]),
        magnitude_bin_edges=MAG_EDGES,
        redshift_bin_edges=Z_EDGES,
        phi_bin_values=PHI_2D_BINS,
    )

    np.testing.assert_allclose(result, np.array([1.0e-5, 6.0e-4, 1.2e-3]))


def test_redshift_binned_lf_uses_fill_value_outside_bins() -> None:
    """Tests that redshift_binned_lf fills outside magnitude or redshift bins."""
    result = redshift_binned_lf(
        np.array([-25.0, -21.0, -21.0]),
        np.array([0.2, 2.0, 0.7]),
        magnitude_bin_edges=MAG_EDGES,
        redshift_bin_edges=Z_EDGES,
        phi_bin_values=PHI_2D_BINS,
        fill_value=8.0,
    )

    np.testing.assert_allclose(result, np.array([8.0, 8.0, 6.0e-4]))


def test_redshift_binned_lf_rejects_negative_redshift() -> None:
    """Tests that redshift_binned_lf rejects negative requested redshift."""
    with pytest.raises(ValueError, match="redshift must be non-negative"):
        redshift_binned_lf(
            -21.0,
            -0.1,
            magnitude_bin_edges=MAG_EDGES,
            redshift_bin_edges=Z_EDGES,
            phi_bin_values=PHI_2D_BINS,
        )


def test_redshift_binned_lf_rejects_wrong_phi_bin_shape() -> None:
    """Tests that redshift_binned_lf validates phi_bin_values shape."""
    with pytest.raises(ValueError, match=r"phi_bin_values must have shape"):
        redshift_binned_lf(
            -21.0,
            0.7,
            magnitude_bin_edges=MAG_EDGES,
            redshift_bin_edges=Z_EDGES,
            phi_bin_values=np.ones((2, 3)),
        )


def test_distance_tabulated_lf_interpolates_in_magnitude_and_distance() -> None:
    """Tests bilinear interpolation for distance_tabulated_lf."""
    result = distance_tabulated_lf(
        -21.0,
        300.0,
        magnitude_grid=MAG_GRID,
        distance_grid=DIST_GRID,
        phi_grid=PHI_2D_GRID,
    )

    assert np.asarray(result).shape == ()
    assert result == pytest.approx(8.0e-4)


def test_distance_tabulated_lf_accepts_broadcastable_inputs() -> None:
    """Tests that distance_tabulated_lf broadcasts magnitude and distance."""
    result = distance_tabulated_lf(
        np.array([-21.0, -21.0]),
        300.0,
        magnitude_grid=MAG_GRID,
        distance_grid=DIST_GRID,
        phi_grid=PHI_2D_GRID,
    )

    assert result.shape == (2,)
    np.testing.assert_allclose(result, np.array([8.0e-4, 8.0e-4]))


def test_distance_tabulated_lf_uses_fill_value_outside_grid() -> None:
    """Tests that distance_tabulated_lf fills outside magnitude or distance grid."""
    result = distance_tabulated_lf(
        np.array([-25.0, -21.0, -21.0]),
        np.array([300.0, 2000.0, 300.0]),
        magnitude_grid=MAG_GRID,
        distance_grid=DIST_GRID,
        phi_grid=PHI_2D_GRID,
        fill_value=4.0,
    )

    np.testing.assert_allclose(result, np.array([4.0, 4.0, 8.0e-4]))


def test_distance_tabulated_lf_interpolates_in_log_phi() -> None:
    """Tests that distance_tabulated_lf supports log interpolation."""
    result = distance_tabulated_lf(
        -21.0,
        300.0,
        magnitude_grid=np.array([-22.0, -20.0]),
        distance_grid=np.array([100.0, 500.0]),
        phi_grid=np.array([[1.0e-4, 1.0e-2], [1.0e-3, 1.0e-1]]),
        log_phi=True,
    )

    assert result == pytest.approx(np.sqrt(1.0e-3 * 1.0e-2))


def test_distance_tabulated_lf_rejects_negative_distance() -> None:
    """Tests that distance_tabulated_lf rejects negative requested distance."""
    with pytest.raises(ValueError, match="comoving_distance must be non-negative"):
        distance_tabulated_lf(
            -21.0,
            -1.0,
            magnitude_grid=MAG_GRID,
            distance_grid=DIST_GRID,
            phi_grid=PHI_2D_GRID,
        )


def test_distance_tabulated_lf_rejects_negative_distance_grid() -> None:
    """Tests that distance_tabulated_lf rejects negative distance grid values."""
    with pytest.raises(ValueError, match="distance_grid contains negative values"):
        distance_tabulated_lf(
            -21.0,
            300.0,
            magnitude_grid=MAG_GRID,
            distance_grid=np.array([-100.0, 500.0, 1000.0]),
            phi_grid=PHI_2D_GRID,
        )


def test_distance_tabulated_lf_rejects_wrong_phi_grid_shape() -> None:
    """Tests that distance_tabulated_lf validates phi_grid shape."""
    with pytest.raises(ValueError, match=r"phi_grid must have shape"):
        distance_tabulated_lf(
            -21.0,
            300.0,
            magnitude_grid=MAG_GRID,
            distance_grid=DIST_GRID,
            phi_grid=np.ones((2, 4)),
        )


def test_distance_binned_lf_returns_piecewise_constant_values() -> None:
    """Tests that distance_binned_lf returns values from magnitude-distance bins."""
    result = distance_binned_lf(
        np.array([-23.0, -21.0, -19.0]),
        np.array([200.0, 700.0, 1200.0]),
        magnitude_bin_edges=MAG_EDGES,
        distance_bin_edges=DIST_EDGES,
        phi_bin_values=PHI_2D_BINS,
    )

    np.testing.assert_allclose(result, np.array([1.0e-5, 6.0e-4, 1.2e-3]))


def test_distance_binned_lf_uses_fill_value_outside_bins() -> None:
    """Tests that distance_binned_lf fills outside magnitude or distance bins."""
    result = distance_binned_lf(
        np.array([-25.0, -21.0, -21.0]),
        np.array([200.0, 2000.0, 700.0]),
        magnitude_bin_edges=MAG_EDGES,
        distance_bin_edges=DIST_EDGES,
        phi_bin_values=PHI_2D_BINS,
        fill_value=8.0,
    )

    np.testing.assert_allclose(result, np.array([8.0, 8.0, 6.0e-4]))


def test_distance_binned_lf_rejects_negative_distance() -> None:
    """Tests that distance_binned_lf rejects negative requested distance."""
    with pytest.raises(ValueError, match="comoving_distance must be non-negative"):
        distance_binned_lf(
            -21.0,
            -1.0,
            magnitude_bin_edges=MAG_EDGES,
            distance_bin_edges=DIST_EDGES,
            phi_bin_values=PHI_2D_BINS,
        )


def test_distance_binned_lf_rejects_negative_distance_edges() -> None:
    """Tests that distance_binned_lf rejects negative distance bin edges."""
    with pytest.raises(ValueError, match="distance_bin_edges contains negative values"):
        distance_binned_lf(
            -21.0,
            300.0,
            magnitude_bin_edges=MAG_EDGES,
            distance_bin_edges=np.array([-100.0, 500.0, 1000.0, 1500.0]),
            phi_bin_values=PHI_2D_BINS,
        )


def test_distance_binned_lf_rejects_wrong_phi_bin_shape() -> None:
    """Tests that distance_binned_lf validates phi_bin_values shape."""
    with pytest.raises(ValueError, match=r"phi_bin_values must have shape"):
        distance_binned_lf(
            -21.0,
            700.0,
            magnitude_bin_edges=MAG_EDGES,
            distance_bin_edges=DIST_EDGES,
            phi_bin_values=np.ones((2, 3)),
        )
