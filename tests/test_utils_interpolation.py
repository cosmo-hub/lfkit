"""Unit tests for ``lfkit.utils.interpolation``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.interpolation import (
    as_1d_finite_grid,
    build_1d_interpolator,
    linear_interp_extrap,
    prep_strictly_increasing_xy,
)


def test_linear_interp_extrap_matches_numpy_interp_in_range():
    """Tests that linear_interp_extrap matches numpy.interp inside the tabulated range."""
    xp = np.array([0.0, 1.0, 2.0])
    fp = np.array([0.0, 1.0, 4.0])
    x = np.array([0.25, 0.5, 1.5, 1.75])
    got = linear_interp_extrap(x, xp, fp)
    exp = np.interp(x, xp, fp)
    assert np.allclose(got, exp, rtol=0.0, atol=0.0)


def test_linear_interp_extrap_extrapolates_linearly_left_and_right():
    """Tests that linear_interp_extrap uses endpoint slopes for linear extrapolation outside the range."""
    xp = np.array([0.0, 1.0, 3.0])
    fp = np.array([10.0, 12.0, 20.0])
    x = np.array([-1.0, 0.0, 3.0, 4.0])

    got = linear_interp_extrap(x, xp, fp)

    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    exp = np.array(
        [
            fp[0] + m_left * (x[0] - xp[0]),
            fp[0],
            fp[-1],
            fp[-1] + m_right * (x[-1] - xp[-1]),
        ]
    )
    assert np.allclose(got, exp, rtol=0.0, atol=0.0)


def test_linear_interp_extrap_raises_on_shape_mismatch():
    """Tests that linear_interp_extrap raises ValueError when xp and fp shapes differ."""
    with pytest.raises(ValueError):
        linear_interp_extrap(np.array([0.0]), np.array([0.0, 1.0]), np.array([1.0]))


def test_prep_strictly_increasing_xy_sorts_dedupes_and_filters():
    """Tests that prep_strictly_increasing_xy sorts by z, removes non-finite entries, and drops duplicate z values."""
    z = np.array([0.2, 0.1, 0.1, np.nan, 0.3])
    y = np.array([2.0, 1.0, 999.0, 4.0, np.inf])

    z_out, y_out = prep_strictly_increasing_xy(z, y)

    assert np.all(np.isfinite(z_out)) and np.all(np.isfinite(y_out))
    assert np.all(z_out[1:] > z_out[:-1])
    # Should keep z=0.1 (first occurrence after sort) and z=0.2; drop duplicate 0.1 and non-finite rows.
    assert np.allclose(z_out, np.array([0.1, 0.2]), rtol=0.0, atol=0.0)
    assert np.allclose(y_out, np.array([1.0, 2.0]), rtol=0.0, atol=0.0)


def test_prep_strictly_increasing_xy_raises_if_too_few_points():
    """Tests that prep_strictly_increasing_xy raises ValueError if fewer than two valid points remain."""
    z = np.array([np.nan, 0.0])
    y = np.array([1.0, np.inf])
    with pytest.raises(ValueError):
        prep_strictly_increasing_xy(z, y)


@pytest.mark.parametrize("method", ["linear", "pchip", "akima"])
def test_build_1d_interpolator_no_extrap_returns_finite_in_range(method):
    """Tests that build_1d_interpolator evaluates finite values within range when extrapolate is False."""
    z = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    f = build_1d_interpolator(z, y, method=method, extrapolate=False)
    xx = np.array([0.5, 1.5, 2.5])
    out = np.asarray(f(xx), float)
    assert out.shape == xx.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", ["linear", "pchip", "akima"])
def test_build_1d_interpolator_native_extrap_produces_finite_outside(method):
    """Tests that build_1d_interpolator with extrap_mode='native' returns finite values outside the tabulated range."""
    z = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    f = build_1d_interpolator(z, y, method=method, extrapolate=True, extrap_mode="native")
    xx = np.array([-0.5, 0.5, 3.5])
    out = np.asarray(f(xx), float)
    assert out.shape == xx.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", ["linear", "pchip", "akima"])
def test_build_1d_interpolator_linear_tail_matches_endpoint_slopes(method):
    """Tests that build_1d_interpolator with extrap_mode='linear_tail' extrapolates using endpoint slopes."""
    z = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 12.0, 20.0])
    f = build_1d_interpolator(z, y, method=method, extrapolate=True, extrap_mode="linear_tail")

    left_x = np.array([-1.0])
    right_x = np.array([3.0])

    m_left = (y[1] - y[0]) / (z[1] - z[0])
    m_right = (y[-1] - y[-2]) / (z[-1] - z[-2])

    left_exp = y[0] + m_left * (left_x - z[0])
    right_exp = y[-1] + m_right * (right_x - z[-1])

    assert np.allclose(np.asarray(f(left_x), float), left_exp, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(f(right_x), float), right_exp, rtol=1e-12, atol=0.0)


def test_build_1d_interpolator_raises_on_unknown_method():
    """Tests that build_1d_interpolator raises ValueError for an unknown interpolation method."""
    z = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        build_1d_interpolator(z, y, method="nope", extrapolate=False)  # type: ignore[arg-type]


def test_build_1d_interpolator_raises_on_unknown_extrap_mode():
    """Tests that build_1d_interpolator raises ValueError for an unknown extrapolation mode."""
    z = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        build_1d_interpolator(z, y, method="linear", extrapolate=True, extrap_mode="nope")  # type: ignore[arg-type]


def test_interpolation_exports_expected_public_names() -> None:
    """Tests that interpolation exposes the expected public API names."""
    import lfkit.utils.interpolation as interpolation

    expected = {
        "linear_interp_extrap",
        "build_1d_interpolator",
        "prep_strictly_increasing_xy",
        "as_1d_finite_grid",
    }

    assert set(interpolation.__all__) == expected


def test_linear_interp_extrap_supports_scalar_query() -> None:
    """Tests that linear_interp_extrap supports scalar query input."""
    result = linear_interp_extrap(
        np.asarray(0.5),
        np.array([0.0, 1.0], dtype=float),
        np.array([0.0, 2.0], dtype=float),
    )

    assert result.shape == ()
    assert result == pytest.approx(1.0)


def test_linear_interp_extrap_returns_numpy_interp_for_single_sample() -> None:
    """Tests that linear_interp_extrap falls back to numpy.interp for one sample."""
    result = linear_interp_extrap(
        np.array([-1.0, 0.0, 1.0], dtype=float),
        np.array([0.0], dtype=float),
        np.array([2.0], dtype=float),
    )

    np.testing.assert_allclose(result, np.array([2.0, 2.0, 2.0]))


def test_linear_interp_extrap_returns_float64_array() -> None:
    """Tests that linear_interp_extrap returns a float64 array."""
    result = linear_interp_extrap(
        np.array([0, 1, 2]),
        np.array([0, 2]),
        np.array([0, 4]),
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_prep_strictly_increasing_xy_converts_integer_inputs() -> None:
    """Tests that interpolation preparation converts integer inputs to floats."""
    z_out, y_out = prep_strictly_increasing_xy(
        np.array([2, 0, 1]),
        np.array([4, 0, 1]),
    )

    assert z_out.dtype == np.float64
    assert y_out.dtype == np.float64
    np.testing.assert_allclose(z_out, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(y_out, np.array([0.0, 1.0, 4.0]))


def test_prep_strictly_increasing_xy_keeps_first_duplicate_after_sort() -> None:
    """Tests that interpolation preparation keeps the first sorted duplicate."""
    z_out, y_out = prep_strictly_increasing_xy(
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.array([0.0, 10.0, 99.0, 20.0]),
    )

    np.testing.assert_allclose(z_out, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(y_out, np.array([0.0, 10.0, 20.0]))


def test_build_1d_interpolator_linear_no_extrap_clamps_outside() -> None:
    """Tests that linear interpolation without extrapolation clamps outside values."""
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([10.0, 20.0, 30.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=False,
    )

    result = f(np.array([-1.0, 0.5, 3.0], dtype=float))

    np.testing.assert_allclose(result, np.array([10.0, 15.0, 30.0]))


def test_build_1d_interpolator_linear_none_mode_clamps_outside() -> None:
    """Tests that linear interpolation with none extrapolation mode clamps outside."""
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([10.0, 20.0, 30.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=True,
        extrap_mode="none",
    )

    result = f(np.array([-1.0, 0.5, 3.0], dtype=float))

    np.testing.assert_allclose(result, np.array([10.0, 15.0, 30.0]))


def test_build_1d_interpolator_linear_native_extrap_matches_linear_extrap() -> None:
    """Tests that linear native extrapolation matches linear_interp_extrap."""
    z = np.array([0.0, 1.0, 3.0], dtype=float)
    y = np.array([10.0, 12.0, 20.0], dtype=float)
    x = np.array([-1.0, 0.5, 4.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=True,
        extrap_mode="native",
    )

    np.testing.assert_allclose(f(x), linear_interp_extrap(x, z, y))


def test_build_1d_interpolator_sorts_unsorted_inputs() -> None:
    """Tests that interpolator construction sorts unsorted tabulated inputs."""
    z = np.array([2.0, 0.0, 1.0], dtype=float)
    y = np.array([20.0, 0.0, 10.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=False,
    )

    result = f(np.array([0.5, 1.5], dtype=float))

    np.testing.assert_allclose(result, np.array([5.0, 15.0]))


def test_build_1d_interpolator_filters_nonfinite_inputs() -> None:
    """Tests that interpolator construction filters non-finite tabulated inputs."""
    z = np.array([0.0, 1.0, np.nan, 2.0], dtype=float)
    y = np.array([0.0, 10.0, 99.0, 20.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=False,
    )

    result = f(np.array([0.5, 1.5], dtype=float))

    np.testing.assert_allclose(result, np.array([5.0, 15.0]))


def test_build_1d_interpolator_rejects_too_few_valid_points() -> None:
    """Tests that interpolator construction rejects too few valid points."""
    with pytest.raises(ValueError, match="Need at least 2 points"):
        build_1d_interpolator(
            np.array([0.0, np.nan], dtype=float),
            np.array([1.0, 2.0], dtype=float),
            method="linear",
            extrapolate=False,
        )


def test_build_1d_interpolator_linear_tail_supports_scalar_query() -> None:
    """Tests that linear-tail extrapolation supports scalar query input."""
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([10.0, 20.0, 30.0], dtype=float)

    f = build_1d_interpolator(
        z,
        y,
        method="linear",
        extrapolate=True,
        extrap_mode="linear_tail",
    )

    result = f(np.asarray(3.0))

    assert result.shape == ()
    assert result == pytest.approx(40.0)


def test_as_1d_finite_grid_accepts_valid_grid() -> None:
    """Tests that as_1d_finite_grid accepts a finite one-dimensional grid."""
    result = as_1d_finite_grid([0.0, 0.5, 1.0], name="z")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]))


def test_as_1d_finite_grid_rejects_scalar_input() -> None:
    """Tests that as_1d_finite_grid rejects scalar input."""
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        as_1d_finite_grid(0.5, name="z")


def test_as_1d_finite_grid_rejects_single_point() -> None:
    """Tests that as_1d_finite_grid rejects one-point grids."""
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        as_1d_finite_grid([0.5], name="z")


def test_as_1d_finite_grid_rejects_multidimensional_input() -> None:
    """Tests that as_1d_finite_grid rejects multidimensional input."""
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        as_1d_finite_grid([[0.0, 0.5], [1.0, 1.5]], name="z")


def test_as_1d_finite_grid_rejects_nan_values() -> None:
    """Tests that as_1d_finite_grid rejects NaN values."""
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        as_1d_finite_grid([0.0, np.nan, 1.0], name="z")


def test_as_1d_finite_grid_rejects_infinite_values() -> None:
    """Tests that as_1d_finite_grid rejects infinite values."""
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        as_1d_finite_grid([0.0, np.inf, 1.0], name="z")
