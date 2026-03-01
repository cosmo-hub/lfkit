"""Unit tests for `lfkit.utils.interpolation` module."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.interpolation import (
    linear_interp_extrap,
    prep_strictly_increasing_xy,
    build_1d_interpolator,
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
