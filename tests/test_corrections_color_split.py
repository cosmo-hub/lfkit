"""Unit tests for `lfkit.corrections.color_split` module."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.corrections.color_split import (
    SDSS_RESPONSES,
    gr_cut_linear,
    default_gr_anchors,
    evaluate_coeffs_maggies_z0,
    validate_color_anchor_gr,
    fit_color_anchor_gr,
    fit_red_blue_anchors,
    validate_sed_sanity,
)


def test_gr_cut_linear_matches_formula():
    """Tests that gr_cut_linear returns a + b*M_r exactly for scalar inputs."""
    M_r = -21.5
    a, b = 0.12, -0.025
    got = gr_cut_linear(M_r, a=a, b=b)
    exp = a + b * M_r
    assert np.isclose(got, exp, rtol=0.0, atol=0.0)


def test_default_gr_anchors_offsets_are_applied():
    """Tests that default_gr_anchors offsets shift anchors relative to the cut by the requested magnitudes."""
    cut, red, blue = default_gr_anchors(
        M_r_ref=-21.5,
        a=0.12,
        b=-0.025,
        red_offset=0.10,
        blue_offset=0.20,
    )

    atol = 1e-12  # tolerate unavoidable float representation noise

    assert np.isfinite(cut) and np.isfinite(red) and np.isfinite(blue)

    # Red should be offset from the cut by +red_offset
    assert np.isclose(red - cut, 0.10, rtol=0.0, atol=atol)

    # Blue should be separated from the cut by blue_offset in magnitude.
    # Allow either sign convention (blue below or above the cut).
    assert np.isclose(abs(blue - cut), 0.20, rtol=0.0, atol=atol)

    # Ensure ordering is sensible: red is redder than blue.
    assert red > blue


def test_validate_color_anchor_gr_requires_g_and_r():
    """Tests that validate_color_anchor_gr raises ValueError if responses do not include sdss_g0 and sdss_r0."""
    coeffs = np.ones(5, dtype=float)
    with pytest.raises(ValueError):
        validate_color_anchor_gr(
            coeffs=coeffs,
            g_minus_r_target=0.7,
            responses=["sdss_u0", "sdss_i0", "sdss_z0"],
        )


def test_fit_color_anchor_gr_requires_g_and_r():
    """Tests that fit_color_anchor_gr raises ValueError if responses do not include sdss_g0 and sdss_r0."""
    with pytest.raises(ValueError):
        fit_color_anchor_gr(
            g_minus_r=0.7,
            responses=["sdss_u0", "sdss_i0", "sdss_z0"],
            validate=False,
        )


@pytest.mark.slow
@pytest.mark.parametrize("gmr", [0.2, 0.8])
def test_fit_color_anchor_gr_returns_finite_coeffs(gmr):
    """Tests that fit_color_anchor_gr returns a finite coefficient vector for a valid (g-r) target."""
    coeffs = fit_color_anchor_gr(g_minus_r=gmr, validate=False)
    coeffs = np.asarray(coeffs, float)
    assert coeffs.ndim == 1
    assert coeffs.size > 0
    assert np.all(np.isfinite(coeffs))


@pytest.mark.slow
def test_evaluate_coeffs_maggies_z0_returns_expected_length():
    """Tests that evaluate_coeffs_maggies_z0 returns one maggies value per response curve."""
    coeffs = fit_color_anchor_gr(g_minus_r=0.6, validate=False)
    maggies = evaluate_coeffs_maggies_z0(coeffs=coeffs, responses=SDSS_RESPONSES)
    maggies = np.asarray(maggies, float)
    assert maggies.shape == (len(SDSS_RESPONSES),)
    assert np.all(np.isfinite(maggies))


@pytest.mark.slow
def test_validate_color_anchor_gr_accepts_fitted_coeffs_within_tolerance():
    """Tests that validate_color_anchor_gr succeeds for coeffs produced by fit_color_anchor_gr within tol_mag."""
    gmr = 0.65
    coeffs = fit_color_anchor_gr(g_minus_r=gmr, validate=True, tol_mag=0.05)
    out = validate_color_anchor_gr(
        coeffs=coeffs,
        g_minus_r_target=gmr,
        tol_mag=0.05,
    )
    assert np.isfinite(out["g_minus_r"])
    assert abs(out["err"]) <= 0.05


@pytest.mark.slow
def test_fit_red_blue_anchors_returns_expected_structure():
    """Tests that fit_red_blue_anchors returns cut/red/blue entries with colors and coefficient vectors."""
    out = fit_red_blue_anchors()
    assert set(out.keys()) == {"cut", "red", "blue"}
    assert "g_minus_r" in out["cut"]
    assert "g_minus_r" in out["red"] and "coeffs" in out["red"]
    assert "g_minus_r" in out["blue"] and "coeffs" in out["blue"]
    assert np.asarray(out["red"]["coeffs"], float).ndim == 1
    assert np.asarray(out["blue"]["coeffs"], float).ndim == 1


@pytest.mark.slow
def test_validate_sed_sanity_accepts_reasonable_coeffs():
    """Tests that validate_sed_sanity returns finite adjacent SDSS colors for fitted coefficients."""
    coeffs = fit_color_anchor_gr(g_minus_r=0.7, validate=False)
    colors = validate_sed_sanity(coeffs=coeffs, max_abs_color=20.0)  # loose threshold for robustness
    assert set(colors.keys()) == {"u-g", "g-r", "r-i", "i-z"}
    assert all(np.isfinite(v) for v in colors.values())
