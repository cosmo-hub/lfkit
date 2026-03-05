"""Unit tests for `lfkit.corrections.color_anchors` module."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.corrections.color_anchors import fit_coeffs_from_bandcolor


class _FakeTemplates:
    """Minimal templates container for fake kcorrect backend."""

    def __init__(self, n_templates: int):
        # Only nt matters for shape checks; contents irrelevant.
        self.restframe_flux = np.ones((n_templates, 3), dtype=float)


class _FakeKcorrect:
    """Minimal fake object returned by build_kcorrect (supports fit_coeffs + templates)."""

    def __init__(
        self,
        *,
        n_templates: int = 5,
        coeffs: np.ndarray | None = None,
        return_shape: tuple[int, ...] | None = None,
    ):
        self.templates = _FakeTemplates(n_templates=n_templates)
        self._coeffs = np.ones(n_templates, dtype=float) if coeffs is None else np.asarray(coeffs, float)
        self._return_shape = return_shape

    def fit_coeffs(self, *, redshift: float, maggies: np.ndarray, ivar: np.ndarray):
        # deterministic + simple sanity: require at least one constrained band
        if np.sum(ivar > 0) < 2:
            raise ValueError("need >=2 constrained bands")
        out = np.array(self._coeffs, float)
        if self._return_shape is not None:
            return np.zeros(self._return_shape, dtype=float)
        return out


def _patch_build_kcorrect(monkeypatch: pytest.MonkeyPatch, fake: _FakeKcorrect):
    """Patch build_kcorrect used inside color_anchors."""
    monkeypatch.setattr(
        "lfkit.corrections.color_anchors.build_kcorrect",
        lambda **kw: fake,
    )


def _patch_mag_to_maggies(monkeypatch: pytest.MonkeyPatch, *, value: float):
    """Patch mag_to_maggies to a fixed value (for validation tests)."""
    monkeypatch.setattr(
        "lfkit.corrections.color_anchors.mag_to_maggies",
        lambda mag: float(value),
    )


def test_fit_coeffs_from_bandcolor_defaults_anchor_to_band_b(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor defaults anchor_band to band_b and includes required bands."""
    fake = _FakeKcorrect(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    coeffs, fit_responses = fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=0.7,
        z_phot=0.1,
        anchor_band=None,
        anchor_mag=20.0,
        responses=None,
    )

    assert np.asarray(coeffs, float).shape == (5,)
    assert fit_responses == ["sdss_g0", "sdss_r0"]  # anchor defaults to band_b -> no duplicate


def test_fit_coeffs_from_bandcolor_custom_anchor_is_included(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor includes anchor_band in fit_responses when distinct."""
    fake = _FakeKcorrect(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    coeffs, fit_responses = fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=0.7,
        anchor_band="sdss_i0",
        responses=None,
    )

    assert np.asarray(coeffs, float).shape == (5,)
    assert fit_responses == ["sdss_g0", "sdss_r0", "sdss_i0"]


def test_fit_coeffs_from_bandcolor_responses_must_include_required_bands(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor raises ValueError if responses omits required bands."""
    fake = _FakeKcorrect(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match="responses is missing required bands"):
        fit_coeffs_from_bandcolor(
            color=("sdss_g0", "sdss_r0"),
            color_value=0.7,
            anchor_band="sdss_r0",
            responses=["sdss_u0", "sdss_i0"],  # missing g and r
        )


def test_fit_coeffs_from_bandcolor_anchor_mag_must_map_to_positive_finite(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor rejects anchor_mag that maps to non-positive or non-finite maggies."""
    fake = _FakeKcorrect(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    _patch_mag_to_maggies(monkeypatch, value=np.nan)
    with pytest.raises(ValueError, match="anchor_mag must map to positive finite maggies"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7, anchor_mag=20.0)

    _patch_mag_to_maggies(monkeypatch, value=0.0)
    with pytest.raises(ValueError, match="anchor_mag must map to positive finite maggies"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7, anchor_mag=20.0)

    _patch_mag_to_maggies(monkeypatch, value=-1.0)
    with pytest.raises(ValueError, match="anchor_mag must map to positive finite maggies"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7, anchor_mag=20.0)


def test_fit_coeffs_from_bandcolor_color_value_must_be_finite(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor rejects non-finite color_value."""
    fake = _FakeKcorrect(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match="color_value must be finite"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=np.nan)


def test_fit_coeffs_from_bandcolor_rejects_wrong_coeff_shape(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor raises ValueError if fit_coeffs returns unexpected shape."""
    # nt=5 but backend returns shape (5,1)
    fake = _FakeKcorrect(n_templates=5, return_shape=(5, 1))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match=r"fit_coeffs returned shape .* expected \(5,\)"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7)


def test_fit_coeffs_from_bandcolor_rejects_nonfinite_coeffs(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor raises ValueError if fit_coeffs returns non-finite coeffs."""
    fake = _FakeKcorrect(n_templates=5, coeffs=np.array([1.0, 1.0, np.nan, 1.0, 1.0]))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match="non-finite coeffs"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7)


def test_fit_coeffs_from_bandcolor_rejects_negative_coeffs(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor raises ValueError if fit_coeffs returns negative coeffs."""
    fake = _FakeKcorrect(n_templates=5, coeffs=np.array([1.0, 1.0, -1e-6, 1.0, 1.0]))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match="negative coeffs"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7)


def test_fit_coeffs_from_bandcolor_rejects_sum_le_zero(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor raises ValueError if fit_coeffs returns coeffs with sum<=0."""
    fake = _FakeKcorrect(n_templates=5, coeffs=np.zeros(5, dtype=float))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    with pytest.raises(ValueError, match="sum<=0"):
        fit_coeffs_from_bandcolor(color=("sdss_g0", "sdss_r0"), color_value=0.7)


def test_fit_coeffs_from_bandcolor_returns_fit_responses_in_given_order(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor returns fit_responses matching provided responses ordering."""
    fake = _FakeKcorrect(n_templates=3, coeffs=np.array([0.2, 0.3, 0.5]))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    coeffs, fit_responses = fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=0.4,
        anchor_band="sdss_r0",
        responses=["sdss_r0", "sdss_g0", "sdss_u0"],  # includes required
    )

    assert np.asarray(coeffs, float).shape == (3,)
    assert fit_responses == ["sdss_r0", "sdss_g0", "sdss_u0"]


def test_fit_coeffs_from_bandcolor_rescale_maggies_does_not_change_validity(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor returns finite coeffs for rescale_maggies True/False."""
    fake = _FakeKcorrect(n_templates=4, coeffs=np.array([0.1, 0.2, 0.3, 0.4]))
    _patch_build_kcorrect(monkeypatch, fake=fake)

    c1, r1 = fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=1.1,
        anchor_mag=20.0,
        rescale_maggies=True,
    )
    c2, r2 = fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=1.1,
        anchor_mag=20.0,
        rescale_maggies=False,
    )

    assert r1 == r2
    assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))
    assert np.allclose(c1, c2)


def test_fit_coeffs_from_bandcolor_passes_redshift_to_backend(monkeypatch: pytest.MonkeyPatch):
    """Tests that fit_coeffs_from_bandcolor passes z_phot to backend fit_coeffs."""
    captured = {}

    class Fake(_FakeKcorrect):
        def fit_coeffs(self, *, redshift: float, maggies: np.ndarray, ivar: np.ndarray):
            captured["redshift"] = float(redshift)
            return super().fit_coeffs(redshift=redshift, maggies=maggies, ivar=ivar)

    fake = Fake(n_templates=5)
    _patch_build_kcorrect(monkeypatch, fake=fake)

    fit_coeffs_from_bandcolor(
        color=("sdss_g0", "sdss_r0"),
        color_value=0.7,
        z_phot=0.321,
    )

    assert "redshift" in captured
    assert captured["redshift"] == pytest.approx(0.321)
