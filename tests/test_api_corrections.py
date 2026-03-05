"""Unit tests for `lfkit.api.corrections` module."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.api.corrections import Corrections


def _fake_interp_factory(x: np.ndarray, y: np.ndarray):
    """
    Return a callable that linearly interpolates over x, y (with edge hold).
    This stands in for build_1d_interpolator / Poggianti interpolators.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def f(z):
        z = np.asarray(z, float)
        return np.interp(z, x, y, left=y[0], right=y[-1])

    return f


def test_k_e_ke_shapes_and_types():
    """Tests that k, e, ke are returned as expected."""
    k = lambda z: np.asarray(z, float) * 2.0
    e = lambda z: np.asarray(z, float) * -1.0
    c = Corrections(k_func=k, e_func=e)

    z = np.array([0.0, 0.5, 1.0])
    K = c.k(z)
    E = c.e(z)
    KE = c.ke(z)

    assert isinstance(K, np.ndarray) and K.dtype == float
    assert isinstance(E, np.ndarray) and E.dtype == float
    assert isinstance(KE, np.ndarray) and KE.dtype == float
    assert np.allclose(K, [0.0, 1.0, 2.0])
    assert np.allclose(E, [0.0, -0.5, -1.0])
    assert np.allclose(KE, [0.0, 0.5, 1.0])


def test_e_none_returns_zeros_like_input():
    """Tests that e(z) returns zeros when e backend is None."""
    k = lambda z: np.asarray(z, float) + 1.0
    c = Corrections(k_func=k, e_func=None)

    z = np.array([0.2, 0.4], dtype=float)
    assert np.allclose(c.e(z), np.zeros_like(z))
    assert np.allclose(c.ke(z), c.k(z))


def test_poggianti_e_model_none_is_zero():
    """Tests that Corrections.poggianti(e_model='none') returns e(z)=0."""
    import lfkit.corrections.poggianti1997 as pogg

    def fake_load_poggianti1997_tables(*, band: str, sed: str):
        z_k = np.array([0.0, 1.0])
        k = np.array([0.0, 1.0])
        z_e = np.array([0.0, 1.0])
        e = np.array([0.0, -0.5])  # should be ignored when e_model='none'
        return z_k, k, z_e, e

    def fake_make_kcorr_interpolator(z, y, **kw):
        return _fake_interp_factory(z, y)

    def fake_make_ecorr_interpolator(z, y, **kw):
        return _fake_interp_factory(z, y)

    old_load = pogg.load_poggianti1997_tables
    old_k = pogg.make_kcorr_interpolator
    old_e = pogg.make_ecorr_interpolator
    try:
        pogg.load_poggianti1997_tables = fake_load_poggianti1997_tables  # type: ignore[assignment]
        pogg.make_kcorr_interpolator = fake_make_kcorr_interpolator  # type: ignore[assignment]
        pogg.make_ecorr_interpolator = fake_make_ecorr_interpolator  # type: ignore[assignment]

        c = Corrections.poggianti(band="V", gal_type="E", e_model="none")
        assert c.meta["k_backend"] == "poggianti1997"
        assert c.meta["e_backend"] == "none"
        assert c.meta["band"] == "V"
        assert c.meta["gal_type"] == "E"

        assert np.allclose(c.k([0.5]), [0.5])
        assert np.allclose(c.e([0.5]), [0.0])
        assert np.allclose(c.ke([0.5]), [0.5])
    finally:
        pogg.load_poggianti1997_tables = old_load  # type: ignore[assignment]
        pogg.make_kcorr_interpolator = old_k  # type: ignore[assignment]
        pogg.make_ecorr_interpolator = old_e  # type: ignore[assignment]


def test_poggianti_rejects_unknown_e_model(monkeypatch: pytest.MonkeyPatch):
    """Tests that Corrections.poggianti rejects unknown e_model."""
    import lfkit.corrections.poggianti1997 as pogg

    monkeypatch.setattr(
        pogg,
        "load_poggianti1997_tables",
        lambda **kw: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -0.5]),
        ),
    )
    monkeypatch.setattr(pogg, "make_kcorr_interpolator", lambda z, y, **kw: _fake_interp_factory(z, y))
    monkeypatch.setattr(pogg, "make_ecorr_interpolator", lambda z, y, **kw: _fake_interp_factory(z, y))

    with pytest.raises(ValueError, match="e_model must be 'none' or 'poggianti'"):
        Corrections.poggianti(band="V", gal_type="E", e_model="parametric")


def test_poggianti_wires_meta_and_interpolators(monkeypatch: pytest.MonkeyPatch):
    """Tests that Corrections.poggianti wires meta and interpolators correctly."""
    import lfkit.corrections.poggianti1997 as pogg

    def fake_load(*, band: str, sed: str):
        z_k = np.array([0.0, 1.0])
        k = np.array([0.0, 1.0])
        z_e = np.array([0.0, 1.0])
        e = np.array([0.0, -0.5])
        return z_k, k, z_e, e

    monkeypatch.setattr(pogg, "load_poggianti1997_tables", fake_load)
    monkeypatch.setattr(pogg, "make_kcorr_interpolator", lambda z, y, **kw: _fake_interp_factory(z, y))
    monkeypatch.setattr(pogg, "make_ecorr_interpolator", lambda z, y, **kw: _fake_interp_factory(z, y))

    c = Corrections.poggianti(band="V", gal_type="E", method="pchip", extrapolate=True, e_model="poggianti")
    assert c.meta["k_backend"] == "poggianti1997"
    assert c.meta["e_backend"] == "poggianti"
    assert c.meta["band"] == "V"
    assert c.meta["gal_type"] == "E"

    assert np.allclose(c.k([0.5]), [0.5])
    assert np.allclose(c.e([0.5]), [-0.25])
    assert np.allclose(c.ke([0.5]), [0.25])


def test_kcorrect_uses_default_grid_when_z_grid_is_none(monkeypatch: pytest.MonkeyPatch):
    """Tests that Corrections.kcorrect builds a default z grid when z_grid is None."""
    captured = {}

    def fake_kcorrect_from_bandcolor(**kwargs):
        captured["z_in"] = np.asarray(kwargs["z"], float)
        # return some valid finite points
        z_ok = np.array([0.0, 1.0, 2.0], dtype=float)
        k_ok = np.array([0.0, 0.5, 1.0], dtype=float)
        return z_ok, k_ok

    monkeypatch.setattr(
        "lfkit.corrections.kcorrect_from_color.kcorrect_from_bandcolor",
        fake_kcorrect_from_bandcolor,
    )

    monkeypatch.setattr(
        "lfkit.api.corrections.build_1d_interpolator",
        lambda x, y, **kw: _fake_interp_factory(x, y),
    )

    c = Corrections.kcorrect(
        z_grid=None,
        response_out="sdss_r0",
        color=("sdss_g0", "sdss_r0"),
        color_value=0.7,
        redshift_range=(0.0, 2.0),
        nredshift=4000,
    )

    assert "z_in" in captured
    assert captured["z_in"].ndim == 1
    assert captured["z_in"].size == 4001  # your implementation uses 4001 hard-coded here
    assert np.isclose(captured["z_in"][0], 0.0)
    assert np.isclose(captured["z_in"][-1], 2.0)

    assert np.allclose(c.k([0.0, 1.0, 2.0]), [0.0, 0.5, 1.0])
    assert np.allclose(c.e([0.0, 1.0]), [0.0, 0.0])

    assert c.meta["k_backend"] == "kcorrect_bandcolor"
    assert c.meta["response_out"] == "sdss_r0"
    assert c.meta["color"] == ("sdss_g0", "sdss_r0")
    assert c.meta["color_value"] == pytest.approx(0.7)
    assert c.meta["n_finite"] == 3
    assert c.meta["z_valid_min"] == pytest.approx(0.0)
    assert c.meta["z_valid_max"] == pytest.approx(2.0)


def test_kcorrect_validates_z_grid(monkeypatch: pytest.MonkeyPatch):
    """Tests that Corrections.kcorrect rejects bad z_grid via as_1d_finite_grid."""
    monkeypatch.setattr(
        "lfkit.corrections.kcorrect_from_color.kcorrect_from_bandcolor",
        lambda **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    with pytest.raises(ValueError):
        Corrections.kcorrect(
            z_grid=np.array([[0.0, 0.1]]),
            response_out="sdss_r0",
            color=("sdss_g0", "sdss_r0"),
            color_value=0.7,
        )

    with pytest.raises(ValueError):
        Corrections.kcorrect(
            z_grid=np.array([0.0, np.nan, 0.2]),
            response_out="sdss_r0",
            color=("sdss_g0", "sdss_r0"),
            color_value=0.7,
        )


def test_kcorrect_calls_backend_with_expected_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Tests that Corrections.kcorrect passes through the expected kwargs to kcorrect_from_bandcolor."""
    captured = {}

    def fake_kcorrect_from_bandcolor(**kwargs):
        captured.update(kwargs)
        return np.array([0.0, 1.0], float), np.array([0.0, 1.0], float)

    monkeypatch.setattr(
        "lfkit.corrections.kcorrect_from_color.kcorrect_from_bandcolor",
        fake_kcorrect_from_bandcolor,
    )
    monkeypatch.setattr(
        "lfkit.api.corrections.build_1d_interpolator",
        lambda x, y, **kw: _fake_interp_factory(x, y),
    )

    z_grid = np.array([0.0, 0.5, 1.0], float)
    c = Corrections.kcorrect(
        z_grid=z_grid,
        response_out="bessell_V",
        color=("sdss_g0", "sdss_r0"),
        color_value=0.4,
        z_phot=0.1,
        anchor_band="sdss_r0",
        anchor_mag=19.5,
        band_shift=0.2,
        response_dir=None,
        redshift_range=(0.0, 1.0),
        nredshift=123,
        ivar_level=1e8,
        anchor_z0=False,
        method="linear",
        extrapolate=False,
    )

    assert np.allclose(c.k([0.0, 1.0]), [0.0, 1.0])
    assert np.allclose(c.e([0.0, 1.0]), [0.0, 0.0])

    assert np.allclose(captured["z"], z_grid)
    assert captured["response_out"] == "bessell_V"
    assert captured["color"] == ("sdss_g0", "sdss_r0")
    assert captured["color_value"] == pytest.approx(0.4)
    assert captured["z_phot"] == pytest.approx(0.1)
    assert captured["anchor_band"] == "sdss_r0"
    assert captured["anchor_mag"] == pytest.approx(19.5)
    assert captured["band_shift"] == pytest.approx(0.2)
    assert captured["redshift_range"] == (0.0, 1.0)
    assert captured["nredshift"] == 123
    assert captured["ivar_level"] == pytest.approx(1e8)
    assert captured["anchor_z0"] is False

    assert c.meta["method"] == "linear"
    assert c.meta["extrapolate"] is False
    assert c.meta["anchor_z0"] is False
