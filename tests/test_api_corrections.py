"""Unit tests for `lfkit.api.corrections` module."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from lfkit.api.corrections import Corrections


def _install_fake_kcorrect_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Corrections.kcorrect does: `import kcorrect.kcorrect as kk`
    even though it doesn't use `kk`. Provide a stub to avoid ImportError.
    """
    pkg = types.ModuleType("kcorrect")
    sub = types.ModuleType("kcorrect.kcorrect")
    monkeypatch.setitem(sys.modules, "kcorrect", pkg)
    monkeypatch.setitem(sys.modules, "kcorrect.kcorrect", sub)


@dataclass
class FakeTemplates:
    """Mock templates for Kcorrect."""
    restframe_wave: np.ndarray  # (Nw,)
    restframe_flux: np.ndarray  # (Ntmpl, Nw)


class FakeKcorrect:
    """
    Minimal object returned by build_kcorrect.
    - .kcorrect(...) returns an array-like with one element
    - .templates provides rest-frame templates for NNLS fit
    """

    def __init__(self, *, z_fail_after: float | None = None, n_templates: int = 5, n_wave: int = 20):
        self.z_fail_after = z_fail_after
        tw = np.linspace(3000.0, 9000.0, n_wave)
        # simple non-negative templates
        tf = np.vstack([(i + 1) * np.exp(-(tw - 5000.0) ** 2 / (2 * (800.0 + 50 * i) ** 2)) for i in range(n_templates)])
        self.templates = FakeTemplates(restframe_wave=tw, restframe_flux=tf)

    def kcorrect(self, *, redshift: float, coeffs: np.ndarray, band_shift: float | None = None):
        if self.z_fail_after is not None and redshift > self.z_fail_after:
            raise ValueError("out of range")
        # deterministic "K": dot with a simple function of redshift; return shape (1,)
        val = float(np.sum(coeffs)) * (0.1 + 0.01 * float(redshift))
        if band_shift is not None:
            val += 0.001 * float(band_shift)
        return np.array([val], dtype=float)


def _fake_interp_factory(x: np.ndarray, y: np.ndarray):
    """
    Return a callable that linearly interpolates over x, y (with np.interp extrap by edge).
    This stands in for build_1d_interpolator / Poggianti interpolators.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def f(z):
        z = np.asarray(z, float)
        return np.interp(z, x, y, left=y[0], right=y[-1])

    return f


def test_K_E_KE_shapes_and_types():
    """Tests that K, E, KE are returned as expected."""
    k = lambda z: np.asarray(z, float) * 2.0
    e = lambda z: np.asarray(z, float) * -1.0
    c = Corrections(k_func=k, e_func=e)

    z = np.array([0.0, 0.5, 1.0])
    K = c.K(z)
    E = c.E(z)
    KE = c.KE(z)

    assert isinstance(K, np.ndarray) and K.dtype == float
    assert isinstance(E, np.ndarray) and E.dtype == float
    assert isinstance(KE, np.ndarray) and KE.dtype == float
    assert np.allclose(K, [0.0, 1.0, 2.0])
    assert np.allclose(E, [0.0, -0.5, -1.0])
    assert np.allclose(KE, [0.0, 0.5, 1.0])


def test_E_none_returns_zeros_like_input():
    """Tests that e(z) returns zeros for z < 0.5."""
    k = lambda z: np.asarray(z, float) + 1.0
    c = Corrections(k_func=k, e_func=None)

    z = np.array([0.2, 0.4], dtype=float)
    assert np.allclose(c.E(z), np.zeros_like(z))
    assert np.allclose(c.KE(z), c.K(z))


def test_build_e_backend_none():
    """Tests that build_e_backend returns None for e_model='none'."""
    f = Corrections._build_e_backend(e_model="none", e_kwargs=None)
    assert f is None


def test_build_e_backend_rejects_unknown():
    """Tests that build_e_backend raises an error for unknown e_model."""
    with pytest.raises(ValueError, match="e_model must be"):
        Corrections._build_e_backend(e_model="parametric", e_kwargs={})


def test_build_e_backend_requires_kwargs_for_poggianti():
    """Tests that build_e_backend raises an error for poggianti without kwargs."""
    with pytest.raises(ValueError, match="e_kwargs required"):
        Corrections._build_e_backend(e_model="poggianti", e_kwargs=None)


def test_build_e_backend_poggianti_calls_constructor(monkeypatch: pytest.MonkeyPatch):
    """Tests that build_e_backend calls poggianti1997 with the correct kwargs."""
    sentinel = object()

    def fake_poggianti1997(**kwargs):
        # return an instance with an _e attribute (callable)
        out = types.SimpleNamespace()
        out._e = lambda z: np.asarray(z, float) * 0.0 + 7.0
        out.sentinel = sentinel
        return out

    monkeypatch.setattr(Corrections, "poggianti1997", staticmethod(fake_poggianti1997))
    efunc = Corrections._build_e_backend(e_model="poggianti", e_kwargs={"band": "r", "sed": "E"})
    assert callable(efunc)
    assert np.allclose(efunc([0.0, 1.0]), [7.0, 7.0])


def test_poggianti1997_wires_meta_and_interpolators(monkeypatch: pytest.MonkeyPatch):
    """Tests that poggianti1997 wires meta and interpolators correctly."""
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

    c = Corrections.poggianti1997(band="r", sed="E", method="pchip", extrapolate=True)
    assert c.meta["backend"] == "poggianti1997"
    assert c.meta["band"] == "r"
    assert c.meta["sed"] == "E"
    assert np.allclose(c.K([0.5]), [0.5])
    assert np.allclose(c.E([0.5]), [-0.25])
    assert np.allclose(c.KE([0.5]), [0.25])


def test_kcorrect_rejects_bad_z(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect raises an error for bad z."""
    _install_fake_kcorrect_module(monkeypatch)

    # Make sure build_kcorrect is never called on bad inputs
    monkeypatch.setattr("lfkit.api.corrections.build_kcorrect", lambda **kw: (_ for _ in ()).throw(AssertionError))

    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        Corrections.kcorrect(z=np.array([[0.0, 0.1]]), response="sdss_r0", coeffs=np.ones(5))
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        Corrections.kcorrect(z=np.array([0.0]), response="sdss_r0", coeffs=np.ones(5))
    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        Corrections.kcorrect(z=np.array([0.0, np.nan]), response="sdss_r0", coeffs=np.ones(5))


def test_kcorrect_requires_at_least_two_finite_points(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect raises an error for insufficient finite points."""
    _install_fake_kcorrect_module(monkeypatch)

    # Fake returns ValueError immediately -> 0 finite
    fake = FakeKcorrect(z_fail_after=-1.0)
    monkeypatch.setattr("lfkit.api.corrections.build_kcorrect", lambda **kw: fake)

    z = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="Too few finite points"):
        Corrections.kcorrect(z=z, response="sdss_r0", coeffs=np.ones(5), anchor_z0=True)


def test_kcorrect_anchor_z0_sets_first_valid_to_zero(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect sets the first valid point to zero."""
    _install_fake_kcorrect_module(monkeypatch)

    # Fail after 0.2 so we have finite at z<=0.2
    fake = FakeKcorrect(z_fail_after=0.2)
    monkeypatch.setattr("lfkit.api.corrections.build_kcorrect", lambda **kw: fake)

    # Use a stable interpolator stub
    monkeypatch.setattr("lfkit.api.corrections.build_1d_interpolator", lambda x, y, **kw: _fake_interp_factory(x, y))

    z = np.array([0.0, 0.1, 0.2, 0.3])
    coeffs = np.ones(5)

    c = Corrections.kcorrect(z=z, response="sdss_r0", coeffs=coeffs, anchor_z0=True, extrapolate=False)
    # At z=0.0 (first valid), anchored to 0
    assert np.allclose(c.K([0.0]), [0.0])

    # Meta sanity
    assert c.meta["backend"] == "kcorrect_compute"
    assert c.meta["response"] == "sdss_r0"
    assert c.meta["n_finite"] == 3
    assert np.isfinite(c.meta["kcorrect_z_valid_max"])


def test_kcorrect_no_anchor_preserves_offset(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect does not set the first valid point to zero."""
    _install_fake_kcorrect_module(monkeypatch)

    fake = FakeKcorrect(z_fail_after=0.2)
    monkeypatch.setattr("lfkit.api.corrections.build_kcorrect", lambda **kw: fake)
    monkeypatch.setattr("lfkit.api.corrections.build_1d_interpolator", lambda x, y, **kw: _fake_interp_factory(x, y))

    z = np.array([0.0, 0.1, 0.2])
    coeffs = np.ones(5)

    c = Corrections.kcorrect(z=z, response="sdss_r0", coeffs=coeffs, anchor_z0=False, extrapolate=False)
    # Without anchoring, z=0.0 returns the model value
    assert c.K([0.0])[0] != 0.0


def test_kcorrect_from_sed_validates_inputs(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect_from_sed validates inputs."""
    _install_fake_kcorrect_module(monkeypatch)

    with pytest.raises(ValueError, match="z must be a finite 1D array"):
        Corrections.kcorrect_from_sed(
            z=np.array([0.0]),
            response="sdss_r0",
            sed_wave_A=np.linspace(4000, 8000, 10),
            sed_flux=np.ones(10),
        )

    with pytest.raises(ValueError, match="must contain >=5 finite points"):
        Corrections.kcorrect_from_sed(
            z=np.array([0.0, 0.1]),
            response="sdss_r0",
            sed_wave_A=np.array([1.0, np.nan, 3.0, 4.0, np.nan]),
            sed_flux=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        )


def test_kcorrect_from_sed_calls_kcorrect_with_nnls_coeffs(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect_from_sed calls kcorrect with NNLS coefficients."""
    _install_fake_kcorrect_module(monkeypatch)

    fake = FakeKcorrect(z_fail_after=None, n_templates=6, n_wave=30)
    monkeypatch.setattr("lfkit.api.corrections.build_kcorrect", lambda **kw: fake)
    monkeypatch.setattr("lfkit.api.corrections.build_1d_interpolator", lambda x, y, **kw: _fake_interp_factory(x, y))

    # Capture coeffs passed into Corrections.kcorrect
    captured = {}

    real_kcorrect = Corrections.kcorrect

    def spy_kcorrect(cls, **kwargs):
        captured["coeffs"] = np.asarray(kwargs["coeffs"], float)
        return real_kcorrect(**kwargs)

    monkeypatch.setattr(Corrections, "kcorrect", classmethod(spy_kcorrect))

    z = np.array([0.0, 0.1, 0.2])
    sed_wave = np.linspace(3500.0, 9500.0, 200)
    sed_flux = np.exp(-(sed_wave - 5500.0) ** 2 / (2 * 1200.0**2)) + 0.05

    c = Corrections.kcorrect_from_sed(
        z=z,
        response="sdss_r0",
        sed_wave_A=sed_wave,
        sed_flux=sed_flux,
        weighted_fit=True,
        anchor_z0=True,
    )

    assert "coeffs" in captured
    assert captured["coeffs"].ndim == 1
    assert np.all(captured["coeffs"] >= -1e-12)  # NNLS non-negative within tol
    assert c.meta["coeffs_source"] == "fit_to_sed"
    assert c.meta["weighted_fit"] is True


def test_kcorrect_from_pkg_schema_validation():
    """Tests that kcorrect_from_pkg validates the input schema."""
    with pytest.raises(KeyError, match="missing required keys"):
        Corrections.kcorrect_from_pkg(pkg={"z": [0, 1]}, gal_type="E", out_band="r")


def test_kcorrect_from_pkg_rejects_bad_z():
    """Tests that kcorrect_from_pkg rejects bad z."""
    pkg = {"z": [0.0], "responses_out": ["r"], "types": ["E"], "K": {"E": np.zeros((1, 1))}}
    with pytest.raises(ValueError, match="pkg\\['z'\\] must be"):
        Corrections.kcorrect_from_pkg(pkg=pkg, gal_type="E", out_band="r")


def test_kcorrect_from_pkg_missing_type_or_band():
    """Tests that kcorrect_from_pkg rejects missing type or band."""
    pkg = {
        "z": [0.0, 1.0],
        "responses_out": ["r"],
        "types": ["E"],
        "K": {"E": np.zeros((2, 1))},
    }
    with pytest.raises(ValueError, match="gal_type=.* not in"):
        Corrections.kcorrect_from_pkg(pkg=pkg, gal_type="Sbc", out_band="r")
    with pytest.raises(ValueError, match="out_band=.* not in"):
        Corrections.kcorrect_from_pkg(pkg=pkg, gal_type="E", out_band="i")


def test_kcorrect_from_pkg_handles_too_few_finite_points(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect_from_pkg handles too few finite points."""
    pkg = {
        "z": np.array([0.0, 1.0]),
        "responses_out": ["r"],
        "types": ["E"],
        "K": {"E": np.array([[np.nan], [np.nan]])},
    }
    c = Corrections.kcorrect_from_pkg(pkg=pkg, gal_type="E", out_band="r")
    out = c.K(np.array([0.2, 0.8]))
    assert np.all(np.isnan(out))
    assert "unavailable" in (c._k.__doc__ or "").lower()
    assert c.meta["n_finite"] == 0


def test_kcorrect_from_pkg_builds_interpolator_when_ok(monkeypatch: pytest.MonkeyPatch):
    """Tests that kcorrect_from_pkg builds an interpolator when all inputs are valid."""
    monkeypatch.setattr("lfkit.api.corrections.build_1d_interpolator", lambda x, y, **kw: _fake_interp_factory(x, y))

    pkg = {
        "z": np.array([0.0, 1.0]),
        "responses_out": ["r", "i"],
        "types": ["E"],
        "K": {"E": np.array([[0.0, 0.0], [1.0, 2.0]])},
    }
    c = Corrections.kcorrect_from_pkg(pkg=pkg, gal_type="E", out_band="i", extrapolate=False)
    assert np.allclose(c.K([0.5]), [1.0])  # halfway between 0 and 2
    assert c.meta["backend"] == "kcorrect_precomputed"
    assert c.meta["n_finite"] == 2


def test_register_kcorrect_response_validates_visibility(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Tests that register_kcorrect_response validates visibility."""
    monkeypatch.setattr("lfkit.api.corrections.write_kcorrect_response", lambda **kw: kw["name"])
    monkeypatch.setattr("lfkit.api.corrections.list_available_responses", lambda out_dir: ["myresp"])

    meta = Corrections.register_kcorrect_response(
        name="myresp",
        wave_angst=np.array([4000.0, 5000.0]),
        throughput=np.array([0.1, 0.2]),
        out_dir=tmp_path,
        validate_visible_to_kcorrect=True,
    )
    assert meta["response"] == "myresp"
    assert str(tmp_path) in meta["out_dir"]
    assert meta["path"].endswith("myresp.dat")


def test_register_kcorrect_response_raises_if_not_discoverable(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Tests that register_kcorrect_response raises if not discoverable."""
    monkeypatch.setattr("lfkit.api.corrections.write_kcorrect_response", lambda **kw: kw["name"])
    monkeypatch.setattr("lfkit.api.corrections.list_available_responses", lambda out_dir: ["other"])

    with pytest.raises(RuntimeError, match="not discoverable"):
        Corrections.register_kcorrect_response(
            name="myresp",
            wave_angst=np.array([4000.0, 5000.0]),
            throughput=np.array([0.1, 0.2]),
            out_dir=tmp_path,
            validate_visible_to_kcorrect=True,
        )
