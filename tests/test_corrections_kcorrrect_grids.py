"""Unit tests for `lfkit.corrections.kcorrect_grid` utilities."""

from __future__ import annotations

import numpy as np
import pytest

import lfkit.corrections.kcorrect_grids as grid


class DummyKC:
    """Simple fake kcorrect backend."""

    def __init__(self, nband=2, nt=3):
        self.nband = nband
        self.templates = type("T", (), {})()
        self.templates.restframe_flux = np.zeros((nt, 10))

    def kcorrect(self, redshift, coeffs, **kw):
        # deterministic fake K(z)
        return np.full(self.nband, redshift)


def test_compute_k_table_validates_z():
    """Tests that compute_k_table rejects invalid z grids."""
    kc = DummyKC()

    with pytest.raises(ValueError):
        grid.compute_k_table(
            kc=kc,
            z_grid=np.array([1.0]),
            coeffs_by_anchor={"a": np.ones(3)},
        )

    with pytest.raises(ValueError):
        grid.compute_k_table(
            kc=kc,
            z_grid=np.array([0.0, np.nan]),
            coeffs_by_anchor={"a": np.ones(3)},
        )


def test_compute_k_table_basic_behavior():
    """Tests that compute_k_table returns expected shape per anchor."""
    kc = DummyKC()

    z = np.linspace(0, 1, 4)
    coeffs = {"a": np.ones(3)}

    out = grid.compute_k_table(
        kc=kc,
        z_grid=z,
        coeffs_by_anchor=coeffs,
    )

    assert "a" in out
    assert out["a"].shape == (len(z), kc.nband)


def test_compute_k_table_anchor_z0_zeroes_first_row():
    """Tests that anchor_z0 subtracts the first row."""
    kc = DummyKC()

    z = np.array([0.0, 1.0, 2.0])
    coeffs = {"a": np.ones(3)}

    out = grid.compute_k_table(
        kc=kc,
        z_grid=z,
        coeffs_by_anchor=coeffs,
        anchor_z0=True,
    )

    assert np.allclose(out["a"][0], 0.0)


def test_compute_k_table_multiple_anchors():
    """Tests that compute_k_table handles multiple anchor labels."""
    kc = DummyKC()

    z = np.linspace(0, 1, 5)

    coeffs = {
        "red": np.ones(3),
        "blue": np.ones(3),
    }

    out = grid.compute_k_table(
        kc=kc,
        z_grid=z,
        coeffs_by_anchor=coeffs,
    )

    assert set(out.keys()) == {"red", "blue"}


def test_build_kcorr_grid_package_calls_backend(monkeypatch):
    """Tests that build_kcorr_grid_package calls build_kcorrect."""
    kc = DummyKC()

    monkeypatch.setattr(grid, "build_kcorrect", lambda **kw: kc)

    z = np.linspace(0, 1, 4)

    pkg = grid.build_kcorr_grid_package(
        responses_in=["r"],
        responses_out=None,
        responses_map=None,
        coeffs_by_anchor={"a": np.ones(3)},
        z_grid=z,
    )

    assert pkg["z"].shape == z.shape
    assert "a" in pkg["K"]
    assert pkg["responses_out"] == ["r"]


def test_build_kcorr_grid_package_validates_coeff_shape(monkeypatch):
    """Tests that coefficient shape mismatch raises."""
    kc = DummyKC()

    monkeypatch.setattr(grid, "build_kcorrect", lambda **kw: kc)

    with pytest.raises(ValueError):
        grid.build_kcorr_grid_package(
            responses_in=["r"],
            responses_out=None,
            responses_map=None,
            coeffs_by_anchor={"a": np.ones(2)},  # wrong length
            z_grid=np.linspace(0, 1, 4),
        )


def test_build_kcorr_grid_package_rejects_negative_coeffs(monkeypatch):
    """Tests that negative coefficients are rejected."""
    kc = DummyKC()

    monkeypatch.setattr(grid, "build_kcorrect", lambda **kw: kc)

    with pytest.raises(ValueError):
        grid.build_kcorr_grid_package(
            responses_in=["r"],
            responses_out=None,
            responses_map=None,
            coeffs_by_anchor={"a": np.array([-1.0, 1.0, 1.0])},
            z_grid=np.linspace(0, 1, 4),
        )


def test_kcorr_interpolators_basic(monkeypatch):
    """Tests that kcorr_interpolators builds interpolators for valid data."""
    z = np.linspace(0, 1, 4)

    pkg = {
        "z": z,
        "responses_out": ["r", "i"],
        "anchors": ["a"],
        "K": {
            "a": np.vstack([z, z]).T,
        },
    }

    def fake_interp(x, y, **kw):
        return ("interp", x, y)

    monkeypatch.setattr(grid, "build_1d_interpolator", fake_interp)

    out = grid.kcorr_interpolators(pkg)

    assert "a" in out
    assert "r" in out["a"]
    assert out["a"]["r"][0] == "interp"


def test_kcorr_interpolators_returns_none_if_too_few_points(monkeypatch):
    """Tests that interpolator is None when fewer than two finite points."""
    z = np.array([0.0, 1.0])

    pkg = {
        "z": z,
        "responses_out": ["r"],
        "anchors": ["a"],
        "K": {
            "a": np.array([[np.nan], [np.nan]]),
        },
    }

    out = grid.kcorr_interpolators(pkg)

    assert out["a"]["r"] is None


def test_kcorr_interpolators_shape_mismatch():
    """Tests that mismatched shapes raise ValueError."""
    z = np.linspace(0, 1, 3)

    pkg = {
        "z": z,
        "responses_out": ["r"],
        "anchors": ["a"],
        "K": {
            "a": np.zeros((2, 1)),  # wrong Nz
        },
    }

    with pytest.raises(ValueError):
        grid.kcorr_interpolators(pkg)
