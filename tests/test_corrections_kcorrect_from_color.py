"""Unit tests for `lfkit.corrections.kcorrect_from_color` module."""

from __future__ import annotations

import numpy as np
import pytest

import lfkit.corrections.kcorrect_from_color as kcmod


def test_kcorrect_from_bandcolor_validates_z():
    """Tests that invalid z arrays raise ValueError."""
    with pytest.raises(ValueError):
        kcmod.kcorrect_from_bandcolor(
            z=np.array([1.0]),
            response_out="r",
            color=("g", "r"),
            color_value=0.5,
        )

    with pytest.raises(ValueError):
        kcmod.kcorrect_from_bandcolor(
            z=np.array([0.0, np.nan]),
            response_out="r",
            color=("g", "r"),
            color_value=0.5,
        )


def test_kcorrect_calls_fit_coeffs(monkeypatch):
    """Tests that fit_coeffs_from_bandcolor is called with expected inputs."""
    called = {}

    def fake_fit(**kwargs):
        called.update(kwargs)
        return np.array([1.0, 2.0]), ["g", "r"]

    monkeypatch.setattr(kcmod, "fit_coeffs_from_bandcolor", fake_fit)

    class DummyKC:
        def kcorrect(self, redshift, coeffs, **kw):
            return [redshift]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    z = np.linspace(0, 1, 5)

    kcmod.kcorrect_from_bandcolor(
        z=z,
        response_out="r",
        color=("g", "r"),
        color_value=0.5,
    )

    assert called["color"] == ("g", "r")
    assert called["color_value"] == 0.5


def test_kcorrect_calls_backend(monkeypatch):
    """Tests that build_kcorrect backend is used to compute K(z)."""

    monkeypatch.setattr(
        kcmod,
        "fit_coeffs_from_bandcolor",
        lambda **kw: (np.array([1.0]), ["g", "r"]),
    )

    calls = []

    class DummyKC:
        def kcorrect(self, redshift, coeffs, **kw):
            calls.append(redshift)
            return [redshift]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    z = np.linspace(0, 1, 4)

    zout, K = kcmod.kcorrect_from_bandcolor(
        z=z,
        response_out="r",
        color=("g", "r"),
        color_value=0.3,
    )

    assert len(calls) == len(z)
    assert np.allclose(zout, z)


def test_kcorrect_band_shift_branch(monkeypatch):
    """Tests that band_shift argument is forwarded to backend."""

    monkeypatch.setattr(
        kcmod,
        "fit_coeffs_from_bandcolor",
        lambda **kw: (np.array([1.0]), ["g", "r"]),
    )

    band_shift_calls = []

    class DummyKC:
        def kcorrect(self, redshift, coeffs, band_shift=None):
            band_shift_calls.append(band_shift)
            return [redshift]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    z = np.linspace(0, 1, 3)

    kcmod.kcorrect_from_bandcolor(
        z=z,
        response_out="r",
        color=("g", "r"),
        color_value=0.5,
        band_shift=0.1,
    )

    assert all(x == 0.1 for x in band_shift_calls)


def test_kcorrect_anchor_z0(monkeypatch):
    """Tests that anchor_z0 subtracts the first valid K(z)."""

    monkeypatch.setattr(
        kcmod,
        "fit_coeffs_from_bandcolor",
        lambda **kw: (np.array([1.0]), ["g", "r"]),
    )

    class DummyKC:
        def kcorrect(self, redshift, coeffs, **kw):
            return [redshift + 1.0]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    z = np.array([0.0, 1.0, 2.0])

    zout, K = kcmod.kcorrect_from_bandcolor(
        z=z,
        response_out="r",
        color=("g", "r"),
        color_value=0.5,
    )

    assert np.isclose(K[0], 0.0)
    assert np.isclose(K[1], 1.0)


def test_kcorrect_filters_nonfinite(monkeypatch):
    """Tests that non-finite backend values are removed."""

    monkeypatch.setattr(
        kcmod,
        "fit_coeffs_from_bandcolor",
        lambda **kw: (np.array([1.0]), ["g", "r"]),
    )

    class DummyKC:
        def kcorrect(self, redshift, coeffs, **kw):
            if redshift == 1.0:
                return [np.nan]
            return [redshift]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    z = np.array([0.0, 1.0, 2.0])

    zout, K = kcmod.kcorrect_from_bandcolor(
        z=z,
        response_out="r",
        color=("g", "r"),
        color_value=0.5,
    )

    assert len(zout) == 2


def test_kcorrect_raises_if_too_few_finite(monkeypatch):
    """Tests that function raises if <2 finite K(z) values."""

    monkeypatch.setattr(
        kcmod,
        "fit_coeffs_from_bandcolor",
        lambda **kw: (np.array([1.0]), ["g", "r"]),
    )

    class DummyKC:
        def kcorrect(self, redshift, coeffs, **kw):
            return [np.nan]

    monkeypatch.setattr(kcmod, "build_kcorrect", lambda **kw: DummyKC())

    with pytest.raises(ValueError):
        kcmod.kcorrect_from_bandcolor(
            z=np.array([0.0, 1.0, 2.0]),
            response_out="r",
            color=("g", "r"),
            color_value=0.5,
        )
