"""Unit tests for `lfkit.corrections.kcorrect_backend` module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lfkit.corrections.kcorrect_backend as backend


def test_kc_cache_key_normalizes_values():
    """Tests that _kc_cache_key normalizes response_dir and numeric values."""
    key = backend._kc_cache_key(
        responses_in=("a",),
        responses_out=("b",),
        responses_map=("c",),
        response_dir=".",
        redshift_range=(0, 2),
        nredshift=4000,
        abcorrect=False,
    )

    assert isinstance(key, tuple)
    assert key[0] == ("a",)
    assert key[1] == ("b",)
    assert key[2] == ("c",)
    assert key[4] == (0.0, 2.0)
    assert key[5] == 4000
    assert key[6] is False


def test_build_kcorrect_passes_defaults(monkeypatch):
    """Tests that build_kcorrect forwards defaults correctly to kk.Kcorrect."""
    calls = {}

    class DummyKC:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(backend.kk, "Kcorrect", DummyKC)
    monkeypatch.setattr(backend, "require_responses", lambda *a, **k: None)
    monkeypatch.setattr(
        backend.inspect,
        "signature",
        lambda x: SimpleNamespace(parameters={}),
    )

    backend._build_kcorrect_cached.cache_clear()

    backend.build_kcorrect(responses_in=["r"])

    assert calls["responses"] == ["r"]
    assert calls["responses_out"] == ["r"]
    assert calls["responses_map"] == ["r"]
    assert calls["redshift_range"] == [0.0, 2.0]
    assert calls["nredshift"] == 4000
    assert calls["abcorrect"] is False


def test_build_kcorrect_calls_require_responses(monkeypatch):
    """Tests that require_responses is called for all response groups."""
    calls = []

    def fake_require(resps, response_dir):
        calls.append((tuple(resps), response_dir))

    monkeypatch.setattr(backend, "require_responses", fake_require)

    class DummyKC:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(backend.kk, "Kcorrect", DummyKC)
    monkeypatch.setattr(
        backend.inspect,
        "signature",
        lambda x: SimpleNamespace(parameters={}),
    )

    backend._build_kcorrect_cached.cache_clear()

    backend.build_kcorrect(responses_in=["a"], responses_out=["b"], responses_map=["c"])

    assert (("a",), None) in calls
    assert (("b",), None) in calls
    assert (("c",), None) in calls


def test_build_kcorrect_with_response_dir(monkeypatch, tmp_path):
    """Tests that response_dir is passed to kk.Kcorrect when supported."""
    calls = {}

    class DummyKC:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(backend.kk, "Kcorrect", DummyKC)
    monkeypatch.setattr(backend, "require_responses", lambda *a, **k: None)

    monkeypatch.setattr(
        backend.inspect,
        "signature",
        lambda x: SimpleNamespace(parameters={"response_dir": None}),
    )

    backend._build_kcorrect_cached.cache_clear()

    backend.build_kcorrect(
        responses_in=["r"],
        response_dir=tmp_path,
    )

    assert calls["response_dir"] == str(tmp_path)


def test_build_kcorrect_raises_if_response_dir_not_supported(monkeypatch, tmp_path):
    """Tests that response_dir raises if kcorrect build does not support it."""

    class DummyKC:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(backend.kk, "Kcorrect", DummyKC)
    monkeypatch.setattr(backend, "require_responses", lambda *a, **k: None)

    monkeypatch.setattr(
        backend.inspect,
        "signature",
        lambda x: SimpleNamespace(parameters={}),
    )

    backend._build_kcorrect_cached.cache_clear()

    with pytest.raises(TypeError):
        backend.build_kcorrect(
            responses_in=["r"],
            response_dir=tmp_path,
        )


def test_build_kcorrect_is_cached(monkeypatch):
    """Tests that identical build_kcorrect calls return cached instance."""
    instances = []

    class DummyKC:
        def __init__(self, **kwargs):
            instances.append(self)

    monkeypatch.setattr(backend.kk, "Kcorrect", DummyKC)
    monkeypatch.setattr(backend, "require_responses", lambda *a, **k: None)
    monkeypatch.setattr(
        backend.inspect,
        "signature",
        lambda x: SimpleNamespace(parameters={}),
    )

    backend._build_kcorrect_cached.cache_clear()

    a = backend.build_kcorrect(responses_in=["r"])
    b = backend.build_kcorrect(responses_in=["r"])

    assert a is b
    assert len(instances) == 1
