"""Unit tests for `lfkit.corrections.poggianti1997` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lfkit.corrections.poggianti1997 import (
    _build_tlb_to_z_grid,
    available_pairs,
    describe_poggianti1997_available,
    load_poggianti1997_tables,
    make_ecorr_interpolator,
    make_kcorr_interpolator,
    poggianti1997_lookback_time_gyr,
    poggianti1997_time_since_bb_gyr,
    poggianti1997_to_accelerating_redshift,
    z_from_lookback_time,
)


def test_available_pairs_returns_mapping_with_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that available_pairs returns a dict of band -> list of SED labels."""
    import lfkit.corrections.poggianti1997 as mod

    # Minimal structured array: only needs to exist; parsing is mocked.
    tab = np.array([(0.0,)], dtype=[("z", "f8")])

    # Pretend the table contains two bands and two SEDs
    monkeypatch.setattr(mod, "available_from_table", lambda _tab: (["r", "i"], ["E", "Sa"]))

    # Make extract_series succeed only for a subset so we test filtering
    def _fake_extract_series(_tab, *, band, sed, min_points=5):
        if (band, sed) in {("r", "E"), ("i", "Sa")}:
            z = np.array([0.0, 0.1, 0.2], float)
            y = np.array([0.0, 0.1, 0.2], float)
            return z, y
        raise ValueError("nope")

    monkeypatch.setattr(mod, "extract_series", _fake_extract_series)

    out = available_pairs(tab, min_points=3)

    assert out == {"r": ["E"], "i": ["Sa"]}
    assert isinstance(out, dict)
    assert all(isinstance(v, list) for v in out.values())


def test_describe_poggianti1997_available_has_expected_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that describe_poggianti1997_available returns dict with kcorr/ecorr keys."""
    # Fake tables + I/O so we don't depend on packaged data paths in unit tests.
    fake = np.array([(0.0, 0.0)], dtype=[("z", "f8"), ("r_E", "f8")])

    def _fake_resolve(*args, **kwargs):
        return tmp_path / "x.csv"

    def _fake_load(*args, **kwargs):
        return fake

    def _fake_available(tab):
        return (["r"], ["E"])

    import lfkit.corrections.poggianti1997 as mod

    monkeypatch.setattr(mod, "resolve_packaged_csv", _fake_resolve)
    monkeypatch.setattr(mod, "load_vizier_csv", _fake_load)
    monkeypatch.setattr(mod, "available_from_table", _fake_available)

    out = describe_poggianti1997_available()
    assert set(out.keys()) == {"kcorr", "ecorr"}
    assert set(out["kcorr"].keys()) == {"bands", "seds"}
    assert set(out["ecorr"].keys()) == {"bands", "seds"}


def test_load_poggianti1997_tables_calls_extract_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that load_poggianti1997_tables returns four arrays from extract_series."""
    fake_tab = np.array([(0.0, 0.0)], dtype=[("z", "f8"), ("r_E", "f8")])
    z = np.array([0.0, 0.1, 0.2], float)
    yk = np.array([0.0, 0.2, 0.5], float)
    ye = np.array([0.0, -0.1, -0.2], float)

    def _fake_resolve(*args, **kwargs):
        return tmp_path / "x.csv"

    def _fake_load(*args, **kwargs):
        return fake_tab

    calls = {"n": 0}

    def _fake_extract(tab, *, band, sed, min_points=5):
        calls["n"] += 1
        if calls["n"] == 1:
            return z, yk
        return z, ye

    import lfkit.corrections.poggianti1997 as mod

    monkeypatch.setattr(mod, "resolve_packaged_csv", _fake_resolve)
    monkeypatch.setattr(mod, "load_vizier_csv", _fake_load)
    monkeypatch.setattr(mod, "extract_series", _fake_extract)

    z_k, kcorr, z_e, ecorr = load_poggianti1997_tables(band="r", sed="E")
    assert np.allclose(z_k, z)
    assert np.allclose(kcorr, yk)
    assert np.allclose(z_e, z)
    assert np.allclose(ecorr, ye)


def test_poggianti1997_time_since_bb_gyr_is_finite_and_nonnegative_at_z0() -> None:
    """Tests that poggianti1997_time_since_bb_gyr returns finite values and is nonnegative at z=0."""
    t0 = poggianti1997_time_since_bb_gyr(0.0)
    assert np.all(np.isfinite(t0))
    assert np.all(t0 >= 0.0)


def test_poggianti1997_lookback_time_gyr_is_zero_at_z0() -> None:
    """Tests that poggianti1997_lookback_time_gyr returns zero at z=0."""
    tlb0 = poggianti1997_lookback_time_gyr(0.0)
    assert np.allclose(tlb0, 0.0)


def test_build_tlb_to_z_grid_rejects_bad_params() -> None:
    """Tests that _build_tlb_to_z_grid raises ValueError for invalid zmax or nz."""
    class _Cosmo:
        pass

    with pytest.raises(ValueError):
        _build_tlb_to_z_grid(_Cosmo(), zmax=0.0, nz=4096)
    with pytest.raises(ValueError):
        _build_tlb_to_z_grid(_Cosmo(), zmax=1.0, nz=16)


def test_build_tlb_to_z_grid_returns_strictly_increasing_t() -> None:
    """Tests that _build_tlb_to_z_grid returns a strictly increasing lookback-time grid."""
    # Patch lookback_time_gyr locally through the module under test.
    import lfkit.corrections.poggianti1997 as mod

    def _fake_lookback(_cosmo, z):
        # strictly increasing function of z
        z = np.asarray(z, float)
        return 2.0 * z + 0.1

    class _Cosmo:
        pass

    orig = mod.lookback_time_gyr
    mod.lookback_time_gyr = _fake_lookback
    try:
        t, z = _build_tlb_to_z_grid(_Cosmo(), zmax=2.0, nz=64)
    finally:
        mod.lookback_time_gyr = orig

    assert t.ndim == 1 and z.ndim == 1
    assert t.size == z.size
    assert np.all(t[1:] > t[:-1])


def test_z_from_lookback_time_raises_outside_range() -> None:
    """Tests that z_from_lookback_time raises ValueError when t_lb is outside the inversion grid."""
    import lfkit.corrections.poggianti1997 as mod

    def _fake_lookback(_cosmo, z):
        z = np.asarray(z, float)
        return z  # t in [0, zmax]

    class _Cosmo:
        pass

    orig = mod.lookback_time_gyr
    mod.lookback_time_gyr = _fake_lookback
    try:
        with pytest.raises(ValueError):
            z_from_lookback_time(_Cosmo(), np.array([-1.0]), zmax=2.0, nz=128)
        with pytest.raises(ValueError):
            z_from_lookback_time(_Cosmo(), np.array([10.0]), zmax=2.0, nz=128)
    finally:
        mod.lookback_time_gyr = orig


def test_z_from_lookback_time_inverts_monotone_mapping() -> None:
    """Tests that z_from_lookback_time approximately inverts a monotone lookback-time mapping."""
    import lfkit.corrections.poggianti1997 as mod

    def _fake_lookback(_cosmo, z):
        z = np.asarray(z, float)
        return 3.0 * z  # invertible scaling

    class _Cosmo:
        pass

    orig = mod.lookback_time_gyr
    mod.lookback_time_gyr = _fake_lookback
    try:
        t = np.array([0.0, 0.6, 3.0], float)
        z = z_from_lookback_time(_Cosmo(), t, zmax=2.0, nz=512)
        assert np.allclose(z, t / 3.0, atol=1e-2)
    finally:
        mod.lookback_time_gyr = orig


def test_poggianti1997_to_accelerating_redshift_calls_inversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that poggianti1997_to_accelerating_redshift delegates through z_from_lookback_time."""
    import lfkit.corrections.poggianti1997 as mod

    def _fake_pog_tlb(z):
        z = np.asarray(z, float)
        return 1.0 + 0.0 * z

    def _fake_z_from(_cosmo, t, *, zmax, nz):
        t = np.asarray(t, float)
        return np.full_like(t, 0.123)

    monkeypatch.setattr(mod, "poggianti1997_lookback_time_gyr", _fake_pog_tlb)
    monkeypatch.setattr(mod, "z_from_lookback_time", _fake_z_from)

    z = poggianti1997_to_accelerating_redshift(np.array([0.0, 0.5]), cosmo_obj=object())
    assert np.allclose(z, 0.123)


def test_make_kcorr_interpolator_anchors_zero_and_is_callable() -> None:
    """Tests that make_kcorr_interpolator anchors K(0)=0 and returns a callable interpolator."""
    z_k = np.array([0.1, 0.2, 0.3], float)
    k = np.array([0.2, 0.5, 0.9], float)

    K = make_kcorr_interpolator(z_k, k, method="linear", extrapolate=True, tail=False)
    assert callable(K)
    assert np.allclose(float(K(0.0)), 0.0)


def test_make_kcorr_interpolator_appends_tail_when_requested() -> None:
    """Tests that make_kcorr_interpolator appends a linear tail to z_end when enabled."""
    z_k = np.array([0.1, 0.2, 0.3], float)
    k = np.array([0.2, 0.5, 0.9], float)

    K = make_kcorr_interpolator(z_k, k, method="linear", extrapolate=True, z_end=2.0, tail=True)
    assert np.isfinite(float(K(2.0)))


def test_make_kcorr_interpolator_raises_on_nonincreasing_tail_slope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that make_kcorr_interpolator raises ValueError when tail slope cannot be formed."""
    import lfkit.corrections.poggianti1997 as mod

    # Return a "prepared" grid whose last three z values are equal -> dz=0 in tail slope calc
    def _bad_prep(z, y):
        z = np.array([0.0, 0.1, 0.1, 0.1], float)
        y = np.array([0.0, 0.2, 0.2, 0.2], float)
        return z, y

    monkeypatch.setattr(mod, "prep_strictly_increasing_xy", _bad_prep)

    z_k = np.array([0.1, 0.1, 0.1], float)
    k = np.array([0.2, 0.2, 0.2], float)

    with pytest.raises(ValueError, match="Non-increasing z near the tail"):
        make_kcorr_interpolator(z_k, k, method="linear", extrapolate=True, z_end=2.0, tail=True)


def test_make_ecorr_interpolator_original_z_does_not_require_cosmo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that make_ecorr_interpolator works without cosmology when original_z is True."""
    import lfkit.corrections.poggianti1997 as mod

    # Ensure we would notice if mapping was called.
    monkeypatch.setattr(mod, "poggianti1997_to_accelerating_redshift", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("called")))

    z_e = np.array([0.1, 0.2, 0.3], float)
    e = np.array([-0.1, -0.2, -0.3], float)
    E = make_ecorr_interpolator(z_e, e, original_z=True, method="linear", extrapolate=True)
    assert callable(E)
    assert np.allclose(float(E(0.0)), 0.0)


def test_make_ecorr_interpolator_remaps_when_original_z_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that make_ecorr_interpolator remaps z_e through lookback-time matching when requested."""
    import lfkit.corrections.poggianti1997 as mod

    def _fake_map(z, *, cosmo_obj, zmax, nz):
        z = np.asarray(z, float)
        return z + 0.5

    monkeypatch.setattr(mod, "poggianti1997_to_accelerating_redshift", _fake_map)

    z_e = np.array([0.1, 0.2, 0.3], float)
    e = np.array([-0.1, -0.2, -0.3], float)
    E = make_ecorr_interpolator(
        z_e,
        e,
        original_z=False,
        cosmo=object(),
        zmap_zmax=5.0,
        zmap_nz=256,
        method="linear",
        extrapolate=True,
    )
    assert callable(E)
    assert np.isfinite(float(E(0.25)))
