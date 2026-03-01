"""Unit tests for `lfkit.corrections.kcorrect` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from lfkit.corrections.kcorrect import (
    _kc_cache_key,
    _cached_available_responses,
    _discover_response_dir_auto,
    build_kcorrect,
    compute_k_table,
    kcorr_interpolators,
    list_available_responses,
    require_responses,
)


def test_kc_cache_key_is_hashable_and_stable() -> None:
    """Tests that _kc_cache_key returns a stable, hashable tuple for same inputs."""
    key1 = _kc_cache_key(
        responses_in=("a", "b"),
        responses_out=("c",),
        responses_map=("a",),
        response_dir=None,
        redshift_range=(0.0, 1.0),
        nredshift=123,
        abcorrect=False,
    )
    key2 = _kc_cache_key(
        responses_in=("a", "b"),
        responses_out=("c",),
        responses_map=("a",),
        response_dir=None,
        redshift_range=(0.0, 1.0),
        nredshift=123,
        abcorrect=False,
    )
    assert key1 == key2
    assert isinstance(key1, tuple)
    hash(key1)


def test_cached_available_responses_raises_for_missing_dir(tmp_path: Path) -> None:
    """Tests that _cached_available_responses raises FileNotFoundError for nonexistent directory."""
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        _cached_available_responses(str(missing))


def test_list_available_responses_uses_explicit_dir(tmp_path: Path) -> None:
    """Tests that list_available_responses lists stems of *.dat files in an explicit directory."""
    (tmp_path / "sdss_r0.dat").write_text("x")
    (tmp_path / "decam_g.dat").write_text("x")
    out = list_available_responses(tmp_path)
    assert out == ["decam_g", "sdss_r0"]


def test_require_responses_accepts_present_responses(tmp_path: Path) -> None:
    """Tests that require_responses does not raise when all responses exist."""
    (tmp_path / "a.dat").write_text("x")
    (tmp_path / "b.dat").write_text("x")
    require_responses(["a", "b"], response_dir=tmp_path)


def test_require_responses_raises_on_missing(tmp_path: Path) -> None:
    """Tests that require_responses raises ValueError when any response is missing."""
    (tmp_path / "a.dat").write_text("x")
    with pytest.raises(ValueError):
        require_responses(["a", "b"], response_dir=tmp_path)


def test_discover_response_dir_auto_returns_existing_path_or_raises() -> None:
    """Tests that _discover_response_dir_auto returns an existing Path or raises FileNotFoundError."""
    try:
        p = _discover_response_dir_auto()
    except FileNotFoundError:
        pytest.skip("kcorrect responses not discoverable in this test environment")
    assert isinstance(p, Path)
    assert p.exists()


def test_build_kcorrect_raises_if_missing_filters(tmp_path: Path) -> None:
    """Tests that build_kcorrect raises ValueError when requested responses are missing."""
    (tmp_path / "only_this.dat").write_text("x")
    with pytest.raises(ValueError):
        build_kcorrect(
            responses_in=["only_this", "missing"],
            responses_out=None,
            responses_map=None,
            response_dir=tmp_path,
        )


def test_compute_k_table_rejects_bad_z_grid() -> None:
    """Tests that compute_k_table rejects non-finite or non-1D z_grid inputs."""
    class _KC:
        def kcorrect(self, *, redshift: float, coeffs: np.ndarray, band_shift: float | None = None):
            return np.array([0.0, 0.1], float)

    kc = _KC()
    coeffs_by_type = {"E": np.ones(3)}

    with pytest.raises(ValueError):
        compute_k_table(kc=kc, z_grid=np.array([[0.0, 0.1]]), coeffs_by_type=coeffs_by_type)

    with pytest.raises(ValueError):
        compute_k_table(kc=kc, z_grid=np.array([0.0, np.nan]), coeffs_by_type=coeffs_by_type)

    with pytest.raises(ValueError):
        compute_k_table(kc=kc, z_grid=np.array([0.0]), coeffs_by_type=coeffs_by_type)


def test_compute_k_table_anchors_z0_to_zero() -> None:
    """Tests that compute_k_table enforces K(z=0)=0 when anchor_z0 is True."""
    class _KC:
        def kcorrect(self, *, redshift: float, coeffs: np.ndarray, band_shift: float | None = None):
            # simple monotone output, two bands
            return np.array([redshift + 1.0, 2.0 * redshift + 3.0], float)

    kc = _KC()
    z = np.array([0.0, 0.2, 0.5], float)
    out = compute_k_table(
        kc=kc,
        z_grid=z,
        coeffs_by_type={"E": np.ones(4)},
        band_shift=None,
        anchor_z0=True,
    )
    k = out["E"]
    assert k.shape == (3, 2)
    assert np.allclose(k[0], 0.0)


def test_compute_k_table_no_anchor_keeps_raw_values() -> None:
    """Tests that compute_k_table does not subtract the first row when anchor_z0 is False."""
    class _KC:
        def kcorrect(self, *, redshift: float, coeffs: np.ndarray, band_shift: float | None = None):
            return np.array([redshift + 1.0, 2.0 * redshift + 3.0], float)

    kc = _KC()
    z = np.array([0.0, 0.2], float)
    out = compute_k_table(
        kc=kc,
        z_grid=z,
        coeffs_by_type={"E": np.ones(4)},
        band_shift=None,
        anchor_z0=False,
    )
    k = out["E"]
    assert np.allclose(k[0], np.array([1.0, 3.0]))


def test_compute_k_table_raises_on_all_nan_after_anchor() -> None:
    """Tests that compute_k_table raises ValueError when anchored grid is all-NaN."""
    class _KC:
        def kcorrect(self, *, redshift: float, coeffs: np.ndarray, band_shift: float | None = None):
            return np.array([np.nan, np.nan], float)

    kc = _KC()
    z = np.array([0.0, 0.2, 0.5], float)

    with pytest.raises(ValueError):
        compute_k_table(
            kc=kc,
            z_grid=z,
            coeffs_by_type={"E": np.ones(2)},
            band_shift=None,
            anchor_z0=True,
        )


def test_kcorr_interpolators_raises_on_shape_mismatch_nz() -> None:
    """Tests that kcorr_interpolators raises ValueError when Nz does not match z grid."""
    pkg = dict(
        z=np.array([0.0, 0.1, 0.2]),
        responses_out=["r"],
        types=["E"],
        K={"E": np.zeros((2, 1))},  # wrong Nz
    )
    with pytest.raises(ValueError):
        kcorr_interpolators(pkg)


def test_kcorr_interpolators_raises_on_shape_mismatch_nband() -> None:
    """Tests that kcorr_interpolators raises ValueError when Nband does not match responses_out."""
    pkg = dict(
        z=np.array([0.0, 0.1, 0.2]),
        responses_out=["r", "i"],
        types=["E"],
        K={"E": np.zeros((3, 1))},  # wrong Nband
    )
    with pytest.raises(ValueError):
        kcorr_interpolators(pkg)


def test_kcorr_interpolators_returns_none_for_insufficient_points() -> None:
    """Tests that kcorr_interpolators returns None for a band with <2 finite points."""
    pkg = dict(
        z=np.array([0.0, 0.1, 0.2]),
        responses_out=["r"],
        types=["E"],
        K={"E": np.array([[np.nan], [np.nan], [1.0]])},  # only one finite point
    )
    out = kcorr_interpolators(pkg, method="pchip", extrapolate=True)
    assert out["E"]["r"] is None
