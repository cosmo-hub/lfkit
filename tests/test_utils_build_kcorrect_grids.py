"""Unit tests for `lfkit.utils.build_kcorrect_grids`."""

from __future__ import annotations

import numpy as np

from lfkit.utils.io import save_kcorr_package
from lfkit.corrections.kcorrect import (
    build_kcorr_grid_package,
    make_type_coeffs_from_colors,
)


def _small_setup():
    """Returns a small set of inputs for testing."""
    responses = ["bessell_U", "bessell_B", "bessell_V"]
    ref_band = "bessell_V"
    z_grid = np.linspace(0.0, 0.5, 11)
    return responses, ref_band, z_grid


def test_make_type_coeffs_from_colors_structure():
    """Tests that make_type_coeffs_from_colors returns a non-empty dict of finite 1D coefficient arrays."""
    responses, ref_band, _ = _small_setup()

    coeffs_by_type = make_type_coeffs_from_colors(
        responses=responses,
        ref_mag=20.0,
        ref_band=ref_band,
        ref_z=0.1,
    )

    assert isinstance(coeffs_by_type, dict)
    assert len(coeffs_by_type) > 0

    for coeffs in coeffs_by_type.values():
        arr = np.asarray(coeffs, dtype=float)
        assert arr.ndim == 1
        assert arr.size > 0
        assert np.all(np.isfinite(arr))


def test_build_kcorr_grid_package_basic_invariants():
    """Tests that build_kcorr_grid_package returns a structured package with finite k grids matching z_grid."""
    responses, ref_band, z_grid = _small_setup()

    coeffs_by_type = make_type_coeffs_from_colors(
        responses=responses,
        ref_mag=20.0,
        ref_band=ref_band,
        ref_z=0.1,
    )

    pkg = build_kcorr_grid_package(
        responses_in=responses,
        responses_out=responses,
        responses_map=responses,
        coeffs_by_type=coeffs_by_type,
        z_grid=z_grid,
        band_shift=None,
        redshift_range=(float(z_grid[0]), float(z_grid[-1])),
        nredshift=200,
    )

    assert "types" in pkg
    assert "K" in pkg
    assert "responses_out" in pkg
    assert list(pkg["responses_out"]) == responses

    for t in pkg["types"]:
        K = np.asarray(pkg["K"][t], dtype=float)
        assert K.shape[0] == len(z_grid)
        assert np.all(np.isfinite(K))


def test_k_at_z0_is_finite():
    """Tests that k(z=0) values are finite for all types and bands."""
    responses, ref_band, z_grid = _small_setup()

    coeffs_by_type = make_type_coeffs_from_colors(
        responses=responses,
        ref_mag=20.0,
        ref_band=ref_band,
        ref_z=0.1,
    )

    pkg = build_kcorr_grid_package(
        responses_in=responses,
        responses_out=responses,
        responses_map=responses,
        coeffs_by_type=coeffs_by_type,
        z_grid=z_grid,
        band_shift=None,
        redshift_range=(float(z_grid[0]), float(z_grid[-1])),
        nredshift=200,
    )

    for t in pkg["types"]:
        K0 = np.asarray(pkg["K"][t])[0]
        assert np.all(np.isfinite(K0))


def test_save_kcorr_package_roundtrip(tmp_path):
    """Tests that save_kcorr_package writes a readable .npz file."""
    responses, ref_band, z_grid = _small_setup()

    coeffs_by_type = make_type_coeffs_from_colors(
        responses=responses,
        ref_mag=20.0,
        ref_band=ref_band,
        ref_z=0.1,
    )

    pkg = build_kcorr_grid_package(
        responses_in=responses,
        responses_out=responses,
        responses_map=responses,
        coeffs_by_type=coeffs_by_type,
        z_grid=z_grid,
        band_shift=None,
        redshift_range=(float(z_grid[0]), float(z_grid[-1])),
        nredshift=200,
    )

    out_file = tmp_path / "test_pkg.npz"
    save_kcorr_package(pkg, out_file)

    assert out_file.exists()

    loaded = np.load(out_file, allow_pickle=True)
    assert len(loaded.files) > 0
