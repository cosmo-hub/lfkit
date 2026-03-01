"""Unit tests for `lfkit.utils.io` module."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.io import (
    load_vizier_csv,
    available_from_table,
    extract_series,
    available_pairs,
    save_kcorr_package,
    load_kcorr_package,
)


def _dummy_poggianti_table() -> np.ndarray:
    # Two bands, two SED columns, includes duplicates + non-finite for cleaning tests.
    dtype = [("recno", "i4"), ("z", "f8"), ("Filt", "U20"), ("E", "f8"), ("Sc", "f8")]
    rows = [
        (1, 0.0, "b1", 0.0, 0.1),
        (2, 0.1, "b1", 0.1, 0.2),
        (3, 0.1, "b1", 0.2, 0.3),  # duplicate z (should be dropped)
        (4, 0.2, "b1", np.nan, 0.4),  # non-finite E (should be dropped)
        (5, 0.3, "b1", 0.3, 0.5),
        (6, 0.4, "b1", 0.4, 0.6),
        (7, 0.0, "b2", 1.0, 1.1),
        (8, 0.1, "b2", 1.1, 1.2),
        (9, 0.2, "b2", 1.2, 1.3),
        (10, 0.3, "b2", 1.3, 1.4),
        (11, 0.4, "b2", 1.4, 1.5),
    ]
    return np.array(rows, dtype=dtype)


def test_load_vizier_csv_reads_headered_table(tmp_path):
    """Tests that load_vizier_csv reads a headered CSV into a structured array with named columns."""
    csv = tmp_path / "tab.csv"
    csv.write_text("recno,z,Filt,E,Sc\n1,0.0,b1,0.0,0.1\n2,0.1,b1,0.1,0.2\n")
    tab = load_vizier_csv(csv)
    assert tab.dtype.names is not None
    assert set(tab.dtype.names) >= {"recno", "z", "Filt", "E", "Sc"}
    assert len(tab) == 2


def test_available_from_table_returns_bands_and_seds():
    """Tests that available_from_table returns sorted unique bands and SED columns excluding metadata."""
    tab = _dummy_poggianti_table()
    bands, seds = available_from_table(tab)
    assert bands == ["b1", "b2"]
    assert set(seds) == {"E", "Sc"}


def test_available_from_table_raises_on_missing_required_cols():
    """Tests that available_from_table raises ValueError if required columns are missing."""
    tab = np.array([(0.0, "b1")], dtype=[("z", "f8"), ("Band", "U10")])
    with pytest.raises(ValueError):
        available_from_table(tab)


def test_extract_series_sorts_dedupes_and_filters_nonfinite():
    """Tests that extract_series returns strictly increasing z and filters duplicates/non-finite values."""
    tab = _dummy_poggianti_table()
    z, y = extract_series(tab, band="b1", sed="E", min_points=4)
    assert z.ndim == 1 and y.ndim == 1
    assert z.size == y.size
    assert np.all(np.isfinite(z)) and np.all(np.isfinite(y))
    assert np.all(z[1:] > z[:-1])  # strictly increasing
    # Duplicate z=0.1 and NaN E row should not survive.
    assert np.isclose(z, np.array([0.0, 0.1, 0.3, 0.4])).all()


def test_extract_series_raises_on_unknown_band_or_sed():
    """Tests that extract_series raises ValueError for unknown band or unknown sed."""
    tab = _dummy_poggianti_table()
    with pytest.raises(ValueError):
        extract_series(tab, band="nope", sed="E", min_points=2)
    with pytest.raises(ValueError):
        extract_series(tab, band="b1", sed="nope", min_points=2)


def test_extract_series_raises_on_too_few_points():
    """Tests that extract_series raises ValueError when fewer than min_points remain after cleaning."""
    tab = _dummy_poggianti_table()
    with pytest.raises(ValueError):
        extract_series(tab, band="b1", sed="E", min_points=10)


def test_available_pairs_filters_by_min_points():
    """Tests that available_pairs returns only (band, sed) combinations that satisfy min_points."""
    tab = _dummy_poggianti_table()
    pairs = available_pairs(tab, min_points=5)
    assert set(pairs.keys()) == {"b1", "b2"}
    # b1/E loses points due to NaN + duplicate
    # -> only 4 unique finite points -> excluded at min_points=5
    assert "E" not in pairs["b1"]
    assert "Sc" in pairs["b1"]
    assert set(pairs["b2"]) == {"E", "Sc"}


def test_save_and_load_kcorr_package_roundtrip(tmp_path):
    """Tests that save_kcorr_package and load_kcorr_package roundtrip core fields and shapes."""
    pkg = dict(
        meta={"tag": "x"},
        z=np.array([0.0, 0.1, 0.2]),
        responses_in=["bessell_V"],
        responses_out=["bessell_V"],
        responses_map=["bessell_V"],
        types=["E", "Sc"],
        K={
            "E": np.array([[0.0], [0.1], [0.2]]),
            "Sc": np.array([[0.0], [0.2], [0.4]]),
        },
    )

    out = tmp_path / "pkg.npz"
    save_kcorr_package(pkg, out)
    loaded = load_kcorr_package(out)

    assert np.allclose(loaded["z"], pkg["z"])
    assert loaded["types"] == pkg["types"]
    assert loaded["responses_in"] == pkg["responses_in"]
    assert loaded["responses_out"] == pkg["responses_out"]
    assert loaded["responses_map"] == pkg["responses_map"]
    for t in pkg["types"]:
        assert np.allclose(loaded["K"][t], pkg["K"][t])
    assert loaded["meta"]["tag"] == "x"
