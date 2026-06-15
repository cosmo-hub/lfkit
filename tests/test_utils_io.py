"""Unit tests for `lfkit.utils.io.py``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.io import (
    POGGIANTI1997_PKG,
    available_from_table,
    available_pairs,
    extract_series,
    load_kcorr_package,
    load_vizier_csv,
    resolve_packaged_csv,
    save_kcorr_package,
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


def test_io_exports_expected_public_names() -> None:
    """Tests that io exposes the expected public API names."""
    import lfkit.utils.io as io

    expected = {
        "POGGIANTI1997_PKG",
        "load_vizier_csv",
        "resolve_packaged_csv",
        "available_from_table",
        "extract_series",
        "available_pairs",
        "save_kcorr_package",
        "load_kcorr_package",
    }

    assert set(io.__all__) == expected


def test_poggianti1997_package_name_is_expected() -> None:
    """Tests that the Poggianti package-data namespace is stable."""
    assert POGGIANTI1997_PKG == "lfkit.data.poggianti1997"


def test_load_vizier_csv_accepts_string_path(tmp_path) -> None:
    """Tests that load_vizier_csv accepts string paths."""
    csv = tmp_path / "tab.csv"
    csv.write_text("recno,z,Filt,E\n1,0.0,b1,0.1\n")

    tab = load_vizier_csv(str(csv))

    assert tab.dtype.names is not None
    assert set(tab.dtype.names) >= {"recno", "z", "Filt", "E"}


def test_resolve_packaged_csv_returns_existing_package_path() -> None:
    """Tests that resolve_packaged_csv returns a path for packaged resources."""
    path = resolve_packaged_csv("__init__.py", pkg="lfkit")

    assert path.name == "__init__.py"
    assert path.exists()


def test_available_from_table_strips_blank_band_labels() -> None:
    """Tests that available_from_table strips band labels and drops blanks."""
    dtype = [("z", "f8"), ("Filt", "U20"), ("E", "f8")]
    tab = np.array(
        [
            (0.0, " b2 ", 0.1),
            (0.1, "b1", 0.2),
            (0.2, "   ", 0.3),
        ],
        dtype=dtype,
    )

    bands, seds = available_from_table(tab)

    assert bands == ["b1", "b2"]
    assert seds == ["E"]


def test_available_from_table_excludes_recno_z_and_filt_only() -> None:
    """Tests that available_from_table excludes only metadata columns from SEDs."""
    dtype = [
        ("recno", "i4"),
        ("z", "f8"),
        ("Filt", "U20"),
        ("E", "f8"),
        ("Sa", "f8"),
    ]
    tab = np.array([(1, 0.0, "b1", 0.1, 0.2)], dtype=dtype)

    _, seds = available_from_table(tab)

    assert seds == ["E", "Sa"]


def test_extract_series_returns_float64_arrays() -> None:
    """Tests that extract_series returns float64 arrays."""
    tab = _dummy_poggianti_table()

    z, y = extract_series(tab, band="b2", sed="Sc", min_points=2)

    assert z.dtype == np.float64
    assert y.dtype == np.float64


def test_extract_series_strips_table_band_values() -> None:
    """Tests that extract_series matches bands after stripping table values."""
    dtype = [("z", "f8"), ("Filt", "U20"), ("E", "f8")]
    tab = np.array(
        [
            (0.0, " b1 ", 0.0),
            (0.1, " b1 ", 0.1),
            (0.2, " b1 ", 0.2),
        ],
        dtype=dtype,
    )

    z, y = extract_series(tab, band="b1", sed="E", min_points=3)

    np.testing.assert_allclose(z, np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(y, np.array([0.0, 0.1, 0.2]))


def test_extract_series_rejects_all_nonfinite_values() -> None:
    """Tests that extract_series rejects selections with no finite values."""
    dtype = [("z", "f8"), ("Filt", "U20"), ("E", "f8")]
    tab = np.array(
        [
            (0.0, "b1", np.nan),
            (0.1, "b1", np.inf),
        ],
        dtype=dtype,
    )

    with pytest.raises(ValueError, match="No finite values"):
        extract_series(tab, band="b1", sed="E", min_points=1)


def test_extract_series_keeps_first_duplicate_after_sorting() -> None:
    """Tests that extract_series keeps the first duplicate after sorting."""
    dtype = [("z", "f8"), ("Filt", "U20"), ("E", "f8")]
    tab = np.array(
        [
            (0.0, "b1", 0.0),
            (0.1, "b1", 1.0),
            (0.1, "b1", 9.0),
            (0.2, "b1", 2.0),
        ],
        dtype=dtype,
    )

    z, y = extract_series(tab, band="b1", sed="E", min_points=3)

    np.testing.assert_allclose(z, np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(y, np.array([0.0, 1.0, 2.0]))


def test_available_pairs_returns_empty_lists_for_unusable_bands() -> None:
    """Tests that available_pairs keeps bands even when no SEDs are usable."""
    dtype = [("z", "f8"), ("Filt", "U20"), ("E", "f8")]
    tab = np.array(
        [
            (0.0, "b1", 0.0),
            (0.1, "b1", 0.1),
            (0.0, "b2", np.nan),
            (0.1, "b2", np.nan),
        ],
        dtype=dtype,
    )

    pairs = available_pairs(tab, min_points=2)

    assert pairs == {"b1": ["E"], "b2": []}


def test_save_kcorr_package_creates_parent_directories(tmp_path) -> None:
    """Tests that save_kcorr_package creates missing parent directories."""
    pkg = dict(
        meta={"tag": "x"},
        z=np.array([0.0, 0.1]),
        responses_in=["r"],
        responses_out=["r"],
        responses_map=["r"],
        types=["E"],
        K={"E": np.array([[0.0], [0.1]])},
    )
    out = tmp_path / "nested" / "pkg.npz"

    save_kcorr_package(pkg, out)

    assert out.exists()


def test_save_and_load_kcorr_package_accepts_string_path(tmp_path) -> None:
    """Tests that k-correction package I/O accepts string paths."""
    pkg = dict(
        meta={"tag": "x"},
        z=np.array([0.0, 0.1]),
        responses_in=["r"],
        responses_out=["r"],
        responses_map=["r"],
        types=["E"],
        K={"E": np.array([[0.0], [0.1]])},
    )
    out = tmp_path / "pkg.npz"

    save_kcorr_package(pkg, str(out))
    loaded = load_kcorr_package(str(out))

    np.testing.assert_allclose(loaded["z"], pkg["z"])
    np.testing.assert_allclose(loaded["K"]["E"], pkg["K"]["E"])


def test_load_kcorr_package_removes_internal_metadata_from_meta(tmp_path) -> None:
    """Tests that load_kcorr_package removes internal metadata keys from meta."""
    pkg = dict(
        meta={"tag": "x", "version": 1},
        z=np.array([0.0, 0.1]),
        responses_in=["r_in"],
        responses_out=["r_out"],
        responses_map=["r_map"],
        types=["E"],
        K={"E": np.array([[0.0], [0.1]])},
    )
    out = tmp_path / "pkg.npz"

    save_kcorr_package(pkg, out)
    loaded = load_kcorr_package(out)

    assert loaded["meta"] == {"tag": "x", "version": 1}
    assert "responses_in" not in loaded["meta"]
    assert "responses_out" not in loaded["meta"]
    assert "responses_map" not in loaded["meta"]
    assert "types" not in loaded["meta"]
