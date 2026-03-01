"""I/O utilities for packaged Poggianti (1997) correction tables.

This module provides helpers to:
- locate Poggianti (1997) CSV files shipped as package data
- load VizieR-style CSV tables into NumPy structured arrays
- list available bands and SED columns
- extract validated (z, y) series for a given (band, sed)

The functions here are intentionally limited to file/resource access and table
parsing.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = (
    "POGGIANTI1997_PKG",
    "load_vizier_csv",
    "resolve_packaged_csv",
    "available_from_table",
    "extract_series",
    "available_pairs",
    "save_kcorr_package",
    "load_kcorr_package",
)

POGGIANTI1997_PKG: Final[str] = "lfkit.data.poggianti1997"


def load_vizier_csv(path: str | Path) -> np.ndarray:
    """Load a VizieR CSV file into a NumPy structured array.

    Args:
        path: Path to a CSV file written with column headers.

    Returns:
        A NumPy structured array with named columns.

    Raises:
        ValueError: If the CSV cannot be read as a headered table.
    """
    arr = np.genfromtxt(
        str(path),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if arr.dtype.names is None:
        raise ValueError(f"Failed to read CSV with headers: {path}")
    return arr


def resolve_packaged_csv(name: str, *, pkg: str = POGGIANTI1997_PKG) -> Path:
    """Resolve a packaged CSV resource to a concrete filesystem path.

    Args:
        name: Filename within the package data directory (e.g. ``kcorr.csv``).
        pkg: Package path containing the data resources.

    Returns:
        A concrete filesystem path to the resource.

    Notes:
        When running from a wheel, resources may not exist as real files.
        ``importlib.resources.as_file`` provides a temporary path-like view.
    """
    with resources.as_file(resources.files(pkg).joinpath(name)) as p:
        return Path(p)


def available_from_table(tab: np.ndarray) -> tuple[list[str], list[str]]:
    """Return available band labels and SED columns in a Poggianti-style table.

    Args:
        tab: Structured array loaded from a Poggianti VizieR CSV.

    Returns:
        A tuple ``(bands, seds)`` where:
        - ``bands`` are unique values from the ``Filt`` column (sorted)
        - ``seds`` are SED column names excluding metadata columns

    Raises:
        ValueError: If required columns are missing.
    """
    cols = list(tab.dtype.names or [])
    if "Filt" not in cols or "z" not in cols:
        raise ValueError(f"Unexpected Poggianti table columns: {cols}")

    filt_vals = np.asarray(tab["Filt"], dtype=str)
    bands = sorted({f.strip() for f in filt_vals if f.strip()})

    drop = {"recno", "z", "Filt"}
    seds = [c for c in cols if c not in drop]
    return bands, seds


def extract_series(
    tab: np.ndarray,
    *,
    band: str,
    sed: str,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a validated (z, y) correction curve for a band and SED.

    Args:
        tab: Structured array for a single correction table (k-corr or e-corr).
        band: Band label to select from the ``Filt`` column.
        sed: SED column name to extract.
        min_points: Minimum number of points required after cleaning.

    Returns:
        A tuple ``(z, y)`` where ``z`` is strictly increasing and ``y`` is the
        corresponding correction values.

    Raises:
        ValueError: If the band/SED is not present, has no finite values, or
            does not contain enough usable samples.
    """
    bands, seds = available_from_table(tab)

    if band not in bands:
        raise ValueError(f"Unknown band={band!r}. Available: {bands}")
    if sed not in seds:
        raise ValueError(f"Unknown sed={sed!r}. Available: {seds}")

    f = np.char.strip(np.asarray(tab["Filt"], dtype=str))
    mask = f == str(band)
    if not np.any(mask):
        raise ValueError(f"No rows found for band={band!r}. Available: {bands}")

    z = np.asarray(tab["z"][mask], float)
    y = np.asarray(tab[sed][mask], float)

    ok = np.isfinite(z) & np.isfinite(y)
    z = z[ok]
    y = y[ok]
    if z.size == 0:
        raise ValueError(
            f"No finite values for band={band!r}, sed={sed!r} in this table."
        )

    order = np.argsort(z)
    z = z[order]
    y = y[order]

    keep = np.ones_like(z, dtype=bool)
    keep[1:] = z[1:] > z[:-1]
    z = z[keep]
    y = y[keep]

    if z.size < int(min_points):
        raise ValueError(
            f"Too few points for band={band!r}, sed={sed!r}: n={z.size}. "
            f"Need at least {min_points}."
        )

    return z, y


def available_pairs(tab: np.ndarray, *, min_points: int = 5) -> dict[str, list[str]]:
    """List usable (band -> SEDs) pairs in a Poggianti-style table.

    Args:
        tab: Structured array for a single correction table (k-corr or e-corr).
        min_points: Minimum number of samples required per extracted series.

    Returns:
        Mapping from band label to a list of SED labels that have usable data.
    """
    bands, seds = available_from_table(tab)
    out: dict[str, list[str]] = {b: [] for b in bands}
    for b in bands:
        for s in seds:
            try:
                extract_series(tab, band=b, sed=s, min_points=min_points)
            except ValueError:
                continue
            out[b].append(s)
    return out



def save_kcorr_package(pkg: dict[str, Any], path: str | Path) -> None:
    """Save the generated package to .npz (portable, fast)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Flatten for NPZ
    z = np.asarray(pkg["z"], float)
    types = list(pkg["types"])
    responses_out = list(pkg["responses_out"])

    arrays: dict[str, np.ndarray] = {"z": z}
    for t in types:
        arrays[f"K__{t}"] = np.asarray(pkg["K"][t], float)

    # Store small metadata as numpy object array (simple + robust)
    meta = dict(pkg["meta"])
    meta.update(
        responses_in=list(pkg["responses_in"]),
        responses_out=responses_out,
        responses_map=list(pkg["responses_map"]),
        types=types,
    )
    arrays["meta"] = np.array([meta], dtype=object)

    np.savez_compressed(p, **arrays)


def load_kcorr_package(path: str | Path) -> dict[str, Any]:
    """Load a saved .npz package created by save_kcorr_package."""
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        z = np.asarray(data["z"], float)
        meta = dict(data["meta"][0])

        types = list(meta["types"])
        kcorr: dict[str, np.ndarray] = {}
        for t in types:
            kcorr[t] = np.asarray(data[f"K__{t}"], float)

    return dict(
        meta={k: meta[k] for k in meta if k not in {"responses_in", "responses_out", "responses_map", "types"}},
        z=z,
        responses_in=list(meta["responses_in"]),
        responses_out=list(meta["responses_out"]),
        responses_map=list(meta["responses_map"]),
        types=types,
        K=kcorr,
    )
