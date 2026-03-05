"""Response / filter-curve utilities for kcorrect.

This module handles:
- discovering where kcorrect response .dat files live
- listing and validating available responses
- writing new response .dat files in kcorrect format (for custom surveys)
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


@lru_cache(maxsize=8)
def discover_response_dir_auto() -> Path:
    """Locate the kcorrect response directory (AUTO discovery), cached.

    Strategy:
    1) Probe a Kcorrect instance for common directory attributes.
    2) Fallback: search inside installed kcorrect package for ``*.dat`` files.

    Returns:
        Path to a directory containing ``*.dat`` response files.

    Raises:
        FileNotFoundError: If no response directory can be found.
    """
    try:
        import kcorrect.kcorrect as kk  # local import to avoid heavy import at module load

        kc = kk.Kcorrect(responses=["sdss_r0"])
        for attr in (
            "response_dir",
            "responses_dir",
            "response_dirname",
            "filters_dir",
            "filter_dir",
            "response_path",
        ):
            if hasattr(kc, attr):
                p = Path(getattr(kc, attr))
                if p.exists():
                    return p
    except Exception:
        pass

    try:
        import kcorrect as _kcorrect_pkg  # type: ignore
    except Exception as e:
        raise FileNotFoundError(
            "Could not import kcorrect to discover response directory automatically."
        ) from e

    base = Path(_kcorrect_pkg.__file__).resolve().parent
    candidates = [
        base,
        base / "data",
        base / "responses",
        base / "response",
        base / "filters",
    ]

    for c in candidates:
        if c.exists() and any(c.glob("*.dat")):
            return c

    dats = list(base.rglob("*.dat"))
    if not dats:
        raise FileNotFoundError(
            "Could not locate kcorrect response .dat files automatically. "
            "Pass response_dir=... explicitly."
        )

    # choose the most common parent directory among *.dat files
    from collections import Counter

    parent = Counter([p.parent for p in dats]).most_common(1)[0][0]
    return Path(parent)


def _normalize_response_dir_key(response_dir: str | Path | None) -> str:
    if response_dir is None:
        return "__AUTO__"
    return str(Path(response_dir).resolve())


@lru_cache(maxsize=32)
def _cached_available_responses(response_dir_key: str) -> tuple[str, ...]:
    """Cached list of available responses for a directory key."""
    if response_dir_key == "__AUTO__":
        rdir = discover_response_dir_auto()
    else:
        rdir = Path(response_dir_key)

    if not rdir.exists():
        raise FileNotFoundError(f"response_dir does not exist: {rdir}")

    return tuple(sorted(p.stem for p in rdir.glob("*.dat")))


def list_available_responses(response_dir: str | Path | None = None) -> list[str]:
    """List available kcorrect response (filter) names."""
    key = _normalize_response_dir_key(response_dir)
    return list(_cached_available_responses(key))


def require_responses(
    responses: Iterable[str],
    response_dir: str | Path | None = None,
) -> None:
    """Validate that requested response names exist in response_dir (or AUTO)."""
    key = _normalize_response_dir_key(response_dir)
    avail = set(_cached_available_responses(key))
    missing = [str(r) for r in responses if str(r) not in avail]
    if missing:
        example = sorted(list(avail))[:25]
        raise ValueError(
            f"Missing kcorrect responses: {missing}. Example available: {example} ..."
        )


def kcorrect_supports_response_dir() -> bool:
    """Return True if installed kcorrect.Kcorrect supports response_dir=..."""
    try:
        import kcorrect.kcorrect as kk

        sig = inspect.signature(kk.Kcorrect)
        return "response_dir" in sig.parameters
    except Exception:
        return False


def write_kcorrect_response(
    *,
    name: str,
    wave_angst: np.ndarray,
    throughput: np.ndarray,
    out_dir: str | Path,
    normalize: bool = True,
) -> str:
    """Write a kcorrect-compatible response file.

    Args:
        name: Response name (file will be {name}.dat).
        wave_angst: Wavelength array in Å.
        throughput: Dimensionless throughput (0..1).
        out_dir: Output directory to write the .dat file into.
        normalize: If True, normalize throughput to max=1.

    Returns:
        The response name (same as input name).
    """
    wave_A = np.asarray(wave_angst, float)
    thr = np.asarray(throughput, float)

    if wave_A.ndim != 1 or thr.ndim != 1:
        raise ValueError("wave_angst and throughput must be 1D arrays.")
    if wave_A.size != thr.size:
        raise ValueError("wave_angst and throughput must have same length.")
    if wave_A.size < 10:
        raise ValueError("Response curve must contain >=10 points.")

    order = np.argsort(wave_A)
    wave_A = wave_A[order]
    thr = thr[order]

    thr = np.clip(thr, 0.0, None)
    if normalize and thr.max() > 0:
        thr = thr / thr.max()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.dat"

    header = (
        "# kcorrect response file\n"
        "# wavelength [Angstrom]   throughput\n"
    )

    np.savetxt(out_path, np.column_stack([wave_A, thr]), header=header)
    return str(name)