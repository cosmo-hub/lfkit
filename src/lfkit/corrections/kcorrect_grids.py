"""kcorrect k(z) grid generation and interpolation utilities.

This module builds tables of k(z) values for one or more **anchors** on a
specified redshift grid and provides helpers to interpolate those tables.

An *anchor* is simply a label associated with a vector of kcorrect template
coefficients. Each coefficient vector represents a particular template
mixture and therefore defines a specific spectral energy distribution.
Evaluating kcorrect with those coefficients produces the corresponding
k(z) curve for a given set of filter responses.

The main purpose of this module is to turn a small set of anchors into
reusable k(z) grids that can be cached and quickly interpolated during
analysis. The resulting tables store k(z) as a function of redshift for
each anchor and response band, allowing fast lookup without repeatedly
calling the kcorrect solver.

Anchors intentionally carry **no semantic meaning** here. They are not
interpreted as galaxy types or populations; they are simply convenient
labels for different template mixtures used to generate k(z) curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lfkit.utils.interpolation import Interpolator, build_1d_interpolator

from .kcorrect_backend import build_kcorrect


def compute_k_table(
    *,
    kc,
    z_grid: np.ndarray,
    coeffs_by_anchor: dict[str, np.ndarray],
    band_shift: float | None = None,
    anchor_z0: bool = True,
) -> dict[str, np.ndarray]:
    """Compute k(z) values on a redshift grid for each anchor.

    This function evaluates k-corrections on a fixed redshift grid using a set
    of precomputed template coefficient vectors (“anchors”). Each anchor
    represents a particular SED mixture and therefore defines a specific
    k(z) curve. The output is a table of k(z) values for every anchor across
    the provided grid, which can later be used for interpolation or lookup.

    By default the table is normalized so that k(z=0)=0 for each band. This
    convention is commonly used when working with k-corrections because it
    removes arbitrary offsets and ensures that all curves are anchored to
    the same reference point.
    """
    z = np.asarray(z_grid, float)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)):
        raise ValueError("z_grid must be a finite 1D array with >=2 points.")

    out: dict[str, np.ndarray] = {}

    for label, coeffs in coeffs_by_anchor.items():
        c = np.asarray(coeffs, float).reshape(-1)

        rows: list[np.ndarray] = []
        nband: int | None = None

        for zi in z:
            zi = float(zi)

            if zi == 0.0 and anchor_z0:
                if nband is None:
                    if band_shift is None:
                        test = kc.kcorrect(redshift=1e-6, coeffs=c)
                    else:
                        test = kc.kcorrect(redshift=1e-6, coeffs=c, band_shift=float(band_shift))
                    nband = int(np.asarray(test, float).size)
                kcorr = np.zeros(nband, dtype=float)
            else:
                if band_shift is None:
                    kcorr = kc.kcorrect(redshift=zi, coeffs=c)
                else:
                    kcorr = kc.kcorrect(redshift=zi, coeffs=c, band_shift=float(band_shift))
                kcorr = np.asarray(kcorr, float)
                if nband is None:
                    nband = int(kcorr.size)

            rows.append(np.asarray(kcorr, float))

        karr = np.vstack(rows)
        if karr.shape[0] != z.size:
            raise ValueError(f"BUG: built K with Nz={karr.shape[0]} "
                             f"but z has Nz={z.size}")

        if anchor_z0:
            if not np.any(np.isfinite(karr)):
                raise ValueError(f"All-NaN K grid for anchor={label!r}.")
            karr = karr - karr[0:1, :]

        out[str(label)] = karr

    return out


def build_kcorr_grid_package(
    *,
    responses_in: list[str],
    responses_out: list[str] | None,
    responses_map: list[str] | None,
    coeffs_by_anchor: dict[str, np.ndarray],
    z_grid: np.ndarray,
    band_shift: float | None = 0.1,
    response_dir: str | None = None,
    redshift_range: tuple[float, float] = (0.0, 3.5),
    nredshift: int = 4000,
) -> dict[str, Any]:
    """Build a packaged k(z) grid for a set of anchors.

    This function generates a self-contained data package holding k(z)
    tables evaluated on a common redshift grid for multiple anchors.
    Each anchor corresponds to a template mixture that defines a specific
    spectral energy distribution and therefore a specific k(z) curve.

    The resulting package contains the redshift grid, the computed k(z)
    tables, the filter responses used, and basic metadata describing the
    setup. It is intended to serve as a reusable object that can be saved,
    cached, or passed to interpolation routines without recomputing the
    k-corrections.
    """
    if responses_out is None:
        responses_out = responses_in
    if responses_map is None:
        responses_map = responses_in

    kc = build_kcorrect(
        responses_in=responses_in,
        responses_out=responses_out,
        responses_map=responses_map,
        response_dir=response_dir,
        redshift_range=redshift_range,
        nredshift=nredshift,
    )

    nt = int(kc.templates.restframe_flux.shape[0])

    for label, c in coeffs_by_anchor.items():
        c = np.asarray(c, float)
        if c.shape != (nt,):
            raise ValueError(f"{label}: coeff shape {c.shape} != ({nt},)")
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{label}: coeffs contain non-finite values.")
        if np.any(c < 0):
            raise ValueError(f"{label}: negative coeffs not allowed.")
        if float(np.sum(c)) <= 0:
            raise ValueError(f"{label}: coeffs sum <= 0.")

    z_grid = np.asarray(z_grid, float)
    K = compute_k_table(
        kc=kc,
        z_grid=z_grid,
        coeffs_by_anchor=coeffs_by_anchor,
        band_shift=band_shift,
        anchor_z0=True,
    )

    return dict(
        meta=dict(
            backend="kcorrect",
            band_shift=band_shift,
            redshift_range=(float(z_grid[0]), float(z_grid[-1])),
            nredshift=int(nredshift),
            response_dir=str(response_dir) if response_dir is not None else None,
        ),
        z=z_grid,
        responses_in=list(map(str, responses_in)),
        responses_out=list(map(str, responses_out)),
        responses_map=list(map(str, responses_map)),
        anchors=sorted(list(map(str, coeffs_by_anchor.keys()))),
        K=K,  # mapping: anchor_label -> (Nz, Nband)
    )


def kcorr_interpolators(
    pkg: dict[str, Any],
    *,
    method: str = "pchip",
    extrapolate: bool = True,
) -> dict[str, dict[str, Interpolator | None]]:
    """Create interpolation functions for k(z).

    This function converts a precomputed k(z) grid package into a set of
    interpolators that return k(z) for any redshift within the grid range.
    An interpolator is produced for each combination of anchor and output
    response band.

    The resulting structure allows k-corrections to be evaluated quickly
    without recomputing the underlying tables, making it convenient to
    integrate precomputed grids into analysis workflows that repeatedly
    query k(z) values.
    """
    z = np.asarray(pkg["z"], float)
    responses_out = list(pkg["responses_out"])
    anchors = list(pkg["anchors"])

    out: dict[str, dict[str, Interpolator | None]] = {}
    for label in anchors:
        ktz = np.asarray(pkg["K"][label], float)  # (Nz, Nband)

        if ktz.shape[0] != z.size:
            raise ValueError(
                f"Shape mismatch for anchor={label!r}:"
                f"K has Nz={ktz.shape[0]} vs z={z.size}."
            )
        if ktz.shape[1] != len(responses_out):
            raise ValueError(
                f"Shape mismatch for anchor={label!r}: "
                f"K has Nband={ktz.shape[1]} "
                f"vs responses_out={len(responses_out)}."
            )

        out[str(label)] = {}
        for j, band in enumerate(responses_out):
            y = np.asarray(ktz[:, j], float)
            ok = np.isfinite(z) & np.isfinite(y)

            if np.count_nonzero(ok) < 2:
                out[str(label)][str(band)] = None
                continue

            out[str(label)][str(band)] = build_1d_interpolator(
                z[ok],
                y[ok],
                method=method,
                extrapolate=extrapolate,
                extrap_mode="linear_tail",
            )

    return out
