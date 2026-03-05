"""Poggianti (1997) k- and e-correction tables and interpolators.

Loads Poggianti (1997) k- and e-correction curves from CSV tables, builds
interpolators for those curves (PCHIP, Akima, or linear), and optionally remaps
the e-correction redshift grid to a target cosmology by matching lookback time.

Table I/O and parsing live in ``lfkit.utils.io``.
Interpolation utilities live in ``lfkit.utils.interpolation``.
Unit conversions live in ``lfkit.utils.units``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lfkit.cosmo.cosmology import cosmo_object as build_cosmo, lookback_time_gyr
from lfkit.utils.interpolation import (
    InterpMethod,
    Interpolator,
    build_1d_interpolator,
    prep_strictly_increasing_xy,
)
from lfkit.utils.io import (
    POGGIANTI1997_PKG,
    available_from_table,
    extract_series,
    load_vizier_csv,
    resolve_packaged_csv,
)
from lfkit.utils.units import h0_km_s_mpc_to_gyr_inv

__all__ = (
    "available_pairs",
    "load_poggianti1997_tables",
    "describe_poggianti1997_available",
    "poggianti1997_time_since_bb_gyr",
    "poggianti1997_lookback_time_gyr",
    "z_from_lookback_time",
    "poggianti1997_to_accelerating_redshift",
    "make_kcorr_interpolator",
    "make_ecorr_interpolator",
    "extract_sed_spectrum",
)


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


def load_poggianti1997_tables(
    *,
    band: str = "r",
    sed: str = "E",
    kcorr_path: str | Path | None = None,
    ecorr_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Poggianti (1997) k- and e-correction curves for a band and SED.

    Args:
        band: Band/filter label in the tables (e.g. ``"r"``, ``"B"``, ``"V"``).
        sed: SED column label (e.g. ``"E"``, ``"Sa"``, ``"Sc"``).
        kcorr_path: Optional path to ``kcorr.csv``. If not provided, the
            packaged file is used.
        ecorr_path: Optional path to ``ecorr.csv``. If not provided, the
            packaged file is used.

    Returns:
        Tuple ``(z_k, kcorr, z_e, ecorr)``:
        - ``z_k`` and ``kcorr`` are the k-correction curve.
        - ``z_e`` and ``ecorr`` are the e-correction curve.

    Raises:
        ValueError: If the requested (band, sed) is not available or contains
            insufficient usable samples.
    """
    if kcorr_path is None:
        kcorr_path = resolve_packaged_csv("kcorr.csv", pkg=POGGIANTI1997_PKG)
    if ecorr_path is None:
        ecorr_path = resolve_packaged_csv("ecorr.csv", pkg=POGGIANTI1997_PKG)

    ktab = load_vizier_csv(kcorr_path)
    etab = load_vizier_csv(ecorr_path)

    z_k, kcorr = extract_series(ktab, band=band, sed=sed)
    z_e, ecorr = extract_series(etab, band=band, sed=sed)
    return z_k, kcorr, z_e, ecorr


def describe_poggianti1997_available(
    *,
    kcorr_path: str | Path | None = None,
    ecorr_path: str | Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Summarize available bands and SED columns in Poggianti CSV tables.

    Args:
        kcorr_path: Optional path to ``kcorr.csv``. If not provided, the
            packaged file is used.
        ecorr_path: Optional path to ``ecorr.csv``. If not provided, the
            packaged file is used.

    Returns:
        A dictionary with keys ``"kcorr"`` and ``"ecorr"``. Each contains:
        - ``"bands"``: list of available band labels
        - ``"seds"``: list of available SED column labels
    """
    if kcorr_path is None:
        kcorr_path = resolve_packaged_csv("kcorr.csv", pkg=POGGIANTI1997_PKG)
    if ecorr_path is None:
        ecorr_path = resolve_packaged_csv("ecorr.csv", pkg=POGGIANTI1997_PKG)

    ktab = load_vizier_csv(kcorr_path)
    etab = load_vizier_csv(ecorr_path)

    k_bands, k_seds = available_from_table(ktab)
    e_bands, e_seds = available_from_table(etab)
    return {
        "kcorr": {"bands": k_bands, "seds": k_seds},
        "ecorr": {"bands": e_bands, "seds": e_seds},
    }


def poggianti1997_time_since_bb_gyr(z: np.ndarray | float) -> np.ndarray:
    """Return cosmic time since the Big Bang in the Poggianti (1997) cosmology.

    Args:
        z: Redshift value(s).

    Returns:
        Cosmic time since the Big Bang at each redshift, in Gyr.

    Notes:
        This uses the decelerating cosmology assumed by Poggianti (1997)
        (q0 = 0.225, H0 = 50 km/s/Mpc). It is intended for lookback-time
        matching when remapping e-about to a different cosmology.
    """
    q0 = 0.225
    h0_km_s_mpc = 50.0

    z = np.asarray(z, dtype=float)
    h0_gyr_inv = h0_km_s_mpc_to_gyr_inv(h0_km_s_mpc)

    term1 = -4.0 * q0 / (h0_gyr_inv * np.power(1.0 - 2.0 * q0, 1.5))
    root_val = np.sqrt((1.0 + 2.0 * q0 * z) / (1.0 - 2.0 * q0))
    term2 = root_val
    term3 = 2.0 * (1.0 - (1.0 + 2.0 * q0 * z) / (1.0 - 2.0 * q0))
    term4 = 0.25 * np.log(np.abs((1.0 + root_val) / (1.0 - root_val)))

    return term1 * (term2 / term3 + term4)


def poggianti1997_lookback_time_gyr(z: np.ndarray | float) -> np.ndarray:
    """Return lookback time in the Poggianti (1997) cosmology.

    Args:
        z: Redshift value(s).

    Returns:
        Lookback time to each redshift (relative to z=0), in Gyr.
    """
    z = np.asarray(z, dtype=float)
    return poggianti1997_time_since_bb_gyr(0.0) - poggianti1997_time_since_bb_gyr(z)


def _build_tlb_to_z_grid(cosmo_obj, *, zmax: float, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a monotonic mapping from lookback time to redshift for a cosmology.

    Args:
        cosmo_obj: Cosmology object accepted by ``lookback_time_gyr``.
        zmax: Maximum redshift for the mapping grid.
        nz: Number of samples used to build the mapping grid.

    Returns:
        Tuple ``(t_grid, z_grid)`` with strictly increasing ``t_grid``.
    """
    if zmax <= 0:
        raise ValueError("zmax must be > 0.")
    if nz < 32:
        raise ValueError("nz is too small; use at least ~256 in practice.")

    z_grid = np.linspace(0.0, float(zmax), int(nz))
    t_grid = lookback_time_gyr(cosmo_obj, z_grid)

    order = np.argsort(t_grid)
    t_grid = t_grid[order]
    z_grid = z_grid[order]

    keep = np.ones_like(t_grid, dtype=bool)
    keep[1:] = t_grid[1:] > t_grid[:-1]
    return t_grid[keep], z_grid[keep]


def z_from_lookback_time(
    cosmo_obj,
    t_lb_gyr: np.ndarray | float,
    *,
    zmax: float = 20.0,
    nz: int = 4096,
) -> np.ndarray:
    """Invert lookback time to redshift using a precomputed interpolation grid.

    Args:
        cosmo_obj: Cosmology object accepted by ``lookback_time_gyr``.
        t_lb_gyr: Lookback time value(s) in Gyr.
        zmax: Maximum redshift used to build the inversion grid.
        nz: Number of samples used to build the inversion grid.

    Returns:
        Redshift value(s) corresponding to ``t_lb_gyr``.

    Raises:
        ValueError: If requested lookback times fall outside the grid range.
    """
    t_lb = np.asarray(t_lb_gyr, dtype=float)
    t_grid, z_grid = _build_tlb_to_z_grid(cosmo_obj, zmax=zmax, nz=nz)

    if np.any(t_lb < t_grid[0]) or np.any(t_lb > t_grid[-1]):
        raise ValueError(
            "Target lookback time outside inversion range. "
            "Increase zmax or check inputs."
        )

    return np.interp(t_lb, t_grid, z_grid)


def poggianti1997_to_accelerating_redshift(
    z_dec: np.ndarray | float,
    *,
    cosmo_obj,
    zmax: float = 20.0,
    nz: int = 4096,
) -> np.ndarray:
    """Map Poggianti (1997) redshifts to a target cosmology via lookback time.

    Args:
        z_dec: Redshift value(s) in the Poggianti (1997) cosmology.
        cosmo_obj: Target cosmology used to compute lookback times.
        zmax: Maximum redshift used to build the inversion grid.
        nz: Number of samples used to build the inversion grid.

    Returns:
        Redshift value(s) in the target cosmology with matched lookback time.
    """
    z_dec = np.asarray(z_dec, dtype=float)
    t_lb = poggianti1997_lookback_time_gyr(z_dec)
    return z_from_lookback_time(cosmo_obj, t_lb, zmax=zmax, nz=nz)


def make_kcorr_interpolator(
    z_k: np.ndarray,
    kcorr: np.ndarray,
    *,
    method: InterpMethod = "pchip",
    extrapolate: bool = True,
    z_end: float = 20.0,
    tail: bool = True,
) -> Interpolator:
    """Create an interpolator for a Poggianti k-correction curve.

    Args:
        z_k: Redshift samples for the k-correction table.
        kcorr: K-correction values at ``z_k``.
        method: Interpolation method (``"pchip"``, ``"akima"``, or ``"linear"``).
        extrapolate: Whether to allow evaluation outside the tabulated range.
        z_end: Redshift endpoint for the optional high-z tail.
        tail: Whether to append a linear high-z tail out to ``z_end``.

    Returns:
        An interpolator callable ``K(z)``.
    """
    z = np.r_[0.0, np.asarray(z_k, float)]
    y = np.r_[0.0, np.asarray(kcorr, float)]
    z, y = prep_strictly_increasing_xy(z, y)

    if tail and z[-1] < z_end:
        i0 = max(0, z.size - 3)
        dz = z[-1] - z[i0]
        if dz <= 0:
            raise ValueError("Non-increasing z near the tail; cannot form slope.")
        slope = (y[-1] - y[i0]) / dz
        y_end = y[-1] + slope * (z_end - z[-1])
        z = np.r_[z, z_end]
        y = np.r_[y, y_end]

    return build_1d_interpolator(z, y, method=method, extrapolate=extrapolate)


def make_ecorr_interpolator(
    z_e: np.ndarray,
    ecorr: np.ndarray,
    *,
    original_z: bool,
    cosmo=None,
    zmap_zmax: float = 20.0,
    zmap_nz: int = 4096,
    method: InterpMethod = "pchip",
    extrapolate: bool = True,
) -> Interpolator:
    """Create an interpolator for a Poggianti e-correction curve.

    Args:
        z_e: Redshift samples for the e-correction table.
        ecorr: E-correction values at ``z_e``.
        original_z: If True, interpret ``z_e`` as Poggianti (1997) redshifts.
            If False, remap ``z_e`` to the target cosmology via lookback time.
        cosmo: Target cosmology used for the remapping when ``original_z=False``.
            If not provided, lfkit's default cosmology is used.
        zmap_zmax: Maximum redshift used to build the remapping inversion grid.
        zmap_nz: Number of samples used to build the remapping inversion grid.
        method: Interpolation method (``"pchip"``, ``"akima"``, or ``"linear"``).
        extrapolate: Whether to extrapolate beyond the tabulated domain.

    Returns:
        An interpolator callable ``E(z)``.
    """
    z_e = np.asarray(z_e, float)
    ecorr = np.asarray(ecorr, float)

    if not original_z:
        if cosmo is None:
            cosmo = build_cosmo()
        z_e = poggianti1997_to_accelerating_redshift(
            z_e, cosmo_obj=cosmo, zmax=zmap_zmax, nz=zmap_nz
        )

    z = np.r_[0.0, z_e]
    y = np.r_[0.0, ecorr]
    return build_1d_interpolator(z, y, method=method, extrapolate=extrapolate)


def extract_sed_spectrum(
    sed_tab: np.ndarray,
    sed_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Poggianti sed.csv column as (wave_A, flux) in rest frame.

    Args:
        sed_tab: Table returned by lfkit.utils.io.load_vizier_csv for sed.csv.
        sed_col: Column name in sed.csv, e.g. "logF03".

    Returns:
        wave_A: Wavelength grid in Å (sorted ascending).
        flux: Linear flux values (10**logF) on the same grid.

    Raises:
        ValueError: If required columns are missing or sed_col not present.
    """
    cols = list(sed_tab.dtype.names or [])
    if "Lam" not in cols:
        raise ValueError(f"sed.csv missing Lam column. cols={cols}")
    if sed_col not in cols:
        raise ValueError(f"sed.csv: sed_col={sed_col!r} not present. "
                         f"Available: {cols[:20]} ...")

    wave_nm = np.asarray(sed_tab["Lam"], float)
    logf = np.asarray(sed_tab[sed_col], float)

    ok = np.isfinite(wave_nm) & np.isfinite(logf)
    wave_A = 10.0 * wave_nm[ok]  # nm -> Å
    flux = 10.0 ** logf[ok]  # log10 -> linear

    order = np.argsort(wave_A)
    return wave_A[order], flux[order]
