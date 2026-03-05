"""
Compare Poggianti (1997) and kcorrect k(z) on a common redshift grid.

- Uses Poggianti tabulated k(z) (low-level loaders).
- Fits kcorrect template coeffs directly from Poggianti sed.csv SED columns (NNLS),
  then evaluates kcorrect k(z) in the matching kcorrect response band.
- Writes ONE figure per band (PNG + PDF), under:
    output/plots/compare_pogg_to_kcorrect/<system>/

Hard-coded choices (no settings / no args):
- E SED column:  logF03
- Sc SED column: logF05   (your best match)
- z grid: 0..3.5, nz=401
- interpolation: pchip
- extrapolate: True
- anchor: enforce K(z0)=0 at the first valid point for kcorrect
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cmasher as cmr
import numpy as np


OUT_DIR = Path("output/plots/compare_poggianti_to_kcorrect")
ZMAX = 3.5
NZ = 401

POGG_METHOD = "pchip"
POGG_EXTRAPOLATE = True

WEIGHTED_FIT = False
BAND_SHIFT = None
ANCHOR_Z0 = True

# Poggianti SED columns (fixed)
SED0_COL = {
    "E": "logF03",
    "Sc": "logF05",
}

# Poggianti band label -> kcorrect response name
SYSTEMS: dict[str, dict[str, str]] = {
    "bessell": {
        "U": "bessell_U", "B": "bessell_B", "V": "bessell_V",
        "R": "bessell_R", "I": "bessell_I",
    },
    "sdss": {
        # Poggianti has g/r/i (lowercase), but NOT u/z.
        "g": "sdss_g0",
        "r": "sdss_r0",
        "i": "sdss_i0",
    },
}

# plotting order
GAL_TYPES = ("E", "Sc")


def _safe_tag(x: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+") else "-" for c in str(x))


def _save_fig(fig, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")


def _extract_poggianti_sed_column(*, sed_tab: np.ndarray, sed_col: str) -> tuple[np.ndarray, np.ndarray]:
    cols = list(sed_tab.dtype.names or [])
    if "Lam" not in cols:
        raise ValueError(f"sed.csv missing Lam column. cols={cols}")
    if sed_col not in cols:
        raise ValueError(f"sed.csv: sed_col={sed_col!r} not present. Available: {cols[:30]} ...")

    wave_nm = np.asarray(sed_tab["Lam"], float)
    logf = np.asarray(sed_tab[sed_col], float)

    ok = np.isfinite(wave_nm) & np.isfinite(logf)
    wave_A = 10.0 * wave_nm[ok]  # nm -> Å
    flux = 10.0 ** logf[ok]      # log10 -> linear

    order = np.argsort(wave_A)
    return wave_A[order], flux[order]


def fit_kcorrect_coeffs_from_sed(
    *,
    sed_wave_A: np.ndarray,
    sed_flux: np.ndarray,
    response_dir: str | Path | None = None,
    weighted_fit: bool = False,
) -> np.ndarray:
    """Fit non-negative kcorrect template coefficients (NNLS) to a rest-frame SED."""
    from scipy.optimize import nnls  # local import (dev-only)
    from lfkit.corrections.kcorrect_backend import build_kcorrect

    sed_wave_A = np.asarray(sed_wave_A, float)
    sed_flux = np.asarray(sed_flux, float)

    ok = np.isfinite(sed_wave_A) & np.isfinite(sed_flux)
    sed_wave_A = sed_wave_A[ok]
    sed_flux = sed_flux[ok]
    if sed_wave_A.size < 5:
        raise ValueError("sed_wave_A/sed_flux must contain >=5 finite points.")

    order = np.argsort(sed_wave_A)
    sed_wave_A = sed_wave_A[order]
    sed_flux = sed_flux[order]

    kc = build_kcorrect(
        responses_in=[],
        responses_out=[],
        responses_map=[],
        response_dir=response_dir,
    )

    tw = np.asarray(kc.templates.restframe_wave, float)  # [Nw]
    tf = np.asarray(kc.templates.restframe_flux, float)  # [Ntmpl, Nw]

    sed_on_tw = np.interp(
        tw,
        sed_wave_A,
        sed_flux,
        left=float(sed_flux[0]),
        right=float(sed_flux[-1]),
    )

    A = tf.T
    b = sed_on_tw

    if weighted_fit:
        w = np.sum(np.abs(tf), axis=0)
        w = np.clip(w, 0.0, None)
        if w.max() > 0:
            w = w / w.max()
        w = np.maximum(w, 1e-4)
        sw = np.sqrt(w)
        A = A * sw[:, None]
        b = b * sw

    scale = np.median(b[b > 0]) if np.any(b > 0) else np.nan
    if np.isfinite(scale) and scale > 0:
        b = b / scale

    coeffs, _ = nnls(A, b)
    return np.asarray(coeffs, float)


def eval_kcorrect_k_of_z(
    *,
    z: np.ndarray,
    response: str,
    coeffs: np.ndarray,
    band_shift: float | None = None,
    anchor_z0: bool = True,
    response_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate kcorrect k(z) on a grid, returning only the finite-supported range."""
    from lfkit.corrections.kcorrect_backend import build_kcorrect

    z = np.asarray(z, float)
    kc = build_kcorrect(
        responses_in=[str(response)],
        responses_out=[str(response)],
        responses_map=[str(response)],
        response_dir=response_dir,
    )

    k_raw = np.full_like(z, np.nan, dtype=float)
    zmax_ok = float("nan")

    for i, zi in enumerate(z):
        try:
            if band_shift is None:
                kval = kc.kcorrect(redshift=float(zi), coeffs=np.asarray(coeffs, float))
            else:
                kval = kc.kcorrect(redshift=float(zi), coeffs=np.asarray(coeffs, float), band_shift=float(band_shift))
            k_raw[i] = float(np.asarray(kval)[0])
            zmax_ok = float(zi)
        except ValueError:
            break

    ok = np.isfinite(k_raw)
    if np.count_nonzero(ok) < 2:
        raise ValueError(f"Too few finite points for kcorrect k(z): response={response!r}")

    if anchor_z0:
        i0 = int(np.where(ok)[0][0])
        k_raw[ok] = k_raw[ok] - k_raw[i0]

    meta = {
        "response": str(response),
        "band_shift": band_shift,
        "anchor_z0": bool(anchor_z0),
        "kcorrect_z_valid_max": zmax_ok,
        "n_finite": int(np.count_nonzero(ok)),
    }
    return z[ok], k_raw[ok], meta


def eval_poggianti_k_of_z(
    *,
    z: np.ndarray,
    band: str,
    gal_type: str,
    method: str = "pchip",
    extrapolate: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate Poggianti (1997) tabulated k(z) for (band, gal_type) on z grid."""
    import lfkit.corrections.poggianti1997 as pogg  # local import (dev-only)

    z_k, kcorr, _, _ = pogg.load_poggianti1997_tables(band=str(band), sed=str(gal_type))
    k_interp = pogg.make_kcorr_interpolator(
        np.asarray(z_k, float),
        np.asarray(kcorr, float),
        method=str(method),
        extrapolate=bool(extrapolate),
    )
    K = np.asarray(k_interp(np.asarray(z, float)), float)
    meta = {
        "poggianti_band": str(band),
        "poggianti_gal_type": str(gal_type),
        "poggianti_method": str(method),
        "poggianti_extrapolate": bool(extrapolate),
    }
    return K, meta


def main() -> None:
    # Load Poggianti sed.csv once
    from lfkit.utils.io import load_vizier_csv, resolve_packaged_csv

    sed_path = resolve_packaged_csv("sed.csv")
    sed_tab = load_vizier_csv(sed_path)

    z_full = np.linspace(0.0, float(ZMAX), int(NZ))

    # Pre-fit coeffs once per galaxy type from Poggianti SED columns
    coeffs_by_type: dict[str, np.ndarray] = {}
    for gt in GAL_TYPES:
        sed_col = SED0_COL[gt]
        sed_wave_A, sed_flux = _extract_poggianti_sed_column(sed_tab=sed_tab, sed_col=sed_col)
        coeffs_by_type[gt] = fit_kcorrect_coeffs_from_sed(
            sed_wave_A=sed_wave_A,
            sed_flux=sed_flux,
            response_dir=None,
            weighted_fit=WEIGHTED_FIT,
        )

    # Plot one figure per (system, band)
    import matplotlib.pyplot as plt

    for system, bandmap in SYSTEMS.items():
        for pog_band, response in bandmap.items():
            fig, ax = plt.subplots(figsize=(7.5, 5.0))

            # Evaluate once-per-type and plot
            for gt in GAL_TYPES:
                # Poggianti
                K_pog, _ = eval_poggianti_k_of_z(
                    z=z_full,
                    band=pog_band,
                    gal_type=gt,
                    method=POGG_METHOD,
                    extrapolate=POGG_EXTRAPOLATE,
                )

                # kcorrect (fit from Poggianti SED)
                z_ok, K_kc, kmeta = eval_kcorrect_k_of_z(
                    z=z_full,
                    response=response,
                    coeffs=coeffs_by_type[gt],
                    band_shift=BAND_SHIFT,
                    anchor_z0=ANCHOR_Z0,
                    response_dir=None,
                )

                # Put Poggianti on the same finite-supported grid as kcorrect
                K_pog_ok = np.asarray(K_pog[np.isin(z_full, z_ok)], float)
                if K_pog_ok.size != z_ok.size:
                    # fallback: interpolate Poggianti onto z_ok
                    K_pog_ok = np.interp(z_ok, z_full, K_pog)

                cmap = "cmr.guppy"

                c_red = cmr.take_cmap_colors(cmap, 3,
                                             cmap_range=(0.0, 0.25))[1]
                c_blue = cmr.take_cmap_colors(cmap, 3,
                                              cmap_range=(0.75, 1.0))[1]

                color = c_red if gt == "E" else c_blue
                ms = 7
                lw = 2.
                fs = 15
                ax.plot(
                    z_ok, K_kc,
                    color=color,
                    lw=lw,
                    marker="o",
                    ms=ms,
                    markevery=25,
                    label=f"${gt}$ kcorrect"
                )

                ax.plot(
                    z_ok, K_pog_ok,
                    color=color,
                    lw=lw,
                    marker="s",
                    ms=ms,
                    markevery=20,
                    label=f"${gt}$ Poggianti97"
                )

            ax.set_xlabel("Redshift $z$", fontsize=fs)
            ax.set_ylabel("$k$-correction [mag]", fontsize=fs)
            ax.set_title(f"{system}: ${pog_band}$ band", fontsize=fs)
            ax.legend(frameon=True, fontsize=fs)

            fig.tight_layout()

            out_base = (
                f"Kcurves__system-{_safe_tag(system)}"
                f"__band-{_safe_tag(pog_band)}"
                f"__zmax-{ZMAX}"
                f"__nz-{NZ}"
                f"__{_safe_tag(POGG_METHOD)}"
                f"__extrap-{int(POGG_EXTRAPOLATE)}"
                f"__anchorz0-{int(ANCHOR_Z0)}"
            )
            _save_fig(fig, OUT_DIR / system / f"{out_base}.png")
            plt.close(fig)

    print(f"Wrote plots under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()