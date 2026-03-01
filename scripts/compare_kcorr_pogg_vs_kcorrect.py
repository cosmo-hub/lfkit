"""Compare Poggianti (1997) and kcorrect K-corrections on a common redshift grid.

This script is a sanity/consistency check for LFKit’s correction backends. It
evaluates Poggianti’s tabulated K(z) and kcorrect-based K(z) constructed from
Poggianti’s representative SEDs, then plots both curves per band and galaxy
type. Both backends are treated the same way at the edges by using the same
interpolation and linear-tail extrapolation strategy, so differences reflect
model/content rather than interpolation artifacts.

The output is one figure per band (PNG + PDF) saved under the configured
output directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

from lfkit.api.corrections import Corrections
from lfkit.utils.io import (
    POGGIANTI1997_PKG,
    load_vizier_csv,
    resolve_packaged_csv,
)


# Mapping: Poggianti "type" -> sed.csv column at z=0
# NOTE: I keep only E here. Sc is chosen dynamically in main()
# because I need to figure out what is the best sed.csv column for Sc.
POGGIANTI_SED0_COL = {"E": "logF03"}  # earliest

# Candidate sed.csv columns to try for Poggianti 'Sc'
SC_CANDIDATE_SED_COLS = ("logF04", "logF05", "logF06", "logF07", "logF08", "logF09", "logF10")

# Poggianti band label -> kcorrect response name
POGGIANTI_TO_KCORRECT = {
    "bessell": {"U": "bessell_U", "B": "bessell_B", "V": "bessell_V", "R": "bessell_R", "I": "bessell_I"},
    "sdss": {"u": "sdss_u0", "g": "sdss_g0", "r": "sdss_r0", "i": "sdss_i0", "z": "sdss_z0"},
}


def _normalize_poggianti_band(band: str) -> str:
    """Normalizes a band label to the Poggianti naming convention."""
    mapping = {"u": "U", "b": "B", "v": "V", "r": "R", "i": "I", "g": "g"}
    return mapping.get(band, band)


def _load_poggianti_tables() -> tuple[np.ndarray, np.ndarray]:
    """Loads the Poggianti (1997) k-correction and SED tables."""
    kcorr_path = resolve_packaged_csv("kcorr.csv", pkg=POGGIANTI1997_PKG)
    sed_path = resolve_packaged_csv("sed.csv", pkg=POGGIANTI1997_PKG)
    return load_vizier_csv(kcorr_path), load_vizier_csv(sed_path)


def extract_sed_spectrum(sed_tab: np.ndarray, sed_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Poggianti SED column as a usable rest-frame spectrum.

    Poggianti’s sed.csv provides log-flux columns sampled on a wavelength grid.
    kcorrect expects a rest-frame spectrum in linear flux units on an Ångstrom
    wavelength grid for template-coefficient fitting. This helper converts the
    table column into (wavelength [Å], flux [linear]) with basic NaN filtering
    and sorting so downstream fitting is stable and reproducible.
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
    wave_nm = wave_nm[ok]
    logf = logf[ok]

    wave_A = 10.0 * wave_nm  # nm -> Angstrom
    flux = 10.0 ** logf

    order = np.argsort(wave_A)
    return wave_A[order], flux[order]


def choose_best_sc_sed_col(
    *,
    sed_tab: np.ndarray,
    band: str,
    system: str,
    z: np.ndarray,
    band_shift: float | None,
    method: str,
    extrapolate: bool,
    z_fit_max: float = 1.2,
    weighted_fit: bool = False,
) -> tuple[str, dict[str, float]]:
    """Choose the sed.csv column that makes kcorrect mimic Poggianti “Sc” most closely.

    Poggianti labels a galaxy type as “Sc”, but the bundled sed.csv provides a
    set of candidate SED columns that do not necessarily map one-to-one onto
    that label in kcorrect template space. For a fair comparison, we select the
    sed.csv column that yields the smallest RMS difference between:
      - Poggianti’s tabulated Sc k(z), and
      - kcorrect k(z) computed from that SED,
    over a conservative redshift range (z <= z_fit_max) where both approaches
    are expected to be well-behaved.

    This is a pragmatic calibration step for the comparison plot; it is not a
    statement about Poggianti’s original classification scheme.
    """
    band_pog = _normalize_poggianti_band(band)

    corr_pog = Corrections.poggianti1997(
        band=band_pog,
        sed="Sc",
        method=method,
        extrapolate=extrapolate,
    )
    k_pog = corr_pog.K(z)

    resp_map = POGGIANTI_TO_KCORRECT.get(system, {})
    band_key = band_pog if system == "bessell" else band
    if band_key not in resp_map:
        raise ValueError(f"band={band!r} not supported for system={system!r}. Have: {sorted(resp_map)}")
    response = resp_map[band_key]

    mask = z <= float(z_fit_max)
    if np.count_nonzero(mask) < 5:
        raise ValueError("z grid too small for selection.")

    scores: dict[str, float] = {}
    for sed_col in SC_CANDIDATE_SED_COLS:
        sed_wave_A, sed_flux = extract_sed_spectrum(sed_tab, sed_col=sed_col)

        corr_kc = Corrections.kcorrect_from_sed(
            z=z,
            response=response,
            sed_wave_A=sed_wave_A,
            sed_flux=sed_flux,
            band_shift=band_shift,
            weighted_fit=weighted_fit,
            method=method,
            extrapolate=extrapolate,
            anchor_z0=True,
        )
        k_kcorr = corr_kc.K(z)

        dk = (k_kcorr - k_pog)[mask]
        ok = np.isfinite(dk)
        if np.count_nonzero(ok) < 5:
            continue
        scores[sed_col] = float(np.sqrt(np.mean(dk[ok] ** 2)))

    if not scores:
        raise ValueError("No usable Sc sed_col candidates produced finite scores.")

    best = min(scores, key=scores.get)
    return best, scores


def _safe_tag(x: str) -> str:
    """Sanitizes a string so it is safe to use in filenames."""
    return "".join(c if (c.isalnum() or c in "-_+") else "-" for c in str(x))


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Saves a Matplotlib figure to PNG and a matching PDF alongside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    """Run the comparison and write one plot per band.

    This routine defines the redshift grid and the set of bands/types to plot,
    loads the packaged Poggianti tables, and performs a small calibration step
    to pick the most compatible sed.csv column for the “Sc” type under kcorrect.
    It then evaluates K(z) for each band and each type using:
      - Poggianti tables via the LFKit Corrections API, and
      - kcorrect computed from the chosen SEDs via the LFKit Corrections API,
    and saves side-by-side curves so discrepancies are easy to diagnose.

    The plots are intended for development/validation: they help identify
    band-mapping issues, template/SED mismatches, anchoring conventions, or
    redshift-range/extrapolation artifacts.
    """
    cfg = {
        "bands": ("u", "b", "v", "r", "i"),
        "systems": ("bessell",),
        "sed": ("E", "Sc"),
        "zmax": 3.5,
        "nz": 401,
        "band_shift": None,
        "weighted_fit": False,
        "method": "pchip",
        "extrapolate": True,
        "out_dir": Path("output/plots/compare_kcorrect_poggianti97"),
    }

    z = np.linspace(0.0, float(cfg["zmax"]), int(cfg["nz"]))
    kcorr_tab, sed_tab = _load_poggianti_tables()

    best_sc_col, sc_scores = choose_best_sc_sed_col(
        sed_tab=sed_tab,
        band="v",
        system=cfg["systems"][0],
        z=z,
        band_shift=cfg["band_shift"],
        method=cfg["method"],
        extrapolate=cfg["extrapolate"],
        z_fit_max=1.2,
        weighted_fit=cfg["weighted_fit"],
    )
    print("Best Sc sed.csv column:", best_sc_col)
    print("Sc candidate RMS (mag) over z<=1.2:", sc_scores)

    sed0_col = {"E": POGGIANTI_SED0_COL["E"], "Sc": best_sc_col}

    lw, fs = 3, 15
    systems_tag = "+".join(_safe_tag(s) for s in cfg["systems"])

    for band in cfg["bands"]:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        band_pog = _normalize_poggianti_band(band)

        colors = cmr.take_cmap_colors(
            "cmr.infinity_s",
            len(cfg["sed"]),
            cmap_range=(0.5, 1.0),
            return_fmt="hex",
        )

        for i, sed in enumerate(cfg["sed"]):
            sed_col = sed0_col[sed]
            sed_wave_A, sed_flux = extract_sed_spectrum(sed_tab, sed_col=sed_col)

            corr_pog = Corrections.poggianti1997(
                band=band_pog,
                sed=sed,
                method=cfg["method"],
                extrapolate=cfg["extrapolate"],
            )
            k_pog = corr_pog.K(z)

            system = cfg["systems"][0]
            response = POGGIANTI_TO_KCORRECT[system][band_pog]
            corr_kc = Corrections.kcorrect_from_sed(
                z=z,
                response=response,
                sed_wave_A=sed_wave_A,
                sed_flux=sed_flux,
                band_shift=cfg["band_shift"],
                weighted_fit=cfg["weighted_fit"],
                method=cfg["method"],
                extrapolate=cfg["extrapolate"],
                anchor_z0=True,
            )
            k_kcorr = corr_kc.K(z)

            ax.plot(z, k_pog, lw=lw, ls="--", color=colors[i], label=f"${sed}$ Poggianti97")
            ax.plot(z, k_kcorr, lw=lw, ls="-", color=colors[i], label=f"${sed}$ kcorrect")

        ax.set_xlabel("Redshift $z$", fontsize=fs)
        ax.set_ylabel("$k$-correction [mag]", fontsize=fs)
        ax.set_title(f"system: {cfg['systems'][0]}, band: ${band}$", fontsize=fs + 2)
        ax.tick_params(labelsize=fs - 2)
        ax.legend(frameon=True, fontsize=fs - 2, loc="upper left")
        fig.tight_layout()

        base = (
            f"sed-{'-'.join(_safe_tag(s) for s in cfg['sed'])}"
            f"__zmax-{cfg['zmax']}"
            f"__systems-{systems_tag}"
            f"__{cfg['method']}"
            f"__extrap-{int(cfg['extrapolate'])}"
            f"__band-{_safe_tag(band)}"
        )
        save_fig(fig, cfg["out_dir"] / f"Kcurves__{base}.png")
        plt.close(fig)

    plt.show()


if __name__ == "__main__":
    main()