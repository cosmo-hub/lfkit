"""Compare kcorrect K-corrections across surveys using a survey-independent red/blue split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from lfkit.api.corrections import Corrections
from lfkit.corrections.color_split import (
    fit_red_blue_anchors,
    validate_color_anchor_gr,
    validate_sed_sanity,
)


def _safe_tag(x: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+") else "-" for c in str(x))


def list_installed_kcorrect_responses() -> Set[str]:
    try:
        import kcorrect  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"Could not import kcorrect: {e}") from e

    pkg_root = Path(__import__("kcorrect").__file__).resolve().parent
    candidates = [
        pkg_root / "data" / "responses",
        pkg_root / "responses",
        pkg_root / "data",
    ]

    resp_dir: Optional[Path] = None
    for c in candidates:
        if c.exists() and c.is_dir():
            dats = list(c.rglob("*.dat"))
            if dats:
                resp_dir = c
                break

    if resp_dir is None:
        dats = list(pkg_root.rglob("*.dat"))
        if not dats:
            raise RuntimeError(f"No .dat response files found under {pkg_root}")
        resp_dir = dats[0].parent

    dat_files = list(resp_dir.rglob("*.dat"))
    return set(map(str, {f.stem for f in dat_files}))


def resolve_optical_responses(installed: Set[str]) -> Dict[str, Dict[str, Optional[str]]]:
    def pick(name: str) -> Optional[str]:
        return name if name in installed else None

    return {
        "SDSS": {"u": pick("sdss_u0"),
                 "g": pick("sdss_g0"),
                 "r": pick("sdss_r0"),
                 "i": pick("sdss_i0"),
                 "z": pick("sdss_z0")},
        "DECam": {"u": pick("decam_u"),
                  "g": pick("decam_g"),
                  "r": pick("decam_r"),
                  "i": pick("decam_i"),
                  "z": pick("decam_z")},
        "HSC": {
            "u": pick("subaru_suprimecam_u"),
            "g": pick("subaru_suprimecam_g"),
            "r": pick("subaru_suprimecam_r"),
            "i": pick("subaru_suprimecam_i"),
            "z": pick("subaru_suprimecam_z"),
        },
    }


def save_fig(fig: plt.Figure, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    cfg = {
        "out_dir": Path("output/plots/compare_kcorrect_surveys_color_split"),
        "surveys": ("SDSS", "DECam", "HSC"),
        "bands": ("u", "g", "r", "i", "z"),
        "zmax": 3.5,
        "nz": 700,
        "method": "pchip",
        "extrapolate": True,
        "e_model": "none",
        "fs": 15,
        "lw": 3,
        # Color-split defaults
        "M_r_ref": -21.5,
        "a": 0.12,
        "b": -0.025,
        "red_offset": 0.10,
        "blue_offset": 0.20,
        "mag_r_fit": 20.0,
        "ivar_level": 1e10,
    }

    anchors = fit_red_blue_anchors(
        M_r_ref=cfg["M_r_ref"],
        a=cfg["a"],
        b=cfg["b"],
        red_offset=cfg["red_offset"],
        blue_offset=cfg["blue_offset"],
        mag_r=cfg["mag_r_fit"],
        responses=None,
        ivar_level=cfg["ivar_level"],
    )

    coeffs_by_color = {
        "blue": np.asarray(anchors["blue"]["coeffs"], float),
        "red": np.asarray(anchors["red"]["coeffs"], float),
    }

    # ---- validate fitted anchor coeffs (fail fast + print what happened) ----
    for pop in ("blue", "red"):
        coeffs = coeffs_by_color[pop]
        target = float(anchors[pop]["g_minus_r"])

        v = validate_color_anchor_gr(
            coeffs=coeffs,
            g_minus_r_target=target,
            responses=None,  # SDSS u/g/r/i/z
            tol_mag=0.02,
        )
        s = validate_sed_sanity(
            coeffs=coeffs,
            responses=None,  # SDSS u/g/r/i/z
            max_abs_color=5.0,
        )

        print(
            f"\nAnchor validation: {pop}\n"
            f"  target g-r = {v['g_minus_r_target']:.4f}\n"
            f"  got    g-r = {v['g_minus_r']:.4f}  (err {v['err']:+.4f})\n"
            f"  sanity: u-g={s['u-g']:.3f}, g-r={s['g-r']:.3f}, r-i={s['r-i']:.3f}, i-z={s['i-z']:.3f}"
        )

    installed = list_installed_kcorrect_responses()
    resp = resolve_optical_responses(installed)

    print("\nResolved optical responses:")
    for survey in cfg["surveys"]:
        bits = []
        for b in cfg["bands"]:
            bits.append(f"{b}:{resp[survey][b] if resp[survey][b] is not None else '-'}")
        print(f"  {survey}: " + ", ".join(bits))

    z = np.linspace(0.0, float(cfg["zmax"]), int(cfg["nz"]))

    cmap_blue = "PiYG_r"
    cmap_red = "RdBu"
    reds = cmr.take_cmap_colors(
        cmap_blue, 3, cmap_range=(0.7, 0.90), return_fmt="hex"
    )
    blues = cmr.take_cmap_colors(
        cmap_red, 3, cmap_range=(0.70, 0.9), return_fmt="hex"
    )

    surveys = list(cfg["surveys"])  # ["SDSS","DECam","HSC"] in your cfg order

    # map (pop, survey) -> color, with blue-pop using the blue-ish group, red-pop using the red-ish group
    color_by_combo = {}
    for i, s in enumerate(surveys):
        color_by_combo[("blue", s)] = blues[i]
        color_by_combo[("red", s)] = reds[i]

    # all solid, since color carries the distinction
    linestyle_by_survey = {"SDSS": "-", "DECam": "-", "HSC": "-"}

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    n_total_plots = 0

    for band in cfg["bands"]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        n_plotted = 0

        # Plot both pops on same axes; line style encodes survey, color encodes pop.
        # We'll add ONE legend that includes all 6 combos.
        for pop in ("blue", "red"):
            coeffs = coeffs_by_color[pop]
            for survey in cfg["surveys"]:
                response_name = resp[survey][band]
                if response_name is None:
                    continue

                corr = Corrections.kcorrect(
                    z=z,
                    response=response_name,
                    coeffs=coeffs,
                    band_shift=None,
                    method=cfg["method"],
                    extrapolate=cfg["extrapolate"],
                    anchor_z0=True,
                    e_model=cfg["e_model"],
                    e_kwargs=None,
                )

                try:
                    k = np.asarray(corr.K(z), float)
                except Exception:
                    continue

                ok = np.isfinite(z) & np.isfinite(k)
                if np.count_nonzero(ok) < 2:
                    continue

                ax.plot(
                    z[ok],
                    k[ok],
                    lw=cfg["lw"],
                    color=color_by_combo[(pop, survey)],
                    ls=linestyle_by_survey.get(survey, "-"),
                    label=f"{survey} {pop}",
                )
                n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel("Redshift $z$", fontsize=cfg["fs"])
        ax.set_ylabel("$k$-correction [mag]", fontsize=cfg["fs"])
        ax.set_title(
            f"${band}$ band (SDSS $g-r$ anchor)",
            fontsize=cfg["fs"] + 2,
        )
        ax.tick_params(labelsize=cfg["fs"] - 2)
        ax.legend(frameon=True, fontsize=cfg["fs"] - 3, loc="upper left")
        fig.tight_layout()

        out_base = (f"kcorr__band-"
                    f"{_safe_tag(band)}__blue-red-split"
                    f"__surveys-linestyles.png")
        save_fig(fig, cfg["out_dir"] / out_base)
        plt.close(fig)

        n_total_plots += 1

    if n_total_plots == 0:
        raise RuntimeError("No optical curves plotted. "
                           "Check response availability and mappings.")

    print("\nDone. Wrote to:", cfg["out_dir"])


if __name__ == "__main__":
    main()