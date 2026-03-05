"""
Compare kcorrect K-corrections across surveys using survey-native red/blue splits.

- Discovers installed kcorrect response curves (.dat)
- Resolves optical band responses for SDSS/DECam/HSC
- Defines "blue" and "red" by a simple linear cut in g-r with offsets
- For each survey and output band, computes k(z) using LFKit Corrections.kcorrect
  anchored to that survey's own (g-r) response pair
- Plots k(z) curves for (survey, pop) combos and saves PNG+PDF

Uses LFKit Corrections.kcorrect API as-is (no API changes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from lfkit.api.corrections import Corrections
from lfkit.corrections.color_anchors import fit_coeffs_from_bandcolor


def _safe_tag(x: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+") else "-" for c in str(x))


def list_installed_kcorrect_responses() -> Set[str]:
    """Return set of installed kcorrect response stems (no .dat suffix)."""
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
    """Map (survey -> band -> response_name or None)."""
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


def _survey_color_pair(
    survey: str,
    installed: Set[str],
    resp: Dict[str, Dict[str, Optional[str]]],
) -> Optional[Tuple[str, str]]:
    """Return (g_resp, r_resp) for survey if available, else None."""
    if survey not in resp:
        return None
    g = resp[survey].get("g", None)
    r = resp[survey].get("r", None)
    if g is None or r is None:
        return None
    if g not in installed or r not in installed:
        return None
    return (g, r)


def _compute_split_colors(
    *,
    M_r_ref: float,
    a: float,
    b: float,
    red_offset: float,
    blue_offset: float,
) -> Dict[str, float]:
    """Return {"red": g-r, "blue": g-r} evaluated at M_r_ref."""
    # "divider" line: (g-r)_div = a + b * (M_r - M_r_ref)
    # evaluated at M_r_ref => a
    gr_div = float(a)

    # Define target anchors as offsets from the divider.
    # Convention here (feel free to flip signs to match your intended meaning):
    # red is redder than divider; blue is bluer than divider.
    return {
        "red": gr_div + float(red_offset),
        "blue": gr_div - float(blue_offset),
    }


def _maybe_print_anchor_sanity(
    *,
    installed: Set[str],
    resp: Dict[str, Dict[str, Optional[str]]],
    surveys: Tuple[str, ...],
    pop_to_color_value: Dict[str, float],
    mag_r_fit: float,
    ivar_level: float,
) -> None:
    """Optional sanity prints: fit coeffs for (g-r) per survey and report."""
    print("\nAnchor sanity (per survey, per pop):")
    for survey in surveys:
        pair = _survey_color_pair(survey, installed, resp)
        if pair is None:
            print(f"  {survey}: missing g/r responses -> skip sanity")
            continue

        gresp, rresp = pair
        for pop in ("blue", "red"):
            gr = float(pop_to_color_value[pop])
            try:
                coeffs, fit_resps = fit_coeffs_from_bandcolor(
                    color=(gresp, rresp),
                    color_value=gr,
                    z_phot=0.0,
                    anchor_band=rresp,
                    anchor_mag=float(mag_r_fit),
                    responses=None,
                    ivar_level=float(ivar_level),
                    response_dir=None,
                    redshift_range=(0.0, 2.0),
                    nredshift=4000,
                    rescale_maggies=True,
                )
                print(
                    f"  {survey:5s} {pop:4s}: anchor {gresp}-{rresp}={gr:+.3f}  "
                    f"(fit_resps={fit_resps}, sum(coeffs)={float(np.sum(coeffs)):.3f})"
                )
            except Exception as e:
                print(f"  {survey:5s} {pop:4s}: sanity fit failed: {e}")


def main() -> None:
    cfg = {
        "out_dir": Path("output/plots/compare_kcorrect_surveys_color_split"),
        "surveys": ("SDSS", "DECam", "HSC"),
        "bands": ("u", "g", "r", "i", "z"),
        "zmax": 3.5,
        "nz": 700,
        "method": "pchip",
        "extrapolate": True,
        "fs": 15,
        "lw": 3,
        # Color-split definition (your knobs)
        "M_r_ref": -21.5,
        "a": 0.65,
        "b": -0.025,
        "red_offset": 0.10,
        "blue_offset": 0.20,
        # anchor normalization (arbitrary)
        "mag_r_fit": 20.0,
        "ivar_level": 1e10,
        # kcorrect internals
        "nredshift": 4000,
        # optional sanity prints
        "print_anchor_sanity": True,
    }

    installed = list_installed_kcorrect_responses()
    resp = resolve_optical_responses(installed)

    print("\nResolved optical responses:")
    for survey in cfg["surveys"]:
        bits = []
        for b in cfg["bands"]:
            bits.append(f"{b}:{resp[survey][b] if resp[survey][b] is not None else '-'}")
        print(f"  {survey}: " + ", ".join(bits))

    # Survey-native anchor band pairs (g, r)
    color_pair_by_survey: Dict[str, Optional[Tuple[str, str]]] = {}
    for survey in cfg["surveys"]:
        color_pair_by_survey[survey] = _survey_color_pair(survey, installed, resp)

    print("\nSurvey-native anchor color pairs (g-r):")
    for survey in cfg["surveys"]:
        pair = color_pair_by_survey[survey]
        print(f"  {survey}: {pair if pair is not None else '-'}")

    # Define the red/blue split values (single scalar per pop)
    pop_to_gr = _compute_split_colors(
        M_r_ref=float(cfg["M_r_ref"]),
        a=float(cfg["a"]),
        b=float(cfg["b"]),
        red_offset=float(cfg["red_offset"]),
        blue_offset=float(cfg["blue_offset"]),
    )

    print("targets:", pop_to_gr)  # should show blue/red g-r
    # ensure both keys exist as ("blue","red")
    pop_to_gr = {"blue": float(pop_to_gr["blue"]), "red": float(pop_to_gr["red"])}

    print("\nSplit targets (g-r):")
    print(f"  blue: {pop_to_gr['blue']:+.3f}")
    print(f"  red : {pop_to_gr['red']:+.3f}")

    if bool(cfg["print_anchor_sanity"]):
        _maybe_print_anchor_sanity(
            installed=installed,
            resp=resp,
            surveys=tuple(cfg["surveys"]),
            pop_to_color_value=pop_to_gr,
            mag_r_fit=float(cfg["mag_r_fit"]),
            ivar_level=float(cfg["ivar_level"]),
        )

    z = np.linspace(0.0, float(cfg["zmax"]), int(cfg["nz"]))

    # Colors: one color per (pop, survey)
    cmap = "cmr.guppy"

    reds = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.0, 0.25))
    blues = cmr.take_cmap_colors(cmap, 3, cmap_range=(0.75, 1.0))

    surveys = list(cfg["surveys"])
    color_by_combo: Dict[Tuple[str, str], str] = {}
    for i, s in enumerate(surveys):
        color_by_combo[("blue", s)] = blues[i]
        color_by_combo[("red", s)] = reds[i]

    linestyle_by_survey = {"SDSS": "-", "DECam": "-", "HSC": "-"}

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    n_total_plots = 0

    for band in cfg["bands"]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        n_plotted = 0

        for pop in ("blue", "red"):
            color_value = float(pop_to_gr[pop])

            for survey in cfg["surveys"]:
                response_out = resp[survey][band]
                if response_out is None:
                    continue

                cpair = color_pair_by_survey.get(survey, None)
                if cpair is None:
                    continue  # cannot anchor without g and r responses

                gresp, rresp = cpair

                try:
                    corr = Corrections.kcorrect(
                        z_grid=z,
                        response_out=str(response_out),
                        color=(str(gresp), str(rresp)),
                        color_value=float(color_value),
                        z_phot=0.0,
                        anchor_band=str(rresp),
                        anchor_mag=float(cfg["mag_r_fit"]),
                        band_shift=None,
                        response_dir=None,
                        redshift_range=(0.0, float(cfg["zmax"])),
                        nredshift=int(cfg["nredshift"]),
                        ivar_level=float(cfg["ivar_level"]),
                        anchor_z0=True,
                        method=str(cfg["method"]),
                        extrapolate=bool(cfg["extrapolate"]),
                    )
                    k = np.asarray(corr.K(z), float)  # or corr.k(z)
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
            f"${band}$ band (survey-native $g-r$ anchor)",
            fontsize=cfg["fs"] + 2,
        )
        ax.tick_params(labelsize=cfg["fs"] - 2)
        ax.legend(frameon=True, fontsize=cfg["fs"] - 3, loc="upper left")
        fig.tight_layout()

        out_base = f"kcorr__band-{_safe_tag(band)}__blue-red__surveys-native-gr.png"
        save_fig(fig, cfg["out_dir"] / out_base)
        plt.close(fig)

        n_total_plots += 1

    if n_total_plots == 0:
        raise RuntimeError("No optical curves plotted. Check response availability/mappings.")

    print("\nDone. Wrote to:", cfg["out_dir"])


if __name__ == "__main__":
    main()
