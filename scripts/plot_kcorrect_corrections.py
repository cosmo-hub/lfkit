import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cmasher as cmr

from lfkit.api.corrections import Corrections
from lfkit.utils.io import load_kcorr_package


def plot_one_package(
    pkg_path: Path,
    *,
    output_root: Path,
    fs: int = 15,
    lw: int = 3,
    method: str = "pchip",
    extrapolate: bool = True,
    z_eval_max: float = 3.5,
    nz_eval: int = 700,
) -> None:
    pkg = load_kcorr_package(pkg_path)

    z = np.linspace(0.0, float(z_eval_max), int(nz_eval))

    types_to_plot = list(pkg["types"])
    bands_to_plot = list(pkg["responses_out"])

    out_dir = output_root / pkg_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # use thse colors
    cmap_blue = "PiYG_r"
    cmap_red = "RdBu"
    cmap_purple = "BuPu"
    reds = cmr.take_cmap_colors(
        cmap_blue, 1, cmap_range=(0.9, 0.99), return_fmt="hex"
    )
    blues = cmr.take_cmap_colors(
        cmap_red, 1, cmap_range=(0.9, 0.99), return_fmt="hex"
    )

    purples = cmr.take_cmap_colors(
        cmap_purple, 1, cmap_range=(0.9, 0.99), return_fmt="hex"
    )

    type_color = {
        "E": reds[0],
        "Sbc": blues[0],
        "Im": purples[0],
    }

    for out_band in bands_to_plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        n_plotted = 0

        for gal_type in types_to_plot:
            corr = Corrections.kcorrect_from_pkg(
                pkg=pkg,
                gal_type=gal_type,
                out_band=out_band,
                method=method,
                extrapolate=extrapolate,
                e_model="none",
            )

            try:
                y = corr.K(z)
            except Exception:
                continue

            c = type_color.get(gal_type, "#666666")
            ax.plot(z, y, color=c, lw=lw, label=f"${gal_type}$")
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            print(
                f"[{pkg_path.name}] Skipping band {out_band}: no finite curves.")
            continue

        ax.set_xlabel("Redshift $z$", fontsize=fs)
        ax.set_ylabel("$k$-correction [mag]", fontsize=fs)

        raw = out_band
        raw_lower = raw.lower()

        if raw_lower.startswith("subaru_"):
            survey = "HSC"
            band = raw.split("_")[-1]
        elif raw_lower.startswith("decam_"):
            survey = "DECam"
            band = raw.split("_")[-1]
        else:
            survey = raw.split("_")[0].upper()
            band = raw.split("_")[-1]

        band = band.replace("0", "")
        ax.set_title(f"{survey} ${band}$", fontsize=fs + 2)
        ax.legend(frameon=True, fontsize=fs - 2)

        plt.tight_layout()
        plt.savefig(out_dir / f"K__band_{out_band}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Done: {pkg_path}")
    print("  Stored z-range:", float(np.min(pkg["z"])), "→", float(np.max(pkg["z"])))


def main() -> None:
    output_root = Path("output") / "plots" / "kcorrect"
    output_root.mkdir(parents=True, exist_ok=True)

    grid_dir = Path("src/lfkit/data/kcorrect/grids")
    pkg_paths = sorted(grid_dir.glob("kcorrect__*.npz"))
    if not pkg_paths:
        raise FileNotFoundError(f"No kcorrect grids found in {grid_dir}")

    fs = 15
    lw = 3
    method = "pchip"
    extrapolate = True

    for pkg_path in pkg_paths:
        plot_one_package(
            pkg_path,
            output_root=output_root,
            fs=fs,
            lw=lw,
            method=method,
            extrapolate=extrapolate,
            z_eval_max=3.5,
            nz_eval=700,
        )

    print("All done. Wrote to:", output_root)


if __name__ == "__main__":
    main()