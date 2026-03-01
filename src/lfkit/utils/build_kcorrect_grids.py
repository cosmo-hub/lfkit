from __future__ import annotations

from pathlib import Path
import numpy as np

from lfkit.utils.io import save_kcorr_package
from lfkit.corrections.kcorrect import (
    build_kcorr_grid_package,
    make_type_coeffs_from_colors,
)


def main() -> None:
    """Runs the script."""
    out_dir = Path("src/lfkit/data/kcorrect/grids")
    out_dir.mkdir(parents=True, exist_ok=True)

    RESPONSE_SETS: dict[str, list[str]] = {
        "sdss": [
            "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
        ],
        "decam": [
            "decam_u", "decam_g", "decam_r", "decam_i", "decam_z", "decam_Y",
        ],
        "bessell": [
            "bessell_U", "bessell_B", "bessell_V", "bessell_R", "bessell_I",
        ],
        # Closest shipped "HSC" analogue in your kcorrect list:
        "subaru_suprimecam": [
            "subaru_suprimecam_B",
            "subaru_suprimecam_V",
            "subaru_suprimecam_Rc",
            "subaru_suprimecam_Ic",
            "subaru_suprimecam_g",
            "subaru_suprimecam_r",
            "subaru_suprimecam_i",
            "subaru_suprimecam_z",
        ],
    }

    z0, z1 = 0.0, 4.0
    Nz = 801
    z_grid = np.linspace(z0, z1, Nz)

    band_shift = None

    for tag, responses in RESPONSE_SETS.items():

        # pick a sensible reference band for this system
        if tag == "sdss":
            ref_band = "sdss_r0"
        elif tag == "decam":
            ref_band = "decam_r"
        elif tag == "bessell":
            ref_band = "bessell_V"
        elif tag == "subaru_suprimecam":
            ref_band = "subaru_suprimecam_r"
        else:
            ref_band = responses[len(responses) // 2]

        coeffs_by_type = make_type_coeffs_from_colors(
            responses=responses,
            ref_mag=20.0,
            ref_band=ref_band,
            ref_z=0.1,
        )

        pkg = build_kcorr_grid_package(
            responses_in=responses,
            responses_out=responses,
            responses_map=responses,
            coeffs_by_type=coeffs_by_type,
            z_grid=z_grid,
            band_shift=band_shift,
            redshift_range=(float(z_grid[0]), float(z_grid[-1])),
            nredshift=8000,
        )

        for t in pkg["types"]:
            K0 = np.asarray(pkg["K"][t])[0]  # first row corresponds to z=0
            print(tag, t, "max|K(z=0)| =", np.nanmax(np.abs(K0)))

        zmin = float(z_grid[0])
        zmax = float(z_grid[-1])

        fname = (
            f"kcorrect__{tag}__z{zmin:.4f}_{zmax:.1f}"
            f"__Nz{Nz}__bs{band_shift if band_shift is not None else 'none'}.npz"
        )
        save_kcorr_package(pkg, out_dir / fname)
        print("Saved:", out_dir / fname)


if __name__ == "__main__":
    main()
