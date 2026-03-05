import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pyccl as ccl
import cmasher as cmr

from lfkit.api.corrections import Corrections
from lfkit.utils.io import (
    resolve_packaged_csv,
    load_vizier_csv,
    available_pairs,
)

output_dir = Path("output") / "plots" / "poggianti1997"
output_dir.mkdir(parents=True, exist_ok=True)

colors = cmr.take_cmap_colors(
    "cmr.infinity_s",
    3,
    cmap_range=(0.5, 1),
    return_fmt="hex",
)

z = np.linspace(0.0, 3.5, 400)

# Cosmology used only for mapped E(z)
cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    h=0.67,
    sigma8=0.8,
    n_s=0.96,
)

# Discover available band/SED pairs from packaged CSV
kcorr_csv = resolve_packaged_csv("kcorr.csv")
ktab = load_vizier_csv(kcorr_csv)
pairs = available_pairs(ktab)

method = "pchip"
extrapolate = True

for band, gal_types in pairs.items():
    for gal_type in gal_types:
        fig, ax = plt.subplots(figsize=(7, 5))

        corr_original = Corrections.poggianti(
            band=band,
            gal_type=gal_type,
            cosmo=cosmo,
            original_z_for_e=True,
            method=method,
            extrapolate=extrapolate,
            e_model="poggianti",  # explicit (default anyway)
        )
        corr_mapped = Corrections.poggianti(
            band=band,
            gal_type=gal_type,
            cosmo=cosmo,
            original_z_for_e=False,
            method=method,
            extrapolate=extrapolate,
            e_model="poggianti",
        )

        K = corr_original.k(z)
        E_original = corr_original.e(z)
        E_mapped = corr_mapped.e(z)

        KE_original = K + E_original
        KE_mapped = K + E_mapped
        # or: KE_original = corr_original.ke(z); KE_mapped = corr_mapped.ke(z)

        # Plot
        lw = 3
        fs = 15
        ax.plot(z, K, label=r"$K(z)$", lw=lw, color=colors[1])
        ax.plot(z, E_original, label=r"$E(z)$ original", lw=lw, ls="--", color=colors[2])
        ax.plot(z, E_mapped, label=r"$E(z)$ mapped", lw=lw, ls="-", color=colors[2])
        ax.plot(z, KE_original, label=r"$K+E$ original", lw=lw, color=colors[0], ls="--")
        ax.plot(z, KE_mapped, label=r"$K+E$ mapped", lw=lw, color=colors[0], ls="-")

        ax.set_xlabel(r"Redshift $z$", fontsize=fs)
        ax.set_ylabel("Correction [mag]", fontsize=fs)
        ax.set_title(f"Poggianti 1997 — band: {band}, type: {gal_type}", fontsize=fs + 2)
        ax.legend(frameon=True, fontsize=fs - 2)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"ke_pogg1997_band_{band}_type_{gal_type}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
