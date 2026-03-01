"""Utilities for registering custom survey response curves with kcorrect.

This allows LFKit to compute k-corrections for arbitrary surveys
(KiDS, Euclid, HSC custom sets, etc.) as long as a bandpass curve
is provided.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def write_kcorrect_response(
    *,
    name: str,
    wave_angst: np.ndarray,
    throughput: np.ndarray,
    out_dir: str | Path,
    normalize: bool = True,
) -> str:
    """
    Write a kcorrect-compatible response file.

    Parameters
    ----------
    name : str
        Name of the response (e.g. "kids_r").
    wave_angst : array
        Wavelength array in Ångstrom.
    throughput : array
        Dimensionless transmission curve (0–1).
    out_dir : str or Path
        Directory to write the response file into.
    normalize : bool
        If True, normalize throughput to max=1.

    Returns
    -------
    str
        The response name (same as input name).
    """

    wave_A = np.asarray(wave_angst, float)
    throughput = np.asarray(throughput, float)

    if wave_A.ndim != 1 or throughput.ndim != 1:
        raise ValueError("wave_A and throughput must be 1D arrays.")
    if wave_A.size != throughput.size:
        raise ValueError("wave_A and throughput must have same length.")
    if wave_A.size < 10:
        raise ValueError("Response curve must contain >=10 points.")

    order = np.argsort(wave_A)
    wave_A = wave_A[order]
    throughput = throughput[order]

    throughput = np.clip(throughput, 0.0, None)

    if normalize and throughput.max() > 0:
        throughput = throughput / throughput.max()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{name}.dat"

    header = (
        "# kcorrect response file\n"
        "# wavelength [Angstrom]   throughput\n"
    )

    np.savetxt(
        out_path,
        np.column_stack([wave_A, throughput]),
        header=header,
    )

    return name