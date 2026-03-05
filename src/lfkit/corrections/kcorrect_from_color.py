"""kcorrect k(z) evaluation from a single color anchor.

This module provides a minimal path from a **two-band color constraint**
to a k(z) curve. The idea is straightforward: a rest-frame color fixes a
flux ratio between two bands, which constrains the mixture of kcorrect
SED templates. Once the corresponding template coefficients are obtained,
kcorrect can be evaluated across redshift to produce the resulting k(z).

Only a single color constraint is used. No galaxy types or population
labels are introduced, and no attempt is made to infer a physical galaxy
classification. The color simply defines a template mixture that is
consistent with the specified flux ratio, which is then used to evaluate
k(z) for the requested output response band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .color_anchors import fit_coeffs_from_bandcolor
from .kcorrect_backend import build_kcorrect

__all__ = [
    "kcorrect_from_bandcolor",
]

# Internal normalization used to construct a concrete photometry vector
# for the fitter. This is a gauge choice in one-color mode.
_ANCHOR_MAG_DEFAULT = 22.0


def kcorrect_from_bandcolor(
    *,
    z: np.ndarray,
    response_out: str,
    color: tuple[str, str],
    color_value: float,
    z_phot: float = 0.0,
    anchor_band: str | None = None,
    band_shift: float | None = None,
    response_dir: str | Path | None = None,
    redshift_range: tuple[float, float] = (0.0, 2.0),
    nredshift: int = 4000,
    ivar_level: float = 1e10,
    anchor_z0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k(z) for a single output response from a two-band color.

    This routine converts a color constraint into kcorrect template
    coefficients and evaluates the resulting k(z) curve on the supplied
    redshift grid. The color defines the relative flux between two bands,
    which constrains the template mixture used by kcorrect.

    Because a color only fixes a flux ratio, the overall normalization of
    the spectrum is arbitrary. An internal anchor magnitude is therefore
    used to define a concrete photometry vector for the fit. The resulting
    coefficients should be interpreted as a convenient representation of
    an SED consistent with the specified color rather than a unique
    physical galaxy model.
    """
    z = np.asarray(z, float)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)):
        raise ValueError("z must be a finite 1D array with >=2 points.")

    coeffs, _fit_responses = fit_coeffs_from_bandcolor(
        color=color,
        color_value=float(color_value),
        z_phot=float(z_phot),
        anchor_band=anchor_band,
        responses=None,
        ivar_level=float(ivar_level),
        response_dir=response_dir,
        redshift_range=redshift_range,
        nredshift=int(nredshift),
        rescale_maggies=True,
    )

    kc = build_kcorrect(
        responses_in=[str(response_out)],
        responses_out=[str(response_out)],
        responses_map=[str(response_out)],
        response_dir=response_dir,
        redshift_range=redshift_range,
        nredshift=nredshift,
    )

    K = np.full_like(z, np.nan, dtype=float)
    for i, zi in enumerate(z):
        if band_shift is None:
            kval = kc.kcorrect(redshift=float(zi), coeffs=coeffs)
        else:
            kval = kc.kcorrect(redshift=float(zi), coeffs=coeffs, band_shift=float(band_shift))
        K[i] = float(np.asarray(kval, float)[0])

    ok = np.isfinite(K)
    if np.count_nonzero(ok) < 2:
        raise ValueError(f"Too few finite K(z) points for response_out={response_out!r}.")

    if anchor_z0:
        i0 = int(np.where(ok)[0][0])
        K[ok] = K[ok] - K[i0]

    return z[ok], K[ok]
