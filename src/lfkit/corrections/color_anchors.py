"""Color-anchor utilities for ``kcorrect``.

Core concept:
  An "anchor" is defined by a single two-band color constraint:

      color = (band_a - band_b) = color_value

where band_a and band_b are *kcorrect response names* (e.g. "sdss_g0").

This module is intentionally agnostic:
- no galaxy "types"
- no "red/blue" naming
- no survey assumptions beyond optional filter mapping wrappers

What this module does:
  The goal is simply to obtain a set of kcorrect template coefficients that
  reproduce a specified two-band color at a given redshift. Since a color only
  fixes a *flux ratio*, the overall normalization of the SED is arbitrary.
  To make the problem well defined we choose an arbitrary reference magnitude
  ("anchor") and construct a minimal synthetic photometry vector consistent
  with the requested color. Those fluxes are then passed to kcorrect to solve
  for the template mixture.

Important limitations:
  The resulting coefficients are **not a unique physical SED fit**. A single
  color constraint leaves large degeneracies in template space. The anchor
  normalization is purely a gauge choice, and no attempt is made to infer
  galaxy type, stellar population parameters, dust, or luminosity evolution.
  The coefficients are therefore best interpreted as a convenient internal
  representation for evaluating K(z) curves consistent with the specified
  color constraint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lfkit.utils.units import mag_to_maggies

from .kcorrect_backend import build_kcorrect

__all__ = [
    "fit_coeffs_from_bandcolor",
]

# Internal normalization (gauge) used to build a concrete photometry vector
# from a pure color (flux ratio) constraint.
_ANCHOR_MAG_DEFAULT = 22.0


def fit_coeffs_from_bandcolor(
    *,
    color: tuple[str, str],
    color_value: float,
    z_phot: float = 0.0,
    anchor_band: str | None = None,
    responses: list[str] | None = None,
    ivar_level: float = 1e10,
    response_dir: str | Path | None = None,
    redshift_range: tuple[float, float] = (0.0, 2.0),
    nredshift: int = 4000,
    rescale_maggies: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Fit kcorrect coefficients from a single two-band color constraint.

    This routine constructs the minimal synthetic photometry required to
    reproduce a given color. Because a color only fixes a flux ratio, the
    overall normalization is arbitrary; we therefore choose an internal
    anchor magnitude to define a concrete flux scale. The resulting fluxes
    are passed to kcorrect to solve for a template mixture whose predicted
    photometry matches the requested color at the specified redshift.

    Args:
        color: Tuple (band_a, band_b) meaning color = m_a - m_b.
               These must be kcorrect response names (file stems).
        color_value: Target color value in magnitudes.
        z_phot: Redshift at which to fit the coefficients.
        anchor_band: Band used to set the arbitrary flux normalization.
                     If None, defaults to band_b.
        responses: Optional explicit list of responses to use in the fit.
                   If None, uses the minimal set {band_a, band_b, anchor_band}.
        ivar_level: Inverse-variance weight for constrained bands.
        response_dir: Optional directory containing custom response .dat files.
        redshift_range: Internal kcorrect lookup redshift range.
        nredshift: Internal kcorrect lookup grid size.
        rescale_maggies: If True, rescale synthetic maggies to O(1) to reduce
                         numerical issues; adjusts ivar accordingly.

    Returns:
        (coeffs, fit_responses)
        - coeffs: array (n_templates,)
        - fit_responses: list of responses actually used in the fit
    """
    band_a, band_b = map(str, color)  # color = m_a - m_b

    if anchor_band is None:
        anchor_band = band_b

    # responses used for fitting
    if responses is None:
        fit_responses = list(dict.fromkeys([band_a, band_b, str(anchor_band)]))
    else:
        fit_responses = list(map(str, responses))
        missing = [x for x in (band_a, band_b, str(anchor_band)) if x not in fit_responses]
        if missing:
            raise ValueError(f"responses is missing required bands: {missing}")

    kc = build_kcorrect(
        responses_in=fit_responses,
        responses_out=fit_responses,
        responses_map=fit_responses,
        response_dir=response_dir,
        redshift_range=redshift_range,
        nredshift=nredshift,
    )

    # anchor magnitude -> anchor flux (maggies)
    f_anchor = float(mag_to_maggies(_ANCHOR_MAG_DEFAULT))
    if (not np.isfinite(f_anchor)) or f_anchor <= 0:
        raise ValueError("anchor_mag must map to positive finite maggies.")

    # color = m_a - m_b => f_a / f_b = 10^(-0.4*color)
    ratio = 10.0 ** (-0.4 * float(color_value))
    if (not np.isfinite(ratio)) or ratio <= 0:
        raise ValueError("color_value must be finite.")

    # choose fluxes consistent with color and anchor band choice
    if anchor_band == band_a:
        f_a = f_anchor
        f_b = f_a / ratio
    elif anchor_band == band_b:
        f_b = f_anchor
        f_a = f_b * ratio
    else:
        # If anchor_band is distinct, anchor it and set (a,b) relative in a simple way.
        f_b = f_anchor
        f_a = f_b * ratio

    flux_map = {band_a: f_a, band_b: f_b, str(anchor_band): f_anchor}

    maggies = np.full(len(fit_responses), np.nan, dtype=float)
    ivar = np.zeros(len(fit_responses), dtype=float)

    w = float(ivar_level)
    for i, band in enumerate(fit_responses):
        if band in flux_map:
            maggies[i] = float(flux_map[band])
            ivar[i] = w

    if rescale_maggies:
        pos = maggies[np.isfinite(maggies) & (maggies > 0)]
        scale = float(np.nanmedian(pos)) if pos.size else 1.0
        if np.isfinite(scale) and scale > 0:
            maggies = maggies / scale
            ivar = ivar * (scale**2)

    coeffs = kc.fit_coeffs(redshift=float(z_phot), maggies=maggies, ivar=ivar)
    coeffs = np.asarray(coeffs, float)

    nt = int(kc.templates.restframe_flux.shape[0])
    if coeffs.shape != (nt,):
        raise ValueError(f"fit_coeffs returned shape {coeffs.shape}, expected ({nt},)")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("fit_coeffs returned non-finite coeffs.")
    if np.any(coeffs < 0):
        raise ValueError("fit_coeffs returned negative coeffs.")
    if float(np.sum(coeffs)) <= 0:
        raise ValueError("fit_coeffs returned coeffs with sum<=0.")

    return coeffs, fit_responses
