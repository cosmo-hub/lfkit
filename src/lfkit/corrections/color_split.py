"""Color-based red/blue SED anchors for kcorrect.

This module defines a *survey-independent* way to construct representative
galaxy SEDs in kcorrect template space using a rest-frame color split at z=0.

kcorrect does not provide named galaxy "types" (E/Sbc/Im); instead, it models
galaxy SEDs as linear combinations of a small set of basis templates. If you
want a stable red-vs-blue split across surveys (SDSS/DECam/HSC), you should:

1) Define a red/blue dividing line in rest-frame color space at z=0.
2) Choose anchor colors for "red" and "blue" around that line.
3) Fit kcorrect coefficient vectors for those anchor colors *once*.
4) Reuse the same coefficient vectors for all filter sets.

This ensures that differences in K(z) across surveys are driven by response
curves (instrument/filter differences), not by refitting drift.

Design
------
We provide two levels:

- A general linear red-sequence color–magnitude relation:
      (g-r)_cut(M_r) = a + b * M_r
  with tunable parameters (a, b).

- Convenience helpers that:
    * evaluate the cut at a representative magnitude M_r_ref, and
    * define two anchor colors (red and blue) as offsets from the cut, and
    * fit kcorrect coefficients for those anchors using only SDSS g and r.

Defaults
--------
The defaults are intentionally simple and tunable:
- a=0.12, b=-0.025 are reasonable low-z values (red sequence tilt).
- M_r_ref=-21.5 is a typical "bright" galaxy reference magnitude.
- red_offset=+0.10, blue_offset=0.20 give a clean separation.

You should treat these as configurable "priors" rather than universal truths.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import kcorrect.kcorrect as kk

from lfkit.utils.units import maggies_to_mag


SDSS_RESPONSES = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]


def gr_cut_linear(M_r: float, *, a: float = 0.12, b: float = -0.025) -> float:
    """Compute a linear red/blue dividing line in rest-frame (g-r).

    This models a red-sequence–like color–magnitude relation:

        (g-r)_cut(M_r) = a + b * M_r

    The parameters (a, b) define the intercept and slope of the
    color–magnitude relation and can be tuned to match a specific dataset.

    Args:
        M_r: Rest-frame absolute magnitude in the r band.
        a: Intercept of the linear relation.
        b: Slope of the linear relation.

    Returns:
        The dividing color (g-r) corresponding to the input M_r.
    """
    return float(a + b * float(M_r))


def default_gr_anchors(
    *,
    M_r_ref: float = -21.5,
    a: float = 0.12,
    b: float = -0.025,
    red_offset: float = 0.10,
    blue_offset: float = 0.20,
) -> Tuple[float, float, float]:
    """Return default (g-r) cut and anchor colors from a parameterized model.

    This evaluates the linear red/blue dividing line at a representative
    absolute magnitude and defines two anchor colors around it:

        cut  = (g-r)_cut(M_r_ref)
        red  = cut + red_offset
        blue = cut - blue_offset

    These anchors can then be used to define representative red and blue
    SEDs in kcorrect template space.

    Args:
        M_r_ref: Representative absolute magnitude at which to evaluate
            the color cut.
        a: Intercept of the linear (g-r) relation.
        b: Slope of the linear (g-r) relation.
        red_offset: Offset above the cut to define the red anchor color.
        blue_offset: Offset below the cut to define the blue anchor color.

    Returns:
        A tuple containing:
            - g_minus_r_cut: Dividing color at M_r_ref.
            - g_minus_r_red: Red anchor color.
            - g_minus_r_blue: Blue anchor color.
    """
    cut = gr_cut_linear(M_r_ref, a=a, b=b)
    g_red = cut + float(red_offset)
    g_blue = cut - float(blue_offset)
    return cut, g_red, g_blue


def evaluate_coeffs_maggies_z0(
    *,
    coeffs: np.ndarray,
    responses: list[str] | None = None,
) -> np.ndarray:
    """Evaluate model broadband maggies at z=0 for a coefficient vector.

    This function forward-models the broadband fluxes (in maggies) implied
    by a fitted kcorrect template coefficient vector at redshift z=0,
    using the specified response curves.

    It is primarily used for validation of color anchors and sanity checks
    of the reconstructed SED.

    Args:
        coeffs: One-dimensional kcorrect template coefficient vector.
        responses: Response list used to initialize kcorrect.
            Defaults to SDSS u/g/r/i/z if None.

    Returns:
        A NumPy array of model maggies with length equal to the number
        of response curves.

    Raises:
        AttributeError: If the installed kcorrect wrapper does not expose
            a compatible forward-model method.
        ValueError: If the input coefficient vector has incompatible shape.
    """
    responses = list(SDSS_RESPONSES if responses is None else responses)
    kc = kk.Kcorrect(responses=responses)

    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)  # MUST be 1D for scalar z

    maggies = kc.reconstruct(redshift=0.0, coeffs=coeffs)

    maggies = np.asarray(maggies, dtype=float)

    # Some wrappers return shape (1, nbands) even for scalar z; flatten if needed
    if maggies.ndim == 2 and maggies.shape[0] == 1:
        maggies = maggies[0]

    return maggies


def validate_color_anchor_gr(
    *,
    coeffs: np.ndarray,
    g_minus_r_target: float,
    responses: list[str] | None = None,
    tol_mag: float = 0.02,
) -> Dict[str, float]:
    """Validate that coeffs reproduce a target (g-r) color at z=0.

    This forward-models fluxes from the coefficient vector and checks the
    achieved (g-r) color against the requested target.

    Args:
        coeffs: Fitted kcorrect coefficient vector.
        g_minus_r_target: Target (g-r) at z=0.
        responses: Response list. Must include sdss_g0 and sdss_r0 if using SDSS.
            Defaults to SDSS u/g/r/i/z.
        tol_mag: Allowed absolute error in (g-r) in magnitudes.

    Returns:
        Dictionary with achieved color and error.

    Raises:
        ValueError: If the achieved color differs from target by > tol_mag.
    """
    responses = list(SDSS_RESPONSES if responses is None else responses)
    if "sdss_g0" not in responses or "sdss_r0" not in responses:
        raise ValueError("responses must include 'sdss_g0' and 'sdss_r0' to validate (g-r).")

    maggies = evaluate_coeffs_maggies_z0(coeffs=coeffs, responses=responses)
    mags = maggies_to_mag(maggies)

    i_g = responses.index("sdss_g0")
    i_r = responses.index("sdss_r0")

    g_minus_r = float(mags[i_g] - mags[i_r])
    err = float(g_minus_r - float(g_minus_r_target))

    out = {"g_minus_r_target": float(g_minus_r_target), "g_minus_r": g_minus_r, "err": err}

    if not np.isfinite(err) or abs(err) > float(tol_mag):
        raise ValueError(
            f"(g-r) anchor validation failed: target={g_minus_r_target:.4f}, "
            f"achieved={g_minus_r:.4f}, err={err:+.4f} mag (tol={tol_mag:.4f})."
        )
    return out

def fit_color_anchor_gr(
    *,
    g_minus_r: float,
    mag_r: float = 20.0,
    responses: list[str] | None = None,
    ivar_level: float = 1e10,
    validate: bool = True,
    tol_mag: float = 0.02,
) -> np.ndarray:
    """Fit a kcorrect coefficient vector matching a target (g-r) color at z=0.

    This constructs synthetic broadband fluxes corresponding to a specified
    rest-frame (g-r) color at z=0 and fits kcorrect template coefficients
    using `fit_coeffs`. Only SDSS g and r bands are constrained; all other
    bands are left unconstrained by setting their inverse-variance weights
    to zero.

    The overall flux normalization is arbitrary; only the relative color
    affects the fitted coefficients.

    Args:
        g_minus_r: Target rest-frame (g - r) color in magnitudes at z=0.
        mag_r: Reference r-band magnitude used for overall normalization.
        responses: Response list used to initialize kcorrect. Must include
            'sdss_g0' and 'sdss_r0'. Defaults to SDSS u/g/r/i/z.
        ivar_level: Inverse-variance weight assigned to the constrained
            g and r bands.

    Returns:
        A NumPy array of fitted kcorrect template coefficients.
    """
    responses = list(SDSS_RESPONSES if responses is None else responses)
    if "sdss_g0" not in responses or "sdss_r0" not in responses:
        raise ValueError("responses must include 'sdss_g0' and 'sdss_r0' for a g-r anchor.")

    kc = kk.Kcorrect(responses=responses)

    m_r = float(mag_r)
    m_g = m_r + float(g_minus_r)

    maggies = np.zeros(len(responses), dtype=float)
    ivar = np.zeros(len(responses), dtype=float)

    i_g = responses.index("sdss_g0")
    i_r = responses.index("sdss_r0")

    maggies[i_g] = 10.0 ** (-0.4 * m_g)
    maggies[i_r] = 10.0 ** (-0.4 * m_r)

    w = float(ivar_level)
    ivar[i_g] = w
    ivar[i_r] = w

    coeffs = kc.fit_coeffs(redshift=0.0, maggies=maggies, ivar=ivar)
    coeffs = np.asarray(coeffs, dtype=float)

    if validate:
        _ = validate_color_anchor_gr(
            coeffs=coeffs,
            g_minus_r_target=float(g_minus_r),
            responses=responses,
            tol_mag=float(tol_mag),
        )

    return coeffs


def fit_red_blue_anchors(
    *,
    M_r_ref: float = -21.5,
    a: float = 0.12,
    b: float = -0.025,
    red_offset: float = 0.10,
    blue_offset: float = 0.20,
    mag_r: float = 20.0,
    responses: list[str] | None = None,
    ivar_level: float = 1e10,
) -> Dict[str, Dict[str, float | np.ndarray]]:
    """Fit red and blue kcorrect coefficient anchors from a parameterized split.

    This function:

    1. Computes a dividing color using a linear (g-r) color–magnitude model.
    2. Defines red and blue anchor colors as offsets from the dividing color.
    3. Fits kcorrect coefficient vectors for both anchor colors at z=0.

    The resulting coefficient vectors can be reused across different survey
    filter sets to compute K-corrections while keeping the SED definition
    fixed.

    Args:
        M_r_ref: Representative absolute magnitude for evaluating the cut.
        a: Intercept of the linear (g-r) relation.
        b: Slope of the linear (g-r) relation.
        red_offset: Offset above the cut for the red anchor color.
        blue_offset: Offset below the cut for the blue anchor color.
        mag_r: Reference r-band magnitude used during fitting.
        responses: Response list used to initialize kcorrect.
            Defaults to SDSS u/g/r/i/z.
        ivar_level: Inverse-variance weight for the constrained g and r bands.

    Returns:
        A dictionary with keys 'cut', 'red', and 'blue'. Each entry contains:
            - 'g_minus_r': The corresponding anchor color.
        Additionally, 'red' and 'blue' include:
            - 'coeffs': The fitted kcorrect coefficient vector.
    """
    cut, g_red, g_blue = default_gr_anchors(
        M_r_ref=M_r_ref,
        a=a,
        b=b,
        red_offset=red_offset,
        blue_offset=blue_offset,
    )

    coeff_red = fit_color_anchor_gr(
        g_minus_r=g_red,
        mag_r=mag_r,
        responses=responses,
        ivar_level=ivar_level,
    )
    coeff_blue = fit_color_anchor_gr(
        g_minus_r=g_blue,
        mag_r=mag_r,
        responses=responses,
        ivar_level=ivar_level,
    )

    return {
        "cut": {"g_minus_r": cut},
        "red": {"g_minus_r": g_red, "coeffs": coeff_red},
        "blue": {"g_minus_r": g_blue, "coeffs": coeff_blue},
    }


def validate_sed_sanity(
    *,
    coeffs: np.ndarray,
    responses: list[str] | None = None,
    max_abs_color: float = 5.0,
) -> Dict[str, float]:
    """Perform basic sanity checks on reconstructed SED colors at z=0.

    This function forward-models broadband fluxes from a coefficient
    vector and computes adjacent SDSS colors (u-g, g-r, r-i, i-z).
    It verifies that the colors are finite and within a physically
    reasonable range.

    This is not a strict physical validation, but a safeguard against
    pathological coefficient solutions.

    Args:
        coeffs: One-dimensional kcorrect template coefficient vector.
        responses: Response list used to initialize kcorrect.
            Defaults to SDSS u/g/r/i/z if None.
        max_abs_color: Maximum allowed absolute magnitude of any
            adjacent color. Values exceeding this threshold are
            treated as unphysical.

    Returns:
        Dictionary containing the computed colors:
            - "u-g"
            - "g-r"
            - "r-i"
            - "i-z"

    Raises:
        ValueError: If any color is non-finite or exceeds the allowed
            maximum absolute color threshold.
    """
    responses = list(SDSS_RESPONSES if responses is None else responses)
    maggies = evaluate_coeffs_maggies_z0(coeffs=coeffs, responses=responses)
    mags = maggies_to_mag(maggies)

    # Quick color sanity checks
    i_u = responses.index("sdss_u0")
    i_g = responses.index("sdss_g0")
    i_r = responses.index("sdss_r0")
    i_i = responses.index("sdss_i0")
    i_z = responses.index("sdss_z0")

    ug = float(mags[i_u] - mags[i_g])
    gr = float(mags[i_g] - mags[i_r])
    ri = float(mags[i_r] - mags[i_i])
    iz = float(mags[i_i] - mags[i_z])

    for name, val in [("u-g", ug), ("g-r", gr), ("r-i", ri), ("i-z", iz)]:
        if (not np.isfinite(val)) or abs(val) > float(max_abs_color):
            raise ValueError(f"SED sanity check failed: {name}={val:.3f} (max_abs_color={max_abs_color}).")

    return {"u-g": ug, "g-r": gr, "r-i": ri, "i-z": iz}
