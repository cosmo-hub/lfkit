"""
Unified k- and e-correction interface.

This module is the user-facing entry point for LFKit photometric corrections.
It provides a small, stable API for evaluating:

    k(z)   bandpass (k-) correction
    e(z)   luminosity evolution (e-) correction
    ke(z)  combined correction, ke(z) = k(z) + e(z)

Design goals
------------
- Keep the runtime API minimal and stable: k(z), e(z), ke(z)
- Make backend choices explicit and composable:
    - k(z) backend: "poggianti" or "kcorrect"
    - e(z) backend: "none" or "poggianti"
- Use astronomy-friendly inputs at the public boundary:
    - filterset: "sdss", "hsc", "decam", "bessell", ...
    - band: "r", "i", "V", ...
    - Poggianti uses a typed table key (gal_type)
    - kcorrect uses an SED mixture specified by a color anchor

Reality check about kcorrect
----------------------------
kcorrect returns k(z) for a chosen SED mixture and filter response. It does not
encode physical luminosity evolution by itself. If you want e(z), you supply a
separate evolution model (for example from Poggianti tabulations), and LFKit
combines them consistently.

Sign convention
---------------
Absolute magnitude is defined as:

    M = m - DM(z) - k(z) + e(z)

With this convention, if galaxies were brighter in the past, then e(z) is
typically negative for z > z_piv in pivoted evolution models.

Notes on filter names and kcorrect responses
--------------------------------------------
In LFKit the public choice of a band is expressed as (filterset, band). kcorrect
internally uses response curve identifiers (file stems). LFKit maps:

    (filterset, band) -> response_name

You can override or extend this mapping via ``response_map`` when working with
custom surveys or custom filter curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from lfkit.utils.interpolation import build_1d_interpolator, as_1d_finite_grid
import lfkit.corrections.poggianti1997 as pogg
from lfkit.corrections.filters import DEFAULT_RESPONSE_MAP

ArrayLike = object  # anything np.asarray can handle

# Default mapping: (filterset, band) -> kcorrect response curve name.
DEFAULT_KCORRECT_RESPONSE_MAP = DEFAULT_RESPONSE_MAP

class Corrections:
    """Evaluator for k(z), e(z), and ke(z).

    This object wraps callables representing the k-correction and the
    luminosity evolution correction and provides a consistent interface
    for evaluating:

        k(z)
        e(z)
        ke(z) = k(z) + e(z)

    Instances are typically created through the class constructors
    ``Corrections.poggianti`` or ``Corrections.kcorrect``.

    Args:
        k_func: Callable returning k(z).
        e_func: Optional callable returning e(z). If None, e(z)=0.
        meta: Optional metadata dictionary describing the backend
            configuration.
    """

    def __init__(
        self,
        k_func: Callable[[ArrayLike], ArrayLike],
        e_func: Callable[[ArrayLike], ArrayLike] | None = None,
        *,
        meta: dict[str, object] | None = None,
    ) -> None:
        self._k = k_func
        self._e = e_func
        self.meta: dict[str, object] = {} if meta is None else dict(meta)

    def k(self, z: ArrayLike) -> np.ndarray:
        """Evaluate the k-correction.

        Args:
            z: Scalar or array-like redshift.

        Returns:
            NumPy array containing k(z).
        """
        return np.asarray(self._k(z), float)

    def e(self, z: ArrayLike) -> np.ndarray:
        """Evaluate the luminosity evolution correction.

        Args:
            z: Scalar or array-like redshift.

        Returns:
            NumPy array containing e(z). If no evolution model is attached,
            zeros are returned.
        """
        if self._e is None:
            return np.zeros_like(np.asarray(z, float))
        return np.asarray(self._e(z), float)

    def ke(self, z: ArrayLike) -> np.ndarray:
        """Evaluate the combined correction ke(z).

        Args:
            z: Scalar or array-like redshift.

        Returns:
            NumPy array containing ke(z) = k(z) + e(z).
        """
        z_arr = np.asarray(z, float)
        return self.k(z_arr) + self.e(z_arr)

    @classmethod
    def poggianti(
        cls,
        *,
        band: str,
        gal_type: str,
        cosmo=None,
        original_z_for_e: bool = True,
        method: str = "pchip",
        extrapolate: bool = True,
        e_model: str = "poggianti",
    ) -> "Corrections":
        """Construct corrections from Poggianti (1997) tabulated models.

        Args:
            band: Poggianti band identifier (e.g. "V", "B").
            gal_type: Galaxy spectral type in the Poggianti tables
                (e.g. "E", "Sc").
            cosmo: Optional cosmology object used for lookback-time
                calculations in the evolution correction.
            original_z_for_e: Whether the evolution correction is evaluated
                on the original Poggianti redshift grid.
            method: Interpolation method for the tabulated curves.
            extrapolate: Whether to allow extrapolation outside the
                tabulated redshift range.
            e_model: Evolution backend ("poggianti" or "none").

        Returns:
            Corrections instance capable of evaluating k(z), e(z), and ke(z).

        Raises:
            ValueError: If an unsupported evolution model is requested.
        """
        z_k, kcorr, z_e, ecorr = pogg.load_poggianti1997_tables(band=band, sed=gal_type)

        k_func = pogg.make_kcorr_interpolator(
            z_k, kcorr, method=method, extrapolate=extrapolate
        )

        e_func: Callable[[ArrayLike], np.ndarray] | None
        if str(e_model).lower() == "none":
            e_func = None
        elif str(e_model).lower() == "poggianti":
            e_func = pogg.make_ecorr_interpolator(
                z_e,
                ecorr,
                original_z=original_z_for_e,
                cosmo=cosmo,
                method=method,
                extrapolate=extrapolate,
            )
        else:
            raise ValueError("e_model must be 'none' or 'poggianti' "
                             "for Corrections.poggianti().")

        out = cls(k_func=k_func, e_func=e_func)
        out.meta.update(
            {
                "k_backend": "poggianti1997",
                "e_backend": str(e_model).lower(),
                "band": str(band),
                "gal_type": str(gal_type),
            }
        )
        return out

    @classmethod
    def kcorrect(
        cls,
        *,
        z_grid: ArrayLike | None = None,
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
        method: str = "pchip",
        extrapolate: bool = True,
    ) -> "Corrections":
        """Construct k(z) using the kcorrect SED template model.

        The spectral energy distribution (SED) is constrained by a two-band
        color. The color fixes a flux ratio between two bands, which
        determines a mixture of kcorrect templates consistent with
        that constraint. The resulting template mixture is then used
        to evaluate k(z) for the requested output response.

        Args:
            z_grid: Redshift grid used to compute the k(z) curve. If None,
                a default grid spanning ``redshift_range`` is used.
            response_out: kcorrect response name for which k(z) is evaluated.
            color: Two-band color constraint (band_a, band_b).
            color_value: Target color value in magnitudes.
            z_phot: Redshift at which the color constraint is applied.
            anchor_band: Band used to set the arbitrary flux normalization.
            band_shift: Optional kcorrect band shift parameter.
            response_dir: Directory containing custom response curves.
            redshift_range: Internal redshift range used by kcorrect.
            nredshift: Internal redshift grid size used by kcorrect.
            ivar_level: Weight assigned to constrained photometric bands.
            anchor_z0: Whether to normalize so that k(z=0)=0.
            method: Interpolation method used to construct k(z).
            extrapolate: Whether to allow extrapolation outside the
                computed redshift range.

        Returns:
            Corrections instance evaluating k(z). In this configuration
            e(z) = 0.
        """
        from lfkit.corrections.kcorrect_from_color import \
            kcorrect_from_bandcolor

        if z_grid is None:
            z_arr = np.linspace(redshift_range[0], redshift_range[1], 4001)
        else:
            z_arr = as_1d_finite_grid(z_grid, name="z_grid")

        z_ok, k_ok = kcorrect_from_bandcolor(
            z=np.asarray(z_arr, float),
            response_out=str(response_out),
            color=(str(color[0]), str(color[1])),
            color_value=float(color_value),
            z_phot=float(z_phot),
            anchor_band=None if anchor_band is None else str(anchor_band),
            band_shift=band_shift,
            response_dir=response_dir,
            redshift_range=(float(redshift_range[0]),
                            float(redshift_range[1])),
            nredshift=int(nredshift),
            ivar_level=float(ivar_level),
            anchor_z0=bool(anchor_z0),
        )

        k_func = build_1d_interpolator(
            np.asarray(z_ok, float),
            np.asarray(k_ok, float),
            method=str(method),
            extrapolate=bool(extrapolate),
            extrap_mode="linear_tail",
        )

        out = cls(k_func=k_func, e_func=None)
        out.meta.update(
            {
                "k_backend": "kcorrect_bandcolor",
                "response_out": str(response_out),
                "color": (str(color[0]), str(color[1])),
                "color_value": float(color_value),
                "z_phot": float(z_phot),
                "anchor_band": None if anchor_band is None else str(
                    anchor_band),
                "band_shift": band_shift,
                "response_dir": str(
                    response_dir) if response_dir is not None else None,
                "redshift_range": (float(redshift_range[0]),
                                   float(redshift_range[1])),
                "nredshift": int(nredshift),
                "ivar_level": float(ivar_level),
                "anchor_z0": bool(anchor_z0),
                "method": str(method),
                "extrapolate": bool(extrapolate),
                "z_valid_min": float(np.asarray(z_ok, float)[0]),
                "z_valid_max": float(np.asarray(z_ok, float)[-1]),
                "n_finite": int(np.asarray(z_ok).size),
            }
        )
        return out
