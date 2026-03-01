"""Unified k- and e-correction interface.

This module provides a high-level API for evaluating photometric
k-corrections and luminosity evolution e-corrections.

The `Corrections` class wraps callable backends for k(z) and e(z)
and exposes a consistent interface:

    k(z)
    e(z)
    ke(z) = k(z) + e(z)

Supported backends:

K-corrections:
    - Poggianti (1997) tabulated models
    - kcorrect (on-the-fly evaluation from SED/coeffs)
    - kcorrect precomputed grids ("pkg" consumer)

E-corrections:
    - None (e = 0)
    - Poggianti (1997) tabulated models
    - Parametric analytic models

All corrections are returned in magnitudes.

Sign convention:
    Absolute magnitude:
        M = m - DM(z) - k(z) + e(z)

    Under this convention, galaxies were brighter in the past,
    so e(z) is typically negative at z > z_piv (for pivoted evolution models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import nnls

from lfkit.corrections.color_split import fit_red_blue_anchors
import lfkit.corrections.poggianti1997 as pogg
from lfkit.corrections.kcorrect import build_kcorrect, list_available_responses
from lfkit.utils.interpolation import build_1d_interpolator
from lfkit.corrections.responses import write_kcorrect_response

ArrayLike = float | np.ndarray


class Corrections:
    """Unified evaluator for k and e corrections.

    This class wraps callable k(z) and e(z) backends and provides
    a consistent public API for evaluating corrections.

    Args:
        k_func: Callable returning k(z) in magnitudes.
        e_func: Callable returning e(z) in magnitudes,
            or None to disable evolution corrections.

    Notes:
        - All functions must accept scalar or array-like redshift inputs.
        - Outputs are always returned as NumPy arrays of dtype float.
    """

    def __init__(
        self,
        k_func: Callable[[ArrayLike], np.ndarray],
        e_func: Callable[[ArrayLike], np.ndarray] | None,
    ) -> None:
        self._k = k_func
        self._e = e_func

        # Optional metadata for debugging/diagnostics
        self.meta: dict[str, object] = {}

    def K(self, z: ArrayLike) -> np.ndarray:
        return np.asarray(self._k(z), float)

    def E(self, z: ArrayLike) -> np.ndarray:
        if self._e is None:
            return np.zeros_like(np.asarray(z, float))
        return np.asarray(self._e(z), float)

    def KE(self, z: ArrayLike) -> np.ndarray:
        z_arr = np.asarray(z, float)
        return self.K(z_arr) + self.E(z_arr)

    @classmethod
    def _build_e_backend(
        cls,
        *,
        e_model: str,
        e_kwargs: dict | None,
    ) -> Callable[[ArrayLike], np.ndarray] | None:
        """Construct an e-correction backend.

        Args:
            e_model: Evolution backend name: "none" or "poggianti".
            e_kwargs: Keyword arguments passed to the Poggianti backend.
                Required iff e_model="poggianti".

        Returns:
            Callable e_func(z) -> e(z) in magnitudes, or None if disabled.
        """
        e_model = str(e_model).lower()

        if e_model == "none":
            return None

        if e_model != "poggianti":
            raise ValueError("e_model must be 'none' or 'poggianti'")

        if e_kwargs is None:
            raise ValueError("e_kwargs required when e_model='poggianti'")

        tmp = cls.poggianti1997(**e_kwargs)
        return tmp._e

    @classmethod
    def poggianti1997(
        cls,
        *,
        band: str,
        sed: str,
        cosmo=None,
        original_z_for_e: bool = True,
        method: str = "pchip",
        extrapolate: bool = True,
    ) -> "Corrections":
        """Construct corrections using Poggianti (1997) tables.

        Args:
            band: Output photometric band name.
            sed: Spectral energy distribution identifier.
            cosmo: Cosmology object used for lookback-time
                conversions if required.
            original_z_for_e: Whether to evaluate e-corrections
                on the original redshift grid.
            method: Interpolation method.
            extrapolate: Whether to allow extrapolation beyond
                tabulated redshift range.

        Returns:
            Corrections instance using tabulated k and e values.
        """
        z_k, kcorr, z_e, ecorr = pogg.load_poggianti1997_tables(band=band, sed=sed)

        k_interp = pogg.make_kcorr_interpolator(
            z_k,
            kcorr,
            method=method,
            extrapolate=extrapolate,
        )

        e_interp = pogg.make_ecorr_interpolator(
            z_e,
            ecorr,
            original_z=original_z_for_e,
            cosmo=cosmo,
            method=method,
            extrapolate=extrapolate,
        )

        out = cls(k_func=k_interp, e_func=e_interp)
        out.meta.update({"backend": "poggianti1997", "band": band, "sed": sed})
        return out

    @classmethod
    def kcorrect(
        cls,
        *,
        z: ArrayLike,
        response: str,
        coeffs: np.ndarray,
        band_shift: float | None = None,
        method: str = "pchip",
        extrapolate: bool = True,
        anchor_z0: bool = True,
        e_model: str = "none",
        e_kwargs: dict | None = None,
        response_dir: str | None = None,
    ) -> "Corrections":
        """Construct corrections by evaluating kcorrect on a z grid.

        This is the "compute" API: it evaluates `kcorrect.kcorrect.Kcorrect.kcorrect`
        on the provided redshift grid, then builds an interpolator and (optionally)
        extrapolates beyond the valid kcorrect range via `linear_tail`.

        Args:
            z: Redshift grid to evaluate on (1D). This grid defines the
                returned interpolator domain (and extrapolation behavior).
            response: kcorrect response name to compute k correction for,
                e.g. "sdss_r0" or "bessell_V".
            coeffs: kcorrect template coefficient vector for the galaxy type.
                Shape must match the number of kcorrect templates.
            band_shift: Optional band-shift parameter passed to kcorrect.
            method: Interpolation method ("pchip", "akima", "linear").
            extrapolate: Whether to allow extrapolation beyond the valid
                evaluation range. Uses `extrap_mode="linear_tail"`.
            anchor_z0: If True, subtract the first finite K value so that
                k(z0)=0 at the first valid grid point (typically z=0).
            e_model: Evolution model selection:
                - "none"
                - "poggianti"
                - "parametric"
            e_kwargs: Keyword arguments passed to the selected e backend.
            response_dir: Optional directory for kcorrect response files.

        Returns:
            Corrections instance combining computed kcorrect k values
            with the selected e backend.

        Raises:
            ValueError: If z is invalid or if too few finite kcorrect points
                are available to build an interpolator.
        """

        z_arr = np.asarray(z, float)
        if z_arr.ndim != 1 or z_arr.size < 2 or np.any(~np.isfinite(z_arr)):
            raise ValueError("z must be a finite 1D array with >=2 points.")

        kc = build_kcorrect(
            responses_in=[str(response)],
            responses_out=[str(response)],
            responses_map=[str(response)],
            response_dir=response_dir,
        )

        # Evaluate kcorrect where supported, stop at first ValueError.
        k_raw = np.full_like(z_arr, np.nan, dtype=float)
        zmax_ok = float("nan")

        for i, zi in enumerate(z_arr):
            try:
                if band_shift is None:
                    kval = kc.kcorrect(redshift=float(zi), coeffs=np.asarray(coeffs, float))
                else:
                    kval = kc.kcorrect(
                        redshift=float(zi),
                        coeffs=np.asarray(coeffs, float),
                        band_shift=float(band_shift),
                    )
                k_raw[i] = float(np.asarray(kval)[0])
                zmax_ok = float(zi)
            except ValueError:
                break

        ok = np.isfinite(k_raw)
        if np.count_nonzero(ok) < 2:
            raise ValueError(
                f"Too few finite points to interpolate K(z): response={response!r}, "
                f"n_finite={int(np.count_nonzero(ok))} out of {z_arr.size}."
            )

        if anchor_z0:
            i0 = int(np.where(ok)[0][0])
            k_raw[ok] = k_raw[ok] - k_raw[i0]

        k_func = build_1d_interpolator(
            z_arr[ok],
            k_raw[ok],
            method=method,
            extrapolate=extrapolate,
            extrap_mode="linear_tail",
        )

        e_func = cls._build_e_backend(e_model=str(e_model), e_kwargs=e_kwargs)

        out = cls(k_func=k_func, e_func=e_func)
        out.meta.update(
            {
                "backend": "kcorrect_compute",
                "response": str(response),
                "band_shift": band_shift,
                "kcorrect_z_valid_max": zmax_ok,
                "n_finite": int(np.count_nonzero(ok)),
            }
        )
        return out

    @classmethod
    def kcorrect_from_sed(
        cls,
        *,
        z: ArrayLike,
        response: str,
        sed_wave_A: np.ndarray,
        sed_flux: np.ndarray,
        band_shift: float | None = None,
        weighted_fit: bool = False,
        z_ref_weight: float = 0.3,
        method: str = "pchip",
        extrapolate: bool = True,
        anchor_z0: bool = True,
        e_model: str = "none",
        e_kwargs: dict | None = None,
        response_dir: str | None = None,
    ) -> "Corrections":
        """Construct corrections from a rest-frame SED by fitting kcorrect coeffs.

        This is a convenience wrapper around `kcorrect(...)` that first fits
        a non-negative kcorrect coefficient vector (NNLS) to the provided
        rest-frame SED (wave, flux), then evaluates kcorrect on `z`.

        Args:
            z: Redshift grid to evaluate on (1D).
            response: kcorrect response name, e.g. "sdss_r0" or "bessell_V".
            sed_wave_A: Rest-frame wavelength array in Angstrom.
            sed_flux: Rest-frame flux array (linear units; arbitrary scaling ok).
            band_shift: Optional band-shift parameter passed to kcorrect.
            weighted_fit: If True, apply a simple rest-grid weighting when
                solving NNLS for the kcorrect coefficients.
            z_ref_weight: Reference redshift used in the weighting heuristic.
            method: Interpolation method ("pchip", "akima", "linear").
            extrapolate: Whether to allow extrapolation beyond the valid
                evaluation range. Uses `extrap_mode="linear_tail"`.
            anchor_z0: If True, enforce K(z0)=0 at the first valid point.
            e_model: Evolution model selection ("none", "poggianti", "parametric").
            e_kwargs: Keyword arguments for the selected E backend.
            response_dir: Optional directory for kcorrect response files.

        Returns:
            Corrections instance built from fitted coeffs + selected E backend.
        """

        z_arr = np.asarray(z, float)
        if z_arr.ndim != 1 or z_arr.size < 2 or np.any(~np.isfinite(z_arr)):
            raise ValueError("z must be a finite 1D array with >=2 points.")

        sed_wave_A = np.asarray(sed_wave_A, float)
        sed_flux = np.asarray(sed_flux, float)
        ok = np.isfinite(sed_wave_A) & np.isfinite(sed_flux)
        sed_wave_A = sed_wave_A[ok]
        sed_flux = sed_flux[ok]
        if sed_wave_A.size < 5:
            raise ValueError("sed_wave_A/sed_flux must contain >=5 finite points.")

        order = np.argsort(sed_wave_A)
        sed_wave_A = sed_wave_A[order]
        sed_flux = sed_flux[order]

        kc = build_kcorrect(
            responses_in=[str(response)],
            responses_out=[str(response)],
            responses_map=[str(response)],
            response_dir=response_dir,
        )

        tw = np.asarray(kc.templates.restframe_wave, float)  # [Nw]
        tf = np.asarray(kc.templates.restframe_flux, float)  # [Ntmpl, Nw]

        sed_on_tw = np.interp(
            tw,
            sed_wave_A,
            sed_flux,
            left=float(sed_flux[0]),
            right=float(sed_flux[-1]),
        )

        A = tf.T  # [Nw, Ntmpl]
        b = sed_on_tw

        if weighted_fit:
            # A minimal, stable weighting heuristic (non-zero everywhere).
            w = np.sum(np.abs(tf), axis=0)
            w = np.clip(w, 0.0, None)
            if w.max() > 0:
                w = w / w.max()
            w = np.maximum(w, 1e-4)
            sw = np.sqrt(w)
            A = A * sw[:, None]
            b = b * sw

        # Scale target to avoid numerical extremes (NNLS is scale-sensitive).
        scale = np.median(b[b > 0]) if np.any(b > 0) else np.nan
        if np.isfinite(scale) and scale > 0:
            b = b / scale

        coeffs, _ = nnls(A, b)

        out = cls.kcorrect(
            z=z_arr,
            response=str(response),
            coeffs=coeffs,
            band_shift=band_shift,
            method=method,
            extrapolate=extrapolate,
            anchor_z0=anchor_z0,
            e_model=e_model,
            e_kwargs=e_kwargs,
            response_dir=response_dir,
        )
        out.meta.update(
            {
                "coeffs_source": "fit_to_sed",
                "weighted_fit": bool(weighted_fit),
                "z_ref_weight": float(z_ref_weight),
            }
        )
        return out

    @classmethod
    def kcorrect_from_pkg(
        cls,
        *,
        pkg: dict,
        gal_type: str,
        out_band: str,
        method: str = "pchip",
        extrapolate: bool = True,
        e_model: str = "none",
        e_kwargs: dict | None = None,
    ) -> "Corrections":
        """Construct corrections using precomputed kcorrect grid K values.

        This is the "grid consumer" backend. It selects a stored K(z) series
        from `pkg` and builds an interpolator (with optional extrapolation).

        Args:
            pkg: Precomputed kcorrect package dictionary containing SED grids.
            gal_type: Galaxy type identifier in the grid.
            out_band: Output photometric band name (must be in pkg["responses_out"]).
            method: Interpolation method ("pchip", "akima", "linear").
            extrapolate: Whether to allow extrapolation.
            e_model: Evolution model selection ("none", "poggianti", "parametric").
            e_kwargs: Keyword arguments passed to the selected E backend.

        Returns:
            Corrections instance combining precomputed kcorrect K values
            with the selected E backend.

        Raises:
            KeyError/ValueError: If pkg is missing required keys,
                or gal_type/out_band are not present.
        """
        # ---- validate pkg schema (minimal + explicit) ----
        required = ("z", "responses_out", "types", "K")
        missing = [k for k in required if k not in pkg]
        if missing:
            raise KeyError(f"pkg is missing required keys: {missing}")

        z = np.asarray(pkg["z"], float)
        if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)):
            raise ValueError("pkg['z'] must be a finite 1D array with >=2 points.")

        types = list(pkg["types"])
        if gal_type not in types:
            raise ValueError(f"gal_type={gal_type!r} not in pkg['types']={types!r}")

        responses_out = list(pkg["responses_out"])
        if out_band not in responses_out:
            raise ValueError(f"out_band={out_band!r} not in pkg['responses_out']={responses_out!r}")

        j = responses_out.index(out_band)
        ktz = np.asarray(pkg["K"][gal_type], float)  # (Nz, Nband)

        if ktz.ndim != 2:
            raise ValueError(f"pkg['K'][{gal_type!r}] must be 2D (Nz, Nband).")

        if ktz.shape[0] != z.size:
            raise ValueError(
                f"Shape mismatch for type={gal_type!r}: K has Nz={ktz.shape[0]} vs z={z.size}."
            )

        if ktz.shape[1] != len(responses_out):
            raise ValueError(
                f"Shape mismatch: K has Nband={ktz.shape[1]} vs responses_out={len(responses_out)}."
            )

        k_series = np.asarray(ktz[:, j], float)
        ok = np.isfinite(z) & np.isfinite(k_series)

        n_ok = int(np.count_nonzero(ok))

        if n_ok < 2:

            def _bad_k(z_in: ArrayLike) -> np.ndarray:
                z_arr = np.asarray(z_in, float)
                return np.full_like(z_arr, np.nan, dtype=float)

            _bad_k.__doc__ = (
                f"K(z) unavailable: gal_type={gal_type!r}, out_band={out_band!r} "
                f"(n_finite={n_ok} of {z.size})."
            )
            k_func = _bad_k
        else:
            k_func = build_1d_interpolator(
                z[ok],
                k_series[ok],
                method=method,
                extrapolate=extrapolate,
                extrap_mode="linear_tail",
            )

        e_func = cls._build_e_backend(e_model=str(e_model), e_kwargs=e_kwargs)

        out = cls(k_func=k_func, e_func=e_func)
        out.meta.update(
            {
                "backend": "kcorrect_precomputed",
                "gal_type": gal_type,
                "out_band": out_band,
                "n_finite": n_ok,
            }
        )
        return out

    @classmethod
    def kcorrect_color_split_coeffs(
        cls,
        *,
        M_r_ref: float = -21.5,
        a: float = 0.12,
        b: float = -0.025,
        red_offset: float = 0.10,
        blue_offset: float = 0.20,
        mag_r: float = 20.0,
        ivar_level: float = 1e10,
    ) -> dict[str, object]:
        """Build survey-independent red/blue kcorrect coefficient anchors.

        This constructs two representative SEDs in kcorrect template space by
        defining a rest-frame (g-r) color cut at z=0 and fitting coefficients for
        one red-sequence anchor and one blue-cloud anchor.

        The cut is parameterized as:

            (g-r)_cut(M_r) = a + b * M_r

        Anchors are defined at a representative magnitude M_r_ref:

            (g-r)_red  = (g-r)_cut(M_r_ref) + red_offset
            (g-r)_blue = (g-r)_cut(M_r_ref) - blue_offset

        The fitted coefficient vectors are intended to be reused across surveys
        (SDSS/DECam/HSC). Compute K(z) for any survey by passing the same coeffs
        to `Corrections.kcorrect(...)` with the desired response curve.

        Args:
            M_r_ref: Representative r-band absolute magnitude to evaluate the cut.
            a: Intercept of the linear cut model.
            b: Slope of the linear cut model.
            red_offset: Offset above the cut for the red anchor.
            blue_offset: Offset below the cut for the blue anchor.
            mag_r: Reference r magnitude used for coefficient fitting (normalization only).
            ivar_level: Inverse-variance weight for the constrained g and r bands.

        Returns:
            A dictionary with entries:
              - 'cut': {'g_minus_r': float}
              - 'red': {'g_minus_r': float, 'coeffs': np.ndarray}
              - 'blue': {'g_minus_r': float, 'coeffs': np.ndarray}
        """
        return fit_red_blue_anchors(
            M_r_ref=M_r_ref,
            a=a,
            b=b,
            red_offset=red_offset,
            blue_offset=blue_offset,
            mag_r=mag_r,
            responses=None,  # SDSS u/g/r/i/z inside the module
            ivar_level=ivar_level,
        )

    @classmethod
    def kcorrect_color_split(
        cls,
        *,
        z: ArrayLike,
        response: str,
        band_shift: float | None = None,
        method: str = "pchip",
        extrapolate: bool = True,
        anchor_z0: bool = True,
        e_model: str = "none",
        e_kwargs: dict | None = None,
        # color-split params
        M_r_ref: float = -21.5,
        a: float = 0.12,
        b: float = -0.025,
        red_offset: float = 0.10,
        blue_offset: float = 0.20,
        mag_r: float = 20.0,
        ivar_level: float = 1e10,
    ) -> dict[str, "Corrections"]:
        """Construct red/blue kcorrect Corrections objects from a color split.

        This is a convenience wrapper that:
          1) fits red/blue coefficient vectors via `kcorrect_color_split_coeffs`, and
          2) constructs `Corrections.kcorrect(...)` for each anchor.

        Args:
            z: Redshift grid for evaluation.
            response: kcorrect response name (e.g. 'sdss_r0', 'decam_r', 'subaru_r').
            band_shift: Optional band shift for kcorrect.
            method: Interpolation method.
            extrapolate: Whether to extrapolate beyond valid evaluation range.
            anchor_z0: Whether to enforce K(z0)=0 at the first valid point.
            e_model: Evolution backend name ('none', 'poggianti', 'parametric').
            e_kwargs: Keyword args for the E backend.
            M_r_ref: Representative magnitude for evaluating the color cut.
            a: Intercept of the linear cut model.
            b: Slope of the linear cut model.
            red_offset: Offset above the cut for the red anchor.
            blue_offset: Offset below the cut for the blue anchor.
            mag_r: Reference r magnitude for coefficient fitting.
            ivar_level: Inverse-variance weight for g/r constraints.

        Returns:
            Dictionary with keys 'red' and 'blue', each mapping to a `Corrections`
            instance configured for the requested response and redshift grid.
        """
        anchors = cls.kcorrect_color_split_coeffs(
            M_r_ref=M_r_ref,
            a=a,
            b=b,
            red_offset=red_offset,
            blue_offset=blue_offset,
            mag_r=mag_r,
            ivar_level=ivar_level,
        )

        out: dict[str, Corrections] = {}
        for label in ("red", "blue"):
            out[label] = cls.kcorrect(
                z=z,
                response=response,
                coeffs=np.asarray(anchors[label]["coeffs"], float),
                band_shift=band_shift,
                method=method,
                extrapolate=extrapolate,
                anchor_z0=anchor_z0,
                e_model=e_model,
                e_kwargs=e_kwargs,
            )
            out[label].meta.update(
                {
                    "coeffs_source": "color_split",
                    "color_model": {"a": a, "b": b, "M_r_ref": M_r_ref},
                    "color_anchors": {
                        "cut": float(anchors["cut"]["g_minus_r"]),
                        "red": float(anchors["red"]["g_minus_r"]),
                        "blue": float(anchors["blue"]["g_minus_r"]),
                    },
                    "color_label": label,
                }
            )
        return out

    @classmethod
    def register_kcorrect_response(
        cls,
        *,
        name: str,
        wave_angst: np.ndarray,
        throughput: np.ndarray,
        out_dir: str | Path,
        normalize: bool = True,
        validate_visible_to_kcorrect: bool = True,
    ) -> dict[str, object]:
        """Register (write) a custom survey response curve for kcorrect.

        This writes a kcorrect-compatible `<name>.dat` file into `out_dir`.
        If your installed `kcorrect` supports `response_dir=...`, you can then
        pass `response_dir=str(out_dir)` to `Corrections.kcorrect(...)`.

        If `validate_visible_to_kcorrect` is True, we also verify the response
        is discoverable via `list_available_responses(out_dir)`.

        Returns:
            Metadata dict including the response name and path.
        """
        out_dir = Path(out_dir)
        resp_name = write_kcorrect_response(
            name=str(name),
            wave_angst=np.asarray(wave_angst, float),
            throughput=np.asarray(throughput, float),
            out_dir=out_dir,
            normalize=bool(normalize),
        )

        meta: dict[str, object] = {
            "response": resp_name,
            "out_dir": str(out_dir),
            "path": str(out_dir / f"{resp_name}.dat"),
        }

        if validate_visible_to_kcorrect:
            avail = set(list_available_responses(out_dir))
            if resp_name not in avail:
                raise RuntimeError(
                    f"Wrote {resp_name}.dat into {out_dir}, but it is not discoverable "
                    f"via list_available_responses(out_dir). Something is off with the file."
                )

        return meta