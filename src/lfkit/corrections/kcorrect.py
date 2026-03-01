"""K-correction grid generator using the kcorrect package.

This module builds redshift-dependent k-correction tables
analogous to Poggianti-style tabulations, but computed using
the `kcorrect` template-fitting framework.

The workflow:

1. Configure a `Kcorrect` instance for a set of input and
   output filter response curves.
2. Evaluate k(z) on a predefined redshift grid for a small
   number of representative galaxy template types.
3. Store the resulting grid for reuse and construct smooth
   interpolators per (type, band).

The resulting tables are suitable for luminosity function
integrals, survey forecasts, and cosmology-dependent modeling.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import kcorrect.kcorrect as kk

from ..utils.units import mag_to_maggies
from ..utils.interpolation import Interpolator, build_1d_interpolator


@lru_cache(maxsize=8)
def _discover_response_dir_auto() -> Path:
    """Locate the kcorrect response directory (AUTO discovery), cached.

    This is the expensive part in many setups (can trigger package scanning).
    We cache the resolved directory Path so repeated calls are cheap.
    """
    # 1) Try via a Kcorrect instance (attr names differ by version)
    try:
        kc = kk.Kcorrect(responses=["sdss_r0"])
        for attr in (
            "response_dir",
            "responses_dir",
            "response_dirname",
            "filters_dir",
            "filter_dir",
            "response_path",
        ):
            if hasattr(kc, attr):
                p = Path(getattr(kc, attr))
                if p.exists():
                    return p
    except Exception:
        pass

    # 2) Fallback: search inside the installed kcorrect package for *.dat
    import kcorrect as _kcorrect_pkg

    base = Path(_kcorrect_pkg.__file__).resolve().parent

    candidates = [
        base,
        base / "data",
        base / "responses",
        base / "response",
        base / "filters",
    ]

    for c in candidates:
        if c.exists() and any(c.glob("*.dat")):
            return c

    dats = list(base.rglob("*.dat"))
    if not dats:
        raise FileNotFoundError(
            "Could not locate kcorrect response .dat files automatically. "
            "Pass response_dir=... explicitly."
        )

    from collections import Counter

    parent = Counter([p.parent for p in dats]).most_common(1)[0][0]
    return Path(parent)


@lru_cache(maxsize=16)
def _cached_available_responses(response_dir_key: str) -> tuple[str, ...]:
    """Cached list of available responses for a given directory key.

    response_dir_key is either "__AUTO__" or a resolved absolute path string.
    """
    if response_dir_key == "__AUTO__":
        response_dir = _discover_response_dir_auto()
    else:
        response_dir = Path(response_dir_key)

    if not response_dir.exists():
        raise FileNotFoundError(f"response_dir does not exist: {response_dir}")

    return tuple(sorted(p.stem for p in response_dir.glob("*.dat")))


def list_available_responses(
    response_dir: str | Path | None = None,
) -> list[str]:
    """List available kcorrect response (filter) names.

    If ``response_dir`` is provided, this function lists all ``*.dat`` files
    in that directory and returns their stem names.

    If ``response_dir`` is None, it attempts to discover the packaged kcorrect
    response directory in a version-robust way:
      1) Probe a ``kk.Kcorrect`` instance for known directory attributes.
      2) Fallback to searching inside the installed kcorrect package for
         ``*.dat`` files.

    Args:
        response_dir: Optional directory containing kcorrect response ``.dat``
            files. If None, attempts automatic discovery.

    Returns:
        Sorted list of available response names (file stems).

    Raises:
        FileNotFoundError: If no response directory can be located or the
            provided ``response_dir`` does not exist.
    """
    if response_dir is None:
        key = "__AUTO__"
    else:
        key = str(Path(response_dir).resolve())

    return list(_cached_available_responses(key))


def require_responses(
    responses: list[str],
    response_dir: str | Path | None = None,
) -> None:
    """Validate that requested response names exist.

    Args:
        responses: List of response names to validate.
        response_dir: Optional directory containing response `.dat`
            files. If None, uses packaged kcorrect responses.

    Raises:
        ValueError: If one or more requested responses are not found.
    """
    key = "__AUTO__" if response_dir is None else str(Path(response_dir).resolve())
    avail = set(_cached_available_responses(key))
    missing = [r for r in responses if r not in avail]
    if missing:
        raise ValueError(
            "Missing kcorrect responses: "
            f"{missing}. Example available: {sorted(list(avail))[:25]} ..."
        )


def _kc_cache_key(
    *,
    responses_in: tuple[str, ...],
    responses_out: tuple[str, ...],
    responses_map: tuple[str, ...],
    response_dir: str | Path | None,
    redshift_range: tuple[float, float],
    nredshift: int,
    abcorrect: bool,
) -> tuple:
    """Build a hashable cache key for Kcorrect construction."""
    if response_dir is None:
        rdir_key = "__AUTO__"
    else:
        rdir_key = str(Path(response_dir).resolve())

    return (
        responses_in,
        responses_out,
        responses_map,
        rdir_key,
        (float(redshift_range[0]), float(redshift_range[1])),
        int(nredshift),
        bool(abcorrect),
    )


@lru_cache(maxsize=8)
def _build_kcorrect_cached(key: tuple) -> kk.Kcorrect:
    """Internal cached constructor; key is produced by _kc_cache_key."""
    (
        responses_in,
        responses_out,
        responses_map,
        rdir_key,
        redshift_range,
        nredshift,
        abcorrect,
    ) = key

    # Validate filters against cached directory listing
    response_dir: str | Path | None
    if rdir_key == "__AUTO__":
        response_dir = None
    else:
        response_dir = Path(rdir_key)

    require_responses(list(responses_in), response_dir)
    require_responses(list(responses_out), response_dir)
    require_responses(list(responses_map), response_dir)

    kwargs: dict[str, Any] = dict(
        responses=list(responses_in),
        responses_out=list(responses_out),
        responses_map=list(responses_map),
        redshift_range=[float(redshift_range[0]), float(redshift_range[1])],
        nredshift=int(nredshift),
        abcorrect=bool(abcorrect),
    )

    if response_dir is not None:
        # Only pass response_dir if this kcorrect version supports it.
        sig = inspect.signature(kk.Kcorrect)
        if "response_dir" in sig.parameters:
            kwargs["response_dir"] = str(response_dir)
        else:
            raise TypeError(
                "This kcorrect version does not support response_dir=... in kk.Kcorrect(...). "
                "To use custom filters, you must either (a) install a kcorrect build that "
                "supports response_dir, or (b) place your *.dat filter files into kcorrect’s "
                "packaged response directory."
            )

    return kk.Kcorrect(**kwargs)


def build_kcorrect(
    *,
    responses_in: list[str],
    responses_out: list[str] | None = None,
    responses_map: list[str] | None = None,
    response_dir: str | Path | None = None,
    redshift_range: tuple[float, float] = (0.0, 2.0),
    nredshift: int = 4000,
    abcorrect: bool = False,
) -> kk.Kcorrect:
    """Construct a ``kcorrect.Kcorrect`` instance for a given response setup.

    Args:
        responses_in: Names of input response curves used to fit SED
            template coefficients.
        responses_out: Names of output (rest-frame) response curves for which
            k-corrections will be computed. Defaults to ``responses_in``.
        responses_map: Names of response curves used internally by
            ``kcorrect`` when mapping observed to rest-frame quantities.
            Defaults to ``responses_in``.
        response_dir: Optional directory containing custom response
            ``.dat`` files. If ``None``, packaged responses are used.
        redshift_range: Minimum and maximum redshift used internally
            by ``kcorrect`` to build its lookup grid.
        nredshift: Number of redshift samples used in the internal grid.
        abcorrect: Whether to apply AB corrections in ``kcorrect``.

    Returns:
        A configured ``kk.Kcorrect`` object ready for k-correction
        evaluation.

    Raises:
        ValueError: If any requested response curve is missing.
    """
    if responses_out is None:
        responses_out = responses_in
    if responses_map is None:
        responses_map = responses_in

    key = _kc_cache_key(
        responses_in=tuple(map(str, responses_in)),
        responses_out=tuple(map(str, responses_out)),
        responses_map=tuple(map(str, responses_map)),
        response_dir=response_dir,
        redshift_range=(float(redshift_range[0]), float(redshift_range[1])),
        nredshift=int(nredshift),
        abcorrect=bool(abcorrect),
    )
    return _build_kcorrect_cached(key)


def compute_k_table(
    *,
    kc: kk.Kcorrect,
    z_grid: np.ndarray,
    coeffs_by_type: dict[str, np.ndarray],
    band_shift: float | None = None,
    anchor_z0: bool = True,
) -> dict[str, np.ndarray]:
    """Compute a k-correction table on a redshift grid for multiple SED types.

    For each galaxy type (identified by a coefficient vector in kcorrect
    template space), this function evaluates ``kc.kcorrect`` across the
    supplied redshift grid and returns an array of shape ``(Nz, Nband)``.

    If ``anchor_z0`` is True, the table is normalized so that ``K(z=0)=0``
    by subtracting the first row from all rows.

    Args:
        kc: Configured ``kk.Kcorrect`` instance.
        z_grid: One-dimensional redshift grid.
        coeffs_by_type: Mapping from type name to a kcorrect coefficient
            vector with shape ``(n_templates,)``.
        band_shift: Optional band shift passed through to ``kc.kcorrect``.
            Use e.g. ``0.1`` for ^0.1 systems. If None, no shift is applied.
        anchor_z0: If True, enforce ``K(z=0)=0`` by subtracting the first
            row of the computed table for each type.

    Returns:
        Dictionary mapping ``type_name -> K`` where ``K`` has shape
        ``(Nz, Nband)`` and ``Nband`` corresponds to ``kc.responses_out``.

    Raises:
        ValueError: If ``z_grid`` is not a finite 1D array with at least two
            points, if an all-NaN grid is produced, or if internal shape
            consistency checks fail.
    """
    z = np.asarray(z_grid, float)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)):
        raise ValueError("z_grid must be a finite 1D array with >=2 points.")

    out: dict[str, np.ndarray] = {}

    for tname, coeffs in coeffs_by_type.items():
        c = np.asarray(coeffs, float)

        rows = []
        nband = None

        for zi in z:
            zi = float(zi)

            if zi == 0.0 and anchor_z0:
                if nband is None:
                    # discover band count robustly
                    if band_shift is None:
                        test = kc.kcorrect(redshift=1e-6, coeffs=c)
                    else:
                        test = kc.kcorrect(redshift=1e-6, coeffs=c, band_shift=float(band_shift))
                    nband = len(np.asarray(test, float))
                kcorr = np.zeros(nband, dtype=float)

            else:
                if band_shift is None:
                    kcorr = kc.kcorrect(redshift=zi, coeffs=c)
                else:
                    kcorr = kc.kcorrect(redshift=zi, coeffs=c, band_shift=float(band_shift))

                kcorr = np.asarray(kcorr, float)
                if nband is None:
                    nband = kcorr.size

            kcorr = np.asarray(kcorr, float)
            kcorr = np.where(np.isfinite(kcorr), kcorr, np.nan)

            if tname == "E" and not np.all(np.isfinite(kcorr)):
                print("BAD:", tname, "z=", zi, "kcorr=", kcorr)
                break

            rows.append(kcorr)

        karr = np.vstack(rows)

        if karr.shape[0] != z.size:
            raise ValueError(f"BUG: built K with Nz={karr.shape[0]} but z has Nz={z.size}")

        if anchor_z0:
            if not np.any(np.isfinite(karr)):
                raise ValueError(f"All-NaN K grid for type={tname!r}.")
            karr = karr - karr[0:1, :]

        out[tname] = karr

    return out


def build_kcorr_grid_package(
    *,
    responses_in: list[str],
    responses_out: list[str] | None,
    responses_map: list[str] | None,
    coeffs_by_type: dict[str, np.ndarray],
    z_grid: np.ndarray,
    band_shift: float | None = 0.1,
    response_dir: str | Path | None = None,
    redshift_range: tuple[float, float] = (0.0, 3.5),
    nredshift: int = 4000,
) -> dict[str, Any]:
    """Generate a cached k-correction grid package.

    This constructs a ``kk.Kcorrect`` instance for the given response setup,
    evaluates k(z) for each galaxy type across the provided redshift grid,
    and returns a dictionary suitable for serialization and interpolation.

    Args:
        responses_in: Input response names used to fit template coefficients.
        responses_out: Output (rest-frame) response names. If None, defaults
            to ``responses_in``.
        responses_map: Response mapping used internally by ``kcorrect``.
        coeffs_by_type: Mapping from galaxy type name to a coefficient vector
            with shape ``(n_templates,)``.
        z_grid: One-dimensional redshift grid.
        band_shift: Optional band shift (e.g. ``0.1`` for ^0.1 systems).
        response_dir: Optional directory containing custom response curves.
        redshift_range: Internal redshift range for ``kcorrect``'s lookup grid.
        nredshift: Number of samples in the internal kcorrect redshift grid.

    Returns:
        A dictionary with keys:
            - ``meta``: Backend and configuration metadata.
            - ``z``: Redshift grid.
            - ``responses_in``: Input response names.
            - ``responses_out``: Output response names.
            - ``responses_map``: Response mapping names.
            - ``types``: Sorted galaxy type names.
            - ``K``: Mapping ``type -> (Nz, Nband)`` k-correction array.

    Raises:
        ValueError: If responses are missing, coefficient vectors have the
            wrong shape, contain non-finite values, contain negative entries,
            or if a valid K table cannot be constructed.
    """
    if responses_out is None:
        responses_out = responses_in
    if responses_map is None:
        responses_map = responses_in

    kc = build_kcorrect(
        responses_in=responses_in,
        responses_out=responses_out,
        responses_map=responses_map,
        response_dir=response_dir,
        redshift_range=redshift_range,
        nredshift=nredshift,
    )

    nt = kc.templates.restframe_flux.shape[0]
    print("n_templates =", nt)

    for name, c in coeffs_by_type.items():
        c = np.asarray(c, float)

        print(
            name,
            "shape", c.shape,
            "finite", np.all(np.isfinite(c)),
            "min", np.min(c) if c.size else None,
            "sum", np.sum(c) if c.size else None,
        )

        if c.shape != (nt,):
            raise ValueError(f"{name}: coeff shape {c.shape} != ({nt},)")
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{name}: coeffs contain non-finite values: {c}")
        if np.any(c < 0):
            raise ValueError(f"{name}: negative coeffs not allowed: {c}")
        if np.sum(c) <= 0:
            raise ValueError(f"{name}: coeffs sum <= 0: {c}")

    kcorr = compute_k_table(
        kc=kc,
        z_grid=z_grid,
        coeffs_by_type=coeffs_by_type,
        band_shift=band_shift,
    )

    return dict(
        meta=dict(
            backend="kcorrect",
            band_shift=band_shift,
            redshift_range=(float(z_grid[0]), float(z_grid[-1])),
            nredshift=int(nredshift),
            response_dir=str(response_dir) if response_dir is not None else None,
        ),
        z=np.asarray(z_grid, float),
        responses_in=list(responses_in),
        responses_out=list(responses_out),
        responses_map=list(responses_map),
        types=sorted(list(coeffs_by_type.keys())),
        K=kcorr,
    )


def make_type_coeffs_from_colors(
    *,
    responses: list[str],
    ref_mag: float = 20.0,
    ref_band: str | None = None,
    ref_z: float = 0.1,
    colors_by_type: dict[str, dict[str, object]] | None = None,
    ivar_level: float = 1e20,
    response_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Build representative kcorrect coefficient vectors from adjacent-band colors.

    You provide an ordered list of response names (bands). For each galaxy
    type, you provide colors between adjacent bands:

        colors[i] = m_i - m_{i+1}  for i = 0..Nband-2

    Given an anchor magnitude ``ref_mag`` in ``ref_band``, this defines a full
    synthetic magnitude vector across all bands, which is converted to maggies
    and fit with ``kcorrect`` to obtain a template coefficient vector.

    Args:
        responses: Ordered response names used for synthetic photometry and to
            initialize ``kk.Kcorrect``. The order defines which bands are
            treated as adjacent.
        ref_mag: AB magnitude assigned to the reference band.
        ref_band: Name of the reference band in ``responses``. If None, uses
            the middle band of ``responses``.
        ref_z: Redshift at which to fit the coefficients.
        colors_by_type: Mapping ``type -> {"colors": [...]}`` where the color
            list has length ``len(responses) - 1`` and encodes adjacent-band
            differences. If None, uses simple placeholder defaults.
        ivar_level: Constant inverse variance assigned to each synthetic maggie.
        response_dir: Optional custom response directory passed to ``kcorrect``.

    Returns:
        Dictionary mapping ``type -> coeffs`` where ``coeffs`` has shape
        ``(n_templates,)``.

    Raises:
        ValueError: If fewer than two responses are provided, if ``ref_band`` is
            not in ``responses``, if colors have the wrong length or contain
            non-finite values, or if ``fit_coeffs`` returns invalid coefficients
            (wrong shape, non-finite, negative, or non-positive sum).
    """
    if len(responses) < 2:
        raise ValueError("responses must have at least 2 bands.")

    # Default anchor band: middle of the list
    if ref_band is None:
        ref_band = responses[len(responses) // 2]

    band_to_idx = {b: i for i, b in enumerate(responses)}
    if ref_band not in band_to_idx:
        raise ValueError(f"ref_band={ref_band!r} not in responses={responses!r}")
    iref = band_to_idx[ref_band]

    ncol = len(responses) - 1
    if colors_by_type is None:
        # These are deliberately simple; tweak per survey later.
        colors_by_type = {
            "E": {"colors": [0.8] * ncol},
            "Sbc": {"colors": [0.4] * ncol},
            "Im": {"colors": [0.1] * ncol},
        }

    kc = build_kcorrect(
        responses_in=list(responses),
        responses_out=list(responses),
        responses_map=list(responses),
        response_dir=response_dir,
        redshift_range=(0.0, 2.0),
        nredshift=4000,
    )

    coeffs_by_type: dict[str, np.ndarray] = {}

    for tname, spec in colors_by_type.items():
        if "colors" not in spec:
            raise ValueError(f"{tname}: colors_by_type entry must"
                             f" include key 'colors'")

        colors = np.asarray(spec["colors"], float)
        if colors.shape != (ncol,):
            raise ValueError(f"{tname}: expected colors length {ncol} "
                             f"(Nband-1), got {colors.shape}")
        if not np.all(np.isfinite(colors)):
            raise ValueError(f"{tname}: non-finite values in colors: {colors}")

        # Construct magnitudes m[i] for all bands.
        # Convention: colors[i] = m[i] - m[i+1]
        m = np.full(len(responses), np.nan, dtype=float)
        m[iref] = float(ref_mag)

        # Propagate to the right: m[i+1] = m[i] - colors[i]
        for i in range(iref, len(responses) - 1):
            m[i + 1] = m[i] - colors[i]

        # Propagate to the left: m[i-1] = m[i] + colors[i-1]
        for i in range(iref, 0, -1):
            m[i - 1] = m[i] + colors[i - 1]

        if not np.all(np.isfinite(m)):
            raise ValueError(f"{tname}: failed to construct finite mags: {m}")

        maggies = mag_to_maggies(m)
        ivar = np.full_like(maggies, float(ivar_level))

        coeffs = kc.fit_coeffs(redshift=float(ref_z), maggies=maggies, ivar=ivar)
        coeffs = np.asarray(coeffs, float)

        nt = kc.templates.restframe_flux.shape[0]
        if coeffs.shape != (nt,):
            raise ValueError(f"{tname}: fit_coeffs returned {coeffs.shape}, "
                             f"expected ({nt},)")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError(f"{tname}: non-finite coeffs from fit: {coeffs}")
        if np.any(coeffs < 0):
            raise ValueError(f"{tname}: negative coeffs returned: {coeffs}")
        if float(np.sum(coeffs)) <= 0:
            raise ValueError(f"{tname}: coeff sum <= 0: {coeffs}")

        coeffs_by_type[tname] = coeffs

    return coeffs_by_type


def kcorr_interpolators(
    pkg: dict[str, Any],
    *,
    method: str = "pchip",
    extrapolate: bool = True,
) -> dict[str, dict[str, Interpolator | None]]:
    """Build interpolation objects for each galaxy type and band.

    Args:
        pkg: k-correction grid package produced by
            ``build_kcorr_grid_package``.
        method: Interpolation method passed to
            ``build_1d_interpolator`` (e.g. ``"pchip"``, ``"akima"``).
        extrapolate: Whether to allow extrapolation beyond
            the tabulated redshift range.

    Returns:
        Nested dictionary mapping:

            ``type -> band_name -> Interpolator``

    Raises:
        ValueError: If the shape of stored K arrays does not
            match the redshift grid or band configuration.
    """
    z = np.asarray(pkg["z"], float)
    responses_out = list(pkg["responses_out"])
    out: dict[str, dict[str, Interpolator | None]] = {}

    for t in pkg["types"]:
        ktz = np.asarray(pkg["K"][t], float)  # (Nz, Nband)

        if ktz.shape[0] != z.size:
            raise ValueError(f"Shape mismatch for type={t!r}: k has Nz"
                             f"={ktz.shape[0]} vs z={z.size}.")
        if ktz.shape[1] != len(responses_out):
            raise ValueError(
                f"Shape mismatch: k has Nband={ktz.shape[1]} vs "
                f"responses_out={len(responses_out)}."
            )

        out[t] = {}
        for j, band in enumerate(responses_out):
            y = np.asarray(ktz[:, j], float)
            ok = np.isfinite(z) & np.isfinite(y)

            if np.count_nonzero(ok) < 2:
                out[t][band] = None
                continue

            out[t][band] = build_1d_interpolator(
                z[ok],
                y[ok],
                method=method,
                extrapolate=extrapolate,
                extrap_mode="linear_tail",
            )

    return out
