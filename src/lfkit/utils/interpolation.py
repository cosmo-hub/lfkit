"""Interpolation utilities for 1D tabulated curves.

This module provides small helpers for preparing tabulated (x, y) data for
interpolation and constructing simple 1D interpolators with optional
extrapolation. These utilities are intended for internal use across LFKit
(e.g., correction tables, response curves, and other tabulated mappings).
"""

from __future__ import annotations

from typing import Callable, Literal, Union

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

InterpMethod = Literal["pchip", "akima", "linear"]
ExtrapMode = Literal["none", "native", "linear_tail"]
Interpolator = Union[PchipInterpolator, Akima1DInterpolator, Callable[[np.ndarray], np.ndarray]]

__all__ = (
    "linear_interp_extrap",
    "build_1d_interpolator",
    "prep_strictly_increasing_xy",
)


def linear_interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Interpolates linearly and extrapolates linearly outside the tabulated range.

    This function behaves like ``numpy.interp`` within the range spanned by
    ``xp``, but extends the curve outside that range using straight-line
    extrapolation from the first/last interval.

    Args:
        x: Query points.
        xp: Monotonic sample locations.
        fp: Sample values at ``xp``.

    Returns:
        Interpolated values at ``x`` with linear extrapolation beyond
        ``[xp[0], xp[-1]]``.

    Raises:
        ValueError: If ``xp`` and ``fp`` have different lengths.
    """
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)

    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape.")

    y = np.interp(x, xp, fp)

    if xp.size < 2:
        return y

    left = x < xp[0]
    right = x > xp[-1]

    if np.any(left):
        m0 = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left] = fp[0] + m0 * (x[left] - xp[0])

    if np.any(right):
        m1 = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right] = fp[-1] + m1 * (x[right] - xp[-1])

    return y


def build_1d_interpolator(
    z: np.ndarray,
    y: np.ndarray,
    *,
    method: InterpMethod,
    extrapolate: bool,
    extrap_mode: ExtrapMode = "native",
) -> Interpolator:
    """Builds a 1D interpolator for tabulated data.

    Args:
        z: Sample locations.
        y: Sample values at ``z``.
        method: Interpolation method. Supported values are ``"pchip"``,
            ``"akima"``, and ``"linear"``.
        extrapolate: Whether to allow evaluation outside the tabulated range.

    Returns:
        A callable interpolator. For ``"pchip"`` and ``"akima"``, this is a SciPy
        interpolator object. For ``"linear"``, this is a function implementing
        linear interpolation (and optional linear extrapolation).

    Raises:
        ValueError: If ``method`` is not recognized or the inputs cannot be
            prepared for interpolation.
    """
    z, y = prep_strictly_increasing_xy(z, y)

    # no extrapolation requested
    if not extrapolate or extrap_mode == "none":
        if method == "pchip":
            return PchipInterpolator(z, y, extrapolate=False)
        if method == "akima":
            return Akima1DInterpolator(z, y, extrapolate=False)
        if method == "linear":
            def f_linear(zz: np.ndarray) -> np.ndarray:
                return np.interp(np.asarray(zz, float), z, y)

            return f_linear
        raise ValueError(f"Unknown method={method!r}.")

    # extrapolation requested
    if extrap_mode == "native":
        if method == "pchip":
            return PchipInterpolator(z, y, extrapolate=True)
        if method == "akima":
            return Akima1DInterpolator(z, y, extrapolate=True)
        if method == "linear":
            def f_linear_extrap(zz: np.ndarray) -> np.ndarray:
                return linear_interp_extrap(np.asarray(zz, float), z, y)

            return f_linear_extrap
        raise ValueError(f"Unknown method={method!r}.")

    if extrap_mode == "linear_tail":
        # build in-range interpolator WITHOUT extrapolation
        if method == "pchip":
            f_in = PchipInterpolator(z, y, extrapolate=False)
        elif method == "akima":
            f_in = Akima1DInterpolator(z, y, extrapolate=False)
        elif method == "linear":
            def f_in(zz: np.ndarray) -> np.ndarray:
                return np.interp(np.asarray(zz, float), z, y)
        else:
            raise ValueError(f"Unknown method={method!r}.")

        z0, z1 = z[0], z[-1]
        # slopes from the first/last interval
        m_left = (y[1] - y[0]) / (z[1] - z[0])
        m_right = (y[-1] - y[-2]) / (z[-1] - z[-2])

        def f(zz):
            zz = np.asarray(zz, float)
            out = np.full_like(zz, np.nan, dtype=float)

            mid = (zz >= z0) & (zz <= z1)
            if np.any(mid):
                out[mid] = f_in(zz[mid])

            left = zz < z0
            if np.any(left):
                out[left] = y[0] + m_left * (zz[left] - z0)

            right = zz > z1
            if np.any(right):
                out[right] = y[-1] + m_right * (zz[right] - z1)

            return out

        return f

    raise ValueError(f"Unknown extrap_mode={extrap_mode!r}.")


def prep_strictly_increasing_xy(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepares tabulated data for 1D interpolation.

    Args:
        z: Sample locations.
        y: Sample values at ``z``.

    Returns:
        A tuple ``(z_out, y_out)`` where ``z_out`` is strictly increasing and
        aligned with ``y_out``. Non-finite entries are removed.

    Raises:
        ValueError: If fewer than two valid sample points remain.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)

    ok = np.isfinite(z) & np.isfinite(y)
    z = z[ok]
    y = y[ok]

    order = np.argsort(z)
    z = z[order]
    y = y[order]

    keep = np.ones_like(z, dtype=bool)
    keep[1:] = z[1:] > z[:-1]
    z = z[keep]
    y = y[keep]

    if z.size < 2:
        raise ValueError("Need at least 2 points to build an interpolator.")

    return z, y
