"""Tabulated and binned luminosity function models.

This module provides non-parametric luminosity function models. Unlike
Schechter, Saunders, Gaussian, or power-law forms, these models do not assume a
fixed analytic shape. Instead, the luminosity function is supplied directly as
values on a magnitude grid or inside magnitude bins.

Two related representations are provided:

``tabulated_lf``
    Interpolates luminosity function values sampled at magnitude grid points.
    This is useful when an LF has already been measured, simulated, or fitted
    elsewhere and should be evaluated smoothly between tabulated points.

``binned_lf``
    Treats the luminosity function as piecewise constant inside magnitude bins.
    This is useful for directly representing measured binned luminosity
    functions without imposing interpolation structure inside each bin.

Redshift-dependent versions are also provided. These allow the LF to vary over
a two-dimensional grid in absolute magnitude and redshift.

These models are especially useful when the bright end, faint end, or redshift
evolution should be data-driven rather than forced into a Schechter-like or
Saunders-like form.
"""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import (
    validate_array,
    validate_tabulated_grid,
    validate_binned_grid,
    validate_2d_tabulated_grid,
    validate_2d_binned_grid,
)


__all__ = [
    "tabulated_lf",
    "binned_lf",
    "redshift_tabulated_lf",
    "redshift_binned_lf",
    "distance_tabulated_lf",
    "distance_binned_lf",
]


def tabulated_lf(
    absolute_mag: FloatInput,
    *,
    magnitude_grid: ParameterValue,
    phi_grid: ParameterValue,
    fill_value: float = 0.0,
    log_phi: bool = False,
) -> FloatArray:
    r"""Return a luminosity function interpolated from tabulated values.

    This evaluates a non-parametric luminosity function by interpolating
    tabulated values,

    .. math::

        \phi(M_i) = \phi_i,

    where ``magnitude_grid`` contains the sampled absolute magnitudes
    :math:`M_i` and ``phi_grid`` contains the corresponding luminosity function
    values :math:`\phi_i`.

    For magnitudes between tabulated points, this function performs linear
    interpolation,

    .. math::

        \phi(M) =
        \mathrm{interp}\left(M; \{M_i\}, \{\phi_i\}\right).

    If ``log_phi`` is True, interpolation is instead performed in
    :math:`\log_{10}\phi`,

    .. math::

        \log_{10}\phi(M) =
        \mathrm{interp}\left(
        M; \{M_i\}, \{\log_{10}\phi_i\}
        \right),

    and the result is transformed back to linear space. Logarithmic
    interpolation is often better for luminosity functions because LF values
    can vary by many orders of magnitude across the magnitude range.

    This model is useful when the luminosity function is known from a table,
    simulation, external fit, or observational measurement and no analytic
    Schechter-like form should be imposed. It gives a smooth representation of
    the tabulated curve, but it does not extrapolate beyond the supplied
    magnitude range. Values outside the table are set to ``fill_value``.

    Args:
        absolute_mag: Absolute magnitude value(s) where the luminosity function
            should be evaluated.
        magnitude_grid: One-dimensional strictly increasing grid of absolute
            magnitudes where the LF is tabulated.
        phi_grid: One-dimensional array of non-negative LF values evaluated at
            ``magnitude_grid``.
        fill_value: Value returned outside the tabulated magnitude range.
        log_phi: If True, interpolate in ``log10(phi)`` instead of directly in
            ``phi``.

    Returns:
        NumPy array containing the interpolated luminosity function evaluated
        at ``absolute_mag``.

    Raises:
        ValueError: If the magnitude grid is not one-dimensional, if the grid
            is not strictly increasing, if the grid and value arrays have
            inconsistent lengths, if any LF value is negative, if
            ``fill_value`` is not finite, or if ``log_phi`` is True and any
            tabulated LF value is not positive.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    magnitude_grid_arr, phi_grid_arr = validate_tabulated_grid(
        magnitude_grid,
        phi_grid,
        coordinate_name="magnitude_grid",
        values_name="phi_grid",
        positive_values=log_phi,
    )

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if log_phi:
        if np.any(phi_grid_arr <= 0.0):
            raise ValueError("phi_grid must be positive when log_phi is True.")

        log_phi_grid = np.log10(phi_grid_arr)

        if fill_value > 0.0:
            log_fill_value = np.log10(fill_value)
        elif fill_value == 0.0:
            log_fill_value = -np.inf
        else:
            raise ValueError("fill_value must be non-negative.")

        interpolated = np.interp(
            absolute_mag_arr,
            magnitude_grid_arr,
            log_phi_grid,
            left=log_fill_value,
            right=log_fill_value,
        )
        phi = 10.0**interpolated
        phi = np.where(np.isfinite(phi), phi, fill_value)
    else:
        if fill_value < 0.0:
            raise ValueError("fill_value must be non-negative.")

        phi = np.interp(
            absolute_mag_arr,
            magnitude_grid_arr,
            phi_grid_arr,
            left=fill_value,
            right=fill_value,
        )

    return np.asarray(phi, dtype=float)


def binned_lf(
    absolute_mag: FloatInput,
    *,
    magnitude_bin_edges: ParameterValue,
    phi_bin_values: ParameterValue,
    fill_value: float = 0.0,
) -> FloatArray:
    r"""Return a piecewise constant binned luminosity function.

    This evaluates a non-parametric luminosity function defined by constant
    values inside absolute magnitude bins. If the bin edges are

    .. math::

        M_0 < M_1 < \cdots < M_N,

    and the supplied bin values are :math:`\phi_j`, then

    .. math::

        \phi(M) = \phi_j
        \quad \mathrm{for} \quad
        M_j \le M < M_{j+1}.

    The final edge is treated as the upper boundary of the last bin. Values
    outside the supplied bin range are set to ``fill_value``.

    This representation is closest to how many observational luminosity
    functions are reported: as measurements in finite magnitude bins. Unlike
    ``tabulated_lf``, this function does not smooth or interpolate between
    measurements. It preserves the step-like structure of the binned estimate.

    This is useful when the bin values themselves are the model parameters,
    when one wants a deliberately non-parametric LF, or when interpolation
    between bins would imply more information than the data actually provide.

    Args:
        absolute_mag: Absolute magnitude value(s) where the luminosity function
            should be evaluated.
        magnitude_bin_edges: One-dimensional strictly increasing absolute
            magnitude bin edges. This array must have one more element than
            ``phi_bin_values``.
        phi_bin_values: One-dimensional array of non-negative LF values inside
            each magnitude bin.
        fill_value: Value returned outside the supplied magnitude bin range.

    Returns:
        NumPy array containing the binned luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If the bin edges or bin values are not one-dimensional, if
            the number of edges is not one larger than the number of values, if
            the edges are not strictly increasing, if any LF value is negative,
            or if ``fill_value`` is not finite and non-negative.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    edges, values = validate_binned_grid(
        magnitude_bin_edges,
        phi_bin_values,
        edges_name="magnitude_bin_edges",
        values_name="phi_bin_values",
    )

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if fill_value < 0.0:
        raise ValueError("fill_value must be non-negative.")

    indices = np.searchsorted(edges, absolute_mag_arr, side="right") - 1
    valid = (indices >= 0) & (indices < values.size)

    phi = np.full(absolute_mag_arr.shape, fill_value, dtype=float)
    phi[valid] = values[indices[valid]]

    return np.asarray(phi, dtype=float)


def redshift_tabulated_lf(
    absolute_mag: FloatInput,
    redshift: FloatInput,
    *,
    magnitude_grid: ParameterValue,
    redshift_grid: ParameterValue,
    phi_grid: ParameterValue,
    fill_value: float = 0.0,
    log_phi: bool = False,
) -> FloatArray:
    r"""Return a luminosity function interpolated over magnitude and redshift.

    This evaluates a redshift-dependent tabulated luminosity function,

    .. math::

        \phi(M_i, z_k) = \phi_{k i},

    where ``magnitude_grid`` contains absolute magnitudes :math:`M_i`,
    ``redshift_grid`` contains redshifts :math:`z_k`, and ``phi_grid`` has
    shape

    .. math::

        (N_z, N_M).

    For an input pair :math:`(M, z)`, the model performs bilinear interpolation
    on the supplied grid. Operationally, it first interpolates in magnitude at
    each tabulated redshift and then interpolates those values in redshift.

    If ``log_phi`` is True, the interpolation is performed in
    :math:`\log_{10}\phi`,

    .. math::

        \log_{10}\phi(M, z)
        =
        \mathrm{interp}_{M,z}
        \left[
        \log_{10}\phi(M_i, z_k)
        \right],

    and the result is converted back to linear space.

    This model is useful when the LF is measured or simulated in several
    redshift slices and one wants a smooth LF surface :math:`\phi(M,z)` without
    assuming an analytic redshift evolution law. It is more flexible than an
    evolving Schechter or evolving Saunders form because each redshift slice can
    have an arbitrary shape.

    Values outside the supplied magnitude or redshift grid are not extrapolated.
    They are set to ``fill_value``.

    Args:
        absolute_mag: Absolute magnitude value(s) where the luminosity function
            should be evaluated.
        redshift: Redshift value(s) where the luminosity function should be
            evaluated. Must be non-negative.
        magnitude_grid: One-dimensional strictly increasing absolute magnitude
            grid.
        redshift_grid: One-dimensional strictly increasing non-negative
            redshift grid.
        phi_grid: Two-dimensional array of non-negative LF values with shape
            ``(redshift_grid.size, magnitude_grid.size)``.
        fill_value: Value returned outside the tabulated magnitude or redshift
            range.
        log_phi: If True, interpolate in ``log10(phi)`` instead of directly in
            ``phi``.

    Returns:
        NumPy array containing the redshift-dependent tabulated luminosity
        function evaluated at ``absolute_mag`` and ``redshift``.

    Raises:
        ValueError: If the magnitude or redshift grid is invalid, if
            ``phi_grid`` has the wrong shape, if any LF value is negative, if
            any requested redshift is negative, if ``fill_value`` is not finite,
            or if ``log_phi`` is True and any tabulated LF value is not
            positive.
    """
    absolute_mag_arr, redshift_arr = np.broadcast_arrays(
        validate_array(absolute_mag, name="absolute_mag"),
        validate_array(redshift, name="redshift"),
    )
    magnitude_grid_arr, redshift_grid_arr, phi_grid_arr = validate_2d_tabulated_grid(
        magnitude_grid,
        redshift_grid,
        phi_grid,
        x_name="magnitude_grid",
        y_name="redshift_grid",
        values_name="phi_grid",
        allow_negative_y=False,
        positive_values=log_phi,
    )

    if np.any(redshift_arr < 0.0):
        raise ValueError("redshift must be non-negative.")

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if fill_value < 0.0:
        raise ValueError("fill_value must be non-negative.")

    if log_phi:
        if np.any(phi_grid_arr <= 0.0):
            raise ValueError("phi_grid must be positive when log_phi is True.")

        table = np.log10(phi_grid_arr)
        outside_value = np.log10(fill_value) if fill_value > 0.0 else -np.inf
    else:
        table = phi_grid_arr
        outside_value = fill_value

    out = np.full(absolute_mag_arr.shape, outside_value, dtype=float)

    for index in np.ndindex(absolute_mag_arr.shape):
        mag = absolute_mag_arr[index]
        z = redshift_arr[index]

        outside_mag = mag < magnitude_grid_arr[0] or mag > magnitude_grid_arr[-1]
        outside_z = z < redshift_grid_arr[0] or z > redshift_grid_arr[-1]

        if outside_mag or outside_z:
            continue

        phi_at_mag_by_redshift = np.array(
            [
                np.interp(mag, magnitude_grid_arr, table[z_index])
                for z_index in range(redshift_grid_arr.size)
            ],
            dtype=float,
        )
        out[index] = np.interp(
            z,
            redshift_grid_arr,
            phi_at_mag_by_redshift,
        )

    if log_phi:
        out = 10.0**out
        out = np.where(np.isfinite(out), out, fill_value)

    return np.asarray(out, dtype=float)


def redshift_binned_lf(
    absolute_mag: FloatInput,
    redshift: FloatInput,
    *,
    magnitude_bin_edges: ParameterValue,
    redshift_bin_edges: ParameterValue,
    phi_bin_values: ParameterValue,
    fill_value: float = 0.0,
) -> FloatArray:
    r"""Return a piecewise constant binned luminosity function in magnitude and redshift.

    This evaluates a two-dimensional binned luminosity function. If the
    magnitude bin edges are

    .. math::

        M_0 < M_1 < \cdots < M_N,

    and the redshift bin edges are

    .. math::

        z_0 < z_1 < \cdots < z_K,

    then the supplied values define

    .. math::

        \phi(M,z) = \phi_{k j}
        \quad \mathrm{for} \quad
        M_j \le M < M_{j+1},
        \quad
        z_k \le z < z_{k+1}.

    The ``phi_bin_values`` array must therefore have shape

    .. math::

        (K, N)
        =
        (N_z - 1, N_M - 1).

    This model is the redshift-dependent analogue of ``binned_lf``. It is
    useful for representing luminosity functions measured independently in
    redshift and magnitude bins. It does not assume smoothness in either
    direction. That makes it useful for conservative non-parametric analyses,
    survey selection studies, or tests where each LF bin is treated as an
    independent degree of freedom.

    Values outside the supplied magnitude or redshift bin ranges are set to
    ``fill_value``.

    Args:
        absolute_mag: Absolute magnitude value(s) where the luminosity function
            should be evaluated.
        redshift: Redshift value(s) where the luminosity function should be
            evaluated. Must be non-negative.
        magnitude_bin_edges: One-dimensional strictly increasing absolute
            magnitude bin edges.
        redshift_bin_edges: One-dimensional strictly increasing non-negative
            redshift bin edges.
        phi_bin_values: Two-dimensional array of non-negative LF values with
            shape ``(redshift_bin_edges.size - 1,
            magnitude_bin_edges.size - 1)``.
        fill_value: Value returned outside the supplied magnitude or redshift
            bin ranges.

    Returns:
        NumPy array containing the binned luminosity function evaluated at
        ``absolute_mag`` and ``redshift``.

    Raises:
        ValueError: If the bin edges are invalid, if ``phi_bin_values`` has the
            wrong shape, if any LF value is negative, if any requested redshift
            is negative, or if ``fill_value`` is not finite and non-negative.
    """
    absolute_mag_arr, redshift_arr = np.broadcast_arrays(
        validate_array(absolute_mag, name="absolute_mag"),
        validate_array(redshift, name="redshift"),
    )
    mag_edges, z_edges, values = validate_2d_binned_grid(
        magnitude_bin_edges,
        redshift_bin_edges,
        phi_bin_values,
        x_edges_name="magnitude_bin_edges",
        y_edges_name="redshift_bin_edges",
        values_name="phi_bin_values",
        allow_negative_y_edges=False,
    )

    if np.any(redshift_arr < 0.0):
        raise ValueError("redshift must be non-negative.")

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if fill_value < 0.0:
        raise ValueError("fill_value must be non-negative.")

    mag_indices = np.searchsorted(mag_edges, absolute_mag_arr, side="right") - 1
    z_indices = np.searchsorted(z_edges, redshift_arr, side="right") - 1

    valid = (
        (mag_indices >= 0)
        & (mag_indices < mag_edges.size - 1)
        & (z_indices >= 0)
        & (z_indices < z_edges.size - 1)
    )

    phi = np.full(absolute_mag_arr.shape, fill_value, dtype=float)
    phi[valid] = values[z_indices[valid], mag_indices[valid]]

    return np.asarray(phi, dtype=float)


def distance_tabulated_lf(
    absolute_mag: FloatInput,
    comoving_distance: FloatInput,
    *,
    magnitude_grid: ParameterValue,
    distance_grid: ParameterValue,
    phi_grid: ParameterValue,
    fill_value: float = 0.0,
    log_phi: bool = False,
) -> FloatArray:
    r"""Return a luminosity function interpolated over magnitude and comoving distance.

    This evaluates a non-parametric luminosity function tabulated on a
    two-dimensional grid,

    .. math::

        \phi(M_i, \chi_k) = \phi_{k i},

    where :math:`M_i` are absolute magnitude grid points and :math:`\chi_k`
    are comoving distance grid points.

    This is analogous to ``redshift_tabulated_lf``, but uses comoving distance
    instead of redshift as the second coordinate. It can be useful for
    projection calculations, tomographic kernels, or simulation analyses where
    the natural radial coordinate is distance rather than redshift.

    Args:
        absolute_mag: Absolute magnitude value(s).
        comoving_distance: Comoving distance value(s). Must be non-negative.
        magnitude_grid: One-dimensional strictly increasing absolute magnitude
            grid.
        distance_grid: One-dimensional strictly increasing non-negative
            comoving distance grid.
        phi_grid: Two-dimensional array of non-negative LF values with shape
            ``(distance_grid.size, magnitude_grid.size)``.
        fill_value: Value returned outside the tabulated magnitude or distance
            range.
        log_phi: If True, interpolate in ``log10(phi)``.

    Returns:
        NumPy array containing the interpolated luminosity function evaluated at
        ``absolute_mag`` and ``comoving_distance``.

    Raises:
        ValueError: If the grids are invalid, if ``phi_grid`` has the wrong
            shape, if LF values are negative, if requested distances are
            negative, or if ``log_phi`` is True and any LF value is not
            positive.
    """
    absolute_mag_arr, distance_arr = np.broadcast_arrays(
        validate_array(absolute_mag, name="absolute_mag"),
        validate_array(comoving_distance, name="comoving_distance"),
    )
    magnitude_grid_arr, distance_grid_arr, phi_grid_arr = validate_2d_tabulated_grid(
        magnitude_grid,
        distance_grid,
        phi_grid,
        x_name="magnitude_grid",
        y_name="distance_grid",
        values_name="phi_grid",
        allow_negative_y=False,
        positive_values=log_phi,
    )

    if np.any(distance_arr < 0.0):
        raise ValueError("comoving_distance must be non-negative.")

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if fill_value < 0.0:
        raise ValueError("fill_value must be non-negative.")

    if log_phi:
        if np.any(phi_grid_arr <= 0.0):
            raise ValueError("phi_grid must be positive when log_phi is True.")

        table = np.log10(phi_grid_arr)
        outside_value = np.log10(fill_value) if fill_value > 0.0 else -np.inf
    else:
        table = phi_grid_arr
        outside_value = fill_value

    out = np.full(absolute_mag_arr.shape, outside_value, dtype=float)

    for index in np.ndindex(absolute_mag_arr.shape):
        mag = absolute_mag_arr[index]
        distance = distance_arr[index]

        outside_mag = mag < magnitude_grid_arr[0] or mag > magnitude_grid_arr[-1]
        outside_distance = (
            distance < distance_grid_arr[0] or distance > distance_grid_arr[-1]
        )

        if outside_mag or outside_distance:
            continue

        phi_at_mag_by_distance = np.array(
            [
                np.interp(mag, magnitude_grid_arr, table[distance_index])
                for distance_index in range(distance_grid_arr.size)
            ],
            dtype=float,
        )
        out[index] = np.interp(
            distance,
            distance_grid_arr,
            phi_at_mag_by_distance,
        )

    if log_phi:
        out = 10.0**out
        out = np.where(np.isfinite(out), out, fill_value)

    return np.asarray(out, dtype=float)


def distance_binned_lf(
    absolute_mag: FloatInput,
    comoving_distance: FloatInput,
    *,
    magnitude_bin_edges: ParameterValue,
    distance_bin_edges: ParameterValue,
    phi_bin_values: ParameterValue,
    fill_value: float = 0.0,
) -> FloatArray:
    r"""Return a piecewise constant binned LF in magnitude and comoving distance.

    This evaluates a luminosity function defined in bins of absolute magnitude
    and comoving distance,

    .. math::

        \phi(M,\chi) = \phi_{k j}

    for

    .. math::

        M_j \le M < M_{j+1},
        \qquad
        \chi_k \le \chi < \chi_{k+1}.

    This is useful when an LF is measured or assigned directly in radial
    distance shells, for example in projection-space calculations or mock
    light-cone analyses.

    Args:
        absolute_mag: Absolute magnitude value(s).
        comoving_distance: Comoving distance value(s). Must be non-negative.
        magnitude_bin_edges: One-dimensional strictly increasing absolute
            magnitude bin edges.
        distance_bin_edges: One-dimensional strictly increasing non-negative
            comoving distance bin edges.
        phi_bin_values: Two-dimensional array of non-negative LF values with
            shape ``(distance_bin_edges.size - 1,
            magnitude_bin_edges.size - 1)``.
        fill_value: Value returned outside the supplied magnitude or distance
            bin ranges.

    Returns:
        NumPy array containing the binned luminosity function evaluated at
        ``absolute_mag`` and ``comoving_distance``.

    Raises:
        ValueError: If bin edges are invalid, if ``phi_bin_values`` has the
            wrong shape, if LF values are negative, or if requested distances
            are negative.
    """
    absolute_mag_arr, distance_arr = np.broadcast_arrays(
        validate_array(absolute_mag, name="absolute_mag"),
        validate_array(comoving_distance, name="comoving_distance"),
    )
    mag_edges, distance_edges, values = validate_2d_binned_grid(
        magnitude_bin_edges,
        distance_bin_edges,
        phi_bin_values,
        x_edges_name="magnitude_bin_edges",
        y_edges_name="distance_bin_edges",
        values_name="phi_bin_values",
        allow_negative_y_edges=False,
    )

    if np.any(distance_arr < 0.0):
        raise ValueError("comoving_distance must be non-negative.")

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    if fill_value < 0.0:
        raise ValueError("fill_value must be non-negative.")

    mag_indices = np.searchsorted(mag_edges, absolute_mag_arr, side="right") - 1
    distance_indices = np.searchsorted(distance_edges, distance_arr, side="right") - 1

    valid = (
        (mag_indices >= 0)
        & (mag_indices < mag_edges.size - 1)
        & (distance_indices >= 0)
        & (distance_indices < distance_edges.size - 1)
    )

    phi = np.full(absolute_mag_arr.shape, fill_value, dtype=float)
    phi[valid] = values[distance_indices[valid], mag_indices[valid]]

    return np.asarray(phi, dtype=float)
