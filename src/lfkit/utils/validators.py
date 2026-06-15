"""Validation utilities."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput

__all__ = [
    "validate_array",
    "validate_luminosity_distance",
    "validate_magnitude_range",
    "validate_strictly_increasing_1d",
    "validate_tabulated_grid",
    "validate_binned_grid",
    "validate_2d_tabulated_grid",
    "validate_2d_binned_grid",
]


def validate_array(
    x: FloatInput,
    *,
    name: str,
    allow_negative: bool = True,
) -> FloatArray:
    """Return a finite float array."""
    arr = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not allow_negative and np.any(arr < 0):
        raise ValueError(f"{name} contains negative values, which are not allowed.")

    return np.asarray(arr, dtype=np.float64)


def validate_luminosity_distance(
    luminosity_distance_mpc: FloatInput,
) -> FloatArray:
    """Return finite positive luminosity distances in Mpc."""
    distance = validate_array(
        luminosity_distance_mpc,
        name="luminosity_distance_mpc",
        allow_negative=False,
    )

    if np.any(distance <= 0.0):
        raise ValueError("luminosity_distance_mpc must contain positive values.")

    return distance


def validate_magnitude_range(
    *,
    m_bright: float,
    m_faint: float,
) -> None:
    """Validate bright and faint magnitude bounds."""
    if not np.isfinite(m_bright):
        raise ValueError("m_bright must be finite.")

    if not np.isfinite(m_faint):
        raise ValueError("m_faint must be finite.")

    if m_faint <= m_bright:
        raise ValueError("m_faint must be larger than m_bright.")


def validate_strictly_increasing_1d(
    x: FloatInput,
    *,
    name: str,
    min_size: int = 2,
    allow_negative: bool = True,
) -> FloatArray:
    """Return a finite strictly increasing one-dimensional float array."""
    arr = validate_array(x, name=name, allow_negative=allow_negative)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values.")

    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    return arr


def validate_tabulated_grid(
    coordinate_grid: FloatInput,
    values: FloatInput,
    *,
    coordinate_name: str,
    values_name: str,
    allow_negative_coordinate: bool = True,
    positive_values: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Return validated one-dimensional tabulated grid coordinates and values."""
    coordinate_arr = validate_strictly_increasing_1d(
        coordinate_grid,
        name=coordinate_name,
        allow_negative=allow_negative_coordinate,
    )
    values_arr = validate_array(values, name=values_name)

    if values_arr.ndim != 1:
        raise ValueError(f"{values_name} must be one-dimensional.")

    if coordinate_arr.size != values_arr.size:
        raise ValueError(
            f"{coordinate_name} and {values_name} must have the same length."
        )

    if positive_values:
        if np.any(values_arr <= 0.0):
            raise ValueError(f"{values_name} must be positive.")
    elif np.any(values_arr < 0.0):
        raise ValueError(f"{values_name} must be non-negative.")

    return coordinate_arr, values_arr


def validate_binned_grid(
    bin_edges: FloatInput,
    bin_values: FloatInput,
    *,
    edges_name: str,
    values_name: str,
    allow_negative_edges: bool = True,
    positive_values: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Return validated one-dimensional bin edges and bin values."""
    edges_arr = validate_strictly_increasing_1d(
        bin_edges,
        name=edges_name,
        allow_negative=allow_negative_edges,
    )
    values_arr = validate_array(bin_values, name=values_name)

    if values_arr.ndim != 1:
        raise ValueError(f"{values_name} must be one-dimensional.")

    if edges_arr.size != values_arr.size + 1:
        raise ValueError(f"{edges_name} must have one more value than {values_name}.")

    if positive_values:
        if np.any(values_arr <= 0.0):
            raise ValueError(f"{values_name} must be positive.")
    elif np.any(values_arr < 0.0):
        raise ValueError(f"{values_name} must be non-negative.")

    return edges_arr, values_arr


def validate_2d_tabulated_grid(
    x_grid: FloatInput,
    y_grid: FloatInput,
    values: FloatInput,
    *,
    x_name: str,
    y_name: str,
    values_name: str,
    allow_negative_x: bool = True,
    allow_negative_y: bool = True,
    positive_values: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return validated two-dimensional tabulated coordinates and values."""
    x_arr = validate_strictly_increasing_1d(
        x_grid,
        name=x_name,
        allow_negative=allow_negative_x,
    )
    y_arr = validate_strictly_increasing_1d(
        y_grid,
        name=y_name,
        allow_negative=allow_negative_y,
    )
    values_arr = validate_array(values, name=values_name)

    if values_arr.ndim != 2:
        raise ValueError(f"{values_name} must be two-dimensional.")

    if values_arr.shape != (y_arr.size, x_arr.size):
        raise ValueError(
            f"{values_name} must have shape ({y_name}.size, {x_name}.size)."
        )

    if positive_values:
        if np.any(values_arr <= 0.0):
            raise ValueError(f"{values_name} must be positive.")
    elif np.any(values_arr < 0.0):
        raise ValueError(f"{values_name} must be non-negative.")

    return x_arr, y_arr, values_arr


def validate_2d_binned_grid(
    x_bin_edges: FloatInput,
    y_bin_edges: FloatInput,
    values: FloatInput,
    *,
    x_edges_name: str,
    y_edges_name: str,
    values_name: str,
    allow_negative_x_edges: bool = True,
    allow_negative_y_edges: bool = True,
    positive_values: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return validated two-dimensional bin edges and bin values."""
    x_edges_arr = validate_strictly_increasing_1d(
        x_bin_edges,
        name=x_edges_name,
        allow_negative=allow_negative_x_edges,
    )
    y_edges_arr = validate_strictly_increasing_1d(
        y_bin_edges,
        name=y_edges_name,
        allow_negative=allow_negative_y_edges,
    )
    values_arr = validate_array(values, name=values_name)

    if values_arr.ndim != 2:
        raise ValueError(f"{values_name} must be two-dimensional.")

    if values_arr.shape != (y_edges_arr.size - 1, x_edges_arr.size - 1):
        raise ValueError(
            f"{values_name} must have shape "
            f"({y_edges_name}.size - 1, {x_edges_name}.size - 1)."
        )

    if positive_values:
        if np.any(values_arr <= 0.0):
            raise ValueError(f"{values_name} must be positive.")
    elif np.any(values_arr < 0.0):
        raise ValueError(f"{values_name} must be non-negative.")

    return x_edges_arr, y_edges_arr, values_arr
