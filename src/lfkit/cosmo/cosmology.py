"""Cosmology utilities for LFKit.

This module provides lightweight wrappers around PyCCL cosmology
objects and background calculations.

It standardizes how cosmology instances are created and how
lookback time is computed, ensuring consistent behavior across LFKit.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

from typing import Any
import pyccl as ccl
import numpy as np


__all__ = (
    "cosmo_object",
    "lookback_time_gyr",
)


def cosmo_object(
    *,
    instance: ccl.Cosmology | None = None,
    **params: Any,
) -> ccl.Cosmology:
    """Return a PyCCL cosmology object.

    This function provides a standardized way to obtain a
    ``pyccl.Cosmology`` instance within LFKit.

    Behavior:
        1. If ``instance`` is provided, it is returned unchanged.
        2. Else if cosmological parameters are provided, a new
           ``ccl.Cosmology`` is constructed using those parameters.
        3. Else, a default ``ccl.CosmologyVanillaLCDM`` is returned.

    Args:
        instance: Pre-built ``ccl.Cosmology`` object.
        **params: Cosmological parameters passed directly to
            ``ccl.Cosmology(**params)``.

    Returns:
        A ``ccl.Cosmology`` object.

    Raises:
        ValueError: If both ``instance`` and cosmological parameters
            are provided.
    """
    if instance is not None:
        if params:
            raise ValueError("Pass instance OR parameters, not both.")
        return instance

    if params:
        return ccl.Cosmology(**params)

    return ccl.CosmologyVanillaLCDM()


def lookback_time_gyr(cosmo_obj: ccl.Cosmology, z):
    """Compute lookback time in gigayears.

    This function evaluates the cosmological lookback time
    using PyCCL background calculations.

    Args:
        cosmo_obj: A ``ccl.Cosmology`` instance.
        z: Redshift value or array of redshift values.

    Returns:
        NumPy array of lookback time values in gigayears.
    """
    z = np.asarray(z, float)
    a = 1.0 / (1.0 + z)
    return np.asarray(ccl.background.lookback_time(cosmo_obj, a), float)
