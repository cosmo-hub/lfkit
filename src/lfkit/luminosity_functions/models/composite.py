"""Composite luminosity function models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lfkit.luminosity_functions.models.gaussian import lognormal_lf
from lfkit.luminosity_functions.models.schechter import schechter
from lfkit.luminosity_functions.models.modifiers import apply_luminosity_cutoff
from lfkit.photometry.luminosities import magnitude_difference_from_luminosity_ratio
from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "two_component_lf",
]


def additive_lf(
    absolute_mag: FloatInput,
    *components: Callable[[FloatInput], FloatArray],
) -> FloatArray:
    """Return the sum of multiple luminosity function components."""
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")

    if len(components) == 0:
        raise ValueError("At least one luminosity function component is required.")

    phi = np.zeros_like(absolute_mag_arr, dtype=float)

    for component in components:
        phi = phi + component(absolute_mag_arr)

    return np.asarray(phi, dtype=float)


def two_component_lf(
    absolute_mag: FloatInput,
    *,
    lognormal_mean_absolute_mag: ParameterValue,
    lognormal_sigma_log_luminosity: ParameterValue,
    modified_phi_star: ParameterValue,
    modified_alpha: ParameterValue,
    lognormal_amplitude: ParameterValue = 1.0,
    modified_m_star: ParameterValue | None = None,
    modified_luminosity_fraction: ParameterValue = 0.562,
) -> FloatArray:
    """Return the sum of lognormal and modified Schechter components."""
    lognormal_mean_absolute_mag_arr = validate_array(
        lognormal_mean_absolute_mag,
        name="lognormal_mean_absolute_mag",
    )

    lognormal_phi = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=lognormal_mean_absolute_mag_arr,
        sigma_log_luminosity=lognormal_sigma_log_luminosity,
        amplitude=lognormal_amplitude,
    )

    if modified_m_star is None:
        modified_luminosity_fraction_arr = validate_array(
            modified_luminosity_fraction,
            name="modified_luminosity_fraction",
        )

        if np.any(modified_luminosity_fraction_arr <= 0.0):
            raise ValueError("modified_luminosity_fraction must be positive.")

        modified_m_star_arr = lognormal_mean_absolute_mag_arr + (
            magnitude_difference_from_luminosity_ratio(
                modified_luminosity_fraction_arr,
            )
        )
    else:
        modified_m_star_arr = validate_array(
            modified_m_star,
            name="modified_m_star",
        )

    modified_phi = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=schechter,
        phi_star=modified_phi_star,
        m_star=modified_m_star_arr,
        alpha=modified_alpha,
    )

    return additive_lf(
        absolute_mag,
        lambda mag: np.asarray(lognormal_phi, dtype=float),
        lambda mag: np.asarray(modified_phi, dtype=float),
    )
