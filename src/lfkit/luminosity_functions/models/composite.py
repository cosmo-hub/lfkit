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
    r"""Return the sum of multiple luminosity function components.

    This evaluates an additive mixture model,

    .. math::

        \phi_{\mathrm{tot}}(M) =
        \sum_i \phi_i(M),

    where each :math:`\phi_i(M)` is an independently defined luminosity
    function component evaluated on the same absolute magnitude grid.

    This helper is useful for building composite populations where different
    physical or phenomenological components contribute to the same observed
    luminosity function, for example a narrow central population plus a broader
    satellite-like component.

    Args:
        absolute_mag: Absolute magnitude value(s).
        *components: Callable luminosity function components. Each component
            must accept ``absolute_mag`` and return values broadcastable to the
            same shape.

    Returns:
        NumPy array containing the summed luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If no luminosity function components are provided.
    """
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
    r"""Return a two-component luminosity function.

    This model combines a lognormal component with a cutoff-modified Schechter
    component,

    .. math::

        \phi_{\mathrm{tot}}(M) =
        \phi_{\mathrm{lognormal}}(M)
        + \phi_{\mathrm{mod}}(M).

    The lognormal term describes a localized component peaked around
    ``lognormal_mean_absolute_mag``. The modified Schechter term provides a
    broader luminosity function component with a Schechter-like faint end and a
    suppressed bright end.

    If ``modified_m_star`` is not supplied, it is inferred from the lognormal
    peak magnitude using

    .. math::

        M_{\star,\mathrm{mod}} =
        M_{\mathrm{lognormal}} + \Delta M(f_L),

    where :math:`f_L` is ``modified_luminosity_fraction`` and
    :math:`\Delta M(f_L)` is the magnitude offset corresponding to that
    luminosity ratio.

    This is mainly a phenomenological composite model: the two terms should be
    interpreted as flexible components of the total luminosity function rather
    than as a unique physical decomposition.

    Args:
        absolute_mag: Absolute magnitude value(s).
        lognormal_mean_absolute_mag: Mean absolute magnitude of the lognormal
            component.
        lognormal_sigma_log_luminosity: Width of the lognormal component in
            log luminosity.
        modified_phi_star: Normalization of the modified Schechter component.
        modified_alpha: Faint-end slope of the modified Schechter component.
        lognormal_amplitude: Amplitude of the lognormal component.
        modified_m_star: Characteristic magnitude of the modified Schechter
            component. If not provided, it is inferred from
            ``modified_luminosity_fraction``.
        modified_luminosity_fraction: Luminosity ratio used to infer
            ``modified_m_star`` when it is not supplied.

    Returns:
        NumPy array containing the combined luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If ``modified_luminosity_fraction`` is not positive.
    """
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
