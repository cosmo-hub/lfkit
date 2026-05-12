"""Conditional luminosity-function model utilities.

This module provides conditional wrappers around existing LFKit luminosity
function models.

A conditional luminosity function has the form ``Phi(M | x)``, where ``M`` is
absolute magnitude and ``x`` is an external conditioning variable. The
conditioning variable is intentionally generic. It may represent redshift,
halo mass, environment, galaxy type, richness, stellar mass, or any other
quantity.

This module does not implement HOD or halo-model machinery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from lfkit.photometry.lf_parameter_models import evaluate_lf_parameters
from lfkit.photometry.luminosities import (
    luminosity_ratio,
    magnitude_difference_from_luminosity_ratio,
)
from lfkit.photometry.luminosity_function import schechter, schechter_double
from lfkit.utils.types import (
    ConditionalParameter,
    FloatArray,
    FloatInput,
    ParameterValue,
)
from lfkit.utils.validators import validate_array


__all__ = [
    "conditional_schechter",
    "conditional_schechter_evolving",
    "conditional_schechter_double",
    "central_lognormal_conditional_lf",
    "satellite_modified_schechter_conditional_lf",
    "central_satellite_conditional_lf",
]


def conditional_schechter(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    phi_star: ConditionalParameter,
    m_star: ConditionalParameter,
    alpha: ConditionalParameter,
) -> FloatArray:
    """Evaluate a conditional Schechter luminosity function.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        phi_star: Schechter normalization. May be scalar, array-like, or
            callable of ``condition``.
        m_star: Characteristic absolute magnitude. May be scalar, array-like,
            or callable of ``condition``.
        alpha: Faint-end slope. May be scalar, array-like, or callable of
            ``condition``.

    Returns:
        Conditional Schechter luminosity-function values.
    """
    condition_arr = validate_array(condition, name="condition")

    return schechter(
        absolute_mag,
        phi_star=_evaluate_conditional_parameter(
            phi_star,
            condition_arr,
            name="phi_star",
        ),
        m_star=_evaluate_conditional_parameter(
            m_star,
            condition_arr,
            name="m_star",
        ),
        alpha=_evaluate_conditional_parameter(
            alpha,
            condition_arr,
            name="alpha",
        ),
    )


def conditional_schechter_evolving(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
) -> FloatArray:
    """Evaluate a conditional Schechter LF using LFKit parameter models.

    This is the conditional-LF analogue of ``schechter_evolving``. The
    conditioning variable is passed to LFKit's registered parameter models.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        phi_model: Evolution/condition model for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected ``phi_star`` model.
        m_star_model: Evolution/condition model for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected ``M_star`` model.
        alpha_model: Evolution/condition model for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected ``alpha`` model.

    Returns:
        Conditional Schechter luminosity-function values.

    Raises:
        ValueError: If an unsupported parameter model is requested.
    """
    condition_arr = validate_array(condition, name="condition")

    phi_star, m_star, alpha = evaluate_lf_parameters(
        condition_arr,
        phi_model=phi_model,
        phi_kwargs=phi_kwargs,
        m_star_model=m_star_model,
        m_star_kwargs=m_star_kwargs,
        alpha_model=alpha_model,
        alpha_kwargs=alpha_kwargs,
    )

    return schechter(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )


def conditional_schechter_double(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    phi_star: ConditionalParameter,
    m_star: ConditionalParameter,
    alpha: float,
    beta: float,
    m_transition: ConditionalParameter,
) -> FloatArray:
    """Evaluate a conditional double-power-law Schechter luminosity function.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        phi_star: Overall normalization. May be scalar, array-like, or
            callable of ``condition``.
        m_star: Characteristic absolute magnitude. May be scalar, array-like,
            or callable of ``condition``.
        alpha: Bright/intermediate faint-end slope parameter.
        beta: Additional faint-end slope modifier.
        m_transition: Transition magnitude. May be scalar, array-like, or
            callable of ``condition``.

    Returns:
        Conditional double-power-law Schechter luminosity-function values.
    """
    condition_arr = validate_array(condition, name="condition")

    return schechter_double(
        absolute_mag,
        phi_star=_evaluate_conditional_parameter(
            phi_star,
            condition_arr,
            name="phi_star",
        ),
        m_star=_evaluate_conditional_parameter(
            m_star,
            condition_arr,
            name="m_star",
        ),
        alpha=alpha,
        beta=beta,
        m_transition=_evaluate_conditional_parameter(
            m_transition,
            condition_arr,
            name="m_transition",
        ),
    )


def central_lognormal_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    mean_absolute_mag: ConditionalParameter,
    sigma_log_luminosity: ConditionalParameter,
    amplitude: ConditionalParameter = 1.0,
) -> FloatArray:
    """Evaluate a central-galaxy lognormal conditional LF in magnitudes.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        mean_absolute_mag: Mean central-galaxy absolute magnitude. May be
            scalar, array-like, or callable of ``condition``.
        sigma_log_luminosity: Scatter in ``log10(L)`` at fixed condition. May
            be scalar, array-like, or callable of ``condition``.
        amplitude: Non-negative amplitude of the central component. May be
            scalar, array-like, or callable of ``condition``.

    Returns:
        Central-galaxy conditional luminosity-function values.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    condition_arr = validate_array(condition, name="condition")

    mean_absolute_mag_arr = _evaluate_conditional_parameter(
        mean_absolute_mag,
        condition_arr,
        name="mean_absolute_mag",
    )
    sigma_log_luminosity_arr = _evaluate_conditional_parameter(
        sigma_log_luminosity,
        condition_arr,
        name="sigma_log_luminosity",
    )
    amplitude_arr = _evaluate_conditional_parameter(
        amplitude,
        condition_arr,
        name="amplitude",
    )

    if np.any(sigma_log_luminosity_arr <= 0.0):
        raise ValueError("sigma_log_luminosity must be positive.")

    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude must be non-negative.")

    delta_log_luminosity = -0.4 * (absolute_mag_arr - mean_absolute_mag_arr)

    phi = (
        amplitude_arr
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity_arr)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity_arr) ** 2.0)
    )

    return _validate_lf_output(phi, name="central_lognormal_conditional_lf")


def satellite_modified_schechter_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    phi_star: ConditionalParameter,
    m_star: ConditionalParameter,
    alpha: ConditionalParameter,
) -> FloatArray:
    """Evaluate a satellite modified-Schechter conditional LF.

    This is the Cacciato-style satellite form with ``exp[-(L/L_star)^2]``
    instead of the standard Schechter ``exp[-L/L_star]``.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        phi_star: Satellite normalization. May be scalar, array-like, or
            callable of ``condition``.
        m_star: Satellite characteristic absolute magnitude. May be scalar,
            array-like, or callable of ``condition``.
        alpha: Satellite faint-end slope. May be scalar, array-like, or
            callable of ``condition``.

    Returns:
        Satellite conditional luminosity-function values.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    condition_arr = validate_array(condition, name="condition")

    phi_star_arr = _evaluate_conditional_parameter(
        phi_star,
        condition_arr,
        name="phi_star",
    )
    m_star_arr = _evaluate_conditional_parameter(
        m_star,
        condition_arr,
        name="m_star",
    )
    alpha_arr = _evaluate_conditional_parameter(
        alpha,
        condition_arr,
        name="alpha",
    )

    if np.any(phi_star_arr < 0.0):
        raise ValueError("phi_star must be non-negative.")

    x = luminosity_ratio(absolute_mag_arr, m_star_arr)

    phi = 0.4 * np.log(10.0) * phi_star_arr * x ** (alpha_arr + 1.0) * np.exp(-(x**2.0))

    return _validate_lf_output(
        phi,
        name="satellite_modified_schechter_conditional_lf",
    )


def central_satellite_conditional_lf(
    absolute_mag: FloatInput,
    condition: FloatInput,
    *,
    central_mean_absolute_mag: ConditionalParameter,
    central_sigma_log_luminosity: ConditionalParameter,
    satellite_phi_star: ConditionalParameter,
    satellite_alpha: ConditionalParameter,
    central_amplitude: ConditionalParameter = 1.0,
    satellite_m_star: ConditionalParameter | None = None,
    satellite_luminosity_fraction: ConditionalParameter = 0.562,
) -> FloatArray:
    """Evaluate the sum of central and satellite conditional LF components.

    Args:
        absolute_mag: Absolute magnitude value(s).
        condition: Values of the conditioning variable.
        central_mean_absolute_mag: Mean central-galaxy absolute magnitude. May
            be scalar, array-like, or callable of ``condition``.
        central_sigma_log_luminosity: Scatter in ``log10(L)`` for the central
            component. May be scalar, array-like, or callable of ``condition``.
        satellite_phi_star: Satellite normalization. May be scalar, array-like,
            or callable of ``condition``.
        satellite_alpha: Satellite faint-end slope. May be scalar, array-like,
            or callable of ``condition``.
        central_amplitude: Non-negative amplitude of the central component. May
            be scalar, array-like, or callable of ``condition``.
        satellite_m_star: Satellite characteristic absolute magnitude. If
            omitted, it is derived from ``central_mean_absolute_mag`` and
            ``satellite_luminosity_fraction``.
        satellite_luminosity_fraction: Ratio ``L_sat_star / L_c`` used only
            when ``satellite_m_star`` is omitted.

    Returns:
        Total central-plus-satellite conditional luminosity-function values.
    """
    condition_arr = validate_array(condition, name="condition")

    central_mean_absolute_mag_arr = _evaluate_conditional_parameter(
        central_mean_absolute_mag,
        condition_arr,
        name="central_mean_absolute_mag",
    )

    central_phi = central_lognormal_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition_arr,
        mean_absolute_mag=central_mean_absolute_mag_arr,
        sigma_log_luminosity=central_sigma_log_luminosity,
        amplitude=central_amplitude,
    )

    if satellite_m_star is None:
        satellite_luminosity_fraction_arr = _evaluate_conditional_parameter(
            satellite_luminosity_fraction,
            condition_arr,
            name="satellite_luminosity_fraction",
        )

        if np.any(satellite_luminosity_fraction_arr <= 0.0):
            raise ValueError("satellite_luminosity_fraction must be positive.")

        satellite_m_star_arr = central_mean_absolute_mag_arr + (
            magnitude_difference_from_luminosity_ratio(
                satellite_luminosity_fraction_arr
            )
        )
    else:
        satellite_m_star_arr = _evaluate_conditional_parameter(
            satellite_m_star,
            condition_arr,
            name="satellite_m_star",
        )

    satellite_phi = satellite_modified_schechter_conditional_lf(
        absolute_mag=absolute_mag,
        condition=condition_arr,
        phi_star=satellite_phi_star,
        m_star=satellite_m_star_arr,
        alpha=satellite_alpha,
    )

    return _validate_lf_output(
        central_phi + satellite_phi,
        name="central_satellite_conditional_lf",
    )


def _evaluate_conditional_parameter(
    parameter: ConditionalParameter,
    condition: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Evaluate a scalar, array-like, or callable conditional parameter."""
    if callable(parameter):
        values = parameter(condition)
    else:
        values = cast(ParameterValue, parameter)

    return validate_array(values, name=name)


def _validate_lf_output(
    phi: FloatInput,
    *,
    name: str,
) -> FloatArray:
    """Validate luminosity-function model output."""
    phi_arr = validate_array(phi, name=name)

    if np.any(phi_arr < 0.0):
        raise ValueError(f"{name} returned negative values, which are not allowed.")

    return np.asarray(phi_arr, dtype=np.float64)
