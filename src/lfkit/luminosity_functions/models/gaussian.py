"""Gaussian-like luminosity function models."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput, ParameterValue
from lfkit.utils.validators import validate_array


__all__ = [
    "gaussian_lf",
    "lognormal_lf",
]


def gaussian_lf(
    absolute_mag: FloatInput,
    *,
    mean_absolute_mag: ParameterValue,
    sigma_absolute_mag: ParameterValue,
    amplitude: ParameterValue = 1.0,
) -> FloatArray:
    r"""Return a Gaussian luminosity function in magnitude space.

    This computes

    .. math::

        \phi(M) =
        \frac{A}{\sqrt{2\pi}\,\sigma_M}
        \exp\left[
            -\frac{1}{2}
            \left(\frac{M - \mu_M}{\sigma_M}\right)^2
        \right],

    where :math:`A` is ``amplitude``, :math:`\mu_M` is
    ``mean_absolute_mag``, and :math:`\sigma_M` is
    ``sigma_absolute_mag``.

    This model describes a symmetric population in absolute magnitude space.
    It is useful for localized luminosity function components whose abundance
    falls off approximately normally around a preferred magnitude.

    Args:
        absolute_mag: Absolute magnitude value(s).
        mean_absolute_mag: Mean absolute magnitude of the Gaussian component.
        sigma_absolute_mag: Standard deviation in absolute magnitude.
        amplitude: Non-negative integrated amplitude of the Gaussian component.

    Returns:
        NumPy array containing the Gaussian luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If ``sigma_absolute_mag`` is not positive or if
            ``amplitude`` is negative.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    mean_absolute_mag_arr = validate_array(
        mean_absolute_mag,
        name="mean_absolute_mag",
    )
    sigma_absolute_mag_arr = validate_array(
        sigma_absolute_mag,
        name="sigma_absolute_mag",
    )
    amplitude_arr = validate_array(amplitude, name="amplitude")

    if np.any(sigma_absolute_mag_arr <= 0.0):
        raise ValueError("sigma_absolute_mag must be positive.")

    if np.any(amplitude_arr < 0.0):
        raise ValueError("amplitude must be non-negative.")

    phi = (
        amplitude_arr
        / (np.sqrt(2.0 * np.pi) * sigma_absolute_mag_arr)
        * np.exp(
            -0.5
            * ((absolute_mag_arr - mean_absolute_mag_arr) / sigma_absolute_mag_arr)
            ** 2.0
        )
    )

    return np.asarray(phi, dtype=float)


def lognormal_lf(
    absolute_mag: FloatInput,
    *,
    mean_absolute_mag: ParameterValue,
    sigma_log_luminosity: ParameterValue,
    amplitude: ParameterValue = 1.0,
) -> FloatArray:
    r"""Return a lognormal luminosity function in magnitude space.

    This computes a Gaussian profile in log-luminosity,

    .. math::

        \phi(M) =
        \frac{0.4 A}{\sqrt{2\pi}\,\sigma_{\log L}}
        \exp\left[
            -\frac{1}{2}
            \left(\frac{\Delta \log_{10} L}{\sigma_{\log L}}\right)^2
        \right],

    where

    .. math::

        \Delta \log_{10} L =
        -0.4\,(M - M_0).

    Here :math:`A` is ``amplitude``, :math:`M_0` is
    ``mean_absolute_mag``, and :math:`\sigma_{\log L}` is
    ``sigma_log_luminosity``.

    This model is symmetric in logarithmic luminosity rather than in
    magnitude. It is useful for compact luminosity components where the natural
    scatter is closer to multiplicative scatter in luminosity than additive
    scatter in magnitude.

    Args:
        absolute_mag: Absolute magnitude value(s).
        mean_absolute_mag: Absolute magnitude corresponding to the lognormal
            peak.
        sigma_log_luminosity: Standard deviation in base-10 log luminosity.
        amplitude: Non-negative integrated amplitude of the lognormal
            component.

    Returns:
        NumPy array containing the lognormal luminosity function evaluated at
        ``absolute_mag``.

    Raises:
        ValueError: If ``sigma_log_luminosity`` is not positive or if
            ``amplitude`` is negative.
    """
    absolute_mag_arr = validate_array(absolute_mag, name="absolute_mag")
    mean_absolute_mag_arr = validate_array(
        mean_absolute_mag,
        name="mean_absolute_mag",
    )
    sigma_log_luminosity_arr = validate_array(
        sigma_log_luminosity,
        name="sigma_log_luminosity",
    )
    amplitude_arr = validate_array(amplitude, name="amplitude")

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

    return np.asarray(phi, dtype=float)
