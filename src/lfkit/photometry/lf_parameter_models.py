r"""Luminosity-function parameter evolution models for LFKit.

This module provides helper functions for evaluating redshift-dependent
luminosity-function parameters such as ``phi_star(z)``, ``M_star(z)``,
and ``alpha(z)``.

These helpers are used by the main luminosity-function evaluators but do
not evaluate the luminosity function themselves.

Built-in options include constant evolution and simple linearized forms
commonly used in the literature.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

from lfkit.utils.validators import validate_array

if TYPE_CHECKING:
    import pyccl as ccl

    Cosmology = ccl.Cosmology
else:
    Cosmology = object

ParameterModel = Callable[..., NDArray[np.float64]]
FloatArray: TypeAlias = NDArray[np.float64]
ParameterValue: TypeAlias = float | FloatArray


__all__ = [
    "ParameterModel",
    "FloatArray",
    "ParameterValue",
    "phi_star_constant",
    "phi_star_linear_p",
    "m_star_constant",
    "m_star_linear_q",
    "alpha_constant",
    "alpha_linear",
    "PHI_STAR_MODELS",
    "M_STAR_MODELS",
    "ALPHA_MODELS",
    "get_parameter_model",
    "evaluate_lf_parameters",
]


def phi_star_constant(
    z: FloatArray,
    *,
    phi_star: float,
) -> NDArray[np.float64]:
    r"""Return a constant Schechter normalization over redshift.

    This uses

    .. math::

        \phi_\star(z) = \phi_\star.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_star: Constant Schechter normalization.

    Returns:
        NumPy array of Schechter normalization values with the same
        broadcast shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return np.full_like(z_arr, fill_value=phi_star, dtype=float)


def phi_star_linear_p(
    z: FloatArray,
    *,
    phi_0_star: float,
    p: float,
) -> NDArray[np.float64]:
    r"""Return a Schechter normalization with density evolution.

    This uses the common density-evolution form

        \phi_\star(z) = \phi_{0,\star} \, 10^{0.4 p z}.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_0_star: Schechter normalization at redshift zero.
        p: Density-evolution parameter.

    Returns:
        NumPy array of redshift-dependent Schechter normalization values.
    """
    z_arr = validate_array(z, name="z")
    return np.asarray(phi_0_star * 10.0 ** (0.4 * p * z_arr), dtype=float)


def m_star_constant(
    z: FloatArray,
    *,
    m_star: float,
) -> NDArray[np.float64]:
    r"""Return a constant characteristic magnitude over redshift

    This uses

    .. math::

        M_\star(z) = M_\star.

    Args:
        z: Redshift value or array-like of redshift values.
        m_star: Constant characteristic magnitude.

    Returns:
        NumPy array of characteristic magnitudes with the same
        broadcast shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return np.full_like(z_arr, fill_value=m_star, dtype=float)


def m_star_linear_q(
    z: FloatArray,
    *,
    m_0_star: float,
    q: float,
    z_ref: float = 0.1,
) -> NDArray[np.float64]:
    r"""Return a characteristic magnitude with luminosity evolution.

    This uses the common luminosity-evolution form

    .. math::

        M_\star(z) = M_{0,\star} - q (z - z_{\mathrm{ref}}).

    Args:
        z: Redshift value or array-like of redshift values.
        m_0_star: Characteristic magnitude at the reference redshift.
        q: Luminosity-evolution parameter.
        z_ref: Reference redshift.

    Returns:
        NumPy array of redshift-dependent characteristic magnitudes.
    """
    z_arr = validate_array(z, name="z")
    return np.asarray(m_0_star - q * (z_arr - z_ref), dtype=float)


def alpha_constant(
    z: FloatArray,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    r"""Return a constant faint-end slope over redshift.

    This uses

    .. math::

        \alpha(z) = \alpha.

    Args:
        z: Redshift value or array-like of redshift values.
        alpha: Constant faint-end slope.

    Returns:
        NumPy array of faint-end slope values with the same
        broadcast shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return np.full_like(z_arr, fill_value=alpha, dtype=float)


def alpha_linear(
    z: FloatArray,
    *,
    alpha_0: float,
    alpha_1: float,
    z_ref: float = 0.1,
) -> NDArray[np.float64]:
    r"""Return a faint-end slope that varies linearly with redshift.

    This uses

    .. math::

        \alpha(z) = \alpha_0 + \alpha_1 (z - z_{\mathrm{ref}}).

    Args:
        z: Redshift value or array-like of redshift values.
        alpha_0: Faint-end slope at the reference redshift.
        alpha_1: Linear slope with redshift.
        z_ref: Reference redshift.

    Returns:
        NumPy array of redshift-dependent faint-end slope values.
    """
    z_arr = validate_array(z, name="z")
    return np.asarray(alpha_0 + alpha_1 * (z_arr - z_ref), dtype=float)


def get_parameter_model(
    model_name: str,
    registry: Mapping[str, ParameterModel],
    *,
    model_kind: str,
) -> ParameterModel:
    r"""Return a registered luminosity-function parameter model.

    Args:
        model_name: Name of the requested model.
        registry: Mapping from model names to model callables.
        model_kind: Human-readable parameter kind used in error messages,
            for example ``"phi_model"``, ``"m_star_model"``, or
            ``"alpha_model"``.

    Returns:
        Registered parameter-model callable.

    Raises:
        ValueError: If ``model_name`` is not present in ``registry``.
    """
    try:
        return registry[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {model_kind} '{model_name}'. Available models: {list(registry)}."
        ) from exc


def evaluate_lf_parameters(
    z: FloatArray,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Evaluate evolving luminosity-function parameters at redshift ``z``.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_model: Evolution model for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected
            ``phi_star`` evolution model.
        m_star_model: Evolution model for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected
            ``M_star`` evolution model.
        alpha_model: Evolution model for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected
            ``alpha`` evolution model.

    Returns:
        Tuple ``(phi_star, m_star, alpha)`` evaluated at ``z``.

    Raises:
        ValueError: If an unsupported evolution model is requested.
    """
    z_arr = validate_array(z, name="z")

    phi_kwargs_dict: dict[str, ParameterValue] = (
        {} if phi_kwargs is None else dict(phi_kwargs)
    )
    m_star_kwargs_dict: dict[str, ParameterValue] = (
        {} if m_star_kwargs is None else dict(m_star_kwargs)
    )
    alpha_kwargs_dict: dict[str, ParameterValue] = (
        {} if alpha_kwargs is None else dict(alpha_kwargs)
    )

    phi_fn = get_parameter_model(
        phi_model,
        PHI_STAR_MODELS,
        model_kind="phi_model",
    )
    m_star_fn = get_parameter_model(
        m_star_model,
        M_STAR_MODELS,
        model_kind="m_star_model",
    )
    alpha_fn = get_parameter_model(
        alpha_model,
        ALPHA_MODELS,
        model_kind="alpha_model",
    )

    phi_star = phi_fn(z_arr, **phi_kwargs_dict)
    m_star = m_star_fn(z_arr, **m_star_kwargs_dict)
    alpha = alpha_fn(z_arr, **alpha_kwargs_dict)

    return phi_star, m_star, alpha


PHI_STAR_MODELS: dict[str, ParameterModel] = {
    "constant": phi_star_constant,
    "linear_p": phi_star_linear_p,
}

M_STAR_MODELS: dict[str, ParameterModel] = {
    "constant": m_star_constant,
    "linear_q": m_star_linear_q,
}

ALPHA_MODELS: dict[str, ParameterModel] = {
    "constant": alpha_constant,
    "linear": alpha_linear,
}
