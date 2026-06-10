"""Public conditional luminosity function constructors."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np

from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.luminosity_functions.registry import (
    CONDITIONAL_LF_MODELS,
    get_conditional_lf_model,
)
from lfkit.utils.types import FloatArray, FloatInput

__all__ = ["ConditionalLuminosityFunction"]


class ConditionalLuminosityFunction(LuminosityFunction):
    """User-facing wrapper for conditional luminosity function models.

    A conditional luminosity function evaluates ``Phi(M | x)``, where
    ``M`` is absolute magnitude and ``x`` is an external conditioning
    variable such as redshift, halo mass, or another model-specific quantity.

    Instances can be created either with the generic constructor or with
    automatically generated model constructors.

    Examples:
        >>> clf = ConditionalLuminosityFunction(
        ...     model="schechter_models.rst",
        ...     parameters={"phi_star": 1e-3, "m_star": -20.5, "alpha": -1.1},
        ... )
        >>> phi = clf.phi(absolute_mag=-20.0, condition=0.5)

        >>> ConditionalLuminosityFunction.available_models()
    """

    def phi(
        self,
        absolute_mag: FloatInput,
        condition: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the conditional luminosity function.

        Args:
            absolute_mag: Absolute magnitude value or array.
            condition: Conditioning variable value or array. The meaning of
                this variable depends on the selected conditional LF model.

        Returns:
            Conditional luminosity function evaluated at ``absolute_mag`` and
            ``condition``.

        Raises:
            ValueError: If ``condition`` is not provided or the model is not
                registered as a conditional luminosity function.
        """
        model_spec = get_conditional_lf_model(self.model)

        if condition is None:
            raise ValueError(
                f"condition is required for conditional luminosity function "
                f"model '{self.model}'."
            )

        return model_spec.function(
            np.asarray(absolute_mag, dtype=float),
            np.asarray(condition, dtype=float),
            **self.parameters_dict,
        )

    @staticmethod
    def available_models() -> list[str]:
        """Return conditional luminosity function model names."""
        return sorted(CONDITIONAL_LF_MODELS)


def _make_conditional_constructor(
    *,
    model_name: str,
    function: Any,
):
    """Create a classmethod constructor from a registered conditional LF model."""
    signature = inspect.signature(function)

    @classmethod
    def constructor(
        cls,
        *,
        meta: Mapping[str, object] | None = None,
        **parameters: Any,
    ) -> ConditionalLuminosityFunction:
        """Create a conditional luminosity function from model parameters."""
        return cls(
            model=model_name,
            parameters=_parameters_from_signature(
                signature=signature,
                provided=parameters,
            ),
            meta=meta,
        )

    constructor.__name__ = model_name
    constructor.__qualname__ = f"ConditionalLuminosityFunction.{model_name}"
    constructor.__doc__ = f"""Create a ``{model_name}`` conditional luminosity function.

The keyword arguments are inferred from the registered low-level model
function. Required model parameters must be supplied by the user. Optional
model parameters use their low-level defaults unless explicitly provided.

Args:
    meta: Optional metadata stored on the luminosity function object.
    **parameters: Parameters passed to the registered conditional LF model.

Returns:
    ConditionalLuminosityFunction: Configured conditional luminosity function.

Examples:
    >>> clf = ConditionalLuminosityFunction.{model_name}(...)
    >>> phi = clf.phi(absolute_mag=-20.0, condition=0.5)
"""

    return constructor


def _parameters_from_signature(
    *,
    signature: inspect.Signature,
    provided: Mapping[str, Any],
) -> dict[str, Any]:
    """Build stored parameters using function defaults plus user values.

    Independent variables such as ``absolute_mag`` and ``condition`` are not
    stored as model parameters. They are supplied later when calling
    :meth:`ConditionalLuminosityFunction.phi`.

    Args:
        signature: Signature of the registered low-level conditional LF model.
        provided: User-supplied constructor keyword arguments.

    Returns:
        Dictionary of model parameters to store on the API object.

    Raises:
        TypeError: If the user provides a keyword that is not accepted by the
            registered low-level model function.
    """
    payload: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if name in {"absolute_mag", "condition", "z", "redshift", "x"}:
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if name in provided:
            payload[name] = provided[name]
        elif parameter.default is not inspect.Parameter.empty:
            payload[name] = parameter.default

    extra = set(provided) - set(signature.parameters)
    if extra:
        raise TypeError(f"Unexpected parameter(s): {sorted(extra)}")

    return payload


for _model_name, _model_spec in CONDITIONAL_LF_MODELS.items():
    setattr(
        ConditionalLuminosityFunction,
        _model_name,
        _make_conditional_constructor(
            model_name=_model_name,
            function=_model_spec.function,
        ),
    )

del _model_name, _model_spec
