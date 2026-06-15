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

    A conditional luminosity function evaluates ``Phi(M | x_1, x_2, ...)``,
    where ``M`` is absolute magnitude and the ``x_i`` are external conditioning
    variables such as redshift, halo mass, environment, richness, stellar mass,
    or other model-specific quantities.

    Instances can be created either with the generic constructor or with
    automatically generated model constructors.
    """

    def phi(
        self,
        absolute_mag: FloatInput,
        *conditions: FloatInput,
    ) -> FloatArray:
        """Evaluate the conditional luminosity function.

        Args:
            absolute_mag: Absolute magnitude value or array.
            *conditions: One or more conditioning variable values or arrays. The
                meaning of each variable depends on the selected conditional
                luminosity function model.

        Returns:
            Conditional luminosity function evaluated at ``absolute_mag`` and
            the supplied conditioning variables.

        Raises:
            ValueError: If no conditioning variables are provided, or if the
                model is not registered as a conditional luminosity function.
        """
        model_spec = get_conditional_lf_model(self.model)

        if not conditions:
            raise ValueError(
                f"At least one conditioning variable is required for conditional "
                f"luminosity function model '{self.model}'."
            )

        condition_arrays = tuple(
            np.asarray(condition_value, dtype=float) for condition_value in conditions
        )

        return model_spec.function(
            np.asarray(absolute_mag, dtype=float),
            *condition_arrays,
            **self.parameters_dict,
        )

    @staticmethod
    def available_models() -> list[str]:
        """Return registered conditional luminosity function model names.

        Returns:
            Sorted list of registered conditional luminosity function model names.
        """
        return sorted(CONDITIONAL_LF_MODELS)


def _make_conditional_constructor(
    *,
    model_name: str,
    function: Any,
):
    """Create a classmethod constructor from a registered conditional model.

    Args:
        model_name: Name of the registered conditional luminosity function model.
        function: Registered low level conditional luminosity function callable.

    Returns:
        Classmethod constructor for ``ConditionalLuminosityFunction``.
    """
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
        
        The keyword arguments are inferred from the registered low level model
        function. Required model parameters must be supplied by the user. Optional
        model parameters use their low level defaults unless explicitly provided.
        
        Args:
            meta: Optional metadata stored on the luminosity function object.
            **parameters: Parameters passed to the registered conditional luminosity
                function model.
        
        Returns:
            Configured conditional luminosity function.
        
        Examples:
            >>> clf = ConditionalLuminosityFunction.{model_name}(...)
            >>> phi = clf.phi(-20.0, 0.5)
            >>> phi = clf.phi(-20.0, halo_mass, redshift)
        """

    return constructor


def _parameters_from_signature(
    *,
    signature: inspect.Signature,
    provided: Mapping[str, Any],
) -> dict[str, Any]:
    """Build stored parameters from a function signature and user values.

    Independent variables such as ``absolute_mag`` and conditioning variables are
    not stored as model parameters. They are supplied later when calling
    :meth:`ConditionalLuminosityFunction.phi`.

    Args:
        signature: Signature of the registered low level conditional luminosity
            function model.
        provided: User-supplied constructor keyword arguments.

    Returns:
        Dictionary of model parameters stored on the API object.

    Raises:
        TypeError: If the user provides a keyword that is not accepted by the
            registered low level model function.
    """
    payload: dict[str, Any] = {}

    independent_names = {
        "absolute_mag",
        "condition",
        "conditions",
        "z",
        "redshift",
        "x",
        "halo_mass",
        "environment",
        "galaxy_type",
    }

    for name, parameter in signature.parameters.items():
        if name in independent_names:
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
