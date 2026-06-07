"""Public conditional luminosity function constructors."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np

from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.luminosity_functions.registry import CONDITIONAL_LF_MODELS
from lfkit.utils.types import FloatArray, FloatInput

__all__ = ["ConditionalLuminosityFunction"]


class ConditionalLuminosityFunction(LuminosityFunction):
    """User-facing wrapper for conditional luminosity function models."""

    def phi(
        self,
        absolute_mag: FloatInput,
        condition: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the conditional luminosity function."""
        try:
            model_spec = CONDITIONAL_LF_MODELS[self.model]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported conditional luminosity function model "
                f"'{self.model}'."
            ) from exc

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
    constructor.__doc__ = f"Create a ``{model_name}`` conditional luminosity function."

    return constructor


def _parameters_from_signature(
    *,
    signature: inspect.Signature,
    provided: Mapping[str, Any],
) -> dict[str, Any]:
    """Build stored parameters using function defaults plus user values."""
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
