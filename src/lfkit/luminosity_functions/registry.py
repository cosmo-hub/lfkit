"""Automatic luminosity-function registries."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from lfkit.luminosity_functions import conditional_models
from lfkit.luminosity_functions._discovery import iter_model_functions


@dataclass(frozen=True)
class LFModel:
    """Description of a registered luminosity-function model."""

    name: str
    function: Callable
    independent_variable: str = "absolute_mag"
    requires_z: bool = False


def discover_lf_models() -> tuple[
    dict[str, LFModel],
    dict[str, LFModel],
    dict[str, Callable],
]:
    """Discover LF models, conditional LF models, and apparent-magnitude evaluators."""
    lf_models: dict[str, LFModel] = {}
    conditional_lf_models: dict[str, LFModel] = {}
    from_m_models: dict[str, Callable] = {}

    _discover_models_package(lf_models, from_m_models)
    _discover_conditional_models(conditional_lf_models)

    return lf_models, conditional_lf_models, from_m_models


def _discover_models_package(
    lf_models: dict[str, LFModel],
    from_m_models: dict[str, Callable],
) -> None:
    """Discover LF models from ``luminosity_functions.models``."""
    for name, obj in iter_model_functions().items():
        _register_lf_model(
            name,
            obj,
            lf_models=lf_models,
            from_m_models=from_m_models,
            name_transform=_public_model_name,
        )


def _discover_conditional_models(
    conditional_lf_models: dict[str, LFModel],
) -> None:
    """Discover conditional LF models."""
    _register_module_lf_models(
        conditional_models,
        lf_models=conditional_lf_models,
        from_m_models=None,
        name_transform=_public_model_name,
    )


def _register_lf_model(
    name: str,
    obj: Callable,
    *,
    lf_models: dict[str, LFModel],
    from_m_models: dict[str, Callable] | None,
    name_transform: Callable[[str], str] | None,
) -> None:
    """Register one public LF function."""
    if not callable(obj):
        return

    sig = inspect.signature(obj)
    params = list(sig.parameters)

    if not params:
        return

    first_arg = params[0]

    if name.endswith("_from_m"):
        if from_m_models is not None:
            from_m_models[name.removesuffix("_from_m")] = obj
        return

    if first_arg not in {"absolute_mag", "magnitude", "luminosity"}:
        return

    public_name = name_transform(name) if name_transform is not None else name

    lf_models[public_name] = LFModel(
        name=public_name,
        function=obj,
        independent_variable=first_arg,
        requires_z=_requires_second_independent_variable(sig),
    )


def _register_module_lf_models(
    module: object,
    *,
    lf_models: dict[str, LFModel],
    from_m_models: dict[str, Callable] | None,
    name_transform: Callable[[str], str] | None,
) -> None:
    """Register public LF functions from one module."""
    for name in getattr(module, "__all__", []):
        _register_lf_model(
            name,
            getattr(module, name),
            lf_models=lf_models,
            from_m_models=from_m_models,
            name_transform=name_transform,
        )


def _public_model_name(name: str) -> str:
    """Return the public model name for LF functions."""
    public_name = name

    if public_name.startswith("conditional_"):
        public_name = public_name.removeprefix("conditional_")

    if public_name.endswith("_lf"):
        public_name = public_name.removesuffix("_lf")

    return public_name


def _requires_second_independent_variable(
    signature: inspect.Signature,
) -> bool:
    """Return whether model needs a second positional independent variable."""
    params = list(signature.parameters.values())

    if len(params) < 2:
        return False

    second = params[1]

    return second.kind in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    } and second.name in {"z", "redshift", "condition", "x"}


LF_MODELS, CONDITIONAL_LF_MODELS, LF_FROM_M_MODELS = discover_lf_models()


def available_lf_models() -> tuple[str, ...]:
    """Return available luminosity-function model names."""
    return tuple(sorted(LF_MODELS))


def available_conditional_lf_models() -> tuple[str, ...]:
    """Return available conditional luminosity-function model names."""
    return tuple(sorted(CONDITIONAL_LF_MODELS))


def available_lf_from_m_models() -> tuple[str, ...]:
    """Return LF models with apparent-magnitude evaluators."""
    return tuple(sorted(LF_FROM_M_MODELS))


def get_lf_model(name: str) -> LFModel:
    """Return a registered luminosity-function model."""
    try:
        return LF_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_lf_models())
        raise ValueError(
            f"Unknown luminosity-function model {name!r}. "
            f"Available models: {available}."
        ) from exc


def get_conditional_lf_model(name: str) -> LFModel:
    """Return a registered conditional luminosity-function model."""
    try:
        return CONDITIONAL_LF_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_conditional_lf_models())
        raise ValueError(
            f"Unknown conditional luminosity-function model {name!r}. "
            f"Available conditional models: {available}."
        ) from exc


def get_lf_from_m_model(name: str) -> Callable:
    """Return an apparent-magnitude LF evaluator."""
    try:
        return LF_FROM_M_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_lf_from_m_models())
        raise ValueError(
            f"phi_from_m is not defined for luminosity-function model {name!r}. "
            f"Available models: {available}."
        ) from exc
