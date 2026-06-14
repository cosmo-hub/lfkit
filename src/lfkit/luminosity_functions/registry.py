"""Automatic luminosity function registries."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from lfkit.luminosity_functions import conditional_models
from lfkit.luminosity_functions._discovery import iter_model_functions


@dataclass(frozen=True)
class LFModel:
    """Description of a registered luminosity function model.

    Attributes:
        name: Public model name.
        function: Registered model callable.
        independent_variable: Name of the first independent variable.
        requires_z: Whether the model requires a second independent variable such
            as redshift or condition.
    """

    name: str
    function: Callable
    independent_variable: str = "absolute_mag"
    requires_z: bool = False


def discover_lf_models() -> tuple[
    dict[str, LFModel],
    dict[str, LFModel],
    dict[str, Callable],
]:
    """Discover luminosity function registries.

    Returns:
        Tuple containing registered luminosity function models, conditional
        luminosity function models, and apparent magnitude evaluators.
    """
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
    """Discover luminosity function models from ``luminosity_functions.models``.

    Args:
        lf_models: Registry updated with discovered luminosity function models.
        from_m_models: Registry updated with apparent magnitude evaluators.
    """
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
    """Discover conditional luminosity function models.

    Args:
        conditional_lf_models: Registry updated with discovered conditional
            luminosity function models.
    """
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
    """Register one public luminosity function callable.

    Args:
        name: Candidate function name.
        obj: Candidate callable.
        lf_models: Registry updated with luminosity function models.
        from_m_models: Optional registry updated with apparent magnitude
            evaluators.
        name_transform: Optional callable used to convert function names to public
            model names.
    """
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
    """Register public luminosity function callables from one module.

    Args:
        module: Module object with an ``__all__`` declaration.
        lf_models: Registry updated with luminosity function models.
        from_m_models: Optional registry updated with apparent magnitude
            evaluators.
        name_transform: Optional callable used to convert function names to public
            model names.
    """
    for name in getattr(module, "__all__", []):
        _register_lf_model(
            name,
            getattr(module, name),
            lf_models=lf_models,
            from_m_models=from_m_models,
            name_transform=name_transform,
        )


def _public_model_name(name: str) -> str:
    """Return the public model name for a luminosity function callable.

    Args:
        name: Function name to convert.

    Returns:
        Public model name with ``conditional_`` and ``_lf`` affixes removed.
    """
    public_name = name

    if public_name.startswith("conditional_"):
        public_name = public_name.removeprefix("conditional_")

    if public_name.endswith("_lf"):
        public_name = public_name.removesuffix("_lf")

    return public_name


def _requires_second_independent_variable(
    signature: inspect.Signature,
) -> bool:
    """Return whether a model requires a second independent variable.

    Args:
        signature: Function signature to inspect.

    Returns:
        ``True`` if the second positional argument is a supported independent
        variable name, otherwise ``False``.
    """
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
    """Return available luminosity function model names.

    Returns:
        Sorted tuple of registered luminosity function model names.
    """
    return tuple(sorted(LF_MODELS))


def available_conditional_lf_models() -> tuple[str, ...]:
    """Return available conditional luminosity function model names.

    Returns:
        Sorted tuple of registered conditional luminosity function model names.
    """
    return tuple(sorted(CONDITIONAL_LF_MODELS))


def available_lf_from_m_models() -> tuple[str, ...]:
    """Return luminosity function models with apparent magnitude evaluators.

    Returns:
        Sorted tuple of model names that provide apparent magnitude evaluators.
    """
    return tuple(sorted(LF_FROM_M_MODELS))


def get_lf_model(name: str) -> LFModel:
    """Return a registered luminosity function model.

    Args:
        name: Name of the registered luminosity function model.

    Returns:
        Registered luminosity function model description.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        return LF_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_lf_models())
        raise ValueError(
            f"Unknown luminosity function model {name!r}. "
            f"Available models: {available}."
        ) from exc


def get_conditional_lf_model(name: str) -> LFModel:
    """Return a registered conditional luminosity function model.

    Args:
        name: Name of the registered conditional luminosity function model.

    Returns:
        Registered conditional luminosity function model description.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        return CONDITIONAL_LF_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_conditional_lf_models())
        raise ValueError(
            f"Unknown conditional luminosity function model {name!r}. "
            f"Available conditional models: {available}."
        ) from exc


def get_lf_from_m_model(name: str) -> Callable:
    """Return an apparent magnitude luminosity function evaluator.

    Args:
        name: Name of the luminosity function model.

    Returns:
        Registered apparent magnitude evaluator.

    Raises:
        ValueError: If ``name`` does not have a registered apparent magnitude
            evaluator.
    """
    try:
        return LF_FROM_M_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_lf_from_m_models())
        raise ValueError(
            f"phi_from_m is not defined for luminosity function model {name!r}. "
            f"Available models: {available}."
        ) from exc
