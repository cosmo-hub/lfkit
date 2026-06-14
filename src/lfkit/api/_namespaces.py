"""User-facing luminosity function API namespaces."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from lfkit.luminosity_functions import completeness as lf_completeness
from lfkit.luminosity_functions import integrals as lf_integrals
from lfkit.luminosity_functions import redshift_density as lf_redshift_density
from lfkit.photometry import luminosities as photo_luminosities
from lfkit.photometry import magnitudes as photo_magnitudes


class LFIntegralsAPI:
    """Grouped API for luminosity function integrals.

    Args:
        lf: Luminosity function object whose callable form is used by bound
            integral methods.
    """

    def __init__(self, lf) -> None:
        self.lf = lf


class LFCompletenessAPI:
    """Grouped API for catalog completeness calculations.

    Args:
        lf: Luminosity function object whose callable form is used by bound
            completeness methods.
    """

    def __init__(self, lf) -> None:
        self.lf = lf


class LFRedshiftDensityAPI:
    """Grouped API for luminosity function weighted redshift density calculations.

    Args:
        lf: Luminosity function object whose callable form is used by bound
            redshift density methods.
    """

    def __init__(self, lf) -> None:
        self.lf = lf


class LFMagnitudesAPI:
    """Grouped API for apparent magnitude and absolute magnitude conversions."""


class LFLuminositiesAPI:
    """Grouped API for luminosity and magnitude conversion helpers."""


_API_NAMESPACES = {
    LFIntegralsAPI: {
        "module": lf_integrals,
        "bound_to_lf": True,
        "lf_arg_name": "lf",
    },
    LFCompletenessAPI: {
        "module": lf_completeness,
        "bound_to_lf": True,
        "lf_arg_name": "lf",
        "static_functions": {"absolute_magnitude_limit"},
    },
    LFRedshiftDensityAPI: {
        "module": lf_redshift_density,
        "bound_to_lf": True,
        "lf_arg_position": 1,
    },
    LFMagnitudesAPI: {
        "module": photo_magnitudes,
        "bound_to_lf": False,
    },
    LFLuminositiesAPI: {
        "module": photo_luminosities,
        "bound_to_lf": False,
    },
}


def expose_lf_function(
    function: Callable[..., Any],
    *,
    lf_arg_position: int | None = None,
    lf_arg_name: str | None = None,
) -> Callable[..., Any]:
    """Expose a low level luminosity function helper as a bound API method.

    Args:
        function: Low level function to expose as a method.
        lf_arg_position: Positional index where the luminosity function callable
            should be inserted. If ``None``, no positional luminosity function
            argument is inserted.
        lf_arg_name: Keyword name used to pass the luminosity function callable.
            If provided, this takes priority over ``lf_arg_position``.

    Returns:
        Bound method that injects ``self.lf._as_callable()`` before calling
        ``function``.
    """

    @wraps(function)
    def method(self, *args, **kwargs):
        lf_callable = self.lf._as_callable()

        if lf_arg_name is not None:
            kwargs[lf_arg_name] = lf_callable
            return function(*args, **kwargs)

        if lf_arg_position is None:
            return function(*args, **kwargs)

        args_list = list(args)
        args_list.insert(lf_arg_position, lf_callable)
        return function(*args_list, **kwargs)

    return method


def _public_functions(module: object) -> dict[str, Callable[..., Any]]:
    """Return callable public functions declared by a module.

    Args:
        module: Module object with an optional ``__all__`` declaration.

    Returns:
        Dictionary mapping public function names to callable objects.
    """
    return {
        name: getattr(module, name)
        for name in getattr(module, "__all__", [])
        if callable(getattr(module, name))
    }


def _method_name(module: object, function_name: str) -> str:
    """Return the API method name for a low level function.

    Args:
        module: Module object that may define ``__api_aliases__``.
        function_name: Name of the low level function.

    Returns:
        Public API method name, using ``__api_aliases__`` when available.
    """
    aliases = getattr(module, "__api_aliases__", {})
    return aliases.get(function_name, function_name)


def _attach_api_methods() -> None:
    """Attach low level functions to their API namespace classes."""
    for api_cls, spec in _API_NAMESPACES.items():
        module = spec["module"]
        bound_to_lf = spec.get("bound_to_lf", False)
        static_functions = spec.get("static_functions", set())

        for function_name, function in _public_functions(module).items():
            method_name = _method_name(module, function_name)

            if not bound_to_lf or function_name in static_functions:
                setattr(api_cls, method_name, staticmethod(function))
                continue

            setattr(
                api_cls,
                method_name,
                expose_lf_function(
                    function,
                    lf_arg_position=spec.get("lf_arg_position"),
                    lf_arg_name=spec.get("lf_arg_name"),
                ),
            )


_attach_api_methods()
