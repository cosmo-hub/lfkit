"""Helpers for exposing low-level functions through API namespaces."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def expose_lf_function(
    function: Callable[..., Any],
    *,
    lf_arg_position: int | None = 1,
    lf_arg_name: str | None = None,
) -> Callable[..., Any]:
    """Expose a low-level LF function as a bound API method.

    Args:
        function: Low-level function to expose.
        lf_arg_position: Positional location where the LF callable is inserted.
            Use this when the low-level function expects ``lf_callable`` as a
            positional argument.
        lf_arg_name: Keyword name that receives the LF callable. Use this when
            the low-level function expects a named LF callable argument.

    Returns:
        Bound method that injects ``self.lf._as_callable()``.
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
