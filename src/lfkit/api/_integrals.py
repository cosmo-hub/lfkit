"""User-facing luminosity function integral API namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfkit.api._expose import expose_lf_function
from lfkit.luminosity_functions import integrals as lf_integrals

if TYPE_CHECKING:
    from lfkit.api.luminosity_function import LuminosityFunction


class LFIntegralsAPI:
    """Grouped API for luminosity function integrals.

    Args:
        lf: Parent luminosity function object.
    """

    def __init__(self, lf: LuminosityFunction) -> None:
        self.lf = lf


for function_name in lf_integrals.__all__:
    function = getattr(lf_integrals, function_name)
    method_name = lf_integrals.__api_aliases__.get(function_name, function_name)

    setattr(
        LFIntegralsAPI,
        method_name,
        expose_lf_function(
            function,
            lf_arg_position=None,
            lf_arg_name="lf",
        ),
    )