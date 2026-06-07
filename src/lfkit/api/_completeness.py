"""User-facing catalog-completeness API namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfkit.api._expose import expose_lf_function
from lfkit.luminosity_functions import completeness as lf_completeness

if TYPE_CHECKING:
    from lfkit.api.luminosity_function import LuminosityFunction


class LFCompletenessAPI:
    """Grouped API for catalog-completeness calculations.

    Args:
        lf: Parent luminosity function object.
    """

    def __init__(self, lf: LuminosityFunction) -> None:
        self.lf = lf


for function_name in lf_completeness.__all__:
    function = getattr(lf_completeness, function_name)

    if function_name == "absolute_magnitude_limit":
        setattr(LFCompletenessAPI, function_name, staticmethod(function))
        continue

    setattr(
        LFCompletenessAPI,
        function_name,
        expose_lf_function(
            function,
            lf_arg_position=None,
            lf_arg_name="lf",
        ),
    )