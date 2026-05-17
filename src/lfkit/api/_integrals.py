"""User-facing luminosity function integral API namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfkit.api._expose import expose_lf_function
from lfkit.photometry.lf_integrals import (
    cumulative_number_density,
    integrated_luminosity_density,
    integrated_number_density,
    lf_weighted_integral,
    magnitude_window_number_density,
    mean_luminosity,
    selection_weighted_number_density,
)

if TYPE_CHECKING:
    from lfkit.api.luminosity_function import LuminosityFunction


class LFIntegralsAPI:
    """Grouped API for luminosity function integrals.

    Args:
        lf: Parent luminosity function object.
    """

    def __init__(self, lf: LuminosityFunction) -> None:
        self.lf = lf


_INTEGRAL_METHODS = {
    "number_density": integrated_number_density,
    "weighted": lf_weighted_integral,
    "selection_weighted_number_density": selection_weighted_number_density,
    "luminosity_density": integrated_luminosity_density,
    "mean_luminosity": mean_luminosity,
    "cumulative_number_density": cumulative_number_density,
    "magnitude_window_number_density": magnitude_window_number_density,
}


for method_name, function in _INTEGRAL_METHODS.items():
    setattr(
        LFIntegralsAPI,
        method_name,
        expose_lf_function(
            function,
            lf_arg_position=None,
            lf_arg_name="lf",
        ),
    )
