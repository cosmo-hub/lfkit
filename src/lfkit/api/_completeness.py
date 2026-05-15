"""User-facing catalog-completeness API namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfkit.api._expose import expose_lf_function
from lfkit.photometry.catalog_completeness import (
    absolute_magnitude_limit,
    catalog_completeness_fraction,
    missing_number_density,
    observed_number_density,
    out_of_catalog_fraction,
)

if TYPE_CHECKING:
    from lfkit.api.luminosity_function import LuminosityFunction


class LFCompletenessAPI:
    """Grouped API for catalog-completeness calculations.

    Args:
        lf: Parent luminosity-function object.
    """

    def __init__(self, lf: LuminosityFunction) -> None:
        self.lf = lf


_COMPLETENESS_METHODS = {
    "observed_number_density": observed_number_density,
    "missing_number_density": missing_number_density,
    "catalog_fraction": catalog_completeness_fraction,
    "out_of_catalog_fraction": out_of_catalog_fraction,
}


for method_name, function in _COMPLETENESS_METHODS.items():
    setattr(
        LFCompletenessAPI,
        method_name,
        expose_lf_function(
            function,
            lf_arg_position=None,
            lf_arg_name="lf",
        ),
    )


LFCompletenessAPI.absolute_magnitude_limit = staticmethod(absolute_magnitude_limit)
