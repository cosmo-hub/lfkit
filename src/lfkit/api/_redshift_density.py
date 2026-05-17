"""User-facing LF redshift-density API namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfkit.api._expose import expose_lf_function
from lfkit.photometry.lf_redshift_density import (
    lf_integrated_number_density,
    lf_weighted_redshift_density,
)

if TYPE_CHECKING:
    from lfkit.api.luminosity_function import LuminosityFunction


class LFRedshiftDensityAPI:
    """Grouped API for LF-weighted redshift-density calculations.

    Args:
        lf: Parent luminosity function object.
    """

    def __init__(self, lf: LuminosityFunction) -> None:
        self.lf = lf


_REDSHIFT_DENSITY_METHODS = {
    "integrated_number_density": lf_integrated_number_density,
    "weighted": lf_weighted_redshift_density,
}


for method_name, function in _REDSHIFT_DENSITY_METHODS.items():
    setattr(
        LFRedshiftDensityAPI,
        method_name,
        expose_lf_function(function, lf_arg_position=1),
    )
