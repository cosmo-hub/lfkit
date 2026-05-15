"""User-facing luminosity and magnitude conversion API namespace."""

from __future__ import annotations

from lfkit.photometry.luminosities import (
    luminosity_from_magnitude,
    luminosity_ratio,
    luminosity_ratio_from_magnitudes,
    luminosity_weight_from_magnitude,
    magnitude_difference_from_luminosity_ratio,
    sample_schechter_luminosity,
    schechter_cumulative_number_density_luminosity,
    schechter_luminosity_density,
    schechter_mean_luminosity,
    schechter_selection_function,
)


class LFLuminositiesAPI:
    """Grouped API for luminosity, magnitude, and Schechter-luminosity helpers."""

    ratio = staticmethod(luminosity_ratio)
    ratio_from_magnitudes = staticmethod(luminosity_ratio_from_magnitudes)
    magnitude_difference_from_ratio = staticmethod(
        magnitude_difference_from_luminosity_ratio
    )
    weight_from_magnitude = staticmethod(luminosity_weight_from_magnitude)
    from_magnitude = staticmethod(luminosity_from_magnitude)

    schechter_cumulative_number_density = staticmethod(
        schechter_cumulative_number_density_luminosity
    )
    schechter_luminosity_density = staticmethod(schechter_luminosity_density)
    schechter_mean_luminosity = staticmethod(schechter_mean_luminosity)
    sample_schechter = staticmethod(sample_schechter_luminosity)
    schechter_selection = staticmethod(schechter_selection_function)
