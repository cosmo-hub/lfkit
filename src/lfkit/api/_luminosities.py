from lfkit.photometry.luminosities import (
    luminosity_from_magnitude,
    luminosity_ratio,
    luminosity_ratio_from_magnitudes,
    luminosity_weight_from_magnitude,
    magnitude_difference_from_luminosity_ratio,
)


class LFLuminositiesAPI:
    """Grouped API for luminosity and magnitude conversion helpers."""

    ratio = staticmethod(luminosity_ratio)
    ratio_from_magnitudes = staticmethod(luminosity_ratio_from_magnitudes)
    magnitude_difference_from_ratio = staticmethod(
        magnitude_difference_from_luminosity_ratio
    )
    weight_from_magnitude = staticmethod(luminosity_weight_from_magnitude)
    from_magnitude = staticmethod(luminosity_from_magnitude)
