"""User-facing magnitude conversion API namespace."""

from __future__ import annotations

from lfkit.photometry.magnitudes import (
    absolute_magnitude,
    absolute_magnitude_from_luminosity_distance,
    apparent_magnitude,
    apparent_magnitude_from_luminosity_distance,
    total_magnitude_correction,
)


class LFMagnitudesAPI:
    """Grouped API for apparent- and absolute-magnitude conversions."""

    correction = staticmethod(total_magnitude_correction)

    absolute = staticmethod(absolute_magnitude)
    absolute_from_luminosity_distance = staticmethod(
        absolute_magnitude_from_luminosity_distance
    )

    apparent = staticmethod(apparent_magnitude)
    apparent_from_luminosity_distance = staticmethod(
        apparent_magnitude_from_luminosity_distance
    )
