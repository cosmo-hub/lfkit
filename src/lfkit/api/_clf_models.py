"""User-facing conditional luminosity-function model API namespace."""

from __future__ import annotations

from lfkit.photometry.conditional_lf_models import (
    conditional_schechter,
    conditional_double_schechter,
    conditional_evolving_schechter,
    lognormal_conditional_lf,
    modified_schechter_conditional_lf,
    two_component_conditional_lf,
)


class LFConditionalModelsAPI:
    """Grouped API for evaluating conditional luminosity-function models."""

    schechter = staticmethod(conditional_schechter)
    evolving_schechter = staticmethod(conditional_evolving_schechter)
    double_schechter = staticmethod(conditional_double_schechter)
    lognormal = staticmethod(lognormal_conditional_lf)
    modified_schechter = staticmethod(modified_schechter_conditional_lf)
    two_component = staticmethod(two_component_conditional_lf)
