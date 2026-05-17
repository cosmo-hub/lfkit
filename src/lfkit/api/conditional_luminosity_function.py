"""Public conditional luminosity function constructors."""

from __future__ import annotations

from collections.abc import Mapping

from lfkit.api.luminosity_function import LuminosityFunction
from lfkit.utils.types import ConditionalParameter, ParameterValue


__all__ = ["ConditionalLuminosityFunction"]


def _make_conditional_lf(
    *,
    model: str,
    parameters: Mapping[str, object],
    meta: Mapping[str, object] | None,
) -> LuminosityFunction:
    """Create a LuminosityFunction backed by a conditional LF model."""
    return LuminosityFunction(
        model=model,
        parameters=parameters,
        meta=meta,
    )


class ConditionalLuminosityFunction:
    """Factory namespace for conditional luminosity function models."""

    @staticmethod
    def schechter(
        *,
        phi_star: ConditionalParameter,
        m_star: ConditionalParameter,
        alpha: ConditionalParameter,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a conditional Schechter luminosity function."""
        return _make_conditional_lf(
            model="conditional_schechter",
            parameters={
                "phi_star": phi_star,
                "m_star": m_star,
                "alpha": alpha,
            },
            meta=meta,
        )

    @staticmethod
    def evolving_schechter(
        *,
        phi_model: str = "linear_p",
        phi_kwargs: Mapping[str, ParameterValue] | None = None,
        m_star_model: str = "linear_q",
        m_star_kwargs: Mapping[str, ParameterValue] | None = None,
        alpha_model: str = "constant",
        alpha_kwargs: Mapping[str, ParameterValue] | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a conditional evolving Schechter luminosity function."""
        return _make_conditional_lf(
            model="conditional_evolving_schechter",
            parameters={
                "phi_model": phi_model,
                "phi_kwargs": {} if phi_kwargs is None else dict(phi_kwargs),
                "m_star_model": m_star_model,
                "m_star_kwargs": {} if m_star_kwargs is None else dict(m_star_kwargs),
                "alpha_model": alpha_model,
                "alpha_kwargs": {} if alpha_kwargs is None else dict(alpha_kwargs),
            },
            meta=meta,
        )

    @staticmethod
    def double_schechter(
        *,
        phi_star: ConditionalParameter,
        m_star: ConditionalParameter,
        alpha: float,
        beta: float,
        m_transition: ConditionalParameter,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a conditional double-power-law Schechter luminosity function."""
        return _make_conditional_lf(
            model="conditional_double_schechter",
            parameters={
                "phi_star": phi_star,
                "m_star": m_star,
                "alpha": alpha,
                "beta": beta,
                "m_transition": m_transition,
            },
            meta=meta,
        )

    @staticmethod
    def lognormal(
        *,
        mean_absolute_mag: ConditionalParameter,
        sigma_log_luminosity: ConditionalParameter,
        amplitude: ConditionalParameter = 1.0,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a lognormal conditional luminosity function."""
        return _make_conditional_lf(
            model="lognormal_conditional_lf",
            parameters={
                "mean_absolute_mag": mean_absolute_mag,
                "sigma_log_luminosity": sigma_log_luminosity,
                "amplitude": amplitude,
            },
            meta=meta,
        )

    @staticmethod
    def modified_schechter(
        *,
        phi_star: ConditionalParameter,
        m_star: ConditionalParameter,
        alpha: ConditionalParameter,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a modified Schechter conditional luminosity function."""
        return _make_conditional_lf(
            model="modified_schechter_conditional_lf",
            parameters={
                "phi_star": phi_star,
                "m_star": m_star,
                "alpha": alpha,
            },
            meta=meta,
        )

    @staticmethod
    def two_component(
        *,
        lognormal_mean_absolute_mag: ConditionalParameter,
        lognormal_sigma_log_luminosity: ConditionalParameter,
        modified_phi_star: ConditionalParameter,
        modified_alpha: ConditionalParameter,
        lognormal_amplitude: ConditionalParameter = 1.0,
        modified_m_star: ConditionalParameter | None = None,
        modified_luminosity_fraction: ConditionalParameter = 0.562,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a two-component conditional luminosity function."""
        return _make_conditional_lf(
            model="two_component_conditional_lf",
            parameters={
                "lognormal_mean_absolute_mag": lognormal_mean_absolute_mag,
                "lognormal_sigma_log_luminosity": lognormal_sigma_log_luminosity,
                "lognormal_amplitude": lognormal_amplitude,
                "modified_phi_star": modified_phi_star,
                "modified_alpha": modified_alpha,
                "modified_m_star": modified_m_star,
                "modified_luminosity_fraction": modified_luminosity_fraction,
            },
            meta=meta,
        )
