r"""Public luminosity-function interface.

This module provides the user-facing :class:`LuminosityFunction` API for
evaluating luminosity functions in absolute- or apparent-magnitude space.

The class stores luminosity-function model state and exposes grouped API
namespaces for related calculations. Low-level numerical and photometric
work remains in the function-based ``lfkit.photometry`` modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from lfkit.api._lf_param_models import LF_FROM_M_MODELS, LF_MODELS
from lfkit.api._completeness import LFCompletenessAPI
from lfkit.api._integrals import LFIntegralsAPI
from lfkit.api._luminosities import LFLuminositiesAPI
from lfkit.api._magnitudes import LFMagnitudesAPI
from lfkit.photometry.lf_parameter_models import (
    available_lf_parameter_models,
    evaluate_lf_parameters,
    register_alpha_model,
    register_m_star_model,
    register_phi_star_model,
)
from lfkit.api._redshift_density import LFRedshiftDensityAPI
from lfkit.utils.types import (
    Cosmology,
    FloatArray,
    FloatInput,
    ParameterModel,
    ParameterValue,
)

if TYPE_CHECKING:
    from lfkit.api.corrections import Corrections
else:
    Corrections = object


__all__ = ["LuminosityFunction"]


class LuminosityFunction:
    """User-facing wrapper for luminosity-function evaluation.

    Args:
        model: Name of the luminosity-function model.
        parameters: Model parameters passed to the underlying LF function.
        meta: Optional metadata describing the LF source or calibration.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: Mapping[str, object],
        meta: Mapping[str, object] | None = None,
    ) -> None:
        self.model = str(model)
        self.parameters_dict = dict(parameters)
        self.meta = {} if meta is None else dict(meta)

        self.integrals = LFIntegralsAPI(self)
        self.redshift_density = LFRedshiftDensityAPI(self)
        self.completeness = LFCompletenessAPI(self)
        self.luminosities = LFLuminositiesAPI()
        self.magnitudes = LFMagnitudesAPI()

    @classmethod
    def schechter(
        cls,
        *,
        phi_star: ParameterValue,
        m_star: ParameterValue,
        alpha: ParameterValue,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a standard Schechter luminosity function.

        Args:
            phi_star: Normalization of the luminosity function.
            m_star: Characteristic absolute magnitude.
            alpha: Faint-end slope.
            meta: Optional metadata describing the LF source or calibration.

        Returns:
            Luminosity-function API object using the standard Schechter model.
        """
        return cls(
            model="schechter",
            parameters={
                "phi_star": phi_star,
                "m_star": m_star,
                "alpha": alpha,
            },
            meta=meta,
        )

    @classmethod
    def evolving_schechter(
        cls,
        *,
        phi_model: str = "linear_p",
        phi_kwargs: Mapping[str, ParameterValue] | None = None,
        m_star_model: str = "linear_q",
        m_star_kwargs: Mapping[str, ParameterValue] | None = None,
        alpha_model: str = "constant",
        alpha_kwargs: Mapping[str, ParameterValue] | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a redshift-evolving Schechter luminosity function.

        Args:
            phi_model: Parameter model used for the normalization evolution.
            phi_kwargs: Keyword arguments for the normalization model.
            m_star_model: Parameter model used for characteristic-magnitude evolution.
            m_star_kwargs: Keyword arguments for the characteristic-magnitude model.
            alpha_model: Parameter model used for faint-end-slope evolution.
            alpha_kwargs: Keyword arguments for the faint-end-slope model.
            meta: Optional metadata describing the LF source or calibration.

        Returns:
            Luminosity-function API object using an evolving Schechter model.
        """
        return cls(
            model="evolving_schechter",
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

    @classmethod
    def double_schechter(
        cls,
        *,
        phi_star: ParameterValue,
        m_star: ParameterValue,
        alpha: float,
        beta: float,
        m_transition: ParameterValue,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Create a double-power-law Schechter luminosity function.

        Args:
            phi_star: Normalization of the luminosity function.
            m_star: Characteristic absolute magnitude.
            alpha: Bright-end or main Schechter slope.
            beta: Additional slope controlling the second power-law component.
            m_transition: Transition magnitude for the second component.
            meta: Optional metadata describing the LF source or calibration.

        Returns:
            Luminosity-function API object using the double Schechter model.
        """
        return cls(
            model="double_schechter",
            parameters={
                "phi_star": phi_star,
                "m_star": m_star,
                "alpha": alpha,
                "beta": beta,
                "m_transition": m_transition,
            },
            meta=meta,
        )

    def phi(
        self,
        absolute_mag: FloatInput,
        z: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the luminosity function in absolute-magnitude space.

        Args:
            absolute_mag: Absolute magnitude values where the LF is evaluated.
            z: Redshift or conditional-coordinate values. Required for evolving
                and conditional models.

        Returns:
            Luminosity-function values evaluated at the input magnitudes.
        """
        try:
            model_spec = LF_MODELS[self.model]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported luminosity function model '{self.model}'."
            ) from exc

        absolute_mag_arr = np.asarray(absolute_mag, dtype=float)

        if model_spec["requires_z"]:
            if z is None:
                raise ValueError(
                    f"z is required for luminosity function model '{self.model}'."
                )

            return model_spec["function"](
                absolute_mag_arr,
                np.asarray(z, dtype=float),
                **self.parameters_dict,
            )

        return model_spec["function"](
            absolute_mag_arr,
            **self.parameters_dict,
        )

    def phi_from_m(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        apparent_mag: FloatInput,
        *,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Evaluate the luminosity function from apparent magnitudes.

        Apparent magnitudes are converted to absolute magnitudes using the
        supplied cosmology, optional reduced Hubble parameter, and optional
        k- and e-correction model.

        Args:
            cosmo_obj: Cosmology object used for distance-modulus conversion.
            z: Redshift values.
            apparent_mag: Apparent magnitude values.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Luminosity-function values evaluated from apparent magnitudes.
        """
        try:
            function = LF_FROM_M_MODELS[self.model]
        except KeyError as exc:
            raise ValueError(
                f"phi_from_m is not defined for luminosity function model "
                f"'{self.model}'."
            ) from exc

        k_corr, e_corr = self._correction_values(corrections, z)

        return function(
            cosmo_obj,
            np.asarray(z, dtype=float),
            np.asarray(apparent_mag, dtype=float),
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
            **self.parameters_dict,
        )

    def parameters(
        self,
        z: FloatInput,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate evolving Schechter parameters at redshift.

        Args:
            z: Redshift values where the evolving LF parameters are evaluated.

        Returns:
            Tuple containing ``phi_star(z)``, ``m_star(z)``, and ``alpha(z)``.
        """
        if self.model != "evolving_schechter":
            raise ValueError("parameters(z) is only defined for evolving_schechter.")

        return evaluate_lf_parameters(
            np.asarray(z, dtype=float),
            **self.parameters_dict,
        )

    def _as_callable(self):
        """Return this object as an ``lf(M, z)`` callable."""
        return lambda absolute_mag, z: self.phi(absolute_mag, z)

    @staticmethod
    def available_models() -> list[str]:
        """Return luminosity-function model names available through the API."""
        return sorted(LF_MODELS)

    @staticmethod
    def available_from_m_models() -> list[str]:
        """Return models that support apparent-magnitude evaluation."""
        return sorted(LF_FROM_M_MODELS)

    @staticmethod
    def available_parameter_models() -> dict[str, list[str]]:
        """Return available LF parameter evolution models."""
        return available_lf_parameter_models()

    @staticmethod
    def register_phi_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a phi-star evolution model.

        Args:
            name: Name used to identify the model.
            model: Callable evaluating ``phi_star(z)``.
            overwrite: If True, replace an existing model with the same name.
        """
        register_phi_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_m_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an M-star evolution model.

        Args:
            name: Name used to identify the model.
            model: Callable evaluating ``M_star(z)``.
            overwrite: If True, replace an existing model with the same name.
        """
        register_m_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_alpha_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an alpha evolution model.

        Args:
            name: Name used to identify the model.
            model: Callable evaluating ``alpha(z)``.
            overwrite: If True, replace an existing model with the same name.
        """
        register_alpha_model(name, model, overwrite=overwrite)

    @staticmethod
    def _correction_values(
        corrections: Corrections | None,
        z: FloatInput,
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """Evaluate optional correction values at redshift.

        Args:
            corrections: Optional correction object with ``k(z)`` and ``e(z)`` methods.
            z: Redshift values where corrections are evaluated.

        Returns:
            Tuple of k-correction and e-correction arrays, or ``None`` values
            when no correction object is supplied.
        """
        if corrections is None:
            return None, None

        return corrections.k(z), corrections.e(z)
