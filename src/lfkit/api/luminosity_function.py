"""Public luminosity function interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from lfkit.api._completeness import LFCompletenessAPI
from lfkit.api._integrals import LFIntegralsAPI
from lfkit.api._luminosities import LFLuminositiesAPI
from lfkit.api._magnitudes import LFMagnitudesAPI
from lfkit.api._redshift_density import LFRedshiftDensityAPI
from lfkit.luminosity_functions.models.parameter_models import (
    available_lf_parameter_models,
    evaluate_lf_parameters,
    register_alpha_model,
    register_m_star_model,
    register_phi_star_model,
)
from lfkit.luminosity_functions.registry import (
    LF_FROM_M_MODELS,
    LF_MODELS,
)
from lfkit.utils.types import (
    Cosmology,
    FloatArray,
    FloatInput,
    ParameterModel,
)

if TYPE_CHECKING:
    from lfkit.api.corrections import Corrections
else:
    Corrections = object


__all__ = ["LuminosityFunction"]


class LuminosityFunction:
    """User-facing wrapper for luminosity function evaluation."""

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

    def phi(
            self,
            absolute_mag: FloatInput,
            z: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the luminosity function in absolute magnitude space."""
        try:
            model_spec = LF_MODELS[self.model]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported luminosity function model '{self.model}'."
            ) from exc

        absolute_mag_arr = np.asarray(absolute_mag, dtype=float)

        if model_spec.requires_z:
            if z is None:
                raise ValueError(
                    f"z is required for luminosity function model '{self.model}'."
                )

            return model_spec.function(
                absolute_mag_arr,
                np.asarray(z, dtype=float),
                **self.parameters_dict,
            )

        return model_spec.function(
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
        """Evaluate the luminosity function from apparent magnitudes."""
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
        """Evaluate evolving Schechter parameters at redshift."""
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
        """Return luminosity function model names available through the API."""
        return sorted(LF_MODELS)

    @staticmethod
    def available_from_m_models() -> list[str]:
        """Return models that support apparent magnitude evaluation."""
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
        """Register a phi-star evolution model."""
        register_phi_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_m_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an M-star evolution model."""
        register_m_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_alpha_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an alpha evolution model."""
        register_alpha_model(name, model, overwrite=overwrite)

    @staticmethod
    def _correction_values(
        corrections: Corrections | None,
        z: FloatInput,
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """Evaluate optional correction values at redshift."""
        if corrections is None:
            return None, None

        return corrections.k(z), corrections.e(z)


def _make_lf_constructor(model_name: str):
    """Create a classmethod constructor for a registered LF model."""

    @classmethod
    def constructor(
        cls,
        *,
        meta: Mapping[str, object] | None = None,
        **parameters: Any,
    ) -> LuminosityFunction:
        return cls(
            model=model_name,
            parameters=_clean_parameters(parameters),
            meta=meta,
        )

    constructor.__name__ = model_name
    constructor.__qualname__ = f"LuminosityFunction.{model_name}"
    constructor.__doc__ = f"Create a ``{model_name}`` luminosity function."

    return constructor


def _clean_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize constructor keyword arguments before storing them."""
    return {
        key: {} if value is None and key.endswith("_kwargs") else value
        for key, value in parameters.items()
    }


for _model_name in LF_MODELS:
    setattr(
        LuminosityFunction,
        _model_name,
        _make_lf_constructor(_model_name),
    )

del _model_name
