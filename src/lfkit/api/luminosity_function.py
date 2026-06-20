"""Public luminosity function interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from lfkit.api._namespaces import (
    LFCompletenessAPI,
    LFFractionsAPI,
    LFIntegralsAPI,
    LFLuminositiesAPI,
    LFMagnitudesAPI,
    LFRedshiftDensityAPI,
)
from lfkit.luminosity_functions.parameter_models import (
    available_lf_parameter_models,
    evaluate_lf_parameters,
    register_alpha_model,
    register_m_star_model,
    register_phi_star_model,
)
from lfkit.luminosity_functions.registry import (
    LF_FROM_M_MODELS,
    LF_MODELS,
    get_lf_from_m_model,
    get_lf_model,
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
    """User-facing wrapper for luminosity function evaluation.

    A luminosity function describes the number density of galaxies as a function of
    absolute magnitude. This class stores a registered model and its parameters,
    then exposes a consistent interface for model evaluation, apparent magnitude
    evaluation, integrals, completeness calculations, redshift density
    calculations, and magnitude or luminosity conversions.

    Instances can be created either with the generic constructor or with
    automatically generated model constructors.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: Mapping[str, object],
        meta: Mapping[str, object] | None = None,
    ) -> None:
        """Create a luminosity function object.

        Args:
            model: Name of a registered luminosity function model.
            parameters: Model parameters passed to the registered model function.
            meta: Optional metadata stored on the luminosity function object.
        """
        self.model = str(model)
        self.parameters_dict = dict(parameters)
        self.meta = {} if meta is None else dict(meta)

        self.integrals = LFIntegralsAPI(self)
        self.redshift_density = LFRedshiftDensityAPI(self)
        self.completeness = LFCompletenessAPI(self)
        self.fractions = LFFractionsAPI(self)
        self.luminosities = LFLuminositiesAPI()
        self.magnitudes = LFMagnitudesAPI()

    def phi(
        self,
        absolute_mag: FloatInput,
        z: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the luminosity function in absolute magnitude space.

        Args:
            absolute_mag: Absolute magnitude value or array.
            z: Optional redshift value or array. This is required only for registered
                models whose parameters evolve with redshift.

        Returns:
            Luminosity function evaluated at ``absolute_mag``. For redshift dependent
            models, the result is evaluated at ``absolute_mag`` and ``z``.

        Raises:
            ValueError: If the model is not registered, or if ``z`` is required by the
                selected model but not provided.
        """
        model_spec = get_lf_model(self.model)

        absolute_mag_arr = np.asarray(absolute_mag, dtype=float)

        if hasattr(self, "_custom_phi"):
            return self._custom_phi(absolute_mag_arr, z)

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
        """Evaluate the luminosity function from apparent magnitude values.

        This converts apparent magnitude to absolute magnitude using the supplied
        cosmology and redshift, then evaluates the registered model.

        Args:
            cosmo_obj: Cosmology object used for the distance conversion.
            z: Redshift value or array.
            apparent_mag: Apparent magnitude value or array.
            h: Optional dimensionless Hubble parameter override.
            corrections: Optional correction object with ``k(z)`` and ``e(z)`` methods.

        Returns:
            Luminosity function evaluated at the absolute magnitude values implied by
            ``apparent_mag`` and ``z``.

        Raises:
            ValueError: If the selected model does not provide a ``phi_from_m``
                evaluator.
        """
        function = get_lf_from_m_model(self.model)

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
            z: Redshift value or array.

        Returns:
            Tuple containing ``phi_star(z)``, ``m_star(z)``, and ``alpha(z)``.

        Raises:
            ValueError: If the current model is not ``"evolving_schechter"``.
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

    def with_luminosity_cutoff(
        self,
        *,
        m_star: float | None = None,
        cutoff_power: float = 2.0,
        cutoff_amplitude: float = 1.0,
        meta: Mapping[str, object] | None = None,
    ) -> LuminosityFunction:
        """Return a copy of this luminosity function with a luminosity cutoff.

        Args:
            m_star: Characteristic magnitude used to define the luminosity ratio. If
                not provided, ``self.parameters_dict["m_star"]`` is used when present.
            cutoff_power: Power applied to the luminosity ratio in the exponential
                cutoff.
            cutoff_amplitude: Amplitude multiplying the cutoff term.
            meta: Optional metadata merged into the returned luminosity function.

        Returns:
            New luminosity function object whose ``phi`` method includes the cutoff.

        Raises:
            ValueError: If ``m_star`` is not supplied and the base model does not store
                an ``m_star`` parameter.
        """
        cutoff_m_star = (
            self.parameters_dict["m_star"]
            if m_star is None and "m_star" in self.parameters_dict
            else m_star
        )

        if cutoff_m_star is None:
            raise ValueError(
                "m_star must be supplied when the base luminosity function "
                "does not have an m_star parameter."
            )

        new = LuminosityFunction(
            model=self.model,
            parameters=self.parameters_dict,
            meta={**self.meta, **({} if meta is None else dict(meta))},
        )

        def modified_phi(
            absolute_mag: FloatInput,
            z: FloatInput | None = None,
        ) -> FloatArray:
            absolute_mag_arr = np.asarray(absolute_mag, dtype=float)
            x = self.luminosities.ratio(absolute_mag_arr, cutoff_m_star)
            modifier = np.exp(-cutoff_amplitude * x**cutoff_power)
            return self.phi(absolute_mag_arr, z) * modifier

        new._custom_phi = modified_phi
        return new

    @staticmethod
    def available_models() -> list[str]:
        """Return luminosity function model names available through the API.

        Returns:
            Sorted list of registered luminosity function model names.
        """
        return sorted(LF_MODELS)

    @staticmethod
    def available_from_m_models() -> list[str]:
        """Return models that support apparent magnitude evaluation.

        Returns:
            Sorted list of registered models with ``phi_from_m`` evaluators.
        """
        return sorted(LF_FROM_M_MODELS)

    @staticmethod
    def available_parameter_models() -> dict[str, list[str]]:
        """Return available LF parameter evolution models.

        Returns:
            Dictionary mapping each LF parameter name to the registered
            evolution models available for that parameter.
        """
        return available_lf_parameter_models()

    @staticmethod
    def register_phi_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a ``phi_star`` evolution model.

        Args:
            name: Name used to identify the parameter model.
            model: Callable parameter model.
            overwrite: Whether to replace an existing model with the same name.
        """
        register_phi_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_m_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an ``m_star`` evolution model.

        Args:
            name: Name used to identify the parameter model.
            model: Callable parameter model.
            overwrite: Whether to replace an existing model with the same name.
        """
        register_m_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_alpha_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an ``alpha`` evolution model.

        Args:
            name: Name used to identify the parameter model.
            model: Callable parameter model.
            overwrite: Whether to replace an existing model with the same name.
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
            z: Redshift value or array.

        Returns:
            Tuple ``(k_correction, e_correction)`` evaluated at ``z``. If no correction
            object is supplied, returns ``(None, None)``.
        """
        if corrections is None:
            return None, None

        return corrections.k(z), corrections.e(z)


def _make_lf_constructor(model_name: str):
    """Create a classmethod constructor for a registered luminosity function model.

    Args:
        model_name: Name of the registered luminosity function model.

    Returns:
        Classmethod constructor for ``LuminosityFunction``.
    """

    @classmethod
    def constructor(
        cls,
        *,
        meta: Mapping[str, object] | None = None,
        **parameters: Any,
    ) -> LuminosityFunction:
        """Create a luminosity function from model parameters."""
        return cls(
            model=model_name,
            parameters=_clean_parameters(parameters),
            meta=meta,
        )

    constructor.__name__ = model_name
    constructor.__qualname__ = f"LuminosityFunction.{model_name}"
    constructor.__doc__ = f"""Create a ``{model_name}`` luminosity function.

The keyword arguments are passed to the registered low level model function.
Required model parameters must be supplied by the user. Optional model
parameters use their low level defaults unless explicitly provided.

Args:
    meta: Optional metadata stored on the luminosity function object.
    **parameters: Parameters passed to the registered luminosity function model.

Returns:
    Configured luminosity function.

Examples:
    >>> lf = LuminosityFunction.{model_name}(...)
    >>> phi = lf.phi(absolute_mag=-20.0)
"""

    return constructor


def _clean_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize constructor keyword arguments before storing them.

    ``None`` values for keyword arguments ending in ``"_kwargs"`` are converted to
    empty dictionaries. This keeps optional nested configuration arguments
    convenient for users while storing a predictable parameter dictionary.

    Args:
        parameters: Raw constructor keyword arguments.

    Returns:
        Normalized parameter dictionary.
    """
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
