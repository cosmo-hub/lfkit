r"""Public luminosity-function interface.

This module provides the user-facing :class:`LuminosityFunction` API for
evaluating luminosity functions in absolute- or apparent-magnitude space.

The class wraps the lower-level LFKit photometry functions behind a small,
stable interface. Users can construct standard, evolving, or double Schechter
models, evaluate :math:`\phi(M, z)`, evaluate from apparent magnitudes, and
compute integrated, observed, and missing number densities for
magnitude-limited catalog selections.

File reading is intentionally not handled here. Catalog-derived LF parameters,
magnitude limits, or correction models should be loaded elsewhere and passed
into this API as scalars, arrays, or correction objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from lfkit.photometry.luminosity_function import (
    schechter,
    schechter_evolving,
    schechter_double,
    schechter_from_m,
    schechter_evolving_from_m,
    schechter_double_from_m,
)
from lfkit.photometry.magnitudes import (
    absolute_magnitude,
    apparent_magnitude,
)
from lfkit.photometry.lf_parameter_models import (
    available_lf_parameter_models,
    evaluate_lf_parameters,
    register_alpha_model,
    register_m_star_model,
    register_phi_star_model,
)
from lfkit.photometry.catalog_completeness import (
    absolute_magnitude_limit,
    catalog_completeness_fraction,
    out_of_catalog_fraction,
    observed_number_density,
    missing_number_density,
    integrated_number_density,
)
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
    """User-facing wrapper for luminosity-function evaluation."""

    def __init__(
        self,
        *,
        model: str,
        parameters: Mapping[str, object],
        meta: Mapping[str, object] | None = None,
    ) -> None:
        """Store a luminosity-function model and its parameters.

        Args:
            model: Name of the luminosity-function model.
            parameters: Model parameters passed to the underlying LF function.
            meta: Optional metadata describing the LF source or calibration.
        """
        self.model = str(model)
        self.parameters_dict = dict(parameters)
        self.meta = {} if meta is None else dict(meta)

    def absolute_magnitude(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        apparent_mag: FloatInput,
        *,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Convert apparent magnitude to absolute magnitude.

        Args:
            cosmo_obj: Cosmology object used for distance-modulus conversion.
            z: Redshift values.
            apparent_mag: Apparent magnitude values.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Absolute magnitudes using the LFKit convention
            ``M = m - mu - K + E``.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return absolute_magnitude(
            cosmo_obj,
            z,
            apparent_mag,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    def apparent_magnitude(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        absolute_mag: FloatInput,
        *,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Convert absolute magnitude to apparent magnitude.

        Args:
            cosmo_obj: Cosmology object used for distance-modulus conversion.
            z: Redshift values.
            absolute_mag: Absolute magnitude values.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Apparent magnitudes using the LFKit convention
            ``m = M + mu + K - E``.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return apparent_magnitude(
            cosmo_obj,
            z,
            absolute_mag,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    @classmethod
    def schechter(
        cls,
        *,
        phi_star: ParameterValue,
        m_star: ParameterValue,
        alpha: ParameterValue,
    ) -> "LuminosityFunction":
        """Create a standard Schechter luminosity function.

        Args:
            phi_star: Normalization of the luminosity function.
            m_star: Characteristic absolute magnitude.
            alpha: Faint-end slope.

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
    ) -> "LuminosityFunction":
        """Create a redshift-evolving Schechter luminosity function.

        Args:
            phi_model: Parameter model used for the normalization evolution.
            phi_kwargs: Keyword arguments for the normalization model.
            m_star_model: Parameter model used for characteristic-magnitude evolution.
            m_star_kwargs: Keyword arguments for the characteristic-magnitude model.
            alpha_model: Parameter model used for faint-end-slope evolution.
            alpha_kwargs: Keyword arguments for the faint-end-slope model.

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
    ) -> "LuminosityFunction":
        """Create a double-power-law Schechter luminosity function.

        Args:
            phi_star: Normalization of the luminosity function.
            m_star: Characteristic absolute magnitude.
            alpha: Bright-end or main Schechter slope.
            beta: Additional slope controlling the second power-law component.
            m_transition: Transition magnitude for the second component.

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
        )

    def phi(
        self,
        absolute_mag: FloatInput,
        z: FloatInput | None = None,
    ) -> FloatArray:
        """Evaluate the luminosity function in absolute-magnitude space.

        Args:
            absolute_mag: Absolute magnitude values where the LF is evaluated.
            z: Redshift values. Required for evolving Schechter models.

        Returns:
            Luminosity-function values evaluated at the input magnitudes.
        """
        if self.model == "schechter":
            return schechter(
                np.asarray(absolute_mag, dtype=float),
                **self.parameters_dict,
            )

        if self.model == "evolving_schechter":
            if z is None:
                raise ValueError("z is required for an evolving luminosity function.")

            return schechter_evolving(
                np.asarray(absolute_mag, dtype=float),
                np.asarray(z, dtype=float),
                **self.parameters_dict,
            )

        if self.model == "double_schechter":
            return schechter_double(
                np.asarray(absolute_mag, dtype=float),
                **self.parameters_dict,
            )

        raise ValueError(f"Unsupported luminosity-function model '{self.model}'.")

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
        k_corr, e_corr = self._correction_values(corrections, z)

        if self.model == "schechter":
            return schechter_from_m(
                cosmo_obj,
                np.asarray(z, dtype=float),
                np.asarray(apparent_mag, dtype=float),
                h=h,
                k_correction=k_corr,
                e_correction=e_corr,
                **self.parameters_dict,
            )

        if self.model == "evolving_schechter":
            return schechter_evolving_from_m(
                cosmo_obj,
                np.asarray(z, dtype=float),
                np.asarray(apparent_mag, dtype=float),
                h=h,
                k_correction=k_corr,
                e_correction=e_corr,
                **self.parameters_dict,
            )

        if self.model == "double_schechter":
            return schechter_double_from_m(
                cosmo_obj,
                np.asarray(z, dtype=float),
                np.asarray(apparent_mag, dtype=float),
                h=h,
                k_correction=k_corr,
                e_correction=e_corr,
                **self.parameters_dict,
            )

        raise ValueError(f"Unsupported luminosity-function model '{self.model}'.")

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

    def absolute_magnitude_limit(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        *,
        m_lim: float,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Return the absolute-magnitude limit of a catalog apparent-magnitude cut.

        Args:
            cosmo_obj: Cosmology object used for distance-modulus conversion.
            z: Redshift values.
            m_lim: Apparent-magnitude limit of the catalog.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Absolute-magnitude limits using the LFKit convention
            ``M_lim = m_lim - mu - K + E``.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return absolute_magnitude_limit(
            cosmo_obj,
            z,
            m_lim=m_lim,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

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
        """Register a phi_star evolution model."""
        register_phi_star_model(name, model, overwrite=overwrite)

    @staticmethod
    def register_m_star_model(
        name: str,
        model: ParameterModel,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an M_star evolution model."""
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

    def integrated_number_density(
        self,
        z: FloatInput,
        *,
        m_bright: ParameterValue,
        m_faint: ParameterValue,
        n_m: int = 512,
    ) -> FloatArray:
        """Integrate the LF over an absolute-magnitude range.

        Args:
            z: Redshift values.
            m_bright: Bright absolute-magnitude integration limit.
            m_faint: Faint absolute-magnitude integration limit.
            n_m: Number of magnitude-grid points used in the integration.

        Returns:
            Number density integrated over the requested magnitude range.
        """
        return integrated_number_density(
            z,
            self._as_callable(),
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
        )

    def observed_number_density(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        *,
        m_lim: float,
        m_bright: float,
        m_faint: float,
        n_m: int = 512,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Return the LF number density observable in a magnitude-limited catalog.

        Args:
            cosmo_obj: Cosmology object used for apparent-to-absolute conversion.
            z: Redshift values.
            m_lim: Apparent-magnitude limit of the catalog.
            m_bright: Bright absolute-magnitude integration limit.
            m_faint: Faint absolute-magnitude integration limit.
            n_m: Number of magnitude-grid points used in the integration.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Number density brighter than the catalog magnitude limit.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return observed_number_density(
            cosmo_obj,
            z,
            self._as_callable(),
            m_lim=m_lim,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    def missing_number_density(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        *,
        m_lim: float,
        m_bright: float,
        m_faint: float,
        n_m: int = 512,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Return the LF number density missing from a magnitude-limited catalog.

        Args:
            cosmo_obj: Cosmology object used for apparent-to-absolute conversion.
            z: Redshift values.
            m_lim: Apparent-magnitude limit of the catalog.
            m_bright: Bright absolute-magnitude integration limit.
            m_faint: Faint absolute-magnitude integration limit.
            n_m: Number of magnitude-grid points used in the integration.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Number density fainter than the catalog magnitude limit but inside
            the requested absolute-magnitude range.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return missing_number_density(
            cosmo_obj,
            z,
            self._as_callable(),
            m_lim=m_lim,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    def catalog_completeness(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        *,
        m_lim: float,
        m_bright: float,
        m_faint: float,
        n_m: int = 512,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Return the observed LF fraction in a magnitude-limited catalog.

        Args:
            cosmo_obj: Cosmology object used for apparent-to-absolute conversion.
            z: Redshift values.
            m_lim: Apparent-magnitude limit of the catalog.
            m_bright: Bright absolute-magnitude integration limit.
            m_faint: Faint absolute-magnitude integration limit.
            n_m: Number of magnitude-grid points used in the integration.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Fraction of the LF number density observable in the catalog.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return catalog_completeness_fraction(
            cosmo_obj,
            z,
            self._as_callable(),
            m_lim=m_lim,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    def out_of_catalog_fraction(
        self,
        cosmo_obj: Cosmology,
        z: FloatInput,
        *,
        m_lim: float,
        m_bright: float,
        m_faint: float,
        n_m: int = 512,
        h: float | None = None,
        corrections: Corrections | None = None,
    ) -> FloatArray:
        """Return the missing LF fraction for a magnitude-limited catalog.

        Args:
            cosmo_obj: Cosmology object used for apparent-to-absolute conversion.
            z: Redshift values.
            m_lim: Apparent-magnitude limit of the catalog.
            m_bright: Bright absolute-magnitude integration limit.
            m_faint: Faint absolute-magnitude integration limit.
            n_m: Number of magnitude-grid points used in the integration.
            h: Optional reduced Hubble parameter used in the magnitude conversion.
            corrections: Optional object providing k-correction and e-correction values.

        Returns:
            Fraction of the LF number density missing from the catalog.
        """
        k_corr, e_corr = self._correction_values(corrections, z)

        return out_of_catalog_fraction(
            cosmo_obj,
            z,
            self._as_callable(),
            m_lim=m_lim,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
            h=h,
            k_correction=k_corr,
            e_correction=e_corr,
        )

    def _as_callable(self):
        """Return this object as an ``lf(M, z)`` callable."""
        return lambda absolute_mag, z: self.phi(absolute_mag, z)

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
