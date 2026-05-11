"""Unit tests for ``lfkit.api.lumfunc.py``."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import lfkit.api.lumfunc as lf_api
from lfkit.api.corrections import Corrections
from lfkit.api.lumfunc import LuminosityFunction


class DummyCorrections:
    """Small correction object used to test public API forwarding."""

    def k(self, z):
        """Tests that k-corrections can be evaluated at z."""
        return np.asarray(z, dtype=float) + 1.0

    def e(self, z):
        """Tests that e-corrections can be evaluated at z."""
        return np.asarray(z, dtype=float) - 1.0


def test_schechter_constructor_stores_model_and_parameters():
    """Tests that the Schechter constructor stores the expected public state."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    assert lf.model == "schechter"
    assert lf.parameters_dict == {
        "phi_star": 1.0e-3,
        "m_star": -20.5,
        "alpha": -1.2,
    }
    assert lf.meta == {}


def test_constructor_copies_parameter_and_metadata_mappings():
    """Tests that input mappings are copied rather than stored by reference."""
    parameters = {"phi_star": 1.0e-3, "m_star": -20.5, "alpha": -1.2}
    meta = {"survey": "test"}

    lf = LuminosityFunction(
        model="schechter",
        parameters=parameters,
        meta=meta,
    )

    parameters["phi_star"] = 9.0
    meta["survey"] = "changed"

    assert lf.parameters_dict["phi_star"] == 1.0e-3
    assert lf.meta["survey"] == "test"


def test_phi_dispatches_schechter_model(monkeypatch):
    """Tests that phi dispatches to the standard Schechter implementation."""
    absolute_mag = np.array([-21.0, -20.0])

    def fake_schechter(mag, *, phi_star, m_star, alpha):
        assert np.allclose(mag, absolute_mag)
        assert phi_star == 1.0e-3
        assert m_star == -20.5
        assert alpha == -1.2
        return np.array([1.0, 2.0])

    monkeypatch.setattr(lf_api, "schechter", fake_schechter)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.phi(absolute_mag)

    assert np.allclose(result, [1.0, 2.0])


def test_phi_requires_redshift_for_evolving_schechter():
    """Tests that evolving Schechter evaluation requires redshift input."""
    lf = LuminosityFunction.evolving_schechter()

    with pytest.raises(ValueError, match="z is required"):
        lf.phi(np.array([-21.0, -20.0]))


def test_phi_dispatches_evolving_schechter_model(monkeypatch):
    """Tests that phi dispatches to the evolving Schechter implementation."""
    absolute_mag = np.array([-21.0, -20.0])
    z = np.array([0.2, 0.8])

    def fake_schechter_evolving(
        mag,
        redshift,
        *,
        phi_model,
        phi_kwargs,
        m_star_model,
        m_star_kwargs,
        alpha_model,
        alpha_kwargs,
    ):
        assert np.allclose(mag, absolute_mag)
        assert np.allclose(redshift, z)
        assert phi_model == "constant"
        assert phi_kwargs == {"value": 1.0e-3}
        assert m_star_model == "constant"
        assert m_star_kwargs == {"value": -20.5}
        assert alpha_model == "constant"
        assert alpha_kwargs == {"value": -1.2}
        return np.array([3.0, 4.0])

    monkeypatch.setattr(lf_api, "schechter_evolving", fake_schechter_evolving)

    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"value": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"value": -20.5},
        alpha_model="constant",
        alpha_kwargs={"value": -1.2},
    )

    result = lf.phi(absolute_mag, z)

    assert np.allclose(result, [3.0, 4.0])


def test_phi_dispatches_double_schechter_model(monkeypatch):
    """Tests that phi dispatches to the double Schechter implementation."""
    absolute_mag = np.array([-21.0, -20.0])

    def fake_schechter_double(
        mag,
        *,
        phi_star,
        m_star,
        alpha,
        beta,
        m_transition,
    ):
        assert np.allclose(mag, absolute_mag)
        assert phi_star == 1.0e-3
        assert m_star == -20.5
        assert alpha == -1.2
        assert beta == -2.0
        assert m_transition == -19.0
        return np.array([5.0, 6.0])

    monkeypatch.setattr(lf_api, "schechter_double", fake_schechter_double)

    lf = LuminosityFunction.double_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
        beta=-2.0,
        m_transition=-19.0,
    )

    result = lf.phi(absolute_mag)

    assert np.allclose(result, [5.0, 6.0])


def test_phi_raises_for_unsupported_model():
    """Tests that unsupported luminosity-function models fail clearly."""
    lf = LuminosityFunction(model="bad_model", parameters={})

    with pytest.raises(ValueError, match="Unsupported luminosity-function model"):
        lf.phi(np.array([-21.0, -20.0]))


def test_phi_from_m_forwards_corrections_to_schechter_from_m(monkeypatch):
    """Tests that phi_from_m forwards correction arrays to magnitude evaluation."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])
    corrections = cast(Corrections, DummyCorrections())

    def fake_schechter_from_m(
        received_cosmo,
        received_z,
        received_apparent_mag,
        *,
        h,
        k_correction,
        e_correction,
        phi_star,
        m_star,
        alpha,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert h == 0.7
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        assert phi_star == 1.0e-3
        assert m_star == -20.5
        assert alpha == -1.2
        return np.array([7.0, 8.0])

    monkeypatch.setattr(lf_api, "schechter_from_m", fake_schechter_from_m)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.phi_from_m(
        cosmo_obj,
        z,
        apparent_mag,
        h=0.7,
        corrections=corrections,
    )

    assert np.allclose(result, [7.0, 8.0])


def test_phi_from_m_passes_none_corrections_when_not_supplied(monkeypatch):
    """Tests that phi_from_m uses None corrections when no correction object is given."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])

    def fake_schechter_from_m(
        received_cosmo,
        received_z,
        received_apparent_mag,
        *,
        h,
        k_correction,
        e_correction,
        phi_star,
        m_star,
        alpha,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert h is None
        assert k_correction is None
        assert e_correction is None
        return np.array([1.0, 1.0])

    monkeypatch.setattr(lf_api, "schechter_from_m", fake_schechter_from_m)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.phi_from_m(cosmo_obj, z, apparent_mag)

    assert np.allclose(result, [1.0, 1.0])


def test_parameters_only_work_for_evolving_schechter(monkeypatch):
    """Tests that parameters delegates only for evolving Schechter models."""
    z = np.array([0.1, 0.5])

    def fake_evaluate_lf_parameters(redshift, **kwargs):
        assert np.allclose(redshift, z)
        assert kwargs["phi_model"] == "constant"
        return (
            np.array([1.0e-3, 1.0e-3]),
            np.array([-20.5, -20.5]),
            np.array([-1.2, -1.2]),
        )

    monkeypatch.setattr(lf_api, "evaluate_lf_parameters", fake_evaluate_lf_parameters)

    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"value": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"value": -20.5},
        alpha_model="constant",
        alpha_kwargs={"value": -1.2},
    )

    phi_star, m_star, alpha = lf.parameters(z)

    assert np.allclose(phi_star, [1.0e-3, 1.0e-3])
    assert np.allclose(m_star, [-20.5, -20.5])
    assert np.allclose(alpha, [-1.2, -1.2])


def test_parameters_raise_for_non_evolving_schechter():
    """Tests that parameters raises for non-evolving luminosity functions."""
    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    with pytest.raises(ValueError, match="only defined for evolving_schechter"):
        lf.parameters(np.array([0.1, 0.5]))


def test_integrated_number_density_uses_api_callable(monkeypatch):
    """Tests that integrated number density evaluates the public LF callable."""
    z = np.array([0.1, 0.5])

    def fake_integrated_number_density(redshift, lf_callable, *, m_bright, m_faint, n_m):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(redshift, z)
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 32
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([10.0, 20.0])

    monkeypatch.setattr(
        lf_api,
        "integrated_number_density",
        fake_integrated_number_density,
    )

    lf = LuminosityFunction(
        model="schechter",
        parameters={"phi_star": 1.0e-3, "m_star": -20.5, "alpha": -1.2},
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.integrated_number_density(
        z,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=32,
    )

    assert np.allclose(result, [10.0, 20.0])


def test_catalog_completeness_forwards_corrections(monkeypatch):
    """Tests that catalog completeness forwards correction arrays."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    corrections = cast(Corrections, DummyCorrections())

    def fake_catalog_completeness_fraction(
        received_cosmo,
        received_z,
        lf_callable,
        *,
        m_lim,
        m_bright,
        m_faint,
        n_m,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 64
        assert h == 0.7
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        assert np.allclose(lf_callable(np.array([-21.0]), np.array([0.1])), [3.0])
        return np.array([0.8, 0.6])

    monkeypatch.setattr(
        lf_api,
        "catalog_completeness_fraction",
        fake_catalog_completeness_fraction,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 3.0),
    )

    result = lf.catalog_completeness(
        cosmo_obj,
        z,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
        h=0.7,
        corrections=corrections,
    )

    assert np.allclose(result, [0.8, 0.6])


def test_observed_and_missing_number_density_are_consistent(monkeypatch):
    """Tests that observed and missing wrappers forward matching inputs."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])

    def fake_observed_number_density(
        received_cosmo,
        received_z,
        lf_callable,
        *,
        m_lim,
        m_bright,
        m_faint,
        n_m,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 128
        assert h is None
        assert k_correction is None
        assert e_correction is None
        return np.array([4.0, 6.0])

    def fake_missing_number_density(
        received_cosmo,
        received_z,
        lf_callable,
        *,
        m_lim,
        m_bright,
        m_faint,
        n_m,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 128
        assert h is None
        assert k_correction is None
        assert e_correction is None
        return np.array([1.0, 2.0])

    monkeypatch.setattr(lf_api, "observed_number_density", fake_observed_number_density)
    monkeypatch.setattr(lf_api, "missing_number_density", fake_missing_number_density)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    observed = lf.observed_number_density(
        cosmo_obj,
        z,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=128,
    )
    missing = lf.missing_number_density(
        cosmo_obj,
        z,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=128,
    )

    assert np.allclose(observed, [4.0, 6.0])
    assert np.allclose(missing, [1.0, 2.0])


def test_catalog_and_out_of_catalog_fractions_sum_to_one(monkeypatch):
    """Tests that catalog and out-of-catalog wrappers preserve fraction semantics."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])

    monkeypatch.setattr(
        lf_api,
        "catalog_completeness_fraction",
        lambda *args, **kwargs: np.array([0.75, 0.25]),
    )
    monkeypatch.setattr(
        lf_api,
        "out_of_catalog_fraction",
        lambda *args, **kwargs: np.array([0.25, 0.75]),
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    completeness = lf.catalog_completeness(
        cosmo_obj,
        z,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
    )
    missing = lf.out_of_catalog_fraction(
        cosmo_obj,
        z,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
    )

    assert np.allclose(completeness + missing, 1.0)
