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
    """Tests that unsupported luminosity function models fail clearly."""
    lf = LuminosityFunction(model="bad_model", parameters={})

    with pytest.raises(ValueError, match="Unsupported luminosity function model"):
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


def test_absolute_magnitude_forwards_corrections(monkeypatch):
    """Tests that absolute_magnitude forwards correction arrays."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])
    corrections = cast(Corrections, DummyCorrections())

    def fake_absolute_magnitude(
        received_cosmo,
        received_z,
        received_apparent_mag,
        *,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert h == 0.7
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        return np.array([-19.0, -20.0])

    monkeypatch.setattr(lf_api, "absolute_magnitude", fake_absolute_magnitude)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.absolute_magnitude(
        cosmo_obj,
        z,
        apparent_mag,
        h=0.7,
        corrections=corrections,
    )

    assert np.allclose(result, [-19.0, -20.0])


def test_apparent_magnitude_forwards_corrections(monkeypatch):
    """Tests that apparent_magnitude forwards correction arrays."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    absolute_mag = np.array([-19.0, -20.0])
    corrections = cast(Corrections, DummyCorrections())

    def fake_apparent_magnitude(
        received_cosmo,
        received_z,
        received_absolute_mag,
        *,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_absolute_mag, absolute_mag)
        assert h == 0.7
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        return np.array([22.0, 24.0])

    monkeypatch.setattr(lf_api, "apparent_magnitude", fake_apparent_magnitude)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.apparent_magnitude(
        cosmo_obj,
        z,
        absolute_mag,
        h=0.7,
        corrections=corrections,
    )

    assert np.allclose(result, [22.0, 24.0])


def test_absolute_magnitude_from_luminosity_distance_forwards_corrections(monkeypatch):
    """Tests that absolute magnitude from distance forwards correction arrays."""
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])
    luminosity_distance_mpc = np.array([500.0, 1500.0])
    corrections = cast(Corrections, DummyCorrections())

    def fake_absolute_magnitude_from_luminosity_distance(
        received_apparent_mag,
        received_luminosity_distance_mpc,
        *,
        k_correction,
        e_correction,
    ):
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert np.allclose(received_luminosity_distance_mpc, luminosity_distance_mpc)
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        return np.array([-18.0, -21.0])

    monkeypatch.setattr(
        lf_api,
        "absolute_magnitude_from_luminosity_distance",
        fake_absolute_magnitude_from_luminosity_distance,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.absolute_magnitude_from_luminosity_distance(
        apparent_mag,
        luminosity_distance_mpc,
        z=z,
        corrections=corrections,
    )

    assert np.allclose(result, [-18.0, -21.0])


def test_apparent_magnitude_from_luminosity_distance_forwards_corrections(monkeypatch):
    """Tests that apparent magnitude from distance forwards correction arrays."""
    z = np.array([0.1, 0.5])
    absolute_mag = np.array([-18.0, -21.0])
    luminosity_distance_mpc = np.array([500.0, 1500.0])
    corrections = cast(Corrections, DummyCorrections())

    def fake_apparent_magnitude_from_luminosity_distance(
        received_absolute_mag,
        received_luminosity_distance_mpc,
        *,
        k_correction,
        e_correction,
    ):
        assert np.allclose(received_absolute_mag, absolute_mag)
        assert np.allclose(received_luminosity_distance_mpc, luminosity_distance_mpc)
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        return np.array([22.0, 24.0])

    monkeypatch.setattr(
        lf_api,
        "apparent_magnitude_from_luminosity_distance",
        fake_apparent_magnitude_from_luminosity_distance,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.apparent_magnitude_from_luminosity_distance(
        absolute_mag,
        luminosity_distance_mpc,
        z=z,
        corrections=corrections,
    )

    assert np.allclose(result, [22.0, 24.0])


def test_absolute_magnitude_limit_forwards_corrections(monkeypatch):
    """Tests that absolute_magnitude_limit forwards correction arrays."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    corrections = cast(Corrections, DummyCorrections())

    def fake_absolute_magnitude_limit(
        received_cosmo,
        received_z,
        *,
        m_lim,
        h,
        k_correction,
        e_correction,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert h == 0.7
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(e_correction, [-0.9, -0.5])
        return np.array([-18.0, -20.0])

    monkeypatch.setattr(lf_api, "absolute_magnitude_limit", fake_absolute_magnitude_limit)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )

    result = lf.absolute_magnitude_limit(
        cosmo_obj,
        z,
        m_lim=24.5,
        h=0.7,
        corrections=corrections,
    )

    assert np.allclose(result, [-18.0, -20.0])


def test_phi_from_m_dispatches_evolving_schechter_from_m(monkeypatch):
    """Tests that phi_from_m dispatches to evolving Schechter from apparent magnitude."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])

    def fake_schechter_evolving_from_m(
        received_cosmo,
        received_z,
        received_apparent_mag,
        *,
        h,
        k_correction,
        e_correction,
        phi_model,
        phi_kwargs,
        m_star_model,
        m_star_kwargs,
        alpha_model,
        alpha_kwargs,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert h is None
        assert k_correction is None
        assert e_correction is None
        assert phi_model == "constant"
        assert phi_kwargs == {"value": 1.0e-3}
        assert m_star_model == "constant"
        assert m_star_kwargs == {"value": -20.5}
        assert alpha_model == "constant"
        assert alpha_kwargs == {"value": -1.2}
        return np.array([9.0, 10.0])

    monkeypatch.setattr(
        lf_api,
        "schechter_evolving_from_m",
        fake_schechter_evolving_from_m,
    )

    lf = LuminosityFunction.evolving_schechter(
        phi_model="constant",
        phi_kwargs={"value": 1.0e-3},
        m_star_model="constant",
        m_star_kwargs={"value": -20.5},
        alpha_model="constant",
        alpha_kwargs={"value": -1.2},
    )

    result = lf.phi_from_m(cosmo_obj, z, apparent_mag)

    assert np.allclose(result, [9.0, 10.0])


def test_phi_from_m_dispatches_double_schechter_from_m(monkeypatch):
    """Tests that phi_from_m dispatches to double Schechter from apparent magnitude."""
    cosmo_obj = object()
    z = np.array([0.1, 0.5])
    apparent_mag = np.array([22.0, 24.0])

    def fake_schechter_double_from_m(
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
        beta,
        m_transition,
    ):
        assert received_cosmo is cosmo_obj
        assert np.allclose(received_z, z)
        assert np.allclose(received_apparent_mag, apparent_mag)
        assert h is None
        assert k_correction is None
        assert e_correction is None
        assert phi_star == 1.0e-3
        assert m_star == -20.5
        assert alpha == -1.2
        assert beta == -2.0
        assert m_transition == -19.0
        return np.array([11.0, 12.0])

    monkeypatch.setattr(
        lf_api,
        "schechter_double_from_m",
        fake_schechter_double_from_m,
    )

    lf = LuminosityFunction.double_schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
        beta=-2.0,
        m_transition=-19.0,
    )

    result = lf.phi_from_m(cosmo_obj, z, apparent_mag)

    assert np.allclose(result, [11.0, 12.0])


def test_phi_from_m_raises_for_unsupported_model():
    """Tests that phi_from_m rejects unsupported LF models."""
    lf = LuminosityFunction(model="bad_model", parameters={})

    with pytest.raises(ValueError, match="Unsupported luminosity function model"):
        lf.phi_from_m(object(), np.array([0.1]), np.array([22.0]))


def test_lf_weighted_integral_forwards_to_module_function(monkeypatch):
    """Tests that lf_weighted_integral forwards inputs to the module function."""
    z = np.array([0.1, 0.5])

    def weight_fn(absolute_mag, redshift):
        return np.ones_like(absolute_mag, dtype=float)

    def fake_lf_weighted_integral(
        received_z,
        lf_callable,
        *,
        m_bright,
        m_faint,
        weight_fn: object,
        n_m,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 64
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([3.0, 4.0])

    monkeypatch.setattr(lf_api, "lf_weighted_integral", fake_lf_weighted_integral)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.lf_weighted_integral(
        z,
        m_bright=-24.0,
        m_faint=-18.0,
        weight_fn=weight_fn,
        n_m=64,
    )

    assert np.allclose(result, [3.0, 4.0])


def test_selection_weighted_number_density_forwards_to_module_function(monkeypatch):
    """Tests that selection_weighted_number_density forwards inputs."""
    z = np.array([0.1, 0.5])

    def selection_fn(absolute_mag, redshift):
        return np.ones_like(absolute_mag, dtype=float)

    def fake_selection_weighted_number_density(
        received_z,
        lf_callable,
        *,
        m_bright,
        m_faint,
        selection_fn: object,
        n_m,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert n_m == 64
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([5.0, 6.0])

    monkeypatch.setattr(
        lf_api,
        "selection_weighted_number_density",
        fake_selection_weighted_number_density,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.selection_weighted_number_density(
        z,
        m_bright=-24.0,
        m_faint=-18.0,
        selection_fn=selection_fn,
        n_m=64,
    )

    assert np.allclose(result, [5.0, 6.0])


def test_integrated_luminosity_density_forwards_to_module_function(monkeypatch):
    """Tests that integrated_luminosity_density forwards inputs."""
    z = np.array([0.1, 0.5])

    def fake_integrated_luminosity_density(
        received_z,
        lf_callable,
        *,
        m_bright,
        m_faint,
        m_reference,
        n_m,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert m_reference == -20.0
        assert n_m == 64
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([7.0, 8.0])

    monkeypatch.setattr(
        lf_api,
        "integrated_luminosity_density",
        fake_integrated_luminosity_density,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.integrated_luminosity_density(
        z,
        m_bright=-24.0,
        m_faint=-18.0,
        m_reference=-20.0,
        n_m=64,
    )

    assert np.allclose(result, [7.0, 8.0])


def test_mean_luminosity_forwards_to_module_function(monkeypatch):
    """Tests that mean_luminosity forwards inputs."""
    z = np.array([0.1, 0.5])

    def fake_mean_luminosity(
        received_z,
        lf_callable,
        *,
        m_bright,
        m_faint,
        m_reference,
        n_m,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert m_reference == -20.0
        assert n_m == 64
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([9.0, 10.0])

    monkeypatch.setattr(lf_api, "mean_luminosity", fake_mean_luminosity)

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.mean_luminosity(
        z,
        m_bright=-24.0,
        m_faint=-18.0,
        m_reference=-20.0,
        n_m=64,
    )

    assert np.allclose(result, [9.0, 10.0])


def test_cumulative_number_density_forwards_to_module_function(monkeypatch):
    """Tests that cumulative_number_density forwards inputs."""
    z = np.array([0.1, 0.5])

    def fake_cumulative_number_density(
        received_z,
        lf_callable,
        *,
        m_threshold,
        m_bright,
        m_faint,
        brighter_than,
        n_m,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_threshold == -20.0
        assert m_bright == -24.0
        assert m_faint == -18.0
        assert brighter_than is False
        assert n_m == 64
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([11.0, 12.0])

    monkeypatch.setattr(
        lf_api,
        "cumulative_number_density",
        fake_cumulative_number_density,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.cumulative_number_density(
        z,
        m_threshold=-20.0,
        m_bright=-24.0,
        m_faint=-18.0,
        brighter_than=False,
        n_m=64,
    )

    assert np.allclose(result, [11.0, 12.0])


def test_lf_integrated_number_density_forwards_corrections(monkeypatch):
    """Tests that apparent-limit LF number density forwards corrections."""
    z = np.array([0.1, 0.5])
    corrections = cast(Corrections, DummyCorrections())

    def luminosity_distance_mpc_fn(redshift):
        return 1000.0 * np.asarray(redshift, dtype=float)

    def fake_lf_integrated_number_density(
        received_z,
        lf_callable,
        *,
        m_lim,
        m_bright,
        n_m,
        luminosity_distance_mpc_fn,
        k_correction,
        evolution_correction,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert m_bright == -24.0
        assert n_m == 64
        assert np.allclose(luminosity_distance_mpc_fn(z), [100.0, 500.0])
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(evolution_correction, [-0.9, -0.5])
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([13.0, 14.0])

    monkeypatch.setattr(
        lf_api,
        "lf_integrated_number_density",
        fake_lf_integrated_number_density,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.lf_integrated_number_density(
        z,
        m_lim=24.5,
        m_bright=-24.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        corrections=corrections,
    )

    assert np.allclose(result, [13.0, 14.0])


def test_lf_weighted_redshift_density_forwards_corrections(monkeypatch):
    """Tests that LF-weighted redshift density forwards corrections."""
    z = np.array([0.1, 0.5])
    corrections = cast(Corrections, DummyCorrections())

    def luminosity_distance_mpc_fn(redshift):
        return 1000.0 * np.asarray(redshift, dtype=float)

    def volume_weight_fn(redshift):
        return 2.0 * np.asarray(redshift, dtype=float)

    def fake_lf_weighted_redshift_density(
        received_z,
        lf_callable,
        *,
        m_lim,
        m_bright,
        n_m,
        luminosity_distance_mpc_fn,
        volume_weight_fn,
        k_correction,
        evolution_correction,
        normalize,
    ):
        mag = np.array([-21.0, -20.0])
        assert np.allclose(received_z, z)
        assert m_lim == 24.5
        assert m_bright == -24.0
        assert n_m == 64
        assert np.allclose(luminosity_distance_mpc_fn(z), [100.0, 500.0])
        assert np.allclose(volume_weight_fn(z), [0.2, 1.0])
        assert np.allclose(k_correction, [1.1, 1.5])
        assert np.allclose(evolution_correction, [-0.9, -0.5])
        assert normalize is False
        assert np.allclose(lf_callable(mag, z), [2.0, 2.0])
        return np.array([15.0, 16.0])

    monkeypatch.setattr(
        lf_api,
        "lf_weighted_redshift_density",
        fake_lf_weighted_redshift_density,
    )

    lf = LuminosityFunction.schechter(
        phi_star=1.0e-3,
        m_star=-20.5,
        alpha=-1.2,
    )
    monkeypatch.setattr(
        lf,
        "phi",
        lambda absolute_mag, z=None: np.full_like(absolute_mag, 2.0),
    )

    result = lf.lf_weighted_redshift_density(
        z,
        m_lim=24.5,
        m_bright=-24.0,
        n_m=64,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        volume_weight_fn=volume_weight_fn,
        corrections=corrections,
        normalize=False,
    )

    assert np.allclose(result, [15.0, 16.0])


def test_available_parameter_models_delegates_to_module_function(monkeypatch):
    """Tests that available_parameter_models delegates to the registry helper."""
    expected = {
        "phi_star": ["constant", "linear_p"],
        "m_star": ["constant", "linear_q"],
        "alpha": ["constant"],
    }

    monkeypatch.setattr(
        lf_api,
        "available_lf_parameter_models",
        lambda: expected,
    )

    result = LuminosityFunction.available_parameter_models()

    assert result == expected


def test_register_phi_star_model_delegates_to_module_function(monkeypatch):
    """Tests that register_phi_star_model delegates to the registry helper."""
    captured = {}

    def model(z, *, value):
        return np.full_like(np.asarray(z, dtype=float), value)

    def fake_register(name, received_model, *, overwrite):
        captured["name"] = name
        captured["model"] = received_model
        captured["overwrite"] = overwrite

    monkeypatch.setattr(lf_api, "register_phi_star_model", fake_register)

    LuminosityFunction.register_phi_star_model(
        "test_phi",
        model,
        overwrite=True,
    )

    assert captured["name"] == "test_phi"
    assert captured["model"] is model
    assert captured["overwrite"] is True


def test_register_m_star_model_delegates_to_module_function(monkeypatch):
    """Tests that register_m_star_model delegates to the registry helper."""
    captured = {}

    def model(z, *, value):
        return np.full_like(np.asarray(z, dtype=float), value)

    def fake_register(name, received_model, *, overwrite):
        captured["name"] = name
        captured["model"] = received_model
        captured["overwrite"] = overwrite

    monkeypatch.setattr(lf_api, "register_m_star_model", fake_register)

    LuminosityFunction.register_m_star_model(
        "test_m_star",
        model,
        overwrite=True,
    )

    assert captured["name"] == "test_m_star"
    assert captured["model"] is model
    assert captured["overwrite"] is True


def test_register_alpha_model_delegates_to_module_function(monkeypatch):
    """Tests that register_alpha_model delegates to the registry helper."""
    captured = {}

    def model(z, *, value):
        return np.full_like(np.asarray(z, dtype=float), value)

    def fake_register(name, received_model, *, overwrite):
        captured["name"] = name
        captured["model"] = received_model
        captured["overwrite"] = overwrite

    monkeypatch.setattr(lf_api, "register_alpha_model", fake_register)

    LuminosityFunction.register_alpha_model(
        "test_alpha",
        model,
        overwrite=True,
    )

    assert captured["name"] == "test_alpha"
    assert captured["model"] is model
    assert captured["overwrite"] is True
