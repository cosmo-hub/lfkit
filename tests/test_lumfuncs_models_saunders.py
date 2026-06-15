"""Unit tests for ``lfkit.luminosity_functions.models.saunders``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.saunders import (
    double_saunders_lf,
    evolving_saunders_lf,
    generalized_saunders_lf,
    saunders_lf,
)
from lfkit.photometry.luminosities import luminosity_ratio


def test_saunders_lf_matches_manual_formula() -> None:
    """Tests that saunders_lf matches the analytic formula."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    phi_star = 0.01
    m_star = -21.0
    alpha = -0.3
    sigma = 0.7

    result = saunders_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        sigma=sigma,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    cutoff = np.exp(-(np.log10(1.0 + x) ** 2) / (2.0 * sigma**2))
    expected = 0.4 * np.log(10.0) * phi_star * x**alpha * cutoff

    np.testing.assert_allclose(result, expected)


def test_saunders_lf_accepts_scalar_input() -> None:
    """Tests that saunders_lf accepts scalar magnitude input."""
    result = saunders_lf(-21.0, phi_star=0.01, m_star=-21.0, alpha=-0.3, sigma=0.7)

    assert np.shape(result) == ()
    assert np.isfinite(result)


def test_saunders_lf_preserves_array_shape() -> None:
    """Tests that saunders_lf preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = saunders_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
    )

    assert result.shape == absolute_mag.shape


def test_saunders_lf_zero_phi_star_returns_zero() -> None:
    """Tests that zero phi_star returns zero values."""
    result = saunders_lf(
        np.array([-23.0, -22.0, -21.0]),
        phi_star=0.0,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
    )

    np.testing.assert_allclose(result, np.zeros(3))


def test_saunders_lf_rejects_negative_phi_star() -> None:
    """Tests that negative phi_star is rejected."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        saunders_lf(
            np.array([-23.0, -22.0]),
            phi_star=-0.01,
            m_star=-21.0,
            alpha=-0.3,
            sigma=0.7,
        )


def test_saunders_lf_rejects_nonpositive_sigma() -> None:
    """Tests that non-positive sigma is rejected."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        saunders_lf(
            np.array([-23.0, -22.0]),
            phi_star=0.01,
            m_star=-21.0,
            alpha=-0.3,
            sigma=0.0,
        )


def test_generalized_saunders_lf_matches_manual_formula() -> None:
    """Tests that generalized_saunders_lf matches the analytic formula."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    phi_star = 0.01
    m_star = -21.0
    alpha = -0.3
    sigma = 0.7
    beta = 1.5

    result = generalized_saunders_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        sigma=sigma,
        beta=beta,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    cutoff_argument = np.log10(1.0 + x) / (np.sqrt(2.0) * sigma)
    cutoff = np.exp(-(cutoff_argument**beta))
    expected = 0.4 * np.log(10.0) * phi_star * x**alpha * cutoff

    np.testing.assert_allclose(result, expected)


def test_generalized_saunders_lf_beta_two_matches_saunders_lf() -> None:
    """Tests that beta=2 gives the standard Saunders model."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])

    result = generalized_saunders_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
        beta=2.0,
    )

    expected = saunders_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
    )

    np.testing.assert_allclose(result, expected)


def test_generalized_saunders_lf_rejects_nonpositive_beta() -> None:
    """Tests that non-positive beta is rejected."""
    with pytest.raises(ValueError, match="beta must be positive"):
        generalized_saunders_lf(
            np.array([-23.0, -22.0]),
            phi_star=0.01,
            m_star=-21.0,
            alpha=-0.3,
            sigma=0.7,
            beta=0.0,
        )


def test_evolving_saunders_lf_matches_saunders_lf_at_zero_evolution() -> None:
    """Tests that zero evolution matches saunders_lf."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    redshift = np.array([0.1, 0.2, 0.3])

    result = evolving_saunders_lf(
        absolute_mag,
        redshift,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
        p=0.0,
        q=0.0,
    )

    expected = saunders_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
    )

    np.testing.assert_allclose(result, expected)


def test_evolving_saunders_lf_applies_density_and_magnitude_evolution() -> None:
    """Tests that evolving_saunders_lf applies p and q evolution."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    redshift = np.array([0.1, 0.2, 0.3])
    phi_star = 0.01
    m_star = -21.0
    p = 2.0
    q = 1.0

    result = evolving_saunders_lf(
        absolute_mag,
        redshift,
        phi_star=phi_star,
        m_star=m_star,
        alpha=-0.3,
        sigma=0.7,
        p=p,
        q=q,
    )

    expected = saunders_lf(
        absolute_mag,
        phi_star=phi_star * (1.0 + redshift) ** p,
        m_star=m_star - q * redshift,
        alpha=-0.3,
        sigma=0.7,
    )

    np.testing.assert_allclose(result, expected)


def test_evolving_saunders_lf_rejects_negative_redshift() -> None:
    """Tests that negative redshift is rejected."""
    with pytest.raises(ValueError, match="redshift must be non-negative"):
        evolving_saunders_lf(
            np.array([-23.0, -22.0]),
            np.array([0.1, -0.2]),
            phi_star=0.01,
            m_star=-21.0,
            alpha=-0.3,
            sigma=0.7,
        )


def test_double_saunders_lf_matches_sum_of_components() -> None:
    """Tests that double_saunders_lf matches the sum of two Saunders models."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])

    result = double_saunders_lf(
        absolute_mag,
        phi_star_1=0.01,
        m_star_1=-21.0,
        alpha_1=-0.3,
        sigma_1=0.7,
        phi_star_2=0.005,
        m_star_2=-20.5,
        alpha_2=-0.8,
        sigma_2=0.5,
    )

    expected = saunders_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-0.3,
        sigma=0.7,
    ) + saunders_lf(
        absolute_mag,
        phi_star=0.005,
        m_star=-20.5,
        alpha=-0.8,
        sigma=0.5,
    )

    np.testing.assert_allclose(result, expected)


def test_double_saunders_lf_preserves_array_shape() -> None:
    """Tests that double_saunders_lf preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = double_saunders_lf(
        absolute_mag,
        phi_star_1=0.01,
        m_star_1=-21.0,
        alpha_1=-0.3,
        sigma_1=0.7,
        phi_star_2=0.005,
        m_star_2=-20.5,
        alpha_2=-0.8,
        sigma_2=0.5,
    )

    assert result.shape == absolute_mag.shape


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            saunders_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -0.3, "sigma": 0.7},
        ),
        (
            generalized_saunders_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
                "beta": 1.5,
            },
        ),
        (
            evolving_saunders_lf,
            {
                "redshift": 0.1,
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
            },
        ),
        (
            double_saunders_lf,
            {
                "phi_star_1": 0.01,
                "m_star_1": -21.0,
                "alpha_1": -0.3,
                "sigma_1": 0.7,
                "phi_star_2": 0.005,
                "m_star_2": -20.5,
                "alpha_2": -0.8,
                "sigma_2": 0.5,
            },
        ),
    ],
)
def test_saunders_models_accept_scalar_inputs(function, kwargs: dict[str, object]) -> None:
    """Tests that all Saunders models accept scalar magnitude input."""
    result = function(-21.0, **kwargs)

    assert np.shape(result) == ()
    assert np.isfinite(result)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            saunders_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -0.3, "sigma": 0.7},
        ),
        (
            generalized_saunders_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
                "beta": 1.5,
            },
        ),
        (
            evolving_saunders_lf,
            {
                "redshift": 0.1,
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
            },
        ),
        (
            double_saunders_lf,
            {
                "phi_star_1": 0.01,
                "m_star_1": -21.0,
                "alpha_1": -0.3,
                "sigma_1": 0.7,
                "phi_star_2": 0.005,
                "m_star_2": -20.5,
                "alpha_2": -0.8,
                "sigma_2": 0.5,
            },
        ),
    ],
)
def test_saunders_models_reject_nonfinite_absolute_mag(
    function,
    kwargs: dict[str, object],
) -> None:
    """Tests that all Saunders models reject non-finite magnitudes."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        function(np.array([-22.0, np.nan]), **kwargs)


@pytest.mark.parametrize(
    ("function", "kwargs", "bad_key", "match"),
    [
        (
            saunders_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -0.3, "sigma": 0.7},
            "phi_star",
            "phi_star contains NaN or infinite values",
        ),
        (
            saunders_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -0.3, "sigma": 0.7},
            "alpha",
            "alpha contains NaN or infinite values",
        ),
        (
            saunders_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -0.3, "sigma": 0.7},
            "sigma",
            "sigma contains NaN or infinite values",
        ),
        (
            generalized_saunders_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
                "beta": 1.5,
            },
            "beta",
            "beta contains NaN or infinite values",
        ),
        (
            evolving_saunders_lf,
            {
                "redshift": 0.1,
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha": -0.3,
                "sigma": 0.7,
            },
            "redshift",
            "redshift contains NaN or infinite values",
        ),
        (
            double_saunders_lf,
            {
                "phi_star_1": 0.01,
                "m_star_1": -21.0,
                "alpha_1": -0.3,
                "sigma_1": 0.7,
                "phi_star_2": 0.005,
                "m_star_2": -20.5,
                "alpha_2": -0.8,
                "sigma_2": 0.5,
            },
            "sigma_2",
            "sigma contains NaN or infinite values",
        ),
    ],
)
def test_saunders_models_reject_nonfinite_validated_parameters(
    function,
    kwargs: dict[str, object],
    bad_key: str,
    match: str,
) -> None:
    """Tests that validated Saunders parameters reject non-finite values."""
    kwargs = dict(kwargs)
    kwargs[bad_key] = np.nan

    with pytest.raises(ValueError, match=match):
        function(np.array([-22.0, -21.0]), **kwargs)


def test_saunders_models_return_float_arrays() -> None:
    """Tests that all Saunders model outputs are floating point."""
    absolute_mag = np.array([-23, -22, -21])

    results = [
        saunders_lf(
            absolute_mag,
            phi_star=1,
            m_star=-21,
            alpha=-0.3,
            sigma=0.7,
        ),
        generalized_saunders_lf(
            absolute_mag,
            phi_star=1,
            m_star=-21,
            alpha=-0.3,
            sigma=0.7,
            beta=1.5,
        ),
        evolving_saunders_lf(
            absolute_mag,
            np.array([0.1, 0.2, 0.3]),
            phi_star=1,
            m_star=-21,
            alpha=-0.3,
            sigma=0.7,
        ),
        double_saunders_lf(
            absolute_mag,
            phi_star_1=1,
            m_star_1=-21,
            alpha_1=-0.3,
            sigma_1=0.7,
            phi_star_2=0.5,
            m_star_2=-20,
            alpha_2=-0.8,
            sigma_2=0.5,
        ),
    ]

    for result in results:
        assert result.dtype.kind == "f"
