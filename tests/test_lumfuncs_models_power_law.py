"""Unit tests for ``lfkit.luminosity_functions.models.power_law``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.power_law import (
    broken_power_law_lf,
    double_power_law_lf,
    log_power_law_lf,
    power_law_lf,
)
from lfkit.photometry.luminosities import luminosity_ratio


def test_power_law_lf_matches_manual_formula() -> None:
    """Tests that power_law_lf matches the analytic formula."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    phi_star = 0.01
    m_star = -21.0
    alpha = -1.2

    result = power_law_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    expected = 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0)

    np.testing.assert_allclose(result, expected)


def test_power_law_lf_accepts_scalar_input() -> None:
    """Tests that power_law_lf accepts scalar magnitude input."""
    result = power_law_lf(-21.0, phi_star=0.01, m_star=-21.0, alpha=-1.0)

    assert np.shape(result) == ()
    assert np.isfinite(result)


def test_power_law_lf_preserves_array_shape() -> None:
    """Tests that power_law_lf preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = power_law_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-1.0,
    )

    assert result.shape == absolute_mag.shape


def test_power_law_lf_zero_phi_star_returns_zero() -> None:
    """Tests that zero phi_star returns zero values."""
    result = power_law_lf(
        np.array([-23.0, -22.0, -21.0]),
        phi_star=0.0,
        m_star=-21.0,
        alpha=-1.0,
    )

    np.testing.assert_allclose(result, np.zeros(3))


def test_power_law_lf_rejects_negative_phi_star() -> None:
    """Tests that negative phi_star is rejected."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        power_law_lf(
            np.array([-23.0, -22.0]),
            phi_star=-0.01,
            m_star=-21.0,
            alpha=-1.0,
        )


def test_double_power_law_lf_matches_manual_formula() -> None:
    """Tests that double_power_law_lf matches the analytic formula."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    phi_star = 0.01
    m_star = -21.0
    alpha = -1.2
    beta = -3.0

    result = double_power_law_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        beta=beta,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    expected = (
        0.4
        * np.log(10.0)
        * phi_star
        / (x ** (-(alpha + 1.0)) + x ** (-(beta + 1.0)))
    )

    np.testing.assert_allclose(result, expected)


def test_double_power_law_lf_value_at_m_star() -> None:
    """Tests that double_power_law_lf has the expected value at x=1."""
    phi_star = 0.01

    result = double_power_law_lf(
        -21.0,
        phi_star=phi_star,
        m_star=-21.0,
        alpha=-1.2,
        beta=-3.0,
    )

    expected = 0.5 * 0.4 * np.log(10.0) * phi_star
    np.testing.assert_allclose(result, expected)


def test_double_power_law_lf_preserves_array_shape() -> None:
    """Tests that double_power_law_lf preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = double_power_law_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha=-1.2,
        beta=-3.0,
    )

    assert result.shape == absolute_mag.shape


def test_double_power_law_lf_rejects_negative_phi_star() -> None:
    """Tests that negative phi_star is rejected."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        double_power_law_lf(
            np.array([-23.0, -22.0]),
            phi_star=-0.01,
            m_star=-21.0,
            alpha=-1.2,
            beta=-3.0,
        )


def test_broken_power_law_lf_matches_manual_formula() -> None:
    """Tests that broken_power_law_lf matches the piecewise analytic formula."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    phi_star = 0.01
    m_star = -21.0
    alpha_faint = -1.1
    alpha_bright = -3.0

    result = broken_power_law_lf(
        absolute_mag,
        phi_star=phi_star,
        m_star=m_star,
        alpha_faint=alpha_faint,
        alpha_bright=alpha_bright,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    expected = np.where(
        x < 1.0,
        phi_star * x ** (alpha_faint + 1.0),
        phi_star * x ** (alpha_bright + 1.0),
    )
    expected = 0.4 * np.log(10.0) * expected

    np.testing.assert_allclose(result, expected)


def test_broken_power_law_lf_uses_bright_branch_at_break() -> None:
    """Tests that x=1 uses the bright-side branch."""
    phi_star = 0.01

    result = broken_power_law_lf(
        -21.0,
        phi_star=phi_star,
        m_star=-21.0,
        alpha_faint=-1.1,
        alpha_bright=-3.0,
    )

    expected = 0.4 * np.log(10.0) * phi_star
    np.testing.assert_allclose(result, expected)


def test_broken_power_law_lf_preserves_array_shape() -> None:
    """Tests that broken_power_law_lf preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = broken_power_law_lf(
        absolute_mag,
        phi_star=0.01,
        m_star=-21.0,
        alpha_faint=-1.1,
        alpha_bright=-3.0,
    )

    assert result.shape == absolute_mag.shape


def test_broken_power_law_lf_rejects_negative_phi_star() -> None:
    """Tests that negative phi_star is rejected."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        broken_power_law_lf(
            np.array([-23.0, -22.0]),
            phi_star=-0.01,
            m_star=-21.0,
            alpha_faint=-1.1,
            alpha_bright=-3.0,
        )


def test_log_power_law_lf_matches_power_law_lf() -> None:
    """Tests that log_power_law_lf matches power_law_lf with phi_star=10**log_phi_star."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    log_phi_star = -2.0

    result = log_power_law_lf(
        absolute_mag,
        log_phi_star=log_phi_star,
        m_star=-21.0,
        alpha=-1.2,
    )

    expected = power_law_lf(
        absolute_mag,
        phi_star=10.0**log_phi_star,
        m_star=-21.0,
        alpha=-1.2,
    )

    np.testing.assert_allclose(result, expected)


def test_log_power_law_lf_accepts_array_log_phi_star() -> None:
    """Tests that log_power_law_lf supports array-valued log_phi_star."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])

    result = log_power_law_lf(
        absolute_mag,
        log_phi_star=np.array([-3.0, -2.0, -1.0]),
        m_star=-21.0,
        alpha=-1.2,
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2},
        ),
        (
            double_power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2, "beta": -3.0},
        ),
        (
            broken_power_law_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha_faint": -1.1,
                "alpha_bright": -3.0,
            },
        ),
        (
            log_power_law_lf,
            {"log_phi_star": -2.0, "m_star": -21.0, "alpha": -1.2},
        ),
    ],
)
def test_power_law_models_accept_scalar_inputs(function, kwargs: dict[str, object]) -> None:
    """Tests that all power-law models accept scalar magnitude input."""
    result = function(-21.0, **kwargs)

    assert np.shape(result) == ()
    assert np.isfinite(result)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2},
        ),
        (
            double_power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2, "beta": -3.0},
        ),
        (
            broken_power_law_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha_faint": -1.1,
                "alpha_bright": -3.0,
            },
        ),
        (
            log_power_law_lf,
            {"log_phi_star": -2.0, "m_star": -21.0, "alpha": -1.2},
        ),
    ],
)
def test_power_law_models_reject_nonfinite_absolute_mag(
    function,
    kwargs: dict[str, object],
) -> None:
    """Tests that all power-law models reject non-finite magnitudes."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        function(np.array([-22.0, np.nan]), **kwargs)


@pytest.mark.parametrize(
    ("function", "kwargs", "bad_key", "match"),
    [
        (
            power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2},
            "phi_star",
            "phi_star contains NaN or infinite values",
        ),
        (
            power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2},
            "alpha",
            "alpha contains NaN or infinite values",
        ),
        (
            double_power_law_lf,
            {"phi_star": 0.01, "m_star": -21.0, "alpha": -1.2, "beta": -3.0},
            "beta",
            "beta contains NaN or infinite values",
        ),
        (
            broken_power_law_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha_faint": -1.1,
                "alpha_bright": -3.0,
            },
            "alpha_faint",
            "alpha_faint contains NaN or infinite values",
        ),
        (
            broken_power_law_lf,
            {
                "phi_star": 0.01,
                "m_star": -21.0,
                "alpha_faint": -1.1,
                "alpha_bright": -3.0,
            },
            "alpha_bright",
            "alpha_bright contains NaN or infinite values",
        ),
        (
            log_power_law_lf,
            {"log_phi_star": -2.0, "m_star": -21.0, "alpha": -1.2},
            "log_phi_star",
            "log_phi_star contains NaN or infinite values",
        ),
    ],
)
def test_power_law_models_reject_nonfinite_validated_parameters(
    function,
    kwargs: dict[str, object],
    bad_key: str,
    match: str,
) -> None:
    """Tests that validated power-law parameters reject non-finite values."""
    kwargs = dict(kwargs)
    kwargs[bad_key] = np.nan

    with pytest.raises(ValueError, match=match):
        function(np.array([-22.0, -21.0]), **kwargs)


def test_power_law_models_return_float_arrays() -> None:
    """Tests that all power-law model outputs are floating point."""
    absolute_mag = np.array([-23, -22, -21])

    results = [
        power_law_lf(absolute_mag, phi_star=1, m_star=-21, alpha=-1),
        double_power_law_lf(
            absolute_mag,
            phi_star=1,
            m_star=-21,
            alpha=-1,
            beta=-3,
        ),
        broken_power_law_lf(
            absolute_mag,
            phi_star=1,
            m_star=-21,
            alpha_faint=-1,
            alpha_bright=-3,
        ),
        log_power_law_lf(absolute_mag, log_phi_star=-2, m_star=-21, alpha=-1),
    ]

    for result in results:
        assert result.dtype.kind == "f"
