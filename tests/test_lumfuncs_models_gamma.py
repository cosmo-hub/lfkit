"""Unit tests for ``lfkit.luminosity_functions.models.gamma``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.gamma import (
    gamma_lf,
    generalized_gamma_lf,
)
from lfkit.luminosity_functions.models.schechter import schechter
from lfkit.utils.integrators import safe_power10


def test_gamma_lf_returns_positive_finite_values() -> None:
    """Tests that gamma_lf returns positive finite values."""
    m = np.linspace(-22.0, -18.0, 10)

    result = gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_gamma_lf_matches_schechter() -> None:
    """Tests that gamma_lf matches the standard Schechter form."""
    m = np.array([-22.0, -21.0, -20.0, -19.0])

    result = gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
    )
    expected = schechter(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
    )

    np.testing.assert_allclose(result, expected)


def test_gamma_lf_matches_manual_formula() -> None:
    """Tests that gamma_lf matches the analytic magnitude-space formula."""
    m = np.array([-22.0, -21.0, -20.0])
    phi_star = 1.0e-3
    m_star = -20.0
    alpha = -1.2

    result = gamma_lf(
        m,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
    )

    x = safe_power10(-0.4 * (m - m_star))
    expected = (
        0.4
        * np.log(10.0)
        * phi_star
        * x ** (alpha + 1.0)
        * np.exp(-x)
    )

    np.testing.assert_allclose(result, expected)


def test_gamma_lf_rejects_negative_phi_star() -> None:
    """Tests that gamma_lf rejects negative phi_star."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        gamma_lf(
            [-20.0],
            phi_star=-1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
        )


def test_gamma_lf_rejects_nonfinite_absolute_mag() -> None:
    """Tests that gamma_lf rejects non-finite absolute magnitudes."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        gamma_lf(
            [np.nan],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
        )


def test_gamma_lf_rejects_nonfinite_phi_star() -> None:
    """Tests that gamma_lf rejects non-finite phi_star."""
    with pytest.raises(ValueError, match="phi_star contains NaN or infinite values"):
        gamma_lf(
            [-20.0],
            phi_star=np.inf,
            m_star=-20.0,
            alpha=-1.0,
        )


def test_gamma_lf_rejects_nonfinite_alpha() -> None:
    """Tests that gamma_lf rejects non-finite alpha."""
    with pytest.raises(ValueError, match="alpha contains NaN or infinite values"):
        gamma_lf(
            [-20.0],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=np.nan,
        )


def test_gamma_lf_accepts_array_parameters() -> None:
    """Tests that gamma_lf supports array-valued parameters."""
    m = np.array([-22.0, -21.0, -20.0])

    result = gamma_lf(
        m,
        phi_star=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        m_star=-20.0,
        alpha=np.array([-1.2, -1.0, -0.8]),
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_gamma_lf_broadcasts_scalar_parameters() -> None:
    """Tests that gamma_lf broadcasts scalar parameters over magnitudes."""
    m = np.array([-22.0, -21.0, -20.0])

    result = gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
    )

    assert result.shape == m.shape


def test_gamma_lf_extreme_magnitudes_remain_finite() -> None:
    """Tests that gamma_lf remains finite for extreme magnitudes."""
    m = np.array([-1000.0, 1000.0])

    result = gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.0,
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_generalized_gamma_lf_returns_positive_finite_values() -> None:
    """Tests that generalized_gamma_lf returns positive finite values."""
    m = np.linspace(-22.0, -18.0, 10)

    result = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
        beta=0.8,
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_generalized_gamma_lf_matches_gamma_when_beta_is_one() -> None:
    """Tests that generalized_gamma_lf reduces to gamma_lf when beta=1."""
    m = np.array([-22.0, -21.0, -20.0, -19.0])

    result = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
        beta=1.0,
    )
    expected = gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
    )

    np.testing.assert_allclose(result, expected)


def test_generalized_gamma_lf_matches_manual_formula() -> None:
    """Tests that generalized_gamma_lf matches the analytic formula."""
    m = np.array([-22.0, -21.0, -20.0])
    phi_star = 1.0e-3
    m_star = -20.0
    alpha = -1.2
    beta = 0.8

    result = generalized_gamma_lf(
        m,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        beta=beta,
    )

    x = safe_power10(-0.4 * (m - m_star))
    expected = (
        0.4
        * np.log(10.0)
        * beta
        * phi_star
        * x ** (alpha + 1.0)
        * np.exp(-(x**beta))
    )

    np.testing.assert_allclose(result, expected)


def test_generalized_gamma_lf_rejects_negative_phi_star() -> None:
    """Tests that generalized_gamma_lf rejects negative phi_star."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=-1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=1.0,
        )


def test_generalized_gamma_lf_rejects_zero_beta() -> None:
    """Tests that generalized_gamma_lf rejects zero beta."""
    with pytest.raises(ValueError, match="beta must be positive"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=0.0,
        )


def test_generalized_gamma_lf_rejects_negative_beta() -> None:
    """Tests that generalized_gamma_lf rejects negative beta."""
    with pytest.raises(ValueError, match="beta must be positive"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=-0.5,
        )


def test_generalized_gamma_lf_rejects_nonfinite_beta() -> None:
    """Tests that generalized_gamma_lf rejects non-finite beta."""
    with pytest.raises(ValueError, match="beta contains NaN or infinite values"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=np.inf,
        )


def test_generalized_gamma_lf_rejects_nonfinite_absolute_mag() -> None:
    """Tests that generalized_gamma_lf rejects non-finite absolute magnitudes."""
    with pytest.raises(ValueError, match="absolute_mag contains NaN or infinite values"):
        generalized_gamma_lf(
            [np.inf],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=1.0,
        )


def test_generalized_gamma_lf_rejects_nonfinite_phi_star() -> None:
    """Tests that generalized_gamma_lf rejects non-finite phi_star."""
    with pytest.raises(ValueError, match="phi_star contains NaN or infinite values"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=np.nan,
            m_star=-20.0,
            alpha=-1.0,
            beta=1.0,
        )


def test_generalized_gamma_lf_rejects_nonfinite_alpha() -> None:
    """Tests that generalized_gamma_lf rejects non-finite alpha."""
    with pytest.raises(ValueError, match="alpha contains NaN or infinite values"):
        generalized_gamma_lf(
            [-20.0],
            phi_star=1.0e-3,
            m_star=-20.0,
            alpha=np.inf,
            beta=1.0,
        )


def test_generalized_gamma_lf_accepts_array_parameters() -> None:
    """Tests that generalized_gamma_lf supports array-valued parameters."""
    m = np.array([-22.0, -21.0, -20.0])

    result = generalized_gamma_lf(
        m,
        phi_star=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        m_star=-20.0,
        alpha=np.array([-1.2, -1.0, -0.8]),
        beta=np.array([0.8, 1.0, 1.2]),
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_generalized_gamma_lf_broadcasts_scalar_parameters() -> None:
    """Tests that generalized_gamma_lf broadcasts scalar parameters over magnitudes."""
    m = np.array([-22.0, -21.0, -20.0])

    result = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.2,
        beta=0.8,
    )

    assert result.shape == m.shape


def test_generalized_gamma_lf_extreme_magnitudes_remain_finite() -> None:
    """Tests that generalized_gamma_lf remains finite for extreme magnitudes."""
    m = np.array([-1000.0, 1000.0])

    result = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.0,
        beta=1.0,
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_generalized_gamma_lf_beta_changes_bright_end_shape() -> None:
    """Tests that beta changes the bright-end cutoff shape."""
    m = np.array([-24.0])

    shallow = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.0,
        beta=0.5,
    )
    steep = generalized_gamma_lf(
        m,
        phi_star=1.0e-3,
        m_star=-20.0,
        alpha=-1.0,
        beta=2.0,
    )

    assert shallow[0] > steep[0]
