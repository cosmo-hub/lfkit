"""Unit tests for ``lfkit.luminosity_functions.models.schechter``."""

import numpy as np
import pytest

from lfkit.photometry.luminosities import  luminosity_ratio
from lfkit.luminosity_functions.models.schechter import (
    schechter,
    evolving_schechter,
    double_schechter,
    schechter_cumulative,
    schechter_cumulative_evolving,
)


def test_schechter_positive_output():
    """Tests that schechter returns positive values for valid inputs."""
    M = np.linspace(-22, -18, 10)
    phi = schechter(M, phi_star=1e-3, m_star=-20.0, alpha=-1.2)
    assert np.all(phi >= 0)


def test_schechter_zero_phi_star_warning():
    """Tests that schechter warns when phi_star is zero."""
    M = np.array([-20.0])
    with pytest.warns(UserWarning):
        phi = schechter(M, phi_star=0.0, m_star=-20.0, alpha=-1.0)
    assert np.all(phi == 0)


def test_schechter_negative_phi_star_error():
    """Tests that schechter raises for negative phi_star."""
    M = np.array([-20.0])
    with pytest.raises(ValueError):
        schechter(M, phi_star=-1.0, m_star=-20.0, alpha=-1.0)


def test_evolving_schechter_matches_constant_case():
    """Tests that evolving_schechter reduces to schechter for constant models."""
    M = np.array([-20.0])
    z = np.array([0.5])

    phi1 = schechter(
        M,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-1.0,
    )

    phi2 = evolving_schechter(
        M,
        z,
        phi_model="constant",
        phi_kwargs={"phi_star": 1e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.0},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.0},
    )

    assert np.allclose(phi1, phi2)


def test_evolving_schechter_invalid_model():
    """Tests that evolving_schechter raises for invalid model names."""
    with pytest.raises(ValueError):
        evolving_schechter(
            [-20.0],
            [0.5],
            phi_model="invalid",
        )


def test_double_schechter_positive():
    """Tests that double_schechter returns finite positive values."""
    M = np.linspace(-22, -18, 10)
    phi = double_schechter(
        M,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-1.2,
        beta=1.0,
        m_transition=-18.0,
    )
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0)


def test_double_schechter_invalid_alpha():
    """Tests that double_schechter raises for non-finite alpha."""
    with pytest.raises(ValueError):
        double_schechter(
            [-20.0],
            phi_star=1e-3,
            m_star=-20.0,
            alpha=np.inf,
            beta=1.0,
            m_transition=-18.0,
        )


def test_schechter_cumulative_positive():
    """Tests that schechter_cumulative returns positive densities."""
    M_lim = np.array([-20.0])
    n = schechter_cumulative(
        M_lim,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-0.5,
    )
    assert np.all(n > 0)


def test_schechter_cumulative_invalid_alpha():
    """Tests that schechter_cumulative raises for alpha <= -1."""
    with pytest.raises(ValueError):
        schechter_cumulative(
            [-20.0],
            phi_star=1e-3,
            m_star=-20.0,
            alpha=-1.5,
        )


def test_schechter_cumulative_evolving_shape():
    """Tests that schechter_cumulative_evolving broadcasts correctly."""
    M_lim = np.array([-20.0, -19.0])
    z = np.array([0.5, 1.0])

    n = schechter_cumulative_evolving(
        M_lim,
        z,
        phi_model="constant",
        phi_kwargs={"phi_star": 1e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.0},
        alpha_model="constant",
        alpha_kwargs={"alpha": -0.5},
    )

    assert n.shape == M_lim.shape


def test_schechter_cumulative_evolving_invalid_model():
    """Tests that schechter_cumulative_evolving raises for invalid models."""
    with pytest.raises(ValueError):
        schechter_cumulative_evolving(
            [-20.0],
            [0.5],
            phi_model="invalid",
        )


def test_schechter_broadcasting_scalar_params():
    """Tests that schechter broadcasts scalar parameters over array magnitudes."""
    M = np.array([-22.0, -21.0, -20.0])
    phi = schechter(M, phi_star=1e-3, m_star=-20.0, alpha=-1.0)

    assert phi.shape == M.shape


def test_luminosity_ratio_accepts_list_input():
    """Tests that luminosity_ratio correctly handles list inputs."""
    M = [-20.0, -21.0]
    out = luminosity_ratio(M, m_star=-20.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


def test_schechter_cumulative_brighter_vs_fainter():
    """Tests that brighter and fainter cumulative sums are consistent."""
    M_lim = np.array([-20.0])

    n_bright = schechter_cumulative(
        M_lim,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-0.5,
        brighter_than=True,
    )

    n_faint = schechter_cumulative(
        M_lim,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-0.5,
        brighter_than=False,
    )

    # total = bright + faint
    total = n_bright + n_faint

    assert np.all(total > 0)


def test_schechter_cumulative_evolving_matches_constant():
    """Tests that evolving cumulative reduces to standard cumulative for constant models."""
    M_lim = np.array([-20.0])
    z = np.array([0.5])

    n1 = schechter_cumulative(
        M_lim,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-0.5,
    )

    n2 = schechter_cumulative_evolving(
        M_lim,
        z,
        phi_model="constant",
        phi_kwargs={"phi_star": 1e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.0},
        alpha_model="constant",
        alpha_kwargs={"alpha": -0.5},
    )

    assert np.allclose(n1, n2)


def test_double_schechter_transition_effect():
    """Tests that double_schechter changes slope across transition magnitude."""
    M = np.array([-19.0, -17.0])

    phi = double_schechter(
        M,
        phi_star=1e-3,
        m_star=-20.0,
        alpha=-1.2,
        beta=2.0,
        m_transition=-18.0,
    )

    assert phi[1] != phi[0]


def test_schechter_extreme_magnitudes_finite():
    """Tests that schechter remains finite for extreme magnitudes."""
    M = np.array([-30.0, -10.0])
    phi = schechter(M, phi_star=1e-3, m_star=-20.0, alpha=-1.0)

    assert np.all(np.isfinite(phi))


def test_evolving_schechter_missing_kwargs():
    """Tests that evolving_schechter raises if required kwargs are missing."""
    with pytest.raises(TypeError):
        evolving_schechter(
            [-20.0],
            [0.5],
            phi_model="linear_p",  # requires phi_0_star and p
        )


def test_schechter_matches_manual_formula() -> None:
    """Tests that schechter matches the analytic magnitude-space formula."""
    m = np.array([-22.0, -21.0, -20.0])
    phi_star = 1e-3
    m_star = -20.0
    alpha = -1.2

    result = schechter(m, phi_star=phi_star, m_star=m_star, alpha=alpha)

    x = luminosity_ratio(m, m_star)
    expected = 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-x)

    np.testing.assert_allclose(result, expected)


def test_schechter_rejects_nonfinite_phi_star() -> None:
    """Tests that schechter rejects non-finite phi_star."""
    with pytest.raises(ValueError, match="phi_star contains NaN or infinite values"):
        schechter([-20.0], phi_star=np.nan, m_star=-20.0, alpha=-1.0)


def test_schechter_rejects_nonfinite_alpha() -> None:
    """Tests that schechter rejects non-finite alpha."""
    with pytest.raises(ValueError, match="alpha contains NaN or infinite values"):
        schechter([-20.0], phi_star=1e-3, m_star=-20.0, alpha=np.inf)


def test_schechter_allows_array_parameters() -> None:
    """Tests that schechter supports array-valued parameters."""
    m = np.array([-22.0, -21.0, -20.0])

    result = schechter(
        m,
        phi_star=np.array([1e-3, 2e-3, 3e-3]),
        m_star=-20.0,
        alpha=np.array([-1.2, -1.0, -0.8]),
    )

    assert result.shape == m.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_double_schechter_matches_manual_formula() -> None:
    """Tests that double_schechter matches the analytic formula."""
    m = np.array([-22.0, -21.0, -20.0])
    phi_star = 1e-3
    m_star = -20.0
    alpha = -1.2
    beta = 1.5
    m_transition = -18.0

    result = double_schechter(
        m,
        phi_star=phi_star,
        m_star=m_star,
        alpha=alpha,
        beta=beta,
        m_transition=m_transition,
    )

    x = luminosity_ratio(m, m_star)
    x_t = luminosity_ratio(m_transition, m_star)
    expected = (
        0.4
        * np.log(10.0)
        * phi_star
        * x ** (alpha + 1.0)
        * np.exp(-x)
        * (1.0 + (x / x_t) ** beta)
    )

    np.testing.assert_allclose(result, expected)


def test_double_schechter_zero_phi_star_warning() -> None:
    """Tests that double_schechter warns when phi_star is zero."""
    with pytest.warns(UserWarning):
        result = double_schechter(
            [-20.0],
            phi_star=0.0,
            m_star=-20.0,
            alpha=-1.0,
            beta=1.0,
            m_transition=-18.0,
        )

    np.testing.assert_allclose(result, np.array([0.0]))


def test_double_schechter_rejects_negative_phi_star() -> None:
    """Tests that double_schechter rejects negative phi_star."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        double_schechter(
            [-20.0],
            phi_star=-1e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=1.0,
            m_transition=-18.0,
        )


def test_double_schechter_rejects_nonfinite_beta() -> None:
    """Tests that double_schechter rejects non-finite beta."""
    with pytest.raises(ValueError, match="beta must be finite"):
        double_schechter(
            [-20.0],
            phi_star=1e-3,
            m_star=-20.0,
            alpha=-1.0,
            beta=np.nan,
            m_transition=-18.0,
        )


def test_schechter_cumulative_rejects_nonfinite_phi_star() -> None:
    """Tests that schechter_cumulative rejects non-finite phi_star."""
    with pytest.raises(ValueError, match="phi_star must be finite"):
        schechter_cumulative(
            [-20.0],
            phi_star=np.inf,
            m_star=-20.0,
            alpha=-0.5,
        )


def test_schechter_cumulative_rejects_negative_phi_star() -> None:
    """Tests that schechter_cumulative rejects negative phi_star."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        schechter_cumulative(
            [-20.0],
            phi_star=-1e-3,
            m_star=-20.0,
            alpha=-0.5,
        )


def test_schechter_cumulative_zero_phi_star_returns_zero() -> None:
    """Tests that zero phi_star gives zero cumulative density."""
    result = schechter_cumulative(
        [-20.0],
        phi_star=0.0,
        m_star=-20.0,
        alpha=-0.5,
    )

    np.testing.assert_allclose(result, np.array([0.0]))


def test_schechter_cumulative_faint_plus_bright_equals_total_gamma() -> None:
    """Tests that bright and faint cumulative branches sum to total density."""
    m_lim = np.array([-20.0, -19.0])
    phi_star = 1e-3
    alpha = -0.5

    n_bright = schechter_cumulative(
        m_lim,
        phi_star=phi_star,
        m_star=-20.0,
        alpha=alpha,
        brighter_than=True,
    )
    n_faint = schechter_cumulative(
        m_lim,
        phi_star=phi_star,
        m_star=-20.0,
        alpha=alpha,
        brighter_than=False,
    )

    expected_total = phi_star * np.sqrt(np.pi)
    np.testing.assert_allclose(n_bright + n_faint, expected_total)


def test_schechter_cumulative_evolving_rejects_negative_phi_star() -> None:
    """Tests that evolving cumulative rejects negative evolved phi_star."""
    with pytest.raises(ValueError, match="phi_star must be non-negative"):
        schechter_cumulative_evolving(
            [-20.0],
            [0.5],
            phi_model="constant",
            phi_kwargs={"phi_star": -1e-3},
            m_star_model="constant",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="constant",
            alpha_kwargs={"alpha": -0.5},
        )


def test_schechter_cumulative_evolving_rejects_divergent_alpha() -> None:
    """Tests that evolving cumulative rejects alpha <= -1."""
    with pytest.raises(ValueError, match="undefined where alpha <= -1"):
        schechter_cumulative_evolving(
            [-20.0],
            [0.5],
            phi_model="constant",
            phi_kwargs={"phi_star": 1e-3},
            m_star_model="constant",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="constant",
            alpha_kwargs={"alpha": -1.0},
        )


def test_double_schechter_accepts_array_phi_star() -> None:
    """Tests that double_schechter broadcasts array phi_star."""

    result = double_schechter(
        [-22.0, -21.0],
        phi_star=np.array([1e-3, 2e-3]),
        m_star=-20.0,
        alpha=-1.0,
        beta=1.0,
        m_transition=-18.0,
    )

    assert result.shape == (2,)
    assert np.all(result >= 0.0)
