"""Unit tests for ``lfkit.luminosity_functions.models.modifiers``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.modifiers import apply_luminosity_cutoff
from lfkit.photometry.luminosities import luminosity_ratio


def constant_lf(absolute_mag, *, m_star, amplitude=1.0):
    """Return a constant base luminosity function."""
    return amplitude * np.ones_like(absolute_mag, dtype=float)


def linear_lf(absolute_mag, *, m_star, slope=1.0):
    """Return a simple magnitude-dependent base luminosity function."""
    return slope * (np.asarray(absolute_mag, dtype=float) - m_star)


def test_apply_luminosity_cutoff_matches_manual_formula() -> None:
    """Tests that apply_luminosity_cutoff matches exp(-A x**p)."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    m_star = -21.0
    cutoff_power = 2.0
    cutoff_amplitude = 0.5

    result = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=constant_lf,
        m_star=m_star,
        cutoff_power=cutoff_power,
        cutoff_amplitude=cutoff_amplitude,
        amplitude=3.0,
    )

    x = luminosity_ratio(absolute_mag, m_star)
    expected = 3.0 * np.exp(-cutoff_amplitude * x**cutoff_power)

    np.testing.assert_allclose(result, expected)


def test_apply_luminosity_cutoff_passes_parameters_to_base_lf() -> None:
    """Tests that extra keyword parameters are forwarded to the base LF."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])
    m_star = -22.0

    result = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=linear_lf,
        m_star=m_star,
        cutoff_amplitude=0.0,
        slope=2.0,
    )

    expected = linear_lf(absolute_mag, m_star=m_star, slope=2.0)
    np.testing.assert_allclose(result, expected)


def test_apply_luminosity_cutoff_zero_amplitude_leaves_base_lf_unchanged() -> None:
    """Tests that zero cutoff amplitude gives the unmodified base LF."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])

    result = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=constant_lf,
        m_star=-22.0,
        cutoff_amplitude=0.0,
        amplitude=4.0,
    )

    np.testing.assert_allclose(result, np.full_like(absolute_mag, 4.0))


def test_apply_luminosity_cutoff_accepts_scalar_input() -> None:
    """Tests that scalar magnitude input is accepted."""
    result = apply_luminosity_cutoff(
        -21.0,
        base_lf=constant_lf,
        m_star=-21.0,
    )

    assert np.shape(result) == ()
    assert np.isfinite(result)


def test_apply_luminosity_cutoff_preserves_array_shape() -> None:
    """Tests that the modified luminosity function preserves input shape."""
    absolute_mag = np.array([[-23.0, -22.0], [-21.0, -20.0]])

    result = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=constant_lf,
        m_star=-21.0,
    )

    assert result.shape == absolute_mag.shape


def test_apply_luminosity_cutoff_returns_float_array() -> None:
    """Tests that the returned values are floating point."""
    result = apply_luminosity_cutoff(
        np.array([-23, -22, -21]),
        base_lf=constant_lf,
        m_star=-21,
    )

    assert result.dtype.kind == "f"


def test_apply_luminosity_cutoff_rejects_zero_cutoff_power() -> None:
    """Tests that zero cutoff power is rejected."""
    with pytest.raises(ValueError, match="cutoff_power must be positive"):
        apply_luminosity_cutoff(
            np.array([-22.0, -21.0]),
            base_lf=constant_lf,
            m_star=-21.0,
            cutoff_power=0.0,
        )


def test_apply_luminosity_cutoff_rejects_negative_cutoff_power() -> None:
    """Tests that negative cutoff power is rejected."""
    with pytest.raises(ValueError, match="cutoff_power must be positive"):
        apply_luminosity_cutoff(
            np.array([-22.0, -21.0]),
            base_lf=constant_lf,
            m_star=-21.0,
            cutoff_power=-1.0,
        )


def test_apply_luminosity_cutoff_rejects_negative_cutoff_amplitude() -> None:
    """Tests that negative cutoff amplitude is rejected."""
    with pytest.raises(ValueError, match="cutoff_amplitude must be non-negative"):
        apply_luminosity_cutoff(
            np.array([-22.0, -21.0]),
            base_lf=constant_lf,
            m_star=-21.0,
            cutoff_amplitude=-1.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"absolute_mag": np.array([-22.0, np.nan])},
            "absolute_mag contains NaN or infinite values",
        ),
        (
            {"cutoff_power": np.nan},
            "cutoff_power contains NaN or infinite values",
        ),
        (
            {"cutoff_amplitude": np.inf},
            "cutoff_amplitude contains NaN or infinite values",
        ),
    ],
)
def test_apply_luminosity_cutoff_rejects_nonfinite_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Tests that non-finite modifier inputs are rejected."""
    params = {
        "absolute_mag": np.array([-22.0, -21.0]),
        "base_lf": constant_lf,
        "m_star": -21.0,
        "cutoff_power": 2.0,
        "cutoff_amplitude": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        apply_luminosity_cutoff(**params)


def test_apply_luminosity_cutoff_allows_array_cutoff_parameters() -> None:
    """Tests that array-valued cutoff parameters broadcast through the modifier."""
    absolute_mag = np.array([-23.0, -22.0, -21.0])

    result = apply_luminosity_cutoff(
        absolute_mag,
        base_lf=constant_lf,
        m_star=-21.0,
        cutoff_power=np.array([1.0, 2.0, 3.0]),
        cutoff_amplitude=np.array([0.1, 0.2, 0.3]),
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))
