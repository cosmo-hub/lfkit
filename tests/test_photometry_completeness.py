"""Unit tests for the ``lfkit.photometry.catalog_completeness.py``"""

import numpy as np
import pytest

import lfkit.photometry.catalog_completeness as cc


def constant_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function."""
    return np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def double_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function with amplitude two."""
    return 2.0 * np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def test_absolute_magnitude_limit_calls_absolute_magnitude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that absolute magnitude limits are delegated to magnitude conversion."""
    calls = {}

    def fake_absolute_magnitude(
        cosmo_obj: object,
        z: np.ndarray,
        apparent_mag: float,
        *,
        h: float | None = None,
        k_correction: float | np.ndarray | None = None,
        e_correction: float | np.ndarray | None = None,
    ) -> np.ndarray:
        calls["cosmo_obj"] = cosmo_obj
        calls["z"] = z
        calls["apparent_mag"] = apparent_mag
        calls["h"] = h
        calls["k_correction"] = k_correction
        calls["e_correction"] = e_correction
        return np.array([-20.0, -19.0])

    monkeypatch.setattr(cc, "absolute_magnitude", fake_absolute_magnitude)

    cosmo_obj = object()
    result = cc.absolute_magnitude_limit(
        cosmo_obj,
        [0.1, 0.2],
        m_lim=24.5,
        h=0.7,
        k_correction=0.1,
        e_correction=0.2,
    )

    np.testing.assert_allclose(result, np.array([-20.0, -19.0]))
    assert calls["cosmo_obj"] is cosmo_obj
    np.testing.assert_allclose(calls["z"], np.array([0.1, 0.2]))
    assert calls["apparent_mag"] == pytest.approx(24.5)
    assert calls["h"] == pytest.approx(0.7)
    assert calls["k_correction"] == pytest.approx(0.1)
    assert calls["e_correction"] == pytest.approx(0.2)


def test_absolute_magnitude_limit_rejects_negative_redshift() -> None:
    """Tests that negative redshifts are rejected."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        cc.absolute_magnitude_limit(object(), [-0.1, 0.2], m_lim=24.5)


def test_absolute_magnitude_limit_rejects_nonfinite_m_lim() -> None:
    """Tests that non-finite apparent magnitude limits are rejected."""
    with pytest.raises(ValueError, match="m_lim must be finite"):
        cc.absolute_magnitude_limit(object(), [0.1, 0.2], m_lim=np.inf)


def test_integrated_number_density_integrates_constant_lf() -> None:
    """Tests that finite-range integration returns the expected width."""
    result = cc.integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_integrated_number_density_integrates_lf_amplitude() -> None:
    """Tests that finite-range integration preserves LF amplitude."""
    result = cc.integrated_number_density(
        [0.1, 0.2],
        double_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([12.0, 12.0]))


def test_integrated_number_density_supports_array_bounds() -> None:
    """Tests that finite-range integration supports redshift-dependent bounds."""
    result = cc.integrated_number_density(
        [0.1, 0.2, 0.3],
        constant_lf,
        m_bright=np.array([-24.0, -23.0, -22.0]),
        m_faint=np.array([-18.0, -18.0, -18.0]),
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 5.0, 4.0]))


def test_integrated_number_density_returns_zero_for_empty_ranges() -> None:
    """Tests that empty magnitude ranges return zero density."""
    result = cc.integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_bright=np.array([-18.0, -20.0]),
        m_faint=np.array([-20.0, -20.0]),
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_integrated_number_density_rejects_negative_redshift() -> None:
    """Tests that finite-range integration rejects negative redshifts."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        cc.integrated_number_density(
            [-0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_small_magnitude_grid() -> None:
    """Tests that finite-range integration requires at least two grid points."""
    with pytest.raises(ValueError, match="n_m must be at least 2"):
        cc.integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_bright=-24.0,
            m_faint=-18.0,
            n_m=1,
        )


def test_integrated_number_density_rejects_nonfinite_lf_values() -> None:
    """Tests that non-finite luminosity-function values are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.full_like(m_abs, np.nan, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) returned non-finite values"):
        cc.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_negative_lf_values() -> None:
    """Tests that negative luminosity-function values are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return -np.ones_like(m_abs, dtype=float)

    with pytest.raises(ValueError, match="lf\\(M, z\\) must be non-negative"):
        cc.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_integrated_number_density_rejects_unbroadcastable_lf_values() -> None:
    """Tests that unbroadcastable luminosity-function outputs are rejected."""

    def bad_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.ones((3, 3), dtype=float)

    with pytest.raises(
        ValueError,
        match="lf\\(M, z\\) must return values broadcastable",
    ):
        cc.integrated_number_density(
            [0.1, 0.2],
            bad_lf,
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_observed_number_density_integrates_to_catalog_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that observed density integrates from bright bound to catalog limit."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    result = cc.observed_number_density(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([3.0, 5.0]))


def test_observed_number_density_clips_catalog_limit_to_faint_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that observed density clips limits fainter than the LF range."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-17.0, -16.0]),
    )

    result = cc.observed_number_density(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=30.0,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_missing_number_density_integrates_from_catalog_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that missing density integrates from catalog limit to faint bound."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    result = cc.missing_number_density(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([3.0, 1.0]))


def test_missing_number_density_clips_catalog_limit_to_bright_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that missing density clips limits brighter than the LF range."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-25.0, -26.0]),
    )

    result = cc.missing_number_density(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=10.0,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([6.0, 6.0]))


def test_observed_number_density_rejects_invalid_magnitude_range() -> None:
    """Tests that observed density rejects invalid LF magnitude bounds."""
    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        cc.observed_number_density(
            object(),
            [0.1, 0.2],
            constant_lf,
            m_lim=24.5,
            m_bright=-18.0,
            m_faint=-24.0,
        )


def test_missing_number_density_rejects_invalid_magnitude_range() -> None:
    """Tests that missing density rejects invalid LF magnitude bounds."""
    with pytest.raises(ValueError, match="m_faint must be larger than m_bright"):
        cc.missing_number_density(
            object(),
            [0.1, 0.2],
            constant_lf,
            m_lim=24.5,
            m_bright=-18.0,
            m_faint=-24.0,
        )


def test_catalog_completeness_fraction_returns_observed_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that catalog completeness returns observed over total density."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    result = cc.catalog_completeness_fraction(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.5, 5.0 / 6.0]))


def test_out_of_catalog_fraction_returns_missing_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that out-of-catalog fraction is one minus completeness."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    result = cc.out_of_catalog_fraction(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.5, 1.0 / 6.0]))


def test_integrated_number_density_accepts_scalar_redshift() -> None:
    """Tests that finite-range integration accepts scalar redshift input."""
    result = cc.integrated_number_density(
        0.1,
        constant_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    assert result.shape == ()
    assert result == pytest.approx(6.0)


def test_integrated_number_density_accepts_broadcastable_scalar_lf_output() -> None:
    """Tests that scalar luminosity-function outputs are broadcast."""

    def scalar_lf(m_abs: np.ndarray, z: np.ndarray) -> float:
        return 2.0

    result = cc.integrated_number_density(
        [0.1, 0.2],
        scalar_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([12.0, 12.0]))


def test_integrated_number_density_rejects_nonfinite_magnitude_bounds() -> None:
    """Tests that non-finite magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_lower contains NaN or infinite values"):
        cc.integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_bright=np.nan,
            m_faint=-18.0,
        )


def test_catalog_completeness_fraction_returns_zero_for_zero_total_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that completeness is zero when total density is zero."""

    def zero_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.zeros_like(m_abs, dtype=float)

    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    result = cc.catalog_completeness_fraction(
        object(),
        [0.1, 0.2],
        zero_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_completeness_and_out_of_catalog_fractions_sum_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that completeness and out-of-catalog fractions are complementary."""
    monkeypatch.setattr(
        cc,
        "absolute_magnitude_limit",
        lambda *args, **kwargs: np.array([-21.0, -19.0]),
    )

    completeness = cc.catalog_completeness_fraction(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )
    missing = cc.out_of_catalog_fraction(
        object(),
        [0.1, 0.2],
        constant_lf,
        m_lim=24.5,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=64,
    )

    np.testing.assert_allclose(completeness + missing, np.ones(2))
