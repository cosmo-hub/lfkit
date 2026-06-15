"""Unit tests for the ``lfkit.photometry.lf_redshift_density``."""

import numpy as np
import pytest

import lfkit.luminosity_functions.redshift_density as lfrd


M_LIM = 26.0
M_BRIGHT = -5.0


def constant_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function."""
    return np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def double_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a constant luminosity function with amplitude two."""
    return 2.0 * np.ones_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def zero_lf(m_abs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return a zero luminosity function."""
    return np.zeros_like(np.broadcast_arrays(m_abs, z)[0], dtype=float)


def constant_luminosity_distance(z: np.ndarray) -> np.ndarray:
    """Return a constant positive luminosity distance in Mpc."""
    return np.full_like(z, 10.0, dtype=float)


def linear_luminosity_distance(z: np.ndarray) -> np.ndarray:
    """Return a simple positive luminosity distance in Mpc."""
    return 10.0 * (1.0 + np.asarray(z, dtype=float))


def constant_volume_weight(z: np.ndarray) -> np.ndarray:
    """Return a constant non-negative volume weight."""
    return np.ones_like(z, dtype=float)


def linear_volume_weight(z: np.ndarray) -> np.ndarray:
    """Return a simple redshift-dependent volume weight."""
    return 1.0 + np.asarray(z, dtype=float)


def expected_absolute_magnitude_limit(
    m_lim: float,
    luminosity_distance_mpc: np.ndarray,
    *,
    k_correction: float | np.ndarray = 0.0,
    evolution_correction: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Return the expected absolute magnitude limit for test inputs."""
    return (
        m_lim
        - 5.0 * np.log10(luminosity_distance_mpc)
        - 25.0
        - k_correction
        + evolution_correction
    )


def test_lf_integrated_number_density_integrates_to_absolute_magnitude_limit() -> None:
    """Tests LF integration to the apparent magnitude-implied absolute limit."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
    )

    np.testing.assert_allclose(result, np.array([1.0, 1.0]))


def test_lf_integrated_number_density_preserves_lf_amplitude() -> None:
    """Tests that LF amplitude is preserved in the redshift-density integral."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        double_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
    )

    np.testing.assert_allclose(result, np.array([2.0, 2.0]))


def test_lf_integrated_number_density_accepts_scalar_redshift() -> None:
    """Tests that LF-integrated number density accepts scalar redshift input."""
    result = lfrd.lf_integrated_number_density(
        0.1,
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
    )

    assert result.shape == ()
    assert result == pytest.approx(1.0)


def test_lf_integrated_number_density_uses_luminosity_distance_function() -> None:
    """Tests that the luminosity-distance callable changes the magnitude limit."""
    z = np.array([0.0, 0.5])
    luminosity_distance = linear_luminosity_distance(z)
    m_faint = expected_absolute_magnitude_limit(M_LIM, luminosity_distance)

    result = lfrd.lf_integrated_number_density(
        z,
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=linear_luminosity_distance,
    )

    np.testing.assert_allclose(result, m_faint - M_BRIGHT)


def test_lf_integrated_number_density_applies_k_correction() -> None:
    """Tests that k-corrections shift the absolute magnitude limit."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        k_correction=0.5,
    )

    np.testing.assert_allclose(result, np.array([0.5, 0.5]))


def test_lf_integrated_number_density_applies_evolution_correction() -> None:
    """Tests that evolution corrections shift the absolute magnitude limit."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        evolution_correction=0.5,
    )

    np.testing.assert_allclose(result, np.array([1.5, 1.5]))


def test_lf_integrated_number_density_accepts_array_corrections() -> None:
    """Tests that redshift-dependent corrections are applied elementwise."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        k_correction=np.array([0.0, 0.5]),
        evolution_correction=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result, np.array([1.0, 1.5]))


def test_lf_integrated_number_density_returns_zero_when_limit_is_too_bright() -> None:
    """Tests that the LF integral is zero when the faint limit is too bright."""
    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        constant_lf,
        m_lim=20.0,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_lf_integrated_number_density_rejects_negative_redshift() -> None:
    """Tests that negative redshifts are rejected."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        lfrd.lf_integrated_number_density(
            [-0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
        )


def test_lf_integrated_number_density_rejects_nonfinite_m_lim() -> None:
    """Tests that non-finite apparent magnitude limits are rejected."""
    with pytest.raises(ValueError, match="m_lim must be finite"):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=np.inf,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
        )


def test_lf_integrated_number_density_rejects_nonfinite_m_bright() -> None:
    """Tests that non-finite bright magnitude bounds are rejected."""
    with pytest.raises(ValueError, match="m_bright must be finite"):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=np.nan,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
        )


def test_lf_integrated_number_density_rejects_nonpositive_luminosity_distance() -> None:
    """Tests that luminosity-distance callables must return positive values."""

    def bad_luminosity_distance(z: np.ndarray) -> np.ndarray:
        """Return invalid luminosity distances."""
        return np.zeros_like(z, dtype=float)

    with pytest.raises(ValueError, match="positive"):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=bad_luminosity_distance,
        )


def test_lf_integrated_number_density_rejects_nonfinite_luminosity_distance() -> None:
    """Tests that luminosity-distance callables must return finite values."""

    def bad_luminosity_distance(z: np.ndarray) -> np.ndarray:
        """Return non-finite luminosity distances."""
        return np.full_like(z, np.nan, dtype=float)

    with pytest.raises(ValueError, match="finite"):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=bad_luminosity_distance,
        )


def test_lf_integrated_number_density_rejects_bad_correction_shape() -> None:
    """Tests that corrections must broadcast to the redshift shape."""
    with pytest.raises(
        ValueError,
        match="k_correction must be scalar or broadcastable to the shape of z",
    ):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            k_correction=np.array([0.0, 0.1, 0.2]),
        )


def test_lf_integrated_number_density_rejects_nonfinite_correction() -> None:
    """Tests that corrections must be finite."""
    with pytest.raises(ValueError, match="evolution_correction contains NaN"):
        lfrd.lf_integrated_number_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            evolution_correction=np.array([0.0, np.nan]),
        )


def test_lf_integrated_number_density_calls_integral_with_expected_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that LF redshift density delegates to the generic LF integral."""
    calls = {}

    def fake_integrated_number_density(
        z: np.ndarray,
        lf: object,
        *,
        m_bright: float,
        m_faint: np.ndarray,
        n_m: int,
    ) -> np.ndarray:
        calls["z"] = z
        calls["lf"] = lf
        calls["m_bright"] = m_bright
        calls["m_faint"] = m_faint
        calls["n_m"] = n_m
        return np.array([10.0, 20.0])

    monkeypatch.setattr(
        lfrd,
        "integrated_number_density",
        fake_integrated_number_density,
    )

    result = lfrd.lf_integrated_number_density(
        [0.1, 0.2],
        double_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=123,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
    )

    np.testing.assert_allclose(result, np.array([10.0, 20.0]))
    np.testing.assert_allclose(calls["z"], np.array([0.1, 0.2]))
    assert calls["lf"] is double_lf
    assert calls["m_bright"] == pytest.approx(M_BRIGHT)
    np.testing.assert_allclose(calls["m_faint"], np.array([-4.0, -4.0]))
    assert calls["n_m"] == 123


def test_lf_weighted_redshift_density_returns_unnormalized_density() -> None:
    """Tests unnormalized LF-weighted redshift density."""
    result = lfrd.lf_weighted_redshift_density(
        [0.0, 1.0],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=linear_volume_weight,
        normalize=False,
    )

    np.testing.assert_allclose(result, np.array([1.0, 2.0]))


def test_lf_weighted_redshift_density_normalizes_density() -> None:
    """Tests normalized LF-weighted redshift density."""
    result = lfrd.lf_weighted_redshift_density(
        [0.0, 1.0],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=linear_volume_weight,
        normalize=True,
    )

    np.testing.assert_allclose(result, np.array([2.0 / 3.0, 4.0 / 3.0]))
    assert np.trapezoid(result, x=np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_lf_weighted_redshift_density_normalizes_on_multi_point_grid() -> None:
    """Tests normalization on a multi-point redshift grid."""
    z = np.array([0.0, 0.5, 1.0])

    result = lfrd.lf_weighted_redshift_density(
        z,
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=linear_volume_weight,
        normalize=True,
    )

    assert np.trapezoid(result, x=z) == pytest.approx(1.0)


def test_lf_weighted_redshift_density_accepts_scalar_redshift_without_normalizing() -> None:
    """Tests scalar redshift input when normalization is disabled."""
    result = lfrd.lf_weighted_redshift_density(
        0.1,
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=constant_volume_weight,
        normalize=False,
    )

    assert result.shape == ()
    assert result == pytest.approx(1.0)


def test_lf_weighted_redshift_density_rejects_scalar_redshift_when_normalizing() -> None:
    """Tests that scalar redshift input cannot be normalized."""
    with pytest.raises(ValueError, match="at least one dimensional"):
        lfrd.lf_weighted_redshift_density(
            0.1,
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            n_m=64,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=constant_volume_weight,
            normalize=True,
        )


def test_lf_weighted_redshift_density_rejects_negative_redshift() -> None:
    """Tests that weighted redshift density rejects negative redshifts."""
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        lfrd.lf_weighted_redshift_density(
            [-0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=constant_volume_weight,
        )


def test_lf_weighted_redshift_density_rejects_negative_volume_weight() -> None:
    """Tests that volume weights must be non-negative."""

    def bad_volume_weight(z: np.ndarray) -> np.ndarray:
        """Return negative volume weights."""
        return -np.ones_like(z, dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        lfrd.lf_weighted_redshift_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=bad_volume_weight,
            normalize=False,
        )


def test_lf_weighted_redshift_density_rejects_nonfinite_volume_weight() -> None:
    """Tests that volume weights must be finite."""

    def bad_volume_weight(z: np.ndarray) -> np.ndarray:
        """Return non-finite volume weights."""
        return np.full_like(z, np.nan, dtype=float)

    with pytest.raises(ValueError, match="finite"):
        lfrd.lf_weighted_redshift_density(
            [0.1, 0.2],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=bad_volume_weight,
            normalize=False,
        )


def test_lf_weighted_redshift_density_rejects_zero_normalization() -> None:
    """Tests that normalization rejects zero-integral redshift densities."""
    with pytest.raises(
        ValueError,
        match="Cannot normalize LF-weighted redshift density",
    ):
        lfrd.lf_weighted_redshift_density(
            [0.0, 1.0],
            zero_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            n_m=64,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=constant_volume_weight,
            normalize=True,
        )


def test_lf_weighted_redshift_density_forwards_corrections_to_lf_density() -> None:
    """Tests that weighted redshift density forwards magnitude corrections."""
    result = lfrd.lf_weighted_redshift_density(
        [0.0, 1.0],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        n_m=64,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=constant_volume_weight,
        k_correction=np.array([0.0, 0.5]),
        evolution_correction=np.array([0.0, 1.0]),
        normalize=False,
    )

    np.testing.assert_allclose(result, np.array([1.0, 1.5]))


def test_lf_weighted_redshift_density_multiplies_lf_density_by_volume_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that LF density is multiplied by the volume-weight callable."""
    monkeypatch.setattr(
        lfrd,
        "lf_integrated_number_density",
        lambda *args, **kwargs: np.array([2.0, 3.0]),
    )

    result = lfrd.lf_weighted_redshift_density(
        [0.0, 1.0],
        constant_lf,
        m_lim=M_LIM,
        m_bright=M_BRIGHT,
        luminosity_distance_mpc_fn=constant_luminosity_distance,
        volume_weight_fn=linear_volume_weight,
        normalize=False,
    )

    np.testing.assert_allclose(result, np.array([2.0, 6.0]))


def test_optional_correction_array_returns_zeros_for_none() -> None:
    """Tests that missing corrections are converted to zeros."""
    result = lfrd._optional_correction_array(
        None,
        np.array([0.1, 0.2]),
        name="k_correction",
    )

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_optional_correction_array_broadcasts_scalar() -> None:
    """Tests that scalar corrections are broadcast to redshift shape."""
    result = lfrd._optional_correction_array(
        0.5,
        np.array([0.1, 0.2]),
        name="k_correction",
    )

    np.testing.assert_allclose(result, np.array([0.5, 0.5]))


def test_optional_correction_array_accepts_matching_array() -> None:
    """Tests that matching correction arrays are preserved."""
    result = lfrd._optional_correction_array(
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
        name="k_correction",
    )

    np.testing.assert_allclose(result, np.array([0.1, 0.2]))


def test_optional_correction_array_rejects_unbroadcastable_array() -> None:
    """Tests that correction arrays must broadcast to redshift shape."""
    with pytest.raises(
        ValueError,
        match="k_correction must be scalar or broadcastable to the shape of z",
    ):
        lfrd._optional_correction_array(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2]),
            name="k_correction",
        )


def test_api_aliases_cover_public_exports() -> None:
    """Tests that public redshift-density functions are included in API aliases."""
    missing_aliases = set(lfrd.__all__) - set(lfrd.__api_aliases__)
    assert missing_aliases == set()


def test_lf_integrated_number_density_rejects_nonfinite_redshift() -> None:
    """Tests that non-finite redshifts are rejected."""
    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        lfrd.lf_integrated_number_density(
            [0.1, np.nan],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
        )


def test_lf_weighted_redshift_density_rejects_nonfinite_redshift() -> None:
    """Tests that weighted redshift density rejects non-finite redshifts."""
    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        lfrd.lf_weighted_redshift_density(
            [0.1, np.inf],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=constant_volume_weight,
            normalize=False,
        )


def test_optional_correction_array_accepts_column_broadcast() -> None:
    """Tests that correction arrays broadcast to multidimensional redshift grids."""
    z = np.array([[0.1, 0.2], [0.3, 0.4]])

    result = lfrd._optional_correction_array(
        np.array([[0.5], [1.0]]),
        z,
        name="k_correction",
    )

    expected = np.array([[0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(result, expected)


def test_lf_weighted_redshift_density_rejects_zero_volume_weight_normalization() -> None:
    """Tests that normalization rejects zero volume-weight density."""

    def zero_volume_weight(z: np.ndarray) -> np.ndarray:
        return np.zeros_like(z, dtype=float)

    with pytest.raises(
        ValueError,
        match="Cannot normalize LF-weighted redshift density",
    ):
        lfrd.lf_weighted_redshift_density(
            [0.0, 1.0],
            constant_lf,
            m_lim=M_LIM,
            m_bright=M_BRIGHT,
            luminosity_distance_mpc_fn=constant_luminosity_distance,
            volume_weight_fn=zero_volume_weight,
            normalize=True,
        )
