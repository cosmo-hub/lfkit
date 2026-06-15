"""Unit tests for ``lfkit.photometry.magnitudes``."""

from __future__ import annotations

import numpy as np
import pytest

import lfkit.photometry.magnitudes as magnitudes
from lfkit.photometry.magnitudes import (
    absolute_magnitude_from_luminosity_distance,
    apparent_magnitude_from_luminosity_distance,
    total_magnitude_correction,
)


def test_total_magnitude_correction_none_inputs() -> None:
    """Tests that total_magnitude_correction returns zero when no corrections are given."""
    out = total_magnitude_correction()

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(0.0)


def test_total_magnitude_correction_k_only_scalar() -> None:
    """Tests that total_magnitude_correction returns only the k-correction."""
    out = total_magnitude_correction(k_correction=0.3)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(0.3)


def test_total_magnitude_correction_e_only_scalar() -> None:
    """Tests that total_magnitude_correction returns minus the e-correction."""
    out = total_magnitude_correction(e_correction=0.2)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(-0.2)


def test_total_magnitude_correction_adds_array_terms() -> None:
    """Tests that total_magnitude_correction combines K and E corrections."""
    k_corr = np.array([0.1, 0.2, 0.3])
    e_corr = np.array([0.05, 0.10, 0.15])

    out = total_magnitude_correction(
        k_correction=k_corr,
        e_correction=e_corr,
    )

    expected = np.array([0.05, 0.10, 0.15])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_total_magnitude_correction_broadcasts_scalar_and_array() -> None:
    """Tests that total_magnitude_correction broadcasts scalar and array inputs."""
    out = total_magnitude_correction(
        k_correction=0.2,
        e_correction=np.array([0.0, 0.1, 0.2]),
    )

    expected = np.array([0.2, 0.1, 0.0])
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_without_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that absolute_magnitude applies M = m - mu."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        apparent_mag=np.array([20.0, 21.5]),
    )

    expected = np.array([-20.0, -19.5])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that absolute_magnitude applies M = m - mu - K + E."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        apparent_mag=np.array([20.0, 21.5]),
        k_correction=np.array([0.1, 0.2]),
        e_correction=np.array([0.3, 0.4]),
    )

    expected = (
        np.array([20.0, 21.5])
        - np.array([40.0, 41.0])
        - np.array([0.1, 0.2])
        + np.array([0.3, 0.4])
    )
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_without_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that apparent_magnitude applies m = M + mu."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        absolute_mag=np.array([-20.0, -19.5]),
    )

    expected = np.array([20.0, 21.5])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that apparent_magnitude applies m = M + mu + K - E."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        absolute_mag=np.array([-20.0, -19.5]),
        k_correction=np.array([0.1, 0.2]),
        e_correction=np.array([0.3, 0.4]),
    )

    expected = (
        np.array([-20.0, -19.5])
        + np.array([40.0, 41.0])
        + np.array([0.1, 0.2])
        - np.array([0.3, 0.4])
    )
    np.testing.assert_allclose(out, expected)


def test_apparent_and_absolute_are_inverse_without_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude and absolute_magnitude invert each other."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-dependent distance modulus."""
        _, _ = cosmo_obj, h
        z_arr = np.asarray(z, dtype=float)
        return 40.0 + z_arr

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    apparent_mag = np.array([20.0, 21.0, 22.0])
    z = np.array([0.1, 0.2, 0.3])

    absolute_mag = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=z,
        apparent_mag=apparent_mag,
    )
    recovered = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=z,
        absolute_mag=absolute_mag,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_apparent_and_absolute_are_inverse_with_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that magnitude conversions invert each other with corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-dependent distance modulus."""
        _, _ = cosmo_obj, h
        z_arr = np.asarray(z, dtype=float)
        return 39.5 + 2.0 * z_arr

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    apparent_mag = np.array([20.0, 21.0, 22.0])
    z = np.array([0.1, 0.2, 0.3])
    k_corr = np.array([0.05, 0.10, 0.15])
    e_corr = np.array([0.20, 0.10, 0.00])

    absolute_mag = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=z,
        apparent_mag=apparent_mag,
        k_correction=k_corr,
        e_correction=e_corr,
    )
    recovered = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=z,
        absolute_mag=absolute_mag,
        k_correction=k_corr,
        e_correction=e_corr,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_absolute_magnitude_passes_h_to_distance_modulus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude forwards h to distance_modulus."""
    captured: dict[str, float | None] = {"h": None}

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Capture h and return a fixed distance modulus."""
        _, _ = cosmo_obj, z
        captured["h"] = h
        return np.array([40.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    _ = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1]),
        apparent_mag=np.array([20.0]),
        h=0.7,
    )

    assert captured["h"] == pytest.approx(0.7)


def test_apparent_magnitude_passes_h_to_distance_modulus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude forwards h to distance_modulus."""
    captured: dict[str, float | None] = {"h": None}

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Capture h and return a fixed distance modulus."""
        _, _ = cosmo_obj, z
        captured["h"] = h
        return np.array([40.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    _ = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1]),
        absolute_mag=np.array([-20.0]),
        h=0.7,
    )

    assert captured["h"] == pytest.approx(0.7)


def test_absolute_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude broadcasts scalar corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus array."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0, 42.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        apparent_mag=np.array([20.0, 21.0, 22.0]),
        k_correction=0.2,
    )

    expected = np.array([20.0, 21.0, 22.0]) - np.array([40.0, 41.0, 42.0]) - 0.2
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude broadcasts scalar corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus array."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0, 42.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        absolute_mag=np.array([-20.0, -20.0, -20.0]),
        e_correction=0.3,
    )

    expected = np.array([-20.0, -20.0, -20.0]) + np.array([40.0, 41.0, 42.0]) - 0.3
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_from_luminosity_distance_scalar() -> None:
    """Tests apparent-to-absolute conversion from scalar luminosity distance."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
    )

    assert result.shape == ()
    assert result == pytest.approx(-4.0)


def test_apparent_magnitude_from_luminosity_distance_scalar() -> None:
    """Tests absolute-to-apparent conversion from scalar luminosity distance."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.0,
        10.0,
    )

    assert result.shape == ()
    assert result == pytest.approx(26.0)


def test_luminosity_distance_magnitude_conversions_are_inverse() -> None:
    """Tests that luminosity-distance magnitude conversions invert each other."""
    apparent_mag = np.array([24.0, 25.0, 26.0])
    luminosity_distance = np.array([10.0, 20.0, 30.0])

    absolute_mag = absolute_magnitude_from_luminosity_distance(
        apparent_mag,
        luminosity_distance,
    )
    recovered = apparent_magnitude_from_luminosity_distance(
        absolute_mag,
        luminosity_distance,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_luminosity_distance_magnitude_conversions_are_inverse_with_corrections() -> None:
    """Tests luminosity-distance conversions invert each other with corrections."""
    apparent_mag = np.array([24.0, 25.0, 26.0])
    luminosity_distance = np.array([10.0, 20.0, 30.0])
    k_corr = np.array([0.0, 0.2, 0.4])
    e_corr = np.array([0.1, 0.1, 0.2])

    absolute_mag = absolute_magnitude_from_luminosity_distance(
        apparent_mag,
        luminosity_distance,
        k_correction=k_corr,
        e_correction=e_corr,
    )
    recovered = apparent_magnitude_from_luminosity_distance(
        absolute_mag,
        luminosity_distance,
        k_correction=k_corr,
        e_correction=e_corr,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_absolute_magnitude_from_luminosity_distance_with_corrections() -> None:
    """Tests apparent-to-absolute conversion with K and evolution corrections."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
        k_correction=0.5,
        e_correction=0.2,
    )

    assert result == pytest.approx(-4.3)


def test_apparent_magnitude_from_luminosity_distance_with_corrections() -> None:
    """Tests absolute-to-apparent conversion with K and evolution corrections."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.3,
        10.0,
        k_correction=0.5,
        e_correction=0.2,
    )

    assert result == pytest.approx(26.0)


def test_absolute_magnitude_from_luminosity_distance_broadcasts_arrays() -> None:
    """Tests apparent-to-absolute conversion with broadcastable inputs."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        np.array([10.0, 100.0]),
    )

    np.testing.assert_allclose(result, np.array([-4.0, -9.0]))


def test_apparent_magnitude_from_luminosity_distance_broadcasts_arrays() -> None:
    """Tests absolute-to-apparent conversion with broadcastable inputs."""
    result = apparent_magnitude_from_luminosity_distance(
        np.array([-4.0, -9.0]),
        np.array([10.0, 100.0]),
    )

    np.testing.assert_allclose(result, np.array([26.0, 26.0]))


def test_absolute_magnitude_from_luminosity_distance_broadcasts_corrections() -> None:
    """Tests apparent-to-absolute conversion with array corrections."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
        k_correction=np.array([0.0, 0.5]),
        e_correction=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result, np.array([-4.0, -3.5]))


def test_apparent_magnitude_from_luminosity_distance_broadcasts_corrections() -> None:
    """Tests absolute-to-apparent conversion with array corrections."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.0,
        10.0,
        k_correction=np.array([0.0, 0.5]),
        e_correction=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result, np.array([26.0, 25.5]))


def test_absolute_magnitude_from_luminosity_distance_rejects_zero_distance() -> None:
    """Tests that apparent-to-absolute conversion rejects zero distance."""
    with pytest.raises(ValueError, match="positive values"):
        absolute_magnitude_from_luminosity_distance(
            26.0,
            0.0,
        )


def test_apparent_magnitude_from_luminosity_distance_rejects_zero_distance() -> None:
    """Tests that absolute-to-apparent conversion rejects zero distance."""
    with pytest.raises(ValueError, match="positive values"):
        apparent_magnitude_from_luminosity_distance(
            -4.0,
            0.0,
        )


def test_absolute_magnitude_from_luminosity_distance_rejects_negative_distance() -> None:
    """Tests that apparent-to-absolute conversion rejects negative distance."""
    with pytest.raises(ValueError, match="negative values"):
        absolute_magnitude_from_luminosity_distance(
            26.0,
            -10.0,
        )


def test_apparent_magnitude_from_luminosity_distance_rejects_negative_distance() -> None:
    """Tests that absolute-to-apparent conversion rejects negative distance."""
    with pytest.raises(ValueError, match="negative values"):
        apparent_magnitude_from_luminosity_distance(
            -4.0,
            -10.0,
        )


def test_absolute_magnitude_from_luminosity_distance_rejects_nonfinite_distance() -> None:
    """Tests that apparent-to-absolute conversion rejects non-finite distance."""
    with pytest.raises(ValueError, match="luminosity_distance_mpc contains NaN"):
        absolute_magnitude_from_luminosity_distance(
            26.0,
            np.nan,
        )


def test_apparent_magnitude_from_luminosity_distance_rejects_nonfinite_distance() -> None:
    """Tests that absolute-to-apparent conversion rejects non-finite distance."""
    with pytest.raises(ValueError, match="luminosity_distance_mpc contains NaN"):
        apparent_magnitude_from_luminosity_distance(
            -4.0,
            np.nan,
        )


def test_total_magnitude_correction_preserves_broadcast_shape() -> None:
    """Tests that total_magnitude_correction returns the broadcasted output shape."""
    out = total_magnitude_correction(
        k_correction=np.array([[0.1], [0.2]]),
        e_correction=np.array([0.0, 0.1, 0.2]),
    )

    expected = np.array(
        [
            [0.1, 0.0, -0.1],
            [0.2, 0.1, 0.0],
        ],
    )
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_from_luminosity_distance_returns_float64_array() -> None:
    """Tests that apparent-to-absolute conversion returns a float64 array."""
    out = absolute_magnitude_from_luminosity_distance(
        np.array([26, 27]),
        np.array([10, 100]),
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_apparent_magnitude_from_luminosity_distance_returns_float64_array() -> None:
    """Tests that absolute-to-apparent conversion returns a float64 array."""
    out = apparent_magnitude_from_luminosity_distance(
        np.array([-4, -8]),
        np.array([10, 100]),
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_luminosity_distance_formula_matches_known_distance_modulus() -> None:
    """Tests that luminosity-distance conversions use mu = 5 log10(dL) + 25."""
    result = absolute_magnitude_from_luminosity_distance(
        apparent_mag=30.0,
        luminosity_distance_mpc=100.0,
    )

    assert result == pytest.approx(-5.0)


def test_apparent_magnitude_from_luminosity_distance_matches_known_distance_modulus() -> None:
    """Tests that absolute-to-apparent conversion uses the expected distance modulus."""
    result = apparent_magnitude_from_luminosity_distance(
        absolute_mag=0.0,
        luminosity_distance_mpc=100.0,
    )

    assert result == pytest.approx(35.0)


def test_absolute_magnitude_supports_scalar_magnitude_with_array_redshift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude broadcasts scalar magnitude over redshift."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-shaped distance modulus."""
        _, _ = cosmo_obj, h
        return np.asarray([40.0, 41.0, 42.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        apparent_mag=22.0,
    )

    np.testing.assert_allclose(out, np.array([-18.0, -19.0, -20.0]))


def test_apparent_magnitude_supports_scalar_magnitude_with_array_redshift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude broadcasts scalar magnitude over redshift."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-shaped distance modulus."""
        _, _ = cosmo_obj, h
        return np.asarray([40.0, 41.0, 42.0], dtype=float)

    monkeypatch.setattr(magnitudes, "distance_modulus", mock_distance_modulus)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        absolute_mag=-20.0,
    )

    np.testing.assert_allclose(out, np.array([20.0, 21.0, 22.0]))


def test_magnitudes_exports_expected_public_names() -> None:
    """Tests that magnitudes exposes the expected public API names."""
    expected = {
        "total_magnitude_correction",
        "absolute_magnitude",
        "absolute_magnitude_from_luminosity_distance",
        "apparent_magnitude",
        "apparent_magnitude_from_luminosity_distance",
    }

    assert set(magnitudes.__all__) == expected


def test_magnitudes_api_aliases_match_public_functions() -> None:
    """Tests that magnitudes API aliases point to public functions."""
    for public_name in magnitudes.__api_aliases__:
        assert public_name in magnitudes.__all__
        assert callable(getattr(magnitudes, public_name))
