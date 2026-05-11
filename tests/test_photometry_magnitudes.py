"""Unit tests for ``lfkit.photometry.magnitudes.py``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.photometry import magnitudes


def test_total_magnitude_correction_none_inputs():
    """Tests that total_magnitude_correction returns zero when no corrections are given."""
    out = magnitudes.total_magnitude_correction()
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert np.allclose(out, 0.0)


def test_total_magnitude_correction_k_only_scalar():
    """Tests that total_magnitude_correction returns only the k-correction for scalar input."""
    out = magnitudes.total_magnitude_correction(k_correction=0.3)
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert np.allclose(out, 0.3)


def test_total_magnitude_correction_e_only_scalar():
    """Tests that total_magnitude_correction returns minus the e-correction for scalar input."""
    out = magnitudes.total_magnitude_correction(e_correction=0.2)
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert np.allclose(out, -0.2)


def test_total_magnitude_correction_adds_array_terms():
    """Tests that total_magnitude_correction combines k- and e-corrections elementwise."""
    k_corr = np.array([0.1, 0.2, 0.3])
    e_corr = np.array([0.05, 0.10, 0.15])

    out = magnitudes.total_magnitude_correction(
        k_correction=k_corr,
        e_correction=e_corr,
    )

    expected = np.array([0.05, 0.10, 0.15])
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert np.allclose(out, expected)


def test_total_magnitude_correction_broadcasts_scalar_and_array():
    """Tests that total_magnitude_correction broadcasts scalar and array inputs correctly."""
    out = magnitudes.total_magnitude_correction(
        k_correction=0.2,
        e_correction=np.array([0.0, 0.1, 0.2]),
    )
    expected = np.array([0.2, 0.1, 0.0])
    assert np.allclose(out, expected)


def test_absolute_magnitude_without_corrections(monkeypatch: pytest.MonkeyPatch):
    """Tests that absolute_magnitude applies M = m - mu without extra corrections."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert out.dtype == float
    assert np.allclose(out, expected)


def test_absolute_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch):
    """Tests that absolute_magnitude applies M = m - mu - K + E."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert np.allclose(out, expected)


def test_apparent_magnitude_without_corrections(monkeypatch: pytest.MonkeyPatch):
    """Tests that apparent_magnitude applies m = M + mu without extra corrections."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert out.dtype == float
    assert np.allclose(out, expected)


def test_apparent_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch):
    """Tests that apparent_magnitude applies m = M + mu + K - E."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert np.allclose(out, expected)


def test_apparent_and_absolute_are_inverse_without_corrections(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that apparent_magnitude and absolute_magnitude invert each other without corrections."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
        _, _, _ = cosmo_obj, z, h
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

    assert np.allclose(recovered, apparent_mag)


def test_apparent_and_absolute_are_inverse_with_corrections(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that apparent_magnitude and absolute_magnitude invert each other with corrections."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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

    assert np.allclose(recovered, apparent_mag)


def test_absolute_magnitude_passes_h_to_distance_modulus(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that absolute_magnitude forwards h to distance_modulus."""
    captured: dict[str, float | None] = {"h": None}

    def mock_distance_modulus(cosmo_obj, z, h=None):
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

    assert captured["h"] == 0.7


def test_apparent_magnitude_passes_h_to_distance_modulus(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that apparent_magnitude forwards h to distance_modulus."""
    captured: dict[str, float | None] = {"h": None}

    def mock_distance_modulus(cosmo_obj, z, h=None):
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

    assert captured["h"] == 0.7


def test_absolute_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that absolute_magnitude broadcasts scalar corrections over array magnitudes."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert np.allclose(out, expected)


def test_apparent_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests that apparent_magnitude broadcasts scalar corrections over array magnitudes."""
    def mock_distance_modulus(cosmo_obj, z, h=None):
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
    assert np.allclose(out, expected)
