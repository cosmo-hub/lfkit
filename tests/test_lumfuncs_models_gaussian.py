"""Unit tests for ``lfkit.luminosity_functions.models.gaussian``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.gaussian import gaussian_lf, lognormal_lf


def test_gaussian_lf_matches_manual_formula() -> None:
    """Tests that gaussian_lf matches the analytic Gaussian formula."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    mean = -21.0
    sigma = 0.5
    amplitude = 2.0

    result = gaussian_lf(
        absolute_mag,
        mean_absolute_mag=mean,
        sigma_absolute_mag=sigma,
        amplitude=amplitude,
    )

    expected = (
        amplitude
        / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-0.5 * ((absolute_mag - mean) / sigma) ** 2.0)
    )
    np.testing.assert_allclose(result, expected)


def test_gaussian_lf_peak_value_matches_normalization() -> None:
    """Tests that gaussian_lf has the expected value at the mean."""
    result = gaussian_lf(
        -21.0,
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=0.5,
        amplitude=2.0,
    )

    expected = 2.0 / (np.sqrt(2.0 * np.pi) * 0.5)
    np.testing.assert_allclose(result, expected)


def test_gaussian_lf_is_symmetric_about_mean() -> None:
    """Tests that gaussian_lf is symmetric in magnitude around the mean."""
    result = gaussian_lf(
        np.array([-22.0, -20.0]),
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=0.5,
        amplitude=1.0,
    )

    np.testing.assert_allclose(result[0], result[1])


def test_gaussian_lf_accepts_scalar_input() -> None:
    """Tests that gaussian_lf accepts scalar magnitude input."""
    result = gaussian_lf(
        -21.0,
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=1.0,
    )

    assert np.shape(result) == ()
    assert np.isfinite(result)


def test_gaussian_lf_preserves_array_shape() -> None:
    """Tests that gaussian_lf preserves the input magnitude shape."""
    absolute_mag = np.array([[-22.0, -21.0], [-20.0, -19.0]])

    result = gaussian_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=1.0,
    )

    assert result.shape == absolute_mag.shape


def test_gaussian_lf_returns_float_array() -> None:
    """Tests that gaussian_lf returns floating-point values."""
    result = gaussian_lf(
        np.array([-22, -21, -20]),
        mean_absolute_mag=-21,
        sigma_absolute_mag=1,
    )

    assert result.dtype.kind == "f"


def test_gaussian_lf_zero_amplitude_returns_zero() -> None:
    """Tests that zero amplitude returns zero luminosity function values."""
    result = gaussian_lf(
        np.array([-22.0, -21.0, -20.0]),
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=1.0,
        amplitude=0.0,
    )

    np.testing.assert_allclose(result, np.zeros(3))


def test_gaussian_lf_scales_linearly_with_amplitude() -> None:
    """Tests that gaussian_lf scales linearly with amplitude."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])

    result_1 = gaussian_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=1.0,
        amplitude=1.0,
    )
    result_2 = gaussian_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_absolute_mag=1.0,
        amplitude=2.0,
    )

    np.testing.assert_allclose(result_2, 2.0 * result_1)


def test_gaussian_lf_allows_array_parameters() -> None:
    """Tests that gaussian_lf supports array-valued broadcasted parameters."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    mean = np.array([-21.5, -21.0, -20.5])

    result = gaussian_lf(
        absolute_mag,
        mean_absolute_mag=mean,
        sigma_absolute_mag=0.5,
        amplitude=np.array([1.0, 2.0, 3.0]),
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


def test_gaussian_lf_rejects_zero_sigma() -> None:
    """Tests that zero sigma_absolute_mag is rejected."""
    with pytest.raises(ValueError, match="sigma_absolute_mag must be positive"):
        gaussian_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_absolute_mag=0.0,
        )


def test_gaussian_lf_rejects_negative_sigma() -> None:
    """Tests that negative sigma_absolute_mag is rejected."""
    with pytest.raises(ValueError, match="sigma_absolute_mag must be positive"):
        gaussian_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_absolute_mag=-1.0,
        )


def test_gaussian_lf_rejects_negative_amplitude() -> None:
    """Tests that negative amplitude is rejected."""
    with pytest.raises(ValueError, match="amplitude must be non-negative"):
        gaussian_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_absolute_mag=1.0,
            amplitude=-1.0,
        )


@pytest.mark.parametrize(
    ("parameter_name", "kwargs", "match"),
    [
        (
            "absolute_mag",
            {"absolute_mag": np.array([-22.0, np.nan])},
            "absolute_mag contains NaN or infinite values",
        ),
        (
            "mean_absolute_mag",
            {"mean_absolute_mag": np.nan},
            "mean_absolute_mag contains NaN or infinite values",
        ),
        (
            "sigma_absolute_mag",
            {"sigma_absolute_mag": np.inf},
            "sigma_absolute_mag contains NaN or infinite values",
        ),
        (
            "amplitude",
            {"amplitude": np.nan},
            "amplitude contains NaN or infinite values",
        ),
    ],
)
def test_gaussian_lf_rejects_nonfinite_inputs(
    parameter_name: str,
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Tests that gaussian_lf rejects non-finite inputs."""
    params = {
        "absolute_mag": np.array([-22.0, -21.0]),
        "mean_absolute_mag": -21.0,
        "sigma_absolute_mag": 1.0,
        "amplitude": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        gaussian_lf(**params)


def test_lognormal_lf_matches_manual_formula() -> None:
    """Tests that lognormal_lf matches the analytic lognormal-in-luminosity formula."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    mean = -21.0
    sigma_log_luminosity = 0.25
    amplitude = 2.0

    result = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=mean,
        sigma_log_luminosity=sigma_log_luminosity,
        amplitude=amplitude,
    )

    delta_log_luminosity = -0.4 * (absolute_mag - mean)
    expected = (
        amplitude
        * 0.4
        / (np.sqrt(2.0 * np.pi) * sigma_log_luminosity)
        * np.exp(-0.5 * (delta_log_luminosity / sigma_log_luminosity) ** 2.0)
    )
    np.testing.assert_allclose(result, expected)


def test_lognormal_lf_peak_value_matches_normalization() -> None:
    """Tests that lognormal_lf has the expected value at the mean."""
    result = lognormal_lf(
        -21.0,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
        amplitude=2.0,
    )

    expected = 2.0 * 0.4 / (np.sqrt(2.0 * np.pi) * 0.25)
    np.testing.assert_allclose(result, expected)


def test_lognormal_lf_is_symmetric_in_log_luminosity_about_mean() -> None:
    """Tests that lognormal_lf is symmetric around the mean magnitude."""
    result = lognormal_lf(
        np.array([-22.0, -20.0]),
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
        amplitude=1.0,
    )

    np.testing.assert_allclose(result[0], result[1])


def test_lognormal_lf_accepts_scalar_input() -> None:
    """Tests that lognormal_lf accepts scalar magnitude input."""
    result = lognormal_lf(
        -21.0,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
    )

    assert np.shape(result) == ()
    assert np.isfinite(result)


def test_lognormal_lf_preserves_array_shape() -> None:
    """Tests that lognormal_lf preserves the input magnitude shape."""
    absolute_mag = np.array([[-22.0, -21.0], [-20.0, -19.0]])

    result = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
    )

    assert result.shape == absolute_mag.shape


def test_lognormal_lf_returns_float_array() -> None:
    """Tests that lognormal_lf returns floating-point values."""
    result = lognormal_lf(
        np.array([-22, -21, -20]),
        mean_absolute_mag=-21,
        sigma_log_luminosity=1,
    )

    assert result.dtype.kind == "f"


def test_lognormal_lf_zero_amplitude_returns_zero() -> None:
    """Tests that zero amplitude returns zero lognormal luminosity function values."""
    result = lognormal_lf(
        np.array([-22.0, -21.0, -20.0]),
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
        amplitude=0.0,
    )

    np.testing.assert_allclose(result, np.zeros(3))


def test_lognormal_lf_scales_linearly_with_amplitude() -> None:
    """Tests that lognormal_lf scales linearly with amplitude."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])

    result_1 = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
        amplitude=1.0,
    )
    result_2 = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=-21.0,
        sigma_log_luminosity=0.25,
        amplitude=2.0,
    )

    np.testing.assert_allclose(result_2, 2.0 * result_1)


def test_lognormal_lf_allows_array_parameters() -> None:
    """Tests that lognormal_lf supports array-valued broadcasted parameters."""
    absolute_mag = np.array([-22.0, -21.0, -20.0])
    mean = np.array([-21.5, -21.0, -20.5])

    result = lognormal_lf(
        absolute_mag,
        mean_absolute_mag=mean,
        sigma_log_luminosity=0.25,
        amplitude=np.array([1.0, 2.0, 3.0]),
    )

    assert result.shape == absolute_mag.shape
    assert np.all(np.isfinite(result))


def test_lognormal_lf_rejects_zero_sigma() -> None:
    """Tests that zero sigma_log_luminosity is rejected."""
    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive"):
        lognormal_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.0,
        )


def test_lognormal_lf_rejects_negative_sigma() -> None:
    """Tests that negative sigma_log_luminosity is rejected."""
    with pytest.raises(ValueError, match="sigma_log_luminosity must be positive"):
        lognormal_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=-0.25,
        )


def test_lognormal_lf_rejects_negative_amplitude() -> None:
    """Tests that negative amplitude is rejected."""
    with pytest.raises(ValueError, match="amplitude must be non-negative"):
        lognormal_lf(
            np.array([-22.0, -21.0]),
            mean_absolute_mag=-21.0,
            sigma_log_luminosity=0.25,
            amplitude=-1.0,
        )


@pytest.mark.parametrize(
    ("parameter_name", "kwargs", "match"),
    [
        (
            "absolute_mag",
            {"absolute_mag": np.array([-22.0, np.nan])},
            "absolute_mag contains NaN or infinite values",
        ),
        (
            "mean_absolute_mag",
            {"mean_absolute_mag": np.nan},
            "mean_absolute_mag contains NaN or infinite values",
        ),
        (
            "sigma_log_luminosity",
            {"sigma_log_luminosity": np.inf},
            "sigma_log_luminosity contains NaN or infinite values",
        ),
        (
            "amplitude",
            {"amplitude": np.nan},
            "amplitude contains NaN or infinite values",
        ),
    ],
)
def test_lognormal_lf_rejects_nonfinite_inputs(
    parameter_name: str,
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Tests that lognormal_lf rejects non-finite inputs."""
    params = {
        "absolute_mag": np.array([-22.0, -21.0]),
        "mean_absolute_mag": -21.0,
        "sigma_log_luminosity": 0.25,
        "amplitude": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        lognormal_lf(**params)
