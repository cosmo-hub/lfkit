"""Unit tests for ``lfkit.photometry.lf_parameter_models.py``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.luminosity_functions.models.parameter_models import (
    ALPHA_MODELS,
    M_STAR_MODELS,
    PHI_STAR_MODELS,
    alpha_constant,
    alpha_linear,
    evaluate_lf_parameters,
    get_parameter_model,
    m_star_constant,
    m_star_linear_q,
    phi_star_constant,
    phi_star_linear_p,
    available_lf_parameter_models,
    register_alpha_model,
    register_m_star_model,
    register_phi_star_model,
)


def test_phi_star_constant_returns_constant_array() -> None:
    """Tests that phi_star_constant returns the same value at all redshifts."""
    z = np.array([0.0, 0.5, 1.0])
    result = phi_star_constant(z, phi_star=1.2e-3)
    expected = np.array([1.2e-3, 1.2e-3, 1.2e-3])

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_phi_star_constant_preserves_input_shape() -> None:
    """Tests that phi_star_constant preserves the shape of the input redshift array."""
    z = np.array([[0.0, 0.2], [0.4, 0.6]])
    result = phi_star_constant(z, phi_star=2.5e-3)

    assert result.shape == z.shape
    np.testing.assert_allclose(result, np.full_like(z, 2.5e-3, dtype=float))


def test_phi_star_linear_p_matches_expected_formula() -> None:
    """Tests that phi_star_linear_p matches the density-evolution formula."""
    z = np.array([0.0, 0.5, 1.0])
    phi_0_star = 1.0e-3
    p = 2.0

    result = phi_star_linear_p(z, phi_0_star=phi_0_star, p=p)
    expected = phi_0_star * 10.0 ** (0.4 * p * z)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_m_star_constant_returns_constant_array() -> None:
    """Tests that m_star_constant returns the same value at all redshifts."""
    z = np.array([0.0, 0.3, 0.8])
    result = m_star_constant(z, m_star=-20.5)
    expected = np.array([-20.5, -20.5, -20.5])

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_m_star_linear_q_matches_expected_formula() -> None:
    """Tests that m_star_linear_q matches the luminosity-evolution formula."""
    z = np.array([0.0, 0.1, 0.5, 1.0])
    m_0_star = -21.0
    q = 1.3
    z_ref = 0.1

    result = m_star_linear_q(z, m_0_star=m_0_star, q=q, z_ref=z_ref)
    expected = m_0_star - q * (z - z_ref)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_m_star_linear_q_uses_default_reference_redshift() -> None:
    """Tests that m_star_linear_q uses the default reference redshift when omitted."""
    z = np.array([0.1, 0.6])
    result = m_star_linear_q(z, m_0_star=-20.0, q=1.0)
    expected = np.array([-20.0, -20.5])

    np.testing.assert_allclose(result, expected)


def test_alpha_constant_returns_constant_array() -> None:
    """Tests that alpha_constant returns the same value at all redshifts."""
    z = np.array([0.0, 0.5, 1.0])
    result = alpha_constant(z, alpha=-1.25)
    expected = np.array([-1.25, -1.25, -1.25])

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_alpha_linear_matches_expected_formula() -> None:
    """Tests that alpha_linear matches the linear redshift-evolution formula."""
    z = np.array([0.0, 0.1, 0.5, 1.0])
    alpha_0 = -1.3
    alpha_1 = 0.2
    z_ref = 0.1

    result = alpha_linear(z, alpha_0=alpha_0, alpha_1=alpha_1, z_ref=z_ref)
    expected = alpha_0 + alpha_1 * (z - z_ref)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_get_parameter_model_returns_registered_model() -> None:
    """Tests that get_parameter_model returns the correct registered callable."""
    model = get_parameter_model(
        "constant",
        PHI_STAR_MODELS,
        model_kind="phi_model",
    )

    assert model is phi_star_constant


def test_get_parameter_model_raises_for_unknown_model() -> None:
    """Tests that get_parameter_model raises a helpful error for unknown models."""
    with pytest.raises(ValueError, match="Unknown phi_model 'bad_model'"):
        get_parameter_model(
            "bad_model",
            PHI_STAR_MODELS,
            model_kind="phi_model",
        )


def test_get_parameter_model_lists_available_models_in_error() -> None:
    """Tests that get_parameter_model includes available model names in the error."""
    with pytest.raises(ValueError) as exc_info:
        get_parameter_model(
            "not_real",
            M_STAR_MODELS,
            model_kind="m_star_model",
        )

    message = str(exc_info.value)
    assert "constant" in message
    assert "linear_q" in message


def test_evaluate_lf_parameters_with_constant_models() -> None:
    """Tests that evaluate_lf_parameters evaluates constant models correctly."""
    z = np.array([0.0, 0.4, 0.9])

    phi_star, m_star, alpha = evaluate_lf_parameters(
        z,
        phi_model="constant",
        phi_kwargs={"phi_star": 1.5e-3},
        m_star_model="constant",
        m_star_kwargs={"m_star": -20.7},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.2},
    )

    np.testing.assert_allclose(phi_star, np.array([1.5e-3, 1.5e-3, 1.5e-3]))
    np.testing.assert_allclose(m_star, np.array([-20.7, -20.7, -20.7]))
    np.testing.assert_allclose(alpha, np.array([-1.2, -1.2, -1.2]))


def test_evaluate_lf_parameters_with_mixed_models() -> None:
    """Tests that evaluate_lf_parameters evaluates mixed evolution models correctly."""
    z = np.array([0.1, 0.6, 1.1])

    phi_star, m_star, alpha = evaluate_lf_parameters(
        z,
        phi_model="linear_p",
        phi_kwargs={"phi_0_star": 2.0e-3, "p": 1.0},
        m_star_model="linear_q",
        m_star_kwargs={"m_0_star": -21.0, "q": 0.8, "z_ref": 0.1},
        alpha_model="linear",
        alpha_kwargs={"alpha_0": -1.1, "alpha_1": -0.05, "z_ref": 0.1},
    )

    expected_phi = 2.0e-3 * 10.0 ** (0.4 * 1.0 * z)
    expected_m_star = -21.0 - 0.8 * (z - 0.1)
    expected_alpha = -1.1 + (-0.05) * (z - 0.1)

    np.testing.assert_allclose(phi_star, expected_phi)
    np.testing.assert_allclose(m_star, expected_m_star)
    np.testing.assert_allclose(alpha, expected_alpha)


def test_evaluate_lf_parameters_accepts_none_kwargs_for_selected_models() -> None:
    """Tests that evaluate_lf_parameters handles None kwargs when models need none beyond provided defaults."""
    z = np.array([0.0, 0.5])

    phi_star, m_star, alpha = evaluate_lf_parameters(
        z,
        phi_model="constant",
        phi_kwargs={"phi_star": 1.0e-3},
        m_star_model="linear_q",
        m_star_kwargs={"m_0_star": -20.5, "q": 1.0},
        alpha_model="constant",
        alpha_kwargs={"alpha": -1.3},
    )

    np.testing.assert_allclose(phi_star, np.array([1.0e-3, 1.0e-3]))
    np.testing.assert_allclose(m_star, np.array([-20.4, -20.9]))
    np.testing.assert_allclose(alpha, np.array([-1.3, -1.3]))


def test_evaluate_lf_parameters_raises_for_unknown_phi_model() -> None:
    """Tests that evaluate_lf_parameters raises for an unknown phi evolution model."""
    z = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="Unknown phi_model 'wrong'"):
        evaluate_lf_parameters(
            z,
            phi_model="wrong",
            phi_kwargs={"phi_star": 1.0e-3},
            m_star_model="constant",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="constant",
            alpha_kwargs={"alpha": -1.0},
        )


def test_evaluate_lf_parameters_raises_for_unknown_m_star_model() -> None:
    """Tests that evaluate_lf_parameters raises for an unknown m_star evolution model."""
    z = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="Unknown m_star_model 'wrong'"):
        evaluate_lf_parameters(
            z,
            phi_model="constant",
            phi_kwargs={"phi_star": 1.0e-3},
            m_star_model="wrong",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="constant",
            alpha_kwargs={"alpha": -1.0},
        )


def test_evaluate_lf_parameters_raises_for_unknown_alpha_model() -> None:
    """Tests that evaluate_lf_parameters raises for an unknown alpha evolution model."""
    z = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="Unknown alpha_model 'wrong'"):
        evaluate_lf_parameters(
            z,
            phi_model="constant",
            phi_kwargs={"phi_star": 1.0e-3},
            m_star_model="constant",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="wrong",
            alpha_kwargs={"alpha": -1.0},
        )


def test_model_registries_contain_expected_keys() -> None:
    """Tests that the model registries contain the expected built-in model names."""
    assert set(PHI_STAR_MODELS) == {"constant", "linear_p"}
    assert set(M_STAR_MODELS) == {"constant", "linear_q"}
    assert set(ALPHA_MODELS) == {"constant", "linear"}


def test_scalar_input_returns_numpy_array() -> None:
    """Tests that scalar redshift input still returns NumPy arrays."""
    phi = phi_star_constant(0.5, phi_star=1.0e-3)
    m_star = m_star_constant(0.5, m_star=-20.0)
    alpha = alpha_constant(0.5, alpha=-1.1)

    assert isinstance(phi, np.ndarray)
    assert isinstance(m_star, np.ndarray)
    assert isinstance(alpha, np.ndarray)
    assert phi.dtype == np.float64
    assert m_star.dtype == np.float64
    assert alpha.dtype == np.float64


def test_available_lf_parameter_models_returns_sorted_builtin_names() -> None:
    """Tests that available_lf_parameter_models returns sorted built-in model names."""
    result = available_lf_parameter_models()

    assert result["phi_star"] == ["constant", "linear_p"]
    assert result["m_star"] == ["constant", "linear_q"]
    assert result["alpha"] == ["constant", "linear"]


def test_register_phi_star_model_adds_custom_model() -> None:
    """Tests that register_phi_star_model adds a custom phi_star model."""
    def custom_phi(z: np.ndarray, *, amplitude: float) -> np.ndarray:
        return np.full_like(z, amplitude, dtype=float)

    register_phi_star_model("custom_phi_test", custom_phi)

    try:
        z = np.array([0.0, 0.5, 1.0])
        model = get_parameter_model(
            "custom_phi_test",
            PHI_STAR_MODELS,
            model_kind="phi_model",
        )
        result = model(z, amplitude=3.0e-3)

        np.testing.assert_allclose(result, np.array([3.0e-3, 3.0e-3, 3.0e-3]))
    finally:
        PHI_STAR_MODELS.pop("custom_phi_test", None)


def test_register_m_star_model_adds_custom_model() -> None:
    """Tests that register_m_star_model adds a custom M_star model."""
    def custom_m_star(z: np.ndarray, *, base: float, slope: float) -> np.ndarray:
        return np.asarray(base + slope * z, dtype=float)

    register_m_star_model("custom_m_star_test", custom_m_star)

    try:
        z = np.array([0.0, 0.5, 1.0])
        model = get_parameter_model(
            "custom_m_star_test",
            M_STAR_MODELS,
            model_kind="m_star_model",
        )
        result = model(z, base=-20.0, slope=-1.0)

        np.testing.assert_allclose(result, np.array([-20.0, -20.5, -21.0]))
    finally:
        M_STAR_MODELS.pop("custom_m_star_test", None)


def test_register_alpha_model_adds_custom_model() -> None:
    """Tests that register_alpha_model adds a custom alpha model."""
    def custom_alpha(z: np.ndarray, *, alpha: float) -> np.ndarray:
        return np.full_like(z, alpha, dtype=float)

    register_alpha_model("custom_alpha_test", custom_alpha)

    try:
        z = np.array([0.0, 0.5, 1.0])
        model = get_parameter_model(
            "custom_alpha_test",
            ALPHA_MODELS,
            model_kind="alpha_model",
        )
        result = model(z, alpha=-1.4)

        np.testing.assert_allclose(result, np.array([-1.4, -1.4, -1.4]))
    finally:
        ALPHA_MODELS.pop("custom_alpha_test", None)


def test_evaluate_lf_parameters_uses_registered_custom_models() -> None:
    """Tests that evaluate_lf_parameters can use registered custom models."""
    def custom_phi(z: np.ndarray, *, amplitude: float) -> np.ndarray:
        return np.full_like(z, amplitude, dtype=float)

    def custom_m_star(z: np.ndarray, *, base: float) -> np.ndarray:
        return np.asarray(base - z, dtype=float)

    def custom_alpha(z: np.ndarray, *, alpha: float) -> np.ndarray:
        return np.full_like(z, alpha, dtype=float)

    register_phi_star_model("eval_custom_phi_test", custom_phi)
    register_m_star_model("eval_custom_m_star_test", custom_m_star)
    register_alpha_model("eval_custom_alpha_test", custom_alpha)

    try:
        z = np.array([0.0, 0.5, 1.0])

        phi_star, m_star, alpha = evaluate_lf_parameters(
            z,
            phi_model="eval_custom_phi_test",
            phi_kwargs={"amplitude": 2.0e-3},
            m_star_model="eval_custom_m_star_test",
            m_star_kwargs={"base": -20.0},
            alpha_model="eval_custom_alpha_test",
            alpha_kwargs={"alpha": -1.2},
        )

        np.testing.assert_allclose(phi_star, np.array([2.0e-3, 2.0e-3, 2.0e-3]))
        np.testing.assert_allclose(m_star, np.array([-20.0, -20.5, -21.0]))
        np.testing.assert_allclose(alpha, np.array([-1.2, -1.2, -1.2]))
    finally:
        PHI_STAR_MODELS.pop("eval_custom_phi_test", None)
        M_STAR_MODELS.pop("eval_custom_m_star_test", None)
        ALPHA_MODELS.pop("eval_custom_alpha_test", None)


def test_register_parameter_model_raises_for_empty_name() -> None:
    """Tests that registering a model with an empty name raises an error."""
    def custom_model(z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float)

    with pytest.raises(ValueError, match="phi_star model name cannot be empty"):
        register_phi_star_model("", custom_model)


def test_register_parameter_model_raises_for_non_callable_model() -> None:
    """Tests that registering a non-callable model raises an error."""
    with pytest.raises(TypeError, match="phi_star model must be callable"):
        register_phi_star_model("not_callable_test", 3.0)  # type: ignore[arg-type]


def test_register_parameter_model_raises_for_duplicate_without_overwrite() -> None:
    """Tests that registering a duplicate model without overwrite raises an error."""
    def custom_model(z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float)

    register_phi_star_model("duplicate_phi_test", custom_model)

    try:
        with pytest.raises(ValueError, match="already registered"):
            register_phi_star_model("duplicate_phi_test", custom_model)
    finally:
        PHI_STAR_MODELS.pop("duplicate_phi_test", None)


def test_register_parameter_model_allows_duplicate_with_overwrite() -> None:
    """Tests that registering a duplicate model with overwrite replaces the model."""
    def first_model(z: np.ndarray) -> np.ndarray:
        return np.full_like(z, 1.0, dtype=float)

    def second_model(z: np.ndarray) -> np.ndarray:
        return np.full_like(z, 2.0, dtype=float)

    register_phi_star_model("overwrite_phi_test", first_model)

    try:
        register_phi_star_model("overwrite_phi_test", second_model, overwrite=True)

        z = np.array([0.0, 0.5])
        result = PHI_STAR_MODELS["overwrite_phi_test"](z)

        np.testing.assert_allclose(result, np.array([2.0, 2.0]))
    finally:
        PHI_STAR_MODELS.pop("overwrite_phi_test", None)


def test_parameter_models_raise_for_non_finite_redshift() -> None:
    """Tests that parameter models reject non-finite redshift values."""
    z = np.array([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        phi_star_constant(z, phi_star=1.0e-3)

    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        m_star_constant(z, m_star=-20.0)

    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        alpha_constant(z, alpha=-1.0)


def test_evaluate_lf_parameters_rejects_non_finite_redshift() -> None:
    """Tests that evaluate_lf_parameters rejects non-finite redshift values."""
    z = np.array([0.0, np.inf])

    with pytest.raises(ValueError, match="z contains NaN or infinite values"):
        evaluate_lf_parameters(
            z,
            phi_model="constant",
            phi_kwargs={"phi_star": 1.0e-3},
            m_star_model="constant",
            m_star_kwargs={"m_star": -20.0},
            alpha_model="constant",
            alpha_kwargs={"alpha": -1.0},
        )
