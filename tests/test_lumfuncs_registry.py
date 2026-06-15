"""Unit tests for ``lfkit.luminosity_functions.registry``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lfkit.luminosity_functions import registry


def model_absolute_mag(absolute_mag: object) -> object:
    """Fake absolute-magnitude luminosity function."""
    return absolute_mag


def model_with_redshift(absolute_mag: object, redshift: object) -> tuple[object, object]:
    """Fake redshift-dependent luminosity function."""
    return absolute_mag, redshift


def model_with_z(absolute_mag: object, z: object) -> tuple[object, object]:
    """Fake z-dependent luminosity function."""
    return absolute_mag, z


def model_with_condition(
    luminosity: object,
    condition: object,
) -> tuple[object, object]:
    """Fake condition-dependent luminosity function."""
    return luminosity, condition


def model_with_x(magnitude: object, x: object) -> tuple[object, object]:
    """Fake generic conditional luminosity function."""
    return magnitude, x


def invalid_first_argument(foo: object) -> object:
    """Fake callable with unsupported first argument."""
    return foo


def no_arguments() -> None:
    """Fake callable with no arguments."""
    return None


def schechter_from_m(magnitude: object) -> object:
    """Fake apparent-magnitude evaluator."""
    return magnitude


def test_lf_model_defaults() -> None:
    """Tests default ``LFModel`` metadata values."""
    model = registry.LFModel(name="test", function=model_absolute_mag)

    assert model.name == "test"
    assert model.function is model_absolute_mag
    assert model.independent_variable == "absolute_mag"
    assert model.requires_z is False


def test_public_model_name_removes_conditional_prefix_and_lf_suffix() -> None:
    """Tests conversion from implementation names to public names."""
    assert registry._public_model_name("conditional_schechter_lf") == "schechter"
    assert registry._public_model_name("schechter_lf") == "schechter"
    assert registry._public_model_name("conditional_gaussian") == "gaussian"
    assert registry._public_model_name("double_schechter") == "double_schechter"


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (model_absolute_mag, False),
        (model_with_redshift, True),
        (model_with_z, True),
        (model_with_condition, True),
        (model_with_x, True),
    ],
)
def test_requires_second_independent_variable(
    function: object,
    expected: bool,
) -> None:
    """Tests supported second independent variable names."""
    import inspect

    assert registry._requires_second_independent_variable(
        inspect.signature(function)
    ) is expected


def test_register_lf_model_adds_valid_model() -> None:
    """Tests registration of a valid luminosity function model."""
    lf_models: dict[str, registry.LFModel] = {}
    from_m_models: dict[str, object] = {}

    registry._register_lf_model(
        "schechter_lf",
        model_absolute_mag,
        lf_models=lf_models,
        from_m_models=from_m_models,
        name_transform=registry._public_model_name,
    )

    assert tuple(lf_models) == ("schechter",)
    assert lf_models["schechter"].name == "schechter"
    assert lf_models["schechter"].function is model_absolute_mag
    assert lf_models["schechter"].independent_variable == "absolute_mag"
    assert lf_models["schechter"].requires_z is False
    assert from_m_models == {}


def test_register_lf_model_records_requires_z() -> None:
    """Tests that redshift-dependent models are marked as requiring z."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "evolving_schechter",
        model_with_redshift,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models["evolving_schechter"].requires_z is True


def test_register_lf_model_accepts_magnitude_and_luminosity_first_arguments() -> None:
    """Tests supported first independent variable names."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "magnitude_model",
        model_with_x,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )
    registry._register_lf_model(
        "luminosity_model",
        model_with_condition,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models["magnitude_model"].independent_variable == "magnitude"
    assert lf_models["luminosity_model"].independent_variable == "luminosity"


def test_register_lf_model_ignores_non_callable() -> None:
    """Tests that non-callable objects are ignored."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "not_callable",
        1,  # type: ignore[arg-type]
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models == {}


def test_register_lf_model_ignores_callable_without_parameters() -> None:
    """Tests that callables without parameters are ignored."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "no_arguments",
        no_arguments,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models == {}


def test_register_lf_model_ignores_unsupported_first_argument() -> None:
    """Tests that callables with unsupported first arguments are ignored."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "bad_model",
        invalid_first_argument,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models == {}


def test_register_lf_model_registers_from_m_model_only() -> None:
    """Tests that ``*_from_m`` callables go into the apparent-magnitude registry."""
    lf_models: dict[str, registry.LFModel] = {}
    from_m_models: dict[str, object] = {}

    registry._register_lf_model(
        "schechter_from_m",
        schechter_from_m,
        lf_models=lf_models,
        from_m_models=from_m_models,
        name_transform=registry._public_model_name,
    )

    assert lf_models == {}
    assert from_m_models == {"schechter": schechter_from_m}


def test_register_lf_model_ignores_from_m_when_registry_is_none() -> None:
    """Tests that ``*_from_m`` callables are ignored without a target registry."""
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_lf_model(
        "schechter_from_m",
        schechter_from_m,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=None,
    )

    assert lf_models == {}


def test_register_module_lf_models_uses_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests registration from a module ``__all__`` declaration."""
    fake_module = SimpleNamespace(
        __all__=["schechter_lf", "CONSTANT"],
        schechter_lf=model_absolute_mag,
        CONSTANT=1.0,
    )
    lf_models: dict[str, registry.LFModel] = {}

    registry._register_module_lf_models(
        fake_module,
        lf_models=lf_models,
        from_m_models=None,
        name_transform=registry._public_model_name,
    )

    assert tuple(lf_models) == ("schechter",)
    assert lf_models["schechter"].function is model_absolute_mag


def test_discover_models_package_uses_iter_model_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests discovery from the models package helper."""
    monkeypatch.setattr(
        registry,
        "iter_model_functions",
        lambda: {
            "schechter_lf": model_absolute_mag,
            "schechter_from_m": schechter_from_m,
            "bad_model": invalid_first_argument,
        },
    )

    lf_models: dict[str, registry.LFModel] = {}
    from_m_models: dict[str, object] = {}

    registry._discover_models_package(lf_models, from_m_models)

    assert tuple(lf_models) == ("schechter",)
    assert lf_models["schechter"].function is model_absolute_mag
    assert from_m_models == {"schechter": schechter_from_m}


def test_discover_conditional_models_uses_conditional_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests discovery from ``conditional_models``."""
    fake_module = SimpleNamespace(
        __all__=["conditional_schechter_lf"],
        conditional_schechter_lf=model_with_condition,
    )

    monkeypatch.setattr(registry, "conditional_models", fake_module)

    conditional_lf_models: dict[str, registry.LFModel] = {}

    registry._discover_conditional_models(conditional_lf_models)

    assert tuple(conditional_lf_models) == ("schechter",)
    assert conditional_lf_models["schechter"].function is model_with_condition
    assert conditional_lf_models["schechter"].requires_z is True


def test_discover_lf_models_returns_all_three_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests full registry discovery."""
    fake_conditional_module = SimpleNamespace(
        __all__=["conditional_schechter_lf"],
        conditional_schechter_lf=model_with_condition,
    )

    monkeypatch.setattr(
        registry,
        "iter_model_functions",
        lambda: {
            "schechter_lf": model_absolute_mag,
            "schechter_from_m": schechter_from_m,
        },
    )
    monkeypatch.setattr(registry, "conditional_models", fake_conditional_module)

    lf_models, conditional_lf_models, from_m_models = registry.discover_lf_models()

    assert tuple(lf_models) == ("schechter",)
    assert tuple(conditional_lf_models) == ("schechter",)
    assert from_m_models == {"schechter": schechter_from_m}


def test_available_lf_models_are_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that available LF model names are sorted."""
    monkeypatch.setattr(
        registry,
        "LF_MODELS",
        {
            "schechter": registry.LFModel("schechter", model_absolute_mag),
            "gaussian": registry.LFModel("gaussian", model_absolute_mag),
        },
    )

    assert registry.available_lf_models() == ("gaussian", "schechter")


def test_available_conditional_lf_models_are_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that available conditional LF model names are sorted."""
    monkeypatch.setattr(
        registry,
        "CONDITIONAL_LF_MODELS",
        {
            "schechter": registry.LFModel("schechter", model_absolute_mag),
            "gaussian": registry.LFModel("gaussian", model_absolute_mag),
        },
    )

    assert registry.available_conditional_lf_models() == ("gaussian", "schechter")


def test_available_lf_from_m_models_are_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that available apparent-magnitude model names are sorted."""
    monkeypatch.setattr(
        registry,
        "LF_FROM_M_MODELS",
        {
            "schechter": schechter_from_m,
            "gaussian": schechter_from_m,
        },
    )

    assert registry.available_lf_from_m_models() == ("gaussian", "schechter")


def test_get_lf_model_returns_registered_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests successful LF model lookup."""
    expected = registry.LFModel("schechter", model_absolute_mag)

    monkeypatch.setattr(registry, "LF_MODELS", {"schechter": expected})

    assert registry.get_lf_model("schechter") is expected


def test_get_conditional_lf_model_returns_registered_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests successful conditional LF model lookup."""
    expected = registry.LFModel("schechter", model_with_condition)

    monkeypatch.setattr(registry, "CONDITIONAL_LF_MODELS", {"schechter": expected})

    assert registry.get_conditional_lf_model("schechter") is expected


def test_get_lf_from_m_model_returns_registered_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests successful apparent-magnitude evaluator lookup."""
    monkeypatch.setattr(registry, "LF_FROM_M_MODELS", {"schechter": schechter_from_m})

    assert registry.get_lf_from_m_model("schechter") is schechter_from_m


def test_get_lf_model_raises_for_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests LF model lookup error message."""
    monkeypatch.setattr(
        registry,
        "LF_MODELS",
        {"schechter": registry.LFModel("schechter", model_absolute_mag)},
    )

    with pytest.raises(ValueError, match="Unknown luminosity function model 'bad'"):
        registry.get_lf_model("bad")


def test_get_conditional_lf_model_raises_for_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests conditional LF model lookup error message."""
    monkeypatch.setattr(
        registry,
        "CONDITIONAL_LF_MODELS",
        {"schechter": registry.LFModel("schechter", model_with_condition)},
    )

    with pytest.raises(
        ValueError,
        match="Unknown conditional luminosity function model 'bad'",
    ):
        registry.get_conditional_lf_model("bad")


def test_get_lf_from_m_model_raises_for_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests apparent-magnitude evaluator lookup error message."""
    monkeypatch.setattr(
        registry,
        "LF_FROM_M_MODELS",
        {"schechter": schechter_from_m},
    )

    with pytest.raises(
        ValueError,
        match="phi_from_m is not defined for luminosity function model 'bad'",
    ):
        registry.get_lf_from_m_model("bad")
