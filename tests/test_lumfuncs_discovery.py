"""Unit tests for ``lfkit.luminosity_functions._discovery``."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lfkit.luminosity_functions import _discovery


def _module_info(name: str) -> SimpleNamespace:
    """Return a small stand-in for ``pkgutil.ModuleInfo``."""
    return SimpleNamespace(name=name)


def test_iter_model_functions_discovers_callables_from_public_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that callable names in ``__all__`` are discovered."""

    def model_function() -> None:
        return None

    fake_module = SimpleNamespace(
        __all__=["model_function"],
        model_function=model_function,
    )

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("fake_models")],
    )
    monkeypatch.setattr(
        _discovery.importlib,
        "import_module",
        lambda name: fake_module,
    )

    result = _discovery.iter_model_functions()

    assert result == {"model_function": model_function}


def test_iter_model_functions_ignores_non_callables_in_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that exported non-callable objects are ignored."""

    def model_function() -> None:
        return None

    fake_module = SimpleNamespace(
        __all__=["model_function", "CONSTANT"],
        model_function=model_function,
        CONSTANT=1.0,
    )

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("fake_models")],
    )
    monkeypatch.setattr(
        _discovery.importlib,
        "import_module",
        lambda name: fake_module,
    )

    result = _discovery.iter_model_functions()

    assert result == {"model_function": model_function}


def test_iter_model_functions_ignores_modules_without_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that modules without ``__all__`` contribute no functions."""

    def hidden_model() -> None:
        return None

    fake_module = SimpleNamespace(hidden_model=hidden_model)

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("fake_models")],
    )
    monkeypatch.setattr(
        _discovery.importlib,
        "import_module",
        lambda name: fake_module,
    )

    result = _discovery.iter_model_functions()

    assert result == {}


def test_iter_model_functions_skips_private_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that modules beginning with an underscore are skipped."""

    import_module = Mock()

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("_private_models")],
    )
    monkeypatch.setattr(_discovery.importlib, "import_module", import_module)

    result = _discovery.iter_model_functions()

    assert result == {}
    import_module.assert_not_called()


def test_iter_model_functions_skips_modifier_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that explicitly skipped modules are not imported."""

    import_module = Mock()

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("modifiers")],
    )
    monkeypatch.setattr(_discovery.importlib, "import_module", import_module)

    result = _discovery.iter_model_functions()

    assert result == {}
    import_module.assert_not_called()


def test_iter_model_functions_imports_expected_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that discovered modules are imported from the models package."""

    fake_module = SimpleNamespace(__all__=[])
    import_module = Mock(return_value=fake_module)

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("schechter")],
    )
    monkeypatch.setattr(_discovery.importlib, "import_module", import_module)

    _discovery.iter_model_functions()

    import_module.assert_called_once_with(
        f"{_discovery.models_pkg.__name__}.schechter"
    )


def test_iter_model_functions_later_modules_override_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that later modules overwrite earlier duplicate exported names."""

    def first_model() -> str:
        return "first"

    def second_model() -> str:
        return "second"

    modules = {
        f"{_discovery.models_pkg.__name__}.first": SimpleNamespace(
            __all__=["same_name"],
            same_name=first_model,
        ),
        f"{_discovery.models_pkg.__name__}.second": SimpleNamespace(
            __all__=["same_name"],
            same_name=second_model,
        ),
    }

    monkeypatch.setattr(
        _discovery.pkgutil,
        "iter_modules",
        lambda path: [_module_info("first"), _module_info("second")],
    )
    monkeypatch.setattr(
        _discovery.importlib,
        "import_module",
        lambda name: modules[name],
    )

    result = _discovery.iter_model_functions()

    assert result == {"same_name": second_model}


def test_iter_model_functions_uses_models_package_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that module discovery uses ``models_pkg.__path__``."""

    seen_paths = []

    def iter_modules(path: object) -> Iterator[SimpleNamespace]:
        seen_paths.append(path)
        return iter(())

    monkeypatch.setattr(_discovery.pkgutil, "iter_modules", iter_modules)

    result = _discovery.iter_model_functions()

    assert result == {}
    assert seen_paths == [_discovery.models_pkg.__path__]
