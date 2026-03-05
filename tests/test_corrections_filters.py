"""Unit tests for `lfkit.corrections.filters` module."""

from __future__ import annotations

import pytest

from lfkit.corrections.filters import (
    DEFAULT_RESPONSE_MAP,
    KNOWN_FILTERSETS,
    list_supported,
    make_response_map,
    normalize_band,
    normalize_filterset,
    resolve_response_name,
    validate_coverage,
)


def test_normalize_filterset_lowercases_and_strips():
    """Tests that normalize_filterset strips whitespace and lowercases."""
    assert normalize_filterset(" SDSS ") == "sdss"
    assert normalize_filterset("BeSsElL") == "bessell"


def test_normalize_band_strips_but_preserves_case():
    """Tests that normalize_band strips whitespace but preserves case."""
    assert normalize_band(" r ") == "r"
    assert normalize_band(" V ") == "V"


def test_make_response_map_defaults_to_module_default():
    """Tests that make_response_map returns DEFAULT_RESPONSE_MAP when no args are provided."""
    m = make_response_map()
    assert isinstance(m, dict)
    # exact same key/value content (but not necessarily the same object)
    assert m == dict(DEFAULT_RESPONSE_MAP)


def test_make_response_map_extra_overrides_base():
    """Tests that make_response_map applies extra overrides on top of base."""
    base = {("sdss", "r"): "sdss_r0", ("sdss", "g"): "sdss_g0"}
    extra = {("sdss", "r"): "custom_r", ("bessell", "V"): "custom_V"}
    m = make_response_map(base=base, extra=extra)

    assert m[("sdss", "g")] == "sdss_g0"
    assert m[("sdss", "r")] == "custom_r"
    assert m[("bessell", "V")] == "custom_V"


def test_resolve_response_name_hits_default_mapping():
    """Tests that resolve_response_name returns the default mapping for known (filterset, band)."""
    assert resolve_response_name(filterset="sdss", band="r") == "sdss_r0"
    assert resolve_response_name(filterset="bessell", band="V") == "bessell_V"


def test_resolve_response_name_normalizes_filterset_and_band_whitespace():
    """Tests that resolve_response_name normalizes filterset case and band whitespace."""
    assert resolve_response_name(filterset=" SDSS ", band=" r ") == "sdss_r0"
    assert resolve_response_name(filterset=" Bessell ", band=" V ") == "bessell_V"


def test_resolve_response_name_uses_custom_response_map():
    """Tests that resolve_response_name uses response_map override when provided."""
    m = {("sdss", "r"): "my_sdss_r"}
    assert resolve_response_name(filterset="sdss", band="r", response_map=m) == "my_sdss_r"


def test_resolve_response_name_unknown_band_lists_known_bands_for_filterset():
    """Tests that resolve_response_name error for unknown band includes known bands for that filterset."""
    with pytest.raises(ValueError, match=r"Known bands for filterset='sdss':"):
        resolve_response_name(filterset="sdss", band="q")  # not present in default


def test_resolve_response_name_unknown_filterset_lists_known_filtersets():
    """Tests that resolve_response_name error for unknown filterset includes known filtersets in mapping."""
    with pytest.raises(ValueError, match=r"Known filtersets in mapping:"):
        resolve_response_name(filterset="not_a_filterset", band="r")


def test_list_supported_returns_expected_structure():
    """Tests that list_supported returns a dict of filterset -> sorted unique bands."""
    out = list_supported()
    assert isinstance(out, dict)
    assert "sdss" in out
    assert "bessell" in out
    assert out["sdss"] == sorted(out["sdss"])
    assert len(out["sdss"]) == len(set(out["sdss"]))  # unique


def test_list_supported_respects_custom_mapping():
    """Tests that list_supported uses the provided response_map."""
    m = {
        ("sdss", "r"): "sdss_r0",
        ("sdss", "x"): "sdss_x0",
        ("foo", "A"): "foo_A",
    }
    out = list_supported(response_map=m)
    assert out == {"foo": ["A"], "sdss": ["r", "x"]}


def test_validate_coverage_accepts_all_present():
    """Tests that validate_coverage does not raise when all requested bands are present."""
    validate_coverage(filterset="sdss", bands=["u", "g", "r", "i", "z"])
    validate_coverage(filterset="bessell", bands=["B", "V"])


def test_validate_coverage_rejects_missing_bands_with_supported_list():
    """Tests that validate_coverage raises ValueError listing missing bands and supported bands."""
    with pytest.raises(ValueError, match=r"Missing response mappings for filterset='sdss'"):
        validate_coverage(filterset="sdss", bands=["r", "q"])  # q missing


def test_validate_coverage_normalizes_filterset_and_band_whitespace():
    """Tests that validate_coverage normalizes filterset and band whitespace."""
    # should not raise: bands are in the default mapping after normalization
    validate_coverage(filterset=" SDSS ", bands=[" r ", " g "])


def test_validate_coverage_with_custom_mapping_can_fill_gaps():
    """Tests that validate_coverage passes when custom response_map provides missing entries."""
    m = {("hsc", "i"): "subaru_hsc_i", ("hsc", "r"): "subaru_hsc_r"}
    validate_coverage(filterset="hsc", bands=["i", "r"], response_map=m)


def test_known_filtersets_contains_expected_canonicals():
    """Tests that KNOWN_FILTERSETS contains canonical lowercase names."""
    assert all(fs == fs.lower() for fs in KNOWN_FILTERSETS)
    assert "sdss" in KNOWN_FILTERSETS
    assert "bessell" in KNOWN_FILTERSETS


def test_resolve_response_name_bessell_lowercase_v_rejected():
    """Tests that bessell band matching is case-sensitive (v != V)."""
    with pytest.raises(ValueError, match=r"Known bands for filterset='bessell':"):
        resolve_response_name(filterset="bessell", band="v")


def test_validate_coverage_unknown_filterset_raises():
    """Tests that validate_coverage raises for an unknown filterset."""
    with pytest.raises(ValueError):
        validate_coverage(filterset="nope", bands=["r"])
