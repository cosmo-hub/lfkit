"""Filter / response mapping utilities.

LFKit uses a survey-oriented way to specify photometric bands:

    (filterset, band)

Examples:
    ("sdss", "r")
    ("hsc", "i")
    ("decam", "r")
    ("bessell", "V")

Internally, kcorrect does not work with survey names. Instead it expects
**response curve identifiers**, which correspond to filter throughput
files distributed with the package.

This module provides a lightweight mapping between the LFKit public
notation

    (filterset, band)

and the corresponding kcorrect response names. The mapping can also be
extended or overridden to support custom filter curves.
"""

from __future__ import annotations

from typing import Iterable

KNOWN_FILTERSETS: tuple[str, ...] = (
    "sdss",
    "hsc",
    "decam",
    "bessell",
)

DEFAULT_RESPONSE_MAP: dict[tuple[str, str], str] = {
    ("sdss", "u"): "sdss_u0",
    ("sdss", "g"): "sdss_g0",
    ("sdss", "r"): "sdss_r0",
    ("sdss", "i"): "sdss_i0",
    ("sdss", "z"): "sdss_z0",

    ("decam", "u"): "decam_u",
    ("decam", "g"): "decam_g",
    ("decam", "r"): "decam_r",
    ("decam", "i"): "decam_i",
    ("decam", "z"): "decam_z",
    ("decam", "Y"): "decam_Y",

    ("hsc", "g"): "subaru_suprimecam_g",
    ("hsc", "r"): "subaru_suprimecam_r",
    ("hsc", "i"): "subaru_suprimecam_i",
    ("hsc", "z"): "subaru_suprimecam_z",

    ("bessell", "U"): "bessell_U",
    ("bessell", "B"): "bessell_B",
    ("bessell", "V"): "bessell_V",
    ("bessell", "R"): "bessell_R",
    ("bessell", "I"): "bessell_I",

    ("2mass", "J"): "twomass_J",
    ("2mass", "H"): "twomass_H",
    ("2mass", "K"): "twomass_Ks",
}


def normalize_filterset(filterset: str) -> str:
    """Return the canonical LFKit representation of a filterset name.

    Filterset names are normalized to lowercase and stripped of
    surrounding whitespace so that user input such as "SDSS", "sdss",
    or " sdss " all resolve to the same canonical form.
    """
    return str(filterset).strip().lower()


def normalize_band(band: str) -> str:
    """Normalize a band label.

    Band names are stripped of surrounding whitespace while preserving
    their case, since some filter systems distinguish between
    uppercase and lowercase band identifiers.
    """
    return str(band).strip()


def make_response_map(
    *,
    base: dict[tuple[str, str], str] | None = None,
    extra: dict[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str], str]:
    """Create a response mapping dictionary.

    The returned mapping associates (filterset, band) pairs with
    kcorrect response identifiers. A custom mapping can be created
    by starting from a base map and overriding or extending entries
    with the ``extra`` dictionary.
    """
    out = dict(DEFAULT_RESPONSE_MAP if base is None else base)
    if extra:
        out.update(extra)
    return out


def resolve_response_name(
    *,
    filterset: str,
    band: str,
    response_map: dict[tuple[str, str], str] | None = None,
) -> str:
    """Resolve a survey band to a kcorrect response identifier.

    Given a (filterset, band) pair, this function returns the
    corresponding kcorrect response name used to load the filter
    throughput curve. A custom mapping can be provided to support
    additional surveys or user-defined filters.
    """
    fs = normalize_filterset(filterset)
    b = normalize_band(band)
    m = DEFAULT_RESPONSE_MAP if response_map is None else response_map

    key = (fs, b)
    if key in m:
        return str(m[key])

    available_for_fs = sorted([bb for (ffs, bb) in m.keys() if ffs == fs])
    if available_for_fs:
        raise ValueError(
            f"No response mapping for (filterset, band)=({fs!r}, {b!r}). "
            f"Known bands for filterset={fs!r}: {available_for_fs}. "
            "Provide response_map=... to extend the mapping."
        )

    known_filtersets = sorted(set(ffs for (ffs, _) in m.keys()))
    raise ValueError(
        f"No response mapping for filterset={fs!r}. "
        f"Known filtersets in mapping: {known_filtersets}. "
        "Provide response_map=... to extend the mapping."
    )


def list_supported(
    response_map: dict[tuple[str, str], str] | None = None,
) -> dict[str, list[str]]:
    """Return the set of supported bands grouped by filterset.

    The output lists all (filterset, band) combinations available
    in the current response mapping. This is useful for inspecting
    which survey filters are currently recognized by LFKit.
    """
    m = DEFAULT_RESPONSE_MAP if response_map is None else response_map
    out: dict[str, list[str]] = {}
    for (fs, band), _resp in m.items():
        out.setdefault(fs, []).append(band)
    for fs in out:
        out[fs] = sorted(set(out[fs]))
    return dict(sorted(out.items()))


def validate_coverage(
    *,
    filterset: str,
    bands: Iterable[str],
    response_map: dict[tuple[str, str], str] | None = None,
) -> None:
    """Check that a set of bands exists in the response mapping.

    This function verifies that all requested bands are defined for
    the given filterset in the response mapping. If any band is
    missing, an informative error is raised listing the supported
    bands for that filterset.
    """
    fs = normalize_filterset(filterset)
    m = DEFAULT_RESPONSE_MAP if response_map is None else response_map

    missing: list[str] = []
    for b in bands:
        key = (fs, normalize_band(b))
        if key not in m:
            missing.append(str(b))

    if missing:
        supported = sorted([bb for (ffs, bb) in m.keys() if ffs == fs])
        raise ValueError(
            f"Missing response mappings for filterset={fs!r}: {missing}. "
            f"Supported bands: {supported}. "
            "Provide response_map=... to extend."
        )
