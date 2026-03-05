"""``kcorrect`` backend construction utilities.

This module provides a small wrapper around ``kcorrect.Kcorrect`` that
standardizes how LFKit creates and reuses backend instances.

A kcorrect backend depends on the set of input responses, output responses,
and the redshift grid used internally by the solver. Constructing this object
can be relatively expensive, so LFKit builds and caches instances associated
with a specific response configuration.

The wrapper also ensures that requested filter response names exist before
initializing the backend and allows optional use of custom response
directories when supported by the installed kcorrect version.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any

import kcorrect.kcorrect as kk

from .responses import require_responses


def _kc_cache_key(
    *,
    responses_in: tuple[str, ...],
    responses_out: tuple[str, ...],
    responses_map: tuple[str, ...],
    response_dir: str | Path | None,
    redshift_range: tuple[float, float],
    nredshift: int,
    abcorrect: bool,
) -> tuple:
    """Build a normalized cache key for a kcorrect backend configuration.

    The key uniquely identifies a backend setup based on the requested
    filter responses, redshift grid configuration, and optional response
    directory. This normalized representation allows identical backend
    configurations to reuse the same cached ``kcorrect.Kcorrect`` instance.
    """
    if response_dir is None:
        rdir_key = "__AUTO__"
    else:
        rdir_key = str(Path(response_dir).resolve())

    return (
        responses_in,
        responses_out,
        responses_map,
        rdir_key,
        (float(redshift_range[0]), float(redshift_range[1])),
        int(nredshift),
        bool(abcorrect),
    )


@lru_cache(maxsize=8)
def _build_kcorrect_cached(key: tuple) -> kk.Kcorrect:
    """Construct and cache a ``kcorrect.Kcorrect`` backend instance.

    The input key encodes the response configuration and solver settings.
    For each unique configuration the corresponding backend is created
    once and stored in the cache, allowing repeated k(z) evaluations to
    reuse the same initialized solver.
    """
    (
        responses_in,
        responses_out,
        responses_map,
        rdir_key,
        redshift_range,
        nredshift,
        abcorrect,
    ) = key

    response_dir: Path | None
    if rdir_key == "__AUTO__":
        response_dir = None
    else:
        response_dir = Path(rdir_key)

    # validate response names exist
    require_responses(list(responses_in), response_dir)
    require_responses(list(responses_out), response_dir)
    require_responses(list(responses_map), response_dir)

    kwargs: dict[str, Any] = dict(
        responses=list(responses_in),
        responses_out=list(responses_out),
        responses_map=list(responses_map),
        redshift_range=[float(redshift_range[0]), float(redshift_range[1])],
        nredshift=int(nredshift),
        abcorrect=bool(abcorrect),
    )

    if response_dir is not None:
        sig = inspect.signature(kk.Kcorrect)
        if "response_dir" in sig.parameters:
            kwargs["response_dir"] = str(response_dir)
        else:
            raise TypeError(
                "This kcorrect version does not support response_dir=... in kk.Kcorrect(...). "
                "To use custom filters, install a kcorrect build that supports response_dir, "
                "or place your *.dat filter files into kcorrect’s packaged response directory."
            )

    return kk.Kcorrect(**kwargs)


def build_kcorrect(
    *,
    responses_in: list[str],
    responses_out: list[str] | None = None,
    responses_map: list[str] | None = None,
    response_dir: str | Path | None = None,
    redshift_range: tuple[float, float] = (0.0, 2.0),
    nredshift: int = 4000,
    abcorrect: bool = False,
) -> kk.Kcorrect:
    """Create a kcorrect backend configured for a specific response setup.

    This function returns a ``kcorrect.Kcorrect`` instance configured with the
    requested input and output filter responses. The backend defines the template
    set and internal redshift grid used to evaluate k-corrections.

    Backend objects are cached so that repeated calls with the same configuration
    reuse the existing instance rather than rebuilding the solver each time. This
    keeps repeated k(z) evaluations fast while ensuring that the requested filter
    responses are validated before use.
    """
    if responses_out is None:
        responses_out = responses_in
    if responses_map is None:
        responses_map = responses_in

    key = _kc_cache_key(
        responses_in=tuple(map(str, responses_in)),
        responses_out=tuple(map(str, responses_out)),
        responses_map=tuple(map(str, responses_map)),
        response_dir=response_dir,
        redshift_range=(float(redshift_range[0]), float(redshift_range[1])),
        nredshift=int(nredshift),
        abcorrect=bool(abcorrect),
    )
    return _build_kcorrect_cached(key)