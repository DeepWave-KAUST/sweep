"""Benchmark velocity models for seismic FWI / LSRTM.

Two flavours share one registry:

* **Embedded demos** — tiny base85-encoded blobs (Marmousi, Overthrust-2D)
  that ship inside the wheel, so notebooks and quick-start examples run with
  no downloads and no extra dependencies.
* **Downloadable benchmarks** — full-size public models (Marmousi II,
  Overthrust 3-D, SEG/EAGE Salt, Marmousi2 elastic, BP 2004, BP 2007 TTI,
  Hess VTI) fetched on demand and cached, each carrying its own
  license / citation.

Unified access::

    from sweep import datasets
    print(datasets.available())                 # [(name, variant), ...]
    prob = datasets.load("overthrust", "3d-acoustic")
    vp = prob["vp"]

Back-compatible helpers (``load_marmousi`` / ``load_overthrust_2d`` and the
``*_DH`` / ``*_UNITS`` constants) are preserved.

Optional dependency: ``pip install sweep-solver[datasets]`` adds the HTTP
client (``requests``/``tqdm``) used to fetch the downloadable benchmarks.
Parsing (SEG-Y / raw grids) is numpy-only — no extra parser dependency.
"""

from __future__ import annotations

from typing import Any

from sweep.datasets.marmousi import (
    GRID_SPACING_M as MARMOUSI_DH,
    UNITS as MARMOUSI_UNITS,
    available as available_marmousi,
    load_marmousi as _load_marmousi_raw,
)
from sweep.datasets.overthrust_2d import (
    GRID_SPACING_M as OVERTHRUST_2D_DH,
    UNITS as OVERTHRUST_2D_UNITS,
    available as available_overthrust_2d,
    load_overthrust_2d,
)
from sweep.datasets.registry import Entry, available, catalog, info, load, register


_MARMOUSI_ALIASES = {
    "true": "vp_true",
    "smooth": "vp_smooth",
    "linear": "vp_linear",
}


def load_marmousi(name: str = "vp_true"):
    """Load an embedded Marmousi preset as a float32 numpy array.

    ``name`` is ``<field>_<kind>`` where ``field`` is ``'vp'`` / ``'vs'`` /
    ``'rho'`` and ``kind`` is ``'true'`` / ``'smooth'`` / ``'linear'``. Bare
    ``'true'`` / ``'smooth'`` / ``'linear'`` are aliases for the vp variants.
    """
    return _load_marmousi_raw(_MARMOUSI_ALIASES.get(name, name))


# ---------------------------------------------------------------- embedded
# Register the embedded demos as first-class registry entries so a single
# ``load(...)`` covers both demo blobs and downloadable benchmarks.
def _embedded_marmousi(*, name: str = "vp_true") -> dict[str, Any]:
    vp = load_marmousi(name)
    dh = (MARMOUSI_DH, MARMOUSI_DH)
    return {"vp": vp, "dh": dh, "units": MARMOUSI_UNITS, "presets": available_marmousi()}


def _embedded_overthrust2d(*, name: str = "true") -> dict[str, Any]:
    vp = load_overthrust_2d(name)
    dh = (OVERTHRUST_2D_DH, OVERTHRUST_2D_DH)
    return {"vp": vp, "dh": dh, "units": OVERTHRUST_2D_UNITS, "presets": available_overthrust_2d()}


register(Entry(
    name="marmousi", variant="2d-demo", loader=_embedded_marmousi,
    description="Embedded Marmousi demo (vp/vs/rho x true/smooth/linear, 281x1361).",
    citation="Versteeg (1994), DOI 10.1190/1.1437051",
    license="Derived demo of SEG open data — bundled for quick-start use.",
    kind="embedded", redistributable=True, default=True, tags=["2d", "demo", "embedded"],
))
register(Entry(
    name="overthrust", variant="2d-demo", loader=_embedded_overthrust2d,
    description="Embedded Overthrust-2D demo slice (true/smooth, 187x801).",
    citation="Aminzadeh, Brac, Kunz (1997), SEG/EAGE 3-D Modeling Series No. 1",
    license="Derived demo (CC-BY-4.0 source) — bundled for quick-start use.",
    kind="embedded", redistributable=True, default=True, tags=["2d", "demo", "embedded"],
))

# Side-effect import: registers all downloadable benchmark entries.
from sweep.datasets import _benchmarks as _benchmarks  # noqa: E402,F401


__all__ = [
    # unified registry API
    "Entry",
    "register",
    "available",
    "catalog",
    "info",
    "load",
    # back-compat embedded helpers
    "MARMOUSI_DH",
    "MARMOUSI_UNITS",
    "available_marmousi",
    "load_marmousi",
    "OVERTHRUST_2D_DH",
    "OVERTHRUST_2D_UNITS",
    "available_overthrust_2d",
    "load_overthrust_2d",
]
