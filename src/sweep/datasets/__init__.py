"""Embedded demo velocity models.

These ship inside the package as compact base85-encoded blobs so that the
notebooks and quick-start examples can run without any external downloads.
For full-size 3D models (Overthrust etc.), see
``examples/models/overthrust/download_release_asset.py``.
"""

from sweep.datasets.marmousi import (
    GRID_SPACING_M as MARMOUSI_DH,
    UNITS as MARMOUSI_UNITS,
    available as available_marmousi,
    load_marmousi as _load_marmousi_raw,
)


_MARMOUSI_ALIASES = {
    "true":   "vp_true",
    "smooth": "vp_smooth",
    "linear": "vp_linear",
}


def load_marmousi(name="vp_true"):
    """Load an embedded Marmousi preset as a float32 numpy array.

    ``name`` is ``<field>_<kind>`` where ``field`` is ``'vp'`` / ``'vs'`` / ``'rho'``
    and ``kind`` is ``'true'`` / ``'smooth'`` / ``'linear'``. Bare ``'true'`` /
    ``'smooth'`` / ``'linear'`` are kept as aliases for the vp variants.
    """
    return _load_marmousi_raw(_MARMOUSI_ALIASES.get(name, name))
from sweep.datasets.overthrust_2d import (
    GRID_SPACING_M as OVERTHRUST_2D_DH,
    UNITS as OVERTHRUST_2D_UNITS,
    available as available_overthrust_2d,
    load_overthrust_2d,
)

__all__ = [
    "MARMOUSI_DH",
    "MARMOUSI_UNITS",
    "available_marmousi",
    "load_marmousi",
    "OVERTHRUST_2D_DH",
    "OVERTHRUST_2D_UNITS",
    "available_overthrust_2d",
    "load_overthrust_2d",
]
