"""Offline tests for the unified sweep.datasets registry.

These never touch the network — they exercise the registry wiring, the
embedded demo loaders, back-compat helpers, and (importantly) verify that
importing ``sweep.datasets`` does not pull in the optional download/parse
dependencies.
"""

import sys

import numpy as np
import pytest

from sweep import datasets
from sweep.datasets import _formats


def test_registry_lists_embedded_and_download():
    keys = datasets.available()
    assert ("marmousi", "2d-demo") in keys        # embedded
    assert ("overthrust", "2d-demo") in keys       # embedded
    assert ("overthrust", "3d-acoustic") in keys   # download (CC-BY)
    assert ("seg-eage-salt", "3d-acoustic") in keys
    assert ("marmousi2", "2d-elastic") in keys
    assert ("bp-2004", "2d-acoustic") in keys
    assert ("bp-2007-tti", "2d-tti") in keys
    assert ("hess-vti", "2d-vti") in keys


def test_excluded_models_absent():
    names = {k[0] for k in datasets.available()}
    # Sigsbee (Data Release Agreement), SEAM (ambiguous license) and OpenFWI
    # (non-commercial CC BY-NC-SA) are intentionally not bundled.
    assert "sigsbee" not in names and "sigsbee2a" not in names
    assert "seam" not in names
    assert "openfwi" not in names


def test_every_entry_has_license_and_citation():
    for name, variant in datasets.available():
        e = datasets.info(name, variant)
        assert e.license, f"{name}:{variant} missing license"
        assert e.citation, f"{name}:{variant} missing citation"


def test_embedded_marmousi_loads_offline():
    prob = datasets.load("marmousi", "2d-demo")
    assert prob["vp"].shape == (281, 1361)
    assert prob["vp"].dtype.name == "float32"
    assert prob["dh"] == (12.5, 12.5)


def test_embedded_overthrust2d_loads_offline():
    prob = datasets.load("overthrust", "2d-demo")
    assert prob["vp"].shape == (187, 801)
    assert prob["dh"] == (25.0, 25.0)


def test_backcompat_helpers():
    assert datasets.load_marmousi("vp_true").shape == (281, 1361)
    assert datasets.load_marmousi("vs_smooth").shape == (281, 1361)
    assert datasets.load_overthrust_2d("true").shape == (187, 801)
    assert datasets.MARMOUSI_DH == 12.5
    assert datasets.OVERTHRUST_2D_DH == 25.0
    assert "vp_true" in datasets.available_marmousi()


def test_import_does_not_pull_optional_deps():
    # Importing the datasets package (and loading embedded demos) must not
    # import requests/tqdm/sweep_io/h5py — those are lazy, download-only.
    for mod in ("requests", "tqdm", "sweep_io", "h5py"):
        assert mod not in sys.modules, f"{mod} imported eagerly by sweep.datasets"


def test_catalog_lists_and_filters(capsys):
    rows = datasets.catalog()                 # prints a table, returns entries
    assert len(rows) == len(datasets.available())
    assert all(isinstance(e, datasets.Entry) for e in rows)
    out = capsys.readouterr().out
    assert "name:variant" in out and "overthrust:3d-acoustic" in out
    # filter to one family
    only = datasets.catalog("marmousi")
    assert {e.name for e in only} == {"marmousi"}


def test_default_variant_resolution():
    # Multi-variant names resolve to the canonical default (not alphabetical).
    assert datasets.info("marmousi").variant == "2d-demo"      # not 2d-acoustic
    assert datasets.info("overthrust").variant == "2d-demo"    # not 3d-acoustic
    # Bare load(name) of a multi-variant model gives the embedded demo (offline).
    assert datasets.load("marmousi")["variant"] == "2d-demo"
    # Sole-variant names resolve unambiguously without a default flag.
    assert datasets.info("bp-2004").variant == "2d-acoustic"
    assert datasets.info("marmousi2").variant == "2d-elastic"


def test_ambiguous_name_without_default_raises():
    # A name with several variants and no default must not silently guess.
    import numpy as np

    def _stub():
        return {"vp": np.zeros((2, 2), dtype="float32"), "dh": (1.0, 1.0)}

    for v in ("va", "vb"):
        datasets.register(
            datasets.Entry(name="_pytest_ambig", variant=v, loader=_stub),
            replace=True,
        )
    with pytest.raises(KeyError, match="multiple variants"):
        datasets.info("_pytest_ambig")


def test_downsample_uniform_and_per_axis():
    a = np.arange(12 * 20, dtype="float32").reshape(12, 20)
    # uniform int
    out, fac = _formats.decimate(a, 2)
    assert out.shape == (6, 10) and fac == (2, 2)
    # per-axis tuple / list
    out, fac = _formats.decimate(a, (2, 4))
    assert out.shape == (6, 5) and fac == (2, 4)
    out, fac = _formats.decimate(a, [1, 5])
    assert out.shape == (12, 4) and fac == (1, 5)
    # 3D per-axis
    b = np.zeros((8, 6, 4), dtype="float32")
    out, fac = _formats.decimate(b, (2, 1, 4))
    assert out.shape == (4, 6, 1) and fac == (2, 1, 4)


def test_downsample_validation():
    a = np.zeros((10, 10), dtype="float32")
    with pytest.raises(ValueError, match="axes but the model is 2-D"):
        _formats.decimate(a, (2, 2, 2))         # wrong ndim
    with pytest.raises(ValueError, match=">= 1"):
        _formats.decimate(a, 0)                  # zero factor
    with pytest.raises(ValueError, match=">= 1"):
        _formats.decimate(a, (2, 0))             # per-axis zero


def test_cli_offline_subcommands(capsys):
    from sweep.datasets.cli import main

    assert main(["list"]) == 0
    assert "overthrust:3d-acoustic" in capsys.readouterr().out
    assert main(["list", "marmousi"]) == 0
    out = capsys.readouterr().out
    assert "marmousi:2d-demo" in out and "bp-2004" not in out
    assert main(["info", "bp-2004"]) == 0
    assert "Billette" in capsys.readouterr().out
    assert main(["where"]) == 0
    assert capsys.readouterr().out.strip()


def test_download_entry_metadata_flags():
    # CC-BY models are re-hostable; AS-IS ones are not.
    assert datasets.info("overthrust", "3d-acoustic").redistributable is True
    assert datasets.info("seg-eage-salt", "3d-acoustic").redistributable is True
    assert datasets.info("bp-2004", "2d-acoustic").redistributable is False
    assert datasets.info("hess-vti", "2d-vti").redistributable is False
