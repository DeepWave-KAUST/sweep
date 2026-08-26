"""Invariants of the topography runtime build that a before/after gate cannot see.

`gate/bitgate.py --tier T` compares one run against a stored baseline, which
catches a changed value. It cannot catch either of these:

* a build that is only *usually* right -- the host-to-device copy of the surface
  rows is asynchronous, and without the synchronize in
  ``build_image_method_topo_rows`` the CUDA forward showed roughly 30%
  non-determinism. A one-shot comparison passes on the runs where the race
  happens not to bite, so the gate would go *flaky* rather than red.
* a y/x transposition in the 3-D runtime pad. Every pad width is equal, so the
  transposed and correct forms agree elementwise whenever ``ny == nx`` -- and
  agree in SHAPE always. Only a non-square grid separates them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.core.topography import (
    build_apm_air_mask,
    build_image_method_topo_rows,
    canonicalise_topography,
    physical_extent,
    resolve_topo_method,
)


def _rows_2d(nx=17):
    x = np.arange(nx)
    return torch.tensor((3 + 5 * np.exp(-((x - nx / 2) ** 2) / 8)).round().astype(np.int64))


# --------------------------------------------------------------------------- #
# dtype / device contracts
# --------------------------------------------------------------------------- #
def test_runtime_rows_are_int32():
    """The compiled side reads ``data_ptr<int>()``.

    A wider dtype here is not a wrong number, it is a pointer reinterpretation
    -- so this is pinned separately from any value comparison.
    """
    out = build_image_method_topo_rows(_rows_2d(), abcn=4, halo=2, device=None)
    assert out.dtype == torch.int32


def test_air_mask_stays_float32():
    """Not bool: replicate padding rejects bool, and the APM classifier
    branches on the tensor having a ``device``."""
    rows = _rows_2d()
    _, mask = canonicalise_topography(rows, 20, rows.shape[0])
    assert mask.dtype == torch.float32
    assert build_apm_air_mask(mask, abcn=3, halo=2, device=None).dtype == torch.float32


def test_air_mask_is_born_on_the_inputs_device():
    """A numpy topography yields a CPU mask even on a CUDA run, and the
    propagator stores that CPU tensor as ``self.topography`` while the equation
    gets the padded copy. Moving it here would change what the caller keeps."""
    rows = _rows_2d()
    _, mask = canonicalise_topography(rows, 20, rows.shape[0])
    assert mask.device == rows.device


# --------------------------------------------------------------------------- #
# the transposition a square grid cannot show
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", ["image", "apm"])
def test_3d_pad_preserves_axis_order(builder):
    ny, nx, nz = 5, 8, 11          # deliberately all different
    rows = torch.arange(ny * nx, dtype=torch.long).reshape(ny, nx) % (nz - 1)
    pad = 3
    if builder == "image":
        out = build_image_method_topo_rows(rows, abcn=pad - 2, halo=2, device=None)
        assert out.shape == (ny + 2 * pad, nx + 2 * pad)
    else:
        _, mask = canonicalise_topography(rows, nz, ny, nx)
        out = build_apm_air_mask(mask, abcn=pad - 2, halo=2, device=None)
        assert out.shape == (nz + 2 * pad, ny + 2 * pad, nx + 2 * pad)


def test_3d_replicate_pad_copies_the_edge_not_the_wrong_edge():
    """A transposed pad tuple would still replicate -- just from the other axis.

    Checking shapes alone would pass. This asserts the padded corner carries the
    value of the nearest CORNER of the original, which a y/x swap breaks once
    the profile varies along both axes.
    """
    ny, nx = 4, 7
    rows = (torch.arange(ny)[:, None] * 10 + torch.arange(nx)[None, :]).to(torch.long)
    out = build_image_method_topo_rows(rows, abcn=1, halo=2, device=None)
    pad = 1 + 2
    assert int(out[0, 0]) == int(rows[0, 0]) + 2          # +halo
    assert int(out[-1, -1]) == int(rows[-1, -1]) + 2
    assert int(out[0, -1]) == int(rows[0, -1]) + 2
    assert int(out[-1, 0]) == int(rows[-1, 0]) + 2
    # and the interior is the original, shifted by the pad
    assert torch.equal(out[pad:pad + ny, pad:pad + nx], (rows + 2).to(torch.int32))


# --------------------------------------------------------------------------- #
# determinism of the build itself
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_repeated_builds_are_identical_on_cuda():
    """Guards the synchronize. Without it the device copy can be read before it
    lands; a single build would still usually look right."""
    rows = _rows_2d(33)
    first = build_image_method_topo_rows(rows, abcn=6, halo=2, device="cuda")
    for _ in range(30):
        again = build_image_method_topo_rows(rows, abcn=6, halo=2, device="cuda")
        assert torch.equal(first, again)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def test_topography_forces_a_free_surface_even_if_asked_otherwise():
    method, fs, image = resolve_topo_method(
        topography=_rows_2d(), topo_method="auto", free_surface=False,
        supports_apm=False, is_curvilinear=False, equation_name="X")
    assert (method, fs, image) == ("image", True, True)


def test_image_method_takes_z_pad_once_apm_twice():
    """``image_method_active`` selects the LAYOUT -- it suppresses the top PML
    band -- so z loses abcn once under image and twice under APM. x is
    unaffected either way."""
    assert physical_extent((100, 60), abcn=10, ndim=2, image_method_active=True) == (90, 40)
    assert physical_extent((100, 60), abcn=10, ndim=2, image_method_active=False) == (80, 40)
