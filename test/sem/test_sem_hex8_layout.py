"""The Hex8Layout is the mesh minus its arrays -- and must stay exactly that.

`build_hex8_mesh` tiles a layout into the per-element-node arrays; the DD
strips read the layout instead and reconstruct what they need from it. Both
rest on three identities that are pinned here, bitwise, on the CPU:

* the layout's axes ARE the mesh's coordinate lines (`x_global` is a meshgrid
  of `x_axis`, so a coordinate is an axis value looked up by column);
* `hex8_ibool` IS the mesh's numbering;
* `geometry=False` changes nothing but the six coordinate arrays.

The negative control at the end shows the strip check that relies on these
identities actually fires when a strip's numbering is foreign.
"""
from __future__ import annotations

import pytest
import torch

from sweep.sem.mesh.hex8 import (Hex8Layout, build_hex8_mesh, hex8_ibool,
                                 hex8_layout)

ARGS = dict(nex=5, ney=3, nez=4, x_max=200.0, y_max=90.0, z_max=160.0,
            n_order=4)


def test_layout_axes_are_the_meshes_coordinate_lines():
    lay = hex8_layout(**ARGS)
    m = build_hex8_mesh(**ARGS)
    nz, ny, nx = lay.nz_global, lay.ny_global, lay.nx_global
    assert (nz, ny, nx) == (m.nez * m.n_order + 1, m.ney * m.n_order + 1,
                            m.nex * m.n_order + 1)
    assert torch.equal(lay.x_axis, m.x_global.reshape(nz, ny, nx)[0, 0, :])
    assert torch.equal(lay.y_axis, m.y_global.reshape(nz, ny, nx)[0, :, 0])
    assert torch.equal(lay.z_axis, m.z_global.reshape(nz, ny, nx)[:, 0, 0])
    # and every per-element-node coordinate is an axis value by column
    g = m.ibool
    assert torch.equal(m.x_e, lay.x_axis[g % nx])
    assert torch.equal(m.y_e, lay.y_axis[(g // nx) % ny])
    assert torch.equal(m.z_e, lay.z_axis[g // (nx * ny)])


def test_layout_scalars_equal_the_meshes():
    lay = hex8_layout(**ARGS)
    m = build_hex8_mesh(**ARGS)
    for k in ("n_order", "nex", "ney", "nez", "x_max", "y_max", "z_max",
              "dx_e", "dy_e", "dz_e", "jacobian_det", "n_global",
              "n_elements", "ngll"):
        assert getattr(lay, k) == getattr(m, k), k
    for k in ("xi", "w_ref", "deriv"):
        assert torch.equal(getattr(lay, k), getattr(m, k)), k


def test_hex8_ibool_is_the_meshes_numbering():
    m = build_hex8_mesh(**ARGS)
    assert torch.equal(hex8_ibool(m.nex, m.ney, m.nez, m.n_order), m.ibool)


def test_geometry_false_keeps_everything_but_the_coordinates():
    full = build_hex8_mesh(**ARGS)
    bare = build_hex8_mesh(**ARGS, geometry=False)
    assert torch.equal(full.ibool, bare.ibool)
    for k in ("n_global", "n_elements", "dx_e", "jacobian_det", "x_max"):
        assert getattr(full, k) == getattr(bare, k), k
    for k in ("x_e", "y_e", "z_e", "x_global", "y_global", "z_global"):
        assert getattr(bare, k) is None, k
        assert getattr(full, k) is not None, k


def test_mesh_carries_the_layout_it_was_tiled_from():
    m = build_hex8_mesh(**ARGS)
    lay = hex8_layout(**ARGS)
    assert isinstance(m.layout, Hex8Layout)
    for k in ("x_axis", "y_axis", "z_axis", "w_ref", "deriv"):
        assert torch.equal(getattr(m.layout, k), getattr(lay, k)), k
    assert m.layout.n_global == m.n_global
    assert build_hex8_mesh(**ARGS, geometry=False).layout is not None


def test_layout_holds_no_per_node_arrays():
    lay = hex8_layout(**ARGS)
    n_line = max(lay.nx_global, lay.ny_global, lay.nz_global)
    for k, v in vars(lay).items():
        if torch.is_tensor(v):
            assert v.numel() <= max(n_line, lay.ngll ** 2), (
                f"{k} has {v.numel()} elements -- a layout is O(nx+ny+nz)")


# --------------------------------------------------------------------------- #
# The strip check against the layout: positive and NEGATIVE control.
# --------------------------------------------------------------------------- #
def _strip(lay: Hex8Layout, ex0, ex1):
    from sweep.sem.dd import _assert_strip_ibool
    no, nex, ney, nez = lay.n_order, lay.nex, lay.ney, lay.nez
    nex_l = ex1 - ex0
    ml = build_hex8_mesh(nex=nex_l, ney=ney, nez=nez,
                         x_max=lay.x_max * nex_l / nex, y_max=lay.y_max,
                         z_max=lay.z_max, n_order=no, geometry=False)
    eids = torch.cat([torch.arange((ez * ney + ey) * nex + ex0,
                                   (ez * ney + ey) * nex + ex1)
                      for ez in range(nez) for ey in range(ney)])
    rows = torch.arange(lay.nz_global * lay.ny_global)
    cols = torch.arange(ex0 * no, ex1 * no + 1)
    gidx = (rows[:, None] * lay.nx_global + cols[None, :]).reshape(-1)
    return ml, eids, gidx, _assert_strip_ibool


@pytest.mark.parametrize("ex0,ex1", [(0, 5), (0, 2), (2, 5), (1, 4)])
def test_strip_numbering_composes_with_gidx_to_the_global_formula(ex0, ex1):
    lay = hex8_layout(**ARGS)
    ml, eids, gidx, check = _strip(lay, ex0, ex1)
    check(ml, lay, eids, gidx)                       # must not raise
    # and it agrees with the REAL global mesh restricted to the strip
    m = build_hex8_mesh(**ARGS)
    assert torch.equal(gidx[ml.ibool], m.ibool[eids])


def test_strip_check_fires_on_a_foreign_numbering():
    """Negative control: reverse the kx order inside every element -- a
    numbering a different element family could plausibly produce -- and the
    check must refuse it, or every field gathered through gidx[ml.ibool]
    would be silently misaligned."""
    lay = hex8_layout(**ARGS)
    ml, eids, gidx, check = _strip(lay, 1, 4)
    ml.ibool = ml.ibool.flip(-1)
    with pytest.raises(AssertionError, match="disagrees with the layout"):
        check(ml, lay, eids, gidx)
