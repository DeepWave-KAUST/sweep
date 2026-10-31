"""PropSEM takes PropTorch's options, with PropTorch's meaning.

``shape`` is the model and ``abcn`` (grid cells, whole elements) the absorbing
layer outside it; ``free_surface`` is PropTorch's edge spec and ``topography``
its per-column surface rows; ``memory`` is ``Full`` / ``BoundarySaving`` /
``Ckpt``; ``impl=None`` is 'auto'; ``nt`` is the wavelet's length; ``device`` or
``dev``. Snapshots are on the grid, in PropTorch's layout. What PropSEM cannot
do is refused, never dropped.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt, Full
from sweep.sem import (PropSEM, AcousticSEM2D, AcousticSEM2DDeformed,
                       AcousticSEM3D, AcousticSEM3DDeformed,
                       AcousticPotentialSEM2D, build_topography_quad9)
from sweep.sem.prop_c import sem_c_available

NZ, NX, NO, DH, F0 = 25, 33, 4, 10.0, 15.0
DT, NT = 6e-4, 120
SRC = np.array([[NX // 2, 8]])
REC = np.array([[[ix, 4] for ix in range(4, NX - 4, 4)]])


def _wav(nt=NT, dtype=torch.float32):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=dtype)


def _prop(equation=AcousticSEM2D, shape=(NZ, NX), **kw):
    kw.setdefault("impl", "eager")
    return PropSEM(equation, shape=shape, dh=DH, dt=DT, n_order=NO,
                   f_dominant=F0, **kw)


def _vp(shape=(NZ, NX), dtype=torch.float32):
    return torch.full(shape, 2000.0, dtype=dtype)


def _hill_rows(nx=NX):
    """Fractional surface rows: a 3-row hill on a 2-row baseline."""
    x = np.arange(nx)
    return 2.0 + 3.0 * np.exp(-((x - nx / 2) / (nx / 6)) ** 2)


# ------------------------- the absorbing layer ------------------------------ #
def test_abcn_is_cells_outside_the_model():
    p = _prop(abcn=8, free_surface=True)
    assert (p.abcn, p.abcn_elements) == (8, 2)
    assert p._grid_shape() == (NZ + 8, NX + 16)       # none under the surface
    assert _prop()._grid_shape() == (NZ + 80, NX + 80)  # ten elements a side


@pytest.mark.parametrize("abcn,err,match", [
    (10, ValueError, "multiple of 4"),
    (-4, ValueError, ">= 0"),
    (True, TypeError, "number of grid cells"),
    (8.0, TypeError, "number of grid cells"),
    ((8, 8, 8, 8), NotImplementedError, "one abcn"),
])
def test_abcn_must_be_whole_elements(abcn, err, match):
    with pytest.raises(err, match=match):
        _prop(abcn=abcn)


@pytest.mark.parametrize("old,match", [("abcn_elements", "abcn= in grid cells"),
                                      ("model_padding", "always added outside")])
def test_a_renamed_option_says_what_replaced_it(old, match):
    with pytest.raises(TypeError, match=match):
        _prop(**{old: 2})


# --------------------------- the free surface ------------------------------- #
@pytest.mark.parametrize("spec", [True, "top", ["top"], {"top": True},
                                  (True, False, False, False)])
def test_free_surface_takes_proptorchs_specs(spec):
    assert _prop(abcn=8, free_surface=spec).free_surface == ("top",)


def test_free_surface_defaults_off_and_takes_several_edges():
    assert _prop(abcn=8).free_surface == ()
    p = _prop(abcn=8, free_surface=["left", "top"])
    assert p.free_surface == ("top", "left")
    assert p._grid_shape() == (NZ + 8, NX + 8)


def test_an_unknown_edge_is_refused():
    with pytest.raises(ValueError, match="unknown free-surface edge"):
        _prop(abcn=8, free_surface=["roof"])


# ----------------------------- topography ----------------------------------- #
def test_topography_rows_build_the_mesh_their_function_did():
    """The mesh's surface passes through the given rows at its control nodes
    (every second column at n_order=4) and is flat across the pad: the mesh
    the function the rows were sampled from builds, to rounding."""
    g = lambda x: (2.0 + 3.0 * torch.exp(-((x / DH - NX / 2) / (NX / 6)) ** 2)) * DH
    rows = g(torch.arange(NX, dtype=torch.float64) * DH) / DH
    p = _prop(AcousticSEM2DDeformed, abcn=8, topography=rows.numpy())
    assert p.free_surface == ("top",)                  # a topography is one
    x0, xm = 8 * DH, (NX - 1) * DH
    ref = build_topography_quad9(
        nex=p._elems_pad[1], nez=p._elems_pad[0], x_max=(NX - 1 + 16) * DH,
        z_bottom=(NZ - 1 + 8) * DH, n_order=NO, dtype=torch.float32,
        z_top_fn=lambda x: g((x.double() - x0).clamp(0.0, xm)).float())
    assert torch.allclose(p.mesh.z_e, ref.z_e, rtol=0, atol=1e-3)
    assert torch.equal(p.mesh.x_e, ref.x_e)


@pytest.mark.parametrize("topo,err,match", [
    (lambda x: x * 0 + 20.0, TypeError, "not a function"),
    (np.zeros(NX - 1), ValueError, "one surface row per column"),
    (np.full(NX, NZ - 1.0), ValueError, "inside the model"),
    (np.full(NX, -1.0), ValueError, "inside the model"),
])
def test_bad_topography_is_refused(topo, err, match):
    with pytest.raises(err, match=match):
        _prop(AcousticSEM2DDeformed, abcn=8, topography=topo)


def test_topography_needs_a_curved_mesh():
    with pytest.raises(ValueError, match="curved-element mesh"):
        _prop(AcousticSEM2D, abcn=8, topography=_hill_rows())


# --------------------------- nt, device, impl ------------------------------- #
def test_nt_is_the_wavelets_length():
    p = _prop(abcn=8, free_surface=True)
    a = p(_wav(80), SRC, REC, models=[_vp()])
    b = p(_wav(120), SRC, REC, models=[_vp()])
    assert (a.shape[1], b.shape[1], p.nt) == (80, 120, 120)
    assert torch.equal(b[:, :80], a)                   # same steps, same numbers


def test_a_given_nt_is_checked():
    with pytest.raises(ValueError, match="built with nt=100"):
        _prop(abcn=8, nt=100)(_wav(120), SRC, REC, models=[_vp()])


def test_device_and_dev_are_both_accepted():
    assert _prop(abcn=8, device="cpu").dev == "cpu"
    assert _prop(abcn=8, dev="cpu").dev == "cpu"
    assert _prop(abcn=8).dev == "cpu"


def test_impl_none_is_auto():
    p = PropSEM(AcousticSEM2D, shape=(NZ, NX), dh=DH, dt=DT, n_order=NO, abcn=8)
    assert p.impl == "eager"                           # on the CPU
    with pytest.raises(ValueError, match="impl must be"):
        _prop(abcn=8, impl="cuda")


@pytest.mark.skipif(not (torch.cuda.is_available() and sem_c_available()),
                    reason="needs CUDA and the SEM entries")
def test_impl_none_takes_the_kernels_on_cuda():
    p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                n_order=NO, abcn=8, device="cuda")
    assert p.impl == "c"


# ------------------------------ memory -------------------------------------- #
def test_memory_is_a_constructor_option():
    with pytest.raises(TypeError, match="constructor option"):
        _prop(abcn=8)(_wav(), SRC, REC, models=[_vp()], memory="boundary")
    with pytest.raises(TypeError, match="storage_dtype"):
        _prop(abcn=8)(_wav(), SRC, REC, models=[_vp()], bs_store_dtype="bf16")


def _record_and_grad(memory, src=SRC, rec=REC):
    p = _prop(abcn=8, free_surface=True, memory=memory, dtype=torch.float64)
    vp = _vp(dtype=torch.float64).requires_grad_(True)
    r = p(_wav(dtype=torch.float64), src, rec, models=[vp])
    r.square().sum().backward()
    return r.detach(), vp.grad


def test_boundary_saving_matches_the_full_tape():
    r0, g0 = _record_and_grad(Full())
    r1, g1 = _record_and_grad(BoundarySaving())
    assert torch.equal(r0, r1)
    assert float((g1 - g0).norm() / g0.norm()) < 1e-9


def test_ckpt_chunks_the_multi_shot_tape():
    src = np.array([[NX // 3, 8], [2 * NX // 3, 8]])
    r0, g0 = _record_and_grad(Full(), src, REC)
    r1, g1 = _record_and_grad(Ckpt(chunks=30), src, REC)
    assert torch.equal(r0, r1)
    assert float((g1 - g0).norm() / g0.norm()) < 1e-9


def test_storage_dtype_sets_the_ring():
    assert _prop(abcn=8, memory=BoundarySaving())._bs_store is None
    assert _prop(abcn=8, memory=BoundarySaving(storage_dtype="bf16"))._bs_store == "bf16"


@pytest.mark.parametrize("memory,match", [
    (BoundarySaving(storage="cpu"), "storage="),
    (BoundarySaving(tail_steps=10), "tail_steps="),
    (Ckpt(mode="recursive", count=2), "mode="),
])
def test_what_propsem_cannot_do_is_refused(memory, match):
    with pytest.raises(NotImplementedError, match=match):
        _prop(abcn=8, memory=memory)


def test_ckpt_chunks_the_single_shot_tape():
    r0, g0 = _record_and_grad(Full())
    r1, g1 = _record_and_grad(Ckpt(chunks=30))
    assert torch.equal(r0, r1)
    assert float((g1 - g0).norm() / g0.norm()) < 1e-9


def test_the_band_knobs_need_boundary_saving():
    with pytest.raises(ValueError, match="needs memory=BoundarySaving"):
        _prop(abcn=8, bs_band_mode="rim")
    with pytest.raises(ValueError, match="needs impl='c' and memory=BoundarySaving"):
        _prop(abcn=8, memory=BoundarySaving(), bs_store_dtype_elem="int8")


# ------------------------------ snapshots ----------------------------------- #
def test_snapshots_are_on_the_grid_in_proptorchs_layout():
    p = _prop(abcn=8, free_surface=True)
    _, snaps = p(_wav(), SRC, REC, models=[_vp()], return_wavefield=True,
                 snapshot_times=[40, 80])
    assert snaps.shape == (2, 1, 1, 1, NZ + 8, NX + 16)
    assert float(snaps.abs().max()) > 0


def _linear_check(p, coeffs, rows, ndim):
    """A linear field is in every element's basis, so on the grid it must be
    the field at the grid point: the element search, the local coordinates
    and the node layout checked at once. Above the surface the grid is 0."""
    m = p.mesh
    axes = (m.z_global, m.x_global) if ndim == 2 else (m.z_global, m.y_global, m.x_global)
    f = sum(c * a.double() for c, a in zip(coeffs, axes)) + 5.0
    got = p._on_grid(f[None]).reshape(p._grid_shape())
    pts = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) * DH
                           for n in p._grid_shape()], indexing="ij")
    want = sum(c * a for c, a in zip(coeffs, pts)) + 5.0
    if rows is None:
        assert torch.allclose(got, want, rtol=0, atol=1e-9)
        return
    # the mesh's surface is quadratic between its control nodes: keep half a
    # cell clear of it either way
    pad = p._pad0[1:]
    lat = np.meshgrid(*[np.clip(np.arange(n) - q, 0, rows.shape[i] - 1)
                        for i, (n, q) in enumerate(zip(p._grid_shape()[1:], pad))],
                      indexing="ij")
    surf = torch.as_tensor(rows[tuple(np.rint(a).astype(int) for a in lat)]) * DH
    wet, air = pts[0] > surf + 0.5 * DH, pts[0] < surf - 0.5 * DH
    assert wet.any() and air.any()
    assert torch.allclose(got[wet], want[wet], rtol=0, atol=1e-9)
    assert (got[air] == 0).all()


@pytest.mark.parametrize("eq,topo", [(AcousticSEM2D, False),
                                     (AcousticSEM2DDeformed, True)])
def test_grid_sampling_is_exact_on_a_linear_field(eq, topo):
    rows = _hill_rows() if topo else None
    p = _prop(eq, abcn=8, topography=rows, free_surface=True, dtype=torch.float64)
    _linear_check(p, (-1.3, 0.7), rows, 2)


@pytest.mark.parametrize("eq,topo", [(AcousticSEM3D, False),
                                     (AcousticSEM3DDeformed, True)])
def test_grid_sampling_is_exact_on_a_linear_field_in_3d(eq, topo):
    shape = (13, 9, 17)
    y, x = np.meshgrid(np.arange(9), np.arange(17), indexing="ij")
    rows = (2.0 + 2.0 * np.exp(-((x - 8) / 3.0) ** 2 - ((y - 4) / 3.0) ** 2)
            if topo else None)
    p = _prop(eq, shape=shape, abcn=4, topography=rows, free_surface=True,
              dtype=torch.float64)
    _linear_check(p, (-1.3, 0.4, 0.7), rows, 3)
