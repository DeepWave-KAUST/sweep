"""The absorbing layer's material is a constant of the model.

The absorbing pad is filled by clamping the interpolation to the model edge.
Kept differentiable, that clamp handed the whole C-PML band's sensitivity to
the outermost ring of cells -- the strip around the edge of every SEM gradient
(measured 24% of the gradient's L2 on the ring, FD 4.2%). FD's EdgePadding
drops the pad in its backward; PropSEM now does the same, and these tests pin:

* a loss that sees only pad nodes has a zero model gradient, and an interior
  loss the gradient of exactly the interior nodes;
* the forward values do not move -- only the backward changed;
* a node ON the model edge counts as inside, on both elements that share it:
  the node-grid strip path (dd.py) has one copy of such a node, so this is
  what keeps the eager and impl='c' gradients equal;
* ``abcn=0`` has no pad and keeps the exact transpose;
* on a real run, the ring stops carrying the band.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D

NZ, NX, DH, NO, PAD = 17, 21, 10.0, 4, 2


def _prop(free_surface=("top",), shape=(NZ, NX), abcn=PAD * NO, **kw):
    return PropSEM(AcousticPotentialSEM2D, shape=shape, dh=DH, dt=1e-3,
                   nt=10, n_order=NO, free_surface=free_surface, abcn=abcn,
                   f_dominant=10.0, dev="cpu", dtype=torch.float64,
                   impl="eager", **kw)


def _grid(shape=(NZ, NX), seed=0):
    g = torch.Generator().manual_seed(seed)
    return (2000.0 + 300.0 * torch.rand(shape, generator=g,
                                        dtype=torch.float64))


def _outside_by_geometry(prop):
    """The pad nodes, from the mesh coordinates alone (not from the code
    under test): outside the model box in metres."""
    m = prop.mesh
    x0 = prop._pad0[-1] * DH
    z0 = prop._pad0[0] * DH
    x, z = m.x_e.reshape(-1), m.z_e.reshape(-1)
    eps = 1e-6 * DH
    return ((x < x0 - eps) | (x > x0 + (NX - 1) * DH + eps)
            | (z < z0 - eps) | (z > z0 + (NZ - 1) * DH + eps))


def test_pad_only_loss_has_no_model_gradient():
    prop = _prop()
    pad = _outside_by_geometry(prop)
    assert pad.any() and not pad.all()
    vp = _grid().requires_grad_(True)
    gll = prop._interp_to_gll(vp).reshape(-1)
    (gll * pad).sum().backward()
    assert torch.count_nonzero(vp.grad) == 0


def test_interior_loss_sees_exactly_the_interior_nodes():
    """The gradient equals the plain interpolant's gradient of the same loss
    restricted to the interior nodes -- the pad is cut, nothing else is."""
    prop = _prop()
    inside = ~_outside_by_geometry(prop)
    c = torch.randn(prop._gll_shape, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1))
    vp = _grid().requires_grad_(True)
    (g_cut,) = torch.autograd.grad((c * prop._interp_to_gll(vp)).sum(), vp)
    pad, prop._g_pad = prop._g_pad, None          # the old, uncut interpolant
    try:
        vq = _grid().requires_grad_(True)
        ref = (c.reshape(-1) * inside * prop._interp_to_gll(vq).reshape(-1))
        (g_ref,) = torch.autograd.grad(ref.sum(), vq)
    finally:
        prop._g_pad = pad
    assert torch.allclose(g_cut, g_ref, rtol=1e-13, atol=0)
    assert torch.count_nonzero(g_cut) > 0


def test_forward_values_do_not_move():
    prop = _prop()
    vp = _grid()
    with_grad = prop._interp_to_gll(vp.clone().requires_grad_(True))
    plain = prop._interp_to_gll(vp)
    assert torch.equal(with_grad.detach(), plain)
    # ... and the pad really reads the clamped edge, as it always did
    pad = _outside_by_geometry(prop).reshape(prop._gll_shape)
    assert float(plain[pad].min()) >= float(vp.min())
    assert float(plain[pad].max()) <= float(vp.max())


def test_edge_nodes_are_inside_on_both_elements():
    """The pad's first element shares its inner face with the model's last
    one; both copies of those nodes must be inside, or eager (element copies)
    and the strip path (one node per position) would cut different sets."""
    prop = _prop()
    prop._build_grid_to_gll()
    m = prop.mesh
    x, z = m.x_e.reshape(-1), m.z_e.reshape(-1)
    x0 = prop._pad0[-1] * DH
    on_left_edge = ((x - x0).abs() < 1e-9) & (z <= (NZ - 1) * DH + 1e-9)
    assert on_left_edge.sum() > NZ        # shared: more copies than nodes
    assert not prop._g_pad[on_left_edge].any()
    # and the cut set is exactly the geometric one
    assert torch.equal(prop._g_pad, _outside_by_geometry(prop))


def test_no_pad_keeps_the_exact_transpose():
    """``abcn=0`` has no absorbing layer: no node is outside the model,
    nothing is cut, and the interpolant's gradient is its exact linear
    transpose."""
    prop = _prop(abcn=0)
    prop._build_grid_to_gll()
    assert prop._g_pad is None
    c = torch.randn(prop._gll_shape, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))
    vp = _grid().requires_grad_(True)
    (g,) = torch.autograd.grad((c * prop._interp_to_gll(vp)).sum(), vp)
    d = torch.randn_like(vp)
    h = 1e-2
    fd = ((c * prop._interp_to_gll(vp.detach() + h * d)).sum()
          - (c * prop._interp_to_gll(vp.detach() - h * d)).sum()) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) <= 1e-9 * abs(float(fd))


def test_free_surface_side_has_no_pad():
    """The top carries the free surface and is not padded, so no node sits
    above the model: the top row is cut only where the side pads reach it."""
    prop = _prop()
    prop._build_grid_to_gll()
    x, z = prop.mesh.x_e.reshape(-1), prop.mesh.z_e.reshape(-1)
    assert float(z.min()) == 0.0
    x0 = prop._pad0[-1] * DH
    top_row = z.abs() < 1e-9
    over_model = (x >= x0 - 1e-9) & (x <= x0 + (NX - 1) * DH + 1e-9)
    assert not prop._g_pad[top_row & over_model].any()
    assert prop._g_pad[top_row & ~over_model].all()


def test_the_ring_no_longer_carries_the_band():
    """Notebook 43's case, smaller: a source and receivers along the whole top,
    the misfit to a two-layer model. Uncut -- the old behaviour, by switching
    the cut off -- the outermost ring carried 31% of the gradient's L2 here; cut,
    8% (the ring is 7% of the cells), against 4.2% vs 24% at notebook 43's
    size."""
    shape, nt = (49, 65), 600
    vp_true = torch.full(shape, 1500.0, dtype=torch.float64)
    vp_true[shape[0] // 2:] = 2500.0

    def run(cut):
        p = PropSEM(AcousticPotentialSEM2D, shape=shape, dh=DH, dt=1e-3, nt=nt,
                    n_order=NO, free_surface=False, abcn=8 * NO,
                    f_dominant=10.0, dev="cpu", dtype=torch.float64,
                    impl="eager")
        p._build_grid_to_gll()
        if not cut:
            p._g_pad = None
        t = np.arange(nt) * 1e-3
        a = (np.pi * 10.0 * (t - 0.12)) ** 2
        w = torch.tensor((1 - 2 * a) * np.exp(-a))
        src = np.array([[shape[1] // 2, 2]])
        rx = np.linspace(0, shape[1] - 1, 32).astype(np.int64)
        rec = np.stack([rx, np.full_like(rx, 4)], -1)[None]
        with torch.no_grad():
            obs = p(w, src, rec, models=[vp_true])
        vp = torch.full(shape, 1500.0, dtype=torch.float64, requires_grad=True)
        (0.5 * (p(w, src, rec, models=[vp]) - obs).pow(2).sum()).backward()
        g = vp.grad
        ring = torch.ones_like(g, dtype=torch.bool)
        ring[1:-1, 1:-1] = False
        return float(g[ring].norm() / g.norm())

    cut, uncut = run(True), run(False)
    assert cut < 0.15, f"ring share {cut:.3f} with the pad cut"
    assert uncut > 2.5 * cut, (
        f"uncut ring share {uncut:.3f} vs cut {cut:.3f}: this case does not "
        "put enough of the band on the ring to tell the two apart")


# ------------------------------ 3-D, node grid ------------------------------ #
def _prop3d(**kw):
    return PropSEM(AcousticSEM3D, shape=(9, 9, 13), dh=DH, dt=1e-3, nt=10,
                   n_order=NO, free_surface=False, abcn=NO, f_dominant=10.0,
                   dev="cpu", dtype=torch.float32, impl="eager", **kw)


def test_3d_interp_cuts_the_pad():
    prop = _prop3d()
    vp = (2000.0 + 100.0 * torch.rand(prop.shape)).requires_grad_(True)
    prop._build_grid_to_gll()
    pad = prop._g_pad.reshape(prop._gll_shape)
    assert pad.any() and not pad.all()
    (prop._interp_to_gll(vp) * pad).sum().backward()
    assert torch.count_nonzero(vp.grad) == 0


def test_3d_strip_tables_cut_the_same_nodes():
    """The single-strip/DD route interpolates on the node grid (dd.py) and
    must cut what the element path cuts, node for node."""
    from sweep.sem.dd import _strip_interp_tables
    prop = _prop3d()
    lay = prop.hex8_layout()

    class _Src:                                  # what the tables read
        dh, pad0, grid_shape, mesh = DH, prop._pad0, prop.shape, lay
        dtype = torch.float32

    _, _, offs = _strip_interp_tables(_Src, 0, len(lay.x_axis) - 1)
    oz, oy, ox = offs
    node_pad = (oz[:, None, None] | oy[None, :, None] | ox[None, None, :])
    # the element path's cut, scattered to the global node grid
    prop._build_grid_to_gll()
    g = torch.zeros(prop.mesh.n_global, dtype=torch.bool)
    g[prop.mesh.ibool.reshape(-1)] = prop._g_pad
    assert torch.equal(node_pad.reshape(-1), g)
