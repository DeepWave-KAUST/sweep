"""Rim-only boundary saving:
``PropSEM(memory=BoundarySaving(), bs_band_mode='rim')``.

Boundary saving normally stores the whole C-PML band. Rim mode stores only
the band's first element layer, the one touching the model: one element
thick instead of ``abcn``. That is everything the model gradient reads -- the
model's own nodes are rebuilt by the reverse march from the layer's edge
line, and the edge nodes' material is shared with the layer's elements -- so
the gradient must equal the full tape over the WHOLE model, edge included.

It used to store the model's outermost element instead (the ``np.gradient``
derivative profiles mark the model's edge line as absorbing, and the old rim
subtracted that set), with the edge line zeroed: cos 0.65 uncropped against
the full tape on impl='c', exact only two elements in. These tests pin the
fix.

The configuration matters more than anything else here. With the source centred,
a free surface and a short record, the absorbing band barely sees the wavefield
and even a broken rim agrees with the full tape to 1e-4 everywhere. So the
agreement tests run a discriminating case: source at 15% of X so the wavefield
hits the left band head-on, no free surface so the band is 4-sided, and a
1500/4500 contrast to drive strong reflections back into it;
``test_the_configuration_actually_discriminates`` checks that storing less
than the layer is visibly wrong on it.

``dt`` comes from the SMALLEST GLL spacing inside an element (~0.172 x element),
not from the nominal node spacing: an FD-style CFL diverges into all-NaN here.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import PropSEM
import sweep.sem.boundary_saving as bsm
from sweep.sem.boundary_saving import cpml_node_masks
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, F0, ABCN, N_ORDER = 49, 65, 20.0, 8.0, 6, 4
NT, VP_LO, VP_HI, SRC_X_FRAC = 900, 1500.0, 4500.0, 0.15
DT = 0.30 * (0.172 * N_ORDER * DH) / VP_HI     # GLL-limited, with headroom


def _vp():
    z, x = np.arange(NZ)[:, None], np.arange(NX)[None, :]
    return (VP_LO + (VP_HI - VP_LO) * z / (NZ - 1)
            + 0.08 * (VP_HI - VP_LO) * np.sin(x / 9.0)).astype(np.float32)


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def _prop(impl, mode, memory=None):
    # receiver_type=["p"]: the pressure, the FWI record. Under PropSEM's
    # pressure-source convention chi is the pressure's double time integral,
    # a near-DC tail whose gradient is not what these gates measure.
    return PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=N_ORDER, free_surface=False,
                   abcn=ABCN * N_ORDER, f_dominant=F0, dev="cuda",
                   dtype=torch.float32, impl=impl, memory=memory,
                   bs_band_mode=mode, receiver_type=["p"])


def _run(impl, memory, mode):
    p = _prop(impl, mode, memory)
    src = np.array([[int(SRC_X_FRAC * NX), 4]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 4)], -1)[None]
    vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
    s = p(_wav(), src, rec, models=[vp])
    assert torch.isfinite(s).all(), "forward diverged (check the GLL CFL)"
    (s ** 2).sum().backward()
    g = vp.grad.detach().cpu().numpy().astype(np.float64)
    assert np.isfinite(g).all(), "gradient not finite"
    del p, vp, s
    gc.collect()
    torch.cuda.empty_cache()
    return g


def _cos(a, b, crop=0):
    sl = (slice(crop, NZ - crop) if crop else slice(None),
          slice(crop, NX - crop) if crop else slice(None))
    u, v = a[sl].reshape(-1), b[sl].reshape(-1)
    return float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))


# ------------------------------ exactness --------------------------------- #
@pytest.mark.parametrize("impl", ["c", "eager"])
def test_rim_matches_full_tape_up_to_the_model_edge(impl):
    """No crop: the outermost model pixels are where the old rim was wrong
    (uncropped cos 0.65, impl='c', on the notebook-54 case)."""
    ref = _run(impl, Full(), "full")
    rim = _run(impl, BoundarySaving(), "rim")
    c = _cos(rim, ref)
    assert c >= 1 - 1e-9, f"uncropped grad 1-cos={1 - c:.2e}"
    rel = np.abs(rim - ref).max() / np.abs(ref).max()
    assert rel < 1e-4, f"max rel error {rel:.2e}"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_full_band_mode_is_still_exact_everywhere(impl):
    """The default must not have moved: exact including the model edge."""
    ref = _run(impl, Full(), "full")
    full = _run(impl, BoundarySaving(), "full")
    assert _cos(full, ref) >= 1 - 1e-9, f"grad cos={_cos(full, ref):.10f}"


def test_the_configuration_actually_discriminates(monkeypatch):
    """Without this the exactness test above would pass on a broken rim.

    Store only the model's edge line -- enough to rebuild the interior, but
    not the band elements the edge nodes share their material with. On a case
    that drives energy into the band that must be visibly wrong."""
    real = bsm.band_from_active

    def edge_line_only(active, elem_nodes, mode="full", model=None):
        band, has = real(active, elem_nodes, mode, model)
        if mode == "rim":
            band &= model
        return band, has

    monkeypatch.setattr(bsm, "band_from_active", edge_line_only)
    ref = _run("c", Full(), "full")
    rim = _run("c", BoundarySaving(), "rim")
    assert _cos(rim, ref) < 0.99, (
        f"edge-line-only cos={_cos(rim, ref):.6f} -- the band sees too little "
        "energy for the exactness test to discriminate")


# ------------------------------ what is stored ----------------------------- #
def _eq(mode):
    p = _prop("c", mode, BoundarySaving())
    return p, p._build_eq(torch.tensor(_vp(), device="cuda"))


def _node_xz(p, eq):
    """Global node coordinates in model pixels (negative / beyond n-1 in the
    band)."""
    m = eq.mesh
    x = torch.zeros(m.n_global, dtype=torch.float64, device="cuda")
    z = torch.zeros_like(x)
    x[m.ibool.reshape(-1)] = m.x_e.reshape(-1).double()
    z[m.ibool.reshape(-1)] = m.z_e.reshape(-1).double()
    return x / DH - p._pad0[1], z / DH - p._pad0[0]


def test_rim_is_the_first_band_element_layer():
    """The stored set, by geometry: every node within one element outside the
    model edge (the layer, edge line included), and not one model node off
    the edge -- the model's outermost element is rebuilt, not stored."""
    p, eq = _eq("rim")
    x, z = _node_xz(p, eq)
    tol = 1e-3          # float32 coordinates; the nearest GLL node is 0.69 px off
    # distance outside the model box, in pixels (0 inside, Chebyshev outside)
    out = torch.stack([-x, x - (NX - 1), -z, z - (NZ - 1)]).amax(0).clamp_min(0)
    on_edge = ((x.abs() < tol) | ((x - (NX - 1)).abs() < tol)
               | (z.abs() < tol) | ((z - (NZ - 1)).abs() < tol)) & (out < tol)
    inside = (out < tol) & ~on_edge
    layer = (out > tol) & (out < N_ORDER + tol)
    stored = torch.zeros_like(inside)
    stored[cpml_node_masks(eq)[1]] = True
    assert not bool((stored & inside).any()), "rim stores model nodes"
    assert bool(stored[on_edge].all()), "rim misses part of the model edge"
    assert bool(stored[layer].all()), "rim misses part of the band's first layer"
    assert int(stored.sum()) == int(on_edge.sum() + layer.sum()), (
        "rim stores band nodes beyond the first element layer")


def test_rim_ignores_profile_round_off_at_the_model_edge():
    """The profile is zero on the model side only up to round-off: measured
    b_z = -3.8e-16 on the top edge line and dbz_dz = 1.8e-17 one GLL node into
    the model. A node-by-node b != 0 test flipped edge nodes into the band,
    and the derivative's crumb made model nodes 'active', so the eager
    reverse zeroed nodes the rim never restores. Put such crumbs on the left
    edge line and the next node line in: neither set may move."""
    p, eq = _eq("rim")
    zero0, save0 = (t.clone() for t in cpml_node_masks(eq))
    x, _ = _node_xz(p, eq)
    edge = (x.abs() < 1e-3)
    next_in = ((x - 0.6907).abs() < 1e-2)
    cp = eq.cpml
    cp.b_x = cp.b_x.clone()
    cp.dbx_dx = cp.dbx_dx.clone()
    cp.b_x[edge & (cp.b_x == 0)] = -3e-16
    cp.dbx_dx[next_in] = 2e-17
    zero1, save1 = cpml_node_masks(eq)
    assert torch.equal(save0, save1), "round-off moved the stored set"
    assert torch.equal(zero0, zero1), "round-off moved the zeroed set"


def test_rim_stores_far_fewer_nodes():
    full_idx = cpml_node_masks(_eq("full")[1])[1]
    rim_idx = cpml_node_masks(_eq("rim")[1])[1]
    assert rim_idx.numel() < full_idx.numel() / 4, (
        f"rim {rim_idx.numel()} vs band {full_idx.numel()} nodes")


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="bs_band_mode"):
        _prop("c", "ring")
