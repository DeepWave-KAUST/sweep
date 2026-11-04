"""The fluid-solid loads' C-PML stretching (``coupling.InterfaceCPML``).

Inside an absorbing band each domain's C-PML carries its stretching into the
interface terms: the solid load becomes ``(s_z*p n_x, s_x*p n_z)``, the fluid
load ``(1/s_x)*u_x n_x + (1/s_z)*u_z n_z``. With the plain terms there a sea
floor through a side band reflected several times what the rock alone does.

* eager: which points carry the stretching, and the sea floor through a side
  band against a box large enough to have no boundary (the plain terms against
  the stretched ones);
* CUDA: a sloped sea floor through the side bands under a multiaxial elastic
  profile, so all four of the stretched terms are live -- the forward and the
  gradients against eager, on the tape and on boundary saving.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import AcousticElasticSEM2D, PropSEM, SEMPropagator
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.mesh.quad9 import Quad9Mesh, quad9_from_control_nodes
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

needs_c = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")


class _Plain(AcousticElasticSEM2D):
    """The plain interface terms inside the bands as well (the old scheme)."""

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, vs_e, rho_e, *, dirichlet=("top",),
              domain=None, cpml_ac=None, cpml_el=None):
        eq = super().build(mesh, vp_e, vs_e, rho_e, dirichlet=dirichlet, domain=domain,
                           cpml_ac=cpml_ac, cpml_el=cpml_el)
        eq.ifc_cpml = None
        return eq


# --------------------------------------------------------------------------- #
# A sloped sea floor through the side bands, low-level build
# --------------------------------------------------------------------------- #
NEX, ROWS = 20, (6, 10)
XMAX, ZMAX = 1000.0, 800.0
DT, NT, F0 = 5e-4, 1200, 10.0
SRC = (300.0, 120.0)
REC = ([(float(x), 100.0) for x in np.linspace(100.0, 900.0, 9)]       # water
       + [(float(x), 500.0) for x in np.linspace(100.0, 900.0, 9)])    # rock


def _floor(x):
    return 300.0 + 0.25 * (x - 500.0)          # 175 m to 425 m, slope 0.25


def _two_layers(dtype):
    """Water over rock across ``_floor``, a flat top: each layer's control rows
    graded linearly in every control column."""
    xa = torch.linspace(0.0, XMAX, 2 * NEX + 1, dtype=dtype)
    tops = [torch.zeros_like(xa), _floor(xa), torch.full_like(xa, ZMAX)]
    z = [tops[k][None, :] + torch.linspace(0.0, 1.0, 2 * n + 1, dtype=dtype)[:, None]
         * (tops[k + 1] - tops[k])[None, :] for k, n in enumerate(ROWS)]
    zg = torch.cat([z[0], z[1][1:]], 0)
    iz = 2 * torch.arange(sum(ROWS))[:, None] + torch.arange(3)[None, :]
    ix = 2 * torch.arange(NEX)[:, None] + torch.arange(3)[None, :]
    take = lambda g: g[iz[:, None, :, None], ix[None, :, None, :]]
    control = torch.stack([take(xa[None, :].expand_as(zg)), take(zg)], -1)
    return quad9_from_control_nodes(control, n_order=4, dtype=dtype)


def _sloped(cls=AcousticElasticSEM2D, grad=False, dtype=torch.float32):
    m = _two_layers(dtype)
    wet = (torch.arange(m.n_elements) // NEX < ROWS[0]).reshape(-1, 1, 1)
    shp = (m.n_elements, m.ngll, m.ngll)
    mats = [torch.where(wet, w, r).to(dtype).expand(shp).contiguous()
            for w, r in ((1500.0, 3000.0), (0.0, 1700.0), (1020.0, 2300.0))]
    if grad:
        mats = [t.requires_grad_(True) for t in mats]
    sides = ("left", "right", "bottom")
    cp_ac = build_cpml(m, dt=DT, sides=sides, pml_thickness_elements=4,
                       c_max=1500.0, f_dominant=F0)
    # multiaxial: the side bands stretch z too, so the solid's s_z*p n_x is live
    cp_el = build_elastic_cpml(m, dt=DT, sides=sides, pml_thickness_elements=4,
                               c_max=3000.0, f_dominant=F0, mpml_ratio=0.2)
    eq = cls.build(m, *mats, dirichlet=("top",), cpml_ac=cp_ac, cpml_el=cp_el)
    return eq, mats


def _wav(dtype=torch.float32):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=dtype)


def _forward(eq, use_cuda, memory=None, dtype=torch.float32):
    prop = (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
        source_kinds=["scalar"], source_payloads=[(1e6,)], pml_guard="off")
    w = _wav(dtype).to("cuda" if use_cuda else "cpu")
    return prop.forward(w, memory=memory) if use_cuda else prop.forward(w)


def _rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm())


def test_band_points_carry_the_stretching_of_their_axes():
    """The table holds exactly the interface points stretched by either profile;
    a point's solid terms follow the elastic profile at its node, its fluid
    terms the acoustic one, and the loads reduce to the plain ones without a
    band."""
    eq, _ = _sloped(dtype=torch.float64)
    ic, (cp_ac, cp_el) = eq.ifc_cpml, eq.cpml
    g = eq.ifc.node_idx.reshape(-1)
    el = eq.domain.is_elastic
    on_nodes = lambda t: torch.zeros(eq.mesh.n_global, dtype=torch.float64).index_copy(
        0, eq.mesh.ibool[el].reshape(-1), t[el].reshape(-1))
    d_x, d_z = on_nodes(cp_el.d_x), on_nodes(cp_el.d_z)
    stretched = ((cp_ac.sigma_x != 0) | (cp_ac.sigma_z != 0) | (d_x != 0) | (d_z != 0))[g]
    assert torch.equal(torch.sort(ic.pt_flat).values, torch.nonzero(stretched).reshape(-1))
    assert torch.equal(ic.pt_g, g[ic.pt_flat])
    assert bool((d_z[ic.pt_g] != 0).any())            # the multiaxial side bands
    tab = ic.table()
    assert tab.shape == (ic.pt.numel(), 24)
    # the fluid's x stretch is the acoustic profile's sigma_x at the node, the
    # solid's x stretch A = d_x dt/2 with the elastic profile's d_x there
    assert torch.allclose(tab[:, 16], cp_ac.sigma_x[ic.pt_g])
    assert torch.allclose(tab[:, 2], 0.5 * DT * d_x[ic.pt_g])
    assert torch.allclose(tab[:, 9], 0.5 * DT * d_z[ic.pt_g])
    # every point of a sloped face has both normal components
    assert bool((tab[:, 0] != 0).all()) and bool((tab[:, 1] != 0).all())
    # no band, no stretching
    m = eq.mesh
    shp = (m.n_elements, m.ngll, m.ngll)
    wet = (torch.arange(m.n_elements) // NEX < ROWS[0]).reshape(-1, 1, 1)
    mats = [torch.where(wet, w, r).double().expand(shp).contiguous()
            for w, r in ((1500.0, 3000.0), (0.0, 1700.0), (1020.0, 2300.0))]
    assert AcousticElasticSEM2D.build(m, *mats, dirichlet=("top",)).ifc_cpml is None


def test_the_stretching_changes_the_records():
    """The sloped floor's records with and without the stretching differ well
    above the c-vs-eager tolerances below -- so those comparisons see it."""
    with torch.no_grad():
        a = _forward(_sloped()[0], False)
        b = _forward(_sloped(_Plain)[0], False)
    assert _rel(b, a) > 1e-3, _rel(b, a)


# --------------------------------------------------------------------------- #
# Absorption of a sea floor through the side band (eager, fp64)
# --------------------------------------------------------------------------- #
A_NO, A_DH, A_DT, A_F0, A_NT = 4, 10.0, 6e-4, 5.0, 1900
A_SHAPE, A_PAD = (81, 161), (180, 180)          # cells; the big box adds A_PAD
A_WATER, A_ROCK = (1500.0, 0.0, 1020.0), (3000.0, 1500.0, 2400.0)
A_DEPTH = 200.0


def _absorb(eq, absorbing):
    nz, nx = A_SHAPE if absorbing else (A_SHAPE[0] + A_PAD[0], A_SHAPE[1] + 2 * A_PAD[1])
    sx = 0 if absorbing else A_PAD[1]
    prop = PropSEM(eq, shape=(nz, nx), dh=A_DH, dt=A_DT, n_order=A_NO,
                   abcn=40 if absorbing else 0, f_dominant=A_F0, impl="eager",
                   dtype=torch.float64, free_surface=True, source_type=["chi"],
                   receiver_type=["p", "vx", "vz"], eager_options={"use_compile": False})
    m = prop.mesh
    wet = m.z_e.mean((1, 2)) < A_DEPTH
    E, n = m.n_elements, m.ngll
    models = [torch.where(wet, torch.tensor(w, dtype=torch.float64),
                          torch.tensor(r, dtype=torch.float64))
              .view(E, 1, 1).expand(E, n, n).contiguous() for w, r in zip(A_WATER, A_ROCK)]
    rx = np.arange(4, A_SHAPE[1] - 3, 12)
    rec = np.concatenate([np.stack([rx, np.full_like(rx, 19)], 1),       # 10 m above the floor
                          np.stack([rx, np.full_like(rx, 21)], 1)])      # 10 m below
    rec[:, 0] += sx
    src = np.array([[40 + sx, 16]])                                    # 40 m above the floor
    t = np.arange(A_NT) * A_DT
    a = (np.pi * A_F0 * (t - 1.2 / A_F0)) ** 2
    w = torch.tensor((1 - 2 * a) * np.exp(-a)) / (A_WATER[2] * A_WATER[0] ** 2)
    with torch.no_grad():
        return prop(w, src, rec[None], models=models)[0].numpy()


def _absorb_error(got, ref):
    per = np.linalg.norm(got - ref, axis=(0, 2)) / np.linalg.norm(ref, axis=(0, 2))
    return float(np.median(per)), float(per.max())


@pytest.mark.slow
def test_sea_floor_through_a_side_band_absorbs():
    """Records 10 m above and below a sea floor that runs into the side bands,
    against a box with no boundary within reach: the largest error per
    receiver is 1.2e-2 with the plain interface terms, 7e-4 with the stretched
    ones."""
    sea = _absorb(AcousticElasticSEM2D, False)
    plain = _absorb_error(_absorb(_Plain, True), sea)
    stretched = _absorb_error(_absorb(AcousticElasticSEM2D, True), sea)
    assert stretched[1] < 0.3 * plain[1] and stretched[1] < 1.5e-3, (stretched, plain)


# --------------------------------------------------------------------------- #
# CUDA against eager
# --------------------------------------------------------------------------- #
@needs_c
def test_c_forward_matches_eager_on_a_sloped_floor_in_the_bands():
    with torch.no_grad():
        oe = _forward(_sloped()[0], False)
        oc = _forward(_sloped()[0], True)
    for j, nm in enumerate(("chi", "u_x", "u_z")):
        assert _rel(oc[..., j], oe[..., j]) < 2e-5, (nm, _rel(oc[..., j], oe[..., j]))


def _grads(use_cuda, memory=None, dtype=torch.float32):
    eq, mats = _sloped(grad=True, dtype=dtype)
    o = _forward(eq, use_cuda, memory, dtype)
    loss = 0.0
    for j in range(o.shape[-1]):
        s = o[..., j]
        W = torch.linspace(0.4 + 0.1 * j, 1.6 - 0.1 * j, s.numel(),
                           device=s.device, dtype=s.dtype).reshape(s.shape)
        loss = loss + (W * s).sum()
    loss.backward()
    return [t.grad.detach().cpu() for t in mats]


@needs_c
@pytest.mark.parametrize("memory", [None, "boundary"], ids=["tape", "bs"])
def test_c_gradients_match_eager_on_a_sloped_floor_in_the_bands(memory):
    """Against the fp64 eager gradient: fp32 lands within 5e-4 of it on either
    backend, with or without the stretching, where the stretching itself moves
    the gradient by 5-7%."""
    ge = _grads(False, dtype=torch.float64)
    gc = _grads(True, memory)
    for nm, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _rel(c, e) < 1e-3, (nm, _rel(c, e))
