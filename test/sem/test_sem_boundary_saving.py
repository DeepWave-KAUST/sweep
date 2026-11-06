"""Boundary-saving correctness gate for the SEM backend, below PropSEM.

Asserts that the boundary-saved runner (reverse-time wavefield reconstruction in
the backward pass) gives a model gradient that matches the full-tape autograd
reference in the interior, while the backward-pass peak memory is strictly
lower. PropSEM's ``memory=BoundarySaving()`` is held to the full tape at 1e-12 in
test_sem_acoustic_bs_exact.py and test_sem_vs_bs_exact.py.
"""
import math

import pytest
import torch

from sweep.sem import ElasticVSSEM2DDeformed, build_axis_aligned_quad9
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.mesh.quad4 import build_quad4_mesh
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic_vs import build_simple_elastic_cpml
from sweep.sem.propagator import SEMPropagator
from sweep.sem.boundary_saving import run_boundary_saved, cpml_node_masks


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ricker(nt, dt, f0=10.0, t0=0.06, scale=1e6, dtype=torch.float64):
    t = torch.arange(nt, device=DEV, dtype=dtype) * dt - t0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a) * scale


def _cos(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return (x @ y / (x.norm() * y.norm() + 1e-300)).item()


# --------------------------------------------------------------------------- #
# 1. Low-level: exact interior gradient + memory reduction (needs CUDA).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(DEV != "cuda", reason="memory measurement needs CUDA")
def test_boundary_saving_interior_exact_and_saves_memory():
    nez, nex, no, dh, dt, nt = 30, 40, 4, 10.0, 0.0015, 500
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=(nex * no) * dh,
                            z_max=(nez * no) * dh, n_order=no,
                            dtype=torch.float64, device=DEV)
    vp0 = 1800 + 600 * mesh.z_e / (nez * no * dh)
    wav = _ricker(nt, dt)
    src_xz = [((nex * no // 2) * dh, (nez * no // 4) * dh)]
    rec_xz = [(x, 2 * dh) for x in (500.0, 800.0, 1100.0)]

    def setup(vp):
        cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                          pml_thickness_elements=8, c_max=2400., f_dominant=10.)
        eq = AcousticSEM2D.build(mesh, vp, dirichlet=("top",), cpml=cpml)
        return eq, SEMPropagator.build(eq, dt=dt, nt=nt,
                                       source_xz=src_xz, receiver_xz=rec_xz)

    ng = mesh.n_global
    rg_w = lambda p: (
        torch.stack([mesh.ibool[r.elem_id].reshape(-1) for r in p.receivers]).to(DEV),
        torch.stack([(r.w_eta[:, None] * r.w_xi[None, :]).reshape(-1)
                     for r in p.receivers]).to(device=DEV, dtype=torch.float64))

    # full tape
    vpf = vp0.clone().requires_grad_(True)
    eqf, pf = setup(vpf)
    spf = pf._build_source_patterns(ng, DEV, torch.float64)[0]   # single source field
    rg, rw = rg_w(pf)
    torch.cuda.reset_peak_memory_stats()
    st = eqf.init_state(dt, DEV, torch.float64); cols = []
    for n in range(nt):
        st = eqf.step(st, [spf * wav[n]], dt)          # positional src
        cols.append((st[0][rg] * rw).sum(1))           # field 0 = u
    torch.stack(cols, 1).pow(2).sum().backward()
    torch.cuda.synchronize()
    mem_full = torch.cuda.max_memory_allocated()
    g_full = vpf.grad.clone()

    # boundary saving
    vpb = vp0.clone().requires_grad_(True)
    eqb, pb = setup(vpb)
    spb = pb._build_source_patterns(ng, DEV, torch.float64)[0]
    torch.cuda.reset_peak_memory_stats()
    sb = run_boundary_saved(eqb, [spb], wav, rg, rw, dt, nt)[0]   # list src; {idx: seis}
    sb.pow(2).sum().backward()
    torch.cuda.synchronize()
    mem_bs = torch.cuda.max_memory_allocated()
    g_bs = vpb.grad.clone()

    cz, _ = cpml_node_masks(eqb)
    interior = torch.ones(ng, dtype=torch.bool, device=DEV); interior[cz] = False
    ni = interior[mesh.ibool]

    assert _cos(g_bs[ni], g_full[ni]) > 0.9999            # interior gradient exact
    assert (g_bs[ni] - g_full[ni]).norm() / g_full[ni].norm() < 1e-3
    assert mem_bs < 0.5 * mem_full                         # at least 2x memory cut


# --------------------------------------------------------------------------- #
# 2. ELASTIC velocity-stress (substep driver): interior gradient + memory.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(DEV != "cuda", reason="memory measurement needs CUDA")
def test_elastic_vs_boundary_saving():
    nex, nez, no, dt, nt = 40, 30, 4, 8e-4, 350
    mesh = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=1600.0, z_max=1200.0,
                                    n_order=no, dtype=torch.float64, device=DEV)
    full = lambda v: torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), v,
                                dtype=torch.float64, device=DEV)
    vp0, vs, rho = full(2400.0), full(1300.0), full(2000.0)
    wav = _ricker(nt, dt, f0=12.0, t0=0.05, scale=1e7)
    src_xz = [(800.0, 500.0)]
    rec_xz = [(600.0, 450.0), (1000.0, 450.0), (800.0, 650.0)]
    mt, pay = ["moment_tensor"], [(1.0, 1.0, 0.0)]

    def build(vp):
        cpml = build_simple_elastic_cpml(
            mesh, dt=dt, sides=("left", "right", "bottom"),
            pml_thickness_elements=8, c_max=2600.0, f_dominant=12.0)
        eq = ElasticVSSEM2DDeformed.build(mesh, vp, vs, rho, free_top=False, cpml=cpml)
        return eq, SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src_xz,
                                       receiver_xz=rec_xz, source_kinds=mt,
                                       source_payloads=pay)

    def loss_grad(memory):
        vp = vp0.clone().requires_grad_(True)
        eq, prop = build(vp)
        torch.cuda.reset_peak_memory_stats()
        out = prop.forward(wav, memory=memory)
        (out[..., 0].T ** 2 + out[..., 1].T ** 2).sum().backward()
        torch.cuda.synchronize()
        return vp.grad.clone(), torch.cuda.max_memory_allocated(), eq, out

    g_full, mem_full, eqf, out_f = loss_grad(None)
    g_bs, mem_bs, eqb, out_b = loss_grad("boundary")

    cz, _ = cpml_node_masks(eqb)
    interior = torch.ones(mesh.n_global, dtype=torch.bool, device=DEV)
    interior[cz] = False
    ni = interior[mesh.ibool]

    # forward seismograms identical; interior model gradient exact; memory cut.
    assert (out_b[..., 0].T - out_f[..., 0].T).norm() / out_f[..., 0].T.norm() < 1e-6
    assert _cos(g_bs[ni], g_full[ni]) > 0.999
    assert (g_bs[ni] - g_full[ni]).norm() / g_full[ni].norm() < 1e-2
    assert mem_bs < 0.5 * mem_full
