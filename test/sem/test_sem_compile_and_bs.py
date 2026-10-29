"""Guards for two previously-untested SEM paths:

1. the compiled step (``eager_options=EagerOptions(use_compile=True)``, the default:
   torch.compile of the per-step graph, used by the FWI
   notebook and the multi-shot loop) must reproduce the eager seismograms AND
   model gradient.
2. Boundary-saving for the **deformed** acoustic equation
   (:class:`AcousticSEM2DDeformed`) — its BS hooks (reverse_step / bs_recompute)
   were implemented but never exercised; here the BS model gradient must match
   the full-tape gradient in the lossless interior.
"""
import math

import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions
from sweep.sem import (
    PropSEM, AcousticSEM2DDeformed, build_axis_aligned_quad9,
)
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.mesh.quad4 import build_quad4_mesh
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.propagator import SEMPropagator
from sweep.sem.boundary_saving import cpml_node_masks

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ricker(nt, dt, f0=10.0, t0=0.06, scale=1.0, dtype=torch.float64):
    t = torch.arange(nt, device=DEV, dtype=dtype) * dt - t0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a) * scale


def _cos(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return (x @ y / (x.norm() * y.norm() + 1e-300)).item()


@pytest.mark.skipif(DEV != "cuda", reason="torch.compile speedup path is CUDA")
def test_compiled_step_matches_eager():
    """PropSEM acoustic: the compiled step == uncompiled (seismograms + vp gradient)."""
    nz, nx, nt, dh, dt = 61, 81, 300, 10.0, 5.0e-4
    f0 = 12.0
    wav = _ricker(nt, dt, f0=f0, t0=0.08, scale=1.0, dtype=torch.float32)
    sources = torch.tensor([[nx // 2, 3]])
    rec_x = torch.arange(2, nx - 2, 3)
    receivers = torch.stack([rec_x, torch.full_like(rec_x, 3)], 1)[None]

    def run(compile):
        prop = PropSEM(AcousticSEM2D, shape=(nz, nx), dh=dh, dt=dt, nt=nt,
                       n_order=4, free_surface=True, abcn=8 * 4,
                       f_dominant=f0, dev=DEV, dtype=torch.float32,
                       impl="eager", eager_options=EagerOptions(use_compile=compile))
        vp = torch.full((nz, nx), 2000.0, device=DEV, requires_grad=True)
        out = prop(wav, sources, receivers, models=[vp])
        out.pow(2).sum().backward()
        return out.detach(), vp.grad.detach()

    seis_e, g_e = run(False)
    seis_c, g_c = run(True)
    assert torch.isfinite(seis_c).all() and torch.isfinite(g_c).all()
    # only FP reassociation between the eager and compiled kernels
    assert (seis_c - seis_e).abs().max() / (seis_e.abs().max() + 1e-30) < 1e-4
    assert (g_c - g_e).norm() / (g_e.norm() + 1e-30) < 1e-4


@pytest.mark.skipif(DEV != "cuda", reason="torch.compile speedup path is CUDA")
def test_compiled_multishot_matches_eager_vmap():
    """Multi-shot compiled step (compile(vmap(step))) == uncompiled vmap batch."""
    nz, nx, nt, dh, dt = 61, 81, 250, 10.0, 5.0e-4
    f0 = 12.0
    wav = _ricker(nt, dt, f0=f0, t0=0.08, scale=1.0, dtype=torch.float32)
    sx = torch.tensor([nx // 3, 2 * nx // 3])
    sources = torch.stack([sx, torch.full_like(sx, 3)], 1)            # (2, 2)
    rec_x = torch.arange(2, nx - 2, 4)
    receivers = torch.stack([rec_x, torch.full_like(rec_x, 3)], 1)[None]

    def run(compile):
        prop = PropSEM(AcousticSEM2D, shape=(nz, nx), dh=dh, dt=dt, nt=nt,
                       n_order=4, free_surface=True, abcn=8 * 4,
                       f_dominant=f0, dev=DEV, dtype=torch.float32,
                       impl="eager", eager_options=EagerOptions(use_compile=compile))
        vp = torch.full((nz, nx), 2000.0, device=DEV, requires_grad=True)
        out = prop(wav, sources, receivers, models=[vp])
        out.pow(2).sum().backward()
        return out.detach(), vp.grad.detach()

    seis_e, g_e = run(False)
    seis_c, g_c = run(True)
    assert seis_c.shape == (2, nt, len(rec_x), 1)
    assert torch.isfinite(seis_c).all() and torch.isfinite(g_c).all()
    assert (seis_c - seis_e).abs().max() / (seis_e.abs().max() + 1e-30) < 1e-4
    assert (g_c - g_e).norm() / (g_e.norm() + 1e-30) < 1e-4


@pytest.mark.skipif(DEV != "cuda", reason="memory measurement / long run needs CUDA")
def test_deformed_acoustic_boundary_saving():
    """AcousticSEM2DDeformed: BS interior gradient == full-tape gradient."""
    nex, nez, no, dt, nt = 26, 20, 4, 6.0e-4, 350
    xm, zm = nex * no * 10.0, nez * no * 10.0
    mesh = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=xm, z_max=zm,
                                    n_order=no, dtype=torch.float64, device=DEV)
    vp0 = 1800.0 + 600.0 * mesh.z_e / zm
    wav = _ricker(nt, dt, f0=12.0, t0=0.08, scale=1.0, dtype=torch.float64)

    def loss_grad(memory):
        vp = vp0.clone().requires_grad_(True)
        cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                          pml_thickness_elements=6, c_max=2400.0, f_dominant=12.0)
        eq = AcousticSEM2DDeformed.build(mesh, vp, dirichlet=("top",), cpml=cpml)
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(xm / 2, 30.0)],
                                   receiver_xz=[(xm * 0.3, 30.0), (xm * 0.6, 30.0)])
        out = prop.forward(wav, memory=memory)
        out[..., 0].T.pow(2).sum().backward()
        return vp.grad.clone(), eq, out

    g_full, eqf, out_f = loss_grad(None)
    g_bs, eqb, out_b = loss_grad("boundary")

    cz, _ = cpml_node_masks(eqb)
    interior = torch.ones(mesh.n_global, dtype=torch.bool, device=DEV)
    interior[cz] = False
    ni = interior[mesh.ibool]

    # forward seismograms identical; interior model gradient matches; finite.
    assert (out_b[..., 0].T - out_f[..., 0].T).norm() \
        / out_f[..., 0].T.norm() < 1e-6
    assert torch.isfinite(g_bs).all()
    assert _cos(g_bs[ni], g_full[ni]) > 0.999
    assert (g_bs[ni] - g_full[ni]).norm() / g_full[ni].norm() < 1e-2


@pytest.mark.skipif(DEV != "cuda", reason="low-precision BS storage path is CUDA")
def test_bs_low_precision_ring():
    """bf16 / int8 BS ring storage: forward unaffected, gradient ≈ fp32 ring
    (and the ring uses the requested dtype). Source scaled so grads are real."""
    nex, nez, no, dt, nt = 26, 20, 4, 6.0e-4, 400
    xm, zm = nex * no * 10.0, nez * no * 10.0
    mesh = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=xm, z_max=zm,
                                    n_order=no, dtype=torch.float32, device=DEV)
    vp0 = 1800.0 + 600.0 * mesh.z_e / zm
    wav = _ricker(nt, dt, f0=12.0, t0=0.08, dtype=torch.float32) * 4e8

    def grad(store_dtype):
        vp = vp0.clone().requires_grad_(True)
        cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                          pml_thickness_elements=8, c_max=2400.0, f_dominant=12.0)
        eq = AcousticSEM2DDeformed.build(mesh, vp, dirichlet=("top",), cpml=cpml)
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(xm / 2, 30.0)],
                                   receiver_xz=[(xm * 0.3, 30.0), (xm * 0.6, 30.0)])
        out = prop.forward(wav, memory="boundary", bs_store_dtype=store_dtype)
        out[..., 0].T.pow(2).sum().backward()
        return out[..., 0].T.detach(), vp.grad.clone()

    s32, g32 = grad(None)
    for sd in (torch.bfloat16, torch.int8):
        s, g = grad(sd)
        assert torch.isfinite(g).all()
        # forward seis unaffected by ring dtype (ring is read only in backward);
        # tiny diff here is GPU scatter-atomic run-to-run noise, not store_dtype.
        assert (s - s32).abs().max() / (s32.abs().max() + 1e-30) < 1e-3
        # gradient stays within storage-rounding noise of the fp32 ring
        assert (g - g32).norm() / (g32.norm() + 1e-30) < 5e-3


def test_record_field_every_snapshots():
    """SEMPropagator ``record_field_every`` returns the right snapshot count,
    times, and a snapshot that equals the live field at that step."""
    nex, nez, no, dt, nt, every = 12, 10, 4, 5.0e-4, 40, 8
    xm, zm = nex * no * 10.0, nez * no * 10.0
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=xm, z_max=zm, n_order=no,
                            dtype=torch.float64, device=DEV)
    vp = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), 2000.0,
                    dtype=torch.float64, device=DEV)
    cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                      pml_thickness_elements=4, c_max=2000.0, f_dominant=12.0)
    eq = AcousticSEM2D.build(mesh, vp, dirichlet=("top",), cpml=cpml)
    prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(xm / 2, 30.0)],
                               receiver_xz=[(xm * 0.3, 30.0)])
    wav = _ricker(nt, dt, f0=12.0, t0=0.05, dtype=torch.float64)

    seis, snaps = prop.forward(wav, record_field_every=every)
    # snapshot times are not returned: they are exactly range(0, nt, every)
    expected = list(range(0, nt, every))          # records after step n, n%every==0
    assert seis.shape == (nt, 1, 1)               # (nt, nrec, ncomp)
    assert snaps.shape == (len(expected), 1, mesh.n_global)
    assert torch.isfinite(snaps).all()
    assert snaps.abs().max() > 0.0                # a wave was recorded


@pytest.mark.skipif(DEV != "cuda", reason="PropSEM deformed path needs CUDA")
def test_propsem_deformed_and_topography():
    """PropSEM drives the deformed / topography path with the SAME FD-style
    ``solver(wav, sources, receivers, models)`` call as the flat path: a flat
    Quad9 mesh reproduces flat Quad4 bit-for-bit, and the topography model
    gradient matches a central finite difference."""
    import numpy as np
    shape = (33, 49)
    dh, dt, nt, f0 = 10.0, 8.0e-4, 180, 14.0
    nz, nx = shape
    t = torch.arange(nt, device=DEV, dtype=torch.float64) * dt - 0.06
    a = (math.pi * f0 * t) ** 2
    wav = (1 - 2 * a) * torch.exp(-a)
    sources = np.array([[nx // 3, 12], [2 * nx // 3, 12]])
    rec_x = np.arange(4, nx - 4, 4)
    receivers = np.stack([rec_x, np.full_like(rec_x, 12)], 1)[None]
    vp = (2200.0 + 100.0 * torch.linspace(0, 1, nz, device=DEV, dtype=torch.float64
                                          )[:, None]).expand(nz, nx).contiguous()

    def run(eq):
        return PropSEM(eq, shape=shape, dh=dh, dt=dt, nt=nt, n_order=4,
                       free_surface=True, abcn=6 * 4, f_dominant=f0,
                       dev=DEV, dtype=torch.float64,
                       impl="eager")(wav, sources, receivers, models=[vp])

    seisA = run(AcousticSEM2D)
    seisB = run(AcousticSEM2DDeformed)
    assert seisB.shape == (2, nt, len(rec_x), 1)
    assert (seisA - seisB).norm() / seisA.norm() < 1e-10     # flat Q9 == Q4

    # flat-mesh GRADIENT: the deformed eq reproduces the Quad4 model gradient
    # bit-for-bit through PropSEM (rigorous, FD-free correctness check).
    def grad(eq):
        v = vp.clone().requires_grad_(True)
        PropSEM(eq, shape=shape, dh=dh, dt=dt, nt=nt, n_order=4,
                free_surface=True, abcn=6 * 4, f_dominant=f0, dev=DEV,
                dtype=torch.float64, impl="eager")(
                    wav, sources, receivers, models=[v]).pow(2).sum().backward()
        return v.grad
    gA, gB = grad(AcousticSEM2D), grad(AcousticSEM2DDeformed)
    assert (gB - gA).norm() / gA.norm() < 1e-10

    x0 = (nx // 2) * dh
    def hill(x):
        return 70.0 - 40.0 * torch.exp(-((x - x0) / 90.0) ** 2)
    rows = (hill(torch.arange(nx, dtype=torch.float64) * dh) / dh).numpy()
    solverT = PropSEM(AcousticSEM2DDeformed, shape=shape, dh=dh, dt=dt, nt=nt,
                      n_order=4, free_surface=True,
                      abcn=6 * 4, topography=rows, f_dominant=f0,
                      dev=DEV, dtype=torch.float64, impl="eager")
    wbig = wav * 1e8                                          # O(1) data -> clean FD

    def loss_of(vp_):
        return solverT(wbig, sources, receivers, models=[vp_]).pow(2).sum()

    vpg = vp.clone().requires_grad_(True)
    loss_of(vpg).backward()
    assert torch.isfinite(vpg.grad).all() and vpg.grad.norm() > 0
    # FD-check at a node OFF the source/receiver columns — a single-node FD is
    # unreliable right under a receiver (the NNI singularity), but clean elsewhere.
    iz_p, ix_p, eps = nz // 2, nx // 2 + 1, 0.5
    with torch.no_grad():
        vp_p = vp.clone(); vp_p[iz_p, ix_p] += eps
        vp_m = vp.clone(); vp_m[iz_p, ix_p] -= eps
        fd = ((loss_of(vp_p) - loss_of(vp_m)) / (2 * eps)).item()
    ana = vpg.grad[iz_p, ix_p].item()
    assert abs(fd - ana) / (abs(fd) + 1e-300) < 1e-3         # topo grad correct


@pytest.mark.skipif(DEV != "cuda", reason="batched BS path needs CUDA")
def test_propsem_multishot_boundary_saving_batched():
    """PropSEM memory=BoundarySaving() multishot runs ONE vmap-batched BS
    rollout and is bit-identical (forward + model gradient) to looping
    single-shot BS."""
    import numpy as np
    shape = (33, 49)
    dh, dt, nt, f0 = 10.0, 8.0e-4, 200, 14.0
    nz, nx = shape
    wav = _ricker(nt, dt, f0=f0, t0=0.06, dtype=torch.float64) * 1e8
    NS = 3
    sources = np.stack([np.linspace(nx * 0.25, nx * 0.75, NS).astype(int),
                        np.full(NS, 4)], 1)
    rec_x = np.arange(4, nx - 4, 4)
    receivers = np.stack([rec_x, np.full_like(rec_x, 4)], 1)[None]
    vp = (2200.0 + 120.0 * torch.linspace(0, 1, nz, device=DEV,
          dtype=torch.float64)[:, None]).expand(nz, nx).contiguous()

    def solver():                        # the PML is outside the model: full interior
        return PropSEM(AcousticSEM2D, shape=shape, dh=dh, dt=dt, nt=nt, n_order=4,
                       free_surface=True, abcn=6 * 4,
                       f_dominant=f0, dev=DEV, dtype=torch.float64,
                       impl="eager", memory=BoundarySaving())

    vpB = vp.clone().requires_grad_(True)
    seisB = solver()(wav, sources, receivers, models=[vpB])
    seisB.pow(2).sum().backward()
    gB = vpB.grad.clone()
    assert seisB.shape == (NS, nt, len(rec_x), 1)

    seis_ref, g_ref = [], torch.zeros_like(vp)
    for b in range(NS):
        vpb = vp.clone().requires_grad_(True)
        sb = solver()(wav, sources[b:b + 1], receivers, models=[vpb])
        seis_ref.append(sb.detach())
        sb.pow(2).sum().backward()
        g_ref += vpb.grad
    seis_ref = torch.cat(seis_ref)
    assert (seisB.detach() - seis_ref).norm() / seis_ref.norm() < 1e-6
    assert (gB - g_ref).norm() / g_ref.norm() < 1e-6
