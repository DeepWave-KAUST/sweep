"""Boundary saving for the 2-D elastic C-PML equation (ElasticSEM2DDeformed).

The interior 2nd-order displacement leapfrog reverses exactly by swap2nd; the
CPML band is restored from the per-step rings. Two pieces are load-bearing and
each is regression-tested here:

* ``node_in_pml`` band detection — the SPECFEM-style elastic profile carries no
  ``b_*`` coefficients, so without it ``cpml_node_masks`` reported an EMPTY band
  and boundary saving silently skipped every band restore.
* ``bs_grad_fields`` — the CPML memory recursion carries a real adjoint. Dropping
  it (the ``boundary_related`` default) is fine while the band is quiet but costs
  ~1% of the gradient once surface waves load the PML (cos 0.988 at nt=1200).

Compared against the full autograd tape: cos >= 1-1e-4 on the forward and on
dL/d{vp, vs, rho}, with and without topography.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import SEMPropagator
from sweep.sem.boundary_saving import cpml_node_masks, cpml_band_elements
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_c = pytest.mark.skipif(not sem_c_available(),
                             reason="needs the compiled sweep._C SEM binding")

NEX, NEZ, XMAX, ZMAX = 22, 16, 1100.0, 800.0
DT, F0, TSH = 7e-4, 13.0, 0.10
VP0, VS0, RHO0 = 2400.0, 1250.0, 2000.0
SIDES, NPML = ("left", "right", "bottom"), 5
MT = [(1e6, 1e6, 0.0)]
SRC = (XMAX / 2, 80.0)
REC = [(float(x), 120.0) for x in np.linspace(280.0, XMAX - 280.0, 18)]


def _mesh(topo, dev, dt_t):
    if not topo:
        return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                        n_order=4, dtype=dt_t, device=dev)
    return build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX, n_order=4, dtype=dt_t,
        device=dev,
        z_top_fn=lambda x: 70.0 * (1.0 - torch.exp(-((x - XMAX / 2) ** 2)
                                                   / (2 * 170.0 ** 2))))


def _cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).cpu())


def _run(topo, nt, mem, dev="cuda", dt_t=torch.float32):
    mesh = _mesh(topo, dev, dt_t)
    zf = mesh.z_e / ZMAX
    g = torch.exp(-(((mesh.x_e - XMAX / 2) ** 2 + (mesh.z_e - 430.0) ** 2)
                    / (2 * 150.0 ** 2)))
    vp = (VP0 * (1 + 0.25 * zf) + 260.0 * g).contiguous().requires_grad_(True)
    vs = (VS0 * (1 + 0.25 * zf) + 140.0 * g).contiguous().requires_grad_(True)
    rho = (RHO0 * (1 + 0.15 * zf)).contiguous().requires_grad_(True)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=SIDES,
                              pml_thickness_elements=NPML, c_max=VP0 * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml)
    P = SEMPropagator.build(eq, dt=DT, nt=nt, source_xz=[SRC], receiver_xz=REC,
                            source_kinds=["moment_tensor"], source_payloads=MT)
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=dt_t, device=dev)
    o = P.forward(wav, memory=mem)
    sx, sz = o[..., 0].T, o[..., 1].T
    # linear weighted loss: keeps the adjoint source O(1) so the fp32 gradient
    # stays off the noise floor (elastic data here is ~1e-7).
    Wx = torch.linspace(0.4, 1.6, sx.numel(), dtype=dt_t, device=dev).reshape(sx.shape)
    Wz = torch.linspace(0.6, 1.4, sz.numel(), dtype=dt_t, device=dev).reshape(sz.shape)
    ((Wx * sx).sum() + (Wz * sz).sum()).backward()
    return (sz.detach(), (vp.grad.detach(), vs.grad.detach(), rho.grad.detach()))


@pytest.mark.parametrize("topo", [False, True], ids=["flat", "topo"])
def test_elastic_bs_matches_full_tape(topo):
    sf, gf = _run(topo, 400, None)
    sb, gb = _run(topo, 400, "boundary")
    assert _cos(sb, sf) >= 1 - 1e-4, f"fwd cos={_cos(sb, sf):.7f}"
    for name, f, b in zip(("vp", "vs", "rho"), gf, gb):
        assert _cos(b, f) >= 1 - 1e-4, f"grad {name} cos={_cos(b, f):.7f}"


def test_elastic_bs_long_record():
    """Long record: the band is strongly loaded, so this is the case that fails
    (cos ~0.99) if the CPML memory adjoint is dropped."""
    _, gf = _run(True, 1000, None)
    _, gb = _run(True, 1000, "boundary")
    for name, f, b in zip(("vp", "vs", "rho"), gf, gb):
        assert _cos(b, f) >= 1 - 1e-4, f"grad {name} cos={_cos(b, f):.7f}"


def test_elastic_cpml_band_is_detected():
    """Regression: the elastic profile has no ``b_*``; the band must still be
    found (via ``node_in_pml``) or BS silently skips every band restore."""
    mesh = _mesh(True, "cuda", torch.float32)
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=SIDES,
                              pml_thickness_elements=NPML, c_max=VP0 * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(
        mesh, torch.full(shp, VP0, device="cuda"),
        torch.full(shp, VS0, device="cuda"),
        torch.full(shp, RHO0, device="cuda"), cpml=cpml)
    active, band = cpml_node_masks(eq)
    assert active.numel() > 0 and band.numel() > active.numel()
    assert band.numel() < mesh.n_global          # a real band, not the whole mesh
    assert 0 < cpml_band_elements(eq).numel() < mesh.n_elements


# ------------------------------- impl='c' (CUDA) ---------------------------- #
# The CUDA backward consumes the four stretched derivatives and the two aPML
# brackets per step. Outside the band those equal the RAW derivatives (Sx0 = 1,
# memories 0) and aPML vanishes, so boundary saving stores only the band
# (displacement on band nodes, the six element-local histories on band elements)
# and the reverse march recomputes the interior from the reconstructed field.


def _run_c(topo, nt, mem, dev="cuda"):
    import gc
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    mesh = _mesh(topo, dev, torch.float32)
    zf = mesh.z_e / ZMAX
    g = torch.exp(-(((mesh.x_e - XMAX / 2) ** 2 + (mesh.z_e - 430.0) ** 2)
                    / (2 * 150.0 ** 2)))
    vp = (VP0 * (1 + 0.25 * zf) + 260.0 * g).contiguous().requires_grad_(True)
    vs = (VS0 * (1 + 0.25 * zf) + 140.0 * g).contiguous().requires_grad_(True)
    rho = (RHO0 * (1 + 0.15 * zf)).contiguous().requires_grad_(True)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=SIDES,
                              pml_thickness_elements=NPML, c_max=VP0 * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml)
    P = SEMCudaPropagator.build(eq, dt=DT, nt=nt, source_xz=[SRC],
                                receiver_xz=REC, source_kinds=["moment_tensor"],
                                source_payloads=MT)
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)
    o = P.forward(wav, memory=mem)
    sx, sz = o[..., 0].T, o[..., 1].T
    Wx = torch.linspace(0.4, 1.6, sx.numel(), device=dev).reshape(sx.shape)
    Wz = torch.linspace(0.6, 1.4, sz.numel(), device=dev).reshape(sz.shape)
    ((Wx * sx).sum() + (Wz * sz).sum()).backward()
    torch.cuda.synchronize()
    return (sz.detach(),
            (vp.grad.detach(), vs.grad.detach(), rho.grad.detach()),
            torch.cuda.max_memory_allocated() / 1e6)


@needs_c
@pytest.mark.parametrize("topo", [False, True], ids=["flat", "topo"])
def test_elastic_cuda_bs_matches_full_and_eager(topo):
    """The three-way triangle: CUDA-BS vs CUDA save-all vs the eager full tape."""
    _, ge = _run(topo, 400, None)                 # eager, full autograd tape
    sf, gf, _ = _run_c(topo, 400, None)           # CUDA save-all
    sb, gb, _ = _run_c(topo, 400, "boundary")     # CUDA boundary saving
    assert _cos(sb, sf) >= 1 - 1e-4, f"fwd cos={_cos(sb, sf):.7f}"
    for name, e, f, b in zip(("vp", "vs", "rho"), ge, gf, gb):
        assert _cos(f, e) >= 1 - 1e-4, f"{name} cuda-full vs eager={_cos(f, e):.7f}"
        assert _cos(b, f) >= 1 - 1e-4, f"{name} cuda-BS vs full={_cos(b, f):.7f}"
        assert _cos(b, e) >= 1 - 1e-4, f"{name} cuda-BS vs eager={_cos(b, e):.7f}"


@needs_c
def test_elastic_cuda_bs_long_record():
    _, gf, _ = _run_c(True, 900, None)
    _, gb, _ = _run_c(True, 900, "boundary")
    for name, f, b in zip(("vp", "vs", "rho"), gf, gb):
        assert _cos(b, f) >= 1 - 1e-4, f"{name} cuda-BS vs full={_cos(b, f):.7f}"


@needs_c
def test_elastic_cuda_bs_actually_saves_memory():
    """Guard against a silent no-op: memory='boundary' must genuinely cut the
    peak. Elastic BS is bounded by the element-local histories (it shrinks them
    by the band fraction, not to a thin ring), so the win is modest -- but it
    must be real."""
    _, _, pk_full = _run_c(True, 900, None)
    _, _, pk_bs = _run_c(True, 900, "boundary")
    assert pk_bs < 0.85 * pk_full, f"no memory saving: full={pk_full:.0f} BS={pk_bs:.0f} MB"


# --------------------------------------------------------------------------- #
# Retraction guard.
#
# This codebase carried the claim that elastic BS degrades as the marched
# interior grows -- "cos 1.0000 at band 100%, 0.99 at 81%, 0.876 at 52%, fixed
# nt=400". Re-measured with the SAME medium in both runs, it is cos 1.0000000
# at every one of those band fractions and down to 36.7%. The original numbers
# are the signature of rebuilding random models between the reference run and
# the test run without a fixed seed, which compares the gradients of two
# different media.
#
# 52% is the fraction the claim put at 0.876, so that is the one pinned here.
# The seeding is the load-bearing part of the test: drop `manual_seed` and it
# reproduces the retracted curve.
# --------------------------------------------------------------------------- #
# the model; with its 3-element band outside it on every side the mesh is
# 65 x 129 nodes, the one the measurements were made on
BF_NZ, BF_NX, BF_ABCN, BF_NT, BF_DT = 41, 105, 3, 400, 4e-4
BF_P = BF_ABCN * 4                     # the band, in cells


def _band_fraction_run(memory):
    import sweep.sem as S
    p = S.PropSEM(ElasticSEM2DDeformed, shape=(BF_NZ, BF_NX), dh=25.0, dt=BF_DT,
                  nt=BF_NT, n_order=4, free_surface=False,
                  abcn=BF_P, f_dominant=8.0, dev="cuda", impl="c",
                  memory=memory)
    m = p.mesh
    torch.manual_seed(0)                       # SAME medium in both runs
    ms = [(lo + w * torch.rand(m.n_elements, m.ngll, m.ngll)).cuda(
        ).requires_grad_(True)
        for lo, w in ((2500.0, 200.0), (1400.0, 100.0), (2000.0, 100.0))]
    # the measured geometry on that mesh, in model indices: the source in the
    # top band (row -4), the receivers 4 rows into the model, across the mesh
    nxm = BF_NX + 2 * BF_P
    src = np.array([[nxm // 8 - BF_P, 8 - BF_P]], dtype=np.int64)
    rix = np.arange(4, nxm - 4, 4) - BF_P
    rec = np.stack([rix, np.full_like(rix, 16 - BF_P)], -1)[None]
    t = np.arange(BF_NT) * BF_DT
    a = (np.pi * 8.0 * (t - 1.3 / 8.0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    s = p(w, src, rec, models=ms)
    (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
    eq, _ = p._resolve_model_batch([t_.detach() for t_ in ms])
    frac = cpml_node_masks(eq)[1].numel() / eq.mesh.n_global
    return [t_.grad.detach().clone() for t_ in ms], float(frac)


@needs_c
def test_elastic_bs_is_exact_with_a_large_marched_interior():
    ref, frac = _band_fraction_run(Full())
    bs, _ = _band_fraction_run(BoundarySaving())
    assert 0.45 < frac < 0.60, (
        f"band is {100 * frac:.1f}% of the nodes -- this test only means "
        "something while roughly half the domain is actually marched")
    for j, name in enumerate(("vp", "vs", "rho")):
        c = _cos(bs[j], ref[j])
        assert c >= 1 - 1e-7, f"{name}: BS vs full tape cos={c:.8f}"
