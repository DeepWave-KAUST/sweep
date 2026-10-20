"""CUDA-graph replay of the SEM impl='c' time loop (SEMGraphRunner).

The loop is launch-bound (measured 32 % of forward wall time and 71 % of
forward+backward wall time spent outside any kernel), so replaying the recorded
launch sequence is the direct fix. These tests pin the three properties that
make it usable:

* capture succeeds at all -- it only does because the drivers are pure stream
  work and the constants (which contain a ``torch.nonzero`` host sync) are cached
* a replay agrees with a fresh eager call to within the atomic-ordering floor,
  which is measured here rather than assumed (the assembly uses ``atomicAdd``, so
  two ordinary eager calls already differ)
* mutating the model in place and replaying gives the answer for the NEW model --
  without this, graphs would be useless for FWI
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.graph_runner import SEMGraphRunner
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, XMAX, ZMAX = 18, 13, 900.0, 650.0
DT, NT, F0, TSH = 7e-4, 200, 13.0, 0.10
VP0, VS0, RHO0 = 2400.0, 1250.0, 2000.0


def _build():
    dev = "cuda"
    mesh = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=4, dtype=torch.float32, device=dev)
    zf = mesh.z_e / ZMAX
    vp = (VP0 * (1 + 0.25 * zf)).contiguous()
    vs = (VS0 * (1 + 0.25 * zf)).contiguous()
    rho = (RHO0 * (1 + 0.15 * zf)).contiguous()
    cpml = build_elastic_cpml(mesh, dt=DT, sides=("left", "right", "bottom"),
                              pml_thickness_elements=5, c_max=VP0 * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml)
    P = SEMCudaPropagator.build(
        eq, dt=DT, nt=NT, source_xz=[(XMAX / 2, 80.0)],
        receiver_xz=[(float(x), 120.0) for x in np.linspace(280.0, XMAX - 280.0, 16)],
        source_kinds=["moment_tensor"], source_payloads=[(1e6, 1e6, 0.0)])
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)
    return P, eq, wav


def _eager(P, wav):
    with torch.no_grad():
        o = P.forward(wav)
    return o[..., 0].T.clone(), o[..., 1].T.clone()


def test_graph_replay_matches_eager():
    P, eq, wav = _build()
    ref_x, ref_z = _eager(P, wav)
    # the assembly uses atomicAdd, so two eager calls already differ: measure
    # that floor instead of demanding bit-exactness.
    a_x, a_z = _eager(P, wav)
    floor = max((a_x - ref_x).abs().max().item(), (a_z - ref_z).abs().max().item())
    amp = ref_x.abs().max().item()

    runner = SEMGraphRunner(P, wav)
    out = runner()
    torch.cuda.synchronize()
    d = max((out[..., 0].T - ref_x).abs().max().item(),
            (out[..., 1].T - ref_z).abs().max().item())
    assert d <= max(10 * floor, 1e-4 * amp), (
        f"replay drift {d:.3e} exceeds the atomic floor {floor:.3e} (amp {amp:.3e})")


def test_graph_replay_is_repeatable():
    P, eq, wav = _build()
    runner = SEMGraphRunner(P, wav)
    first = runner()[..., 0].T.clone()
    second = runner()[..., 0].T.clone()
    amp = first.abs().max().item()
    d = (second - first).abs().max().item()
    assert d <= 1e-4 * amp, f"two replays disagree by {d:.3e} (amp {amp:.3e})"


def test_graph_tracks_model_mutation():
    """The property FWI needs: write new model values into the captured buffers,
    replay, and get the answer for the NEW model."""
    P, eq, wav = _build()
    ref_x, _ = _eager(P, wav)
    runner = SEMGraphRunner(P, wav)

    # what the perturbed model SHOULD give, computed independently
    P2, eq2, wav2 = _build()
    with torch.no_grad():
        eq2._lam.mul_(1.10)
    want_x, _ = _eager(P2, wav2)
    moved = (want_x - ref_x).abs().max().item()
    amp = ref_x.abs().max().item()
    assert moved > 1e-3 * amp, "perturbation too small to be a meaningful test"

    with torch.no_grad():
        eq._lam.mul_(1.10)                 # same change, in the captured buffers
    got = runner()[..., 0].T
    torch.cuda.synchronize()
    d = (got - want_x).abs().max().item()
    assert d <= 1e-4 * amp, (
        f"graph did not track the model update: differs from the recomputed "
        f"answer by {d:.3e} (the update itself moves the answer by {moved:.3e})")


def test_graph_tracks_wavelet_mutation():
    P, eq, wav = _build()
    runner = SEMGraphRunner(P, wav)
    base = runner()[..., 0].T.clone()
    with torch.no_grad():
        runner.wavelet.mul_(2.0)           # the wavelet is a captured buffer too
    got = runner()[..., 0].T
    torch.cuda.synchronize()
    # the equation is linear in the source, so doubling it doubles the record
    d = (got - 2.0 * base).abs().max().item()
    assert d <= 1e-4 * base.abs().max().item(), f"wavelet update not tracked ({d:.3e})"


# --------------------------- graphed GRADIENT path -------------------------- #
# SEMGraphGradRunner graphs forward AND backward. What matters beyond the PoC is
# that autograd still reaches vp/vs/rho: only the C loop is graphed, the cheap
# build() chain (vp -> lam/mu) stays eager and must still thread through.


def _mesh_cpml(dev="cuda"):
    mesh = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=4, dtype=torch.float32, device=dev)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=("left", "right", "bottom"),
                              pml_thickness_elements=5, c_max=VP0 * 1.2,
                              f_dominant=F0)
    return mesh, cpml


def _fresh_eq(mesh, cpml, scale=1.0):
    """One FWI iteration's equation: new leaves, a new vp -> lam autograd chain."""
    zf = mesh.z_e / ZMAX
    vp = (VP0 * scale * (1 + 0.25 * zf)).contiguous().requires_grad_(True)
    vs = (VS0 * (1 + 0.25 * zf)).contiguous().requires_grad_(True)
    rho = (RHO0 * (1 + 0.15 * zf)).contiguous().requires_grad_(True)
    return ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml), (vp, vs, rho)


def _build_grad(dev="cuda"):
    mesh = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=4, dtype=torch.float32, device=dev)
    zf = mesh.z_e / ZMAX
    vp = (VP0 * (1 + 0.25 * zf)).contiguous().requires_grad_(True)
    vs = (VS0 * (1 + 0.25 * zf)).contiguous().requires_grad_(True)
    rho = (RHO0 * (1 + 0.15 * zf)).contiguous().requires_grad_(True)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=("left", "right", "bottom"),
                              pml_thickness_elements=5, c_max=VP0 * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml)
    P = SEMCudaPropagator.build(
        eq, dt=DT, nt=NT, source_xz=[(XMAX / 2, 80.0)],
        receiver_xz=[(float(x), 120.0) for x in np.linspace(280.0, XMAX - 280.0, 16)],
        source_kinds=["moment_tensor"], source_payloads=[(1e6, 1e6, 0.0)])
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)
    return P, eq, wav, (vp, vs, rho)


def _grad_wrt_models(out, models):
    """Linear weighted loss: a plain .sum() leaves the elastic material gradients
    at the fp32 denormal floor, where any comparison is noise."""
    sx, sz = out[..., 0].T, out[..., 1].T
    Wx = torch.linspace(0.4, 1.6, sx.numel(), device=sx.device).reshape(sx.shape)
    Wz = torch.linspace(0.6, 1.4, sz.numel(), device=sz.device).reshape(sz.shape)
    for m in models:
        m.grad = None
    ((Wx * sx).sum() + (Wz * sz).sum()).backward()
    return [m.grad.detach().clone() for m in models]


def _eager_grad(P_ref, eq, wav):
    """Ordinary (ungraphed) forward for a freshly built equation."""
    P = SEMCudaPropagator.build(
        eq, dt=DT, nt=NT, source_xz=[(XMAX / 2, 80.0)],
        receiver_xz=[(float(x), 120.0) for x in np.linspace(280.0, XMAX - 280.0, 16)],
        source_kinds=["moment_tensor"], source_payloads=[(1e6, 1e6, 0.0)])
    return P.forward(wav)


def _cos_nodeps(a, b):
    """No additive epsilon in the denominator: with gradients down at 1e-15 a
    `+1e-30` guard silently halves the cosine and fakes a mismatch."""
    aa = a.reshape(-1).double()
    bb = b.reshape(-1).double()
    na, nb = float(aa.norm()), float(bb.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(aa @ bb) / (na * nb)


def test_graphed_gradient_reaches_vp_vs_rho():
    from sweep.sem.graph_runner import SEMGraphGradRunner

    P, eq, wav, models = _build_grad()
    mesh, cpml = _mesh_cpml()

    # each gradient evaluation rebuilds the equation, exactly as an FWI iteration
    # does (build() computes lam/mu eagerly, so its graph is freed per backward)
    e1, m1 = _fresh_eq(mesh, cpml)
    ref = _grad_wrt_models(_eager_grad(P, e1, wav), m1)
    e2, m2 = _fresh_eq(mesh, cpml)
    ref2 = _grad_wrt_models(_eager_grad(P, e2, wav), m2)      # atomicAdd floor

    runner = SEMGraphGradRunner(P)
    e3, m3 = _fresh_eq(mesh, cpml)
    got = _grad_wrt_models(runner(wav, eq=e3), m3)

    for name, r, r2, g in zip(("vp", "vs", "rho"), ref, ref2, got):
        assert g is not None, f"no gradient reached {name} through the graphed node"
        assert torch.isfinite(g).all(), f"{name} gradient is not finite"
        floor = _cos_nodeps(r, r2)
        cg = _cos_nodeps(r, g)
        assert cg >= floor - 1e-6, (
            f"{name}: graphed cos={cg:.8f} below the eager-vs-eager floor {floor:.8f}")


def test_graphed_gradient_tracks_a_model_change():
    """Two different models must give two different gradients through the graph."""
    from sweep.sem.graph_runner import SEMGraphGradRunner

    P, eq, wav, _ = _build_grad()
    mesh, cpml = _mesh_cpml()
    runner = SEMGraphGradRunner(P)

    e0, m0 = _fresh_eq(mesh, cpml, scale=1.0)
    g0 = _grad_wrt_models(runner(wav, eq=e0), m0)
    e1, m1 = _fresh_eq(mesh, cpml, scale=1.15)     # a genuinely different model
    g1 = _grad_wrt_models(runner(wav, eq=e1), m1)

    moved = max(float((b - a).norm() / (a.norm() + 1e-300))
                for a, b in zip(g0, g1))
    assert moved > 1e-3, (
        f"a 20% lambda change moved the graphed gradient by only {moved:.2e} "
        "relative -- the graph is not seeing the update")


# ------------------- graphing MUST compose with boundary saving ------------- #
# Boundary saving is the production memory lever, so a dispatch optimisation
# that silently disabled it would be useless in production. An earlier version
# of SEMGraphGradRunner hard-coded bs=False on the assumption that the band
# buffers could not be static; measuring showed that assumption was wrong.


def test_graphed_gradient_supports_boundary_saving():
    from sweep.sem.graph_runner import SEMGraphGradRunner

    P, eq, wav, _ = _build_grad()
    mesh, cpml = _mesh_cpml()

    # reference: ordinary impl='c' with the save-all tape
    e_ref, m_ref = _fresh_eq(mesh, cpml)
    ref = _grad_wrt_models(_eager_grad(P, e_ref, wav), m_ref)
    e_ref2, m_ref2 = _fresh_eq(mesh, cpml)
    ref2 = _grad_wrt_models(_eager_grad(P, e_ref2, wav), m_ref2)   # atomic floor

    runner = SEMGraphGradRunner(P, memory="boundary")
    e_bs, m_bs = _fresh_eq(mesh, cpml)
    got = _grad_wrt_models(runner(wav, eq=e_bs), m_bs)

    for name, r, r2, g in zip(("vp", "vs", "rho"), ref, ref2, got):
        floor = _cos_nodeps(r, r2)
        cg = _cos_nodeps(r, g)
        assert cg >= floor - 1e-6, (
            f"{name}: graphed+boundary-saved cos={cg:.8f} below the eager-vs-eager "
            f"floor {floor:.8f}")


def test_graphing_does_not_undo_the_boundary_saving():
    """Graphing must not eat the memory that boundary saving buys: measured
    0.98-0.99x of the ungraphed boundary-saved peak at nt = 300..1800."""
    import gc

    from sweep.sem.graph_runner import SEMGraphGradRunner

    P, eq, wav, _ = _build_grad()
    mesh, cpml = _mesh_cpml()

    def peak(memory, graphed):
        gc.collect(); torch.cuda.empty_cache()
        runner = SEMGraphGradRunner(P, memory=memory) if graphed else None
        e, m = _fresh_eq(mesh, cpml)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        out = runner(wav, eq=e) if graphed else (P.__setattr__("eq", e) or P.forward(wav))
        _grad_wrt_models(out, m)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6

    bs_plain = peak("boundary", False)
    bs_graph = peak("boundary", True)
    assert bs_graph <= 1.5 * bs_plain, (
        f"graphing inflated the boundary-saved peak from {bs_plain:.0f} to "
        f"{bs_graph:.0f} MB -- that would defeat the point of boundary saving")
