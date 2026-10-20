"""ElasticVSSEM3DDeformed: 3-D elastic + topography + absorbing boundary.

Before this equation the choice was ElasticSEM3DDeformed (topography, no C-PML
at all) or ElasticVSSEM3D (C-PML, axis-aligned only). This one is a subclass of
the latter overriding exactly the two geometry operators, so the C-PML, the
time loop and the BS hooks are the VERIFIED ones, not copies.

The absorption gate here is big-domain SUBTRACTION with receivers strictly
outside the PML band -- not decay-to-floor. Measured on the existing suite,
decay-to-floor passes with two orders of magnitude to spare at a reflection
coefficient of 0.4-0.5 and SATURATES below R~0.2 (a worse PML scores better),
so it certifies stability, not absorption. The companion test proves this
gate discriminates by feeding it a deliberately bad profile.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM, SEMPropagator, ElasticVSSEM3D
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.equations.elastic_vs3d_deformed import ElasticVSSEM3DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import build_axis_aligned_hex27, build_topography_hex27

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NO, DH = 4, 25.0
VP, VS, RHO, F0 = 2400.0, 1250.0, 2000.0, 8.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)


def _wav(nt, device, amp=1e10):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor((((1 - 2 * a) * np.exp(-a)) * amp).astype(np.float32),
                        device=device).to(torch.float64 if device == "cpu"
                                          else torch.float32)


def _full(mesh, v, dtype):
    return torch.full((mesh.n_elements,) + (mesh.ngll,) * 3, v,
                      dtype=dtype, device=mesh.ibool.device)


# ------------------------- flat-mesh reduction ----------------------------- #
def test_reduces_to_the_flat_equation_on_a_flat_mesh():
    """On an axis-aligned Hex27 the full chain rule must reproduce the Hex8
    constant-scaling equation to float64 round-off. This pins the inv_J index
    convention; agreement is NOT bit-for-bit (the zero cross terms still enter
    the sums and re-associate them)."""
    D = torch.float64
    ne, L = 3, 300.0
    h8 = build_hex8_mesh(nex=ne, ney=ne, nez=ne, x_max=L, y_max=L, z_max=L,
                         n_order=NO, dtype=D)
    h27 = build_axis_aligned_hex27(nex=ne, ney=ne, nez=ne, x_max=L, y_max=L,
                                   z_max=L, n_order=NO, dtype=D)
    assert torch.equal(h8.ibool, h27.ibool)
    nt = 60
    w = _wav(nt, "cpu")

    def run(cls, m):
        eq = cls.build(m, _full(m, VP, D), _full(m, VS, D), _full(m, RHO, D))
        P = SEMPropagator.build(
            eq, dt=DT, nt=nt, source_xz=[(150.0, 150.0, 80.0)],
            receiver_xz=[(x, 150.0, 60.0) for x in (90.0, 150.0, 210.0)],
            source_kinds=["force"], source_payloads=[(0.0, 0.0, 1.0)],
            pml_guard="off")
        return P.forward(w)

    a = run(ElasticVSSEM3D, h8)
    b = run(ElasticVSSEM3DDeformed, h27)
    rel = float((a - b).abs().max() / a.abs().max())
    assert rel < 1e-13, f"flat-mesh reduction rel={rel:.3e}"


def test_divergence_is_the_transpose_of_the_gradient_on_a_curved_mesh():
    """<div(sigma), v> == <sigma, grad(v)>. This CANNOT be checked on a flat
    mesh -- the cross terms are identically zero there, so a transposed
    operator still reduces correctly and the test above cannot see it. The
    curved mesh is verified to actually exercise the cross terms."""
    from sweep.sem._assembly import phys_grads3, stress_divergence3
    D = torch.float64
    pi = np.pi
    topo = lambda x, y: 40.0 * torch.sin(2 * pi * x / 300.0) * torch.cos(
        2 * pi * y / 300.0)
    m = build_topography_hex27(nex=3, ney=3, nez=3, x_max=300.0, y_max=300.0,
                               z_bottom=300.0, z_top_fn=topo, n_order=NO,
                               dtype=D)
    off = max(float(m.inv_J[..., mm, d].abs().max())
              for mm in range(3) for d in range(3) if mm != d)
    diag = float(m.inv_J[..., 0, 0].abs().max())
    assert off > 0.1 * diag, (
        f"mesh does not exercise the cross terms (off {off:.2e} vs diag "
        f"{diag:.2e}); the adjoint check would be vacuous")
    torch.manual_seed(1)
    E, ng = m.n_elements, m.ngll
    v = [torch.randn(m.n_global, dtype=D) for _ in range(3)]
    sig = tuple(torch.randn(E, ng, ng, ng, dtype=D) for _ in range(9))
    like = torch.zeros(m.n_global, dtype=D)
    R = stress_divergence3(m.deriv, m.ibool, m.inv_J, m.w_jac, sig, like)
    lhs = sum(float(Ri @ vi) for Ri, vi in zip(R, v))
    g = phys_grads3(m.deriv, m.ibool, m.inv_J, *v)
    pair = [(i, j) for i in range(3) for j in range(3)]
    rhs = sum(float((m.w_jac * sig[k] * g[3 * i + j]).sum())
              for k, (i, j) in enumerate(pair))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel < 1e-12, f"divergence is not the gradient's transpose: {rel:.3e}"


# ----------------------------- absorption gate ----------------------------- #
_PML_EL = 3
_SM = 10                    # 1000 m box
_BIG = 24                   # 2400 m laterally, same depth
_NT = 612                   # 1.10 s: the full lateral S echo fits, and the
#                             reference's own P echo (1.13 s) still does not
# Timing that makes the gate bite, all with the Ricker delay 1.5/f0 included:
#   lateral echo path (500 + 350) m: P at 0.54 s, S (the strong one from a
#     vertical force) at 0.87 s -- a 0.70 s record saw neither and the bare
#     box "leaked" 1%, which the discrimination assert below caught.
#   reference lateral P echo lands at 1.13 s > the 1.00 s record: clean.
#   the BOTTOM echo (0.90 s) is common mode: the C-PML goes on the four
#     lateral sides only, so all three runs share the same reflecting bottom
#     and it cancels in the subtraction. The gate measures LATERAL absorption.


def _topo(L):
    def f(x, y):
        c = L / 2.0
        dx, dy = x - c, y - c
        r = torch.sqrt(dx * dx + dy * dy)
        taper = 0.5 * (1.0 + torch.cos(np.pi * (r / 450.0).clamp(max=1.0)))
        return 40.0 * (torch.sin(2 * np.pi * dx / 500.0)
                       * torch.cos(2 * np.pi * dy / 500.0)) * taper
    return f


def _absorption_run(ne_lat, cpml_profile_or_none, ne_dep=_SM):
    L = ne_lat * NO * DH
    c = L / 2.0
    Lz = ne_dep * NO * DH
    m = build_topography_hex27(nex=ne_lat, ney=ne_lat, nez=ne_dep, x_max=L,
                               y_max=L, z_bottom=Lz, z_top_fn=_topo(L),
                               n_order=NO, dtype=torch.float32, device="cuda")
    F = torch.float32
    cp = None
    if cpml_profile_or_none is not None:
        cp = build_simple_elastic_cpml_3d(
            m, dt=DT, sides=("left", "right", "front", "back"),
            pml_thickness_elements=_PML_EL, c_max=VP * 1.2, f_dominant=F0,
            reflection_target=cpml_profile_or_none)
    eq = ElasticVSSEM3DDeformed.build(m, _full(m, VP, F), _full(m, VS, F),
                                      _full(m, RHO, F), cpml=cp)
    assert (cpml_profile_or_none is None) == (eq.cpml is None)
    # receivers strictly INSIDE the interior: the PML band is _PML_EL elements
    # = 300 m, so |offset| <= 150 keeps them 50 m clear of it in the small box.
    rec = [(c + o, c, 150.0) for o in (-150.0, -75.0, 0.0, 75.0, 150.0)]
    P = SEMPropagator.build(eq, dt=DT, nt=_NT, source_xz=[(c, c, 100.0)],
                            receiver_xz=rec, source_kinds=["force"],
                            source_payloads=[(0.0, 0.0, 1.0)],
                            pml_guard="off")
    with torch.no_grad():
        s = P.forward(_wav(_NT, "cuda"))
    out = s[..., 2].double().cpu().numpy()
    del P, eq, m
    torch.cuda.empty_cache()
    return out


def _leak(small, ref):
    """Peak residual after the first small-box echo, per trace, vs ref peak."""
    i0 = int(((500.0 + (500.0 - 150.0)) / VP + 1.5 / F0) / DT)   # P echo + Ricker delay
    res = np.abs(small[i0:] - ref[i0:]).max(axis=0)
    return float((res / np.abs(ref).max(axis=0)).max())


@cuda
def test_cpml_absorbs_against_a_reflection_free_reference():
    """max(residual peak / reference peak) < 5% outside the PML band. The
    reference shares the small box's DEPTH (a different depth redistributes
    the topography stretch and changes the source element -- measured 3x on
    the first frame) and its surface (the relief is confined to the shared
    footprint)."""
    ref = _absorption_run(_BIG, None)
    box = _absorption_run(_SM, None)
    pml = _absorption_run(_SM, 1e-3)
    leak_box = _leak(box, ref)
    leak_pml = _leak(pml, ref)
    assert leak_pml < 0.05, f"C-PML leak {leak_pml:.3f} >= 5%"
    # and the gate must have something to remove: the bare box must sit well
    # above the gate, or passing it proves nothing. (A first configuration had
    # box leak 1% -- the record ended before any echo arrived.)
    assert leak_box > 3 * 0.05, (
        f"the bare box only leaks {leak_box:.3f}; this configuration cannot "
        f"discriminate and the 5% assertion above proves nothing")


@cuda
def test_the_absorption_gate_discriminates():
    """Feed the SAME gate a deliberately bad profile (reflection_target=0.7,
    i.e. sigma_0 scaled by ln(0.7)/ln(1e-3) ~ 5% of the proper damping). If
    this passes the 5% gate, the gate is measuring nothing."""
    ref = _absorption_run(_BIG, None)
    bad = _absorption_run(_SM, 0.7)
    assert _leak(bad, ref) > 0.05, (
        "a nearly-undamped profile passed the absorption gate")


# --------------------------- PropSEM integration --------------------------- #
@cuda
def test_propsem_drives_it_with_topography_and_cpml():
    nz = ny = nx = 8 * NO + 1
    # a +-30 m relief, 50 m down (and the source and receivers 2 rows = 50 m
    # with it): a topography is the surface row of every column, inside the
    # model, so the relief can no longer straddle z = 0
    topo = lambda x, y: 50.0 + 30.0 * torch.sin(2 * np.pi * x / 400.0)
    Y, X = torch.meshgrid(torch.arange(ny, dtype=torch.float64) * DH,
                          torch.arange(nx, dtype=torch.float64) * DH,
                          indexing="ij")
    p = PropSEM(ElasticVSSEM3DDeformed, shape=(nz, ny, nx), dh=DH, dt=DT,
                nt=80, n_order=NO, free_surface=("top",),
                abcn=2 * NO, f_dominant=F0, dev="cuda", impl="eager",
                topography=(topo(X, Y) / DH).numpy())
    models = [torch.full((nz, ny, nx), v, device="cuda")
              for v in (VP, VS, RHO)]
    src = np.array([[nx // 2, ny // 2, 3 + 2]])
    rix = np.arange(4, nx - 4, 2)
    rec = np.stack([rix, np.full_like(rix, ny // 2),
                    np.full_like(rix, 2 + 2)], -1)[None]
    with torch.no_grad():
        s = p(_wav(80, "cuda"), src, rec, models=models)
    assert s.shape[-1] == 3 and torch.isfinite(s).all()
    assert float(s.abs().max()) > 0


def test_impl_c_is_refused_clearly():
    """The CUDA kernel assumes flat hex8 metrics; silently running it on a
    curved mesh would use the wrong geometry. The gate must refuse."""
    nz = ny = nx = 4 * NO + 1
    with pytest.raises((NotImplementedError, RuntimeError)):
        PropSEM(ElasticVSSEM3DDeformed, shape=(nz, ny, nx), dh=DH, dt=DT,
                nt=8, n_order=NO, free_surface=("top",), abcn=2 * NO,
                f_dominant=F0, dev="cuda", impl="c")
