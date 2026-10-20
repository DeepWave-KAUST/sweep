"""3-D SEM (Hex8) — acoustic + isotropic elastic.

The rigorous, FD-free core:
  * operator algebra — K symmetric, PSD, correct null space (constant for
    acoustic; the three rigid-body translations for elastic);
  * **reduction to 2-D** — a y-invariant 3-D field reproduces the already
    validated 2-D plane-strain operator exactly, because the extra y-quadrature
    factor cancels in the acceleration ``-K u / M`` node-by-node.

Plus CUDA-gated propagation checks that the measured wavefront speed matches the
physical wave speed (the strongest end-to-end test of the geometric direction
factors + the source/receiver/CPML plumbing).
"""
import numpy as np
import pytest
import torch

from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import build_axis_aligned_hex27, build_topography_hex27
from sweep.sem.mesh.quad4 import build_quad4_mesh
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.equations.elastic3d import ElasticSEM3D
from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.equations.elastic2d import ElasticSEM2D
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.boundary_saving import cpml_node_masks
from sweep.sem.propagator import SEMPropagator
from sweep.sem import PropSEM
from sweep.propagator import BoundarySaving

DT = torch.float64
N = 4


# --------------------------------------------------------------------------- #
# Mesh structure
# --------------------------------------------------------------------------- #
def test_hex8_mesh_structure():
    m = build_hex8_mesh(3, 4, 5, 30.0, 40.0, 50.0, n_order=N, dtype=DT)
    ng = m.ngll
    assert m.ibool.shape == (3 * 4 * 5, ng, ng, ng)
    nx_g, ny_g, nz_g = 3 * N + 1, 4 * N + 1, 5 * N + 1
    assert m.n_global == nx_g * ny_g * nz_g
    assert int(m.ibool.min()) == 0 and int(m.ibool.max()) == m.n_global - 1
    # every global node hit at least once
    assert torch.unique(m.ibool).numel() == m.n_global
    # element-0 corner coords sit at the origin; opposite corner at (dx,dy,dz)
    assert torch.allclose(m.x_e[0, 0, 0, 0], torch.tensor(0.0, dtype=DT))
    assert torch.allclose(m.x_e[0, 0, 0, -1],
                          torch.tensor(m.dx_e, dtype=DT), atol=1e-9)
    assert torch.allclose(m.z_e[0, -1, 0, 0],
                          torch.tensor(m.dz_e, dtype=DT), atol=1e-9)


# --------------------------------------------------------------------------- #
# Operator algebra
# --------------------------------------------------------------------------- #
def test_acoustic3d_operator_sym_psd_null():
    torch.manual_seed(0)
    m = build_hex8_mesh(3, 3, 3, 300.0, 300.0, 300.0, n_order=N, dtype=DT)
    sh = (m.n_elements, m.ngll, m.ngll, m.ngll)
    vp = (2000.0 + 400.0 * torch.rand(sh, dtype=DT)).contiguous()
    eq = AcousticSEM3D.build(m, vp, dirichlet=())          # pure Neumann
    a, b = torch.randn(m.n_global, dtype=DT), torch.randn(m.n_global, dtype=DT)
    Ka, Kb = eq.stiffness_apply(a), eq.stiffness_apply(b)
    assert abs((a @ Kb - b @ Ka).item()) / abs((a @ Kb).item()) < 1e-10   # sym
    assert (a @ Ka).item() > 0                                            # PSD
    nullc = eq.stiffness_apply(torch.ones(m.n_global, dtype=DT)).norm()
    assert (nullc / eq.mass.norm()).item() < 1e-10                        # const null


def test_elastic3d_operator_sym_psd_rigidbody():
    torch.manual_seed(1)
    m = build_hex8_mesh(3, 3, 3, 300.0, 300.0, 300.0, n_order=N, dtype=DT)
    sh = (m.n_elements, m.ngll, m.ngll, m.ngll)
    vp = (3000.0 + 300.0 * torch.rand(sh, dtype=DT)).contiguous()
    vs = (1700.0 + 200.0 * torch.rand(sh, dtype=DT)).contiguous()
    rho = (2200.0 + 200.0 * torch.rand(sh, dtype=DT)).contiguous()
    eq = ElasticSEM3D.build(m, vp, vs, rho)
    nG = m.n_global
    rnd = lambda: torch.randn(nG, dtype=DT)
    a = (rnd(), rnd(), rnd()); b = (rnd(), rnd(), rnd())
    Ka, Kb = eq.stiffness_apply(*a), eq.stiffness_apply(*b)
    aKb = sum(a[i] @ Kb[i] for i in range(3)).item()
    bKa = sum(b[i] @ Ka[i] for i in range(3)).item()
    assert abs(aKb - bKa) / abs(aKb) < 1e-10                              # sym
    assert sum(a[i] @ Ka[i] for i in range(3)).item() > 0                 # PSD
    one, zero = torch.ones(nG, dtype=DT), torch.zeros(nG, dtype=DT)
    mref = eq.mass.norm().item()
    for trans in [(one, zero, zero), (zero, one, zero), (zero, zero, one)]:
        K = eq.stiffness_apply(*trans)
        assert sum(k.norm() for k in K).item() / mref < 1e-10            # rigid null


# --------------------------------------------------------------------------- #
# Reduction to 2-D (exact, FD-free) — a y-invariant 3-D field reproduces the
# validated 2-D plane-strain acceleration node-by-node.
# --------------------------------------------------------------------------- #
def _lift_y_invariant(u2, nz_g, ny_g, nx_g):
    """(nz_g·nx_g,) 2-D field -> (nz_g·ny_g·nx_g,) 3-D field, copied across y."""
    u3 = u2.reshape(nz_g, 1, nx_g).expand(nz_g, ny_g, nx_g).reshape(-1)
    return u3.contiguous()


def test_acoustic3d_reduces_to_2d():
    torch.manual_seed(2)
    nex, nez = 3, 3
    m3 = build_hex8_mesh(nex, 1, nez, 300.0, 100.0, 300.0, n_order=N, dtype=DT)
    m2 = build_quad4_mesh(nex, nez, 300.0, 300.0, n_order=N, dtype=DT)
    nx_g, ny_g, nz_g = nex * N + 1, 1 * N + 1, nez * N + 1
    c0 = 2500.0
    vp3 = torch.full((m3.n_elements, N + 1, N + 1, N + 1), c0, dtype=DT)
    vp2 = torch.full((m2.n_elements, N + 1, N + 1), c0, dtype=DT)
    e3 = AcousticSEM3D.build(m3, vp3, dirichlet=())
    e2 = AcousticSEM2D.build(m2, vp2, dirichlet=())
    u2 = torch.randn(m2.n_global, dtype=DT)
    u3 = _lift_y_invariant(u2, nz_g, ny_g, nx_g)
    a2 = (-e2.stiffness_apply(u2) * e2._inv_M).reshape(nz_g, nx_g)
    a3 = (-e3.stiffness_apply(u3) * e3._inv_M).reshape(nz_g, ny_g, nx_g)
    for j in range(ny_g):                                   # every y-layer == 2-D
        assert (a3[:, j, :] - a2).norm() / a2.norm() < 1e-12


def test_elastic3d_reduces_to_2d():
    torch.manual_seed(3)
    nex, nez = 3, 3
    m3 = build_hex8_mesh(nex, 1, nez, 300.0, 100.0, 300.0, n_order=N, dtype=DT)
    m2 = build_quad4_mesh(nex, nez, 300.0, 300.0, n_order=N, dtype=DT)
    nx_g, ny_g, nz_g = nex * N + 1, 1 * N + 1, nez * N + 1
    vp, vs, rho = 3000.0, 1700.0, 2200.0
    full3 = lambda v: torch.full((m3.n_elements, N + 1, N + 1, N + 1), v, dtype=DT)
    full2 = lambda v: torch.full((m2.n_elements, N + 1, N + 1), v, dtype=DT)
    e3 = ElasticSEM3D.build(m3, full3(vp), full3(vs), full3(rho))
    e2 = ElasticSEM2D.build(m2, full2(vp), full2(vs), full2(rho))
    ux2, uz2 = torch.randn(m2.n_global, dtype=DT), torch.randn(m2.n_global, dtype=DT)
    ux3 = _lift_y_invariant(ux2, nz_g, ny_g, nx_g)
    uz3 = _lift_y_invariant(uz2, nz_g, ny_g, nx_g)
    uy3 = torch.zeros(m3.n_global, dtype=DT)
    Kx2, Kz2 = e2.stiffness_apply(ux2, uz2)
    Kx3, Ky3, Kz3 = e3.stiffness_apply(ux3, uy3, uz3)
    ax2 = (-Kx2 * e2._inv_M).reshape(nz_g, nx_g)
    az2 = (-Kz2 * e2._inv_M).reshape(nz_g, nx_g)
    ax3 = (-Kx3 * e3._inv_M).reshape(nz_g, ny_g, nx_g)
    ay3 = (-Ky3 * e3._inv_M).reshape(nz_g, ny_g, nx_g)
    az3 = (-Kz3 * e3._inv_M).reshape(nz_g, ny_g, nx_g)
    for j in range(ny_g):
        assert (ax3[:, j, :] - ax2).norm() / ax2.norm() < 1e-12
        assert (az3[:, j, :] - az2).norm() / az2.norm() < 1e-12
    # uy stays decoupled in the y-interior. The two y-faces legitimately carry
    # the σ_yy = λ(∂x ux + ∂z uz) traction (a boundary term of ∫ σ_yy ∂_y v_y),
    # so only the interior y-layers are exactly zero.
    assert ay3[:, 1:-1, :].norm() / ax2.norm() < 1e-12


# --------------------------------------------------------------------------- #
# Propagation wave speed (CUDA-gated — heavier).
# --------------------------------------------------------------------------- #
def _ricker(t, f, delay):
    a = (np.pi * f * (t - delay)) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float64)


def _speed_from_arrivals(seis, offs, dt):
    tt = [int(np.argmax(np.abs(seis[i]))) * dt for i in range(len(offs))]
    return np.polyfit(tt, offs, 1)[0]


def _speed_onset(seis, offs, dt, frac=0.03):
    """First-break (wavefront onset) speed — robust to near-field, which
    contaminates the peak when receiver offset ≲ 1 wavelength."""
    tt = [int(np.argmax(np.abs(seis[i]) > frac * np.abs(seis[i]).max())) * dt
          for i in range(len(offs))]
    return np.polyfit(tt, offs, 1)[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="propagation is heavy")
def test_acoustic3d_wavespeed():
    dev = "cuda"
    nel, de = 20, 20.0
    L, c0 = nel * de, 3000.0
    m = build_hex8_mesh(nel, nel, nel, L, L, L, n_order=N, dtype=DT, device=dev)
    vp = torch.full((m.n_elements, N + 1, N + 1, N + 1), c0, dtype=DT, device=dev)
    f0, dt, nt, delay = 14.0, 5e-4, 360, 0.08
    cpml = build_cpml3d(m, dt=dt, c_max=c0, f_dominant=f0, pml_thickness_elements=3,
                        sides=("left", "right", "front", "back", "top", "bottom"))
    eq = AcousticSEM3D.build(m, vp, dirichlet=(), cpml=cpml)
    sx, sy, sz, offs = 100.0, 200.0, 200.0, [50.0, 100.0, 150.0, 200.0]
    prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(sx, sy, sz)],
                               receiver_xz=[(sx + d, sy, sz) for d in offs])
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT, device=dev)
    seis = prop.forward(wav)[..., 0].T.cpu().numpy()
    c_meas = _speed_from_arrivals(seis, offs, dt)
    assert abs(c_meas - c0) / c0 < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="propagation is heavy")
def test_elastic3d_pwave_speed():
    dev = "cuda"
    nel, de = 20, 20.0
    L = nel * de
    vp0, vs0, rho0 = 3000.0, 1732.0, 2200.0
    m = build_hex8_mesh(nel, nel, nel, L, L, L, n_order=N, dtype=DT, device=dev)
    full = lambda v: torch.full((m.n_elements, N + 1, N + 1, N + 1), v,
                                dtype=DT, device=dev)
    eq = ElasticSEM3D.build(m, full(vp0), full(vs0), full(rho0))
    f0, dt, nt, delay = 14.0, 5e-4, 360, 0.08
    sx, sy, sz, offs = 200.0, 200.0, 100.0, [50.0, 100.0, 150.0, 200.0]
    prop = SEMPropagator.build(
        eq, dt=dt, nt=nt, source_xz=[(sx, sy, sz)],
        receiver_xz=[(sx, sy, sz + d) for d in offs],
        source_kinds=["force"], source_payloads=[(0.0, 0.0, 1.0)])
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT, device=dev)
    uz = prop.forward(wav)[..., 2].T.cpu().numpy()
    vp_meas = _speed_from_arrivals(uz, offs, dt)            # leading arrival = P
    assert abs(vp_meas - vp0) / vp0 < 0.05


# --------------------------------------------------------------------------- #
# Velocity-stress 3-D elastic (ElasticVSSEM3D) + simple CPML.
# --------------------------------------------------------------------------- #
def test_elastic_vs3d_autograd():
    """Full-tape autograd model gradient FD-checked (vp, vs, rho)."""
    torch.manual_seed(7)
    nel = 4
    Lc = nel * 25.0
    m = build_hex8_mesh(nel, nel, nel, Lc, Lc, Lc, n_order=N, dtype=DT)
    sh = (m.n_elements, N + 1, N + 1, N + 1)
    f0, dt, nt, delay = 16.0, 8e-4, 70, 0.05
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT)
    src = [(Lc / 2, Lc / 2, Lc / 4)]
    rec = [(Lc * 0.4, Lc * 0.5, Lc * 0.6), (Lc * 0.6, Lc * 0.5, Lc * 0.6)]
    params = [(3200 + 200 * torch.rand(sh, dtype=DT)).contiguous(),
              (1800 + 150 * torch.rand(sh, dtype=DT)).contiguous(),
              (2300 + 150 * torch.rand(sh, dtype=DT)).contiguous()]
    for p in params:
        p.requires_grad_(True)

    def misfit():
        eq = ElasticVSSEM3D.build(m, params[0], params[1], params[2])
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src, receiver_xz=rec,
                                   source_kinds=["force"], source_payloads=[(1.0, 0.5, 0.8)])
        out = prop.forward(wav)
        return out.sum(-1).T.pow(2).sum()

    J = misfit()
    grads = torch.autograd.grad(J, params)
    dirs = [torch.randn_like(p) for p in params]
    analytic = sum((g * d).sum() for g, d in zip(grads, dirs)).item()
    h = 1e-3
    with torch.no_grad():
        for p, d in zip(params, dirs):
            p += h * d
    Jp = misfit().item()
    with torch.no_grad():
        for p, d in zip(params, dirs):
            p -= 2 * h * d
    Jm = misfit().item()
    fd = (Jp - Jm) / (2 * h)
    assert abs(analytic - fd) / (abs(fd) + 1e-300) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="propagation is heavy")
def test_elastic_vs3d_cpml_speed_and_absorb():
    """P/S wave speeds + CPML absorption (vs a reflecting box)."""
    dev = "cuda"
    nel, de = 20, 24.0
    L = nel * de
    vp0, vs0, rho0 = 3000.0, 1732.0, 2200.0
    m = build_hex8_mesh(nel, nel, nel, L, L, L, n_order=N, dtype=DT, device=dev)
    full = lambda v: torch.full((m.n_elements, N + 1, N + 1, N + 1), v,
                                dtype=DT, device=dev)
    f0, dt, nt, delay = 14.0, 6e-4, 420, 0.07
    cpml = build_simple_elastic_cpml_3d(
        m, dt=dt, c_max=vp0, f_dominant=f0, pml_thickness_elements=3,
        sides=("left", "right", "front", "back", "top", "bottom"))
    eq = ElasticVSSEM3D.build(m, full(vp0), full(vs0), full(rho0), cpml=cpml)
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT, device=dev)
    offs = [60.0, 120.0, 180.0, 240.0]
    sy = L / 2

    def run(src, rec):
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[src], receiver_xz=rec,
                                   source_kinds=["force"], source_payloads=[(0.0, 0.0, 1.0)])
        return prop.forward(wav)

    vzP = run((L / 2, sy, 100.0), [(L / 2, sy, 100.0 + d) for d in offs]
              )[..., 2].T.cpu().numpy()
    vzS = run((120.0, sy, L / 2), [(120.0 + d, sy, L / 2) for d in offs]
              )[..., 2].T.cpu().numpy()
    vP = _speed_onset(vzP, offs, dt)              # along force axis -> near field
    vS = _speed_from_arrivals(vzS, offs, dt)      # clean S radiation lobe
    assert abs(vP - vp0) / vp0 < 0.06
    assert abs(vS - vs0) / vs0 < 0.08

    # CPML absorbs: late-window energy at the source point << reflecting box.
    nt_long = 700
    wav_l = torch.tensor(_ricker(np.arange(nt_long) * dt, f0, delay), dtype=DT, device=dev)
    bare = ElasticVSSEM3D.build(m, full(vp0), full(vs0), full(rho0), cpml=None)
    sc = (L / 2, sy, L / 2)
    pc = SEMPropagator.build(eq, dt=dt, nt=nt_long, source_xz=[sc], receiver_xz=[sc],
                             source_kinds=["force"], source_payloads=[(0.0, 0.0, 1.0)])
    pb = SEMPropagator.build(bare, dt=dt, nt=nt_long, source_xz=[sc], receiver_xz=[sc],
                             source_kinds=["force"], source_payloads=[(0.0, 0.0, 1.0)])
    tc = pc.forward(wav_l)[..., 2].T.cpu().numpy()[0]
    tb = pb.forward(wav_l)[..., 2].T.cpu().numpy()[0]
    assert np.isfinite(tc).all()
    late = slice(int(0.75 * nt_long), nt_long)
    assert np.abs(tb[late]).max() / (np.abs(tc[late]).max() + 1e-300) > 4


# --------------------------------------------------------------------------- #
# 3-D boundary saving (run_boundary_saved): forward bit-identical to the full
# tape, interior model gradient FWI-grade (cosine > 0.999). The PML-band
# gradient is not reconstructed by design — compare on the interior nodes only.
# --------------------------------------------------------------------------- #
def _cos(x, y):
    return (x @ y / (x.norm() * y.norm() + 1e-300)).item()


def _bs_vs_fulltape(dev, nel, pml, build_eq, make_params, cpml, source_kinds,
                    source_payloads):
    L = nel * 25.0
    m = build_hex8_mesh(nel, nel, nel, L, L, L, n_order=N, dtype=DT, device=dev)
    f0, dt, nt, delay = 12.0, 5e-4, 200, 0.06   # dt < SEM-4 CFL (~5.7e-4 here)
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT, device=dev)
    src = [(L / 2, L / 2, L / 2)]
    rec = [(L * 0.35, L * 0.5, L * 0.45), (L * 0.6, L * 0.45, L * 0.55)]

    def run(memory):
        params = make_params(m)
        for p in params:
            p.requires_grad_(True)
        eq = build_eq(m, params, cpml(m, dt, f0))
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src, receiver_xz=rec,
                                   source_kinds=source_kinds, source_payloads=source_payloads)
        out = prop.forward(wav, memory=memory)
        seis = out.sum(-1).T
        seis.pow(2).sum().backward()
        return seis.detach(), [p.grad.clone() for p in params], eq

    s_full, g_full, _ = run(None)
    s_bs, g_bs, _ = run("boundary")
    assert (s_bs - s_full).norm() / s_full.norm() < 1e-9        # forward exact
    # True interior = elements at depth > pml (exclude the PML + 1-element skin
    # that BS does not reconstruct by design — FWI masks it). There the
    # reconstruction is exact (cosine ~1) for both acoustic and velocity-stress.
    e = torch.arange(m.n_elements, device=dev)
    ex, ey, ez = e % nel, (e // nel) % nel, e // (nel * nel)
    depth = torch.stack([ex, nel - 1 - ex, ey, nel - 1 - ey, ez, nel - 1 - ez]).amin(0)
    deep = (depth >= pml + 1)[:, None, None, None].expand_as(m.ibool)
    for gf, gb in zip(g_full, g_bs):
        assert _cos(gb[deep], gf[deep]) > 0.999                # interior exact


def _acoustic_bs_case(dev, nel):
    _bs_vs_fulltape(
        dev, nel, pml=2,
        build_eq=lambda m, P, cp: AcousticSEM3D.build(m, P[0], cpml=cp),
        make_params=lambda m: [torch.full((m.n_elements, N + 1, N + 1, N + 1),
                                          3000.0, dtype=DT, device=m.ibool.device)],
        cpml=lambda m, dt, f0: build_cpml3d(
            m, dt=dt, c_max=3000.0, f_dominant=f0, pml_thickness_elements=2,
            sides=("left", "right", "front", "back", "bottom")),
        source_kinds=None, source_payloads=None)


def _elastic_vs_bs_case(dev, nel):
    fill = lambda m, v: torch.full((m.n_elements, N + 1, N + 1, N + 1), v,
                                   dtype=DT, device=m.ibool.device)
    _bs_vs_fulltape(
        dev, nel, pml=2,
        build_eq=lambda m, P, cp: ElasticVSSEM3D.build(m, P[0], P[1], P[2], cpml=cp),
        make_params=lambda m: [fill(m, 3000.0), fill(m, 1732.0), fill(m, 2200.0)],
        cpml=lambda m, dt, f0: build_simple_elastic_cpml_3d(
            m, dt=dt, c_max=3000.0, f_dominant=f0, pml_thickness_elements=2,
            sides=("left", "right", "front", "back", "bottom")),
        source_kinds=["force"], source_payloads=[(1.0, 0.5, 0.8)])


# A meaningful interior (vs the PML band) needs a non-trivial mesh, so the
# gradient-quality checks are CUDA-gated (nel=12), matching the 2D BS tests.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="BS gradient needs a real mesh")
def test_acoustic3d_boundary_saving():
    _acoustic_bs_case("cuda", 12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BS gradient needs a real mesh")
def test_elastic_vs3d_boundary_saving():
    _elastic_vs_bs_case("cuda", 12)


# --------------------------------------------------------------------------- #
# PropSEM 3-D: the unified FD-style grid call for AcousticSEM3D.
# --------------------------------------------------------------------------- #
def _propsem3d(shape, dev="cpu", memory=None):
    return PropSEM(AcousticSEM3D, shape=shape, dh=25.0, dt=4e-4, nt=100,
                   n_order=N, free_surface=False,
                   abcn=2 * N, f_dominant=12.0, dev=dev, dtype=DT,
                   impl="eager", memory=memory)


def test_propsem3d_forward_interp_and_equivalence():
    """Forward shape; the (nz,ny,nx)->GLL interp gradient is the exact linear
    transpose; and PropSEM == a hand-built SEMPropagator with vp_e=interp(grid)."""
    nz, ny, nx = 17, 13, 17
    prop = _propsem3d((nz, ny, nx))
    nt = prop.nt
    wav = torch.tensor(_ricker(np.arange(nt) * prop.dt, 12.0, 0.05), dtype=DT)
    vp = torch.full((nz, ny, nx), 3000.0, dtype=DT)
    src = np.array([[6, 6, 5]], dtype=np.int64)
    rec = np.array([[[4, 6, 7], [12, 6, 7]]], dtype=np.int64)
    seis = prop(wav, src, rec, models=[vp])
    assert seis.shape == (1, nt, 2, 1) and torch.isfinite(seis).all()

    # interp gradient = exact transpose of the linear map -- on the model's
    # nodes: the absorbing pad is a constant of the model (test_sem_pad_gradient)
    torch.manual_seed(0)
    prop._build_grid_to_gll()
    c = torch.randn(prop._gll_shape, dtype=DT) * ~prop._g_pad.reshape(
        prop._gll_shape)
    g = vp.clone().requires_grad_(True)
    (gi,) = torch.autograd.grad((c * prop._interp_to_gll(g)).sum(), g)
    d = torch.randn_like(g)
    an = (gi * d).sum().item()
    h = 1e-1
    fd = ((c * prop._interp_to_gll(g + h * d)).sum().item()
          - (c * prop._interp_to_gll(g - h * d)).sum().item()) / (2 * h)
    assert abs(an - fd) / (abs(fd) + 1e-300) < 1e-10

    # PropSEM forward == manual SEMPropagator (bit-identical)
    vp_e = prop._interp_to_gll(vp)
    cpml = build_cpml3d(prop.mesh, dt=prop.dt, c_max=float(vp_e.max()),
                        f_dominant=12.0, pml_thickness_elements=2,
                        sides=("left", "right", "front", "back", "bottom", "top"))
    eq = AcousticSEM3D.build(prop.mesh, vp_e, dirichlet=(), cpml=cpml)
    sp = SEMPropagator.build(
        eq, dt=prop.dt, nt=nt, source_xz=[prop._coords(src[0])],
        receiver_xz=[prop._coords(rec[0, j]) for j in range(2)])
    ref = sp.forward(wav)[..., 0].T                  # (nrec, nt)
    # PropSEM returns the FD layout (B, nt, nrec, ncomp); comparing it to the
    # native (nrec, nt) directly does not raise, it BROADCASTS to (nt, nrec, nt)
    # and quietly reports a ratio of ~10 instead of ~1e-11.
    assert (seis[0, :, :, 0].T - ref).norm() / ref.norm() < 1e-11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="multishot + BS is heavy")
def test_propsem3d_multishot_and_boundary_saving():
    """Multishot vmap batch == per-shot loop (forward + grad), and BS == full
    tape (forward bit-identical, interior grid gradient cosine ~1)."""
    nz, ny, nx = 25, 21, 25
    prop = _propsem3d((nz, ny, nx), dev="cuda")
    nt = prop.nt
    wav = torch.tensor(_ricker(np.arange(nt) * prop.dt, 12.0, 0.06),
                       dtype=DT, device="cuda")
    vp = torch.full((nz, ny, nx), 3000.0, dtype=DT, device="cuda")
    src = np.array([[8, 10, 6], [16, 10, 6]], dtype=np.int64)
    rx = np.arange(4, nx - 4, 4)
    rec = np.repeat(np.stack([rx, np.full_like(rx, 10), np.full_like(rx, 10)], 1)[None],
                    2, 0)

    # multishot vmap batch vs per-shot loop
    gb = vp.clone().requires_grad_(True)
    seis_b = _propsem3d((nz, ny, nx), "cuda")(wav, src, rec, models=[gb])
    seis_b.pow(2).sum().backward()
    grad_b = gb.grad.clone()
    seis_loop, grad_loop = [], torch.zeros_like(vp)
    for b in range(2):
        gl = vp.clone().requires_grad_(True)
        sb = _propsem3d((nz, ny, nx), "cuda")(wav, src[b:b + 1], rec[b:b + 1], models=[gl])
        seis_loop.append(sb.detach())
        sb.pow(2).sum().backward()
        grad_loop += gl.grad
    seis_loop = torch.cat(seis_loop)
    assert (seis_b.detach() - seis_loop).norm() / seis_loop.norm() < 1e-10
    assert (grad_b - grad_loop).norm() / grad_loop.norm() < 1e-9

    # BS == full tape (single shot)
    gf = vp.clone().requires_grad_(True)
    sf = _propsem3d((nz, ny, nx), "cuda")(wav, src[:1], rec[:1], models=[gf])
    sf.pow(2).sum().backward()
    gbs = vp.clone().requires_grad_(True)
    sbs = _propsem3d((nz, ny, nx), "cuda", memory=BoundarySaving())(
        wav, src[:1], rec[:1], models=[gbs])
    sbs.pow(2).sum().backward()
    assert (sbs.detach() - sf.detach()).norm() / sf.detach().norm() < 1e-9
    core = (slice(3, nz - 3), slice(3, ny - 3), slice(3, nx - 3))
    gA, gB = gf.grad[core].flatten(), gbs.grad[core].flatten()
    assert (gA @ gB / (gA.norm() * gB.norm() + 1e-300)).item() > 0.999


# --------------------------------------------------------------------------- #
# Hex27 deformed mesh + AcousticSEM3DDeformed (curved free surface / topography).
# --------------------------------------------------------------------------- #
def test_hex27_reduces_to_hex8():
    """Axis-aligned Hex27 == Hex8 (mesh), and the deformed acoustic operator
    reduces to AcousticSEM3D exactly — bare stiffness + full CPML propagation."""
    nex, ney, nez = 3, 4, 3
    xm, ym, zm = 300.0, 400.0, 300.0
    h8 = build_hex8_mesh(nex, ney, nez, xm, ym, zm, n_order=N, dtype=DT)
    h27 = build_axis_aligned_hex27(nex, ney, nez, xm, ym, zm, n_order=N, dtype=DT)
    assert (h27.ibool == h8.ibool).all()
    assert (h27.x_e - h8.x_e).norm() / h8.x_e.norm() < 1e-12
    assert (h27.z_e - h8.z_e).norm() / h8.z_e.norm() < 1e-12
    assert float((h27.det_J - h8.jacobian_det).abs().max()) / h8.jacobian_det < 1e-12

    vp = (2500.0 + 0.8 * h8.z_e).contiguous()          # continuous (depth ramp)
    # bare stiffness
    e8 = AcousticSEM3D.build(h8, vp, dirichlet=(), cpml=None)
    e27 = AcousticSEM3DDeformed.build(h27, vp, dirichlet=(), cpml=None)
    u = torch.randn(h8.n_global, dtype=DT)
    assert (e27.stiffness_apply(u) - e8.stiffness_apply(u)).norm() \
        / e8.stiffness_apply(u).norm() < 1e-11
    # full CPML propagation (share one profile; meshes share node ordering)
    f0, dt, nt, delay = 12.0, 5e-4, 80, 0.06
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT)
    cp = build_cpml3d(h8, dt=dt, c_max=float(vp.max()), f_dominant=f0,
                      pml_thickness_elements=1,
                      sides=("left", "right", "front", "back", "bottom"))
    e8c = AcousticSEM3D.build(h8, vp, dirichlet=("top",), cpml=cp)
    e27c = AcousticSEM3DDeformed.build(h27, vp, dirichlet=("top",), cpml=cp)
    src = [(xm / 2, ym / 2, zm / 3)]
    rec = [(xm * 0.35, ym * 0.5, zm * 0.4), (xm * 0.65, ym * 0.5, zm * 0.5)]
    o8 = SEMPropagator.build(e8c, dt=dt, nt=nt, source_xz=src, receiver_xz=rec).forward(wav)
    o27 = SEMPropagator.build(e27c, dt=dt, nt=nt, source_xz=src, receiver_xz=rec).forward(wav)
    rel = (o27[..., 0].T - o8[..., 0].T).norm() / o8[..., 0].T.norm()
    assert rel < 1e-10


def test_hex27_topography_forward_and_gradient():
    """Gaussian-hill topography: surface follows z_top(x,y); stable forward;
    autograd vp gradient finite-difference checked."""
    nex, ney, nez = 3, 3, 3
    xm, ym = 300.0, 300.0
    def z_top(x, y):
        return 50.0 - 40.0 * torch.exp(-(((x - xm / 2) / 80) ** 2 + ((y - ym / 2) / 80) ** 2))
    ht = build_topography_hex27(nex, ney, nez, xm, ym, 360.0, z_top, n_order=N, dtype=DT)
    assert bool((ht.det_J > 0).all())
    # surface (top element row) follows z_top at control-coincident GLL nodes
    surf = slice(0, ney * nex)
    c = slice(0, N + 1, N // 2)
    zt = ht.z_e[surf, 0][:, c, c]
    assert float((zt - z_top(ht.x_e[surf, 0][:, c, c],
                             ht.y_e[surf, 0][:, c, c])).abs().max()) < 1e-9

    f0, dt, nt, delay = 12.0, 5e-4, 80, 0.06
    wav = torch.tensor(_ricker(np.arange(nt) * dt, f0, delay), dtype=DT)
    vp = (2600.0 + 0.7 * ht.z_e).contiguous()          # continuous
    src = [(xm / 2, ym / 2, 120.0)]
    rec = [(xm * 0.35, ym * 0.5, 100.0), (xm * 0.65, ym * 0.5, 110.0)]

    def misfit(vp_e):
        eq = AcousticSEM3DDeformed.build(ht, vp_e, dirichlet=("top",), cpml=None)
        return SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src,
                                   receiver_xz=rec).forward(wav)[..., 0].T.pow(2).sum()

    vp.requires_grad_(True)
    J = misfit(vp)
    assert torch.isfinite(J)
    (grad,) = torch.autograd.grad(J, vp)
    torch.manual_seed(0)
    d = torch.randn_like(vp)
    analytic = (grad * d).sum().item()
    h = 1e-2
    with torch.no_grad():
        fd = (misfit(vp + h * d).item() - misfit(vp - h * d).item()) / (2 * h)
    assert abs(analytic - fd) / (abs(fd) + 1e-300) < 1e-5


# --------------------------------------------------------------------------- #
# PropSEM elastic 3-D front end (ElasticVSSEM3D): models=[vp,vs,rho], force
# source, 3-component receivers, multishot vmap batch + boundary saving.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="elastic multishot is heavy")
def test_propsem_elastic3d_multishot_and_bs():
    from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
    dev = "cuda"
    nz, ny, nx = 21, 17, 21
    dh, dt, nt = 25.0, 4e-4, 90

    def prop(memory=None):
        return PropSEM(ElasticVSSEM3D, shape=(nz, ny, nx), dh=dh, dt=dt, nt=nt,
                       n_order=N, free_surface=False,
                       abcn=2 * N, f_dominant=12.0, dev=dev, dtype=DT,
                       impl="eager", memory=memory)

    wav = torch.tensor(_ricker(np.arange(nt) * dt, 12.0, 0.06), dtype=DT, device=dev)
    src = np.array([[7, 8, 5], [14, 8, 5]], dtype=np.int64)
    rx = np.arange(3, nx - 3, 4)
    rec = np.repeat(np.stack([rx, np.full_like(rx, 8), np.full_like(rx, 9)], 1)[None], 2, 0)
    mk = lambda v: torch.full((nz, ny, nx), v, dtype=DT, device=dev)
    base = [mk(3000.0), mk(1732.0), mk(2200.0)]

    def run(memory=None, idx=slice(None)):
        ms = [m.clone().requires_grad_(True) for m in base]
        s = prop(memory)(wav, src[idx], rec[idx], models=ms)
        s.pow(2).sum().backward()
        return s.detach(), [m.grad.clone() for m in ms]

    # forward shape (nshots, ncomp=3, nrec, nt)
    s_full, g_b = run()
    assert s_full.shape == (2, nt, len(rx), 3) and torch.isfinite(s_full).all()

    # multishot vmap batch == per-shot loop (forward + grad)
    s_loop, g_loop = [], [torch.zeros_like(base[k]) for k in range(3)]
    for b in range(2):
        s1, g1 = run(idx=slice(b, b + 1))
        s_loop.append(s1)
        for k in range(3):
            g_loop[k] += g1[k]
    s_loop = torch.cat(s_loop)
    assert (s_full - s_loop).norm() / s_loop.norm() < 1e-10
    for k in range(3):
        assert (g_b[k] - g_loop[k]).norm() / g_loop[k].norm() < 1e-9

    # boundary saving == full tape (single shot): forward exact, interior cos ~1
    sf, gf = run(idx=slice(0, 1))
    sbs, gbs = run(memory=BoundarySaving(), idx=slice(0, 1))
    assert (sbs - sf).norm() / sf.norm() < 1e-9
    core = (slice(3, nz - 3), slice(3, ny - 3), slice(3, nx - 3))
    for k in range(3):                      # bare fp64 cosine (grads ~1e-27)
        a, b = gbs[k][core].flatten(), gf[k][core].flatten()
        assert (a @ b / (a.norm() * b.norm())).item() > 0.999
