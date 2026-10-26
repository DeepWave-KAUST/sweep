"""The invariant behind the C-PML interior fast path in the pointwise kernels.

``sem2d_acoustic_cpml_step_newmark_f`` and ``_step_T`` branch on
``b_x == b_z == dbx_dx == dbz_dz == 0`` and then skip everything that depends on
the C-PML memory variables: psi and zeta are identically zero wherever b is
zero, because b is the only thing that ever feeds them and they start at zero.

That is a property of the C-PML *profile builder*, not of these kernels, and
nothing else in the codebase pins it -- a future profile that leaks a non-zero
psi outside the band would silently corrupt the gradient in the whole model
region while every existing propagation test still passed. Hence this file.

The measured payoff for the branch (idle RTX 6000 Ada, 3.5M GLL nodes):
step_newmark_f -38%, step_T -43%, whole step -11%. Splitting only the ELEMENT
stiffness kernel gave far less, because in SEM the C-PML's extra work lives in
the pointwise update (6 profile arrays + 4 memory fields read, 4 written), not
in the stiffness contraction.

Note ``gax``/``gaz`` are deliberately still read on the fast path: a node with
b = 0 can belong to an element that DOES contain absorbing nodes, and the
assembled value there is non-zero.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import PropSEM
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, DH, N_ORDER, F0, VP0 = 72, 48, 20.0, 4, 8.0, 2500.0
XMAX, ZMAX = NEX * N_ORDER * DH, NEZ * N_ORDER * DH
DT = 0.30 * (0.172 * N_ORDER * DH) / (VP0 * 1.4)


def test_memory_fields_are_exactly_zero_wherever_b_vanishes():
    """Run far enough that the wavefield is well inside the absorbing layer."""
    nt = 800
    mesh = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=N_ORDER, dtype=torch.float64,
                                    device="cuda")
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp = (VP0 * (1 + 0.4 * mesh.z_e / ZMAX)).contiguous()
    rho = torch.full(shp, 2000.0, dtype=torch.float64, device="cuda")
    cpml = build_cpml(mesh, dt=DT, sides=("left", "right", "bottom", "top"),
                      pml_thickness_elements=4, c_max=VP0 * 1.5, f_dominant=F0)
    eq = AcousticPotentialSEM2D.build(mesh, vp, rho, cpml=cpml)

    zero_b = (cpml.b_x == 0) & (cpml.b_z == 0)
    # the band is a rim, so on a small mesh it can dominate; make sure this one
    # really has an interior worth testing before believing the result
    frac = float(zero_b.float().mean())
    assert int(zero_b.sum()) > 10000 and frac > 0.3, (
        f"only {int(zero_b.sum())} nodes ({100*frac:.0f}%) have b = 0; "
        "this mesh cannot discriminate")

    ng = mesh.n_global
    z = lambda: torch.zeros(ng, dtype=torch.float64, device="cuda")
    fields = tuple(z() for _ in range(6))       # chi, chi_prev, psi_x/z, zeta_x/z
    d2 = (mesh.x_e - XMAX * 0.15) ** 2 + (mesh.z_e - ZMAX * 0.30) ** 2
    e, a, b = np.unravel_index(int(d2.argmin().item()), tuple(d2.shape))
    g_src = int(mesh.ibool[e, a, b])
    t = np.arange(nt) * DT
    w = (1 - 2 * (np.pi * F0 * (t - 1.5 / F0)) ** 2) * np.exp(
        -(np.pi * F0 * (t - 1.5 / F0)) ** 2)

    worst = 0.0
    with torch.no_grad():
        for n in range(nt):
            s = z()
            s[g_src] = float(w[n])
            fields = eq.step(fields, [s], DT)
            if n % 25 == 0:
                for f in fields[2:]:            # psi_x, psi_z, zeta_x, zeta_z
                    worst = max(worst, float(f[zero_b].abs().max()))
    assert float(fields[0].abs().max()) > 1.0, "the run carried no energy"
    assert worst == 0.0, (
        f"a C-PML memory field reached {worst:.3e} where b = 0; the interior "
        "fast path in step_newmark_f / step_T assumes it stays exactly zero")


def _run(impl, fs, abcn, memory, topo=None, nz=49, nx=65, nt=700):
    lo, hi = 1500.0, 4500.0
    dt = 0.30 * (0.172 * N_ORDER * DH) / hi
    zc, xc = np.arange(nz)[:, None], np.arange(nx)[None, :]
    vp = (lo + (hi - lo) * zc / (nz - 1)
          + 0.08 * (hi - lo) * np.sin(xc / 9.0)).astype(np.float32)
    if topo is not None:            # z_top(x) in metres -> a row per column
        topo = (topo(torch.arange(nx, dtype=torch.float64) * DH) / DH).numpy()
    p = PropSEM(AcousticPotentialSEM2D, shape=(nz, nx), dh=DH, dt=dt, nt=nt,
                n_order=N_ORDER, free_surface=fs,
                abcn=abcn * N_ORDER, f_dominant=F0, dev="cuda",
                dtype=torch.float32, impl=impl, topography=topo, memory=memory)
    t = np.arange(nt, dtype=np.float32) * dt
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")
    src = np.array([[int(0.15 * nx), 4]], dtype=np.int64)
    rix = np.arange(3, nx - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 4)], -1)[None]
    m = torch.tensor(vp, device="cuda", requires_grad=True)
    s = p(wav, src, rec, models=[m])
    assert torch.isfinite(s).all()
    (s ** 2).sum().backward()
    out = m.grad.cpu().numpy().astype(np.float64)
    del p, m, s
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("memory", [Full(), BoundarySaving()], ids=["full", "bs"])
@pytest.mark.parametrize("fs,abcn", [(True, 6), (False, 12)],
                         ids=["fs-thin", "nofs-thick"])
def test_fast_path_gradient_matches_eager(fs, abcn, memory):
    ge = _run("eager", fs, abcn, memory)
    gc_ = _run("c", fs, abcn, memory)
    assert _cos(gc_, ge) >= 1 - 1e-6, f"grad cos={_cos(gc_, ge):.8f}"


def test_fast_path_gradient_matches_eager_on_topography():
    topo = lambda x: 60.0 * torch.exp(-((x - 650.0) ** 2) / (2 * 180.0 ** 2))
    ge = _run("eager", True, 6, BoundarySaving(), topo)
    gc_ = _run("c", True, 6, BoundarySaving(), topo)
    assert _cos(gc_, ge) >= 1 - 1e-6, f"grad cos={_cos(gc_, ge):.8f}"


# --------------------------------------------------------------------------- #
# 3-D. The same fast path lives in sem3d_acoustic_cpml_step_newmark / _step_T,
# with nine profile arrays and six memory fields instead of six and four. The
# payoff is smaller than in 2-D and that is geometry, not implementation: the
# band wraps five or six faces of a volume, so the interior share is 37% at
# 24^3 elements with a 4-element band where 2-D reaches 92-96%. Measured
# step_newmark -16% at 37% interior and -24% at 68%.
# --------------------------------------------------------------------------- #
NE3, DH3, F0_3, VP3 = 14, 40.0, 6.0, 2500.0


def test_3d_memory_fields_are_exactly_zero_wherever_b_vanishes():
    nt = 300
    L = NE3 * N_ORDER * DH3
    mesh = build_hex8_mesh(nex=NE3, ney=NE3, nez=NE3, x_max=L, y_max=L, z_max=L,
                           n_order=N_ORDER, dtype=torch.float64, device="cuda")
    dt = 0.30 * (0.172 * N_ORDER * DH3) / (VP3 * 1.3)
    vp = torch.full_like(mesh.z_e, VP3)
    cpml = build_cpml3d(mesh, dt=dt,
                        sides=("left", "right", "front", "back", "bottom"),
                        pml_thickness_elements=2, c_max=VP3 * 1.3,
                        f_dominant=F0_3)
    eq = AcousticSEM3D.build(mesh, vp, cpml=cpml)

    zero_b = (cpml.b_x == 0) & (cpml.b_y == 0) & (cpml.b_z == 0)
    frac = float(zero_b.float().mean())
    assert frac > 0.35, (
        f"only {100*frac:.0f}% of nodes have b = 0 -- the band wraps five faces of\n"
        "        a volume, so a small 3-D mesh has almost no interior and this test\n"
        "        would prove nothing")

    ng = mesh.n_global
    z = lambda: torch.zeros(ng, dtype=torch.float64, device="cuda")
    fields = tuple(z() for _ in range(8))   # u, u_prev, psi_x/y/z, zeta_x/y/z
    d2 = ((mesh.x_e - L * 0.5) ** 2 + (mesh.y_e - L * 0.5) ** 2
          + (mesh.z_e - L * 0.25) ** 2)
    idx = np.unravel_index(int(d2.argmin().item()), tuple(d2.shape))
    g_src = int(mesh.ibool[idx])
    t = np.arange(nt) * dt
    a = (np.pi * F0_3 * (t - 1.5 / F0_3)) ** 2
    w = (1 - 2 * a) * np.exp(-a)

    worst = 0.0
    with torch.no_grad():
        for n in range(nt):
            s = z()
            s[g_src] = float(w[n])
            fields = eq.step(fields, [s], dt)
            if n % 20 == 0:
                for f in fields[2:]:        # psi_x/y/z, zeta_x/y/z
                    worst = max(worst, float(f[zero_b].abs().max()))
    assert float(fields[0].abs().max()) > 0.0, "the run carried no energy"
    assert worst == 0.0, (
        f"a 3-D C-PML memory field reached {worst:.3e} where b = 0; the interior "
        "fast path in sem3d step_newmark / step_T assumes it stays exactly zero")
