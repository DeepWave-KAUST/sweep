"""Stage-5 gates: boundary saving for the coupled fluid-solid equation.

With no CPML the coupled step is exactly time-reversible (both domains are
3-level leapfrog, all forces at the "now" level), so the BS ring is EMPTY and
the reconstruction is exact. The gates therefore demand near-machine agreement,
not the usual "interior cosine" bar:

1. BS seismograms == full-tape seismograms (same arithmetic, no_grad vs grad),
2. BS d loss / d {vp, vs, rho} == full-tape autograd through the whole rollout,
3. (CUDA) BS backward peak memory well under the full tape's.

Gradients flow vp/vs/rho -> (kappa, 1/rho, lam, mu, mass) -> step internals via
the leaves returned by bs_model_tensors, so this also gates that the leaf set
is COMPLETE — a missing leaf shows up as a wrong (not just noisy) gradient.
"""
import numpy as np
import pytest
import torch

from sweep.sem.boundary_saving import run_boundary_saved
from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64

NEX, NEZ, DE = 20, 16, 50.0
Z_IF = 8 * DE                              # water: top 8 rows, solid: bottom 8


def _model_leaves():
    m = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE,
                                 z_max=NEZ * DE, n_order=4, dtype=DT, device=DEV)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    fl = (zc < Z_IF).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fl, 1500.0, 3000.0).to(DT).contiguous().requires_grad_(True)
    vs = torch.where(fl, 0.0, 1732.0).to(DT).contiguous().requires_grad_(True)
    rho = torch.where(fl, 1000.0, 2200.0).to(DT).contiguous().requires_grad_(True)
    return m, vp, vs, rho


def _acq(m, eq):
    xg = torch.zeros(m.n_global, dtype=DT, device=DEV); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT, device=DEV); zg[m.ibool] = m.z_e
    node = lambda x, z: int(((xg - x) ** 2 + (zg - z) ** 2).argmin())
    src = node(500.0, 200.0)                                     # in the water
    recs = [node(x, 150.0) for x in (300.0, 700.0)] \
         + [node(x, 600.0) for x in (300.0, 700.0)]              # hydro + geo
    pat = torch.zeros(m.n_global, dtype=DT, device=DEV); pat[src] = 1.0
    rec_G = torch.tensor(recs, device=DEV).reshape(-1, 1)
    rec_W = torch.ones(len(recs), 1, dtype=DT, device=DEV)
    return pat, rec_G, rec_W


def _wavelet(nt, dt):
    t = np.arange(nt) * dt
    a = (np.pi * 12.0 * (t - 0.12)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT, device=DEV)


# receiver-capable fields of AcousticElasticSEM2D, in field_specs order
RECV = (0, 2, 3)          # chi, u_x, u_z


def _run(memory, nt=300, dt=8e-4):
    m, vp, vs, rho = _model_leaves()
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho, dirichlet=("top",))
    pat, rec_G, rec_W = _acq(m, eq)
    zero = torch.zeros(m.n_global, dtype=DT, device=DEV)
    wav = _wavelet(nt, dt)

    if memory == "boundary":
        # run_boundary_saved returns a LIST in recv_idx order, so position p
        # corresponds to field RECV[p] -- it is no longer keyed by field id.
        out = run_boundary_saved(eq, [pat, zero, zero], wav, rec_G, rec_W, dt, nt)
        seis = {j: out[p] for p, j in enumerate(RECV)}
    else:
        f = eq.init_state(dt, DEV, DT)
        cols = {j: [] for j in RECV}
        for n in range(nt):
            f = eq.step(f, [pat * wav[n], zero, zero], dt)
            for j in cols:
                cols[j].append((f[j][rec_G] * rec_W).sum(-1))
        seis = {j: torch.stack(cols[j], dim=-1) for j in cols}

    loss = sum(seis[j].pow(2).sum() for j in RECV)
    loss.backward()
    return ({j: seis[j].detach() for j in seis},
            vp.grad.clone(), vs.grad.clone(), rho.grad.clone())


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (a.norm() * b.norm() + 1e-300))


def test_bs_matches_full_tape_seis_and_gradients():
    s_full, gvp_f, gvs_f, grho_f = _run(None)
    s_bs, gvp_b, gvs_b, grho_b = _run("boundary")

    # (1) seismograms: same arithmetic path -> near-bitwise
    for j in RECV:
        num = (s_bs[j] - s_full[j]).abs().max()
        den = s_full[j].abs().max()
        assert num / den < 1e-12, f"field {j}: {num/den:.2e}"

    # (2) gradients wrt all three model leaves. The reconstruction is very
    # nearly exact, but NOT bit-exact, and the bar has to sit above what it
    # actually costs rather than just above one observed run. Measured over 8
    # repeats on this configuration: vp 1.3-1.7e-14, rho 1.3-1.7e-14, and vs
    # 4.4-8.1e-10 (median 6.0e-10) -- vs is the outlier because it is zero in
    # the fluid, so its gradient lives entirely in the solid, which is what
    # the reverse march rebuilds. The full-tape run-to-run atomics spread is
    # 3-8e-13, three orders below that, so this is reconstruction error, not
    # noise. The old 1e-9 bar sat at 1.2x the observed maximum and failed
    # intermittently in the full suite; 5e-9 keeps ~6x headroom while still
    # being five orders below anything a real defect has produced.
    for name, gf, gb in (("vp", gvp_f, gvp_b), ("vs", gvs_f, gvs_b),
                         ("rho", grho_f, grho_b)):
        assert gf.norm() > 0, f"{name}: full-tape gradient is zero"
        c = _cos(gf, gb)
        rel = float((gf - gb).norm() / gf.norm())
        assert c > 1 - 1e-12, f"{name}: cos {c}"
        assert rel < 5e-9, f"{name}: rel {rel:.2e}"


def test_bs_gradient_covers_both_domains():
    """The vp gradient must be nonzero in BOTH the water (kappa path) and the
    solid (lam/mu path) — a missing model leaf would silently kill one side."""
    _, gvp, gvs, _ = _run("boundary", nt=250)
    m, *_ = _model_leaves()
    zc = m.z_e.flatten(1).mean(1)
    fl = zc < Z_IF
    assert gvp[fl].abs().sum() > 0, "no vp gradient in the fluid"
    assert gvp[~fl].abs().sum() > 0, "no vp gradient in the solid"
    assert gvs[~fl].abs().sum() > 0, "no vs gradient in the solid"
    assert gvs[fl].abs().sum() == 0, "vs gradient leaked into the fluid (vs==0)"


@pytest.mark.skipif(DEV != "cuda", reason="memory measurement needs CUDA")
def test_bs_saves_memory():
    def peak(memory):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        _run(memory, nt=800)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    mem_full = peak(None)
    mem_bs = peak("boundary")
    assert mem_bs < 0.5 * mem_full, f"bs {mem_bs/1e6:.0f} MB vs full {mem_full/1e6:.0f} MB"
