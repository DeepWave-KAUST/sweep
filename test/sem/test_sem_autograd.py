"""Verify the SEM acoustic 2-D PoC is end-to-end differentiable.

Sets ``vp_e.requires_grad_(True)``, runs forward, builds an L2 loss against
a synthetic target trace, calls ``loss.backward()``, and:
  1. Checks the gradient is finite and non-zero.
  2. Spot-checks the sign of d(loss)/d(vp) against a finite-difference
     reference at two elements (one near source, one near receiver).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import AcousticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t: torch.Tensor, f0: float, delay: float) -> torch.Tensor:
    a = (torch.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _run_forward(vp_e: torch.Tensor, mesh, dt: float, nt: int,
                 src_xz, rec_xz, source_wave: torch.Tensor) -> torch.Tensor:
    eq = AcousticSEM2D.build(mesh, vp_e)
    prop = SEMPropagator.build(eq, dt=dt, nt=nt,
                               source_xz=[src_xz], receiver_xz=rec_xz)
    return prop.forward(source_wave)[..., 0].T


def test_backward_runs_and_grad_is_finite():
    """loss.backward() succeeds and produces a finite, non-zero gradient."""
    dtype = torch.float64
    mesh = build_quad4_mesh(nex=8, nez=6, x_max=400.0, z_max=300.0,
                            n_order=4, dtype=dtype)
    nt, dt = 300, 0.0008
    f0, delay = 15.0, 0.06
    src_xz = (200.0, 80.0)
    rec_xz = [(300.0, 80.0), (100.0, 80.0)]

    vp0 = torch.full(
        (mesh.n_elements, mesh.ngll, mesh.ngll), 2000.0,
        dtype=dtype, requires_grad=True,
    )
    t = torch.arange(nt, dtype=dtype) * dt
    src_wave = ricker(t, f0=f0, delay=delay)

    seis_obs = _run_forward(
        # target: same model but +50 m/s near (mid-x, mid-z) so the trace is
        # measurably different.
        (2000.0 + 50.0 * torch.ones_like(vp0).detach()),
        mesh, dt, nt, src_xz, rec_xz, src_wave,
    ).detach()

    seis_pred = _run_forward(vp0, mesh, dt, nt, src_xz, rec_xz, src_wave)
    loss = ((seis_pred - seis_obs) ** 2).sum()
    loss.backward()

    assert vp0.grad is not None, "no gradient produced"
    g = vp0.grad
    assert torch.isfinite(g).all(), "grad contains NaN/Inf"
    assert g.abs().sum() > 0, "grad is identically zero"
    print(f"||grad||_inf = {g.abs().max().item():.4e}, "
          f"||grad||_2 = {g.norm().item():.4e}")


def test_backward_sign_matches_finite_difference():
    """At a hand-picked element, sign of analytic grad matches FD."""
    dtype = torch.float64
    mesh = build_quad4_mesh(nex=8, nez=6, x_max=400.0, z_max=300.0,
                            n_order=4, dtype=dtype)
    nt, dt = 300, 0.0008
    f0, delay = 15.0, 0.06
    src_xz = (200.0, 80.0)
    rec_xz = [(300.0, 80.0)]

    vp_base = 2000.0 * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                  dtype=dtype)
    t = torch.arange(nt, dtype=dtype) * dt
    src_wave = ricker(t, f0=f0, delay=delay)
    seis_obs = _run_forward(vp_base + 80.0, mesh, dt, nt,
                            src_xz, rec_xz, src_wave).detach()

    def loss_at(vp_e):
        s = _run_forward(vp_e, mesh, dt, nt, src_xz, rec_xz, src_wave)
        return ((s - seis_obs) ** 2).sum()

    # Analytic gradient
    vp = vp_base.clone().requires_grad_(True)
    L0 = loss_at(vp)
    L0.backward()
    g_analytic = vp.grad.detach().clone()

    # Pick a representative element (one near the receiver) and FD-perturb
    # one of its GLL nodes.
    e_idx = 5 * mesh.nex + 6     # row 5, col 6 (near receiver at x=300, z=80)
    α, β = 2, 2                  # central GLL node
    eps = 1.0                    # 1 m/s perturbation
    vp_plus = vp_base.clone()
    vp_minus = vp_base.clone()
    vp_plus[e_idx, α, β]  += eps
    vp_minus[e_idx, α, β] -= eps
    L_plus  = loss_at(vp_plus).item()
    L_minus = loss_at(vp_minus).item()
    fd = (L_plus - L_minus) / (2 * eps)
    an = g_analytic[e_idx, α, β].item()
    print(f"analytic d(loss)/d(vp[{e_idx},{α},{β}]) = {an:+.6e}")
    print(f"finite-diff (eps=1 m/s)              = {fd:+.6e}")
    # Same sign and within ~10% (FD is itself approximate for finite eps).
    assert an * fd > 0, f"sign mismatch: analytic={an}, FD={fd}"
    rel = abs(an - fd) / max(abs(fd), 1e-30)
    assert rel < 0.1, f"rel mismatch {rel:.2%} (analytic={an}, FD={fd})"


if __name__ == "__main__":
    test_backward_runs_and_grad_is_finite()
    print()
    test_backward_sign_matches_finite_difference()
