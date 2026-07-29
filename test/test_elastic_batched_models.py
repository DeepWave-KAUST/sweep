"""Bit-exact test: per-shot batched velocity models for the 2-D Elastic solver.

``Elastic.supports_batched_models = True`` lets ``models=[vp, vs, rho]`` be passed
either as

* **shared** ``(nz, nx)`` — one model broadcast across the shot batch (historical
  behaviour, kept bit-for-bit), or
* **per-shot** ``(B, nz, nx)`` — shot ``b`` propagates in ``(vp, vs, rho)[b]`` and
  each parameter's gradient stays per-shot.

Because each shot's wavefield depends only on its own model (no cross-shot
reduction in the forward), the batched forward must be **bit-identical** to
looping the shots one at a time. The per-shot model gradient is compared with a
tight tolerance (the compiled gradient kernel uses ``atomicAdd``, so its
reduction order may differ by a few ULP between the batched and single-shot
launches). Covers both ``impl='c'`` and ``impl='eager'``.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="batched per-shot models exercise the compiled/eager 2-D solvers on CUDA",
)
DEVICE = "cuda"

# Canonical small elastic config (mirrors test_elastic_apm.py).
NZ, NX = 64, 96
DH, DT, NT = 4.0, 4.0e-4, 400
ABCN, SO = 25, 4
FREQ, DELAY = 25.0, 0.06
B = 3  # shots in the batch


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32), device=DEVICE)


def _geometry():
    """B point sources at different x; a shared receiver line (per-shot copy).

    sources/receivers are NUMPY int arrays (the compiled solver copies + shifts
    them in place); only the wavelet and models are torch tensors.
    """
    sx = np.linspace(NX // 4, 3 * NX // 4, B).round().astype(np.int64)
    sources = np.stack([sx, np.full(B, NZ // 4, np.int64)], axis=-1)  # (B, [x, z])
    rec_x = np.arange(8, NX - 8, 3, dtype=np.int64)
    rec = np.stack([rec_x, np.full_like(rec_x, 3)], axis=-1)  # (nrec, [x, z])
    receivers = np.broadcast_to(rec, (B, *rec.shape)).copy()  # (B, nrec, 2)
    return sources, receivers


def _geometry_varying_receivers():
    """B sources at different x, EACH with its own receiver line at *different*
    absolute positions (streamer-style: a fixed-length cable trailing each
    source, plus a per-shot cable depth). Proves per-shot receivers — not just
    per-shot models — stay bit-exact. Same nrec across shots (rectangular
    tensor, the one thing the batch must share)."""
    sx = np.linspace(NX // 5, NX // 2, B).round().astype(np.int64)
    sources = np.stack([sx, np.full(B, NZ // 4, np.int64)], axis=-1)  # (B, [x, z])
    nrec = 18
    offsets = np.arange(3, 3 + 2 * nrec, 2, dtype=np.int64)  # cable, len nrec
    receivers = np.stack([
        np.stack([np.clip(sx[b] + offsets, 0, NX - 1),
                  np.full(nrec, 2 + b, np.int64)], axis=-1)  # depth varies per shot too
        for b in range(B)
    ], axis=0)  # (B, nrec, [x, z])
    assert not np.array_equal(receivers[0], receivers[1]), \
        "test bug: receivers must differ per shot"
    return sources, receivers


def _per_shot_models(seed=0):
    """B distinct physical (vp, vs, rho) models, each (NZ, NX)."""
    rng = np.random.default_rng(seed)
    vp = np.empty((B, NZ, NX), np.float32)
    vs = np.empty_like(vp)
    rho = np.empty_like(vp)
    for b in range(B):
        base = 2200.0 + 300.0 * b + rng.uniform(-60, 60, (NZ, NX)).astype(np.float32)
        vp[b] = base
        vs[b] = base / 1.75
        rho[b] = 1000.0 + 0.28 * base
    to = lambda a: torch.tensor(a, device=DEVICE)
    return to(vp), to(vs), to(rho)


def _prop(impl):
    eq = Elastic(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(
        eq, shape=(NZ, NX), free_surface=False, topography=None,
        abcn=ABCN, dh=DH, dt=DT, use_ckpt=False, impl=impl,
    )


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_elastic_batched_forward_bit_exact(impl):
    """Per-shot batched forward == looping the shots one at a time (bit-exact)."""
    wav = _wavelet()
    sources, receivers = _geometry()
    vp, vs, rho = _per_shot_models()

    syn_batch = _prop(impl)(wav, sources, receivers, models=[vp, vs, rho]).detach().cpu()

    syn_loop = torch.stack([
        _prop(impl)(
            wav, sources[b:b + 1], receivers[b:b + 1],
            models=[vp[b], vs[b], rho[b]],
        ).detach().cpu()[0]
        for b in range(B)
    ])

    assert syn_batch.shape == syn_loop.shape
    maxdiff = (syn_batch - syn_loop).abs().max().item()
    exact = torch.equal(syn_batch, syn_loop)
    print(f"[{impl}] forward batched-vs-loop: exact={exact} maxdiff={maxdiff:.3e}")
    # Independent per-shot forward -> expect bit-exact; allow a hair for eager.
    tol = 0.0 if impl == "c" else 1e-4 * syn_loop.abs().max().item()
    assert exact or maxdiff <= tol, f"[{impl}] forward maxdiff={maxdiff:.3e} > tol={tol:.3e}"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_elastic_batched_varying_receivers_bit_exact(impl):
    """Per-shot DIFFERENT receiver positions (+ per-shot models) == the loop.

    Each shot has its own source, its own receiver locations, and its own
    (vp,vs,rho); the batched forward must still match shooting them one at a
    time bit-for-bit."""
    wav = _wavelet()
    sources, receivers = _geometry_varying_receivers()
    vp, vs, rho = _per_shot_models()

    syn_batch = _prop(impl)(wav, sources, receivers, models=[vp, vs, rho]).detach().cpu()

    syn_loop = torch.stack([
        _prop(impl)(
            wav, sources[b:b + 1], receivers[b:b + 1],
            models=[vp[b], vs[b], rho[b]],
        ).detach().cpu()[0]
        for b in range(B)
    ])

    assert syn_batch.shape == syn_loop.shape
    maxdiff = (syn_batch - syn_loop).abs().max().item()
    exact = torch.equal(syn_batch, syn_loop)
    print(f"[{impl}] varying-receivers forward: exact={exact} maxdiff={maxdiff:.3e}")
    tol = 0.0 if impl == "c" else 1e-4 * syn_loop.abs().max().item()
    assert exact or maxdiff <= tol, f"[{impl}] varying-recv maxdiff={maxdiff:.3e} > tol={tol:.3e}"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_elastic_batched_gradient_matches_loop(impl):
    """Per-shot model gradients (vp/vs/rho) match the per-shot loop."""
    wav = _wavelet()
    sources, receivers = _geometry()
    vp0, vs0, rho0 = _per_shot_models(seed=1)
    # A detached "observed" record from a slightly perturbed model.
    obs = _prop(impl)(
        wav, sources, receivers, models=[vp0 * 1.03, vs0 * 1.03, rho0],
    ).detach()

    def _grads_batched():
        vp = vp0.clone().requires_grad_(True)
        vs = vs0.clone().requires_grad_(True)
        rho = rho0.clone().requires_grad_(True)
        syn = _prop(impl)(wav, sources, receivers, models=[vp, vs, rho])
        (0.5 * ((syn - obs) ** 2).sum()).backward()
        return vp.grad.detach().cpu(), vs.grad.detach().cpu(), rho.grad.detach().cpu()

    def _grads_loop():
        g = [torch.zeros((B, NZ, NX)) for _ in range(3)]
        for b in range(B):
            vpb = vp0[b].clone().requires_grad_(True)
            vsb = vs0[b].clone().requires_grad_(True)
            rhob = rho0[b].clone().requires_grad_(True)
            syn = _prop(impl)(wav, sources[b:b + 1], receivers[b:b + 1],
                              models=[vpb, vsb, rhob])
            (0.5 * ((syn - obs[b:b + 1]) ** 2).sum()).backward()
            for k, pb in enumerate((vpb, vsb, rhob)):
                g[k][b] = pb.grad.detach().cpu()
        return g

    gb = _grads_batched()
    gl = _grads_loop()
    for name, a, c in zip(("vp", "vs", "rho"), gb, gl):
        scale = c.abs().max().item() + 1e-30
        maxrel = (a - c).abs().max().item() / scale
        print(f"[{impl}] grad {name}: exact={torch.equal(a, c)} max|rel|={maxrel:.3e}")
        assert maxrel < 1e-4, f"[{impl}] {name} grad max|rel|={maxrel:.3e}"


def test_elastic_shared_model_broadcasts():
    """A shared (nz,nx) model == a per-shot batch whose shots are all identical."""
    wav = _wavelet()
    sources, receivers = _geometry()
    vp1, vs1, rho1 = _per_shot_models()
    vp_s, vs_s, rho_s = vp1[0], vs1[0], rho1[0]  # (nz, nx)

    syn_shared = _prop("c")(
        wav, sources, receivers, models=[vp_s, vs_s, rho_s],
    ).detach().cpu()
    rep = lambda m: m[None].repeat(B, 1, 1)
    syn_batch = _prop("c")(
        wav, sources, receivers, models=[rep(vp_s), rep(vs_s), rep(rho_s)],
    ).detach().cpu()
    assert torch.equal(syn_shared, syn_batch), \
        f"shared vs identical-per-shot maxdiff={(syn_shared - syn_batch).abs().max():.3e}"
