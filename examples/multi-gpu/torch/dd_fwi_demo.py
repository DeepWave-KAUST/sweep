"""ModelParallel end-to-end demo: build a domain-decomposed solver, run the
forward to get synthetic data, then a plain ``loss.backward()`` to get the
model gradient of an FWI misfit — the three things you do every FWI iteration.

DD behaves like the single-domain ``PropTorch``: same call signature, and the
forward is autograd-transparent, so the ONLY DD-specific lines below are two
``all_reduce`` calls (global misfit + global gradient assembly).

Run (model split across N GPUs along x):

    torchrun --standalone --nproc-per-node=2 examples/multi-gpu/torch/dd_fwi_demo.py

What DD gives you: the *global* model never has to fit on one GPU. Each rank
owns one x-tile (+ a thin halo); the forward/backward are bit-identical to the
single-domain run (the halo exchange makes the split invisible to physics).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic                       # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel      # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.08):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def make_models(nz, nx):
    """A layered true model + a box anomaly; the 'init' is the layered
    background WITHOUT the anomaly (so the gradient should light up the box)."""
    z = np.linspace(0, 1, nz, dtype=np.float32)[:, None]
    base = np.broadcast_to(1800.0 + 700.0 * z, (nz, nx)).copy()
    base[nz // 2:, :] += 300.0                       # a deeper fast layer
    true, init = base.copy(), base.copy()
    true[nz // 3: nz // 2, 2 * nx // 5: 3 * nx // 5] += 400.0   # the box to recover
    return true, init


def main():
    # ---- 0. distributed init (one process per GPU, launched by torchrun) ----
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    # ---- 1. problem definition (GLOBAL — every rank uses the same shapes) ----
    nz, nx = 96, 48 * world          # x is split across `world` GPUs
    so, abcn, nt = 4, 20, 700
    dh = 15.0
    true_vp, init_vp = make_models(nz, nx)
    wav = ricker(nt, DT, fm=10.0)
    src = np.array([[[nx // 2, 3]]], dtype=np.int32)                        # surface shot
    rec = np.array([[[ix, 3] for ix in range(2, nx - 2, 2)]], dtype=np.int32)  # surface line

    # ---- 2. BUILD THE DD SOLVER: build a normal global propagator, then
    #         ModelParallel(prop, mesh) runs it decomposed (1-D x-cut, py=1) ----
    mesh = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    eq = Acoustic(spatial_order=so, device=dev, backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=(nz, nx), dh=dh, dt=DT,
                     nt=nt, abcn=abcn, source_type=["h1"], receiver_type=["h1"],
                     dev=dev, free_surface=False)
    ddp = ModelParallel(prop, mesh)
    if rank == 0:
        print(f"[demo] world={world}  global=(nz={nz}, nx={nx})  "
              f"tile per GPU = (nz={nz}, nxp={nx // world})  nt={nt}")

    # ---- 3. FORWARD -> synthetic data (exactly like single-domain) ----------
    # "observed" data: forward through the TRUE model (no grad needed).
    obs = ddp.forward(wav, src, rec, models=[true_vp]).clone()
    # the model we invert: here a replicated global vp leaf (small demo, fits
    # each GPU). For a model too big for one card, pass this rank's TILE leaf
    # instead (vp.grad then comes back tile-shaped; skip the grad all_reduce).
    vp = torch.tensor(init_vp, device=dev, requires_grad=True)
    syn = ddp(wav, src, rec, models=[vp])              # autograd-transparent record

    # ---- 4. MISFIT + 5. BACKWARD — just loss.backward(), like non-DD --------
    loss = 0.5 * (syn - obs).pow(2).sum()
    dist.all_reduce(loss)                              # [DD] global misfit over tiles
    loss.backward()                                    # autograd -> vp.grad
    dist.all_reduce(vp.grad)                           # [DD] assemble global gradient
    # vp.grad is now the full global FWI gradient, identical on every rank.

    if rank == 0:
        grad_full = vp.grad.detach().cpu().numpy()
        print(f"[demo] global misfit J = {loss.item():.6e}")
        print(f"[demo] |grad|max = {np.abs(grad_full).max():.3e}  "
              f"grad shape = {grad_full.shape}")
        # one illustrative steepest-descent step: vp_new = vp - alpha * grad
        alpha = 5.0 / (np.abs(grad_full).max() + 1e-30)
        vp_updated = init_vp - alpha * grad_full
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            lo, hi = np.percentile(grad_full, [2, 98])      # signed two-sided range
            fig, ax = plt.subplots(2, 2, figsize=(13, 7))
            ax[0, 0].imshow(true_vp, cmap="jet", aspect="auto"); ax[0, 0].set_title("true vp (with box)")
            ax[0, 1].imshow(init_vp, cmap="jet", aspect="auto"); ax[0, 1].set_title("init vp (no box)")
            im = ax[1, 0].imshow(grad_full, cmap="seismic", vmin=lo, vmax=hi, aspect="auto")
            ax[1, 0].set_title(f"DD model gradient  (J={loss.item():.3e}, {world} tiles)")
            fig.colorbar(im, ax=ax[1, 0], shrink=0.8)
            ax[1, 1].imshow(vp_updated, cmap="jet", aspect="auto"); ax[1, 1].set_title("vp after 1 GD step")
            for a in ax.ravel():
                a.set_xlabel("x"); a.set_ylabel("z")
                for k in range(1, world):                    # tile cut lines
                    a.axvline(k * (nx // world), color="k", ls="--", lw=0.6, alpha=0.5)
            fig.suptitle("ModelParallel: build solver -> forward (syn) -> loss.backward() (gradient)")
            fig.tight_layout()
            out = Path(__file__).resolve().parent / "dd_fwi_demo.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            print(f"[demo] saved figure: {out}")
        except Exception as e:                              # plotting is optional
            print(f"[demo] (plot skipped: {e})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
