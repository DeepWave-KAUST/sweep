"""One FWI gradient of the eager propagator with Marmousi split across GPUs,
checked against the same gradient on one GPU.

``ModelParallel`` takes an ``impl="eager"`` propagator with the same call as
the stepped one: each rank runs the eager rollout on its x strip plus a halo,
and the halo is refilled from the neighbours every ``exchange_steps`` steps.
The gradient comes from autograd.

    torchrun --standalone --nproc-per-node=2 \\
        examples/multi-gpu/torch/dd_eager_marmousi_2d.py --outdir out

Rank 0 also runs the single-domain gradient and writes
``dd_eager_marmousi_2d.png``.  ``--device cpu`` runs it on CPU processes
(the exchange then goes through gloo); add ``--downsample 16`` there.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.datasets import load                          # noqa: E402
from sweep.equations import Acoustic                     # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel   # noqa: E402
from sweep.propagator.torch import PropTorch             # noqa: E402


def ricker(nt, dt, fm):
    t = np.arange(nt, dtype=np.float32) * dt - 1.5 / fm
    a = (np.pi * fm * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a) * 1e3).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--downsample", type=int, default=8, help="of the 1.25 m Marmousi")
    p.add_argument("--nshot", type=int, default=4)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--fm", type=float, default=5.0)
    p.add_argument("--exchange-steps", type=int, default=None)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    dist.init_process_group("nccl" if a.device == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if a.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dev = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = torch.device("cpu")

    # Marmousi, cropped to a width the tiles divide; a smoothed start
    d = load("marmousi", variant="2d-acoustic")
    vp_true = d["vp"][::a.downsample, ::a.downsample]
    nx = vp_true.shape[1] // (16 * world) * (16 * world)
    vp_true = np.ascontiguousarray(vp_true[:, :nx], dtype=np.float32)
    dh = d["dh"][0] * a.downsample
    k = np.ones(15, np.float32) / 15
    vp_init = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 7, "edge"), k, "valid"), 0, vp_true)
    vp_init = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 7, "edge"), k, "valid"), 1, vp_init)
    nz = vp_true.shape[0]
    dt = 0.4 * dh / float(vp_true.max())
    nt = int(a.seconds / dt)

    def build():
        return PropTorch(Acoustic(spatial_order=8, device=dev, backend="torch"), impl="eager",
                         shape=(nz, nx), dev=dev, dh=dh, dt=dt, nt=nt, abcn=40,
                         source_type=["h1"], receiver_type=["h1"], free_surface=False,
                         B=a.nshot)

    wav = torch.tensor(ricker(nt, dt, a.fm), device=dev)
    sx = np.linspace(20, nx - 20, a.nshot).astype(int)
    src = np.array([[x, 4] for x in sx], dtype=np.int32)              # (nshots, 2): one point each
    rec = np.array([[[x, 4] for x in range(2, nx - 2, 2)]] * a.nshot, dtype=np.int32)

    def gradient(prop):
        """0.5 ||d(vp_init) - d(vp_true)||^2: its gradient, and the record."""
        with torch.no_grad():
            obs = prop(wav, src, rec, models=[torch.tensor(vp_true, device=dev)])
        vp = torch.tensor(vp_init, device=dev, requires_grad=True)
        syn = prop(wav, src, rec, models=[vp])
        (0.5 * (syn - obs).pow(2).sum()).backward()
        return vp.grad, syn.detach()

    # --- the decomposed run: every rank calls the same thing ---------------
    mesh = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    ddp = ModelParallel(build(), mesh, exchange_steps=a.exchange_steps)
    g_dd, syn_dd = gradient(ddp)
    dist.all_reduce(g_dd)                 # each rank holds its tile: sum = the global gradient
    rec_dd = ddp.gather_record(syn_dd)    # the full record, on rank 0 (on the CPU)

    if rank == 0:
        g_one, rec_one = gradient(build())
        rel_g = float((g_dd - g_one).abs().max() / g_one.abs().max())
        rec_one = rec_one.cpu()
        rel_d = float((rec_dd - rec_one).abs().max() / rec_one.abs().max())
        print(f"{world} tiles (halo {ddp.halo_width['x']} cells), {nz}x{nx} at {dh:.0f} m, "
              f"{nt} steps: record rel. diff {rel_d:.1e}, gradient rel. diff {rel_g:.1e}")
        plot(g_dd.cpu().numpy(), g_one.cpu().numpy(), dh, world, rel_g, Path(a.outdir))
    dist.destroy_process_group()


def plot(g_dd, g_one, dh, world, rel_g, out):
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    nz, nx = g_dd.shape
    ext = [0, nx * dh / 1e3, nz * dh / 1e3, 0]
    v = np.percentile(np.abs(g_one), 98)
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
    panels = [(g_dd, f"(a) {world} tiles"), (g_one, "(b) One domain"), (g_one - g_dd, f"(c) (b) - (a), max {rel_g:.0e} of max |(b)|")]
    for ax, (g, title) in zip(axs, panels):
        im = ax.imshow(g, cmap="seismic", vmin=-v, vmax=v, extent=ext, aspect="auto")
        ax.set_title(title, loc="left")
    for i in range(1, world):
        axs[0].axvline(i * nx // world * dh / 1e3, color="k", ls="--", lw=0.8)
    axs[-1].set_xlabel("Distance (km)")
    fig.supylabel("Depth (km)")
    fig.colorbar(im, ax=axs, shrink=0.6, label="vp gradient")
    fig.savefig(out / "dd_eager_marmousi_2d.png", dpi=200)


if __name__ == "__main__":
    main()
