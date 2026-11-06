"""One 3-D acoustic SEM shot across several GPUs, by domain decomposition.

    torchrun --standalone --nproc_per_node=4 \
        examples/multi-gpu/torch/sem_dd_acoustic3d.py [--check]

``sweep.sem.dd.SEMModelParallel`` cuts the SEM mesh into x strips, one per
rank, and runs the ``impl='c'`` kernels of ``AcousticSEM3D`` on each strip,
summing the interface nodes between neighbouring strips after every step's
assembly (NCCL). The call is ``PropSEM``'s, for one shot: the wavelet, the
source and the receivers as global node ids, and the models. Under
``torchrun`` each rank's record holds the receivers of its own strip and its
model gradient the nodes of its own strip, so a sum over the ranks assembles
both; the loss below is a plain sum of squares, which adds up the same way.

The script records data in a model with a velocity anomaly, takes the
gradient of the misfit at the background model, and prints the time and the
peak GPU memory of each rank. ``--check`` runs the same shot on one GPU with
``PropSEM`` (rank 0) and compares the records and the gradients.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
import torch.distributed as dist

from sweep.propagator import BoundarySaving, Full
from sweep.sem import PropSEM
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import AcousticSEM3D

NO = 4                     # GLL order: the elements are NO grid cells wide
ABCN = 4 * NO              # the C-PML, four elements, on every side but the free top


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", type=int, nargs=3, default=(97, 129, 257),
                    metavar=("NZ", "NY", "NX"),
                    help="the model grid; every n - 1 a multiple of 4")
    ap.add_argument("--dh", type=float, default=10.0, help="grid spacing (m)")
    ap.add_argument("--dt", type=float, default=5e-4, help="time step (s)")
    ap.add_argument("--nt", type=int, default=2000, help="time steps")
    ap.add_argument("--f0", type=float, default=10.0, help="Ricker peak frequency (Hz)")
    ap.add_argument("--memory", choices=("rim", "boundary", "full"), default="rim",
                    help="what the backward keeps: each strip's ring of the C-PML band's "
                         "first element layer (rim), of the whole band, or every step")
    ap.add_argument("--check", action="store_true",
                    help="rank 0 also runs the shot on one GPU and compares")
    return ap.parse_args()


def node_id(prop, ix, iy, iz):
    """The global SEM node at the model grid point ``(ix, iy, iz)``.

    The node grid of the flat 3-D mesh is the model grid with the C-PML added
    (row-major, z slowest), and its nodes coincide with grid points at the
    element corners, where ``ix``, ``iy`` and ``iz`` are multiples of NO."""
    assert ix % NO == 0 and iy % NO == 0 and iz % NO == 0, "an element corner"
    nz, ny, nx = prop.shape
    nyg, nxg = ny + 2 * prop.abcn, nx + 2 * prop.abcn
    z0 = 0 if "top" in prop.free_surface else prop.abcn
    return ((iz + z0) * nyg + iy + prop.abcn) * nxg + ix + prop.abcn


def models(shape, dh):
    """A velocity rising from 2000 to 3000 m/s with depth, and the same with a
    +10% Gaussian anomaly 400 m deep under the middle of the model, which
    straddles the central cut."""
    nz, ny, nx = shape
    z, y, x = np.meshgrid(*(np.arange(n) * dh for n in shape), indexing="ij")
    background = 2000.0 + 1000.0 * z / z.max()
    r2 = ((x - x.mean()) ** 2 + (y - y.mean()) ** 2 + (z - 400.0) ** 2) / 80.0 ** 2
    true = background * (1.0 + 0.1 * np.exp(-r2))
    as_t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")
    return as_t(true), as_t(background)


def memory_options(kind):
    """The propagator's gradient memory: boundary saving of the band's first
    element layer (``'rim'``), of the whole band, or the full tape."""
    if kind == "full":
        return dict(memory=Full())
    return dict(memory=BoundarySaving(), bs_band_mode="rim" if kind == "rim" else "full")


def ricker(nt, dt, f0):
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.tensor((1.0 - 2.0 * a) * np.exp(-a), dtype=torch.float32, device="cuda")


def main():
    args = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    nz, ny, nx = args.shape

    # every rank builds the same propagator, the configuration the strips read
    prop = PropSEM(AcousticSEM3D, shape=args.shape, dh=args.dh, dt=args.dt, nt=args.nt,
                   n_order=NO, abcn=ABCN, free_surface=True, f_dominant=args.f0,
                   dev="cuda", impl="c", **memory_options(args.memory))
    vp_true, vp_init = models(args.shape, args.dh)
    wavelet = ricker(args.nt, args.dt, args.f0)

    # the source 80 m deep in the middle; receivers 40 m deep every 80 m
    corner = lambda n: (n // 2) // NO * NO
    sxyz = (corner(nx), corner(ny), 2 * NO)
    rxy = [(x, y) for y in range(2 * NO, ny - 2 * NO, 2 * NO)
           for x in range(2 * NO, nx - 2 * NO, 2 * NO)]
    src = node_id(prop, *sxyz)
    rec = torch.tensor([node_id(prop, x, y, NO) for x, y in rxy], device="cuda")

    ddp = SEMModelParallel(prop, StripTopology(world, rank=rank))

    # the observed data, scaled to order one (the wave equation is linear in the source)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        obs = ddp(wavelet, src, rec, [vp_true])
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    peak = obs.abs().max()
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    wavelet, obs = wavelet / peak, obs / peak

    # the gradient of the misfit at the background model
    vp = vp_init.clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = ddp(wavelet, src, rec, [vp])
    loss = 0.5 * (pred - obs).pow(2).sum()          # this rank's receivers
    loss.backward()
    torch.cuda.synchronize()
    t_grad = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30

    loss = loss.detach()
    dist.all_reduce(loss)                           # the global misfit
    grad = vp.grad
    dist.all_reduce(grad)                           # the global gradient
    record = ddp.gather_record(pred.detach())       # (nt, nrec) on rank 0, None elsewhere
    stats = torch.tensor([t_fwd, t_grad, mem], device="cuda")
    every = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(every, stats)

    if rank == 0:
        print(f"AcousticSEM3D, model {nz} x {ny} x {nx} ({nz * ny * nx / 1e6:.1f} M points), "
              f"{args.nt} steps, {len(rxy)} receivers, {world} strips, memory={args.memory}")
        print(f"misfit {float(loss):.4e}, |gradient| {float(grad.norm()):.4e}")
        print(f"{'rank':>4s} {'forward (s)':>12s} {'fwd+bwd (s)':>12s} {'peak (GiB)':>11s}")
        for r, s in enumerate(every):
            print(f"{r:4d} {float(s[0]):12.2f} {float(s[1]):12.2f} {float(s[2]):11.2f}")

    if not args.check:
        dist.destroy_process_group()
        return

    # The check compares the gradient of the record's energy, which the direct
    # wave dominates: the misfit's gradient comes from a small residual, and the
    # run-to-run spread of the CUDA kernels' atomic sums is a percent of it.
    ve = vp_init.clone().requires_grad_(True)
    (0.5 * ddp(wavelet, src, rec, [ve]).pow(2).sum()).backward()
    g_energy = ve.grad
    dist.all_reduce(g_energy)

    if rank == 0:
        # the same shot on this GPU alone, through PropSEM: the sources and
        # receivers as model grid points, (x, y, z), at the same element corners.
        # The strips' memory goes first: this GPU has to hold the whole problem.
        g_energy = g_energy.cpu()
        del ddp, prop, pred, vp, ve, grad
        gc.collect()
        torch.cuda.empty_cache()
        one = PropSEM(AcousticSEM3D, shape=args.shape, dh=args.dh, dt=args.dt, nt=args.nt,
                      n_order=NO, abcn=ABCN, free_surface=True, f_dominant=args.f0,
                      dev="cuda", impl="c", **memory_options(args.memory))
        s1 = np.array([sxyz], dtype=np.float32)
        r1 = np.array([[(x, y, NO) for x, y in rxy]], dtype=np.float32)

        def single():
            v = vp_init.clone().requires_grad_(True)
            p = one(wavelet, s1, r1, models=[v])[0, :, :, 0]
            (0.5 * p.pow(2).sum()).backward()
            return p.detach().cpu(), v.grad.cpu()

        p_a, g_a = single()
        _, g_b = single()                           # again: the atomics' own spread
        rel = lambda a, b: float((a - b).norm() / b.norm())
        print(f"one GPU against {world}: record {rel(record, p_a):.1e}, gradient of the record's "
              f"energy {rel(g_energy, g_a):.1e} (two runs on one GPU: {rel(g_b, g_a):.1e})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
