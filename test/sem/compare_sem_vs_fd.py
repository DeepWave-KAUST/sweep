"""Compare SEM acoustic 2-D against sweep's FD Acoustic on the same setup.

Both solvers run on a uniform vp domain with a Ricker source near the top.
Boundary conditions don't fully match (SEM has free top + Stacey sides/bottom;
sweep FD here uses free top + CPML on all four sides), so we restrict the
comparison to a windowed direct arrival, before any reflection reaches the
receivers.

Run:
    PYTHONPATH=src python test/sem/compare_sem_vs_fd.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.sem import AcousticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t: torch.Tensor, f0: float, delay: float) -> torch.Tensor:
    a = (torch.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum() / (na * nb))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Common physical setup
    # ------------------------------------------------------------------
    x_max, z_max = 1000.0, 750.0
    vp_const = 2000.0
    dt = 0.0008
    nt = 700
    f0 = 12.0
    delay = 0.10

    t = torch.arange(nt, dtype=torch.float64) * dt
    src_xz = (500.0, 100.0)
    rec_x_array = np.arange(50.0, 1000.0, 50.0)
    rec_xz = [(x, 100.0) for x in rec_x_array]
    print(f"physical setup: domain {x_max}x{z_max} m, vp={vp_const}, "
          f"f0={f0}, dt={dt}, nt={nt}")

    # ------------------------------------------------------------------
    # SEM run
    # ------------------------------------------------------------------
    nex, nez = 20, 15            # dh_e = 50 m, N=4 -> ~12 m GLL spacing
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=4, dtype=torch.float64, device="cpu")
    vp_e = vp_const * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                 dtype=torch.float64)
    sem = AcousticSEM2D.build(mesh, vp_e)
    sem_prop = SEMPropagator.build(sem, dt=dt, nt=nt,
                                   source_xz=[src_xz], receiver_xz=rec_xz)
    src_wave = ricker(t, f0=f0, delay=delay)
    sem_out = sem_prop.forward(src_wave)
    sem_seis = sem_out[..., 0].T.numpy()             # (n_rec, nt)
    print(f"SEM forward done: seis shape {sem_seis.shape}")

    # ------------------------------------------------------------------
    # FD run (sweep.equations.Acoustic)
    # ------------------------------------------------------------------
    dh_fd = 10.0
    nx_fd = int(round(x_max / dh_fd)) + 1
    nz_fd = int(round(z_max / dh_fd)) + 1
    vp_fd = torch.full((nz_fd, nx_fd), vp_const, dtype=torch.float32)

    ix_s = int(round(src_xz[0] / dh_fd))
    iz_s = int(round(src_xz[1] / dh_fd))
    sources_fd = torch.tensor([[ix_s, iz_s]], dtype=torch.int64)
    iz_r = int(round(100.0 / dh_fd))
    ix_r = np.round(rec_x_array / dh_fd).astype(np.int64)
    receivers_fd = torch.tensor(
        np.stack([ix_r, np.full_like(ix_r, iz_r)], axis=-1)
    ).to(torch.int64).unsqueeze(0)   # (1, n_rec, 2)
    src_wave_fd = src_wave.to(torch.float32)

    eq_fd = Acoustic(spatial_order=4, device="cpu", backend="torch")
    prop_fd = PropTorch(
        eq_fd,
        shape=(nz_fd, nx_fd),
        free_surface=True,
        abcn=40,
        dh=dh_fd,
        dt=dt,
        impl="eager",
        use_ckpt=False,
    )
    with torch.no_grad():
        rec_fd = prop_fd(src_wave_fd, sources_fd, receivers_fd, models=[vp_fd])
    # rec_fd shape: (nshot=1, nt, n_rec, nfield=1)
    fd_seis = rec_fd[0, :, :, 0].cpu().numpy().T          # (n_rec, nt)
    print(f"FD forward done: rec shape {tuple(rec_fd.shape)}, "
          f"trimmed to {fd_seis.shape}")

    # ------------------------------------------------------------------
    # Windowed comparison: direct arrival only.
    # window = [delay - 2/f0, delay + offset/c + 2/f0]
    # ------------------------------------------------------------------
    src_x, src_z = src_xz
    print()
    print(f"{'rec_x':>7s} {'win[s]':>14s} {'cos':>10s} {'rel_L2':>10s}")
    cos_all = []
    relL2_all = []
    for j, (rx, rz) in enumerate(rec_xz):
        offset = np.hypot(rx - src_x, rz - src_z)
        t0 = max(0.0, delay - 2 / f0)
        t1 = min(nt * dt, delay + offset / vp_const + 2 / f0)
        idx = (t.numpy() >= t0) & (t.numpy() <= t1)
        if idx.sum() < 5:
            continue
        a = sem_seis[j, idx].astype(np.float64)
        b = fd_seis[j, idx].astype(np.float64)
        # Normalize each to unit peak so amplitude differences in source
        # convention don't dominate the comparison (we care about waveform).
        if np.max(np.abs(a)) > 0:
            a = a / np.max(np.abs(a))
        if np.max(np.abs(b)) > 0:
            b = b / np.max(np.abs(b))
        c = cos_sim(a, b)
        r = rel_l2(a, b)
        cos_all.append(c)
        relL2_all.append(r)
        print(f"{rx:7.0f} {t0:.3f}..{t1:.3f}  {c:10.6f} {r:10.4f}")
    print()
    print(f"cosine    median = {np.median(cos_all):.6f}, "
          f"min = {np.min(cos_all):.6f}")
    print(f"rel_L2    median = {np.median(relL2_all):.4f}, "
          f"max = {np.max(relL2_all):.4f}")

    # ------------------------------------------------------------------
    # Plot overlays at a few representative receivers
    # ------------------------------------------------------------------
    pick_indices = [3, 9, 14]            # near, mid, far
    fig, axes = plt.subplots(len(pick_indices), 1, figsize=(8, 6), sharex=True)
    for ax, j in zip(axes, pick_indices):
        rx, rz = rec_xz[j]
        a = sem_seis[j] / max(np.max(np.abs(sem_seis[j])), 1e-30)
        b = fd_seis[j] / max(np.max(np.abs(fd_seis[j])), 1e-30)
        ax.plot(t.numpy(), a, label="SEM", lw=1.4)
        ax.plot(t.numpy(), b, "--", label="FD", lw=1.0)
        ax.set_ylabel("amp (norm)")
        ax.set_title(f"receiver x={rx:.0f}, offset={abs(rx-src_x):.0f} m")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    overlay_path = out / "sem_vs_fd_traces.png"
    fig.savefig(overlay_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {overlay_path.resolve()}")

    # Gather side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, seis, title in [
        (axes[0], sem_seis, "SEM"),
        (axes[1], fd_seis,  "FD (sweep)"),
    ]:
        vmax = np.percentile(np.abs(seis), 99.0)
        ax.imshow(seis.T, aspect="auto", cmap="seismic",
                  vmin=-vmax, vmax=vmax,
                  extent=[rec_x_array[0], rec_x_array[-1], t[-1].item(), 0])
        ax.set_xlabel("x (m)")
        ax.set_title(title)
    axes[0].set_ylabel("t (s)")
    fig.tight_layout()
    gath_path = out / "sem_vs_fd_gathers.png"
    fig.savefig(gath_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {gath_path.resolve()}")


if __name__ == "__main__":
    main(os.environ.get("OUTDIR", "_smoketest"))
