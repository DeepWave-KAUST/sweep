"""3-way seismogram gather: SEM | sweep FD | SPECFEM2D on the same line.

Uses the same physical setup as ``run_compare.py``: 40×30-element mesh,
nstep from geometry.npz, long-nt comparison so boundary-condition
behaviour shows up in the gather.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.sem import AcousticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t: np.ndarray, f0: float, delay: float) -> np.ndarray:
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="_smoketest/specfem_acoustic_flat")
    parser.add_argument(
        "--out_dir",
        default="_smoketest")
    args = parser.parse_args()

    work = Path(args.workdir)
    out = work / "OUTPUT_FILES"
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    geom = np.load(work / "geometry.npz", allow_pickle=True)
    x_max = float(geom["x_max"]); z_max = float(geom["z_max"])
    vp_const = float(geom["vp"])
    nex = int(geom["nex"]); nez = int(geom["nez"])
    n_order = int(geom["n_order"])
    dt = float(geom["dt"]); nstep = int(geom["nstep"])
    f0 = float(geom["f0"]); tshift = float(geom["tshift"])
    src_xz_sem = (float(geom["src_xz"][0]), float(geom["src_xz"][1]))

    # ------------------------------------------------------------------ #
    # Receivers — match the SPECFEM STATIONS file
    # ------------------------------------------------------------------ #
    stations_lines = (work / "DATA" / "STATIONS").read_text().splitlines()
    rec_x_array = []
    rec_z_array = []
    for ln in stations_lines:
        parts = ln.split()
        rec_x_array.append(float(parts[2]))
        # SPECFEM z is up-positive; convert to SEM down-positive
        rec_z_array.append(z_max - float(parts[3]))
    rec_x_array = np.array(rec_x_array)
    rec_z_array = np.array(rec_z_array)
    rec_xz_sem = list(zip(rec_x_array.tolist(), rec_z_array.tolist()))
    print(f"{len(rec_xz_sem)} receivers, x∈[{rec_x_array.min():.0f},"
          f" {rec_x_array.max():.0f}] m, z = {rec_z_array[0]:.0f} m")

    # ------------------------------------------------------------------ #
    # Load SPECFEM seismograms (AA.S*.PRE.semp)
    # ------------------------------------------------------------------ #
    spec_traces = []
    spec_t_file = None
    for j in range(len(rec_xz_sem)):
        d = np.loadtxt(out / f"AA.S{j:03d}.PRE.semp")
        if spec_t_file is None:
            spec_t_file = d[:, 0]
        spec_traces.append(d[:, 1])
    spec_seis = np.stack(spec_traces, axis=0)              # (n_rec, nt_spec)

    # SPECFEM time-shift to align with SEM physical-time axis. Use the
    # source-co-located receiver to read the response-peak time.
    j_src = int(np.argmin(np.abs(rec_x_array - src_xz_sem[0])))
    peak_file_t = spec_t_file[np.argmax(np.abs(spec_seis[j_src]))]
    spec_shift = tshift - peak_file_t
    print(f"SPECFEM peak at file-time {peak_file_t:.4f} s -> shift +"
          f"{spec_shift*1e3:.1f} ms")
    spec_t_phys = spec_t_file + spec_shift

    # ------------------------------------------------------------------ #
    # SEM run (long nt, all receivers)
    # ------------------------------------------------------------------ #
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=n_order, dtype=torch.float64)
    vp_e = vp_const * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                 dtype=torch.float64)
    from sweep.sem import build_cpml
    cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                      pml_thickness_elements=12,
                      c_max=vp_const, f_dominant=f0)
    sem = AcousticSEM2D.build(mesh, vp_e,
                              dirichlet=("top",), cpml=cpml)
    sem_prop = SEMPropagator.build(sem, dt=dt, nt=nstep,
                                   source_xz=[src_xz_sem],
                                   receiver_xz=rec_xz_sem)
    t_sem = np.arange(nstep) * dt
    src_wave = torch.from_numpy(ricker(t_sem, f0, tshift)).to(torch.float64)
    sem_seis = sem_prop.forward(src_wave)[..., 0].T.numpy()
    print(f"SEM gather: {sem_seis.shape}")

    # ------------------------------------------------------------------ #
    # FD run (CPML)
    # ------------------------------------------------------------------ #
    dh_fd = 10.0
    nx_fd = int(round(x_max / dh_fd)) + 1
    nz_fd = int(round(z_max / dh_fd)) + 1
    vp_fd = torch.full((nz_fd, nx_fd), vp_const, dtype=torch.float32)
    ix_s = int(round(src_xz_sem[0] / dh_fd))
    iz_s = int(round(src_xz_sem[1] / dh_fd))
    sources_fd = torch.tensor([[ix_s, iz_s]], dtype=torch.int64)
    receivers_fd = torch.tensor(
        [[[int(round(rx / dh_fd)), int(round(rz / dh_fd))]
          for rx, rz in rec_xz_sem]],
        dtype=torch.int64,
    )
    src_fd = torch.from_numpy(ricker(t_sem, f0, tshift).astype(np.float32))
    fd_eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    fd_prop = PropTorch(fd_eq, shape=(nz_fd, nx_fd),
                       free_surface=True, abcn=30, dh=dh_fd, dt=dt,
                       impl="eager", use_ckpt=False)
    with torch.no_grad():
        rec_fd = fd_prop(src_fd, sources_fd, receivers_fd, models=[vp_fd])
    fd_seis = rec_fd[0, :, :, 0].cpu().numpy().T          # (n_rec, nstep)
    print(f"FD gather: {fd_seis.shape}")

    # ------------------------------------------------------------------ #
    # Align polarity (SPECFEM has opposite sign of SEM/FD)
    # ------------------------------------------------------------------ #
    sem_seis = -sem_seis
    fd_seis  = -fd_seis

    # ------------------------------------------------------------------ #
    # 3-way gather image
    # ------------------------------------------------------------------ #
    def gather_panel(ax, t_axis, seis, title, vmax=0.15,
                     per_trace=True):
        s = seis.copy()
        if per_trace:
            peaks = np.maximum(np.abs(s).max(axis=1, keepdims=True), 1e-30)
            s = s / peaks
        else:
            peak_global = max(np.max(np.abs(s)), 1e-30)
            s = s / peak_global
        im = ax.imshow(s, aspect="auto", cmap="seismic",
                       vmin=-vmax, vmax=vmax,
                       extent=[t_axis[0], t_axis[-1],
                               rec_x_array[-1], rec_x_array[0]])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("receiver x (m)")
        ax.tick_params(labelsize=8)
        return im

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    gather_panel(axes[0], t_sem, sem_seis,    "SEM  (CPML 12 elem ≈ 300 m)")
    gather_panel(axes[1], t_sem, fd_seis,     "sweep FD  (CPML 30 cells = 300 m)")
    gather_panel(axes[2], spec_t_phys, spec_seis, "SPECFEM2D  (PML 12 elem = 300 m)")
    for ax in axes:
        ax.set_xlim(0, t_sem[-1])
    fig.suptitle("clipped at ±15% peak so reflections are visible", fontsize=9)
    fig.tight_layout()
    g_path = fig_dir / "gather_3way_longnt.png"
    fig.savefig(g_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_path.resolve()}")

    # Windowed late-time gather. Normalize by EACH solver's global peak
    # (NOT per-trace), so the absolute leak amplitude is shown honestly.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    gather_panel(axes[0], t_sem, sem_seis, "SEM CPML — t > 0.5 s (global norm)",
                 vmax=0.01, per_trace=False)
    gather_panel(axes[1], t_sem, fd_seis,  "FD CPML — t > 0.5 s (global norm)",
                 vmax=0.01, per_trace=False)
    gather_panel(axes[2], spec_t_phys, spec_seis,
                 "SPECFEM PML — t > 0.5 s (global norm)",
                 vmax=0.01, per_trace=False)
    for ax in axes:
        ax.set_xlim(0.5, t_sem[-1])
    fig.tight_layout()
    g_path2 = fig_dir / "gather_3way_late_only.png"
    fig.savefig(g_path2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_path2.resolve()}")

    # ------------------------------------------------------------------ #
    # Trace overlay at 3 selected receivers
    # ------------------------------------------------------------------ #
    pick_x = [200.0, 500.0, 900.0]
    pick_idx = [int(np.argmin(np.abs(rec_x_array - x))) for x in pick_x]

    def norm(a):
        return a / max(np.max(np.abs(a)), 1e-30)

    fig, axes = plt.subplots(len(pick_idx), 1, figsize=(12, 2.6 * len(pick_idx)),
                             sharex=True)
    for ax, j in zip(axes, pick_idx):
        rx = rec_x_array[j]
        offset = abs(rx - src_xz_sem[0])
        ax.plot(t_sem,   norm(sem_seis[j]),    "C0",  lw=1.4,
                label="SEM CPML 12-elem")
        ax.plot(t_sem,   norm(fd_seis[j]),     "C1--", lw=1.0,
                label="FD CPML 30-cell")
        ax.plot(spec_t_phys, norm(spec_seis[j]), "C3",  lw=0.8,
                label="SPECFEM PML 12-elem")
        # Geometric direct-arrival marker
        t_direct = tshift + offset / vp_const
        ax.axvline(t_direct, ls=":", lw=0.6, color="gray",
                   label=f"direct {t_direct*1e3:.0f} ms")
        ax.set_xlim(0, t_sem[-1])
        ax.set_title(f"x={rx:.0f} m  (offset {offset:.0f} m)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("physical time (s)")
    fig.tight_layout()
    t_path = fig_dir / "trace_3way_longnt.png"
    fig.savefig(t_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {t_path.resolve()}")


if __name__ == "__main__":
    main()
