"""Overlay traces at / near the source from SEM, sweep FD, and SPECFEM2D
to see what the actually-injected wavelet looks like in each backend.

A receiver at ~50 m offset is used (not the singular zero-offset spot
where the 2D Green function has a 1/sqrt singularity that distorts the
recorded shape).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.sem import AcousticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main():
    work = Path("_smoketest/specfem_acoustic_flat")
    out_dir = Path("_smoketest")

    geom = np.load(work / "geometry.npz", allow_pickle=True)
    x_max = float(geom["x_max"]); z_max = float(geom["z_max"])
    vp_const = float(geom["vp"])
    dt = float(geom["dt"]); nstep = int(geom["nstep"])
    f0 = float(geom["f0"]); tshift = float(geom["tshift"])
    src_xz = (float(geom["src_xz"][0]), float(geom["src_xz"][1]))

    # ------------------------------------------------------------------
    # SPECFEM trace at receivers spanning near→far offset, so we can see
    # whether mismatch is from source spatial distribution (near-field) or
    # from source time function (which should now match via external file).
    # ------------------------------------------------------------------
    stations = (work / "DATA" / "STATIONS").read_text().splitlines()
    rec_x_probe = [500.0, 600.0, 800.0]   # offsets 0, 100, 300 m
    spec_traces = []
    for rx in rec_x_probe:
        j = None
        for k, ln in enumerate(stations):
            if abs(float(ln.split()[2]) - rx) < 1.0:
                j = k; break
        if j is None:
            raise RuntimeError(f"no station at x={rx}")
        d = np.loadtxt(work / "OUTPUT_FILES" / f"AA.S{j:03d}.PRE.semp")
        spec_traces.append((rx, d[:, 0], d[:, 1]))
    # Use the source-co-located trace to measure SPECFEM's time-shift.
    _, t_spec_file_at_src, p_spec_at_src = spec_traces[0]
    peak_file_t = t_spec_file_at_src[np.argmax(np.abs(p_spec_at_src))]
    spec_shift = tshift - peak_file_t
    print(f"SPECFEM peak file-time = {peak_file_t:.4f}, shift={spec_shift*1e3:.1f} ms")

    # ------------------------------------------------------------------
    # SEM run with all probe receivers at z=100
    # ------------------------------------------------------------------
    nex, nez = 20, 15
    n_order = 4
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=n_order, dtype=torch.float64)
    vp_e = vp_const * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                 dtype=torch.float64)
    sem = AcousticSEM2D.build(mesh, vp_e)
    rec_xz_sem = [(rx, 100.0) for rx in rec_x_probe]
    sem_prop = SEMPropagator.build(sem, dt=dt, nt=nstep,
                                   source_xz=[src_xz], receiver_xz=rec_xz_sem)
    t_sem = np.arange(nstep) * dt
    src_wave = torch.from_numpy(ricker(t_sem, f0, tshift)).to(torch.float64)
    sem_seis_all = sem_prop.forward(src_wave)[..., 0].T.numpy()  # (n_rec, nt)

    # ------------------------------------------------------------------
    # sweep FD run with all probe receivers
    # ------------------------------------------------------------------
    dh_fd = 10.0
    nx_fd = int(round(x_max / dh_fd)) + 1
    nz_fd = int(round(z_max / dh_fd)) + 1
    vp_fd = torch.full((nz_fd, nx_fd), vp_const, dtype=torch.float32)
    ix_s = int(round(src_xz[0] / dh_fd))
    iz_s = int(round(src_xz[1] / dh_fd))
    sources_fd = torch.tensor([[ix_s, iz_s]], dtype=torch.int64)
    receivers_fd = torch.tensor(
        [[[int(rx / dh_fd), int(100 / dh_fd)] for rx in rec_x_probe]],
        dtype=torch.int64,
    )
    src_fd = torch.from_numpy(ricker(t_sem, f0, tshift).astype(np.float32))
    fd_eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    fd_prop = PropTorch(fd_eq, shape=(nz_fd, nx_fd),
                       free_surface=True, abcn=30, dh=dh_fd, dt=dt,
                       impl="eager", use_ckpt=False)
    with torch.no_grad():
        rec_fd = fd_prop(src_fd, sources_fd, receivers_fd, models=[vp_fd])
    fd_seis_all = rec_fd[0, :, :, 0].cpu().numpy().T   # (n_rec, nt)

    # ------------------------------------------------------------------
    # Plot one row per probe offset; show SEM, FD, and SPECFEM aligned.
    # ------------------------------------------------------------------
    def norm(a):
        return a / max(np.max(np.abs(a)), 1e-30)

    fig, axes = plt.subplots(len(rec_x_probe), 1, figsize=(10, 2.4 * len(rec_x_probe)),
                             sharex=True)
    for ax, j, (rx, t_file, p_s) in zip(axes, range(len(rec_x_probe)), spec_traces):
        t_phys = t_file + spec_shift
        offset = abs(rx - src_xz[0])
        ax.plot(t_sem, norm(-sem_seis_all[j]), "C0", lw=1.5, label=f"SEM × (−1)")
        ax.plot(t_sem, norm(-fd_seis_all[j]), "C1--", lw=1.2, label="sweep FD × (−1)")
        ax.plot(t_phys, norm(p_s), "C3", lw=1.0, label="SPECFEM")
        # Expected geometric direct arrival from a point source:
        t_arr = tshift + offset / vp_const
        ax.axvline(t_arr, color="gray", ls=":", lw=0.6, alpha=0.6,
                   label=f"geometric arrival {t_arr*1e3:.1f} ms")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f"x={rx:.0f} m  (offset {offset:.0f} m)")
        ax.set_xlim(0.0, 0.3)
    axes[-1].set_xlabel("physical time (s)")
    fig.tight_layout()
    out_path = out_dir / "diag_source_shape_traces.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
