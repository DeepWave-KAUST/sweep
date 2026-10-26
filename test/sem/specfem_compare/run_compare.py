"""Run SEM + sweep FD on the same setup as the SPECFEM2D acoustic case and
plot three-way wavefield snapshot comparison panels.

Prerequisites:
  1. ``build_inputs.py`` has been run (DATA/ is populated).
  2. SPECFEM2D has been run from the workdir:
       cd <workdir> && ./xmeshfem2D && ./xspecfem2D
     which produces ``OUTPUT_FILES/wavefield00000XXX_01.bin`` and
     ``OUTPUT_FILES/wavefield_grid_for_dumps.bin``.

The script then:
  - rebuilds the SEM PoC at exactly the same physical parameters
  - rebuilds sweep's FD Acoustic at the same geometry (dh chosen to give a
    similarly resolved grid)
  - aligns snapshots by **physical time** (SPECFEM starts the time series at
    -t0, while SEM and sweep-FD both start at t = 0)
  - renders side-by-side panels.
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
    return (1.0 - 2.0 * a) * np.exp(-a)


# ------------------------------------------------------------------ #
# SPECFEM2D output readers
# ------------------------------------------------------------------ #
def _read_f32(path: Path) -> np.ndarray:
    """Read a Fortran-unformatted single-prec binary as a flat float32 array.

    SPECFEM2D writes these with a 4-byte record marker on each side (Fortran
    sequential access). We strip those.
    """
    raw = np.fromfile(path, dtype=np.float32)
    # Fortran record markers are int32 byte counts; first and last are == length.
    if raw.size >= 2 and int(raw.view(np.int32)[0]) == (raw.size - 2) * 4:
        return raw[1:-1].copy()
    return raw


def load_specfem_snapshots(out_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Returns (coords[N,2], snaps[T,N], step_indices).
    """
    grid_path = out_dir / "wavefield_grid_for_dumps.bin"
    grid_flat = _read_f32(grid_path)
    # Layout in SPECFEM2D: pairs (x, z) for each dumped GLL point.
    coords = grid_flat.reshape(-1, 2)
    n = coords.shape[0]

    snap_files = sorted(out_dir.glob("wavefield*_01.bin"))
    step_indices = []
    snaps = []
    for p in snap_files:
        # filename: wavefield00000XXXX_01.bin
        step = int(p.name[len("wavefield"):len("wavefield") + 7])
        vals = _read_f32(p)
        if vals.size != n:
            print(f"warn: skipping {p.name}: size {vals.size} != grid {n}")
            continue
        step_indices.append(step)
        snaps.append(vals)
    if not snaps:
        raise FileNotFoundError(f"no usable wavefield dumps in {out_dir}")
    return coords, np.stack(snaps, axis=0), step_indices


def read_specfem_t0(seismogram_path: Path) -> float:
    """Read first column of a .semp file to recover the start time."""
    with open(seismogram_path) as f:
        first = f.readline().split()
    return float(first[0])


# ------------------------------------------------------------------ #
# Plot helpers
# ------------------------------------------------------------------ #
def percentile_sym(a: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(np.abs(a), q))


def render_panel_grid(
    pannels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    times_s: list[float],
    x_max: float, z_max: float, src_xz: tuple[float, float],
    out_path: Path,
    *,
    vmax_global: float | None = None,
):
    """For each time snapshot, render one row of (SEM | FD | SPECFEM) panels.

    All ``snaps`` arrays are expected to already be on a regular fine grid
    of the same physical extent — non-uniform GLL data should be
    pre-interpolated via :func:`gll_regular_to_image`.
    """
    n_times = len(times_s)
    n_cols = len(pannels)
    fig, axes = plt.subplots(n_times, n_cols,
                             figsize=(4.5 * n_cols, 2.8 * n_times),
                             squeeze=False)
    if vmax_global is None:
        vmax_global = max(percentile_sym(p[3], 99.0) for p in pannels)
    for r, t in enumerate(times_s):
        for c, (label, X, Z, snaps) in enumerate(pannels):
            ax = axes[r, c]
            im = ax.imshow(snaps[r],
                           extent=[X.min(), X.max(), Z.max(), Z.min()],
                           cmap="seismic", vmin=-vmax_global, vmax=vmax_global,
                           aspect="equal")
            ax.plot(src_xz[0], src_xz[1], "ko", ms=4, mew=0)
            if r == 0:
                ax.set_title(label, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"t={t:.3f}s", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def gll_array_to_image(
    values: np.ndarray, x: np.ndarray, z: np.ndarray,
    *, nx_img: int = 801, nz_img: int = 601,
    x_max: float, z_max: float,
):
    """Linearly interpolate scattered GLL samples onto a regular image grid
    for visual comparison."""
    from scipy.interpolate import griddata
    xi = np.linspace(0, x_max, nx_img)
    zi = np.linspace(0, z_max, nz_img)
    XI, ZI = np.meshgrid(xi, zi)
    img = griddata(
        (x, z), values, (XI, ZI),
        method="linear", fill_value=0.0,
    )
    return XI, ZI, img


def gll_regular_to_image(
    values_yx: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray,
    *, nx_img: int = 801, nz_img: int = 601, x_max: float, z_max: float,
):
    """Interpolate a tensor-product non-uniform GLL grid (as produced by
    the SEM mesh) to a fine regular grid for crisp rendering."""
    from scipy.interpolate import RegularGridInterpolator
    xi = np.linspace(0, x_max, nx_img)
    zi = np.linspace(0, z_max, nz_img)
    XI, ZI = np.meshgrid(xi, zi)
    f = RegularGridInterpolator((z_axis, x_axis), values_yx,
                                bounds_error=False, fill_value=0.0)
    return f((ZI, XI))


# ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------
    # 1. Load SPECFEM snapshots
    # ------------------------------------------------------------------
    coords_spec, snaps_spec, step_indices = load_specfem_snapshots(out)
    t0_spec = -read_specfem_t0(out / "AA.S000.PRE.semp")
    print(f"SPECFEM start time = -{t0_spec:.4f} s ; snapshot steps: "
          f"{step_indices[:3]} ... {step_indices[-3:]}")
    # SPECFEM's seismotype=4 acoustic output looks like a Ricker-shaped
    # (sign-flipped) pulse at the source — empirically it's ≈ -χ or similar
    # zeroth-order map of the potential, NOT the full -ρ ∂²χ/∂t². So no
    # 2nd-derivative post-process is needed; align by matching the
    # response-peak time between SEM (peak at physical time = tshift) and
    # SPECFEM (peak at file-time `peak_file_t`).
    src_x = src_xz_sem[0]
    stations = (work / "DATA" / "STATIONS").read_text().splitlines()
    j_at_src = None
    for k, ln in enumerate(stations):
        parts = ln.split()
        if abs(float(parts[2]) - src_x) < 1.0:
            j_at_src = k
            break
    if j_at_src is None:
        raise RuntimeError("no SPECFEM station co-located with source")
    seis = np.loadtxt(out / f"AA.S{j_at_src:03d}.PRE.semp")
    peak_file_t = seis[np.argmax(np.abs(seis[:, 1])), 0]
    spec_time_shift = tshift - peak_file_t
    print(f"SPECFEM peak observed at file-time {peak_file_t:.4f} s; "
          f"applying shift +{spec_time_shift*1e3:.1f} ms to align with SEM")
    phys_times_spec = np.array(
        [(k - 1) * dt - t0_spec + spec_time_shift for k in step_indices]
    )

    # SPECFEM coords have z increasing UP; convert to SEM down-positive.
    coords_spec_xz_sem = np.stack(
        [coords_spec[:, 0], z_max - coords_spec[:, 1]], axis=-1
    )

    # ------------------------------------------------------------------
    # 2. Pick comparison times spanning multiple boundary round-trips so we
    #    can SEE how each backend's BC accumulates reflections. With
    #    domain 1000×750 m and c=2000 m/s, a wave from centre reaches the
    #    nearest boundary at ~0.18 s and returns at ~0.36 s; long-time
    #    samples reveal Stacey vs CPML leakage.
    target_t = np.array([0.10, 0.20, 0.40, 0.70, 1.00, 1.40, 1.80, 2.20])
    nearest = []
    for tt in target_t:
        j = int(np.argmin(np.abs(phys_times_spec - tt)))
        nearest.append(j)
    spec_idx = nearest
    spec_t   = phys_times_spec[spec_idx]
    print(f"comparing at SPECFEM physical times: "
          f"{[f'{t:.3f}' for t in spec_t]}")

    # ------------------------------------------------------------------
    # 3. SEM run
    # ------------------------------------------------------------------
    n_sem_steps = int(np.ceil(spec_t.max() / dt)) + 10
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=n_order, dtype=torch.float64)
    vp_e = vp_const * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                 dtype=torch.float64)
    # Match the 300 m physical PML thickness of sweep FD (abcn=30 cells × 10 m)
    # and SPECFEM (NELEM_PML_THICKNESS=12 × 25 m).
    from sweep.sem import build_cpml
    cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                      pml_thickness_elements=12,
                      c_max=vp_const, f_dominant=f0)
    sem = AcousticSEM2D.build(mesh, vp_e,
                              dirichlet=("top",), cpml=cpml)
    rec_xz_dummy = [(50.0, 100.0)]
    prop = SEMPropagator.build(sem, dt=dt, nt=n_sem_steps,
                               source_xz=[src_xz_sem], receiver_xz=rec_xz_dummy)
    t_sem = np.arange(n_sem_steps) * dt
    sem_src = torch.from_numpy(ricker(t_sem, f0=f0, delay=tshift)).to(torch.float64)
    # Sparse snapshots — long-nt runs would otherwise blow up memory
    # (3000 × 19481 × 8 B ≈ 470 MB).
    sem_record_every = 50
    _, sem_snaps_t = prop.forward(sem_src, record_field_every=sem_record_every)
    sem_snaps = sem_snaps_t[:, 0].numpy()
    sem_snap_steps = np.arange(0, sem_nt, sem_record_every)

    nx_g = mesh.nex * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1
    sem_snap_grid = sem_snaps.reshape(-1, nz_g, nx_g)

    # Pick the SEM snapshot closest to each target physical time.
    sem_pick_step = [int(round(t / dt)) for t in spec_t]
    sem_pick_idx_array = [int(np.argmin(np.abs(sem_snap_steps - s)))
                          for s in sem_pick_step]
    sem_pick = sem_snap_grid[sem_pick_idx_array]
    sem_x = np.linspace(0, x_max, nx_g)
    sem_z = np.linspace(0, z_max, nz_g)
    # Variable kept for the FD section below so it knows which physical
    # step each panel corresponds to.
    sem_pick_idx = sem_pick_step

    # ------------------------------------------------------------------
    # 4. FD run (sweep.equations.Acoustic) on a matching but finer grid
    # ------------------------------------------------------------------
    dh_fd = 10.0
    nx_fd = int(round(x_max / dh_fd)) + 1
    nz_fd = int(round(z_max / dh_fd)) + 1
    vp_fd = torch.full((nz_fd, nx_fd), vp_const, dtype=torch.float32)
    ix_s = int(round(src_xz_sem[0] / dh_fd))
    iz_s = int(round(src_xz_sem[1] / dh_fd))
    sources_fd = torch.tensor([[ix_s, iz_s]], dtype=torch.int64)
    receivers_fd = torch.tensor([[[0, 0]]], dtype=torch.int64)  # placeholder
    src_wave_fd = torch.from_numpy(ricker(t_sem, f0, tshift).astype(np.float32))

    fd_eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    fd_prop = PropTorch(
        fd_eq, shape=(nz_fd, nx_fd),
        free_surface=True, abcn=30, dh=dh_fd, dt=dt,
        impl="eager", use_ckpt=False,
    )
    # Need 3 consecutive timesteps at each target so we can compute the
    # centered 2nd-time derivative (= SPECFEM-style pressure output).
    snap_indices_fd = sorted({i + d for i in sem_pick_idx for d in (-1, 0, 1)
                              if 0 < i + d < n_sem_steps - 1})
    with torch.no_grad():
        _, fd_snaps_pad = fd_prop(
            src_wave_fd, sources_fd, receivers_fd, models=[vp_fd],
            return_wavefield=True, snapshot_times=snap_indices_fd,
        )
    # shape: (n_snap, n_wavefield, B=1, C=1, nz_pad, nx_pad)
    fd_snaps_pad = fd_snaps_pad[:, 0, 0, 0].cpu().numpy()
    abcn = 30
    fd_snaps_phys_all = fd_snaps_pad[:, :nz_fd, abcn:abcn + nx_fd]
    fd_x = np.arange(nx_fd) * dh_fd
    fd_z = np.arange(nz_fd) * dh_fd

    # ------------------------------------------------------------------
    # 5. Map SPECFEM scattered snapshots to a regular grid (for imshow).
    # ------------------------------------------------------------------
    spec_snaps_img = []
    for j in spec_idx:
        XI, ZI, img = gll_array_to_image(
            snaps_spec[j], coords_spec_xz_sem[:, 0], coords_spec_xz_sem[:, 1],
            nx_img=nx_g, nz_img=nz_g, x_max=x_max, z_max=z_max,
        )
        spec_snaps_img.append(img)
    spec_x = np.linspace(0, x_max, nx_g)
    spec_z = np.linspace(0, z_max, nz_g)

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    fd_pick = []
    for idx in sem_pick_idx:
        fd_pick.append(fd_snaps_phys_all[snap_indices_fd.index(idx)])
    fd_pick = np.stack(fd_pick, axis=0)

    sign = -1.0
    sem_aligned = sign * sem_pick
    fd_aligned  = sign * fd_pick

    # Interpolate SEM (non-uniform GLL axes) and FD (uniform but coarse) to
    # the SAME fine 801×601 regular grid as SPECFEM, so all three render at
    # equal resolution.
    nx_img, nz_img = 801, 601
    fine_x = np.linspace(0, x_max, nx_img)
    fine_z = np.linspace(0, z_max, nz_img)
    sem_fine = np.stack([
        gll_regular_to_image(sem_aligned[k], sem_x, sem_z,
                             nx_img=nx_img, nz_img=nz_img,
                             x_max=x_max, z_max=z_max)
        for k in range(sem_aligned.shape[0])
    ])
    fd_fine = np.stack([
        gll_regular_to_image(fd_aligned[k], fd_x, fd_z,
                             nx_img=nx_img, nz_img=nz_img,
                             x_max=x_max, z_max=z_max)
        for k in range(fd_aligned.shape[0])
    ])
    spec_fine = np.stack(spec_snaps_img, axis=0)  # already 601×801

    def normalize(arr):
        out = arr.copy().astype(np.float32)
        for k in range(out.shape[0]):
            m = np.percentile(np.abs(out[k]), 99.0)
            if m > 0:
                out[k] = out[k] / m
        return out

    panels = [
        ("SEM  (−u)",            fine_x, fine_z, normalize(sem_fine)),
        ("sweep FD (−h1)",       fine_x, fine_z, normalize(fd_fine)),
        ("SPECFEM2D (pressure)", fine_x, fine_z, normalize(spec_fine)),
    ]
    out_path = fig_dir / "sem_vs_fd_vs_specfem_snapshots.png"
    render_panel_grid(panels, list(spec_t), x_max, z_max, src_xz_sem, out_path,
                      vmax_global=1.0)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
