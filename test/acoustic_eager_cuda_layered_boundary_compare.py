import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_repo_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


REPO_ROOT = find_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))
from bootstrap import configure_example_imports

IMPORT_MODE = configure_example_imports(EXAMPLES_DIR)

from sweep import is_torch_binding_available
from sweep.equations import Acoustic
from sweep.propagator.options import CUDAOptions, EagerOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


OUTPUT_DIR = REPO_ROOT / "test" / "test_outputs" / "acoustic_eager_cuda_layered_boundary_compare"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare eager and CUDA acoustic forward wavefields/records on a layered model."
    )
    parser.add_argument("--import-mode", choices=("env", "source"), default=IMPORT_MODE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nz", type=int, default=140)
    parser.add_argument("--nx", type=int, default=220)
    parser.add_argument("--dz", type=float, default=5.0)
    parser.add_argument("--dx", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=2200)
    parser.add_argument("--fm", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--spatial-order", type=int, default=8)
    parser.add_argument("--src-z", type=int, default=2)
    parser.add_argument("--rec-z", type=int, default=2)
    parser.add_argument("--nrec", type=int, default=96)
    parser.add_argument("--snapshot-times", type=int, nargs="*", default=[700, 1100, 1600, 2000])
    parser.add_argument("--cuda-memory", choices=("full", "bs", "ckpt"), default="full")
    return parser


def require_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if not is_torch_binding_available():
            raise RuntimeError("sweep._C is not available or the CUDA binding cannot be imported.")
        return torch.device("cuda")
    return torch.device(name)


def percentile_range(data, low=2.0, high=98.0):
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        peak = max(float(np.max(np.abs(finite))), 1e-6)
        return -peak, peak
    return float(lo), float(hi)


def symmetric_percentile(data, p=98.0):
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    peak = np.percentile(np.abs(finite), p)
    return max(float(peak), 1e-6)


def make_layered_model(shape):
    nz, nx = shape
    vp = np.full((nz, nx), 1800.0, dtype=np.float32)
    z1 = int(round(nz * 0.25))
    z2 = int(round(nz * 0.50))
    z3 = int(round(nz * 0.72))
    vp[z1:, :] = 2200.0
    vp[z2:, :] = 2600.0
    vp[z3:, :] = 3100.0
    return vp


def make_geometry(shape, src_z, rec_z, nrec):
    nz, nx = shape
    src = np.array([[nx // 2, src_z]], dtype=np.int64)
    rec_x = np.linspace(8, nx - 9, nrec, dtype=np.int64)
    rec = np.stack([rec_x, np.full(rec_x.shape[0], rec_z, dtype=np.int64)], axis=-1)[None, ...]
    return src, rec


def make_wavelet(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt
    if delay is None:
        delay = 1.2 / float(fm)
    return ricker(t - delay, f=float(fm)).astype(np.float32)


def build_cuda_options(mode):
    if mode == "full":
        return CUDAOptions(memory=None)
    if mode == "bs":
        return CUDAOptions(memory={"strategy": "boundary", "boundary": {"storage": "gpu", "transfer_interval": 1, "pinned_memory": False}})
    if mode == "ckpt":
        return CUDAOptions(memory={"strategy": "ckpt", "ckpt": {"mode": "chunk", "chunks": 64}})
    raise ValueError(f"Unsupported CUDA memory mode '{mode}'.")


def build_solver(backend, shape, dh, dt, device, args):
    equation = Acoustic(spatial_order=args.spatial_order, device=device, backend="torch")
    common = dict(
        shape=shape,
        dev=device,
        dh=dh,
        dt=dt,
        nt=args.nt,
        abcn=args.abcn,
        source_type=["h1"],
        receiver_type=["h1"],
        pml_type="cpmlr",
        free_surface=False,
        use_ckpt=False,
    )
    if backend == "eager":
        return PropTorch(equation, backend="eager", eager_options=EagerOptions(use_compile=False), **common)
    return PropTorch(equation, backend="cuda", cuda_options=build_cuda_options(args.cuda_memory), **common)


def normalize_record_layout(record, nt, nrec):
    if isinstance(record, torch.Tensor):
        record = record.detach().cpu().numpy()
    arr = np.asarray(record)
    if arr.ndim == 4:
        if arr.shape[-1] != 1:
            raise ValueError(f"Unsupported record tensor shape {arr.shape}")
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D record tensor, got shape {arr.shape}")
    if arr.shape == (1, nt, nrec):
        return arr[0]
    if arr.shape == (1, nrec, nt):
        return arr[0].T
    raise ValueError(f"Unsupported record layout {arr.shape}")


def extract_h1_snapshots(wavefields):
    if isinstance(wavefields, torch.Tensor):
        data = wavefields.detach().cpu().numpy()
    else:
        data = np.asarray(wavefields)
    if data.ndim == 5:
        return data[:, 0, 0]
    if data.ndim == 6:
        return data[:, 0, 0, 0]
    raise ValueError(f"Unsupported wavefield snapshot shape {data.shape}")


def crop_eager_field(field, abcn):
    arr = np.asarray(field, dtype=np.float32)
    z0 = abcn
    z1 = arr.shape[-2] - abcn
    x0 = abcn
    x1 = arr.shape[-1] - abcn
    return arr[z0:z1, x0:x1]


def crop_cuda_field(field, abcn, spatial_order):
    arr = np.asarray(field, dtype=np.float32)
    m = spatial_order // 2
    z0 = abcn + m
    z1 = arr.shape[-2] - abcn - m
    x0 = abcn + m
    x1 = arr.shape[-1] - abcn - m
    return arr[z0:z1, x0:x1]


def run_backend(name, solver, wavelet, sources, receivers, model, snapshot_times, device, args):
    vp = torch.tensor(model, device=device, dtype=torch.float32)
    if name == "eager":
        with torch.no_grad():
            record, wavefields = solver(
                wavelet,
                sources,
                receivers,
                models=[vp],
                return_wavefield=True,
                snapshot_times=snapshot_times,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        snapshots = extract_h1_snapshots(wavefields)
        snapshots = np.stack(
            [crop_eager_field(s, solver.abcn) for s in snapshots],
            axis=0,
        )
        return {
            "record": normalize_record_layout(record, wavelet.shape[-1], receivers.shape[1]),
            "snapshots": snapshots,
        }

    with torch.no_grad():
        record = solver(
            wavelet,
            sources,
            receivers,
            models=[vp],
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    h1_idx = solver.wavefield_names.index("h1")
    snapshots = []
    for t in snapshot_times:
        sub_wavelet = wavelet[..., : t + 1]
        sub_args = argparse.Namespace(**vars(args))
        sub_args.nt = t + 1
        sub_solver = build_solver("cuda", model.shape, tuple(solver._grid_spacing), solver._dt, device, sub_args)
        vp_snap = torch.tensor(model, device=device, dtype=torch.float32, requires_grad=True)
        _ = sub_solver(
            sub_wavelet,
            sources,
            receivers,
            models=[vp_snap],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        field = sub_solver.forward_wavefields[h1_idx][0, 0].detach().cpu().numpy()
        snapshots.append(crop_cuda_field(field, sub_solver.abcn, sub_solver.equation.so))

    return {
        "record": normalize_record_layout(record, wavelet.shape[-1], receivers.shape[1]),
        "snapshots": np.stack(snapshots, axis=0),
    }


def summarize_boundary_energy(snapshot, edge_width=20):
    field = np.asarray(snapshot, dtype=np.float32)
    nz, nx = field.shape
    edge = np.zeros_like(field, dtype=bool)
    edge[:edge_width, :] = True
    edge[-edge_width:, :] = True
    edge[:, :edge_width] = True
    edge[:, -edge_width:] = True
    interior = ~edge
    edge_max = float(np.max(np.abs(field[edge])))
    interior_max = max(float(np.max(np.abs(field[interior]))), 1e-6)
    return edge_max / interior_max


def save_snapshot_figure(eager_snapshots, cuda_snapshots, snapshot_times, dh, out_path):
    panels = []
    titles = []
    for idx, t in enumerate(snapshot_times):
        diff = cuda_snapshots[idx] - eager_snapshots[idx]
        panels.extend([eager_snapshots[idx], cuda_snapshots[idx], diff])
        titles.extend([f"eager | t={t}", f"cuda | t={t}", f"diff | t={t}"])

    ncols = 3
    nrows = len(snapshot_times)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    extent = (0.0, eager_snapshots.shape[-1] * dh[1], eager_snapshots.shape[-2] * dh[0], 0.0)
    for i in range(nrows):
        row = [eager_snapshots[i], cuda_snapshots[i], cuda_snapshots[i] - eager_snapshots[i]]
        for j in range(ncols):
            ax = axes[i, j]
            panel = row[j]
            if j < 2:
                vmin, vmax = percentile_range(panel, 2, 98)
            else:
                amp = symmetric_percentile(panel, 98)
                vmin, vmax = -amp, amp
            im = ax.imshow(panel, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(titles[i * ncols + j])
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_record_figure(eager_record, cuda_record, out_path):
    diff = cuda_record - eager_record
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    panels = [eager_record, cuda_record, diff]
    titles = ["eager total record", "cuda total record", "difference"]
    for j, panel in enumerate(panels):
        ax = axes[0, j]
        if j < 2:
            vmin, vmax = percentile_range(panel, 2, 98)
        else:
            amp = symmetric_percentile(panel, 98)
            vmin, vmax = -amp, amp
        im = ax.imshow(panel, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(titles[j])
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Time Sample")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_figure(model, sources, receivers, dh, out_path):
    extent = (0.0, model.shape[1] * dh[1], model.shape[0] * dh[0], 0.0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(model, cmap="viridis", aspect="auto", extent=extent)
    ax.scatter(sources[:, 0] * dh[1], sources[:, 1] * dh[0], s=110, c="gold", marker="*", edgecolors="black", linewidths=0.6)
    ax.scatter(receivers[0, :, 0] * dh[1], receivers[0, :, 1] * dh[0], s=12, c="white", marker="v", edgecolors="black", linewidths=0.3)
    ax.set_title("Layered model and acquisition geometry")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Vp (m/s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    device = require_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shape = (args.nz, args.nx)
    dh = (float(args.dz), float(args.dx))
    model = make_layered_model(shape)
    sources, receivers = make_geometry(shape, args.src_z, args.rec_z, args.nrec)
    wavelet = make_wavelet(args.nt, args.dt, args.fm, args.delay)

    eager_solver = build_solver("eager", shape, dh, args.dt, device, args)
    cuda_solver = build_solver("cuda", shape, dh, args.dt, device, args)

    eager = run_backend("eager", eager_solver, wavelet, sources, receivers, model, args.snapshot_times, device, args)
    cuda = run_backend("cuda", cuda_solver, wavelet, sources, receivers, model, args.snapshot_times, device, args)

    save_model_figure(model, sources, receivers, dh, OUTPUT_DIR / "layered_model_geometry.png")
    save_snapshot_figure(eager["snapshots"], cuda["snapshots"], args.snapshot_times, dh, OUTPUT_DIR / "wavefield_snapshots.png")
    save_record_figure(eager["record"], cuda["record"], OUTPUT_DIR / "records.png")

    summary_path = OUTPUT_DIR / "summary.txt"
    rel_record = np.linalg.norm(cuda["record"] - eager["record"]) / max(np.linalg.norm(eager["record"]), 1e-12)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"import_mode={args.import_mode}\n")
        f.write(f"shape={shape}\n")
        f.write(f"dh=(dz={dh[0]}, dx={dh[1]})\n")
        f.write(f"dt={args.dt}\n")
        f.write(f"nt={args.nt}\n")
        f.write(f"source=top_center\n")
        f.write(f"source_grid={sources.tolist()}\n")
        f.write(f"receiver_z={args.rec_z}\n")
        f.write(f"cuda_memory={args.cuda_memory}\n")
        f.write(f"record_rel_l2={rel_record:.6e}\n")
        for idx, t in enumerate(args.snapshot_times):
            eager_ratio = summarize_boundary_energy(eager["snapshots"][idx], edge_width=args.abcn)
            cuda_ratio = summarize_boundary_energy(cuda["snapshots"][idx], edge_width=args.abcn)
            f.write(
                f"t={t}: eager_edge_ratio={eager_ratio:.6f}, "
                f"cuda_edge_ratio={cuda_ratio:.6f}\n"
            )

    print(f"python={sys.executable}")
    print(f"import_mode={args.import_mode}")
    print(f"outputs={OUTPUT_DIR}")
    print(f"record_rel_l2={rel_record:.6e}")


if __name__ == "__main__":
    main()
