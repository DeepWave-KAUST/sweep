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

from sweep.equations import Acoustic3D, AcousticLSRTM3D
from sweep.propagator.options import BoundaryOptions, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


OUTPUT_DIR = REPO_ROOT / "test" / "test_outputs" / "lsrtm3d_reflectivity_grad_smoke"


def build_parser():
    parser = argparse.ArgumentParser(description="Compare 3D acoustic LSRTM toy-model gradients between eager and CUDA.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dh", type=float, default=20.0)
    parser.add_argument("--fm", type=float, default=8.0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--abcn", type=int, default=10)
    parser.add_argument("--cuda-memory", choices=("full", "bs", "chunk", "recursive"), default="bs")
    parser.add_argument("--checkpoint-chunks", type=int, default=16)
    parser.add_argument("--checkpoint-count", type=int, default=6)
    return parser


def build_geometry(shape):
    nz, ny, nx = shape
    sources = np.array([[nx // 2, ny // 2, 2]], dtype=np.int32)
    rec_x = np.arange(4, nx - 4, 4, dtype=np.int32)
    rec_y = np.full_like(rec_x, ny // 2)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_y, rec_z], axis=-1)[None, ...]
    return sources, receivers


def build_cuda_options(args):
    if args.cuda_memory == "full":
        return CUDAOptions(memory=None)
    if args.cuda_memory == "bs":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(storage="gpu"),
            )
        )
    if args.cuda_memory == "chunk":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="chunk", chunks=args.checkpoint_chunks),
            )
        )
    return CUDAOptions(
        memory=MemoryOptions(
            strategy="ckpt",
            ckpt=CkptOptions(mode="recursive", count=args.checkpoint_count),
        )
    )


def build_common_solver(equation, shape, backend, device, args, receiver_type):
    kwargs = dict(
        shape=shape,
        dev=device,
        dh=args.dh,
        dt=args.dt,
        source_type=["h1"],
        receiver_type=receiver_type,
        abcn=args.abcn,
        pml_type="cpmlr",
        free_surface=False,
    )
    if backend == "eager":
        return PropTorch(
            equation,
            backend="eager",
            eager_options=EagerOptions(use_compile=True),
            use_ckpt=True,
            ckpt_chunks=args.checkpoint_chunks,
            **kwargs,
        )
    return PropTorch(
        equation,
        backend="cuda",
        cuda_options=build_cuda_options(args),
        **kwargs,
    )


def build_acoustic_solver(shape, backend, device, args):
    return build_common_solver(
        Acoustic3D(spatial_order=args.spatial_order, device=device, backend="torch"),
        shape,
        backend,
        device,
        args,
        ["h1"],
    )


def build_lsrtm_solver(shape, backend, device, args):
    return build_common_solver(
        AcousticLSRTM3D(spatial_order=args.spatial_order, device=device, backend="torch"),
        shape,
        backend,
        device,
        args,
        ["sh1"],
    )


def make_models(shape):
    nz, ny, nx = shape
    true_model = np.full(shape, 2000.0, dtype=np.float32)
    true_model[nz // 2 :, :, :] = 2600.0
    true_model[nz // 3 : (2 * nz) // 3, ny // 4 : (3 * ny) // 4, nx // 4 : (3 * nx) // 4] += 120.0
    smooth_model = np.full(shape, 2200.0, dtype=np.float32)
    return true_model, smooth_model


def make_wavelet(args):
    t = np.arange(args.nt, dtype=np.float32) * args.dt
    return ricker(t - args.delay, f=args.fm).astype(np.float32)


def normalize_record_layout(record, nshots, nreceivers, nt):
    if record.ndim == 4:
        if record.shape[-1] != 1:
            raise ValueError(f"Unsupported 4-D record tensor shape {tuple(record.shape)}")
        record = record[..., 0]
    if record.ndim != 3:
        raise ValueError(f"Expected 3-D record tensor, got shape {tuple(record.shape)}")
    if tuple(record.shape) == (nshots, nreceivers, nt):
        return record
    if tuple(record.shape) == (nshots, nt, nreceivers):
        return record.transpose(1, 2)
    raise ValueError(
        f"Unsupported record layout {tuple(record.shape)}; expected {(nshots, nreceivers, nt)} or {(nshots, nt, nreceivers)}."
    )


def generate_observed_scattered_data(shape, backend, device, args):
    true_model, smooth_model = make_models(shape)
    sources, receivers = build_geometry(shape)
    wave = make_wavelet(args)
    acoustic = build_acoustic_solver(shape, backend, device, args)
    true_vp = torch.tensor(true_model, device=device)
    smooth_vp = torch.tensor(smooth_model, device=device)
    solver_kwargs = {}
    if backend == "cuda" and args.cuda_memory == "bs":
        solver_kwargs["use_boundary_saving"] = False
    with torch.no_grad():
        obs = acoustic(wave, sources, receivers, models=[true_vp], **solver_kwargs).detach()
        bg = acoustic(wave, sources, receivers, models=[smooth_vp], **solver_kwargs).detach()
    obs = normalize_record_layout(obs, sources.shape[0], receivers.shape[1], args.nt)
    bg = normalize_record_layout(bg, sources.shape[0], receivers.shape[1], args.nt)
    return {
        "wave": wave,
        "sources": sources,
        "receivers": receivers,
        "smooth_model": smooth_model,
        "observed_scattered": (obs - bg).detach(),
    }


def run_reflectivity_grad(shape, backend, device, args):
    bundle = generate_observed_scattered_data(shape, backend, device, args)
    solver = build_lsrtm_solver(shape, backend, device, args)
    vp = torch.tensor(bundle["smooth_model"], device=device)
    ref = torch.zeros_like(vp, requires_grad=True)
    solver_kwargs = {}
    if backend == "cuda" and args.cuda_memory == "bs":
        solver_kwargs["use_boundary_saving"] = True
    syn = solver(
        bundle["wave"],
        bundle["sources"],
        bundle["receivers"],
        models=[vp, ref],
        **solver_kwargs,
    )
    syn = normalize_record_layout(
        syn,
        bundle["sources"].shape[0],
        bundle["receivers"].shape[1],
        bundle["wave"].shape[-1],
    )
    observed = bundle["observed_scattered"]
    loss = (syn - observed).pow(2).mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    grad = ref.grad
    if grad is None:
        raise RuntimeError(f"{backend} reflectivity gradient is missing.")
    grad = grad.detach().cpu()
    if not torch.isfinite(grad).all():
        raise RuntimeError(f"{backend} reflectivity gradient contains NaN/Inf.")
    return {
        "loss": float(loss.detach().cpu().item()),
        "grad": grad,
        "syn": syn.detach().cpu(),
        "obs": observed.detach().cpu(),
    }


def compute_metrics(reference, candidate):
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    diff = cand_flat - ref_flat
    denom = torch.linalg.norm(ref_flat).clamp_min(1e-8)
    rel_l2 = (torch.linalg.norm(diff) / denom).item()
    cosine = torch.nn.functional.cosine_similarity(
        cand_flat.reshape(1, -1),
        ref_flat.reshape(1, -1),
    ).item()
    return {
        "rel_l2": rel_l2,
        "cosine": cosine,
        "diff_l2": torch.linalg.norm(diff).item(),
        "diff_linf": torch.max(torch.abs(diff)).item(),
    }


def percentile_with_nonzero_fallback(arr, lo=2.0, hi=98.0, eps=1e-12):
    arr = np.asarray(arr)
    vmin, vmax = np.percentile(arr, [lo, hi])
    if not np.isclose(vmin, vmax):
        return float(vmin), float(vmax)
    active = arr[np.abs(arr) > eps]
    if active.size > 0:
        vmin, vmax = np.percentile(active, [lo, hi])
        if not np.isclose(vmin, vmax):
            return float(vmin), float(vmax)
    peak = float(np.max(np.abs(arr))) if arr.size else 1.0
    peak = peak if peak > eps else 1.0
    return -peak, peak


def middle_slices(volume):
    nz, ny, nx = volume.shape
    return [
        ("XY @ Zmid", volume[nz // 2], "X", "Y"),
        ("XZ @ Ymid", volume[:, ny // 2, :], "X", "Z"),
        ("YZ @ Xmid", volume[:, :, nx // 2], "Y", "Z"),
    ]


def save_records_figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    panels = [
        ("eager obs", results["eager"]["obs"][0].numpy().T),
        ("eager syn", results["eager"]["syn"][0].numpy().T),
        ("cuda obs", results["cuda"]["obs"][0].numpy().T),
        ("cuda syn", results["cuda"]["syn"][0].numpy().T),
    ]
    for ax, (title, gather) in zip(axes.reshape(-1), panels):
        vmin, vmax = percentile_with_nonzero_fallback(gather)
        im = ax.imshow(gather, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Receiver Index")
        ax.set_ylabel("Time Sample")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")
    fig.tight_layout()
    output = OUTPUT_DIR / "shot_records.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def save_gradients_figure(results):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    for row, mode in enumerate(("eager", "cuda")):
        grad = results[mode]["grad"].numpy()
        for ax, (slice_title, data, xlabel, ylabel) in zip(axes[row], middle_slices(grad)):
            vmin, vmax = percentile_with_nonzero_fallback(data)
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
            ax.set_title(f"{mode}: {slice_title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, shrink=0.8, label="Gradient")
    fig.tight_layout()
    output = OUTPUT_DIR / "reflectivity_gradients.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable PyTorch environment.")

    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shape = (args.nz, args.ny, args.nx)
    device = torch.device("cuda")
    print(f"python={sys.executable}")
    print(f"import_mode={args.import_mode}")
    print(f"shape={shape}, nt={args.nt}, cuda_memory={args.cuda_memory}")

    results = {
        "eager": run_reflectivity_grad(shape, "eager", device, args),
        "cuda": run_reflectivity_grad(shape, "cuda", device, args),
    }
    metrics = compute_metrics(results["eager"]["grad"], results["cuda"]["grad"])

    record_path = save_records_figure(results)
    grad_path = save_gradients_figure(results)

    print(f"eager loss: {results['eager']['loss']:.6e}")
    print(f"cuda loss:  {results['cuda']['loss']:.6e}")
    print(f"gradient rel_l2: {metrics['rel_l2']:.6e}")
    print(f"gradient cosine: {metrics['cosine']:.6f}")
    print(f"gradient diff_l2: {metrics['diff_l2']:.6e}")
    print(f"gradient diff_linf: {metrics['diff_linf']:.6e}")
    print(f"records figure: {record_path}")
    print(f"gradient figure: {grad_path}")


if __name__ == "__main__":
    main()
