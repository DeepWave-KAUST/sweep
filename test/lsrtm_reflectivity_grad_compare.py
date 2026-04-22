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

import configure_marmousi as shared_config
from sweep.equations import Acoustic, AcousticLSRTM
from sweep.propagator.options import BoundaryOptions, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True

OUTPUT_DIR = REPO_ROOT / "test" / "test_outputs" / "lsrtm_reflectivity_grad_compare"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare AcousticLSRTM reflectivity gradients between eager and CUDA modes."
    )
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument("--nz", type=int, default=None, help="Depth samples for the Marmousi crop. Defaults to the full model.")
    parser.add_argument("--nx", type=int, default=None, help="Horizontal samples for the Marmousi crop. Defaults to the full model.")
    parser.add_argument("--z0", type=int, default=None, help="Top index of the Marmousi crop. Defaults to a centered crop.")
    parser.add_argument("--x0", type=int, default=None, help="Left index of the Marmousi crop. Defaults to a centered crop.")
    parser.add_argument("--nt", type=int, default=None, help="Number of time samples. Defaults to the shared Marmousi config.")
    parser.add_argument("--dt", type=float, default=None, help="Time step. Defaults to shared Marmousi config.")
    parser.add_argument("--dh", type=float, default=None, help="Grid spacing. Defaults to shared Marmousi config.")
    parser.add_argument("--fm", type=float, default=None, help="Ricker dominant frequency. Defaults to the shared Marmousi config.")
    parser.add_argument("--delay", type=float, default=None, help="Ricker delay in seconds. Defaults to the shared Marmousi config.")
    parser.add_argument("--spatial-order", type=int, default=8)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--src-step", type=int, default=24)
    parser.add_argument("--rec-step", type=int, default=4)
    parser.add_argument("--src-z", type=int, default=2)
    parser.add_argument("--rec-z", type=int, default=6)
    parser.add_argument("--nshots", type=int, default=1, help="Number of shots to compare.")
    parser.add_argument("--boundary-transfer-interval", type=int, default=10)
    parser.add_argument("--checkpoint-chunks", type=int, default=64)
    parser.add_argument("--checkpoint-count", type=int, default=6)
    parser.add_argument("--show-shot", type=int, default=-1, help="Shot index to visualize for the observed data figure.")
    return parser


def load_models(args):
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    smooth_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)

    model_nz, model_nx = true_model.shape
    crop_nz = model_nz if args.nz is None else args.nz
    crop_nx = model_nx if args.nx is None else args.nx
    if crop_nz > model_nz or crop_nx > model_nx:
        raise ValueError(
            f"Requested crop ({crop_nz}, {crop_nx}) exceeds Marmousi bounds ({model_nz}, {model_nx})."
        )

    z0 = args.z0 if args.z0 is not None else max(0, (model_nz - crop_nz) // 2)
    x0 = args.x0 if args.x0 is not None else max(0, (model_nx - crop_nx) // 2)
    z1 = z0 + crop_nz
    x1 = x0 + crop_nx
    true_crop = true_model[z0:z1, x0:x1].copy()
    smooth_crop = smooth_model[z0:z1, x0:x1].copy()
    if true_crop.shape != (crop_nz, crop_nx) or smooth_crop.shape != (crop_nz, crop_nx):
        raise ValueError(
            f"Requested crop z[{z0}:{z1}) x[{x0}:{x1}) exceeds Marmousi bounds ({model_nz}, {model_nx})."
        )
    return true_crop, smooth_crop


def build_geometry(shape, args):
    _, nx = shape

    if args.nshots == 1:
        src_x = np.array([nx // 2], dtype=np.int64)
    else:
        src_x = np.arange(12, nx - 12, args.src_step, dtype=np.int64)
        if src_x.size == 0:
            raise ValueError("No shots fit within the requested crop. Reduce src-step or enlarge nx.")
        center_idx = src_x.size // 2
        half = args.nshots // 2
        start = max(0, center_idx - half)
        end = min(src_x.size, start + args.nshots)
        start = max(0, end - args.nshots)
        src_x = src_x[start:end]
    sources = np.stack([src_x, np.full(src_x.shape[0], args.src_z, dtype=np.int64)], axis=-1)

    rec_x = np.arange(8, nx - 8, args.rec_step, dtype=np.int64)
    receivers = np.stack([rec_x, np.full(rec_x.shape[0], args.rec_z, dtype=np.int64)], axis=-1)[None, ...]
    receivers = receivers.repeat(sources.shape[0], axis=0)
    return sources, receivers


def build_wavelet(args):
    t = np.arange(args.nt, dtype=np.float32) * args.dt
    return ricker(t - args.delay, f=args.fm).astype(np.float32)


def build_cuda_options(mode, args):
    if mode == "full":
        return CUDAOptions(memory=None)
    if mode == "bs":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(
                    storage="gpu",
                    transfer_interval=1,
                    pinned_memory=False,
                ),
            )
        )
    if mode == "bs_cpu":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(
                    storage="cpu",
                    transfer_interval=args.boundary_transfer_interval,
                    pinned_memory=True,
                ),
            )
        )
    if mode == "ckpt":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="chunk", chunks=args.checkpoint_chunks),
            )
        )
    if mode == "recursive":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="recursive", count=args.checkpoint_count),
            )
        )
    raise ValueError(f"Unsupported CUDA mode '{mode}'.")


def build_solver(equation, shape, dh, dt, device, backend, mode, args, receiver_type):
    common_kwargs = dict(
        shape=shape,
        dev=device,
        dh=dh,
        dt=dt,
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
            eager_options=EagerOptions(use_compile=False),
            **common_kwargs,
        )

    return PropTorch(
        equation,
        backend="cuda",
        cuda_options=build_cuda_options(mode, args),
        **common_kwargs,
    )


def _normalize_record_layout(record, nshots, nreceivers, nt):
    if record.ndim == 4:
        if record.shape[-1] != 1:
            raise ValueError(f"Unsupported 4-D record tensor shape {tuple(record.shape)}")
        record = record[..., 0]

    if record.ndim != 3:
        raise ValueError(f"Expected 3-D record tensor after squeezing channel dim, got shape {tuple(record.shape)}")

    shape = tuple(record.shape)
    if shape == (nshots, nreceivers, nt):
        return record
    if shape == (nshots, nt, nreceivers):
        return record.transpose(1, 2)

    raise ValueError(
        f"Unsupported record layout {shape}; expected {(nshots, nreceivers, nt)} or {(nshots, nt, nreceivers)}."
    )


def generate_observed_scattered_data(shape, true_model, smooth_model, wavelet, sources, receivers, dh, dt, device, args, backend, mode):
    acoustic = build_solver(
        Acoustic(spatial_order=args.spatial_order, device=device, backend="torch"),
        shape,
        dh,
        dt,
        device,
        backend=backend,
        mode=mode,
        args=args,
        receiver_type=["h1"],
    )

    true_vp = torch.tensor(true_model, device=device, dtype=torch.float32)
    background_vp = torch.tensor(smooth_model, device=device, dtype=torch.float32)

    with torch.no_grad():
        obs = acoustic(wavelet, sources, receivers, models=[true_vp]).detach().clone()
        background = acoustic(wavelet, sources, receivers, models=[background_vp]).detach().clone()
    obs = _normalize_record_layout(obs, sources.shape[0], receivers.shape[1], args.nt)
    background = _normalize_record_layout(background, sources.shape[0], receivers.shape[1], args.nt)
    scattered = (obs - background).detach()
    return {
        "true_total": obs,
        "background_total": background,
        "observed_scattered": scattered,
    }


def run_reflectivity_grad(solver, wavelet, sources, receivers, smooth_model, observed, device):
    vp = torch.tensor(smooth_model, device=device, dtype=torch.float32)
    ref = torch.zeros_like(vp, requires_grad=True)
    syn = solver(wavelet, sources, receivers, models=[vp, ref])
    syn = _normalize_record_layout(syn, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
    loss = (syn - observed).pow(2).mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    grad = ref.grad
    if grad is None:
        raise RuntimeError("Reflectivity gradient is missing.")
    grad = grad.detach().cpu()
    if not torch.isfinite(grad).all():
        raise RuntimeError("Reflectivity gradient contains NaN or Inf.")
    return {
        "loss": float(loss.detach().cpu().item()),
        "grad": grad,
        "syn": syn.detach().cpu(),
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
    ref_img = reference.to(torch.float64)
    cand_img = candidate.to(torch.float64)
    data_min = torch.minimum(ref_img.min(), cand_img.min())
    data_max = torch.maximum(ref_img.max(), cand_img.max())
    data_range = (data_max - data_min).clamp_min(torch.tensor(1e-8, dtype=torch.float64))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = ref_img.mean()
    mu_y = cand_img.mean()
    sigma_x = ((ref_img - mu_x) ** 2).mean()
    sigma_y = ((cand_img - mu_y) ** 2).mean()
    sigma_xy = ((ref_img - mu_x) * (cand_img - mu_y)).mean()
    ssim = (
        ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
        / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    ).item()
    return {
        "rel_l2": rel_l2,
        "cosine": cosine,
        "ssim": ssim,
        "diff_l2": torch.linalg.norm(diff).item(),
        "diff_linf": torch.max(torch.abs(diff)).item(),
    }


def finite_percentiles(arr, lo=2.0, hi=98.0):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    return float(vmin), float(vmax)


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


def save_observed_figure(observed_results, args):
    shot_idx = args.show_shot if args.show_shot >= 0 else 0
    groups = [
        ("true_total", "True Total"),
        ("background_total", "Background Total"),
        ("observed_scattered", "Observed Scattered"),
    ]
    fig, axes = plt.subplots(len(groups), 3, figsize=(18, 12), squeeze=False)

    eager = observed_results["eager"]
    cuda = observed_results["cuda_full"]

    for row, (key, title) in enumerate(groups):
        eager_obs = eager[key]
        cuda_obs = cuda[key]
        shot_idx_clamped = min(max(shot_idx, 0), eager_obs.shape[0] - 1)
        eager_gather = eager_obs[shot_idx_clamped].squeeze().T
        cuda_gather = cuda_obs[shot_idx_clamped].squeeze().T
        diff_gather = cuda_gather - eager_gather

        panels = [
            ("eager", eager_gather, "seismic"),
            ("cuda_full", cuda_gather, "seismic"),
            ("difference", diff_gather, "seismic"),
        ]

        for col, (label, gather, cmap) in enumerate(panels):
            ax = axes[row, col]
            vmin, vmax = percentile_with_nonzero_fallback(gather, 2, 98)
            im = ax.imshow(gather, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            cbar = fig.colorbar(im, ax=ax, shrink=0.9, label="Amplitude")
            cbar.ax.yaxis.get_offset_text().set_visible(True)
            ax.set_title(f"{title}: {label}")
            ax.set_xlabel("Receiver Index")
            ax.set_ylabel("Time Sample")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output = OUTPUT_DIR / "observed_scattered_data.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def save_gradient_figure(results):
    mode_names = list(results.keys())
    ncols = int(np.ceil(len(mode_names) / 2))
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8), squeeze=False)
    axes = axes.reshape(-1)

    for ax, mode in zip(axes, mode_names):
        grad = results[mode]["grad"].numpy()
        vmin, vmax = np.percentile(grad, [2, 98])
        im = ax.imshow(grad, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(mode)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.yaxis.get_offset_text().set_visible(True)

    for ax in axes[len(mode_names):]:
        ax.axis("off")

    fig.suptitle("AcousticLSRTM Reflectivity Gradient Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = OUTPUT_DIR / "reflectivity_gradient_compare.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def save_difference_figure(results, reference_name="eager"):
    other_modes = [name for name in results if name != reference_name]
    fig, axes = plt.subplots(1, len(other_modes), figsize=(4.5 * len(other_modes), 4), squeeze=False)
    axes = axes[0]

    ref = results[reference_name]["grad"].numpy()
    for ax, mode in zip(axes, other_modes):
        diff = results[mode]["grad"].numpy() - ref
        vmin, vmax = np.percentile(diff, [2, 98])
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        im = ax.imshow(diff, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(f"{mode} - {reference_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("AcousticLSRTM Reflectivity Gradient Differences")
    fig.tight_layout()
    output = OUTPUT_DIR / "reflectivity_gradient_diff.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable PyTorch environment.")

    args = build_parser().parse_args()
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    if args.nt is None:
        args.nt = cfg["nt"]
    if args.dt is None:
        args.dt = cfg["dt"]
    if args.dh is None:
        args.dh = cfg["dh"]
    if args.fm is None:
        args.fm = cfg["fm"]
    if args.delay is None:
        args.delay = cfg["delay"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"python={sys.executable}")
    print(f"import_mode={args.import_mode}")

    true_model, smooth_model = load_models(args)
    shape = true_model.shape
    sources, receivers = build_geometry(shape, args)
    wavelet = build_wavelet(args)
    device = torch.device("cuda")

    modes = [
        ("eager", "full"),
        ("cuda_full", "full"),
        ("cuda_bs", "bs"),
        ("cuda_bs_cpu", "bs_cpu"),
        ("cuda_ckpt", "ckpt"),
        ("cuda_recursive", "recursive"),
    ]

    results = {}
    obs_path = None
    observed_results = {}
    for label, mode in modes:
        backend = "eager" if label == "eager" else "cuda"
        print(f"\nRunning {label} ...")
        observed_bundle = generate_observed_scattered_data(
            shape=shape,
            true_model=true_model,
            smooth_model=smooth_model,
            wavelet=wavelet,
            sources=sources,
            receivers=receivers,
            dh=args.dh,
            dt=args.dt,
            device=device,
            args=args,
            backend=backend,
            mode=mode,
        )
        if label in {"eager", "cuda_full"}:
            observed_results[label] = {
                key: value.detach().cpu().numpy() for key, value in observed_bundle.items()
            }
        if label == "eager":
            print(
                "  observed_scattered stats:",
                f"min={observed_results[label]['observed_scattered'].min():.6e}, "
                f"max={observed_results[label]['observed_scattered'].max():.6e},",
                f"absmax={np.abs(observed_results[label]['observed_scattered']).max():.6e}",
            )
        if label == "cuda_full":
            print(
                "  cuda observed_scattered stats:",
                f"min={observed_results[label]['observed_scattered'].min():.6e}, "
                f"max={observed_results[label]['observed_scattered'].max():.6e},",
                f"absmax={np.abs(observed_results[label]['observed_scattered']).max():.6e}",
            )
        if {"eager", "cuda_full"}.issubset(observed_results.keys()) and obs_path is None:
            obs_path = save_observed_figure(observed_results, args)
        equation = AcousticLSRTM(spatial_order=args.spatial_order, device=device, backend="torch")
        solver = build_solver(
            equation=equation,
            shape=shape,
            dh=args.dh,
            dt=args.dt,
            device=device,
            backend=backend,
            mode=mode,
            args=args,
            receiver_type=["sh1"],
        )
        result = run_reflectivity_grad(
            solver=solver,
            wavelet=wavelet,
            sources=sources,
            receivers=receivers,
            smooth_model=smooth_model,
            observed=observed_bundle["observed_scattered"],
            device=device,
        )
        results[label] = result
        print(f"  loss={result['loss']:.6e}")

    ref_grad = results["eager"]["grad"]
    summary_lines = [
        f"observed_data={obs_path}",
        f"shape={shape}, nt={args.nt}, dt={args.dt}, dh={args.dh}",
        f"sources={sources.shape}, receivers={receivers.shape}",
        "",
        "Gradient consistency relative to eager:",
    ]
    print("\nGradient consistency relative to eager:")
    for label in [name for name in results if name != "eager"]:
        metrics = compute_metrics(ref_grad, results[label]["grad"])
        results[label]["metrics"] = metrics
        line = (
            f"{label}: rel_l2={metrics['rel_l2']:.6f}, cosine={metrics['cosine']:.6f}, ssim={metrics['ssim']:.6f}, "
            f"diff_l2={metrics['diff_l2']:.6e}, diff_linf={metrics['diff_linf']:.6e}"
        )
        print(" ", line)
        summary_lines.append(line)

    grad_fig = save_gradient_figure(results)
    diff_fig = save_difference_figure(results)
    summary_lines.extend(["", f"gradient_figure={grad_fig}", f"diff_figure={diff_fig}"])

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nSaved outputs to {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
