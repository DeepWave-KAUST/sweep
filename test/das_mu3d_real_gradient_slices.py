#!/usr/bin/env python3
"""Plot DAS-Mu 3D real-residual gradient slices.

This script builds a true model and an initial model, generates observed data
from the true model, then computes gradients from synthetic data on the initial
model.  The output figures show central z/y/x slices for vp, vs, and rho.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sweep.equations import DASMu3D
from sweep.propagator.options import CUDAOptions, CkptOptions, MemoryOptions
from sweep.propagator.torch import PropTorch


def ricker(nt: int, dt: float, fm: float, delay: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def build_models(shape: tuple[int, int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nz, ny, nx = shape
    z, y, x = np.meshgrid(
        np.linspace(0.0, 1.0, nz, dtype=np.float32),
        np.linspace(0.0, 1.0, ny, dtype=np.float32),
        np.linspace(0.0, 1.0, nx, dtype=np.float32),
        indexing="ij",
    )
    background = 2150.0 + 380.0 * z + 80.0 * np.sin(np.pi * x) * np.sin(np.pi * y)
    anomaly_pos = np.exp(-(((z - 0.55) / 0.16) ** 2 + ((y - 0.45) / 0.18) ** 2 + ((x - 0.48) / 0.16) ** 2))
    anomaly_neg = np.exp(-(((z - 0.72) / 0.12) ** 2 + ((y - 0.62) / 0.16) ** 2 + ((x - 0.65) / 0.14) ** 2))
    anomaly = 260.0 * anomaly_pos - 150.0 * anomaly_neg

    vp_true = (background + anomaly).astype(np.float32)
    vs_true = (0.55 * background + 0.45 * anomaly + 40.0).astype(np.float32)
    rho_true = (2050.0 + 0.18 * background + 0.10 * anomaly).astype(np.float32)

    vp_init = (background + 0.28 * anomaly).astype(np.float32)
    vs_init = (0.55 * background + 0.13 * anomaly + 40.0).astype(np.float32)
    rho_init = (2050.0 + 0.18 * background + 0.03 * anomaly).astype(np.float32)
    return [vp_true, vs_true, rho_true], [vp_init, vs_init, rho_init]


def build_geometry(shape: tuple[int, int, int], abcn: int, receiver_stride: int) -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = shape
    src_z = min(max(abcn + 4, nz // 4), nz - abcn - 5)
    rec_z = min(src_z + 1, nz - abcn - 5)
    sources = np.array([[[nx // 2, ny // 2, src_z]]], dtype=np.int32)
    xs = np.arange(abcn + 3, nx - abcn - 3, receiver_stride, dtype=np.int32)
    ys = np.arange(abcn + 3, ny - abcn - 3, receiver_stride, dtype=np.int32)
    if xs.size == 0:
        xs = np.array([nx // 2], dtype=np.int32)
    if ys.size == 0:
        ys = np.array([ny // 2], dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    receivers = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.full(xx.size, rec_z, dtype=np.int32)],
        axis=-1,
    )[None, ...]
    return sources, receivers


def tensor_models(models: list[np.ndarray], device: torch.device, requires_grad: bool) -> list[torch.Tensor]:
    return [torch.tensor(array, dtype=torch.float32, device=device, requires_grad=requires_grad) for array in models]


def make_solver(
    *,
    impl: str,
    mode: str,
    device: torch.device,
    shape: tuple[int, int, int],
    nt: int,
    dt: float,
    dh: float,
    abcn: int,
    spatial_order: int,
    free_surface: bool,
) -> PropTorch:
    equation = DASMu3D(spatial_order=spatial_order, device=device, backend="torch")
    common = dict(
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["exx", "eyy", "ezz", "exy", "exz", "eyz"],
        abcn=abcn,
        dh=dh,
        dt=dt,
        dev=device,
        nt=nt,
        pml_type="cpmls",
        free_surface=free_surface,
    )
    if impl == "eager":
        return PropTorch(
            equation,
            backend="torch",
            impl="eager",
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
            **common,
        )
    if mode == "full":
        return PropTorch(
            equation,
            backend="torch",
            impl="c",
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
            **common,
        )
    if mode == "bs":
        return PropTorch(
            equation,
            backend="torch",
            impl="c",
            use_ckpt=False,
            boundary_saving_config={
                "enabled": True,
                "storage": "gpu" if device.type == "cuda" else "cpu",
                "transfer_interval": 1,
            },
            **common,
        )
    if mode == "ckpt_chunk":
        opts = CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=6)))
        return PropTorch(equation, backend="torch", impl="c", cuda_options=opts, **common)
    if mode == "ckpt_recursive":
        opts = CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="recursive", count=4)))
        return PropTorch(equation, backend="torch", impl="c", cuda_options=opts, **common)
    raise ValueError(f"Unknown mode {mode!r}")


def normalize_record(record: torch.Tensor, nt: int, nrec: int, nfields: int) -> torch.Tensor:
    if tuple(record.shape) == (1, nt, nrec, nfields):
        return record
    if tuple(record.shape) == (nfields, 1, nrec, nt):
        return record.permute(1, 3, 2, 0).contiguous()
    raise RuntimeError(f"Unexpected record shape {tuple(record.shape)}")


def run_forward(
    *,
    impl: str,
    mode: str,
    device: torch.device,
    wavelet: torch.Tensor,
    sources: np.ndarray,
    receivers: np.ndarray,
    models: list[np.ndarray],
    args: argparse.Namespace,
) -> torch.Tensor:
    solver = make_solver(
        impl=impl,
        mode=mode,
        device=device,
        shape=tuple(args.shape),
        nt=args.nt,
        dt=args.dt,
        dh=args.dh,
        abcn=args.abcn,
        spatial_order=args.spatial_order,
        free_surface=args.free_surface,
    )
    with torch.no_grad():
        record = solver(wavelet, sources=sources, receivers=receivers, models=tensor_models(models, device, False))
        record = normalize_record(record, args.nt, receivers.shape[1], 6)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return record.detach()


def run_gradient(
    *,
    name: str,
    impl: str,
    mode: str,
    device: torch.device,
    wavelet: torch.Tensor,
    obs: torch.Tensor,
    sources: np.ndarray,
    receivers: np.ndarray,
    init_models: list[np.ndarray],
    args: argparse.Namespace,
) -> dict:
    solver = make_solver(
        impl=impl,
        mode=mode,
        device=device,
        shape=tuple(args.shape),
        nt=args.nt,
        dt=args.dt,
        dh=args.dh,
        abcn=args.abcn,
        spatial_order=args.spatial_order,
        free_surface=args.free_surface,
    )
    models = tensor_models(init_models, device, True)
    syn = solver(wavelet, sources=sources, receivers=receivers, models=models)
    syn = normalize_record(syn, args.nt, receivers.shape[1], 6)
    residual = syn - obs.to(device)
    loss = residual.pow(2).mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    grads = {
        key: tensor.grad.detach().cpu().numpy()[0, 0] if tensor.grad.dim() == 5 else tensor.grad.detach().cpu().numpy()
        for key, tensor in zip(("vp", "vs", "rho"), models)
    }
    return {
        "name": name,
        "loss": float(loss.detach().cpu()),
        "syn": syn.detach().cpu(),
        "residual": residual.detach().cpu(),
        "grads": grads,
    }


def percentile_limits(data: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, percentiles)
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.mean(finite))
        radius = float(np.max(np.abs(finite - center)))
        if not np.isfinite(radius) or radius == 0.0:
            radius = 1.0
        return center - radius, center + radius
    return vmin, vmax


def gradient_slices(data: np.ndarray) -> list[tuple[str, np.ndarray]]:
    nz, ny, nx = data.shape
    return [
        (f"z={nz // 2}", data[nz // 2, :, :]),
        (f"y={ny // 2}", data[:, ny // 2, :]),
        (f"x={nx // 2}", data[:, :, nx // 2]),
    ]


def plot_gradient_figures(results: list[dict], output_dir: Path, percentiles: tuple[float, float]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for model_name in ("vp", "vs", "rho"):
        fig, axes = plt.subplots(
            len(results),
            3,
            figsize=(8.8, 2.15 * len(results)),
            squeeze=False,
        )
        for row, result in enumerate(results):
            for col, (label, data) in enumerate(gradient_slices(result["grads"][model_name])):
                ax = axes[row, col]
                vmin, vmax = percentile_limits(data, percentiles)
                im = ax.imshow(data, cmap="seismic", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
                ax.set_title(f"{result['name']} {model_name} {label}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        path = output_dir / f"das_mu3d_real_grad_{model_name}_slices.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_model_figure(true_models: list[np.ndarray], init_models: list[np.ndarray], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("vp", "vs", "rho")
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.4), squeeze=False)
    for row, (label, models) in enumerate((("true", true_models), ("init", init_models))):
        for col, (name, data) in enumerate(zip(names, models)):
            zslice = data[data.shape[0] // 2]
            vmin, vmax = percentile_limits(zslice, (2.0, 98.0))
            im = axes[row, col].imshow(zslice, cmap="viridis", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{label} {name} z={data.shape[0] // 2}", fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.tight_layout()
    path = output_dir / "das_mu3d_true_init_model_slices.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_record_figure(obs: torch.Tensor, result: dict, output_dir: Path, percentiles: tuple[float, float]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel_names = ("exx", "eyy", "ezz", "exy", "exz", "eyz")
    obs_np = obs[0].numpy()
    syn_np = result["syn"][0].numpy()
    res_np = result["residual"][0].numpy()
    cols = [0, 2, 4]
    fig, axes = plt.subplots(3, len(cols), figsize=(8.5, 5.8), squeeze=False)
    for col_idx, channel in enumerate(cols):
        for row, (label, data) in enumerate((("obs", obs_np), ("syn", syn_np), ("syn-obs", res_np))):
            panel = data[:, :, channel]
            vmin, vmax = percentile_limits(panel, percentiles)
            im = axes[row, col_idx].imshow(panel, cmap="seismic", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
            axes[row, col_idx].set_title(f"{label} {channel_names[channel]}", fontsize=9)
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row, col_idx], fraction=0.046, pad=0.02)
    fig.tight_layout()
    path = output_dir / "das_mu3d_obs_syn_residual_records.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("test/test_outputs/das_mu3d_real_gradient_slices"))
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 28, 32), metavar=("NZ", "NY", "NX"))
    parser.add_argument("--nt", type=int, default=90)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--freq", type=float, default=18.0)
    parser.add_argument("--delay", type=float, default=0.025)
    parser.add_argument("--source-scale", type=float, default=1.0e6)
    parser.add_argument("--abcn", type=int, default=8)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--receiver-stride", type=int, default=3)
    parser.add_argument("--free-surface", action="store_true")
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(2.0, 98.0))
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    true_models, init_models = build_models(tuple(args.shape))
    sources, receivers = build_geometry(tuple(args.shape), args.abcn, args.receiver_stride)
    wave_np = args.source_scale * ricker(args.nt, args.dt, args.freq, args.delay).reshape(1, 1, args.nt)

    if torch.cuda.is_available():
        obs_device = torch.device("cuda")
    else:
        obs_device = torch.device("cpu")
    wave_obs = torch.tensor(wave_np, dtype=torch.float32, device=obs_device)
    obs = run_forward(
        impl="eager",
        mode="full",
        device=obs_device,
        wavelet=wave_obs,
        sources=sources,
        receivers=receivers,
        models=true_models,
        args=args,
    ).cpu()

    cases = []
    if torch.cuda.is_available() and not args.cpu_only:
        cases.extend(
            [
                ("eager-gpu", "eager", "full", torch.device("cuda")),
                ("c-gpu full", "c", "full", torch.device("cuda")),
                ("c-gpu bs", "c", "bs", torch.device("cuda")),
                ("c-gpu ckpt", "c", "ckpt_chunk", torch.device("cuda")),
                ("c-gpu rckpt", "c", "ckpt_recursive", torch.device("cuda")),
            ]
        )
    cases.append(("eager-cpu", "eager", "full", torch.device("cpu")))
    if args.include_cpu or args.cpu_only:
        cases.extend(
            [
                ("c-cpu full", "c", "full", torch.device("cpu")),
                ("c-cpu bs", "c", "bs", torch.device("cpu")),
                ("c-cpu ckpt", "c", "ckpt_chunk", torch.device("cpu")),
            ]
        )

    results = []
    for name, impl, mode, device in cases:
        wavelet = torch.tensor(wave_np, dtype=torch.float32, device=device)
        result = run_gradient(
            name=name,
            impl=impl,
            mode=mode,
            device=device,
            wavelet=wavelet,
            obs=obs,
            sources=sources,
            receivers=receivers,
            init_models=init_models,
            args=args,
        )
        print(f"{name:12s} loss={result['loss']:.6e}")
        results.append(result)

    reference = results[0]
    metrics = {
        "shape": list(args.shape),
        "nt": args.nt,
        "dt": args.dt,
        "dh": args.dh,
        "abcn": args.abcn,
        "spatial_order": args.spatial_order,
        "source_scale": args.source_scale,
        "sources": sources.tolist(),
        "receivers_shape": list(receivers.shape),
        "results": [],
    }
    for result in results:
        item = {"name": result["name"], "loss": result["loss"], "cosine_vs_first": {}}
        for model_name in ("vp", "vs", "rho"):
            item["cosine_vs_first"][model_name] = cosine(result["grads"][model_name], reference["grads"][model_name])
        metrics["results"].append(item)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    paths = []
    paths.append(plot_model_figure(true_models, init_models, output_dir))
    paths.append(plot_record_figure(obs, results[0], output_dir, tuple(args.percentiles)))
    paths.extend(plot_gradient_figures(results, output_dir, tuple(args.percentiles)))
    print("Saved:")
    for path in paths:
        print(path)
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
