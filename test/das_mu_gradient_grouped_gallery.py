#!/usr/bin/env python3
"""Generate grouped DAS-Mu gradient plots by solver mode.

The model, acquisition geometry, wavelet, and default numerical parameters
match ``test/solver_gradient_mode_suite.py``. Each output image contains one
memory mode, with rows for implementations/devices and columns for model
gradients, making it easy to flip between mode images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from solver_gradient_mode_suite import (
    SCENARIOS,
    SolverSpec,
    build_solver,
    make_geometry,
    make_models,
    metric_pair,
    normalize_record,
    ricker,
    tensors_from_models,
)
from sweep.equations import DASMu


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="full,bs_gpu,ckpt_chunk,ckpt_recursive")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "test" / "test_outputs" / "das_mu_gradient_grouped")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scenario", default="interior", choices=("interior", "free_surface", "fd_edge"))
    parser.add_argument("--percentiles", type=float, nargs=2, default=(2.0, 98.0))
    parser.add_argument("--source-impl", default="eager_gpu", choices=("eager_gpu", "eager_cpu"))
    return parser


def suite_args(output_dir: Path, run_name: str | None):
    return SimpleNamespace(
        nz2d=48,
        nx2d=56,
        nz3d=24,
        ny3d=20,
        nx3d=24,
        nt=120,
        dt=0.0015,
        dh=10.0,
        freq=10.0,
        delay=0.06,
        spatial_order=4,
        abcn=30,
        receiver_stride2d=6,
        receiver_stride3d=8,
        ckpt_chunks=24,
        ckpt_count=4,
        transfer_interval=4,
        disk_ring_buffers=2,
        disk_dir=None,
        output_dir=output_dir,
        run_name=run_name,
    )


def das_mu_spec() -> SolverSpec:
    return SolverSpec(
        "das_mu2d",
        DASMu,
        2,
        ("vp", "vs", "rho"),
        ("sxx", "szz"),
        ("exx", "ezz", "exz"),
        "cpmls",
        elastic=True,
        supported_modes=(
            "full",
            "bs_gpu",
            "bs_cpu",
            "bs_cpu_pinned",
            "bs_disk",
            "bs_disk_async",
            "ckpt_chunk",
            "ckpt_chunk_cpu",
            "ckpt_recursive",
            "ckpt_recursive_cpu",
        ),
        supported_scenarios=("interior", "free_surface", "fd_edge"),
    )


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def guarded_solver_call(solver, wavelet, sources, receivers, *, models):
    sources_in = np.array(sources, dtype=np.int32, copy=True)
    receivers_in = np.array(receivers, dtype=np.int32, copy=True)
    sources_before = sources_in.copy()
    receivers_before = receivers_in.copy()
    record = solver(wavelet, sources_in, receivers_in, models=models)
    if not np.array_equal(sources_in, sources_before):
        raise RuntimeError("Solver mutated source coordinates.")
    if not np.array_equal(receivers_in, receivers_before):
        raise RuntimeError("Solver mutated receiver coordinates.")
    return record


def run_forward(solver, wavelet, sources, receivers, models_np, device):
    models = tensors_from_models(models_np, [False] * len(models_np), device)
    with torch.no_grad():
        record = guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
        maybe_sync(device)
    if not torch.isfinite(record).all():
        raise RuntimeError("Forward record contains NaN or Inf.")
    return record.detach().cpu()


def run_gradient(solver, wavelet, sources, receivers, observed, models_np, grad_flags, model_names, device):
    models = tensors_from_models(models_np, grad_flags, device)
    observed_t = observed.to(device)
    record = guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
    loss = (record - observed_t).pow(2).mean()
    loss.backward()
    maybe_sync(device)
    grads = {}
    for name, tensor, needs_grad in zip(model_names, models, grad_flags):
        if not needs_grad:
            continue
        if tensor.grad is None:
            raise RuntimeError(f"{name} gradient is missing.")
        if not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{name} gradient contains NaN or Inf.")
        grads[name] = tensor.grad.detach().cpu()
    return {
        "loss": float(loss.detach().cpu()),
        "record": record.detach().cpu(),
        "grads": grads,
    }


def gradient_limits(data: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, percentiles)
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.nanmean(finite)) if finite.size else 0.0
        radius = float(np.nanmax(np.abs(finite - center))) if finite.size else 1.0
        radius = radius if radius > 0.0 else 1.0
        return center - radius, center + radius
    return vmin, vmax


def plot_grouped_mode(path: Path, mode: str, results: dict, metrics: dict, sources, receivers, percentiles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    impl_order = [name for name in ("eager_cpu", "eager_gpu", "c_cpu", "c_gpu") if name in results]
    model_names = list(next(iter(results.values()))["grads"])
    fig, axes = plt.subplots(
        len(impl_order),
        len(model_names),
        figsize=(3.25 * len(model_names), 2.65 * len(impl_order)),
        squeeze=False,
    )

    rec = receivers.reshape(-1, receivers.shape[-1])
    src = sources.reshape(-1, sources.shape[-1])
    for row, impl in enumerate(impl_order):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            data = results[impl]["grads"][model_name].numpy()
            vmin, vmax = gradient_limits(data, tuple(percentiles))
            im = ax.imshow(data, cmap="seismic", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
            ax.scatter(rec[:, 0], rec[:, 1], s=9, marker="o", facecolors="none", edgecolors="black", linewidths=0.35)
            ax.scatter(src[:, 0], src[:, 1], s=60, marker="*", c="cyan", edgecolors="black", linewidths=0.45)
            metric = metrics.get(impl, {}).get(model_name)
            metric_text = ""
            if metric is not None and math.isfinite(metric.get("cosine", math.nan)):
                metric_text = f" cos={metric['cosine']:.4f}"
            ax.set_title(f"{impl} {model_name}{metric_text}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"DAS-Mu gradients grouped by mode: {mode}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parsed = build_parser().parse_args()
    run_name = parsed.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (parsed.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    args = suite_args(parsed.output_dir, run_name)
    modes = [item.strip() for item in parsed.modes.split(",") if item.strip()]

    spec = das_mu_spec()
    scenario = SCENARIOS[parsed.scenario]
    shape = (args.nz2d, args.nx2d)
    sources, receivers = make_geometry(spec, shape, scenario, args)
    true_models, init_models, grad_flags = make_models(spec, shape)

    torch.manual_seed(0)
    np.random.seed(0)

    devices = {"eager_cpu": torch.device("cpu"), "c_cpu": torch.device("cpu")}
    if torch.cuda.is_available():
        devices.update({"eager_gpu": torch.device("cuda"), "c_gpu": torch.device("cuda")})
    elif parsed.source_impl == "eager_gpu":
        raise RuntimeError("CUDA is not available, but --source-impl=eager_gpu was requested.")

    wavelets = {
        name: torch.tensor(ricker(args.nt, args.dt, args.freq, args.delay), device=device)
        for name, device in devices.items()
    }

    source_impl = parsed.source_impl if parsed.source_impl in devices else "eager_cpu"
    source_device = devices[source_impl]
    case_key = f"das_mu2d_{scenario.key}"
    source_solver = build_solver(spec, "eager", "eager", scenario, shape, source_device, args, run_dir, case_key)
    observed = run_forward(source_solver, wavelets[source_impl], sources, receivers, true_models, source_device)
    del source_solver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eager_results = {}
    for impl in ("eager_cpu", "eager_gpu"):
        if impl not in devices:
            continue
        device = devices[impl]
        solver = build_solver(spec, "eager", "eager", scenario, shape, device, args, run_dir, case_key)
        eager_results[impl] = run_gradient(
            solver,
            wavelets[impl],
            sources,
            receivers,
            observed,
            init_models,
            grad_flags,
            spec.model_names,
            device,
        )
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reference_impl = "eager_gpu" if "eager_gpu" in eager_results else "eager_cpu"
    reference = eager_results[reference_impl]

    rows = []
    for mode in modes:
        print(f"[mode={mode}]")
        results = dict(eager_results)
        for impl in ("c_cpu", "c_gpu"):
            if impl not in devices:
                continue
            started = time.time()
            device = devices[impl]
            solver = build_solver(spec, "cuda", mode, scenario, shape, device, args, run_dir, case_key)
            results[impl] = run_gradient(
                solver,
                wavelets[impl],
                sources,
                receivers,
                observed,
                init_models,
                grad_flags,
                spec.model_names,
                device,
            )
            print(f"  {impl}: loss={results[impl]['loss']:.9e} seconds={time.time() - started:.2f}")
            del solver
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics = {}
        for impl, result in results.items():
            metrics[impl] = {
                name: metric_pair(reference["grads"][name], result["grads"][name])
                for name in reference["grads"]
            }
            for name, item in metrics[impl].items():
                rows.append(
                    {
                        "mode": mode,
                        "implementation": impl,
                        "model": name,
                        "loss": f"{result['loss']:.9e}",
                        "cosine_vs_reference": f"{item['cosine']:.9e}",
                        "rel_l2_vs_reference": f"{item['rel_l2']:.9e}",
                        "reference_impl": reference_impl,
                    }
                )

        image_path = plot_grouped_mode(
            run_dir / f"{case_key}_{mode}_grouped_gradients.png",
            mode,
            results,
            metrics,
            sources,
            receivers,
            parsed.percentiles,
        )
        print(f"  plot={image_path}")

    metadata = {
        "shape": shape,
        "nt": args.nt,
        "dt": args.dt,
        "dh": args.dh,
        "freq": args.freq,
        "delay": args.delay,
        "spatial_order": args.spatial_order,
        "abcn": args.abcn,
        "sources": sources.tolist(),
        "receivers_shape": tuple(receivers.shape),
        "scenario": scenario.key,
        "free_surface": scenario.free_surface,
        "edge_source": scenario.edge_source,
        "source_impl": source_impl,
        "reference_impl": reference_impl,
        "modes": modes,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "implementation",
                "model",
                "loss",
                "cosine_vs_reference",
                "rel_l2_vs_reference",
                "reference_impl",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
