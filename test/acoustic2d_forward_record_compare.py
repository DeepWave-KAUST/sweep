#!/usr/bin/env python3
"""Save Acoustic2D eager-vs-CUDA forward record comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import solver_gradient_mode_suite as suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare dense Acoustic2D forward records for eager and CUDA.")
    parser.add_argument("--scenarios", default="interior,fd_edge,free_surface", help="Comma list or 'all'.")
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--nx", type=int, default=56)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--abcn", type=int, default=8)
    parser.add_argument("--receiver-stride", type=int, default=1)
    parser.add_argument("--model", choices=("init", "true"), default="init")
    parser.add_argument("--cuda-mode", default="full")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=suite.REPO_ROOT / "test" / "test_outputs" / "acoustic2d_forward_record_compare",
    )
    parser.add_argument("--run-name", default=None)
    return parser


def suite_args(args: argparse.Namespace):
    base = suite.build_parser().parse_args([])
    base.nz2d = args.nz
    base.nx2d = args.nx
    base.nt = args.nt
    base.dt = args.dt
    base.dh = args.dh
    base.freq = args.freq
    base.delay = args.delay
    base.spatial_order = args.spatial_order
    base.abcn = args.abcn
    base.receiver_stride2d = args.receiver_stride
    base.no_plot = True
    base.no_fail = True
    return base


def record_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.reshape(-1).double()
    cand = candidate.reshape(-1).double()
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    ref = ref[finite]
    cand = cand[finite]
    diff = cand - ref
    ref_l2 = torch.linalg.vector_norm(ref)
    cand_l2 = torch.linalg.vector_norm(cand)
    return {
        "rel_l2": float(torch.linalg.vector_norm(diff) / max(float(ref_l2), 1e-30)),
        "cosine": float(torch.dot(ref, cand) / max(float(ref_l2 * cand_l2), 1e-30)),
        "eager_l2": float(ref_l2),
        "cuda_l2": float(cand_l2),
        "norm_ratio_cuda_over_eager": float(cand_l2 / max(float(ref_l2), 1e-30)),
        "diff_linf": float(diff.abs().max()) if diff.numel() else float("nan"),
    }


def percentile_limit(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(finite), 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def save_record_plot(path: Path, title: str, eager: torch.Tensor, cuda: torch.Tensor, sources, receivers, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eager_np = eager[0].numpy()
    cuda_np = cuda[0].numpy()
    diff_np = cuda_np - eager_np
    vmin, vmax = percentile_limit(np.concatenate([eager_np.reshape(-1), cuda_np.reshape(-1)]))
    dmin, dmax = percentile_limit(diff_np)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    rows = [
        ("eager", eager_np, vmin, vmax),
        ("cuda", cuda_np, vmin, vmax),
        ("cuda - eager", diff_np, dmin, dmax),
    ]
    for ax, (name, image, lo, hi) in zip(axes, rows):
        im = ax.imshow(image, cmap="seismic", aspect="auto", origin="upper", vmin=lo, vmax=hi)
        ax.set_ylabel("receiver")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("time sample")
    metric_text = (
        f"source={np.asarray(sources).tolist()}  receivers={tuple(np.asarray(receivers).shape)}  "
        f"rel_l2={metrics['rel_l2']:.3e}  cos={metrics['cosine']:.6f}  "
        f"norm_ratio={metrics['norm_ratio_cuda_over_eager']:.3e}"
    )
    fig.suptitle(f"{title}\n{metric_text}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_geometry_plot(path: Path, title: str, shape, sources, receivers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz, nx = shape
    src = np.asarray(sources).reshape(-1, 2)
    rec = np.asarray(receivers).reshape(-1, 2)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(nz - 0.5, -0.5)
    ax.scatter(rec[:, 0], rec[:, 1], s=18, marker="o", facecolors="none", edgecolors="black", label="receiver")
    ax.scatter(src[:, 0], src[:, 1], s=110, marker="*", c="cyan", edgecolors="black", label="source")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    sargs = suite_args(args)
    scenario_keys = suite.parse_csv(args.scenarios, suite.SCENARIOS, label="scenario")
    suite.require_cuda_bindings(["acoustic2d"])

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    spec = suite.SOLVERS["acoustic2d"]
    wavelet = torch.tensor(suite.ricker(args.nt, args.dt, args.freq, args.delay), device=device)
    rows = []
    for scenario_key in scenario_keys:
        scenario = suite.SCENARIOS[scenario_key]
        shape = suite.shape_for(spec, sargs)
        sources, receivers = suite.make_geometry(spec, shape, scenario, sargs)
        true_models, init_models, _ = suite.make_models(spec, shape)
        models = true_models if args.model == "true" else init_models
        case_dir = run_dir / scenario_key
        case_dir.mkdir(parents=True, exist_ok=True)

        eager_solver = suite.build_solver(spec, "eager", "eager", scenario, shape, device, sargs, run_dir, scenario_key)
        cuda_solver = suite.build_solver(spec, "cuda", args.cuda_mode, scenario, shape, device, sargs, run_dir, scenario_key)
        eager_record = suite.run_forward(eager_solver, wavelet, sources, receivers, models, device)
        cuda_record = suite.run_forward(cuda_solver, wavelet, sources, receivers, models, device)
        metrics = record_metrics(eager_record, cuda_record)

        plot_path = case_dir / f"acoustic2d_{scenario_key}_{args.model}_record_compare.png"
        npz_path = case_dir / f"acoustic2d_{scenario_key}_{args.model}_records.npz"
        geometry_path = case_dir / f"acoustic2d_{scenario_key}_geometry.png"
        save_record_plot(plot_path, f"acoustic2d {scenario_key} forward record ({args.model} model)", eager_record, cuda_record, sources, receivers, metrics)
        save_geometry_plot(geometry_path, f"acoustic2d {scenario_key} geometry", shape, sources, receivers)
        np.savez(
            npz_path,
            eager=eager_record.numpy(),
            cuda=cuda_record.numpy(),
            sources=sources,
            receivers=receivers,
            shape=np.asarray(shape, dtype=np.int32),
        )
        metadata = {
            "scenario": scenario_key,
            "shape": shape,
            "model": args.model,
            "sources": sources.tolist(),
            "receivers_shape": tuple(receivers.shape),
            "receiver_stride": args.receiver_stride,
            "metrics": metrics,
            "record_plot": str(plot_path),
            "geometry_plot": str(geometry_path),
            "records_npz": str(npz_path),
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        row = {
            "scenario": scenario_key,
            "model": args.model,
            "sources": sources.tolist(),
            "receivers_shape": tuple(receivers.shape),
            "record_plot": str(plot_path),
            "geometry_plot": str(geometry_path),
            "records_npz": str(npz_path),
            **{key: f"{value:.9e}" for key, value in metrics.items()},
        }
        rows.append(row)
        print(
            f"{scenario_key}: receivers={receivers.shape} rel_l2={metrics['rel_l2']:.3e} "
            f"cos={metrics['cosine']:.6f} norm_ratio={metrics['norm_ratio_cuda_over_eager']:.3e}"
        )

    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"run_dir={run_dir}")
    print(f"summary_csv={csv_path}")


if __name__ == "__main__":
    main()
