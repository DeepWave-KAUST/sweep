import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.propagator.options import BoundaryOptions, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


def default_methods():
    return [
        {
            "name": "eager_full",
            "backend": "eager",
            "eager_options": EagerOptions(use_compile=False),
            "use_ckpt": False,
        },
        {
            "name": "eager_compile_full",
            "backend": "eager",
            "eager_options": EagerOptions(use_compile=True, compile_mode="default"),
            "use_ckpt": False,
        },
        {
            "name": "eager_ckpt_100",
            "backend": "eager",
            "eager_options": EagerOptions(use_compile=False),
            "use_ckpt": True,
            "ckpt_chunks": 100,
        },
        {
            "name": "eager_compile_ckpt_100",
            "backend": "eager",
            "eager_options": EagerOptions(use_compile=True, compile_mode="default"),
            "use_ckpt": True,
            "ckpt_chunks": 100,
        },
        {
            "name": "cuda_full",
            "backend": "cuda",
            "cuda_options": CUDAOptions(memory=None),
        },
        {
            "name": "cuda_boundary_gpu",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=BoundaryOptions(storage="gpu"),
                )
            ),
        },
        {
            "name": "cuda_boundary_cpu_i8",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=BoundaryOptions(storage="cpu", transfer_interval=8, pinned_memory=True),
                )
            ),
        },
        {
            "name": "cuda_boundary_cpu_i16",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=BoundaryOptions(storage="cpu", transfer_interval=16, pinned_memory=True),
                )
            ),
        },
        {
            "name": "cuda_ckpt_chunk_100",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="ckpt",
                    ckpt=CkptOptions(mode="chunk", chunks=100),
                )
            ),
        },
        {
            "name": "cuda_ckpt_chunk_50",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="ckpt",
                    ckpt=CkptOptions(mode="chunk", chunks=50),
                )
            ),
        },
        {
            "name": "cuda_ckpt_recursive_4",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="ckpt",
                    ckpt=CkptOptions(mode="recursive", count=4),
                )
            ),
        },
        {
            "name": "cuda_ckpt_recursive_8",
            "backend": "cuda",
            "cuda_options": CUDAOptions(
                memory=MemoryOptions(
                    strategy="ckpt",
                    ckpt=CkptOptions(mode="recursive", count=8),
                )
            ),
        },
    ]


def parse_args(import_mode, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=import_mode,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of method names to run.",
    )
    return parser.parse_args()


def select_methods(methods, wanted_names):
    if not wanted_names:
        return methods
    wanted = set(wanted_names)
    selected = [method for method in methods if method["name"] in wanted]
    missing = wanted - {method["name"] for method in selected}
    if missing:
        raise ValueError(f"Unknown methods requested: {sorted(missing)}")
    return selected


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ricker_wavelet(config):
    t = np.arange(config["nt"], dtype=np.float32) * config["dt"]
    return ricker(t - config["delay"], f=config["fm"]).astype(np.float32)


def build_prop_torch_solver(*, equation, method, dev, config):
    # Keep the top-level benchmark scripts focused on geometry/model setup.
    return PropTorch(
        equation,
        shape=config["shape"],
        dev=dev,
        dh=config["dh"],
        dt=config["dt"],
        nt=config["nt"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=config["abcn"],
        free_surface=config["free_surface"],
        pml_type="cpmlr",
        backend=method["backend"],
        eager_options=method.get("eager_options"),
        cuda_options=method.get("cuda_options"),
        use_ckpt=method.get("use_ckpt", False),
        ckpt_chunks=method.get("ckpt_chunks", 100),
    )


def time_one_run(solver, wavelet, sources, receivers, dev, config, build_models_fn, *, return_grad=False):
    model = build_models_fn(config["shape"], dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    syn = solver(wavelet, sources, receivers, models=[model])
    loss = syn.pow(2).mean()
    loss.backward()

    if dev.type == "cuda":
        end.record()
        torch.cuda.synchronize(dev)
        elapsed_ms = float(start.elapsed_time(end))
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = float("nan")

    grad = model.grad.detach().cpu().numpy()
    grad_norm = float(np.linalg.norm(grad))
    if return_grad:
        return elapsed_ms, peak_mb, grad_norm, grad
    return elapsed_ms, peak_mb, grad_norm


def benchmark_method(method, wavelet, sources, receivers, dev, config, build_solver_fn, build_models_fn):
    solver = build_solver_fn(method, dev)

    for _ in range(config["warmup"]):
        time_one_run(solver, wavelet, sources, receivers, dev, config, build_models_fn)

    times = []
    peaks = []
    grad_norms = []
    for _ in range(config["repeat"]):
        elapsed_ms, peak_mb, grad_norm = time_one_run(
            solver, wavelet, sources, receivers, dev, config, build_models_fn
        )
        times.append(elapsed_ms)
        peaks.append(peak_mb)
        grad_norms.append(grad_norm)

    _, _, _, grad = time_one_run(
        solver, wavelet, sources, receivers, dev, config, build_models_fn, return_grad=True
    )

    return {
        "name": method["name"],
        "backend": method["backend"],
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "peak_mb": float(np.mean(peaks)),
        "grad_norm": float(np.mean(grad_norms)),
        "grad": grad,
    }


def print_table(results):
    header = f"{'method':<24} {'backend':<8} {'time_ms':>12} {'peak_mb':>12} {'grad_norm':>12}"
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['name']:<24} {item['backend']:<8} "
            f"{item['mean_ms']:>12.2f} {item['peak_mb']:>12.2f} {item['grad_norm']:>12.4e}"
        )


def save_markdown(results, out_path, title):
    lines = [
        f"# {title}",
        "",
        "| method | backend | mean time (ms) | peak memory (MB) | grad norm |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['backend']} | {item['mean_ms']:.2f} | "
            f"{item['peak_mb']:.2f} | {item['grad_norm']:.4e} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_peak_mb(value):
    if np.isnan(value):
        return "N/A"
    return f"{value:.1f} MB"


def _gradient_display_scale(panel):
    # Use a per-panel robust scale so weaker 3D slices are still visible.
    vmax = float(np.percentile(np.abs(panel), 98.0))
    if vmax <= 0.0:
        vmax = float(np.max(np.abs(panel)))
    return max(vmax, 1e-8)


def get_platform_label(dev):
    if dev.type == "cuda":
        return f"CPU: {get_cpu_model()} | GPU: {torch.cuda.get_device_name(dev)}"
    return f"CPU: {get_cpu_model()}"


def get_cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "Unknown CPU"


def plot_summary(results, out_path, title, platform_label):
    names = [item["name"] for item in results]
    times = [item["mean_ms"] for item in results]
    time_err = [item["std_ms"] for item in results]
    peaks = [item["peak_mb"] for item in results]

    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)

    time_bars = axes[0].bar(x, times, yerr=time_err, color="#4C78A8", capsize=4)
    axes[0].set_title("Runtime Comparison")
    axes[0].set_ylabel("Forward + Backward Time (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].bar_label(time_bars, labels=[f"{v:.1f}" for v in times], padding=3, fontsize=8)

    peak_bars = axes[1].bar(x, peaks, color="#F58518")
    axes[1].set_title("Peak Memory Comparison")
    axes[1].set_ylabel("Peak GPU Memory (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    axes[1].bar_label(peak_bars, labels=[_format_peak_mb(v) for v in peaks], padding=3, fontsize=8)

    fig.suptitle(f"{title} | {platform_label}")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gradients(results, out_path, panel_fn, title, x_label, y_label, platform_label):
    names = [item["name"] for item in results]
    ncols = min(3, len(results))
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, name, item in zip(axes.ravel(), names, results):
        panel, extent = panel_fn(item["grad"])
        vmax = _gradient_display_scale(panel)
        ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(
            f"{name}\n{item['mean_ms']:.1f} ms | {_format_peak_mb(item['peak_mb'])} | +/-{vmax:.2e}",
            fontsize=10,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    for ax in axes.ravel()[len(results):]:
        ax.axis("off")

    fig.suptitle(f"{title} | {platform_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    results,
    output_dir,
    markdown_title,
    gradient_panel_fn,
    gradient_title,
    x_label,
    y_label,
    platform_label,
):
    json_path = output_dir / "results.json"
    md_path = output_dir / "results.md"
    summary_path = output_dir / "summary.png"
    grad_path = output_dir / "gradients.png"
    json_results = [{k: v for k, v in item.items() if k != "grad"} for item in results]
    json_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    save_markdown(results, md_path, markdown_title)
    plot_summary(results, summary_path, markdown_title, platform_label)
    plot_gradients(results, grad_path, gradient_panel_fn, gradient_title, x_label, y_label, platform_label)
    return json_path, md_path, summary_path, grad_path


def run_benchmark_suite(
    *,
    methods,
    config,
    dev,
    wavelet,
    sources,
    receivers,
    build_solver_fn,
    build_models_fn,
):
    results = []
    for method in methods:
        print(f"Running {method['name']}...")
        results.append(
            benchmark_method(
                method,
                wavelet,
                sources,
                receivers,
                dev,
                config,
                build_solver_fn,
                build_models_fn,
            )
        )
    return results
