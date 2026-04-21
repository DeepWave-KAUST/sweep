from pathlib import Path
import sys


def find_examples_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if candidate.name == "examples":
            return candidate
    raise RuntimeError("Could not locate the examples directory.")


EXAMPLES_DIR = find_examples_root()
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))
from bootstrap import configure_example_imports

IMPORT_MODE = configure_example_imports(EXAMPLES_DIR)

import numpy as np
import torch

from common_benchmark import (
    build_prop_torch_solver,
    build_ricker_wavelet,
    default_methods,
    get_platform_label,
    parse_args,
    print_table,
    run_benchmark_suite,
    save_outputs,
    seed_everything,
    select_methods,
)
from sweep.equations import Acoustic3D


OUTPUT_DIR = Path(__file__).resolve().parent / "acoustic3d_memory_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "shape": (72, 96, 96),  # (nz, ny, nx)
    "dh": 10.0,
    "dt": 0.001,
    "nt": 600,
    "fm": 8.0,
    "delay": 0.12,
    "spatial_order": 8,
    "abcn": 20,
    "free_surface": False,
    "nshots_y": 2,
    "nshots_x": 2,
    "srcz": 4,
    "recz": 6,
    "warmup": 1,
    "repeat": 3,
}


def build_geometry(shape, cfg):
    _, ny, nx = shape
    src_y = np.linspace(16, ny - 16, cfg["nshots_y"], dtype=np.int64)
    src_x = np.linspace(16, nx - 16, cfg["nshots_x"], dtype=np.int64)
    yy, xx = np.meshgrid(src_y, src_x, indexing="ij")
    sources = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.full(yy.size, cfg["srcz"], dtype=np.int64)],
        axis=1,
    )

    rec_y = np.arange(8, ny - 8, 4, dtype=np.int64)
    rec_x = np.arange(8, nx - 8, 4, dtype=np.int64)
    ryy, rxx = np.meshgrid(rec_y, rec_x, indexing="ij")
    receivers_single = np.stack(
        [rxx.reshape(-1), ryy.reshape(-1), np.full(ryy.size, cfg["recz"], dtype=np.int64)],
        axis=1,
    )
    receivers = receivers_single[None, ...].repeat(sources.shape[0], axis=0)
    return sources, receivers


def build_models(shape, dev):
    nz, ny, nx = shape
    vp = np.full(shape, 2200.0, dtype=np.float32)
    vp[nz // 2 :, :, :] = 2600.0
    vp[nz // 3 : nz // 2, ny // 3 : 2 * ny // 3, nx // 3 : 2 * nx // 3] += 150.0
    return torch.from_numpy(vp).to(dev).requires_grad_(True)


def build_solver(method, dev):
    equation = Acoustic3D(
        spatial_order=CONFIG["spatial_order"],
        device=dev,
        backend="torch",
    )
    return build_prop_torch_solver(equation=equation, method=method, dev=dev, config=CONFIG)


def gradient_panel(grad):
    nz, ny, nx = grad.shape
    # Slice along y instead of z so the cross-section is more likely to pass
    # through the vertical anomaly structure in this 3D setup.
    y_index = int(np.argmax(np.mean(np.abs(grad), axis=(0, 2))))
    panel = grad[:, y_index, :]
    extent = [0.0, nx * CONFIG["dh"], nz * CONFIG["dh"], 0.0]
    return panel, extent


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    args = parse_args(IMPORT_MODE, "Benchmark eager/cuda memory-saving methods on Acoustic3D.")
    methods = select_methods(default_methods(), args.methods)

    seed_everything(0)

    dev = torch.device("cuda")
    wavelet = build_ricker_wavelet(CONFIG)
    sources, receivers = build_geometry(CONFIG["shape"], CONFIG)
    results = run_benchmark_suite(
        methods=methods,
        config=CONFIG,
        dev=dev,
        wavelet=wavelet,
        sources=sources,
        receivers=receivers,
        build_solver_fn=build_solver,
        build_models_fn=build_models,
    )

    print("")
    print_table(results)
    paths = save_outputs(
        results,
        OUTPUT_DIR,
        "Acoustic3D Memory Benchmark",
        gradient_panel,
        "Gradient Comparison (Max-Energy Y Slice)",
        "x (m)",
        "z (m)",
        get_platform_label(dev),
    )
    print("")
    print(f"Saved benchmark results to {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
