"""Save gradient plots for the backend gradient consistency test cases.

Each gradient tensor is plotted with its own scale. 2D tensors are rendered as
one image, 3D tensors as three center slices, and 1D tensors as a line plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TEST_ROOT = REPO_ROOT / "test"
for path in (SRC_ROOT, TEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from backend_gradient_matrix import BACKENDS, backend_supported, run_backend  # noqa: E402
from cpu_binding_gradient_consistency import CASES, MODES, config_for_scale, scaled_cases  # noqa: E402


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(dtype=torch.float32).numpy()


def scale_for(array: np.ndarray) -> tuple[float, float, float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0, 0.0, 0.0
    max_abs = float(np.max(np.abs(finite)))
    p2, p98 = np.percentile(finite, [2, 98])
    vmin = float(p2)
    vmax = float(p98)
    if vmin == vmax:
        pad = abs(vmin) * 0.1 if vmin else 1.0
        vmin -= pad
        vmax += pad
    return vmin, vmax, max_abs, float(np.mean(finite))


def one_line_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return text.splitlines()[0]
    return "".join(traceback.format_exception_only(type(exc), exc)).strip().splitlines()[0]


def save_line_plot(array: np.ndarray, path: Path, title: str) -> None:
    y = array.reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.plot(np.arange(y.size), y, linewidth=1.5)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sample")
    ax.set_ylabel("gradient")
    ax.grid(True, alpha=0.25)
    finite = y[np.isfinite(y)]
    if finite.size:
        ymin, ymax = np.percentile(finite, [2, 98])
        ymin = float(ymin)
        ymax = float(ymax)
        if ymin == ymax:
            pad = abs(ymin) * 0.1 if ymin else 1.0
            ymin -= pad
            ymax += pad
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_2d_plot(array: np.ndarray, path: Path, title: str) -> None:
    vmin, vmax, max_abs, _ = scale_for(array)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(array, cmap="seismic", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
    ax.set_title(f"{title}\np2={vmin:.3e}, p98={vmax:.3e}, max_abs={max_abs:.3e}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_3d_plot(array: np.ndarray, path: Path, title: str) -> None:
    zc = array.shape[0] // 2
    yc = array.shape[1] // 2
    xc = array.shape[2] // 2
    slices = (
        (f"z={zc}", array[zc, :, :]),
        (f"y={yc}", array[:, yc, :]),
        (f"x={xc}", array[:, :, xc]),
    )
    vmin, vmax, max_abs, _ = scale_for(array)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150)
    last_im = None
    for ax, (label, image) in zip(axes, slices):
        last_im = ax.imshow(image, cmap="seismic", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        ax.set_title(label, fontsize=8)
    fig.suptitle(f"{title}\np2={vmin:.3e}, p98={vmax:.3e}, max_abs={max_abs:.3e}", fontsize=9)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_gradient(array: np.ndarray, path: Path, title: str) -> tuple[str, float, float, float]:
    squeezed = np.squeeze(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = squeezed[np.isfinite(squeezed)]
    min_value = float(np.min(finite)) if finite.size else float("nan")
    max_value = float(np.max(finite)) if finite.size else float("nan")
    max_abs = float(np.max(np.abs(finite))) if finite.size else float("nan")
    p2 = float(np.percentile(finite, 2)) if finite.size else float("nan")
    p98 = float(np.percentile(finite, 98)) if finite.size else float("nan")

    if squeezed.ndim <= 1:
        save_line_plot(squeezed.reshape(-1), path, title)
        kind = "line"
    elif squeezed.ndim == 2:
        save_2d_plot(squeezed, path, title)
        kind = "image2d"
    elif squeezed.ndim == 3:
        save_3d_plot(squeezed, path, title)
        kind = "slices3d"
    else:
        flattened = squeezed.reshape((-1, *squeezed.shape[-2:]))
        save_3d_plot(flattened, path, title)
        kind = "reshaped_slices"
    return kind, min_value, max_value, max_abs, p2, p98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES], choices=[case.name for case in CASES])
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    parser.add_argument(
        "--scale",
        choices=("tiny", "cuda-suite"),
        default="tiny",
        help=(
            "Problem size profile. 'cuda-suite' matches the older CUDA gradient "
            "suite defaults: 2D=48x56, 3D=24x20x24, nt=120, so=4, abcn=30."
        ),
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, torch.get_num_threads())))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "test" / "outputs" / "gradient_plots",
        help="Directory where PNGs and gradient_index.csv are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_available = torch.cuda.is_available()
    config = config_for_scale(args.scale)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    selected_cases = list(scaled_cases(config, tuple(selected_cases)))
    selected_modes = list(dict.fromkeys(args.modes))
    rows: list[dict[str, object]] = []

    for case in selected_cases:
        for mode in selected_modes:
            for backend in BACKENDS:
                if not backend_supported(mode, backend, cuda_available):
                    continue
                print(f"running {case.name} {mode} {backend.label}", flush=True)
                try:
                    _, grads = run_backend(case, mode, backend, config)
                except Exception as exc:  # noqa: BLE001 - keep saving later plots
                    rows.append(
                        {
                            "solver": case.name,
                            "mode": mode,
                            "backend": backend.label,
                            "gradient": "",
                            "plot_kind": "ERROR",
                            "min": "",
                            "max": "",
                            "max_abs": "",
                            "p2": "",
                            "p98": "",
                            "path": "",
                            "error": one_line_error(exc),
                        }
                    )
                    print(f"failed {case.name} {mode} {backend.label}: {one_line_error(exc)}", flush=True)
                    continue
                for grad_name, grad_tensor in grads.items():
                    rel_path = (
                        Path(safe_name(case.name))
                        / safe_name(mode)
                        / safe_name(backend.label)
                        / f"{safe_name(grad_name)}.png"
                    )
                    title = f"{case.name} | {mode} | {backend.label} | grad:{grad_name}"
                    kind, min_value, max_value, max_abs, p2, p98 = save_gradient(
                        tensor_to_numpy(grad_tensor),
                        out_dir / rel_path,
                        title,
                    )
                    rows.append(
                        {
                            "solver": case.name,
                            "mode": mode,
                            "backend": backend.label,
                            "gradient": grad_name,
                            "plot_kind": kind,
                            "min": f"{min_value:.9e}",
                            "max": f"{max_value:.9e}",
                            "max_abs": f"{max_abs:.9e}",
                            "p2": f"{p2:.9e}",
                            "p98": f"{p98:.9e}",
                            "path": str(rel_path),
                            "error": "",
                        }
                    )

    index_path = out_dir / "gradient_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["solver", "mode", "backend", "gradient", "plot_kind", "min", "max", "max_abs", "p2", "p98", "path", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_plots={len(rows)}")
    print(f"out_dir={out_dir}")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
