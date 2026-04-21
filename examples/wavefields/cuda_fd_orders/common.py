from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


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

from sweep.propagator.cuda import PropCUDA
from sweep.signal import ricker


ORDERS = (2, 6, 10, 14)
ABCN = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-enabled PyTorch environment.")
    return torch.device("cuda")


def make_wavelet(nt, dt, dom_freq, delay=0.12, amplitude=1e6):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    return amplitude * ricker(t, f=dom_freq).astype(np.float32)

def make_geometry(shape):
    nz, nx = shape
    source = np.array([[nx // 2, nz // 2]], dtype=np.int64)
    receivers = source[:, None, :].copy()
    return source, receivers


def crop_panel(panel, shape, spatial_order):
    halo = ABCN + spatial_order // 2
    nz, nx = shape
    return panel[halo : halo + nz, halo : halo + nx]


def extract_saved_wavefield(record):
    grad_fn = getattr(record, "grad_fn", None)
    if grad_fn is None or not hasattr(grad_fn, "saved_tensors"):
        raise RuntimeError(
            "CUDA wavefield history was not saved. Make sure at least one model tensor "
            "requires gradients and checkpointing/boundary saving are disabled."
        )
    saved = grad_fn.saved_tensors
    if not saved:
        raise RuntimeError("No saved CUDA tensors were found on the custom autograd node.")
    return saved[0]


def extent_meters(shape, dh):
    nz, nx = shape
    return (0.0, (nx - 1) * dh, (nz - 1) * dh, 0.0)


def clip_value(panels, percentile=99.5):
    amp = np.percentile(np.abs(np.stack(panels)), percentile)
    return max(float(amp), 1e-6)


def plot_order_grid(panels, titles, out_path, figure_title, shape, dh):
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.5), squeeze=False)
    vmax = clip_value(panels)
    ext = extent_meters(shape, dh)

    for ax, panel, title in zip(axes[0], panels, titles):
        ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax, extent=ext)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

