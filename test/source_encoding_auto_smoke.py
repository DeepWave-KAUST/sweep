#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "test" / "test_outputs" / "source_encoding_auto_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def build_problem():
    nz, nx = 41, 51
    nt = 180
    dt = 0.0015
    dh = 10.0
    nsrc = 3

    vp = np.full((nz, nx), 1800.0, dtype=np.float32)
    vp[nz // 2 :, :] = 2200.0
    vp[nz // 3 : (2 * nz) // 3, nx // 3 : (2 * nx) // 3] += 120.0

    base_wavelet = ricker(nt=nt, dt=dt, freq=10.0, delay=0.08)
    scales = np.array([1.0, -0.7, 0.45], dtype=np.float32)
    encoded_wavelet = (scales[:, None] * base_wavelet[None, :])[None, ...]

    src_x = np.array([nx // 4, nx // 2, (3 * nx) // 4], dtype=np.int32)
    src_z = np.full_like(src_x, 2)
    sources = np.stack([src_x, src_z], axis=-1)[None, ...]

    rec_x = np.arange(4, nx - 4, 2, dtype=np.int32)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]

    return {
        "shape": (nz, nx),
        "nt": nt,
        "dt": dt,
        "dh": dh,
        "vp": vp,
        "wavelet": encoded_wavelet,
        "sources": sources,
        "receivers": receivers,
    }


def normalize_record(record):
    arr = np.asarray(record)
    if arr.ndim == 4:
        return arr[0, :, :, 0]
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"Unsupported record shape {arr.shape}")


def save_record_figure(label, explicit_record, auto_record):
    explicit = normalize_record(explicit_record)
    auto = normalize_record(auto_record)
    vmax = np.percentile(np.abs(np.concatenate([explicit.ravel(), auto.ravel()])), 99.0)
    vmax = max(float(vmax), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    panels = [
        ("Explicit", explicit),
        ("Auto", auto),
        ("Difference", auto - explicit),
    ]
    for ax, (title, panel) in zip(axes[0], panels):
        local_vmax = vmax if title != "Difference" else max(float(np.max(np.abs(panel))), 1e-6)
        im = ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-local_vmax, vmax=local_vmax)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")
        ax.set_title(f"{label}: {title}")
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Time Sample")
    fig.tight_layout()
    output = OUTPUT_DIR / f"{label}_records.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def build_torch_solver(backend, device, shape, dt, dh):
    equation = Acoustic(spatial_order=4, device=device, backend="torch")
    return PropTorch(
        equation,
        shape=shape,
        dt=dt,
        dh=dh,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=device,
        backend=backend,
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
        pml_type='cpmlr',
    )


def run_torch_case(backend, problem):
    if backend == "cuda" and not torch.cuda.is_available():
        print("cuda: skipped (CUDA not available)")
        return None

    device = torch.device("cuda" if backend == "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    solver = build_torch_solver(backend, device, problem["shape"], problem["dt"], problem["dh"])

    def forward_and_grad(enable_source_encoding):
        vp = torch.tensor(problem["vp"], device=device, dtype=torch.float32, requires_grad=True)
        record = solver(
            wavelet=problem["wavelet"],
            sources=problem["sources"],
            receivers=problem["receivers"],
            models=[vp],
            source_encoding=enable_source_encoding,
        )
        loss = record.square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        grad = vp.grad.detach().cpu().numpy()
        return record.detach().cpu().numpy(), grad, float(loss.item())

    explicit_record, explicit_grad, explicit_loss = forward_and_grad(True)
    auto_record, auto_grad, auto_loss = forward_and_grad(False)

    record_path = save_record_figure(f"torch_{backend}", explicit_record, auto_record)
    record_diff = np.linalg.norm(normalize_record(auto_record) - normalize_record(explicit_record))
    grad_diff = np.linalg.norm(auto_grad - explicit_grad)
    grad_base = max(np.linalg.norm(explicit_grad), 1e-8)
    grad_rel = grad_diff / grad_base

    print(f"[torch:{backend}] explicit_loss={explicit_loss:.6e}")
    print(f"[torch:{backend}] auto_loss={auto_loss:.6e}")
    print(f"[torch:{backend}] record_diff_l2={record_diff:.6e}")
    print(f"[torch:{backend}] grad_rel_l2={grad_rel:.6e}")
    print(f"[torch:{backend}] record_figure={record_path}")

    assert np.allclose(auto_record, explicit_record, rtol=1e-5, atol=1e-6), f"{backend} record mismatch"
    assert np.allclose(auto_grad, explicit_grad, rtol=1e-4, atol=1e-6), f"{backend} gradient mismatch"
    return {
        "backend": f"torch:{backend}",
        "record_figure": record_path,
        "explicit_loss": explicit_loss,
        "auto_loss": auto_loss,
        "grad_rel_l2": grad_rel,
    }


def maybe_run_jax_case(problem):
    try:
        import jax
        import jax.numpy as jnp
        from sweep.propagator.jax import PropJax
    except Exception as exc:
        print(f"jax: skipped ({exc})")
        return None

    equation = Acoustic(spatial_order=4, backend="jax")
    solver = PropJax(
        equation,
        shape=problem["shape"],
        dt=problem["dt"],
        dh=problem["dh"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=None,
        use_ckpt=False,
    )

    wavelet = jnp.asarray(problem["wavelet"], dtype=jnp.float32)
    sources = jnp.asarray(problem["sources"], dtype=jnp.int32)
    receivers = jnp.asarray(problem["receivers"], dtype=jnp.int32)

    def run(enable_source_encoding):
        def loss_fn(vp):
            record = solver(
                wavelet=wavelet,
                sources=sources,
                receivers=receivers,
                models=[vp],
                source_encoding=enable_source_encoding,
            )
            return jnp.mean(record ** 2), record

        vp = jnp.asarray(problem["vp"], dtype=jnp.float32)
        (loss, record), grad = jax.value_and_grad(loss_fn, has_aux=True)(vp)
        return np.asarray(record), np.asarray(grad), float(loss)

    explicit_record, explicit_grad, explicit_loss = run(True)
    auto_record, auto_grad, auto_loss = run(False)
    record_path = save_record_figure("jax", explicit_record, auto_record)

    record_diff = np.linalg.norm(normalize_record(auto_record) - normalize_record(explicit_record))
    grad_diff = np.linalg.norm(auto_grad - explicit_grad)
    grad_base = max(np.linalg.norm(explicit_grad), 1e-8)
    grad_rel = grad_diff / grad_base

    print(f"[jax] explicit_loss={explicit_loss:.6e}")
    print(f"[jax] auto_loss={auto_loss:.6e}")
    print(f"[jax] record_diff_l2={record_diff:.6e}")
    print(f"[jax] grad_rel_l2={grad_rel:.6e}")
    print(f"[jax] record_figure={record_path}")

    assert np.allclose(auto_record, explicit_record, rtol=1e-5, atol=1e-6), "jax record mismatch"
    assert np.allclose(auto_grad, explicit_grad, rtol=1e-4, atol=1e-6), "jax gradient mismatch"
    return {
        "backend": "jax",
        "record_figure": record_path,
        "explicit_loss": explicit_loss,
        "auto_loss": auto_loss,
        "grad_rel_l2": grad_rel,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke test for automatic source-encoding detection.")
    parser.add_argument(
        "--backend",
        choices=["torch-eager", "torch-cuda", "jax", "all"],
        default="all",
        help="Which backend variant to run.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    print(f"python={sys.executable}")
    print(f"output_dir={OUTPUT_DIR}")

    problem = build_problem()
    results = []

    if args.backend in ("torch-eager", "all"):
        results.append(run_torch_case("eager", problem))
    if args.backend in ("torch-cuda", "all"):
        results.append(run_torch_case("cuda", problem))
    if args.backend in ("jax", "all"):
        results.append(maybe_run_jax_case(problem))

    results = [result for result in results if result is not None]
    if not results:
        raise RuntimeError("No backend run completed.")

    print("source-encoding auto smoke passed.")


if __name__ == "__main__":
    main()
