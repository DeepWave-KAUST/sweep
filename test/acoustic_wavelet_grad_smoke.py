import argparse
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch

torch.backends.cudnn.benchmark = True


def _ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def _build_problem_2d():
    nz, nx = 41, 41
    nt = 200
    dt = 0.0015
    dh = 10.0

    vp = np.full((nz, nx), 1800.0, dtype=np.float32)
    vp[nz // 2 :, :] = 2200.0

    wavelet = _ricker(nt=nt, dt=dt, freq=12.0, delay=0.1)
    sources = np.array([[nx // 2, 2]], dtype=np.int32)

    rec_x = np.arange(6, nx - 6, 4, dtype=np.int32)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]

    return {
        "dim": "2d",
        "name": "acoustic2d",
        "equation": Acoustic,
        "shape": (nz, nx),
        "vp": vp,
        "wavelet": wavelet,
        "sources": sources,
        "receivers": receivers,
        "dt": dt,
        "dh": (dh, dh),
    }


def _build_problem_3d():
    nz, ny, nx = 31, 31, 31
    nt = 160
    dt = 0.0015
    dh = 10.0

    vp = np.full((nz, ny, nx), 1800.0, dtype=np.float32)
    vp[nz // 2 :, :, :] = 2200.0
    vp[nz // 3 : (2 * nz) // 3, ny // 4 : (3 * ny) // 4, nx // 4 : (3 * nx) // 4] += 120.0

    wavelet = _ricker(nt=nt, dt=dt, freq=10.0, delay=0.1)
    sources = np.array([[nx // 2, ny // 2, 2]], dtype=np.int32)

    rec_x, rec_y = np.meshgrid(
        np.arange(6, nx - 6, 6, dtype=np.int32),
        np.arange(6, ny - 6, 6, dtype=np.int32),
        indexing="xy",
    )
    rec_z = np.full(rec_x.size, 2, dtype=np.int32)
    receivers = np.stack([rec_x.reshape(-1), rec_y.reshape(-1), rec_z], axis=-1)[None, ...]

    return {
        "dim": "3d",
        "name": "acoustic3d",
        "equation": Acoustic3D,
        "shape": (nz, ny, nx),
        "vp": vp,
        "wavelet": wavelet,
        "sources": sources,
        "receivers": receivers,
        "dt": dt,
        "dh": (dh, dh, dh),
    }


def _run_wavelet_grad(solver, wavelet, vp_np, sources, receivers, device):
    solver_name = solver.__class__.__name__
    print(f"  [{solver_name}] building tensors")
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32)

    print(f"  [{solver_name}] forward")
    record = solver(
        wavelet=wavelet,
        sources=sources,
        receivers=receivers,
        models=[vp],
    )
    loss = record.square().mean()
    print(f"  [{solver_name}] backward")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    grad = wavelet.grad
    assert grad is not None
    grad = grad.detach().cpu()
    assert torch.isfinite(grad).all()
    return grad


def _save_wavelet_grad_figure(case_name, mode, wavelet, cuda_grad, torch_grad):
    wavelet_np = np.asarray(wavelet, dtype=np.float32)
    cuda_grad_np = cuda_grad.numpy()
    torch_grad_np = torch_grad.numpy()
    diff_grad_np = cuda_grad_np - torch_grad_np
    time_idx = np.arange(wavelet_np.shape[-1], dtype=np.int32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), squeeze=False)
    axes = axes[0]

    axes[0].plot(time_idx, wavelet_np, color="black", linewidth=1.5)
    axes[0].set_title("Initial Wavelet")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_idx, cuda_grad_np, color="tab:red", linewidth=1.5)
    axes[1].set_title("CUDA Wavelet Grad")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Gradient")
    axes[1].grid(alpha=0.3)

    axes[2].plot(time_idx, torch_grad_np, color="tab:blue", linewidth=1.5)
    axes[2].set_title("Torch Wavelet Grad")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Gradient")
    axes[2].grid(alpha=0.3)

    axes[3].plot(time_idx, diff_grad_np, color="tab:green", linewidth=1.5)
    axes[3].set_title("CUDA - Torch")
    axes[3].set_xlabel("Sample")
    axes[3].set_ylabel("Gradient Difference")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"{case_name} Wavelet Gradient Smoke: {mode}")
    fig.tight_layout()
    output_path = f"{case_name}_wavelet_grad_{mode}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_case(problem, mode, boundary_cfg):
    device = torch.device("cuda")
    print(f"\nRunning {problem['name']} mode={mode}")

    common_kwargs = dict(
        shape=problem["shape"],
        dt=problem["dt"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=device,
        pml_type="cpmlr",
    )

    equation = problem["equation"](spatial_order=4, device=device)
    cuda_solver = PropCUDA(
        equation,
        dh=problem["dh"],
        **common_kwargs,
        boundary_saving_config=boundary_cfg,
    )
    torch_solver = PropTorch(
        equation,
        dh=problem["dh"][0] if isinstance(problem["dh"], tuple) else problem["dh"],
        **common_kwargs,
    )

    print(" CUDA gradient run")
    cuda_grad = _run_wavelet_grad(
        cuda_solver,
        torch.tensor(problem["wavelet"], device=device, dtype=torch.float32, requires_grad=True),
        problem["vp"],
        problem["sources"],
        problem["receivers"],
        device,
    )
    print(" Torch gradient run")
    torch_grad = _run_wavelet_grad(
        torch_solver,
        torch.tensor(problem["wavelet"], device=device, dtype=torch.float32, requires_grad=True),
        problem["vp"],
        problem["sources"],
        problem["receivers"],
        device,
    )

    denom = torch.linalg.norm(torch_grad).clamp_min(1e-8)
    rel_l2 = torch.linalg.norm(cuda_grad - torch_grad) / denom
    cosine = torch.nn.functional.cosine_similarity(
        cuda_grad.reshape(1, -1),
        torch_grad.reshape(1, -1),
    ).item()
    diff = cuda_grad - torch_grad
    diff_l2 = torch.linalg.norm(diff).item()
    diff_linf = torch.max(torch.abs(diff)).item()

    output_path = _save_wavelet_grad_figure(problem["name"], mode, problem["wavelet"], cuda_grad, torch_grad)

    assert rel_l2.item() < 0.15, (
        f"{problem['name']} {mode} wavelet grad relative L2 mismatch: {rel_l2.item():.4f}"
    )
    assert cosine > 0.99, f"{problem['name']} {mode} wavelet grad cosine similarity too low: {cosine:.4f}"
    return {
        "case": problem["name"],
        "mode": mode,
        "rel_l2": rel_l2.item(),
        "cosine": cosine,
        "diff_l2": diff_l2,
        "diff_linf": diff_linf,
        "figure": output_path,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Acoustic 2D/3D CUDA wavelet gradient smoke test.")
    parser.add_argument(
        "--case",
        choices=["2d", "3d", "all"],
        default="all",
        help="Which acoustic wave equation case to run.",
    )
    return parser


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device.")

    import sweep
    import sweep._C as sweep_c

    args = _build_parser().parse_args()

    print(f"python={sys.executable}")
    print(f"sweep={sweep.__file__}")
    print(f"sweep._C={sweep_c.__file__}")

    cases = []
    if args.case in ("2d", "all"):
        cases.append(_build_problem_2d())
    if args.case in ("3d", "all"):
        cases.append(_build_problem_3d())

    modes = [
        ("full", None),
        ("boundary_gpu", {"enabled": True, "storage": "gpu", "transfer_interval": 2}),
    ]

    print("Running acoustic wavelet gradient smoke...")
    for problem in cases:
        for mode, boundary_cfg in modes:
            result = _run_case(problem, mode, boundary_cfg)
            print(
                f"[{result['case']}:{result['mode']}] rel_l2={result['rel_l2']:.6f}, "
                f"cosine={result['cosine']:.6f}, "
                f"diff_l2={result['diff_l2']:.6e}, "
                f"diff_linf={result['diff_linf']:.6e}, "
                f"figure={result['figure']}"
            )


if __name__ == "__main__":
    main()
