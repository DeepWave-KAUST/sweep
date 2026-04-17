import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHED_FILES = [
    "src/sweep/equations/acoustic.py",
    "src/sweep/equations/acoustic_vrz.py",
    "src/sweep/operators/general.py",
    "src/sweep/operators/torch.py",
    "src/sweep/propagator/base.py",
    "src/sweep/propagator/torch.py",
    "src/sweep/receivers/torch.py",
    "src/sweep/sources/torch.py",
]


RUNNER = r"""
import json
import os
import time
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch


def ricker(t, fm):
    pi2 = np.pi * 2.0
    return (1.0 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)


def get_config():
    return {
        "nz": int(os.environ["SWEEP_BENCH_NZ"]),
        "nx": int(os.environ["SWEEP_BENCH_NX"]),
        "nt": int(os.environ["SWEEP_BENCH_NT"]),
        "dt": float(os.environ["SWEEP_BENCH_DT"]),
        "dh": float(os.environ["SWEEP_BENCH_DH"]),
        "abcn": int(os.environ["SWEEP_BENCH_ABCN"]),
        "nshots": int(os.environ["SWEEP_BENCH_NSHOTS"]),
        "nreceivers": int(os.environ["SWEEP_BENCH_NRECEIVERS"]),
        "fm": float(os.environ["SWEEP_BENCH_FM"]),
        "delay": float(os.environ["SWEEP_BENCH_DELAY"]),
        "spatial_order": int(os.environ["SWEEP_BENCH_SPATIAL_ORDER"]),
        "src_z": int(os.environ["SWEEP_BENCH_SRC_Z"]),
        "rec_z": int(os.environ["SWEEP_BENCH_REC_Z"]),
        "repeats": int(os.environ["SWEEP_BENCH_REPEATS"]),
        "warmup": int(os.environ["SWEEP_BENCH_WARMUP"]),
        "device": os.environ["SWEEP_BENCH_DEVICE"],
        "use_compile": os.environ.get("SWEEP_BENCH_USE_COMPILE", "0") == "1",
        "compile_mode": os.environ.get("SWEEP_BENCH_COMPILE_MODE", "default"),
    }


def make_case(device, cfg):
    nz, nx = cfg["nz"], cfg["nx"]
    nt = cfg["nt"]
    dt = cfg["dt"]
    dh = cfg["dh"]
    abcn = cfg["abcn"]
    nshots = cfg["nshots"]
    nreceivers = cfg["nreceivers"]
    vp_np = np.full((nz, nx), 1800.0, dtype=np.float32)
    vp_np[nz // 2 :, :] = 2400.0
    vp_np[nz // 3 : (2 * nz) // 3, nx // 4 : (3 * nx) // 4] += 150.0

    t = np.arange(nt, dtype=np.float32) * dt - cfg["delay"]
    wave = ricker(t, fm=cfg["fm"]).astype(np.float32)

    src_x = np.linspace(8, nx - 9, nshots, dtype=np.int64)
    src_z = np.full(nshots, cfg["src_z"], dtype=np.int64)
    sources = np.stack([src_x, src_z], axis=1)

    rec_x = np.linspace(0, nx - 1, nreceivers, dtype=np.int64)
    rec_z = np.full(nreceivers, cfg["rec_z"], dtype=np.int64)
    receiver_template = np.stack([rec_x, rec_z], axis=1)
    receivers = np.repeat(receiver_template[None, :, :], nshots, axis=0)

    model = PropTorch(
        Acoustic(spatial_order=cfg["spatial_order"], device=device, backend="torch"),
        shape=(nz, nx),
        dev=device,
        dh=dh,
        dt=dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=False,
        use_ckpt=False,
        pml_type="cpmlr",
        use_compile=cfg["use_compile"],
        compile_mode=cfg["compile_mode"],
    )
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32, requires_grad=True)
    return model, vp, wave, sources, receivers


def measure(device, fn, repeats, warmup):
    timings = []
    for _ in range(warmup):
        fn()
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    timings = np.asarray(timings, dtype=np.float64)
    return {
        "mean_s": float(timings.mean()),
        "std_s": float(timings.std(ddof=0)),
        "min_s": float(timings.min()),
        "max_s": float(timings.max()),
    }


def main():
    torch.backends.cudnn.benchmark = True
    cfg = get_config()
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    repeats = cfg["repeats"]
    warmup = cfg["warmup"]

    model, vp, wave, sources, receivers = make_case(device, cfg)

    def run_forward():
        with torch.no_grad():
            out = model(wave, sources, receivers, models=[vp])
            if device.type != "cuda":
                _ = float(out.abs().sum())

    def run_backward():
        if vp.grad is not None:
            vp.grad = None
        out = model(wave, sources, receivers, models=[vp])
        loss = out.square().mean()
        loss.backward()
        if device.type != "cuda":
            _ = float(vp.grad.abs().sum())

    result = {
        "config": cfg,
        "device": str(device),
        "torch": torch.__version__,
        "forward": measure(device, run_forward, repeats=repeats, warmup=warmup),
        "forward_backward": measure(device, run_backward, repeats=repeats, warmup=warmup),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark PropTorch speedup versus the git HEAD baseline."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--nshots", type=int, default=6)
    parser.add_argument("--nreceivers", type=int, default=96)
    parser.add_argument("--fm", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--spatial-order", type=int, default=4, dest="spatial_order")
    parser.add_argument("--src-z", type=int, default=4, dest="src_z")
    parser.add_argument("--rec-z", type=int, default=4, dest="rec_z")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--use-compile", action="store_true", dest="use_compile")
    parser.add_argument("--compile-mode", default="default")
    return parser


def copy_tree(src_root: Path, dst_root: Path) -> None:
    shutil.copytree(src_root, dst_root)


def restore_head_versions(dst_repo: Path) -> None:
    for rel_path in PATCHED_FILES:
        content = subprocess.check_output(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=REPO_ROOT,
            text=True,
        )
        target = dst_repo / rel_path
        target.write_text(content)


def run_case(src_dir: Path, args) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)
    env["SWEEP_BENCH_DEVICE"] = args.device
    env["SWEEP_BENCH_NZ"] = str(args.nz)
    env["SWEEP_BENCH_NX"] = str(args.nx)
    env["SWEEP_BENCH_NT"] = str(args.nt)
    env["SWEEP_BENCH_DT"] = str(args.dt)
    env["SWEEP_BENCH_DH"] = str(args.dh)
    env["SWEEP_BENCH_ABCN"] = str(args.abcn)
    env["SWEEP_BENCH_NSHOTS"] = str(args.nshots)
    env["SWEEP_BENCH_NRECEIVERS"] = str(args.nreceivers)
    env["SWEEP_BENCH_FM"] = str(args.fm)
    env["SWEEP_BENCH_DELAY"] = str(args.delay)
    env["SWEEP_BENCH_SPATIAL_ORDER"] = str(args.spatial_order)
    env["SWEEP_BENCH_SRC_Z"] = str(args.src_z)
    env["SWEEP_BENCH_REC_Z"] = str(args.rec_z)
    env["SWEEP_BENCH_REPEATS"] = str(args.repeats)
    env["SWEEP_BENCH_WARMUP"] = str(args.warmup)
    env["SWEEP_BENCH_USE_COMPILE"] = "1" if args.use_compile else "0"
    env["SWEEP_BENCH_COMPILE_MODE"] = args.compile_mode
    out = subprocess.check_output(
        [sys.executable, "-c", RUNNER],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    return json.loads(out)


def format_speedup(baseline_s: float, current_s: float) -> dict:
    speedup = baseline_s / current_s
    reduction = (baseline_s - current_s) / baseline_s * 100.0
    return {
        "baseline_s": baseline_s,
        "current_s": current_s,
        "speedup_x": speedup,
        "time_reduction_pct": reduction,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 8 if args.device in {"auto", "cuda"} else 4
    if args.warmup is None:
        args.warmup = 3 if args.device in {"auto", "cuda"} else 1

    with tempfile.TemporaryDirectory(prefix="sweep-bench-") as tmpdir:
        tmpdir = Path(tmpdir)
        current_root = tmpdir / "current"
        baseline_root = tmpdir / "baseline"

        copy_tree(REPO_ROOT / "src", current_root / "src")
        copy_tree(REPO_ROOT / "src", baseline_root / "src")
        restore_head_versions(baseline_root)

        current = run_case(current_root / "src", args)
        baseline = run_case(baseline_root / "src", args)

    summary = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "config": vars(args),
        "device": current["device"],
        "torch": current["torch"],
        "forward": format_speedup(baseline["forward"]["mean_s"], current["forward"]["mean_s"]),
        "forward_backward": format_speedup(
            baseline["forward_backward"]["mean_s"],
            current["forward_backward"]["mean_s"],
        ),
        "raw": {
            "baseline": baseline,
            "current": current,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
