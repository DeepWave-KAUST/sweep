#!/usr/bin/env python3
"""Topography gradient consistency suite (eager vs CUDA).

Mirrors :file:`solver_gradient_mode_suite.py` but focused on the irregular
free-surface topography paths added to the CUDA backend:

* **Acoustic** image / vacuum-staircase (Mittet 2002) — forward + backward
* **Elastic** image method (Robertsson 1996, ``topo_method='image'``) —
  forward + backward
* **Elastic APM** (Cao & Chen 2018, ``topo_method='apm'``) — forward only
  (CUDA backward is a stub; gradients flow through eager autograd).

For each (solver × topo scenario × memory mode) the suite runs both eager
and CUDA propagators, computes gradients via autograd, and reports cosine
similarity + relative L2.  Acceptance defaults mirror the original suite
(``rel_l2 < 1.5``, ``cosine > 0.8``).

The model grid is the same canonical 48×56 used by the main suite so the
two scripts cover comparable physical regimes.

Run::

    python test/topo_gradient_mode_suite.py             # default scenarios
    python test/topo_gradient_mode_suite.py --solvers acoustic2d_topo
    python test/topo_gradient_mode_suite.py --modes full,bs_gpu
    python test/topo_gradient_mode_suite.py --topo gentle_hill,mountain
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "src" / "sweep").exists() and (candidate / "test").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root from this script path.")


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
if "--source-import" in sys.argv:
    sys.argv.remove("--source-import")
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from sweep.equations import Acoustic, Elastic  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopoSolver:
    key: str
    equation_cls: type
    elastic: bool
    topo_method: str        # 'image' or 'apm'
    forward_only: bool      # True for APM (CUDA backward is a stub)


@dataclass(frozen=True)
class TopoScenario:
    key: str
    description: str
    builder: callable       # nx -> 1-D int row array


# Topography builders (return a 1-D int64 array of length nx).
def _gentle_hill(nx: int) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32)
    h = 4.0 * np.exp(-((x - nx / 2) ** 2) / (2.0 * 8.0 ** 2))
    return (3 + h).round().astype(np.int64)


def _mountain(nx: int) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32)
    h = 8.0 * np.exp(-((x - nx * 0.4) ** 2) / (2.0 * 6.0 ** 2)) \
      + 5.0 * np.exp(-((x - nx * 0.7) ** 2) / (2.0 * 4.0 ** 2)) \
      + 1.5 * np.sin(2.0 * np.pi * x / 14.0)
    return (3 + h).round().astype(np.int64)


def _stairs(nx: int) -> np.ndarray:
    out = np.full(nx, 4, dtype=np.int64)
    out[nx // 3 : 2 * nx // 3] = 8
    out[2 * nx // 3 :] = 6
    return out


SOLVERS = {
    "acoustic2d_topo": TopoSolver(
        "acoustic2d_topo", Acoustic, False, "image", False,
    ),
    "elastic2d_topo_image": TopoSolver(
        "elastic2d_topo_image", Elastic, True, "image", False,
    ),
    "elastic2d_topo_apm": TopoSolver(
        "elastic2d_topo_apm", Elastic, True, "apm", False,  # gradient enabled
    ),
}

SCENARIOS = {
    "gentle_hill": TopoScenario("gentle_hill", "single Gaussian, max ~4 rows", _gentle_hill),
    "mountain":    TopoScenario("mountain", "two Gaussians + sine ripple, max ~10 rows", _mountain),
    "stairs":      TopoScenario("stairs", "discrete-step topography (worst case for staircase)", _stairs),
}

DEFAULT_SOLVERS = "acoustic2d_topo,elastic2d_topo_image,elastic2d_topo_apm"
DEFAULT_SCENARIOS = "gentle_hill,mountain,stairs"
DEFAULT_MODES = "full,bs_gpu,bs_cpu"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def depth_ramp(shape, top, bottom):
    nz = shape[0]
    depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    ramp = top + (bottom - top) * depth
    return np.broadcast_to(ramp[:, None], shape).astype(np.float32).copy()


def add_box(arr, value):
    out = arr.copy()
    nz, nx = out.shape
    out[nz // 3 : (2 * nz) // 3, nx // 4 : (3 * nx) // 4] += value
    return out


def make_models(solver: TopoSolver, shape):
    vp_init = depth_ramp(shape, 1800.0, 2400.0)
    vp_true = add_box(vp_init, 180.0)
    if not solver.elastic:
        return [vp_true], [vp_init], [True]
    vs_init = (vp_init / 1.73).astype(np.float32)
    vs_true = (vp_true / 1.73).astype(np.float32)
    rho_init = depth_ramp(shape, 1000.0, 1200.0)
    rho_true = add_box(rho_init, 60.0)
    return [vp_true, vs_true, rho_true], [vp_init, vs_init, rho_init], [True, True, True]


def make_geometry(shape, topo, args):
    """Source at ~src_z above the per-column surface;
    receivers 1 cell below the local surface, every receiver_stride columns."""
    nz, nx = shape
    src_x = nx // 2
    src_z = int(topo[src_x]) + args.src_depth
    sources = np.array([[src_x, src_z]], dtype=np.int32)
    rec_x = np.arange(args.margin, nx - args.margin, args.receiver_stride, dtype=np.int32)
    rec_z = (topo[rec_x] + 1).astype(np.int32)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
    return sources, receivers


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


# ---------------------------------------------------------------------------
# Propagator construction
# ---------------------------------------------------------------------------


def build_propagator(solver: TopoSolver, backend_impl: str, mode: str, topo, shape, args, device):
    """backend_impl ∈ {'eager', 'c'};  mode ∈ {'full', 'bs_gpu', 'bs_cpu'}."""
    eq = solver.equation_cls(spatial_order=args.spatial_order, device=device, backend="torch")
    kwargs = dict(
        shape=shape, dev=device,
        dh=(args.dh, args.dh),
        dt=args.dt,
        abcn=args.abcn,
        nt=args.nt,
        B=1,
        topography=topo,
        topo_method=solver.topo_method,
        allow_growth=True,
    )
    # Image method: free_surface auto-set to True by propagator's
    # _resolve_topo_method; APM auto-sets False.  Don't override.
    if backend_impl == "eager":
        from sweep.propagator.options import EagerOptions
        return PropTorch(
            eq, backend="torch", impl="eager",
            eager_options=EagerOptions(use_compile=False),
            use_ckpt=False, **kwargs,
        )
    if mode == "full":
        return PropTorch(
            eq, backend="torch", impl="c",
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
            **kwargs,
        )
    cfg = {"enabled": True}
    if mode == "bs_gpu":
        cfg["storage"] = "gpu"
    elif mode == "bs_cpu":
        cfg["storage"] = "cpu"; cfg["pinned_memory"] = False
    else:
        raise ValueError(f"unknown mode: {mode}")
    return PropTorch(eq, backend="torch", impl="c",
                     boundary_saving_config=cfg, **kwargs)


def tensors_from_models(models_np, grad_flags, device):
    out = []
    for arr, needs_grad in zip(models_np, grad_flags):
        t = torch.tensor(arr, device=device, dtype=torch.float32)
        if needs_grad:
            t.requires_grad_(True)
        out.append(t)
    return out


def call_solver(solver, prop, wavelet, sources, receivers, models):
    """eager wants torch sources/receivers; CUDA wants numpy."""
    if prop._backend_impl.__class__.__name__.startswith("_PropTorchEager") if hasattr(prop, "_backend_impl") else False:
        # eager
        srcs = torch.tensor(sources, device=models[0].device, dtype=torch.long)
        recs = torch.tensor(receivers, device=models[0].device, dtype=torch.long)
        return prop(wavelet, srcs, recs, models=models)
    # CUDA: numpy ok
    try:
        return prop(wavelet, sources, receivers, models=models)
    except TypeError:
        # fallback torch tensors
        srcs = torch.tensor(sources, device=models[0].device, dtype=torch.long)
        recs = torch.tensor(receivers, device=models[0].device, dtype=torch.long)
        return prop(wavelet, srcs, recs, models=models)


def run_forward(solver, prop, wavelet_t, sources, receivers, models_np, device):
    """Forward pass; no gradient.  Returns numpy record."""
    models = tensors_from_models(models_np, [False] * len(models_np), device)
    with torch.no_grad():
        record = call_solver(solver, prop, wavelet_t, sources, receivers, models)
    return record.detach().cpu().numpy().squeeze()


def run_forward_backward(solver, prop, wavelet_t, sources, receivers,
                          observed_np, models_np, grad_flags, model_names, device):
    """Forward + autograd backward.  Returns {grads: {name: cpu_tensor}}."""
    models = tensors_from_models(models_np, grad_flags, device)
    observed_t = torch.tensor(observed_np, device=device, dtype=torch.float32)
    record = call_solver(solver, prop, wavelet_t, sources, receivers, models)
    record_squeezed = record.squeeze()
    if observed_t.shape != record_squeezed.shape:
        # Best-effort broadcast (mostly removes the lone (1,) batch dim)
        record_squeezed = record_squeezed.reshape(observed_t.shape)
    loss = (record_squeezed - observed_t).pow(2).mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    grads = {}
    for name, t, gf in zip(model_names, models, grad_flags):
        if not gf: continue
        if t.grad is None:
            raise RuntimeError(f"{name} grad is None")
        if not torch.isfinite(t.grad).all():
            raise RuntimeError(f"{name} grad has NaN/Inf")
        grads[name] = t.grad.detach().cpu()
    return {"loss": float(loss.detach().cpu()),
            "record": record.detach().cpu(),
            "grads": grads}


def metric_pair(ref, cand):
    ref = ref.detach().cpu().to(torch.float64).reshape(-1)
    cand = cand.detach().cpu().to(torch.float64).reshape(-1)
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    if not bool(finite.any()):
        return {"rel_l2": math.inf, "cosine": math.nan, "diff_linf": math.inf}
    ref = ref[finite]; cand = cand[finite]
    diff = cand - ref
    ref_l2 = float(torch.linalg.vector_norm(ref))
    cand_l2 = float(torch.linalg.vector_norm(cand))
    diff_l2 = float(torch.linalg.vector_norm(diff))
    denom = max(ref_l2, 1e-30)
    cosine = math.nan if ref_l2 <= 0 or cand_l2 <= 0 else \
        float(torch.dot(ref, cand) / (ref_l2 * cand_l2))
    return {"rel_l2": diff_l2 / denom, "cosine": cosine,
            "diff_linf": float(diff.abs().max()),
            "ref_l2": ref_l2, "cand_l2": cand_l2}


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_csv(value, valid, label):
    if value == "all": return list(valid)
    out = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in out if x not in valid]
    if bad: raise ValueError(f"Unknown {label}: {bad}; valid: {sorted(valid)}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--solvers", default=DEFAULT_SOLVERS)
    p.add_argument("--topo", default=DEFAULT_SCENARIOS, help="comma list of topography scenarios")
    p.add_argument("--modes", default=DEFAULT_MODES)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--nx", type=int, default=56)
    p.add_argument("--nt", type=int, default=120)
    p.add_argument("--dt", type=float, default=0.0015)
    p.add_argument("--dh", type=float, default=10.0)
    p.add_argument("--freq", type=float, default=10.0)
    p.add_argument("--delay", type=float, default=0.06)
    p.add_argument("--spatial-order", type=int, default=4)
    p.add_argument("--abcn", type=int, default=30)
    p.add_argument("--receiver-stride", type=int, default=6)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--src-depth", type=int, default=2, help="cells below local surface for source")
    p.add_argument("--rel-l2-threshold", type=float, default=1.5)
    p.add_argument("--cosine-threshold", type=float, default=0.8)
    p.add_argument("--fwd-rel-threshold", type=float, default=1e-3,
                   help="forward-only solvers (APM): max relative record diff")
    p.add_argument("--no-fail", action="store_true",
                   help="always exit 0 (collect all results)")
    return p


def status_for_grad(grad_metrics, args):
    fails = []
    for name, m in grad_metrics.items():
        if not math.isfinite(m["cosine"]) or m["cosine"] < args.cosine_threshold:
            fails.append(f"{name}.cosine={m['cosine']:.4g}<{args.cosine_threshold}")
        if not math.isfinite(m["rel_l2"]) or m["rel_l2"] > args.rel_l2_threshold:
            fails.append(f"{name}.rel_l2={m['rel_l2']:.4g}>{args.rel_l2_threshold}")
    return ("FAIL" if fails else "OK"), fails


def status_for_fwd(rec_metric, args):
    fails = []
    if rec_metric["rel_l2"] > args.fwd_rel_threshold:
        fails.append(f"record.rel_l2={rec_metric['rel_l2']:.4g}>{args.fwd_rel_threshold}")
    return ("FAIL" if fails else "OK"), fails


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr); sys.exit(2)

    solver_keys = parse_csv(args.solvers, SOLVERS.keys(), "solvers")
    scenario_keys = parse_csv(args.topo, SCENARIOS.keys(), "topo")
    mode_keys = parse_csv(args.modes, ("full", "bs_gpu", "bs_cpu"), "modes")

    shape = (args.nz, args.nx)
    wavelet_np = ricker(args.nt, args.dt, args.freq, args.delay)
    wavelet_t = torch.tensor(wavelet_np, device=device)

    rows = []
    print(f"\nTopography gradient consistency suite — grid {shape}, "
          f"NT={args.nt}, dt={args.dt}, abcn={args.abcn}\n")
    print(f"{'solver':24s} {'topo':14s} {'mode':10s}  cosine_grad / rel_l2 / fwd_rel        status")
    print("-" * 110)
    n_fail = 0

    for sk in solver_keys:
        solver = SOLVERS[sk]
        true_models, init_models, grad_flags = make_models(solver, shape)
        model_names = ("vp", "vs", "rho") if solver.elastic else ("vp",)
        for ck in scenario_keys:
            scenario = SCENARIOS[ck]
            topo = scenario.builder(args.nx)
            sources, receivers = make_geometry(shape, topo, args)

            # Reference (eager) — one observed record
            try:
                eager_prop = build_propagator(solver, "eager", "full", topo, shape, args, device)
                observed_np = run_forward(solver, eager_prop, wavelet_t, sources, receivers,
                                          true_models, device)
                if solver.forward_only:
                    eager_rec = observed_np
                else:
                    eager_out = run_forward_backward(
                        solver, eager_prop, wavelet_t, sources, receivers, observed_np,
                        init_models, grad_flags, model_names, device,
                    )
            except Exception as exc:
                rows.append((sk, ck, "eager", str(exc)))
                n_fail += 1
                print(f"{sk:24s} {ck:14s} eager       ERROR: {exc}")
                continue

            for mk in mode_keys:
                try:
                    cuda_prop = build_propagator(solver, "c", mk, topo, shape, args, device)
                    if solver.forward_only:
                        cuda_rec_np = run_forward(solver, cuda_prop, wavelet_t, sources,
                                                  receivers, true_models, device)
                        rec_metric = metric_pair(
                            torch.tensor(eager_rec, dtype=torch.float32),
                            torch.tensor(cuda_rec_np, dtype=torch.float32),
                        )
                        st, fails = status_for_fwd(rec_metric, args)
                        line = (f"{sk:24s} {ck:14s} {mk:10s}  fwd rel={rec_metric['rel_l2']:.3e}  "
                                f"cos={rec_metric['cosine']:.4f}  {st}")
                    else:
                        cuda_out = run_forward_backward(
                            solver, cuda_prop, wavelet_t, sources, receivers, observed_np,
                            init_models, grad_flags, model_names, device,
                        )
                        grad_metrics = {
                            name: metric_pair(eager_out["grads"][name], cuda_out["grads"][name])
                            for name in eager_out["grads"]
                        }
                        st, fails = status_for_grad(grad_metrics, args)
                        cos = " ".join(f"{n}={m['cosine']:.4f}" for n, m in grad_metrics.items())
                        rel = " ".join(f"{n}={m['rel_l2']:.3e}" for n, m in grad_metrics.items())
                        line = f"{sk:24s} {ck:14s} {mk:10s}  cos: {cos:30s} rel: {rel:20s}  {st}"

                    if st == "FAIL":
                        n_fail += 1
                        line += "  [" + "; ".join(fails) + "]"
                    print(line)
                except NotImplementedError as exc:
                    print(f"{sk:24s} {ck:14s} {mk:10s}  SKIP (not implemented): {exc}")
                except Exception as exc:
                    n_fail += 1
                    print(f"{sk:24s} {ck:14s} {mk:10s}  ERROR: {type(exc).__name__}: {exc}")

    print("-" * 110)
    if n_fail and not args.no_fail:
        print(f"FAILED: {n_fail} cases")
        sys.exit(1)
    print(f"DONE ({n_fail} failures)" if n_fail else "ALL PASSED")


if __name__ == "__main__":
    main()
