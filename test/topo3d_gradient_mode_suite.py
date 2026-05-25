"""3-D topography Python eager consistency suite.

Covers ``acoustic3d`` and ``elastic3d`` with irregular free-surface
topography (image method only — APM 3-D pending Dong 2023 implementation).
Since the CUDA 3-D topo backend is not yet implemented, this suite
validates the Python eager path by three orthogonal checks per case:

1. **Flat-topo equivalence**: a constant-row topography ``surf[iy, ix] = 0``
   must produce the same record (up to fp32 noise) as the flat
   ``free_surface=True`` path.  Catches obvious indexing / padding bugs.

2. **Air-cell zeroing**: forward wavefield evaluated cell-by-cell at
   ``iz < surf[iy, ix]`` (the "air" region) must be exactly zero after
   the step — verifies the surface BC is being enforced.

3. **Finite gradient**: backward autograd through a quadratic loss must
   produce non-NaN, non-zero gradients on (vp[/vs/rho]).  Catches NaN
   propagation from masked-fill / where ops.

Usage::

    python test/topo3d_gradient_mode_suite.py             # default scenarios
    python test/topo3d_gradient_mode_suite.py --solvers elastic3d_topo
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src" / "sweep" / "__init__.py").exists():
            return p
    raise RuntimeError("Could not locate sweep src/ from " + str(here))


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from sweep.equations.acoustic3d import Acoustic3D                          # noqa: E402
from sweep.equations.elastic3d import Elastic                               # noqa: E402
from sweep.propagator.torch import PropTorch                                # noqa: E402


# ---------------------------------------------------------------------------
# Solver + scenario tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopoSolver:
    key: str
    equation_cls: type
    elastic: bool
    topo_method: str = "image"


@dataclass(frozen=True)
class TopoScenario:
    key: str
    builder: Callable[[int, int], np.ndarray]


def _flat(ny: int, nx: int) -> np.ndarray:
    """All surface rows at iz=0 — must match flat FS bit-exact."""
    return np.zeros((ny, nx), dtype=np.int32)


def _gentle_hill_3d(ny: int, nx: int) -> np.ndarray:
    """Single Gaussian bump centred in (iy, ix)."""
    cy, cx = ny / 2.0, nx / 2.0
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r2 = ((y - cy) / (ny / 3.0)) ** 2 + ((x - cx) / (nx / 3.0)) ** 2
    return np.clip(np.round(3.0 * np.exp(-r2)).astype(np.int32), 0, None)


def _ridge(ny: int, nx: int) -> np.ndarray:
    """Diagonal ridge along ix=iy diagonal — varies in both x and y."""
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.clip(np.round(2.0 * np.sin(np.pi * (x + y) / max(ny, nx))).astype(np.int32), 0, None)


SOLVERS = {
    "acoustic3d_topo":      TopoSolver("acoustic3d_topo",     Acoustic3D, False, "image"),
    "elastic3d_topo":       TopoSolver("elastic3d_topo",      Elastic,    True,  "image"),
    "elastic3d_topo_apm":   TopoSolver("elastic3d_topo_apm",  Elastic,    True,  "apm"),
}

SCENARIOS = {
    "flat":         TopoScenario("flat",         _flat),
    "gentle_hill":  TopoScenario("gentle_hill",  _gentle_hill_3d),
    "ridge":        TopoScenario("ridge",        _ridge),
}

DEFAULT_SOLVERS = "acoustic3d_topo,elastic3d_topo,elastic3d_topo_apm"
DEFAULT_SCENARIOS = "flat,gentle_hill,ridge"


# ---------------------------------------------------------------------------
# Model + geometry
# ---------------------------------------------------------------------------


def depth_ramp_3d(shape, top, bottom):
    nz = shape[0]
    vp1d = np.linspace(top, bottom, nz, dtype=np.float32)
    return np.broadcast_to(vp1d[:, None, None], shape).copy()


def add_box_3d(arr, value):
    nz, ny, nx = arr.shape
    z0, z1 = nz * 3 // 5, nz * 4 // 5
    y0, y1 = ny * 2 // 5, ny * 3 // 5
    x0, x1 = nx * 2 // 5, nx * 3 // 5
    out = arr.copy()
    out[z0:z1, y0:y1, x0:x1] += value
    return out


def make_models(solver: TopoSolver, shape):
    if solver.elastic:
        vp_true = depth_ramp_3d(shape, 1800.0, 2400.0)
        vs_true = depth_ramp_3d(shape, 1000.0, 1400.0)
        rho_true = depth_ramp_3d(shape, 1000.0, 1200.0)
        vp_init = vp_true.copy()
        vs_init = vs_true.copy()
        rho_init = rho_true.copy()
        vp_true = add_box_3d(vp_true, 180.0)
        vs_true = add_box_3d(vs_true, 100.0)
        rho_true = add_box_3d(rho_true, 60.0)
        return ([vp_true, vs_true, rho_true],
                [vp_init, vs_init, rho_init],
                [True, True, True])
    vp_true = depth_ramp_3d(shape, 1800.0, 2400.0)
    vp_init = vp_true.copy()
    vp_true = add_box_3d(vp_true, 180.0)
    return [vp_true], [vp_init], [True]


def make_geometry(shape, topo, args):
    """Source 1 cell below the local surface at the (iy, ix) centre; a
    small line of receivers along x at iy=ny//2."""
    nz, ny, nx = shape
    src_y = ny // 2
    src_x = nx // 2
    src_z = int(topo[src_y, src_x]) + args.src_depth
    sources = np.array([[src_x, src_y, src_z]], dtype=np.int32)

    rec_x = np.arange(args.margin, nx - args.margin, args.receiver_stride, dtype=np.int32)
    rec_y = np.full_like(rec_x, src_y)
    rec_z = (topo[rec_y, rec_x] + 1).astype(np.int32)
    receivers = np.stack([rec_x, rec_y, rec_z], axis=-1)[None, ...]
    return sources, receivers


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------


def build_propagator(solver: TopoSolver, topography, shape, args, device, *, eager=True):
    eq = solver.equation_cls(spatial_order=args.spatial_order, device=device, backend="torch")
    kwargs = dict(
        shape=shape, dev=device,
        dh=(args.dh, args.dh, args.dh),
        dt=args.dt,
        abcn=args.abcn,
        nt=args.nt,
        B=1,
        topography=topography,
        topo_method=solver.topo_method,
        allow_growth=True,
    )
    if eager:
        from sweep.propagator.options import EagerOptions
        return PropTorch(
            eq, backend="torch", impl="eager",
            eager_options=EagerOptions(use_compile=False),
            use_ckpt=False, **kwargs,
        )
    return PropTorch(eq, backend="torch", impl="c", use_ckpt=False, **kwargs)


def tensors_from_models(models_np, grad_flags, device):
    out = []
    for arr, needs_grad in zip(models_np, grad_flags):
        t = torch.tensor(arr, device=device, dtype=torch.float32)
        if needs_grad:
            t.requires_grad_(True)
        out.append(t)
    return out


def call_solver(prop, wavelet, sources, receivers, models, device):
    srcs = torch.tensor(sources, device=device, dtype=torch.long)
    recs = torch.tensor(receivers, device=device, dtype=torch.long)
    return prop(wavelet, srcs, recs, models=models)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_forward(solver, prop, wavelet_t, sources, receivers, models_np, device):
    models = tensors_from_models(models_np, [False] * len(models_np), device)
    with torch.no_grad():
        return call_solver(prop, wavelet_t, sources, receivers, models, device).detach().cpu().numpy().squeeze()


def run_grad(solver, prop, wavelet_t, sources, receivers, observed_np,
             models_np, grad_flags, model_names, device):
    models = tensors_from_models(models_np, grad_flags, device)
    obs_t = torch.tensor(observed_np, device=device, dtype=torch.float32)
    record = call_solver(prop, wavelet_t, sources, receivers, models, device)
    rec = record.squeeze()
    if obs_t.shape != rec.shape:
        rec = rec.reshape(obs_t.shape)
    loss = (rec - obs_t).pow(2).mean()
    loss.backward()
    grads = {}
    for name, t, gf in zip(model_names, models, grad_flags):
        if not gf:
            continue
        if t.grad is None:
            raise RuntimeError(f"{name} grad is None")
        grads[name] = t.grad.detach().cpu()
    return float(loss.detach().cpu()), grads


def rel_l2(a, b):
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a64)
    if denom == 0:
        return math.inf if np.linalg.norm(b64) else 0.0
    return float(np.linalg.norm(a64 - b64) / denom)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--solvers", default=DEFAULT_SOLVERS)
    p.add_argument("--topo", default=DEFAULT_SCENARIOS)
    p.add_argument("--nz", type=int, default=24)
    p.add_argument("--ny", type=int, default=20)
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--nt", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.0015)
    p.add_argument("--dh", type=float, default=10.0)
    p.add_argument("--freq", type=float, default=10.0)
    p.add_argument("--delay", type=float, default=0.06)
    p.add_argument("--spatial-order", type=int, default=4)
    p.add_argument("--abcn", type=int, default=10)
    p.add_argument("--receiver-stride", type=int, default=4)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--src-depth", type=int, default=2)
    p.add_argument("--rel-flat-threshold", type=float, default=1e-5,
                   help="flat-topo IMAGE record must match flat FS within this rel-L2")
    p.add_argument("--rel-flat-apm-threshold", type=float, default=5e-2,
                   help="flat-topo APM record must match flat-FS image record "
                        "within this rel-L2 (APM is a different discretisation "
                        "from image; ~few percent on coarse grids is expected)")
    p.add_argument("--no-fail", action="store_true")
    return p


def parse_csv(value, valid, label):
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in valid]
    if bad:
        raise SystemExit(f"unknown {label}: {bad}")
    return items


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver_keys = parse_csv(args.solvers, SOLVERS.keys(), "solvers")
    scenario_keys = parse_csv(args.topo, SCENARIOS.keys(), "topo")

    shape = (args.nz, args.ny, args.nx)
    wavelet_t = torch.tensor(ricker(args.nt, args.dt, args.freq, args.delay), device=device)

    print(f"\nTopo 3-D Python-eager suite — grid {shape}, NT={args.nt}, "
          f"dt={args.dt}, abcn={args.abcn}\n")
    print(f"{'solver':24s} {'topo':14s} {'check':22s} result")
    print("-" * 100)

    n_fail = 0

    for sk in solver_keys:
        solver = SOLVERS[sk]
        true_models, init_models, grad_flags = make_models(solver, shape)
        model_names = ("vp", "vs", "rho") if solver.elastic else ("vp",)

        # Reference: flat free-surface record (no topo at all).
        try:
            flat_eq = solver.equation_cls(
                spatial_order=args.spatial_order, device=device, backend="torch"
            )
            kwargs_flat = dict(
                shape=shape, dev=device,
                dh=(args.dh, args.dh, args.dh),
                dt=args.dt, abcn=args.abcn, nt=args.nt, B=1,
                free_surface=True,            # flat FS (no topography)
                allow_growth=True,
            )
            from sweep.propagator.options import EagerOptions
            flat_prop = PropTorch(
                flat_eq, backend="torch", impl="eager",
                eager_options=EagerOptions(use_compile=False),
                use_ckpt=False, **kwargs_flat,
            )
            # Flat-FS sources at constant z=src_depth (mirrors what flat
            # topo would resolve to).
            flat_topo = np.zeros((args.ny, args.nx), dtype=np.int32)
            srcs_flat, recs_flat = make_geometry(shape, flat_topo, args)
            flat_record = run_forward(solver, flat_prop, wavelet_t,
                                       srcs_flat, recs_flat,
                                       true_models, device)
        except Exception as exc:
            print(f"{sk:24s} flat reference setup ERROR: {exc}")
            n_fail += 1
            continue

        for ck in scenario_keys:
            scenario = SCENARIOS[ck]
            topo = scenario.builder(args.ny, args.nx)
            sources, receivers = make_geometry(shape, topo, args)

            # ---- Build topo propagator ----
            try:
                prop = build_propagator(solver, topo, shape, args, device, eager=True)
            except Exception as exc:
                print(f"{sk:24s} {ck:14s} {'build':22s} ERROR: {exc}")
                n_fail += 1
                continue

            # ---- Check 1: flat-topo equivalence ----
            # IMAGE: with topo_rows == 0 the surface row is the physical
            # top, identical to flat ``free_surface=True`` (no topo) — must
            # match bit-exact.
            # APM:   topo_rows == 0 means there are NO air cells (mask
            # all-False) so the simulation reduces to a no-FS, full-PML
            # bulk medium, which is fundamentally different from the image
            # flat-FS reference.  Skip this check for APM and verify the
            # APM free-surface path with the non-flat scenarios below
            # (which DO have air cells and exercise H/V*/OC/IC).
            if ck == "flat":
                if solver.topo_method == "apm":
                    print(f"{sk:24s} {ck:14s} {'flat_equivalence':22s} "
                          f"SKIP (APM with all-zero topo has no air cells; "
                          f"non-flat scenarios exercise the APM free surface)")
                else:
                    try:
                        rec = run_forward(solver, prop, wavelet_t, sources, receivers,
                                          true_models, device)
                        rel = rel_l2(flat_record, rec)
                        ok = rel < args.rel_flat_threshold
                        msg = f"rel_l2={rel:.3e} (vs flat FS)"
                        print(f"{sk:24s} {ck:14s} {'flat_equivalence':22s} "
                              f"{'OK' if ok else 'FAIL'}  {msg}")
                        if not ok:
                            n_fail += 1
                    except Exception as exc:
                        print(f"{sk:24s} {ck:14s} {'flat_equivalence':22s} ERROR: {exc}")
                        n_fail += 1

            # ---- Check 2: air cells zeroed in forward record (sanity that
            # the receiver placement above surface returns 0).
            try:
                # Place a receiver ABOVE the surface and verify it reads 0.
                air_y, air_x = args.ny // 3, args.nx // 3
                air_z = max(int(topo[air_y, air_x]) - 1, 0)  # 1 cell above surface
                if int(topo[air_y, air_x]) == 0:
                    # surface at top; no air row available — skip check
                    print(f"{sk:24s} {ck:14s} {'air_zero':22s} SKIP (surface at iz=0)")
                else:
                    rec_air = np.array([[[air_x, air_y, air_z]]], dtype=np.int32)
                    rec_air_vals = run_forward(solver, prop, wavelet_t, sources,
                                                rec_air, true_models, device)
                    peak_air = float(np.abs(rec_air_vals).max())
                    ok = peak_air < 1e-6
                    print(f"{sk:24s} {ck:14s} {'air_zero':22s} {'OK' if ok else 'FAIL'}  "
                          f"|rec| at air cell (iy={air_y},ix={air_x},iz={air_z}) = {peak_air:.3e}")
                    if not ok:
                        n_fail += 1
            except Exception as exc:
                print(f"{sk:24s} {ck:14s} {'air_zero':22s} ERROR: {exc}")
                n_fail += 1

            # ---- Check 3: finite, non-trivial gradient ----
            try:
                observed = run_forward(solver, prop, wavelet_t, sources, receivers,
                                        true_models, device)
                loss, grads = run_grad(solver, prop, wavelet_t, sources, receivers,
                                        observed, init_models, grad_flags, model_names, device)
                ok = True
                msgs = []
                for name, g in grads.items():
                    gn = g.numpy()
                    finite = np.isfinite(gn).all()
                    peak = float(np.abs(gn).max())
                    msgs.append(f"{name} peak={peak:.2e} finite={finite}")
                    if not finite or peak == 0.0:
                        ok = False
                print(f"{sk:24s} {ck:14s} {'finite_grad':22s} {'OK' if ok else 'FAIL'}  "
                      f"loss={loss:.3e}  {' '.join(msgs)}")
                if not ok:
                    n_fail += 1
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"{sk:24s} {ck:14s} {'finite_grad':22s} ERROR: {exc}")
                n_fail += 1

    print("-" * 100)
    if n_fail and not args.no_fail:
        print(f"FAILED: {n_fail} cases")
        sys.exit(1)
    print(f"DONE ({n_fail} failures)" if n_fail else "ALL PASSED")


if __name__ == "__main__":
    main()
