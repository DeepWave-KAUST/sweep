#!/usr/bin/env python3
"""Bit-exact before/after gate for the sweep refactor.

Records the forward record AND the inversion gradient for a matrix of
``(solver, impl, memory mode, scenario, grid)`` configurations, then compares
two recordings for **exact** equality (``torch.equal`` / exact float loss).

The whole refactor's acceptance criterion is "forward and inversion gradient
identical to before", so this is the only thing standing between a structural
change and a silent numerical regression.

Design notes
------------
* **One subprocess per config.** A single process would let one config's OOM or
  CUDA-sticky-error cascade into the next and fabricate both false reds and
  false greens (``lesson_pytest_ab_needs_per_file_isolation``).
* **``observed`` is code-independent.** It is a fixed-seed random tensor, not a
  forward of the true model -- an ``observed`` produced by the code under test
  would move with the refactor and mask exactly what we are trying to catch.
  A random residual also excites every receiver/time sample, so the adjoint is
  more discriminating than a physical residual that is near-zero somewhere.
* **Reproducibility is verified, not assumed.** ``--verify-reproducible`` runs
  the matrix twice on unmodified code and asserts bit-equality first; any
  config that cannot reach it is reported and must be demoted to a tolerance
  criterion *with its measured two-run noise floor recorded*
  (``lesson_bitexact_criterion_must_verify_attainable``).
* **The gate self-checks both directions.** ``--self-test`` perturbs one stored
  tensor by 1 ULP and asserts the comparator goes RED; a gate that can only say
  PASS is not a gate (``lesson_pipefail_inverts_safety_gates``).

Usage
-----
    . gate/env.sh
    $PY gate/bitgate.py --tier A --save    gate/base_A.pt
    $PY gate/bitgate.py --tier A --compare gate/base_A.pt
    $PY gate/bitgate.py --tier A --verify-reproducible
    $PY gate/bitgate.py --self-test gate/base_A.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))


# --------------------------------------------------------------------------- #
# config matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cfg:
    solver: str        # key into solver_gradient_mode_suite.SOLVERS
    impl: str          # "eager" | "c"
    mode: str          # eager: full|bs|ckpt ; c: full|bs_*|ckpt_*
    scenario: str      # key into solver_gradient_mode_suite.SCENARIOS
    grid: str          # "canon" | "phys"
    #: which surface discretisation: "auto" | "image" | "apm".
    topo_method: str = "auto"
    #: irregular free-surface topography: "" (flat) | "hill" | "stairs".
    #:
    #: Every other scenario in this file runs a FLAT model, so until these
    #: existed no topography code path had a bit-exact gate at all -- not the
    #: image-method rows, not the APM air mask, not the DD air-clear pre-pass.
    topo: str = ""

    @property
    def id(self) -> str:
        # Flat ids keep their spelling so existing baselines keep matching.
        t = (f"|topo_{self.topo}_{self.topo_method}" if self.topo else "")
        return f"{self.solver}|{self.impl}|{self.mode}|{self.scenario}|{self.grid}{t}"

    @classmethod
    def from_id(cls, cid: str) -> "Cfg":
        """Rebuild from an id. Used by the per-config subprocess.

        Parsed rather than positionally unpacked so that adding an axis to Cfg
        cannot silently break the child process -- which is exactly what a bare
        5-tuple unpack did when ``topo`` arrived.
        """
        parts = cid.split("|")
        solver, impl, mode, scenario, grid = parts[:5]
        topo = topo_method = ""
        for extra in parts[5:]:
            if extra.startswith("topo_"):
                _, topo, topo_method = extra.split("_", 2)
        return cls(solver, impl, mode, scenario, grid,
                   topo_method=topo_method or "auto", topo=topo)


# Grids.  ``canon`` mirrors solver_gradient_mode_suite's defaults (48x56, abcn
# 30) -- fast, but its physical interior is NEGATIVE (shape includes the PML),
# i.e. everything sits inside the absorbing layer.  That is fine for a
# before/after comparison but blind to anything that only shows up with a real
# interior, so ``phys`` (140x160, abcn 20 -> 100x120 interior) is always run
# alongside it.  See reference_canonical_gradient_test_model.
GRIDS = {
    "canon": dict(nz2d=48, nx2d=56, nz3d=24, ny3d=20, nx3d=24, abcn=30, nt=120),
    "phys": dict(nz2d=140, nx2d=160, nz3d=60, ny3d=56, nx3d=60, abcn=20, nt=200),
}

_C_MEM_MODES = ("full", "bs_gpu", "bs_cpu", "bs_disk", "bs_gpu_int8",
                "ckpt_chunk", "ckpt_recursive")

# Tier A -- fast, run after every commit.  Covers both time-discretisation
# families (2nd-order acoustic, 1st-order staggered elastic), both impls, the
# three memory strategies, flat/free-surface/per-edge-FS, and both grids.
TIER_A = [
    Cfg("acoustic2d", "eager", "full", "interior", "canon"),
    Cfg("acoustic2d", "eager", "full", "free_surface", "canon"),
    Cfg("acoustic2d", "eager", "full", "free_surface_all4", "canon"),
    Cfg("acoustic2d", "eager", "bs", "interior", "canon"),
    Cfg("acoustic2d", "eager", "ckpt", "interior", "canon"),
    Cfg("elastic2d", "eager", "full", "interior", "canon"),
    Cfg("elastic2d", "eager", "full", "free_surface", "canon"),
    Cfg("vrz2d", "eager", "full", "interior", "canon"),
    Cfg("acoustic3d", "eager", "full", "interior", "canon"),
    Cfg("acoustic2d", "c", "full", "interior", "canon"),
    Cfg("acoustic2d", "c", "bs_gpu", "interior", "canon"),
    Cfg("acoustic2d", "c", "bs_cpu", "interior", "canon"),
    Cfg("acoustic2d", "c", "bs_gpu_int8", "interior", "canon"),
    Cfg("acoustic2d", "c", "ckpt_chunk", "interior", "canon"),
    Cfg("acoustic2d", "c", "ckpt_recursive", "interior", "canon"),
    Cfg("acoustic2d", "c", "bs_gpu", "free_surface", "canon"),
    Cfg("acoustic2d", "c", "bs_gpu", "free_surface_all4", "canon"),
    Cfg("acoustic3d", "c", "bs_gpu", "interior", "canon"),
    Cfg("elastic2d", "c", "bs_gpu", "interior", "canon"),
    Cfg("elastic2d", "c", "bs_gpu", "free_surface", "canon"),
    Cfg("elastic3d", "c", "bs_gpu", "interior", "canon"),
    Cfg("vrz2d", "c", "bs_gpu", "interior", "canon"),
    Cfg("lsrtm2d", "c", "bs_gpu", "interior", "canon"),
    # real physical interior
    Cfg("acoustic2d", "eager", "full", "interior", "phys"),
    Cfg("acoustic2d", "c", "bs_gpu", "interior", "phys"),
    Cfg("acoustic2d", "c", "bs_gpu", "free_surface", "phys"),
    Cfg("elastic2d", "c", "bs_gpu", "free_surface", "phys"),
]

# Every solver key in solver_gradient_mode_suite.SOLVERS.  NOTE: ElasticVRR
# (csrc elastic_vr2d) has no entry there, so its compiled binding is NOT gated
# by anything here -- the one equation with a CUDA kernel outside the suite.
# Run gate/evr_ab.py for it; gate/verify_all.sh does.
ALL_SOLVERS = (
    "acoustic2d", "acoustic3d", "vrz2d", "vrz3d", "lsrtm2d", "lsrtm3d",
    "das2d", "das3d", "das_mu2d", "das_mu3d", "elastic2d", "elastic3d",
    "acoustic_vti_1st_2d", "acoustic_vti_1st_3d", "elastic_tti_sg2d",
    # The suite has had specs for these two since it was written; the list
    # simply never picked them up, so two compiled equations -- one of them a
    # whole displacement-based formulation -- had no bit-exact coverage at all.
    "elastic_tti_sg3d", "elastic_tti_2nd2d",
)

# das2d / das3d declare supported_scenarios=('interior',); everything else takes
# free_surface too.
_INTERIOR_ONLY = ("das2d", "das3d")


def _topography(kind: str, nx: int, ny: int | None = None):
    """Surface row index per physical column; mirrors topo_gradient_mode_suite.

    2-D wants ``(nx_phys,)``, 3-D wants ``(ny_phys, nx_phys)``. The 3-D profile
    deliberately varies along BOTH axes: a y-invariant surface would not catch a
    y/x transposition in the runtime pad, and the pad tuples in the image and
    APM builders are in reversed axis order with equal widths, so a
    transposition is invisible unless something breaks the symmetry.
    """
    import numpy as np
    x = np.arange(nx)
    if kind == "hill":
        prof = 3.0 + 6.0 * np.exp(-((x - nx / 2) ** 2) / (2 * (nx / 8) ** 2))
    elif kind == "stairs":
        prof = np.full(nx, 4.0)
        prof[nx // 3: 2 * nx // 3] = 8.0
        prof[2 * nx // 3:] = 6.0
    else:
        raise ValueError(f"unknown topography {kind!r}")
    if ny is None:
        return prof.round().astype(np.int64)
    yv = np.arange(ny)
    tilt = 2.0 * np.sin(2 * np.pi * yv / max(ny, 1))[:, None]
    return (prof[None, :] + tilt).round().astype(np.int64)


# Topography configs, and the trap in choosing them: `impl='c'` REFUSES
# boundary saving with topography and refuses the APM backward outright. A
# refused config still "passes" the comparison -- an error compares equal to an
# error -- so it would be a green that checks nothing. Topo configs on the
# compiled path therefore use `full` or `ckpt_chunk`, never `bs_*`.
#
# The second trap: APM is the DEFAULT surface method for elastic equations, and
# the compiled APM BACKWARD is a stub that refuses. So an elastic topo config on
# impl='c' must ask for 'image' explicitly; APM is covered on eager, where its
# gradient does work.
_TOPO = [
    # Both grids: the shape given to PropTorch is the physical extent (the PML
    # is added on top), so canon is a perfectly good surface, just a small one.
    # The 3-D configs matter most -- canon 3-D is 24x20x24 and phys 3-D is
    # 60x56x60, both with ny != nx, which is what lets a y/x transposition in
    # the runtime pad show up at all.
    #
    # image method (vacuum staircase) -- acoustic, both impls
    Cfg("acoustic2d", "eager", "full", "interior", "phys", topo="hill"),
    Cfg("acoustic2d", "eager", "full", "interior", "phys", topo="stairs"),
    Cfg("acoustic2d", "c", "full", "interior", "phys", topo="hill"),
    Cfg("acoustic2d", "c", "ckpt_chunk", "interior", "phys", topo="hill"),
    Cfg("acoustic2d", "c", "full", "interior", "phys", topo="stairs"),
    Cfg("acoustic3d", "eager", "full", "interior", "phys", topo="hill"),
    Cfg("acoustic3d", "c", "full", "interior", "phys", topo="hill"),
    # elastic: image on both impls, APM on eager only
    Cfg("elastic2d", "eager", "full", "interior", "phys", topo="hill",
        topo_method="image"),
    Cfg("elastic2d", "c", "full", "interior", "phys", topo="hill",
        topo_method="image"),
    Cfg("elastic2d", "eager", "full", "interior", "phys", topo="hill",
        topo_method="apm"),
]


def _tier_c() -> list[Cfg]:
    """Coverage tier: touch EVERY equation once on both impls.

    This is the tier for changes that are per-equation but shallow -- the
    ``_C()`` de-duplication, the registry, the slot table. Tier A is deeper on
    a few equations; tier C is one config on all of them.
    """
    out: list[Cfg] = []
    for s in ALL_SOLVERS:
        out.append(Cfg(s, "eager", "full", "interior", "canon"))
        out.append(Cfg(s, "c", "bs_gpu", "interior", "canon"))
    return out


def _tier_b() -> list[Cfg]:
    """Full sweep: every equation x every memory mode, plus free surface and a
    real physical interior. Run at the end of each phase."""
    out: list[Cfg] = []
    for s in ALL_SOLVERS:
        for mode in _C_MEM_MODES:
            out.append(Cfg(s, "c", mode, "interior", "canon"))
        out.append(Cfg(s, "c", "bs_gpu", "interior", "phys"))
        out.append(Cfg(s, "eager", "full", "interior", "canon"))
        if s not in _INTERIOR_ONLY:
            out.append(Cfg(s, "c", "bs_gpu", "free_surface", "canon"))
            out.append(Cfg(s, "eager", "full", "free_surface", "canon"))
    for s in ("acoustic2d", "elastic2d"):
        out.append(Cfg(s, "c", "bs_gpu", "free_surface_all4", "canon"))
        out.append(Cfg(s, "eager", "full", "free_surface_all4", "canon"))
    return out


def _tier_t() -> list[Cfg]:
    """Topography tier. Its own tier because it needs its own baseline and
    because none of the other tiers touch a non-flat model."""
    return list(_TOPO)


TIERS = {"A": TIER_A, "B": _tier_b, "C": _tier_c, "T": _tier_t}


def tier_configs(tier: str) -> list[Cfg]:
    t = TIERS[tier]
    return list(t() if callable(t) else t)


# --------------------------------------------------------------------------- #
# single-config runner (executed in its own subprocess)
# --------------------------------------------------------------------------- #
def _suite_args(grid: str):
    """A namespace shaped like solver_gradient_mode_suite's argparse result."""
    import solver_gradient_mode_suite as S

    parser = S.build_parser()
    args = parser.parse_args([])
    g = GRIDS[grid]
    for k, v in g.items():
        setattr(args, k, v)
    args.no_plot = True
    return args


def run_one(cfg: Cfg, device: str = "cuda") -> dict:
    """Build + run one configuration; return the tensors to be compared."""
    import solver_gradient_mode_suite as S

    torch.manual_seed(0)
    np.random.seed(0)

    spec = S.SOLVERS[cfg.solver]
    scenario = S.SCENARIOS[cfg.scenario]
    args = _suite_args(cfg.grid)
    shape = S.shape_for(spec, args)
    sources, receivers = S.make_geometry(spec, shape, scenario, args)
    true_models, init_models, grad_flags = S.make_models(spec, shape)
    wavelet = torch.tensor(
        S.ricker(args.nt, args.dt, args.freq, args.delay), device=device)

    run_dir = Path(tempfile.mkdtemp(prefix="bitgate_"))
    case_key = f"{cfg.solver}_{cfg.scenario}_{cfg.grid}"

    if cfg.topo:
        # The shape passed to PropTorch is the PHYSICAL extent -- the propagator
        # adds the PML itself, so prop.shape comes back larger (140x160 at
        # abcn=20 becomes 180x200 internally). Topography is indexed in physical
        # columns, so it is exactly as long as the shape given here. Deriving it
        # by subtracting abcn -- which is what prop.shape - 2*abcn does
        # INTERNALLY -- double-counts and is rejected.
        topo = _topography(cfg.topo, shape[-1],
                           shape[1] if len(shape) == 3 else None)
    else:
        topo = None
    if topo is not None:
        solver = _build_with_topo(S, spec, scenario, shape, device, args, cfg,
                                  topo, run_dir, case_key)
    elif cfg.impl == "eager":
        solver = _build_eager(S, spec, scenario, shape, device, args, cfg.mode)
    else:
        solver = S.build_solver(spec, "c", cfg.mode, scenario, shape, device,
                                args, run_dir, case_key)

    # PropTorch(impl='c') does NOT raise for an equation with no CUDA binding: it
    # warns and sets impl='eager' (propagator/torch.py). A gate that trusted the
    # kwarg would then run the SAME eager code for both the 'c' and the 'eager'
    # config and pass every comparison while testing nothing -- the exact shape
    # of failure this gate exists to prevent. Assert what was BUILT, not what was
    # asked for.
    realised = getattr(solver, "impl", None)
    if realised is not None and realised != cfg.impl:
        raise RuntimeError(
            f"{cfg.id}: asked for impl={cfg.impl!r} but the propagator built "
            f"impl={realised!r} (equation has no compiled kernel?). Refusing to "
            f"record this config: it would compare eager against eager.")

    # ``observed`` must not come from the code under test -- see module docstring.
    nrec = receivers.shape[1]
    nshot = sources.shape[0]
    ncomp = max(1, len(spec.receiver_type))
    gen = torch.Generator(device="cpu").manual_seed(1234)
    observed = torch.randn(nshot, nrec, args.nt, generator=gen) * 1e-3
    if ncomp > 1:
        observed = observed  # normalize_record collapses components; keep 3-D

    fwd = S.run_forward(solver, wavelet, sources, receivers, init_models, device)
    if observed.shape != fwd.shape:
        observed = torch.randn(*fwd.shape, generator=gen) * 1e-3

    grad = S.run_gradient(solver, wavelet, sources, receivers, observed,
                          init_models, grad_flags, spec.model_names, device)

    out = {
        "record": fwd.to(torch.float32).contiguous(),
        "loss": float(grad["loss"]),
        "grads": {k: v.to(torch.float32).contiguous()
                  for k, v in grad["grads"].items()},
        "shape": tuple(int(s) for s in shape),
    }
    return out


def _build_with_topo(S, spec, scenario, shape, device, args, cfg, topo,
                     run_dir, case_key):
    """Propagator with an irregular free surface, on either impl.

    ``topography=`` makes the free surface implicit, so ``free_surface`` is NOT
    passed -- the propagator resolves it from ``topo_method``.
    """
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions

    equation = spec.equation_cls(spatial_order=args.spatial_order,
                                 device=device, backend="torch")
    common = dict(
        shape=shape, dev=device, dh=S.spacing_for(spec, args), dt=args.dt,
        source_type=list(spec.source_type), receiver_type=list(spec.receiver_type),
        abcn=args.abcn, pml_type=spec.pml_type, nt=args.nt, B=1,
        allow_growth=True, topography=topo, topo_method=cfg.topo_method,
    )
    if cfg.impl == "eager":
        return PropTorch(equation, backend="torch", impl="eager",
                         eager_options=EagerOptions(use_compile=False),
                         use_ckpt=(cfg.mode == "ckpt"), **common)
    cuda_options = S.build_cuda_options(cfg.mode, args, run_dir, case_key)
    if cfg.mode == "full":
        return PropTorch(equation, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **common)
    return PropTorch(equation, backend="torch", impl="c",
                     cuda_options=cuda_options, **common)


def _build_eager(S, spec, scenario, shape, device, args, mode):
    """Eager propagator with the requested memory strategy.

    ``S.build_solver(..., 'eager', ...)`` only ever builds the full-wavefield
    variant; the eager path also has boundary saving and checkpointing, and
    both are on the refactor's blast radius, so they get their own configs.
    """
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions, MemoryOptions, BoundaryOptions

    equation = spec.equation_cls(spatial_order=args.spatial_order,
                                 device=device, backend="torch")
    common = dict(
        shape=shape, dev=device, dh=S.spacing_for(spec, args), dt=args.dt,
        source_type=list(spec.source_type), receiver_type=list(spec.receiver_type),
        abcn=args.abcn, pml_type=spec.pml_type,
        free_surface=scenario.free_surface, nt=args.nt, B=1, allow_growth=True,
    )
    kw = dict(backend="torch", impl="eager",
              eager_options=EagerOptions(use_compile=False))
    if mode == "full":
        return PropTorch(equation, use_ckpt=False, **kw, **common)
    if mode == "ckpt":
        return PropTorch(equation, use_ckpt=True, **kw, **common)
    if mode == "bs":
        return PropTorch(
            equation,
            memory=MemoryOptions(strategy="boundary",
                                 boundary=BoundaryOptions(storage="gpu")),
            **kw, **common)
    raise ValueError(f"unknown eager mode {mode!r}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _run_matrix(cfgs: list[Cfg], inproc: bool, verbose: bool) -> dict:
    results: dict[str, dict] = {}
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        if inproc:
            try:
                results[cfg.id] = run_one(cfg)
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                results[cfg.id] = {"error": f"{type(exc).__name__}: {exc}"}
                status = "ERR"
        else:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
                out = Path(fh.name)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--one", cfg.id, "--out", str(out)],
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode == 0 and out.exists() and out.stat().st_size:
                results[cfg.id] = torch.load(out, map_location="cpu",
                                             weights_only=False)
                status = "ok"
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                results[cfg.id] = {
                    "error": tail[-1] if tail else f"rc={proc.returncode}"}
                status = "ERR"
            out.unlink(missing_ok=True)
        dt = time.time() - t0
        if verbose:
            mark = "." if status == "ok" else "E"
            print(f"[{i:3d}/{len(cfgs)}] {mark} {dt:6.1f}s  {cfg.id}", flush=True)
            if status == "ERR":
                print(f"          -> {results[cfg.id]['error']}", flush=True)
    return results


# Multiple of a config's MEASURED two-run spread that still counts as noise.
# The floor itself is never guessed -- see --measure-noise -- so widening the
# criterion means editing this one number, visibly, rather than nudging a table.
NOISE_HEADROOM = 5.0


def _grad_rel(ga: torch.Tensor, gb: torch.Tensor) -> float:
    scale = ga.abs().max().item()
    if scale == 0.0:
        return 0.0 if torch.equal(ga, gb) else float("inf")
    return (ga - gb).abs().max().item() / scale


def _cmp_entry(a: dict, b: dict, tol: float | None = None) -> list[str]:
    """Compare one config's payload.

    ``tol`` is a per-config RELATIVE gradient tolerance, and it is only ever set
    for configs proven (by ``--measure-noise``) not to reproduce themselves.
    The forward record and the loss stay STRICT for every config regardless: a
    solver whose gradient reduction order wanders must still reproduce its
    forward exactly, so the relaxation cannot swallow a forward regression --
    the failure mode that would actually matter.
    """
    fails = []
    if ("error" in a) != ("error" in b):
        return [f"error-status changed: before={a.get('error', 'ok')!r} "
                f"after={b.get('error', 'ok')!r}"]
    if "error" in a:
        if a["error"] != b["error"]:
            fails.append(f"error text changed: {a['error']!r} -> {b['error']!r}")
        return fails
    if a["record"].shape != b["record"].shape:
        fails.append(f"record shape {tuple(a['record'].shape)} -> "
                     f"{tuple(b['record'].shape)}")
    elif not torch.equal(a["record"], b["record"]):
        n = int((a["record"] != b["record"]).sum())
        m = (a["record"] - b["record"]).abs().max().item()
        fails.append(f"record differs ({n} elems, max|d|={m:.3e})")
    if a["loss"] != b["loss"]:
        fails.append(f"loss {a['loss']!r} -> {b['loss']!r}")
    ka, kb = set(a["grads"]), set(b["grads"])
    if ka != kb:
        fails.append(f"grad keys {sorted(ka)} -> {sorted(kb)}")
    for k in sorted(ka & kb):
        ga, gb = a["grads"][k], b["grads"][k]
        if ga.shape != gb.shape:
            fails.append(f"grad[{k}] shape {tuple(ga.shape)} -> {tuple(gb.shape)}")
            continue
        if torch.equal(ga, gb):
            continue
        n = int((ga != gb).sum())
        m = (ga - gb).abs().max().item()
        if tol is None:
            fails.append(f"grad[{k}] differs ({n} elems, max|d|={m:.3e})")
            continue
        rel = _grad_rel(ga, gb)
        if rel > tol:
            fails.append(f"grad[{k}] rel {rel:.2e} EXCEEDS the measured noise "
                         f"floor {tol:.2e} ({n} elems, max|d|={m:.3e})")
    return fails


def load_floors(path: Path | None):
    """Per-config relative gradient tolerances, from a --measure-noise run."""
    if path is None or not Path(path).exists():
        return {}
    blob = json.loads(Path(path).read_text())
    return {k: v["grad_rel"] * NOISE_HEADROOM
            for k, v in blob.get("floors", {}).items()
            if v and v.get("grad_rel")}


def compare(before: dict, after: dict, verbose: bool = True, floors=None) -> int:
    keys = sorted(set(before) | set(after))
    floors = floors or {}
    npass = nfail = nmiss = nrelaxed = nnew = nerr = 0
    for k in keys:
        if k not in before:
            print(f"NEW      {k}")
            nnew += 1
            continue
        if k not in after:
            print(f"MISSING  {k}")
            nmiss += 1
            continue
        tol = floors.get(k)
        fails = _cmp_entry(before[k], after[k], tol=tol)
        if not fails and tol is not None and not all(
                torch.equal(before[k]["grads"][g], after[k]["grads"][g])
                for g in before[k].get("grads", {})):
            nrelaxed += 1
        if fails:
            nfail += 1
            print(f"FAIL     {k}")
            for f in fails:
                print(f"           {f}")
        elif fails:
            nfail += 1
            print(f"FAIL     {k}")
            for f in fails:
                print(f"           {f}")
        else:
            npass += 1
            if "error" in before[k]:
                nerr += 1
            if verbose:
                errm = " (both errored)" if "error" in before[k] else ""
                tolm = f"  [within measured noise floor {tol:.1e}]" if tol else ""
                print(f"pass     {k}{errm}{tolm}")
    tail = f"   (of which {nrelaxed} within a measured noise floor)" if nrelaxed else ""
    if nerr:
        tail += f"   [{nerr} compared ERROR TEXT only -- no numbers checked]"
    print(f"\n{'='*70}\nPASS {npass}   FAIL {nfail}   MISSING {nmiss}   NEW {nnew}{tail}\n{'='*70}")
    return 0 if (nfail == 0 and nmiss == 0) else 1


def measure_noise(cfgs, inproc, out: Path, passes: int = 4) -> int:
    """Run the matrix ``passes`` times on the CURRENT tree and record what does
    not reproduce itself.

    This is the prerequisite for using ``torch.equal`` as an acceptance
    criterion at all: a config that cannot reproduce itself will fail every
    comparison forever, and 'fixing' it means chasing a bug that is not there
    (lesson_bitexact_criterion_must_verify_attainable).

    ``passes`` defaults to 4 (6 pairings), not 2, because two runs measure ONE
    sample of the spread and int8 storage in particular has a long tail: two
    passes put acoustic_vti_1st_3d's int8 floor at 6.3e-05, while eight passes
    showed its true spread reaching 1.1e-03 -- a factor of 18. A floor sampled
    that far below the truth turns every later comparison into a false alarm,
    which costs more than the extra passes do.

    Configs that ARE bit-stable are written as null, so the file is a census
    rather than an opt-in list of excuses.
    """
    print(f"# measuring the spread of {len(cfgs)} configs over {passes} passes "
          f"({passes * (passes - 1) // 2} pairings) on the CURRENT tree\n")
    runs = []
    for p in range(passes):
        if p:
            print(f"\n--- pass {p + 1}/{passes} ---\n")
        runs.append(_run_matrix(cfgs, inproc, True))

    floors = {}
    for cfg in cfgs:
        k = cfg.id
        entries = [r.get(k, {}) for r in runs]
        if any("error" in e for e in entries):
            floors[k] = None
            continue
        rec_stable = all(torch.equal(entries[0]["record"], e["record"]) for e in entries[1:])
        loss_stable = all(entries[0]["loss"] == e["loss"] for e in entries[1:])
        # Worst over EVERY pairing, not just consecutive ones.
        rel, worst = 0.0, None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                for g in entries[i]["grads"]:
                    r = _grad_rel(entries[i]["grads"][g], entries[j]["grads"][g])
                    if r > rel:
                        rel, worst = r, g
        if rec_stable and loss_stable and rel == 0.0:
            floors[k] = None
        else:
            floors[k] = {"grad_rel": rel, "worst_grad": worst,
                         "record": "stable" if rec_stable else "UNSTABLE",
                         "loss": "stable" if loss_stable else "UNSTABLE",
                         "passes": passes}

    blob = {"meta": {"gpu": torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None,
                     "torch": torch.__version__, "passes": passes,
                     "n": len(cfgs)},
            "floors": floors}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n")

    bad = {k: v for k, v in floors.items() if v}
    print(f"\n{'='*70}")
    print(f"{len(floors) - len(bad)}/{len(floors)} configs reproduce themselves bit-exactly")
    if bad:
        print(f"\n{len(bad)} do NOT -- their gradients get a tolerance derived "
              f"from these measurements:\n")
        for k, v in sorted(bad.items()):
            flag = ""
            if v["record"] != "stable" or v["loss"] != "stable":
                flag = f"   <-- record={v['record']} loss={v['loss']} (SERIOUS)"
            print(f"  {k}\n      worst grad[{v['worst_grad']}] rel={v['grad_rel']:.2e}{flag}")
    print(f"{'='*70}\nwrote {out}")
    # A config whose RECORD or LOSS is unstable is a different, much worse
    # finding than a wandering gradient reduction; do not let it pass quietly.
    serious = [k for k, v in bad.items()
               if v["record"] != "stable" or v["loss"] != "stable"]
    if serious:
        print(f"\n!! {len(serious)} config(s) have an unstable FORWARD, not just "
              f"a gradient. That is not a reduction-order artefact -- investigate "
              f"before relying on any of this.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="A", choices=sorted(TIERS))
    ap.add_argument("--only", default=None,
                    help="Substring filter on config ids.")
    ap.add_argument("--only-unstable", type=Path, default=None,
                    help="Restrict to the configs a previous --measure-noise "
                         "found non-reproducible (re-measure just those).")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--verify-reproducible", action="store_true",
                    help="Run the matrix twice and require bit-equality.")
    ap.add_argument("--self-test", type=Path, default=None,
                    help="Perturb a stored tensor by 1 ULP; the comparator "
                         "MUST report FAIL (proves the gate can go red).")
    ap.add_argument("--measure-noise", type=Path, default=None,
                    help="Run the tier --passes times on the current tree and "
                         "write each config's measured spread to this JSON.")
    ap.add_argument("--passes", type=int, default=4,
                    help="Passes for --measure-noise. 2 measures a single "
                         "sample and reads int8 floors up to 18x too low.")
    ap.add_argument("--floors", type=Path,
                    default=HERE / "noise_floors.json",
                    help="Per-config gradient tolerances from --measure-noise. "
                         "Configs absent from it stay strictly bit-exact.")
    ap.add_argument("--strict", action="store_true",
                    help="Ignore the floors file; require torch.equal everywhere.")
    ap.add_argument("--inproc", action="store_true",
                    help="Run configs in-process (fast, no crash isolation).")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.one:                                   # child process
        payload = run_one(Cfg.from_id(args.one))
        torch.save(payload, args.out)
        return 0

    cfgs = tier_configs(args.tier)
    if args.only:
        cfgs = [c for c in cfgs if args.only in c.id]
    if args.only_unstable:
        keep = {k for k, v in json.loads(args.only_unstable.read_text())["floors"].items() if v}
        cfgs = [c for c in cfgs if c.id in keep]

    if args.list:
        for c in cfgs:
            print(c.id)
        print(f"\n{len(cfgs)} configs in tier {args.tier}")
        return 0

    if args.self_test:
        return _self_test(args.self_test)

    if args.measure_noise:
        return measure_noise(cfgs, args.inproc, args.measure_noise, args.passes)

    floors = {} if args.strict else load_floors(args.floors)

    if args.verify_reproducible:
        print(f"# reproducibility check: tier {args.tier}, {len(cfgs)} configs, "
              f"two identical runs must be bit-equal\n")
        a = _run_matrix(cfgs, args.inproc, True)
        print("\n--- second pass ---\n")
        b = _run_matrix(cfgs, args.inproc, True)
        print("\n--- comparison ---\n")
        rc = compare(a, b)   # deliberately strict: floors are not applied here
        if rc:
            print("\n!! Some configs are NOT bit-reproducible on unmodified code.\n"
                  "   Do NOT use torch.equal as their acceptance criterion --\n"
                  "   locate WHERE they differ first, then either narrow the\n"
                  "   criterion (e.g. exclude PML) or record the measured\n"
                  "   two-run noise floor as the tolerance.")
        return rc

    print(f"# tier {args.tier}: {len(cfgs)} configs\n")
    res = _run_matrix(cfgs, args.inproc, True)
    nerr = sum(1 for v in res.values() if "error" in v)
    print(f"\nran {len(res)} configs, {nerr} errored")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        meta = {"tier": args.tier, "n": len(res),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None,
                "configs": [asdict(c) for c in cfgs]}
        torch.save({"meta": meta, "results": res}, args.save)
        print(f"saved -> {args.save}")
        (args.save.with_suffix(".json")).write_text(
            json.dumps(meta, indent=2) + "\n")

    if args.compare:
        blob = torch.load(args.compare, map_location="cpu", weights_only=False)
        return compare(blob["results"], res, floors=floors)
    return 1 if nerr else 0


def _self_test(baseline: Path) -> int:
    """The comparator must FAIL on a 1-ULP perturbation of a single element."""
    blob = torch.load(baseline, map_location="cpu", weights_only=False)
    before = blob["results"]
    good = [k for k, v in before.items() if "error" not in v]
    if not good:
        print("self-test: baseline has no successful configs")
        return 1
    import copy
    checks = []
    for key in good[:3]:
        after = copy.deepcopy(before)
        tgt = after[key]
        # perturb the largest-magnitude gradient element by ONE ulp
        gname = sorted(tgt["grads"])[0]
        g = tgt["grads"][gname]
        idx = int(g.abs().argmax())
        flat = g.reshape(-1)
        flat[idx] = torch.nextafter(flat[idx], torch.tensor(float("inf")))
        rc = compare(before, after, verbose=False)
        checks.append((key, gname, rc))
    ok = all(rc == 1 for _, _, rc in checks)
    print("\n" + "=" * 70)
    for key, gname, rc in checks:
        print(f"{'RED (correct)' if rc == 1 else 'GREEN (BROKEN GATE)'}"
              f"  1-ulp in grad[{gname}] of {key}")
    print(f"self-test: {'PASS -- the gate can go red' if ok else 'FAIL -- THE GATE IS BLIND'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
