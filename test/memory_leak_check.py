"""Does repeating a forward+backward leak, for every equation and memory mode?

Nothing in the tree asked this. ``dd_regeom_leak_check.py`` does ask it, but
only for ModelParallel and only for ``Acoustic3D`` -- the same hard-coding that
made ``dd_pad_grad_check`` look like DD coverage when it was one equation's.
``dd_mem_probe`` / ``dd_mem_check`` report PEAK memory for capacity planning;
a peak says nothing about whether the tenth iteration holds more than the third.

WHAT COUNTS AS A LEAK HERE. Not "memory returns to zero" -- a live propagator
legitimately holds its wavefields, its boundary ring and its caches, and
demanding zero would flag every equation. The question is whether the steady
state IS steady: after the lazily-allocated buffers exist, does iteration N+1
hold more than iteration N. A real leak is linear in the iteration count, so it
separates from noise by simply running longer.

WHAT IS MEASURED. ``torch.cuda.memory_allocated()`` -- live tensor bytes -- not
``memory_reserved()``, which is the caching allocator's pool and grows by design
without anything leaking. Host RSS is read alongside it because this tree has
had host-side churn before (the boundary-saving backward used to build a
two-wavefield PINNED buffer per call), and that is invisible to every CUDA
counter.

TWO SCENARIOS, because they catch different bugs:

  reuse  one propagator, many calls -- the FWI epoch. Leaks here are per-call
         allocations that stay reachable: a retained autograd graph, a list
         appended to, a cache keyed on something that varies per call.
  rebuild  a new propagator per iteration -- the per-shot / per-config loop.
         Leaks here are module registration, process-global caches, C-side
         handles and temp directories that outlive the object that made them.

Run:
    python test/memory_leak_check.py --solvers acoustic2d,elastic2d --modes full,bs_gpu
    python test/memory_leak_check.py --self-test      # prove it can see a leak
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def allocated_mb() -> float:
    return torch.cuda.memory_allocated() / (1024.0 ** 2)


def _args(**overrides):
    """The suite's builders read their configuration off an argparse namespace.

    Taken from the suite's OWN parser defaults rather than retyped: this check
    must not silently run a different grid, dt or order than the gate does, and
    guessing the field set is how the first attempt died on a missing nz2d.
    """
    ns = suite.build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def one_iteration(solver, wavelet, sources, receivers, models_np, device, leak_sink=None):
    """One forward + backward, with every reference dropped on the way out.

    ``leak_sink`` is the self-test's deliberate leak: appending the output keeps
    the whole autograd graph alive, which is exactly the shape of the bug this
    check exists to find.
    """
    # tensors_from_models, not a comprehension: make_models returns a
    # (true, init, grad_flags) triple and not every model is a float that can
    # carry a gradient -- the first attempt here died on exactly that.
    models_init, grad_flags = models_np
    models = suite.tensors_from_models(models_init, grad_flags, device)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    loss = (syn.double() ** 2).sum()
    loss.backward()
    if leak_sink is not None:
        leak_sink.append(syn)
    del syn, loss, models


def measure(spec, backend, mode, scenario, shape, device, args, iters, warmup,
            rebuild, leak_sink=None):
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    models_np = (models_init, grad_flags)
    # exactly how the suite itself does it (solver_gradient_mode_suite.py:995-997)
    sources, receivers = suite.make_geometry(spec, shape, scenario, args)
    wavelet = torch.tensor(suite.ricker(args.nt, args.dt, args.freq, args.delay),
                           device=device)

    # Start this configuration from a clean pool. Without it the previous
    # mode's tail lands inside this one's measurement window as a single
    # isolated step -- measured: elastic_tti_sg3d|bs_gpu read
    # 212.715 213.146 212.715 212.715 ... (flat for the remaining sixteen) and
    # was called a 0.244 MB leak, while the same configuration in its own
    # process is 212.715 / 213.294 forever and reads 0.000. dd_regeom_leak_check
    # says the same thing in its docstring -- "memory phases run ONE solver each
    # so the shared CUDA pool can't cross-contaminate the readings" -- and this
    # check did not do it.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    solver = None
    if not rebuild:
        solver = suite.build_solver(spec, backend, mode, scenario, shape, device,
                                    args, Path("/tmp"), "leakcheck")

    samples = []
    for i in range(warmup + iters):
        if rebuild:
            solver = suite.build_solver(spec, backend, mode, scenario, shape,
                                        device, args, Path("/tmp"), "leakcheck")
        one_iteration(solver, wavelet, sources, receivers, models_np, device, leak_sink)
        if rebuild:
            del solver
            solver = None
        # gc BEFORE reading: a reference cycle collected late is not a leak, and
        # counting it as one would make every run red for the wrong reason.
        gc.collect()
        torch.cuda.synchronize()
        if i >= warmup:
            samples.append((allocated_mb(), rss_mb()))
    del solver
    gc.collect()
    return samples


def verdict(samples, tol_mb):
    """Least-squares slope over the whole measured run, scaled to the run.

    NOT last-minus-first, which is what this check did first and which was
    wrong. Several equations alternate between two steady states -- 261.398 /
    261.977 MB on elastic_tti_sg3d, every other iteration, forever. Endpoints
    then report the difference between two PHASES as growth, and report it with
    the opposite sign depending on where the run happened to stop:
    acoustic_vti_1st_3d came back -0.148 at eight iterations and +0.148 at
    twenty, from the same flat trace.

    The alternation is the ALLOCATOR'S ACCOUNTING, not an alternating
    allocation -- measured, not assumed, and the opposite of what this docstring
    claimed before. `requested_bytes.all.current` and `reserved_bytes.all.current`
    are bit-identical on every iteration for all four oscillating configurations;
    only `allocated_bytes` moves, because a reused free block whose remainder is
    too small to split off (large pool: the remainder must exceed 1 MB) is
    charged to the new tensor whole. acoustic_vti_1st_3d's padded model, from
    one call site, is charged 2,601,984 bytes on one iteration and 2,757,632 on
    the next -- a 155,648-byte difference that is exactly the reported swing and
    is comfortably under the 1 MB split threshold. Every swing seen here
    (0.148 / 0.194 / 0.579 / 0.709 MB) is under 1 MB, as that rule requires.

    A leak is a trend. Fitting one over every sample makes an oscillation
    cancel, and makes the reported number the growth ACROSS THE RUN, so a
    genuine per-iteration leak still scales with the iteration count exactly as
    it should.
    """
    n = len(samples)
    alloc = [a for a, _ in samples]
    rss = [r for _, r in samples]
    # Each parity gets n/2 points, and a fit over three of them is dominated by
    # its own residual: elastic_tti_sg3d|bs_gpu reported 0.338 MB of "growth" at
    # 12 measured iterations and 0.000 at 20, from a trace that is dead flat at
    # 212.715 / 213.294 either way. Refuse to answer rather than answer badly.
    if n < 16:
        raise ValueError(
            f"{n} measured iterations is too few to separate a trend from an "
            f"alternating steady state: each parity would get {n // 2} points. "
            f"Use --iters 16 or more.")
    # Compare LIKE WITH LIKE. Several equations alternate between two steady
    # states, so a fit over consecutive samples still sees a residual slope
    # whenever the sample count is even. Fitting each parity separately and
    # taking the larger removes the oscillation exactly rather than
    # approximately, and a real leak shows up identically in both.
    xs = list(range(n))
    xbar = sum(xs) / n

    def slope(ys):
        ybar = sum(ys) / n
        den = sum((x - xbar) ** 2 for x in xs)
        return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / den

    def phase_growth(ys):
        slopes = []
        for off in (0, 1):
            sub = ys[off::2]
            if len(sub) < 3:
                continue
            m = len(sub)
            sxs = list(range(m))
            sxbar = sum(sxs) / m
            sybar = sum(sub) / m
            den = sum((x - sxbar) ** 2 for x in sxs)
            # per ITERATION, not per sample: each parity steps by two
            slopes.append(
                sum((x - sxbar) * (y - sybar) for x, y in zip(sxs, sub)) / den / 2.0)
        if not slopes:                      # too few samples to split by parity
            return abs(slope(ys))
        # max, not mean: a leak that somehow showed in one parity only is still
        # a leak. `if slopes` rather than `if best` -- an exactly-zero slope is
        # the ANSWER, and falling back to the whole-run fit there is what made a
        # perfectly flat elastic_tti_sg3d read as a leak.
        return max(abs(x) for x in slopes)

    # growth over the run, so the number means the same thing whatever --iters is
    d_alloc = phase_growth(alloc) * (n - 1)
    d_rss = phase_growth(rss) * (n - 1)
    # The peak-to-peak of the oscillation is reported separately. It costs
    # nothing -- see above, reserved and requested bytes do not move -- but it
    # is reported rather than hidden because folding it into the leak number is
    # what made this check wrong twice.
    swing = max(alloc) - min(alloc)
    leaked = d_alloc > tol_mb
    return d_alloc, d_rss, leaked, swing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", default="acoustic2d,elastic2d")
    ap.add_argument("--modes", default="full,bs_gpu,ckpt_chunk")
    ap.add_argument("--backend", default="c", choices=("c", "eager"))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3,
                    help="iterations discarded before measuring; the lazily "
                         "allocated buffers must already exist or their first "
                         "appearance reads as a leak")
    ap.add_argument("--tol-mb", type=float, default=0.05,
                    help="growth over the whole measured run that still counts "
                         "as steady state")
    ap.add_argument("--rebuild", action="store_true",
                    help="construct a NEW propagator every iteration")
    ap.add_argument("--self-test", action="store_true",
                    help="inject a deliberate leak and require this check to "
                         "SEE it; a leak check that has never gone red is not "
                         "evidence of anything")
    ap.add_argument("--nt", type=int, default=60)
    ap.add_argument("--trace", action="store_true",
                    help="print every iteration's allocated bytes. A leak is a "
                         "straight line; a one-off step that never repeats is "
                         "not, and the two are indistinguishable from endpoints "
                         "alone.")
    ap.add_argument("--isolate", action="store_true",
                    help="run every (solver, mode) in its OWN process. The "
                         "shared CUDA pool carries readings across "
                         "configurations otherwise: measured, das_mu3d|bs_gpu "
                         "read 0.879 MB of growth after two other modes had run "
                         "in the same process, and 0.000 on its own, from the "
                         "same flat 205.157/205.867 trace. gc + empty_cache "
                         "between configurations was not enough; a process "
                         "boundary is. dd_regeom_leak_check.py runs one solver "
                         "per phase for exactly this reason.")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device"); return 2
    args = _args(nt=a.nt)
    device = torch.device("cuda:0")
    scenario = suite.SCENARIOS["interior"]

    if a.isolate:
        import subprocess
        rc = 0
        print(f"{'solver':<20s} {'mode':<12s} {'growth MB':>10s} {'swing MB':>9s} "
              f"{'dRSS MB':>8s}  verdict")
        print("-" * 76)
        for key in a.solvers.split(","):
            for mode in a.modes.split(","):
                cmd = [sys.executable, __file__, "--solvers", key,
                       "--modes", mode, "--iters", str(a.iters),
                       "--warmup", str(a.warmup), "--tol-mb", str(a.tol_mb),
                       "--backend", a.backend, "--nt", str(a.nt)]
                if a.rebuild:
                    cmd.append("--rebuild")
                if a.self_test:
                    cmd.append("--self-test")
                out = subprocess.run(cmd, capture_output=True, text=True)
                for line in out.stdout.splitlines():
                    if line.startswith(key):
                        print(line)
                rc |= (out.returncode != 0)
        print("-" * 76)
        print(f"-> {'PASS' if not rc else 'FAILED'}   "
              f"({a.warmup} warmup + {a.iters} measured, tol {a.tol_mb} MB, "
              f"one process per configuration)")
        return 1 if rc else 0

    bad = 0
    print(f"{'solver':<20s} {'mode':<12s} {'growth MB':>10s} {'swing MB':>9s} "
          f"{'dRSS MB':>8s}  verdict")
    print("-" * 76)
    for key in a.solvers.split(","):
        spec = suite.SOLVERS[key]
        shape = suite.shape_for(spec, args)
        modes = suite.filter_supported_modes(spec, a.modes.split(","))
        for mode in modes:
            sink = [] if a.self_test else None
            try:
                s = measure(spec, a.backend, mode, scenario, shape, device, args,
                            a.iters, a.warmup, a.rebuild, sink)
            except NotImplementedError as e:
                print(f"{key:<20s} {mode:<12s} {'-':>10s} {'-':>9s} {'-':>8s}  "
                      f"refused: {str(e)[:28]}")
                continue
            if a.trace:
                print(f"  {key}|{mode} per-iteration allocated MB:")
                print("   ", " ".join(f"{x:.3f}" for x, _ in s))
            d_alloc, d_rss, leaked, swing = verdict(s, a.tol_mb)
            if a.self_test:
                # inverted: the injected leak MUST be visible
                ok = leaked
                print(f"{key:<20s} {mode:<12s} {d_alloc:10.3f} {swing:9.3f} "
                      f"{d_rss:8.1f}  "
                      f"{'saw the injected leak' if ok else 'MISSED IT -- check is blind'}")
                bad += not ok
            else:
                print(f"{key:<20s} {mode:<12s} {d_alloc:10.3f} {swing:9.3f} "
                      f"{d_rss:8.1f}  {'LEAK' if leaked else 'steady'}")
                bad += leaked
    print("-" * 76)
    print(f"-> {'PASS' if bad == 0 else f'{bad} FAILED'}"
          f"   ({a.warmup} warmup + {a.iters} measured, tol {a.tol_mb} MB)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
