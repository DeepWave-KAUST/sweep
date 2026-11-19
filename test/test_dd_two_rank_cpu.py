"""Two-rank domain decomposition on the CPU (host core, gloo) vs one domain.

An x cut with every source clear of it takes ModelParallel's overlapped
forward: phase 1 computes the cut strips, their shipment starts, phase 2
computes the rest.  That loop was written against CUDA streams and
``_overlap_ok`` checks the cut geometry only, so a CPU tile reached
``torch.cuda.Stream()`` -- and a process without CUDA died there.

Both ranks run with no GPU visible and the source mid tile 0, so the
overlapped loop is the one that runs (asserted).  The gathered record and the
summed tile gradients must equal the single-domain propagator's.

The 2-D case also runs with ``overlap=False``: the serial step-then-exchange
loop must be the one that runs then (the overlapped one not at all), and it
must give the same record and gradient -- the two schedules are bit-identical
by construction, and this is the test that holds them to it.  (The 3-D case
takes minutes per run on the CPU; one schedule of it is enough.)

DD steps its tiles with ``impl='c'``; where that cannot run on the CPU (no host
core in this build) the test skips.
"""
from __future__ import annotations

import datetime
import os
import time
import traceback
import warnings

import numpy as np
import pytest
import torch

dist = pytest.importorskip("torch.distributed")
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo is required", allow_module_level=True)

import torch.multiprocessing as mp

WORLD = 2
DT, NT, DH, SO, ABCN = 0.0015, 90, 10.0, 4, 12
#: Global shapes, cut in half along x.  AcousticVRZ3D is the case that failed.
SHAPES = {"Acoustic": (48, 64), "AcousticVRZ3D": (24, 20, 32)}
#: (equation, overlap) runs; the ranks take minutes per run on the CPU.
CASES = [("Acoustic", True), ("Acoustic", False), ("AcousticVRZ3D", True)]
#: For every run of both ranks together; a hang is reported after this.
DEADLINE_S = 900


def _prop(name):
    import sweep.equations as E
    from sweep.propagator.torch import PropTorch

    eq = getattr(E, name)(spatial_order=SO, device="cpu", backend="torch")
    return PropTorch(eq, backend="torch", impl="c", shape=SHAPES[name], dev="cpu",
                     dh=DH, dt=DT, source_type=["h1"], receiver_type=["h1"],
                     abcn=ABCN, nt=NT, B=1)


def _host_core_serves(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the falling-back-to-eager warning
        return _prop(name).impl == "c"


def _models(name):
    shape = SHAPES[name]
    nz = shape[0]
    vp = np.linspace(1800.0, 2400.0, nz, dtype=np.float32)
    vp = np.broadcast_to(vp.reshape((nz,) + (1,) * (len(shape) - 1)), shape).copy()
    vp[tuple(slice(s // 3, 2 * s // 3) for s in shape)] += 180.0   # straddles the cut
    out = [vp]
    if name == "AcousticVRZ3D":
        out.append(np.full(shape, 2000.0, dtype=np.float32))       # impedance
    return [torch.tensor(m, requires_grad=True) for m in out]


def _geometry(name):
    shape = SHAPES[name]
    nx = shape[-1]
    sx = nx // WORLD // 2        # mid tile 0, well over M from the cut at nx // 2
    if len(shape) == 2:
        src = [[[sx, shape[0] // 5]]]
        rec = [[[x, 3] for x in range(4, nx - 4, 5)]]
    else:
        src = [[[sx, shape[1] // 2, shape[0] // 5]]]
        rec = [[[x, shape[1] // 2, 3] for x in range(4, nx - 4, 5)]]
    return np.array(src, dtype=np.int64), np.array(rec, dtype=np.int64)


def _wavelet():
    return torch.randn(NT, generator=torch.Generator().manual_seed(4321)) * 1e-2


def _adjoint(shape):
    return torch.randn(shape, generator=torch.Generator().manual_seed(7)) * 1e-3


def _dd_case(name, rank, overlap):
    from sweep.parallel import MeshTopology, ModelParallel

    topo = MeshTopology(py=1, px=WORLD, shot_groups=1, world_size=WORLD, rank=rank)
    ddp = ModelParallel(_prop(name), topo, overlap=overlap)
    models = _models(name)
    src, rec = _geometry(name)
    syn = ddp(_wavelet(), src, rec, models=models)
    full = list(syn.shape)
    full[-2] = rec.shape[1]                  # nrec is axis -2 of the record
    syn.backward(gradient=_adjoint(full)[..., list(ddp.own_receiver_indices), :])
    grads = [m.grad for m in models]
    for g in grads:                          # each rank filled only its own tile
        dist.all_reduce(g)
    return {"record": ddp.gather_record(syn.detach()), "grads": grads}


def _worker(rank, init_file, out_dir, cases):
    from sweep.parallel import ModelParallel

    ran = []
    overlapped = ModelParallel._run_dd_loop_overlapped

    def spy(self, *args, **kwargs):
        ran.append(rank)
        return overlapped(self, *args, **kwargs)

    ModelParallel._run_dd_loop_overlapped = spy
    dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=WORLD,
                            rank=rank, timeout=datetime.timedelta(seconds=120))
    try:
        for name, overlap in cases:
            ran.clear()
            try:
                res = _dd_case(name, rank, overlap)
                res["overlapped"] = len(ran)
            except Exception:
                res = {"error": traceback.format_exc()}
            torch.save(res, os.path.join(out_dir, _result_name(name, overlap, rank)))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _result_name(name, overlap, rank):
    return f"{name}.overlap{int(overlap)}.rank{rank}.pt"


@pytest.fixture(scope="module")
def dd_runs(tmp_path_factory):
    """Every case through one spawn of the two ranks: ``(out_dir, cases)``."""
    served = {n for n in SHAPES if _host_core_serves(n)}
    cases = [(n, ov) for n, ov in CASES if n in served]
    if not cases:
        pytest.skip("impl='c' does not run on the CPU in this build (no host core)")
    out = tmp_path_factory.mktemp("dd_two_rank_cpu")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, str(out / "store"), str(out), cases))
             for r in range(WORLD)]
    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setenv("CUDA_VISIBLE_DEVICES", "")     # the ranks see no GPU
        for p in procs:
            p.start()
    deadline = time.monotonic() + DEADLINE_S
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.terminate()
    if hung:
        pytest.fail(f"{len(hung)} rank(s) did not finish in {DEADLINE_S} s")
    codes = [p.exitcode for p in procs]
    if any(codes):
        pytest.fail(f"rank exit codes {codes}")
    return out, cases


@pytest.mark.parametrize("name,overlap", CASES,
                         ids=[f"{n}-{'overlap' if ov else 'serial'}" for n, ov in CASES])
def test_two_rank_cpu_dd_matches_single_domain(dd_runs, name, overlap):
    out, cases = dd_runs
    if (name, overlap) not in cases:
        pytest.skip(f"the host core does not run {name}")
    res = [torch.load(out / _result_name(name, overlap, r)) for r in range(WORLD)]
    want_overlapped = 1 if overlap else 0
    for r, x in enumerate(res):
        if "error" in x:
            pytest.fail(f"rank {r}:\n{x['error']}", pytrace=False)
        assert x["overlapped"] == want_overlapped, (
            f"rank {r}: the overlapped forward ran {x['overlapped']} time(s) with "
            f"overlap={overlap}, expected {want_overlapped}")

    models = _models(name)
    src, rec = _geometry(name)
    syn = _prop(name)(_wavelet(), src, rec, models=models)
    syn.backward(gradient=_adjoint(syn.shape))

    _assert_equal(res[0]["record"], syn.detach(), "record")
    for i, (got, m) in enumerate(zip(res[0]["grads"], models)):
        _assert_equal(got, m.grad, f"gradient of model {i}")


def _assert_equal(got, want, what):
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    assert torch.equal(got, want), (
        f"{what}: two-rank DD differs from one domain, max |diff| "
        f"{(got - want).abs().max().item():.3e} (max |ref| {want.abs().max().item():.3e})")
