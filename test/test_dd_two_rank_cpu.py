"""Two-rank domain decomposition on the CPU (host core, gloo) vs one domain.

An x cut with every source clear of it takes ModelParallel's overlapped
forward: phase 1 computes the cut strips, their shipment starts, phase 2
computes the rest.  That loop was written against CUDA streams and
``_overlap_ok`` checks the cut geometry only, so a CPU tile reached
``torch.cuda.Stream()`` -- and a process without CUDA died there.

Both ranks run with no GPU visible and the source mid tile 0, so the
overlapped loop is the one that runs (asserted).  The gathered record and the
summed tile gradients must equal the single-domain propagator's.

DD steps its tiles with ``impl='c'``; where that cannot run on the CPU (no host
core in this build) the test skips.
"""
from __future__ import annotations

import datetime
import os
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


def _dd_case(name, rank):
    from sweep.parallel import MeshTopology, ModelParallel

    topo = MeshTopology(py=1, px=WORLD, shot_groups=1, world_size=WORLD, rank=rank)
    ddp = ModelParallel(_prop(name), topo)
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


def _worker(rank, init_file, out_dir, names):
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
        for name in names:
            ran.clear()
            try:
                res = _dd_case(name, rank)
                res["overlapped"] = len(ran)
            except Exception:
                res = {"error": traceback.format_exc()}
            torch.save(res, os.path.join(out_dir, f"{name}.rank{rank}.pt"))
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def dd_runs(tmp_path_factory):
    """Both equations through one spawn of the two ranks: ``(out_dir, names)``."""
    names = [n for n in SHAPES if _host_core_serves(n)]
    if not names:
        pytest.skip("impl='c' does not run on the CPU in this build (no host core)")
    out = tmp_path_factory.mktemp("dd_two_rank_cpu")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, str(out / "store"), str(out), names))
             for r in range(WORLD)]
    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setenv("CUDA_VISIBLE_DEVICES", "")     # the ranks see no GPU
        for p in procs:
            p.start()
    for p in procs:
        p.join(timeout=300)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.terminate()
    if hung:
        pytest.fail(f"{len(hung)} rank(s) did not finish in 300 s")
    codes = [p.exitcode for p in procs]
    if any(codes):
        pytest.fail(f"rank exit codes {codes}")
    return out, names


@pytest.mark.parametrize("name", list(SHAPES))
def test_two_rank_cpu_dd_matches_single_domain(dd_runs, name):
    out, names = dd_runs
    if name not in names:
        pytest.skip(f"the host core does not run {name}")
    res = [torch.load(out / f"{name}.rank{r}.pt") for r in range(WORLD)]
    for r, x in enumerate(res):
        if "error" in x:
            pytest.fail(f"rank {r}:\n{x['error']}", pytrace=False)
        assert x["overlapped"] == 1, f"rank {r} did not take the overlapped forward"

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
