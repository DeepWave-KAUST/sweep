"""DD halo exchange over CUDA IPC (``ModelParallel(halo="ipc")``) vs NCCL: bit-identical.

The strips carry the same bytes whichever way they travel, so every rank's
tile record and tile gradients must be ``torch.equal`` between the two
transports.  Cases cover the per-tensor exchanger (acoustic), the overlapped
forward (source clear of the cut) and the serial one (source on the cut), the
batched field groups (elastic) and, with four GPUs, interior tiles with two
neighbours (px=4) and the cross-axis batch (2x2).

The source-clear acoustic case also runs ``overlap=False`` (serial
step-then-exchange) and must equal the overlapped forward.

Needs as many CUDA devices as ranks; otherwise skipped.
"""
from __future__ import annotations

import datetime
import os
import traceback

import numpy as np
import pytest
import torch

dist = pytest.importorskip("torch.distributed")
import torch.multiprocessing as mp  # noqa: E402


DT, NT, SO, ABCN = 1.0e-3, 120, 4, 10
NXP, NZ, NY = 28, 40, 24


def _cases(world):
    if world == 2:
        return [("acoustic", 2, 1, 2, "far"), ("acoustic", 2, 1, 2, "cut"),
                ("elastic", 2, 1, 2, "far"), ("acoustic", 3, 1, 2, "far"),
                ("elastic", 3, 1, 2, "far")]
    return [("acoustic", 2, 1, 4, "far"), ("elastic", 2, 1, 4, "far"),
            ("acoustic", 3, 2, 2, "far"), ("elastic", 3, 2, 2, "far")]


def _run(case, rank, world, dev, transport, overlap=True):
    from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D
    from sweep.parallel import MeshTopology, ModelParallel
    from sweep.propagator.torch import PropTorch

    fam, ndim, py, px, where = case
    nx = NXP * px
    ny = NY * py
    shape = (NZ, nx) if ndim == 2 else (NZ, ny, nx)
    grid = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if fam == "acoustic":
        models = [1800.0 + 600.0 * grid]
        st = rt = ["h1"]
        eq = Acoustic if ndim == 2 else Acoustic3D
        scale = 1.0
    else:
        models = [2200.0 + 400.0 * grid, 1200.0 + 200.0 * grid, 2000.0 + 100.0 * grid]
        st = ["sxx", "szz"] if ndim == 2 else ["sxx", "syy", "szz"]
        rt = ["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"]
        eq = Elastic if ndim == 2 else Elastic3D
        scale = 1.0e6
    # px=4: source in the middle of interior tile 1 and a record long enough for the
    # wave to reach tile 3 (420 m at >= 1800 m/s), so every cut carries non-zero halos
    nt = 300 if px == 4 else NT
    t = np.arange(nt, dtype=np.float32) * DT - 0.05
    a = (np.pi * 12.0 * t) ** 2
    wav = (scale * (1 - 2 * a) * np.exp(-a)).astype(np.float32)
    sx = NXP if where == "cut" else (NXP + NXP // 2 if px == 4 else NXP // 2)
    if ndim == 2:
        src = np.array([[[sx, NZ // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    else:
        # one receiver row per y-half so every tile of a 2x2 mesh owns receivers
        src = np.array([[[sx, ny // 2 - 3, NZ // 4]]], dtype=np.int32)
        rec = np.array([[[ix, iy, 2] for iy in (ny // 2 - 3, ny // 2 + 3)
                         for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    prop = PropTorch(eq(spatial_order=SO, device=dev, backend="torch"), backend="torch",
                     impl="c", shape=shape, dh=10.0, dt=DT, nt=nt, abcn=ABCN,
                     source_type=st, receiver_type=rt, dev=dev)
    if prop.impl != "c":
        # the eager fallback is not bit-reproducible across runs; this test is about
        # the compiled DD step, so a core built without this GPU's arch must fail loudly
        raise RuntimeError("impl='c' fell back to eager: no compiled core for this GPU")
    ddp = ModelParallel(prop, topo, halo=transport, overlap=overlap)
    y0, nyp = getattr(ddp, "y0", 0), getattr(ddp, "nyp", ny)
    tiles = []
    for m in models:
        sl = m[:, ddp.x0:ddp.x0 + ddp.nxp] if ndim == 2 else \
            m[:, y0:y0 + nyp, ddp.x0:ddp.x0 + ddp.nxp]
        tiles.append(torch.tensor(sl, device=dev, requires_grad=True))
    rec_tile = ddp.forward(torch.tensor(wav, device=dev), src, rec, models=tiles)
    rec_tile.backward(gradient=rec_tile.detach())
    torch.cuda.synchronize()
    return [rec_tile.detach().cpu()] + [m.grad.cpu() for m in tiles]


def _worker(rank, world, init_file, out_dir):
    from sweep.parallel import ModelParallel
    from sweep.parallel import fast_halo

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    counts = {"overlapped": 0, "ipc_batches": 0}
    overlapped = ModelParallel._run_dd_loop_overlapped
    batch_init = fast_halo._IpcBatch.__init__

    def spy_overlap(self, *a, **k):
        counts["overlapped"] += 1
        return overlapped(self, *a, **k)

    def spy_batch(self, *a, **k):
        counts["ipc_batches"] += 1
        return batch_init(self, *a, **k)

    ModelParallel._run_dd_loop_overlapped = spy_overlap
    fast_halo._IpcBatch.__init__ = spy_batch
    dist.init_process_group("nccl", init_method=f"file://{init_file}", world_size=world,
                            rank=rank, timeout=datetime.timedelta(seconds=180),
                            device_id=dev)
    res = {}
    try:
        for case in _cases(world):
            key = "/".join(map(str, case))
            try:
                out = {}
                for tr in ("nccl", "ipc"):
                    counts.update(overlapped=0, ipc_batches=0)
                    out[tr] = _run(case, rank, world, dev, tr)
                    out[tr + "_counts"] = dict(counts)
                if case[4] == "far" and case[0] == "acoustic" and case[1] == 2:
                    # the serial schedule against the overlapped one, same transport
                    counts.update(overlapped=0, ipc_batches=0)
                    serial = _run(case, rank, world, dev, "nccl", overlap=False)
                    assert counts["overlapped"] == 0
                    out["serial_equal"] = all(torch.equal(a, b)
                                              for a, b in zip(out["nccl"], serial))
                res[key] = {
                    "serial_equal": out.get("serial_equal", True),
                    "equal": all(torch.equal(a, b) for a, b in zip(out["nccl"], out["ipc"])),
                    "maxdiff": max(float((a - b).abs().max()) for a, b in
                                   zip(out["nccl"], out["ipc"])),
                    "nonzero": all(float(a.abs().max()) > 0 for a in out["nccl"][:2]),
                    "nccl_counts": out["nccl_counts"], "ipc_counts": out["ipc_counts"]}
            except Exception:
                res[key] = {"error": traceback.format_exc()}
            dist.barrier()
    finally:
        torch.save(res, os.path.join(out_dir, f"rank{rank}.pt"))
        dist.destroy_process_group()


def _spawn(world, tmp_path):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, world, str(tmp_path / "store"), str(tmp_path)))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.terminate()
    assert not hung, "a rank hung"
    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(world)]


@pytest.mark.parametrize("world", [2, 4])
def test_ipc_matches_nccl(world, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("needs NCCL")
    pytest.importorskip("torch.distributed._symmetric_memory")
    results = _spawn(world, tmp_path)
    for rank, res in enumerate(results):
        for key, r in res.items():
            assert "error" not in r, f"rank {rank} {key}:\n{r['error']}"
            assert r["nonzero"], f"rank {rank} {key}: empty record/gradient"
            assert r["equal"], f"rank {rank} {key}: ipc != nccl (max|d| {r['maxdiff']:.3e})"
            assert r["serial_equal"], f"rank {rank} {key}: overlap=False != overlap=True"
            assert r["ipc_counts"]["ipc_batches"] > 0, f"{key}: ipc never built a batch"
            assert r["nccl_counts"]["ipc_batches"] == 0, f"{key}: nccl built an ipc batch"
            assert r["ipc_counts"]["overlapped"] == r["nccl_counts"]["overlapped"]
    if world == 2:     # the source-clear acoustic case takes the overlapped forward
        assert results[0]["acoustic/2/1/2/far"]["ipc_counts"]["overlapped"] > 0
        assert results[0]["acoustic/2/1/2/cut"]["ipc_counts"]["overlapped"] == 0


def test_halo_name_checked_at_construction():
    """A wrong ``halo=`` fails in ``ModelParallel.__init__``, on every device."""
    from sweep.parallel.fast_halo import check_transport_name
    assert check_transport_name("nccl") == "nccl"
    assert check_transport_name("ipc") == "ipc"
    for bad in ("IPC", "nvshmem", None, 1):
        with pytest.raises(ValueError, match="halo="):
            check_transport_name(bad)
