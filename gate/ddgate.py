#!/usr/bin/env python3
"""Bit-exact before/after gate for the domain-decomposed propagator.

`bitgate.py` covers the single-domain propagators; nothing in it touches
`sweep/parallel/` or `sweep/propagator/_stepped.py`, which are exactly what P2
and P3 rewrite. This is their gate.

Two rungs, because they need different hardware:

* **`world_size=1`** -- a single-tile `ModelParallel`. Needs no
  ``torch.distributed`` and no second GPU, yet still drives the whole stepped
  machinery: the one-time capture, the lazy adjoint promotion, per-shot geometry
  rebinding, the per-family forward/backward step loops, and the wavefield-role
  rotation in ``_stepped.py``. That is the bulk of what a Python-side refactor
  can break, and it runs on the dev box.
* **`world_size>1`** -- real tiles, real NCCL halo exchange, real
  ``cut_face_mask``. Only this rung can catch a halo/cut-face regression, and it
  needs >= 2 GPUs (launch under ``torchrun``; see ``--ranks``).

Both rungs record the *tile* record and the *tile* model gradients and compare
them with ``torch.equal``.

Usage:
    . gate/env.sh
    $PY gate/ddgate.py --save gate/base_dd1.pt              # world=1, this box
    $PY gate/ddgate.py --compare gate/base_dd1.pt
    $PY gate/ddgate.py --verify-reproducible

    torchrun --standalone --nproc-per-node=2 gate/ddgate.py \
        --ranks 2 --save gate/base_dd2.pt                   # >=2 GPUs
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

DT = 0.0015
NT = 90
DH = 10.0
SO = 4
ABCN = 12


@dataclass(frozen=True)
class DDCfg:
    equation: str          # Acoustic | Acoustic3D | AcousticVRZ3D | Elastic | Elastic3D
    free_surface: bool
    py: int
    px: int
    #: elastic only. "stress" is the default acquisition (stress source,
    #: velocity receivers); "bodyforce" swaps in a velocity source.
    #:
    #: This is not variety for its own sake. The elastic DD backward has a
    #: CONDITIONAL halo exchange after its injection phase, guarded by
    #: `inj_cross` -- true when a body-force source writes recon velocity or a
    #: stress receiver writes adjoint stress, both of which phase 1 then reads
    #: across the cut. With the default acquisition the guard is FALSE, so that
    #: shipment never fires and no gate in the tree covered it. A wrong guard,
    #: or a group attached to the wrong phase, would have passed everything.
    src_kind: str = "stress"

    #: Where the source sits relative to the x-cut lattice, and therefore WHICH
    #: forward loop runs.
    #:
    #: The acoustic/VRZ forward has an overlapped variant that computes the cut
    #: strips first, ships them on a comm stream, and computes the interior
    #: while the exchange flies. `_forward_loop` only takes it when
    #: `_src_away_from_cuts` holds, because the source injection atomically adds
    #: into the very buffer the comm stream is packing.
    #:
    #: The default geometry put the source at `x = nx//2`, and the cut lattice
    #: is `k * (nx//px)` -- so for EVERY even px the source sat exactly ON a cut
    #: and the predicate was False. The overlapped path therefore ran in 0 of
    #: this gate's configs: a broken strip bound, a wrong exchange point or a
    #: mis-joined comm stream would have passed green. "off_cut" moves the
    #: source clear of every cut so the phased loop actually runs; "on_cut"
    #: keeps the original placement, because the serial fallback is a real path
    #: too and losing coverage of it to gain the other would be no better.
    src_pos: str = "on_cut"

    @property
    def id(self) -> str:
        fs = "fs" if self.free_surface else "nofs"
        # Keep the default spelling of the id unchanged so existing baselines
        # keep matching; only the new variant carries a suffix.
        kind = "" if self.src_kind == "stress" else f"|{self.src_kind}"
        # Keep the default spelling unchanged so existing baselines still match.
        pos = "" if self.src_pos == "on_cut" else f"|{self.src_pos}"
        return f"{self.equation}|{fs}|{self.py}x{self.px}{kind}{pos}"

    @property
    def world(self) -> int:
        return self.py * self.px


# ``ModelParallel`` accepts only equations whose CUDA forward AND backward honour
# the stepped range -- see ``_DD_EQUATIONS`` in dd_propagator.py. Anything else
# is refused at construction, so there is nothing to gate.
DD_EQUATIONS = ("Acoustic", "Acoustic3D", "AcousticVRZ3D", "Elastic", "Elastic3D")


def configs(ranks: int) -> list[DDCfg]:
    out = []
    for eq in DD_EQUATIONS:
        for fs in (False, True):
            if ranks == 1:
                out.append(DDCfg(eq, fs, 1, 1))
            else:
                out.append(DDCfg(eq, fs, 1, ranks))
                if eq.endswith("3D") and ranks % 2 == 0 and ranks >= 4:
                    out.append(DDCfg(eq, fs, 2, ranks // 2))
    # Body-force elastic: the only configuration that makes the elastic
    # backward's `inj_cross` shipment fire. See DDCfg.src_kind.
    for eq in ("Elastic", "Elastic3D"):
        px = 1 if ranks == 1 else ranks
        out.append(DDCfg(eq, False, 1, px, src_kind="bodyforce"))
    # Source clear of the cuts, so the acoustic/VRZ forward takes its
    # OVERLAPPED loop. See DDCfg.src_pos: without these the phased forward is
    # not executed by any config in this gate. Only meaningful at world > 1,
    # where `_overlap_ok` can be true at all.
    if ranks > 1:
        for eq in DD_EQUATIONS:
            for fs in (False, True):
                out.append(DDCfg(eq, fs, 1, ranks, src_pos="off_cut"))
    return out


def _shape(eq: str) -> tuple[int, ...]:
    # Physical interior stays real (not swallowed by the PML) so a cut face has
    # something meaningful on both sides of it.
    return (72, 96) if not eq.endswith("3D") else (40, 40, 48)


def _make(cfg: DDCfg, rank: int, dev: str):
    import sweep.equations as E
    from sweep.parallel import MeshTopology, ModelParallel
    from sweep.propagator.torch import PropTorch

    cls = getattr(E, cfg.equation)
    shape = _shape(cfg.equation)
    ndim = len(shape)
    elastic = cfg.equation.startswith("Elastic")
    eq = cls(spatial_order=SO, device=dev, backend="torch")
    src_t = ["sxx", "szz"] if elastic else ["h1"]
    if elastic and ndim == 3:
        src_t = ["sxx", "syy", "szz"]
    rec_t = (["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"]) if elastic else ["h1"]
    if elastic and cfg.src_kind == "bodyforce":
        src_t = list(rec_t)          # velocity source -> inj_cross is True

    prop = PropTorch(
        eq, backend="torch", impl="c", shape=shape, dev=dev, dh=DH, dt=DT,
        source_type=src_t, receiver_type=rec_t, abcn=ABCN,
        free_surface=cfg.free_surface, nt=NT, B=1,
    )
    topo = MeshTopology(py=cfg.py, px=cfg.px, shot_groups=1,
                        world_size=cfg.world, rank=rank)
    return ModelParallel(prop, topo), shape, ndim


def _models(cfg: DDCfg, shape, dev):
    """Deterministic global models; DD slices each rank's tile itself."""
    nz = shape[0]
    ramp = np.linspace(1800.0, 2400.0, nz, dtype=np.float32)
    vp = np.broadcast_to(ramp.reshape((nz,) + (1,) * (len(shape) - 1)), shape).copy()
    sl = tuple(slice(s // 3, 2 * s // 3) for s in shape)
    vp[sl] += 180.0
    out = [vp]
    if cfg.equation.startswith("Elastic"):
        out.append(vp / 1.73)                                   # vs
        out.append(np.full(shape, 2000.0, dtype=np.float32))    # rho
    elif cfg.equation.startswith("AcousticVRZ"):
        out.append(np.full(shape, 2000.0, dtype=np.float32))    # z (impedance)
    return [torch.tensor(m, device=dev) for m in out]


def _src_x(nx: int, px: int, pos: str) -> int:
    """Source x for a given placement, checked against the cut lattice.

    Cuts sit at ``k * (nx // px)``; the driver's predicate is
    ``|src_x - cut| > M`` for every cut, with ``M = spatial_order // 2``.
    "on_cut" reproduces the original ``nx // 2``, which for every even px IS a
    cut. "off_cut" is the midpoint of the first tile, which is the furthest a
    single source can get from both ``0`` and the first cut; it is asserted
    clear rather than assumed, so a future shape or px cannot silently put the
    gate back on the serial path."""
    if pos == "on_cut" or px == 1:
        return nx // 2
    nxp = nx // px
    x = nxp // 2
    cuts = [k * nxp for k in range(1, px)]
    assert all(abs(x - c) > SO // 2 for c in cuts), (
        f"off_cut source x={x} is within M={SO // 2} of a cut in {cuts} "
        f"(nx={nx}, px={px}) -- the phased forward would not run")
    return x


def _geometry(shape, ndim, px=1, src_pos="on_cut"):
    nx = shape[-1]
    sx = _src_x(nx, px, src_pos)
    if ndim == 2:
        nz = shape[0]
        src = np.array([[[sx, nz // 5]]], dtype=np.int64)
        rec = np.array([[[x, 3] for x in range(4, nx - 4, 5)]], dtype=np.int64)
    else:
        nz, ny = shape[0], shape[1]
        src = np.array([[[sx, ny // 2, nz // 5]]], dtype=np.int64)
        rec = np.array([[[x, ny // 2, 3] for x in range(4, nx - 4, 5)]], dtype=np.int64)
    return src, rec


def run_one(cfg: DDCfg, rank: int, dev: str) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    ddp, shape, ndim = _make(cfg, rank, dev)
    models = [m.clone().requires_grad_() for m in _models(cfg, shape, dev)]
    src, rec = _geometry(shape, ndim, cfg.px, cfg.src_pos)
    gen = torch.Generator(device="cpu").manual_seed(4321)
    wav = (torch.randn(NT, generator=gen) * 1e-2).to(dev)

    syn = ddp(wav, src, rec, models=models)
    # A fixed pseudo-random adjoint source, so the backward is exercised
    # independently of whatever the forward happened to produce.
    #
    # Drawn in the RAW record layout and converted, not drawn in syn's shape.
    # ModelParallel now returns the single-card layout (B, nt, nrec, nfield)
    # where it used to return (B, nrec, nt); torch.randn fills in MEMORY order,
    # so drawing at the new shape would put the same flat stream at different
    # (t, receiver) positions -- a different adjoint source, hence different
    # gradients, hence every stored baseline invalidated. Re-baselining is how a
    # real regression gets laundered, so the draw is pinned to the layout the
    # baselines were recorded in and every base_dd*.pt stays valid.
    from sweep.propagator._c import (_canonical_to_cuda_record,
                                     _cuda_record_to_canonical)
    adj_raw = (torch.randn(ddp.record.shape, generator=gen) * 1e-3).to(dev)
    syn.backward(gradient=_cuda_record_to_canonical(adj_raw))

    # WHICH forward loop actually ran. Recorded in the baseline so a future
    # change that silently drops back to the serial path is a gate FAILURE
    # rather than an invisible loss of coverage -- which is exactly how the
    # phased loop came to be executed by no config at all.
    phased = bool(getattr(ddp, "_overlap_ok", False)
                  and getattr(ddp, "_spec", None) is not None
                  and ddp._spec.forward_overlapped is not None
                  and ddp._src_away_from_cuts(
                      torch.as_tensor(src).reshape(1, -1, len(shape))))

    return {
        "forward_loop": "overlapped" if phased else "serial",
        # Stored in the RAW layout for the same reason the adjoint source is
        # drawn there: the baselines predate the layout change and must keep
        # comparing bit-for-bit across it.
        "record": _canonical_to_cuda_record(
            syn.detach(), ddp.record.ndim).float().cpu().contiguous(),
        "grads": {f"m{i}": m.grad.detach().float().cpu().contiguous()
                  for i, m in enumerate(models)},
        "shape": tuple(int(s) for s in shape),
    }


def _cmp(a: dict, b: dict) -> list[str]:
    fails = []
    if ("error" in a) != ("error" in b):
        return [f"error-status changed: {a.get('error','ok')!r} -> {b.get('error','ok')!r}"]
    if "error" in a:
        return [] if a["error"] == b["error"] else [
            f"error text changed: {a['error']!r} -> {b['error']!r}"]
    # A config that used to run the phased forward and now runs the serial one
    # produces the SAME numbers -- the two loops are bit-identical by design --
    # so nothing below would notice. Check it explicitly.
    # Only when BOTH sides recorded it: a baseline captured before this field
    # existed must keep comparing clean rather than fail on its absence.
    if "forward_loop" in a and "forward_loop" in b and a["forward_loop"] != b["forward_loop"]:
        fails.append(f"forward loop changed: {a['forward_loop']} -> "
                     f"{b['forward_loop']} (same numbers, lost coverage)")
    if not torch.equal(a["record"], b["record"]):
        d = (a["record"] - b["record"]).abs().max().item()
        n = int((a["record"] != b["record"]).sum())
        fails.append(f"record differs ({n} elems, max|d|={d:.3e})")
    for k in sorted(set(a["grads"]) | set(b["grads"])):
        if k not in a["grads"] or k not in b["grads"]:
            fails.append(f"grad[{k}] present on only one side")
            continue
        ga, gb = a["grads"][k], b["grads"][k]
        if ga.shape != gb.shape:
            fails.append(f"grad[{k}] shape {tuple(ga.shape)} -> {tuple(gb.shape)}")
        elif not torch.equal(ga, gb):
            d = (ga - gb).abs().max().item()
            n = int((ga != gb).sum())
            fails.append(f"grad[{k}] differs ({n} elems, max|d|={d:.3e})")
    return fails


def compare(before: dict, after: dict) -> int:
    npass = nfail = nmiss = nnew = nerr = 0
    for k in sorted(set(before) | set(after)):
        if k not in before:
            print(f"NEW      {k}")
            nnew += 1
            continue
        if k not in after:
            print(f"MISSING  {k}")
            nmiss += 1
            continue
        fails = _cmp(before[k], after[k])
        if fails:
            nfail += 1
            print(f"FAIL     {k}")
            for f in fails:
                print(f"           {f}")
        else:
            npass += 1
            if "error" in before[k]:
                nerr += 1
            print(f"pass     {k}" + ("  (both errored)" if "error" in before[k] else ""))
    tail = f"   [{nerr} compared ERROR TEXT only -- no numbers checked]" if nerr else ""
    print(f"\n{'='*70}\nPASS {npass}   FAIL {nfail}   MISSING {nmiss}   NEW {nnew}{tail}\n{'='*70}")
    return 0 if (nfail == 0 and nmiss == 0) else 1


def _run_all(cfgs, rank, dev):
    res = {}
    for i, cfg in enumerate(cfgs, 1):
        try:
            res[cfg.id] = run_one(cfg, rank, dev)
            mark = "."
        except Exception as exc:                      # noqa: BLE001
            res[cfg.id] = {"error": f"{type(exc).__name__}: {exc}"}
            mark = "E"
        if rank == 0:
            print(f"[{i:2d}/{len(cfgs)}] {mark} {cfg.id}", flush=True)
            if mark == "E":
                print(f"          -> {res[cfg.id]['error']}", flush=True)
        # Each config builds its own propagator; free it before the next one.
        torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", type=int, default=1,
                    help="world size; >1 must be launched under torchrun")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--verify-reproducible", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfgs = configs(args.ranks)
    if args.only:
        cfgs = [c for c in cfgs if args.only in c.id]
    if args.list:
        for c in cfgs:
            print(c.id)
        print(f"\n{len(cfgs)} configs at world={args.ranks}")
        return 0

    rank = 0
    dev = "cuda"
    if args.ranks > 1:
        import torch.distributed as dist
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dev = f"cuda:{os.environ['LOCAL_RANK']}"
        dist.init_process_group("nccl")

    if args.verify_reproducible:
        if rank == 0:
            print(f"# DD reproducibility: world={args.ranks}, {len(cfgs)} configs\n")
        a = _run_all(cfgs, rank, dev)
        if rank == 0:
            print("\n--- second pass ---\n")
        b = _run_all(cfgs, rank, dev)
        return compare(a, b) if rank == 0 else 0

    res = _run_all(cfgs, rank, dev)
    rc = 0
    if rank == 0:
        nerr = sum(1 for v in res.values() if "error" in v)
        print(f"\nran {len(res)} configs, {nerr} errored")
        if args.save:
            torch.save({"meta": {"ranks": args.ranks,
                                 "gpu": torch.cuda.get_device_name(0)},
                        "results": res}, args.save)
            print(f"saved -> {args.save}")
        if args.compare:
            blob = torch.load(args.compare, map_location="cpu", weights_only=False)
            rc = compare(blob["results"], res)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
