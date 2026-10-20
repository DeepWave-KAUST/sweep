"""SEM DD distributed check (not pytest-collected; launch with torchrun).

    torchrun --nproc_per_node=P test/sem/dd_api_check_sem.py [2d|3d]

Every rank steps its own strip through the impl='c' stepped loop with NCCL
interface exchanges; rank 0 also runs the FUSED single-domain C driver and
grades the assembled record three-way, mirroring FD ``dd_api_check.py``:

    PASS      rel <= 25x the fused driver's own run-to-run atomics floor
    PASS_TOL  rel <= 1e-4
    FAIL      otherwise

(SEM assembles by SUMMING interface contributions, so FD's bitwise bar is a
copy-semantics privilege this path never had.)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import torch.distributed as dist

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import AcousticPotentialSEM2D, AcousticSEM3D
from sweep.sem.prop_c import _SEMAcousticCpmlC

NO, DH, F0 = 4, 25.0, 8.0
# the models; the absorbing layer outside them (3 elements in 2-D, 2 in 3-D,
# none on the free top) makes the meshes 6 x 12 and 4 x 5 x 6 elements
NZ, NX = 13, 25
NT, DT = 250, 5e-4
# the 3-D case: a cut is a YZ PLANE, so ny > 1 is what makes the check
# meaningful (a plane collapsed to one row would pass any ny=1 geometry)
NZ3, NY3, NX3 = 9, 5, 9
NT3, DT3 = 200, 1.2e-3


def _wavelet(nt, dt):
    t = np.arange(nt) * dt
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def _setup_2d(memory=None):
    prop = S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=3 * NO, f_dominant=F0,
                     dev="cuda", impl="c")
    torch.manual_seed(0)
    vp = (2000.0 + 300.0 * torch.rand(NZ, NX)).cuda()
    eq, _ = prop._resolve_model_batch([vp])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + 5 * NO + 1
    rec = (torch.arange(3, nxg - 3, 4) + 1 * nxg).cuda()

    def fused(wav):
        pattern = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                              device=eq.mesh.ibool.device)
        pattern[int(src)] = 1.0
        parts, mats = SEMModelParallel(prop, StripTopology(1))._parts(
            eq, pattern, rec)
        c = dict(parts[0].c)
        c["bs"], c["bs_kind"] = memory == "boundary", 0
        with torch.no_grad():
            return _SEMAcousticCpmlC.apply(
                mats[0], mats[1], mats[2], wav, c).transpose(0, 1).cpu()

    return prop, vp, _wavelet(NT, DT), src, rec, fused


def _setup_3d(memory=None):
    # the memory is the propagator's own (a constructor option)
    prop = S.PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=DH, dt=DT3,
                     nt=NT3, n_order=NO, free_surface=("top",),
                     abcn=2 * NO, f_dominant=F0,
                     memory=BoundarySaving() if memory == "boundary" else None,
                     dev="cuda", impl="c")
    torch.manual_seed(0)
    vp = (2500.0 + 200.0 * torch.rand(NZ3, NY3, NX3)).cuda()
    _, nyg, nxg = prop._grid_shape()           # the model and its layer

    def gid(ix, iy, iz):
        return (iz * nyg + iy) * nxg + ix

    src = gid(nxg // 2, nyg // 2, 2 * NO)
    rec = torch.tensor([gid(x, y, NO)
                        for y in (nyg // 3, 2 * nyg // 3)
                        for x in range(3, nxg - 3, 3)]).cuda()

    def fused(wav):
        with torch.no_grad():
            return SEMModelParallel(prop, StripTopology(1))(wav, src, rec, [vp]).cpu()

    return prop, vp, _wavelet(NT3, DT3), src, rec, fused


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "2d"
    if case not in ("2d", "3d"):
        raise SystemExit(f"usage: dd_api_check_sem.py [2d|3d]; got {case!r}")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # the record is boundary-saving-independent (BS only changes how the
    # backward reconstructs), so this check stays forward/save-all; the BS
    # question is a GRADIENT one and lives in dd_grad_compare.py --memory
    prop, vp, wav, src, rec, fused = (_setup_3d() if case == "3d"
                                      else _setup_2d())

    ddp = SEMModelParallel(prop, StripTopology(world, rank=rank))
    with torch.no_grad():
        part = ddp(wav, src, rec, [vp])
    full = ddp.gather_record(part)

    if rank == 0:
        r1, r2 = fused(wav), fused(wav)
        assert float(r1.abs().max()) > 0.0, "reference record is all zeros"
        floor = float((r1 - r2).abs().max() / r1.abs().max())
        err = float((full - r1).abs().max() / r1.abs().max())
        if err <= max(25.0 * floor, 1e-6):
            verdict = "PASS"
        elif err <= 1e-4:
            verdict = "PASS_TOL"
        else:
            verdict = "FAIL"
        print(f"[sem-dd] case={case} world={world} nccl rel={err:.3e} "
              f"floor={floor:.3e}  {verdict}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
