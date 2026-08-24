"""3-D elastic 1x2 DD half-step bitwise audit (FS off).

Pass 1: drive run_phase(1) -> v-exchange -> run_phase(2) -> s-exchange,
compare every slot's owned region vs full per half step, find the first
diverging (it, phase, slot, cell).

Pass 2: rebuild fresh runners, stop right BEFORE phase 2 of the bad step,
dump every input of the syy update at the diverging cell for BOTH runs
(bit hex) incl. CPML a/b coefficients, then run phase 2 and dump outputs.

Usage: python dbg_elastic_3d_audit.py [src_gx]   (default 16 = on-cut column)
"""
from pathlib import Path
import struct
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "test"))

import test_dd_elastic_tiles_3d as T  # noqa: E402

NZ, NY, NX, NT, DT = T.NZ, T.NY, T.NX, T.NT, T.DT
PAD, M, NWF, NPHYS = T.PAD, T.M, T.NWF, T.NPHYS
NXP = NX // 2

NAMES = [
    "vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz",
    "m_vxx", "m_vxy", "m_vxz", "m_vyx", "m_vyy", "m_vyz",
    "m_vzx", "m_vzy", "m_vzz",
    "m_sxxx", "m_sxxy", "m_sxxz", "m_syyx", "m_syyy", "m_syyz",
    "m_szzx", "m_szzy", "m_szzz", "m_sxyx", "m_sxyy", "m_sxyz",
    "m_sxzx", "m_sxzy", "m_sxzz", "m_syzx", "m_syzy", "m_syzz",
]
PML_NAMES = ["az", "bz", "azh", "bzh", "ay", "by", "ayh", "byh",
             "ax", "bx", "axh", "bxh"]


def fhex(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def show(tag, tile_val, full_val):
    t, f = float(tile_val), float(full_val)
    mark = "" if fhex(t) == fhex(f) else "   <<< DIFFERS"
    print(f"  {tag:22s} tile={t: .9e} {fhex(t)}   full={f: .9e} {fhex(f)}{mark}")


def build(src_gx, free_surface=False):
    models_np = T.global_models()
    wavelet = T.ricker(NT, DT, scale=T.WAVELET_SCALE)
    src_g = (src_gx, NY // 2, NZ // 4)
    rec_g = [(gx, NY // 2, 2) for gx in range(2, NX - 2, 5)]

    full_src = np.array([[list(src_g)]], dtype=np.int32)
    full_rec = np.array([[list(r) for r in rec_g]], dtype=np.int32)
    prop_full = T.make_prop((NZ, NY, NX), free_surface)
    runner_full, _ = T.make_runner(prop_full, wavelet, full_src, full_rec, models_np)

    tiles = []
    for xi in range(2):
        topo = T.MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP
        tile_models = [a[:, :, x0:x0 + NXP].copy() for a in models_np]
        sx, sy, sz = src_g
        if x0 <= sx < x0 + NXP:
            t_src = np.array([[[sx - x0, sy, sz]]], dtype=np.int32)
            t_wav = wavelet
        else:
            t_src = np.array([[[1, 1, 1]]], dtype=np.int32)
            t_wav = np.zeros_like(wavelet)
        own = [(i, r) for i, r in enumerate(rec_g) if x0 <= r[0] < x0 + NXP]
        t_rec = np.array([[[r[0] - x0, r[1], r[2]] for _, r in own]], dtype=np.int32)
        prop = T.make_prop((NZ, NY, NXP), free_surface, topo=topo)
        runner, _ = T.make_runner(prop, t_wav, t_src, t_rec, tile_models)
        T.fix_tile_models(runner.p, runner_full.p, 0, x0)
        runner.p.cut_face_mask = 1 if xi == 1 else 2
        tiles.append(runner)
    return runner_full, tiles


def make_helpers(runner_full, tiles):
    r0, r1 = tiles
    lo, hi = PAD, PAD + NXP

    def exchange(slots):
        for f in slots:
            a, b = r0.L[f], r1.L[f]
            a[..., hi:hi + M].copy_(b[..., lo:lo + M])
            b[..., lo - M:lo].copy_(a[..., hi - M:hi])

    def compare(it, where, verbose=True):
        bad = []
        for xi, r in enumerate(tiles):
            for f in range(NWF):
                ref = runner_full.L[f][..., PAD + xi * NXP: PAD + xi * NXP + NXP]
                got = r.L[f][..., lo:hi]
                if not torch.equal(got, ref):
                    d = (got - ref).abs()
                    nz_ = (d > 0).nonzero()
                    bad.append((xi, f, d.max().item(), nz_.size(0), nz_[0].tolist()))
        if bad and verbose:
            for xi, f, dmax, n, first in bad:
                print(f"it={it} {where}: tile{xi} slot{f}={NAMES[f]} "
                      f"n={n} maxabs={dmax:.3e} first(B,C,z,y,xloc)={first}")
        return bad

    return exchange, compare


def audit_cell(runner_full, r1, tag, zz, yy, xloc):
    xt = PAD + xloc
    xf = PAD + NXP + xloc
    print(f"\n=== {tag}: tile1 run(z={zz},y={yy},x={xt}) vs "
          f"full run(z={zz},y={yy},x={xf}) ===")
    Lt, Lf = r1.L, runner_full.L

    def cell(slot, z, y, x_t, x_f, ctag):
        show(ctag, Lt[slot][0, 0, z, y, x_t].item(), Lf[slot][0, 0, z, y, x_f].item())

    cell(4, zz, yy, xt, xf, "syy")
    for dy in (-2, -1, 0, 1):
        cell(1, zz, yy + dy, xt, xf, f"vy[y{dy:+d}]")
    for dx in (-2, -1, 0, 1):
        cell(0, zz, yy, xt + dx, xf + dx, f"vx[x{dx:+d}]")
    for dz in (-2, -1, 0, 1):
        cell(2, zz + dz, yy, xt, xf, f"vz[z{dz:+d}]")
    for slot, nm in ((9, "m_vxx"), (13, "m_vyy"), (17, "m_vzz")):
        cell(slot, zz, yy, xt, xf, nm)
    for mi, nm in enumerate(["vp", "vs", "rho"]):
        show(nm, list(r1.p.models)[mi][0, 0, zz, yy, xt].item(),
             list(runner_full.p.models)[mi][0, 0, zz, yy, xf].item())
    pt = [t.detach().cpu().numpy().ravel() for t in list(r1.p.pml_vals)]
    pf = [t.detach().cpu().numpy().ravel() for t in list(runner_full.p.pml_vals)]
    for k, nm in enumerate(PML_NAMES):
        if "z" in nm:
            show(f"cpml.{nm}[iz={zz}]", pt[k][zz], pf[k][zz])
        elif "y" in nm:
            show(f"cpml.{nm}[iy={yy}]", pt[k][yy], pf[k][yy])
        else:
            show(f"cpml.{nm}[ix]", pt[k][xt], pf[k][xf])
    nx_t, nx_f = NXP + 2 * PAD, NX + 2 * PAD

    def interior(ix, nx):
        return (ix >= PAD + 1) and (ix < nx - PAD - 1)

    print(f"  interior-x predicate (3-D kernel, no cut-awareness): "
          f"tile1 ix={xt} -> {interior(xt, nx_t)} | full ix={xf} -> {interior(xf, nx_f)}")


def main(src_gx=16):
    print(f"--- PASS 1: find first divergence (src_gx={src_gx}, cut at gx=16) ---")
    runner_full, tiles = build(src_gx)
    exchange, compare = make_helpers(runner_full, tiles)
    first_bad = None
    with torch.no_grad():
        for it in range(NT):
            for r in tiles:
                r.run_phase(it + 1, 1)
            runner_full.run_phase(it + 1, 1)
            exchange(range(3))
            bad_v = compare(it, "after v-phase+vex")
            for r in tiles:
                r.run_phase(it + 1, 2)
            runner_full.run_phase(it + 1, 2)
            exchange(range(3, NPHYS))
            bad_s = compare(it, "after s-phase+sex")
            if bad_v or bad_s:
                first_bad = (it, bool(bad_v), (bad_v or bad_s))
                break
    if first_bad is None:
        print(f"CLEAN for all {NT} steps (bitwise).")
        return

    it_bad, in_vphase, bad = first_bad
    xi, f, dmax, n, first = bad[0]
    print(f"\nFIRST DIVERGENCE: it={it_bad} phase={'v' if in_vphase else 's'} "
          f"tile{xi} slot{f}={NAMES[f]} first(B,C,z,y,xloc)={first} maxabs={dmax:.3e}")

    if not (xi == 1 and f == 4 and not in_vphase):
        print("(divergence pattern differs from the canonical syy case; "
              "audit skipped — inspect manually)")
        return

    _, _, zz, yy, xloc = first
    print(f"\n--- PASS 2: rebuild, stop before phase 2 of it={it_bad}, audit ---")
    runner_full, tiles = build(src_gx)
    exchange, compare = make_helpers(runner_full, tiles)
    r1 = tiles[1]
    with torch.no_grad():
        for it in range(it_bad):
            for r in tiles:
                r.run_phase(it + 1, 1)
            runner_full.run_phase(it + 1, 1)
            exchange(range(3))
            for r in tiles:
                r.run_phase(it + 1, 2)
            runner_full.run_phase(it + 1, 2)
            exchange(range(3, NPHYS))
        for r in tiles:
            r.run_phase(it_bad + 1, 1)
        runner_full.run_phase(it_bad + 1, 1)
        exchange(range(3))
        audit_cell(runner_full, r1, f"INPUTS before phase2 of it={it_bad}",
                   zz, yy, xloc)
        for r in tiles:
            r.run_phase(it_bad + 1, 2)
        runner_full.run_phase(it_bad + 1, 2)
        audit_cell(runner_full, r1, f"OUTPUTS after phase2 of it={it_bad}",
                   zz, yy, xloc)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
