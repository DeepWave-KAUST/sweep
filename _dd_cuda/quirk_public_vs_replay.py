"""Root-cause repro: acoustic3d public Warpper forward vs stepped replay.

Hypothesis: the public BS forward passes an EMPTY wavefields list, so C++
AcousticWavefieldTensor::allocate() (which never allocates psi*n) leaves
double_buffer_psi=false and the kernel takes the legacy in-place psi branch
  (f.psixn ? f.psixn : f.psix)[idx] = ...
which RACES against the gradient<>(f.psix, ...) neighbour reads in the same
launch.  The stepped replay binds 12 tensors -> bind() sets
double_buffer_psi=true -> race-free read-old/write-new branch.

Experiments (expected outcomes BEFORE -> AFTER the allocate() fix that
passes double_buffer_psi=true at the forward call sites):
  E1  public vs replay-12 (the reported quirk)        DIFF   -> EQUAL
  E2  public vs replay-9  (legacy in-place branch)    DIFF*  -> DIFF*
  E3  public vs replay-empty (internal allocate)      DIFF*  -> EQUAL
  E4  replay-9 vs replay-9   (legacy determinism)     DIFF*  -> DIFF*
  E5  replay-12 vs replay-12 (race-free determinism)  EQUAL  -> EQUAL
  E6  public vs public (fresh prop)                   DIFF*  -> EQUAL
  E7  lockstep 9 vs 12: first diff is u/zeta in the y/z-PML while psi is
      still bitwise clean (the dpsi neighbour-read race signature)
  E8  2D control: public vs replay-7 vs replay-9      EQUAL (race never
      manifests at 48x56 in 2-D on this GPU)
* = live race: every legacy-branch run differs from every other run.
Verified 2026-06-12 on RTX 6000 Ada: post-fix column reproduced exactly.
"""

import sys

import numpy as np
import torch

SRC = "/home/wangs0j/sweep-local/sweep-dd-cuda/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sweep  # noqa: E402
assert sweep.__file__.startswith(SRC), f"wrong sweep: {sweep.__file__}"

from sweep.equations import Acoustic, Acoustic3D  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
    rotate_wavefield_roles,
)

DEV = torch.device("cuda")
NZ, NY, NX = 24, 20, 32
NT, DT, SO, ABCN = 60, 0.0015, 4, 10
M = SO // 2
PAD = ABCN + M


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def make_prop(ndim):
    if ndim == 3:
        eq = Acoustic3D(spatial_order=SO, device=DEV, backend="torch")
        shape = (NZ, NY, NX)
    else:
        eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
        shape = (48, 56)
    return PropTorch(
        eq, backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
        source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": True, "storage": "gpu",
                                "transfer_interval": 1,
                                "pinned_memory": False},
    )


def geometry(ndim):
    wavelet = ricker(NT, DT)
    if ndim == 3:
        vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None, None]
        vp = np.broadcast_to(vp, (NZ, NY, NX)).copy()
        vp[8:14, 5:15, 10:22] += 180.0
        src = np.array([[[NX // 2, NY // 2, NZ // 4]]], dtype=np.int32)
        rec = np.array([[[gx, NY // 2, 2] for gx in range(2, NX - 2, 5)]],
                       dtype=np.int32)
    else:
        nz, nx = 48, 56
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None]
        vp = np.broadcast_to(vp, (nz, nx)).copy()
        src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32)
    return vp, src, rec, wavelet


def capture(prop):
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(params):
        out = orig(params)
        cap["fp"], cap["raw"] = params, out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def public_run(ndim):
    prop = make_prop(ndim)
    cap = capture(prop)
    vp, src, rec, wavelet = geometry(ndim)
    models = [torch.tensor(vp, device=DEV, requires_grad=True)]
    prop(wavelet, src, rec, models=models)
    fp = cap["fp"]
    assert len(list(fp.wavefields)) == 0, \
        f"premise broken: public BS run bound {len(list(fp.wavefields))} wavefields"
    return cap


def snap(cap):
    fp = cap["fp"]
    rec = fp.record_out
    if rec is None or not rec.numel():
        rec = cap["raw"][2]
    return {
        "ring": [t.clone() for t in fp.boundary_gpu],
        "record": rec.clone(),
        "last_two": fp.last_two.clone(),
    }


def replay(cap, n_wf, ndim=3):
    """n_wf: 0 = empty list (C++ internal allocate); 7/9 (2D), 9/12 (3D)."""
    fp, func = cap["fp"], cap["func"]
    rec = torch.zeros_like(cap["raw"][2])
    fp.record_out = rec
    for t in fp.boundary_gpu:
        t.zero_()
    fp.last_two.zero_()
    dbuf = (n_wf == 12) if ndim == 3 else (n_wf == 9)
    if n_wf == 0:
        fp.wavefields = []
        fp.it_begin, fp.it_end = 0, NT
        with torch.no_grad():
            func(fp)
        fp.it_begin, fp.it_end = 0, -1
        return snap(cap), None
    L = [torch.zeros_like(fp.models[0]) for _ in range(n_wf)]
    r = SteppedBindingRunner(func, fp, L,
                             acoustic_psi_pairs(ndim) if dbuf else ())
    with torch.no_grad():
        r.run_to(NT)
    return snap(cap), L


def cmp(tag, sa, sb):
    bad = []
    for key in ("ring", "record", "last_two"):
        a, b = sa[key], sb[key]
        pairs = list(zip(a, b)) if isinstance(a, list) else [(a, b)]
        for i, (x, y) in enumerate(pairs):
            ne = (x != y)
            n = int(ne.sum())
            if n:
                d = (x - y).abs().max().item()
                bad.append(f"{key}[{i}]: {n}/{x.numel()} elems, max|d|={d:.3e}")
    status = "EQUAL (bitwise)" if not bad else "DIFFER"
    print(f"  {tag}: {status}")
    for line in bad[:6]:
        print(f"      {line}")
    return not bad


def ring_first_diff_step(sa, sb):
    """transfer_interval=1 -> dim0 of each ring tensor is the step slot."""
    first = None
    for i, (x, y) in enumerate(zip(sa["ring"], sb["ring"])):
        ne = (x != y).reshape(x.shape[0], -1).any(dim=1)
        steps = ne.nonzero().flatten()
        if steps.numel():
            s = int(steps[0])
            if first is None or s < first[1]:
                first = (i, s)
    return first


def main():
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    print("\n=== 3-D acoustic (nz,ny,nx)=(24,20,32) abcn=10 so=4 nt=60 BS gpu ===")
    cap = public_run(3)
    pub = snap(cap)
    fp = cap["fp"]
    print(f"  ring tensors: {[tuple(t.shape) for t in fp.boundary_gpu]}")
    print(f"  last_two: {tuple(fp.last_two.shape)}")

    s12a, _ = replay(cap, 12)
    s12b, _ = replay(cap, 12)
    s9a, _ = replay(cap, 9)
    s9b, _ = replay(cap, 9)
    s0, _ = replay(cap, 0)

    print("E1 public vs replay-12 (the quirk):")
    e1 = cmp("pub vs 12", pub, s12a)
    if not e1:
        fd = ring_first_diff_step(pub, s12a)
        print(f"      first differing ring slot: tensor {fd[0]}, step {fd[1]}")
    print("E2 public vs replay-9 (legacy in-place path, same binding style):")
    cmp("pub vs 9", pub, s9a)
    print("E3 public vs replay-empty (internal allocate again):")
    cmp("pub vs empty", pub, s0)
    print("E4 replay-9 determinism:")
    cmp("9 vs 9", s9a, s9b)
    print("E5 replay-12 determinism:")
    cmp("12 vs 12", s12a, s12b)

    print("E6 public vs public (fresh prop instance):")
    cap2 = public_run(3)
    pub2 = snap(cap2)
    cmp("pub vs pub'", pub, pub2)

    # ---------------- E7: lockstep 9 vs 12, first differing step ----------
    print("E7 lockstep 9 vs 12 (state compared after every step):")
    fp, func = cap["fp"], cap["func"]
    rec = torch.zeros_like(cap["raw"][2])
    fp.record_out = rec
    for t in fp.boundary_gpu:
        t.zero_()
    fp.last_two.zero_()
    L9 = [torch.zeros_like(fp.models[0]) for _ in range(9)]
    L12 = [torch.zeros_like(fp.models[0]) for _ in range(12)]
    r9 = SteppedBindingRunner(func, fp, L9, ())
    r12 = SteppedBindingRunner(func, fp, L12, acoustic_psi_pairs(3))
    nz_rt, ny_rt, nx_rt = fp.models[0].shape[-3:]
    print(f"      runtime grid (nz,ny,nx)=({nz_rt},{ny_rt},{nx_rt}), PML pad={PAD}")
    hit = 0
    with torch.no_grad():
        for k in range(1, NT + 1):
            r9.run_to(k)
            r12.run_to(k)
            rot9 = rotate_wavefield_roles(L9, k)
            rot12 = rotate_wavefield_roles(L12, k, psi_pairs=acoustic_psi_pairs(3))
            fields = {
                "u_prev": (rot9[0], rot12[0]),
                "u_now": (rot9[1], rot12[1]),
                "psix": (rot9[3], rot12[3]),
                "psiz": (rot9[4], rot12[4]),
                "zetax": (rot9[5], rot12[5]),
                "zetaz": (rot9[6], rot12[6]),
                "psiy": (rot9[7], rot12[7]),
                "zetay": (rot9[8], rot12[8]),
            }
            report = {}
            for name, (a, b) in fields.items():
                n = int((a != b).sum())
                if n:
                    report[name] = (n, (a - b).abs().max().item())
            if report:
                hit += 1
                print(f"      step {k}: " + ", ".join(
                    f"{nm} {n} cells max|d|={d:.2e}" for nm, (n, d) in report.items()))
                for name, (a, b) in fields.items():
                    ne = (a != b)
                    if not ne.any():
                        continue
                    coords = ne.nonzero()[:8]
                    locs = []
                    for c in coords:
                        iz, iy, ix = int(c[-3]), int(c[-2]), int(c[-1])
                        zone = []
                        if ix < PAD or ix >= nx_rt - PAD:
                            zone.append("x")
                        if iy < PAD or iy >= ny_rt - PAD:
                            zone.append("y")
                        if iz < PAD or iz >= nz_rt - PAD:
                            zone.append("z")
                        locs.append(f"(z{iz},y{iy},x{ix})pml[{''.join(zone) or '-'}]")
                    print(f"        {name}: {' '.join(locs)}")
                if hit >= 3:
                    print("      ... (stopping detail after 3 differing steps)")
                    break

    # ---------------- E8: 2D control --------------------------------------
    print("\n=== 2-D control (48x56) abcn=10 so=4 nt=60 BS gpu ===")
    cap2d = public_run(2)
    pub2d = snap(cap2d)
    s9, _ = replay(cap2d, 9, ndim=2)
    s7, _ = replay(cap2d, 7, ndim=2)
    print("E8a public vs replay-9 (double-buffered):")
    cmp("pub vs 9", pub2d, s9)
    print("E8b public vs replay-7 (legacy in-place):")
    cmp("pub vs 7", pub2d, s7)


if __name__ == "__main__":
    main()
