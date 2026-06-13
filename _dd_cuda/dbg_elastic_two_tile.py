"""Per-step bisect of the elastic two-tile DD mismatch (FS off, src 14)."""
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

sys.path.insert(0, str(REPO_ROOT / "test"))
import test_dd_elastic_two_tile as T  # noqa: E402

NZ, NX, NT, DT = T.NZ, T.NX, T.NT, T.DT
PAD, W, M, NWF, NPHYS = T.PAD, T.W, T.M, T.NWF, T.NPHYS

FIELD_NAMES = [
    "vx", "vz", "sxx", "szz", "sxz",
    "m_vxx", "m_vxz", "m_vzx", "m_vzz",
    "m_sxxx", "m_sxxz", "m_szzx", "m_szzz", "m_sxzx", "m_sxzz",
]


def main(src_gx=14, free_surface=False):
    nxp = NX // 2
    models_np = T.global_models()
    wavelet = T.ricker(NT, DT, scale=T.WAVELET_SCALE)
    src_z = 12
    rec_gx = list(range(2, NX - 2, 6))
    rec_z = 2

    full_sources = np.array([[[src_gx, src_z]]], dtype=np.int32)
    full_receivers = np.array([[[gx, rec_z] for gx in rec_gx]], dtype=np.int32)
    prop_full = T.make_prop((NZ, NX), free_surface)
    runner_full, record_full = T.make_runner(
        prop_full, wavelet, full_sources, full_receivers, models_np
    )

    runners = []
    for xi in range(2):
        topo = T.MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * nxp
        tile_models = [a[:, x0:x0 + nxp].copy() for a in models_np]
        owns_src = (x0 <= src_gx < x0 + nxp)
        if owns_src:
            tile_sources = np.array([[[src_gx - x0, src_z]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)
        own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
        tile_receivers = np.array(
            [[[gx - x0, rec_z] for gx in own_rec]], dtype=np.int32
        )
        prop = T.make_prop((NZ, nxp), free_surface, topo=topo)
        runner, record = T.make_runner(
            prop, tile_wavelet, tile_sources, tile_receivers, tile_models
        )
        T.fix_tile_models(runner.p, runner_full.p, x0)
        runners.append(runner)

    # model fix sanity
    for xi, r in enumerate(runners):
        for mi, (mt, mf) in enumerate(zip(list(r.p.models),
                                          list(runner_full.p.models))):
            x0 = xi * nxp
            assert torch.equal(mt, mf[..., x0:x0 + mt.size(-1)]), (xi, mi)
    print("model halo fix verified")

    r0, r1 = runners
    lo, hi = PAD, PAD + nxp

    def compare(it, where):
        bad = []
        for xi, r in enumerate(runners):
            for f in range(NWF):
                ref = runner_full.L[f][..., PAD + xi * nxp: PAD + xi * nxp + nxp]
                got = r.L[f][..., lo:hi]
                if not torch.equal(got, ref):
                    d = (got - ref).abs()
                    nz_ = (d > 0).nonzero()
                    zmin = nz_[:, 2].min().item(); zmax = nz_[:, 2].max().item()
                    xmin = nz_[:, 3].min().item(); xmax = nz_[:, 3].max().item()
                    bad.append(
                        f"tile{xi} {FIELD_NAMES[f]} n={nz_.size(0)} "
                        f"maxabs={d.max().item():.3e} "
                        f"z[{zmin},{zmax}] xloc[{xmin},{xmax}]"
                    )
        if bad:
            print(f"it={it} ({where}): " + " | ".join(bad))
        return bool(bad)

    with torch.no_grad():
        for it in range(NT):
            runner_full.run_to(it + 1)
            r0.run_to(it + 1)
            r1.run_to(it + 1)
            pre = compare(it, "pre-exchange(interior)")
            for f in range(NPHYS):
                a, b = r0.L[f], r1.L[f]
                a[..., hi:hi + W] = b[..., lo:lo + W]
                b[..., lo - W:lo] = a[..., hi - W:hi]
            if pre:
                if it > 3:
                    break

    print("done")


if __name__ == "__main__":
    main()
