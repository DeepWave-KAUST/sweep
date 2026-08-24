"""Stage-wise snapshots to find where cross-process bit-divergence enters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))

import numpy as np
import torch

import test_dd_backward_two_tile as T


def main(tag):
    out = {}

    # Stage 1: bare public forward only (no autograd backward)
    sources = np.array([[[T.SRC_GX, T.SRC_GZ]]], dtype=np.int32)
    receivers = np.array([[[gx, T.REC_GZ] for gx in T.REC_GX]], dtype=np.int32)
    prop = T.make_prop((T.NZ, T.NX))
    cap = T.capture_both(prop)
    models = [torch.tensor(T.vp_global(), device=T.DEV, requires_grad=True)]
    syn = prop(torch.as_tensor(T.ricker(T.NT, T.DT)), sources, receivers,
               models=models)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    torch.cuda.synchronize()
    out["s1_rec"] = cap["fwd_raw_out"][2].clone().cpu()
    out["s1_ring"] = [b.clone().cpu() for b in cap["fp"].boundary_gpu]
    out["s1_lt"] = cap["fp"].last_two.clone().cpu()

    # Stage 2: public autograd backward
    rec.sum().backward()
    torch.cuda.synchronize()
    out["s2_ring"] = [b.clone().cpu() for b in cap["fp"].boundary_gpu]
    out["s2_grad"] = models[0].grad.clone().cpu()

    torch.save(out, f"/tmp/stage_{tag}.pt")


if __name__ == "__main__":
    main(sys.argv[1])
