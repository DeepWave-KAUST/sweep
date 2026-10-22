#!/usr/bin/env python3
"""s_gradient='adjoint' under the eager default (torch.compile on) against uncompiled."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "test"))
import qp_aniso_consistency as Q  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

cfg = Q.CONFIGS["canonical"]
for cls in (Q.EQUATIONS["vti"], Q.EQUATIONS["tti"]):
    for sg in ("adjoint", "frozen"):
        outs = []
        for compile_ in (False, True):
            eq = cls(spatial_order=4, device="cuda", backend="torch", s_gradient=sg)
            prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=compile_), memory=Full(),
                             shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev="cuda",
                             nt=cfg["nt"], B=1)
            names = eq.models
            wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device="cuda")
            ms = [torch.tensor(a, device="cuda", requires_grad=True) for a in Q.model_arrays(names, cfg["shape"], "eps>delta")]
            src, rec = Q.geometry(cfg["shape"], 4, cfg["rec_stride"])
            d = prop(wav, src, rec, models=ms)
            d.pow(2).sum().backward()
            outs.append([d.detach()] + [m.grad.detach() for m in ms])
        rel = [float((a - b).norm() / b.norm()) for a, b in zip(outs[1], outs[0])]
        print(f"{cls.__name__} {sg}: compiled vs uncompiled rel (record, {', '.join(names)}) = "
              + " ".join(f"{r:.1e}" for r in rel), flush=True)
