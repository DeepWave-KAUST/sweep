"""ElasticVRR's free surface is Elastic's.

With zero reflectivity and constant velocity the vector-reflectivity equation
is constant-density Elastic with rho = 1 (p = v), so with ``free_surface=True``
the two must still record the same thing -- including the Rayleigh wave along
the surface.  They used not to: ElasticVRR mirrored sxz and pz about the
integer surface row and zeroed the surface-row sxz, the two things Elastic had
dropped (Kristek, Moczo & Archuleta 2002, Table 1), which made its Rayleigh
wave 2-7 % slow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ElasticVRR
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

NZ, NX = 40, 96
DH, DT, NT = 10.0, 1.5e-3, 500
ABCN, SO = 20, 4


def _impls():
    impls = ["eager"]
    if torch.cuda.is_available():
        impls.append("c")
    return impls


def _run(cls, impl, models, receiver_type):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(use_ckpt=False, eager_options={"use_compile": False}) if impl == "eager" else {}
    prop = PropTorch(cls(spatial_order=SO, device=dev, backend="torch"), shape=(NZ, NX),
                     abcn=ABCN, dh=DH, dt=DT, impl=impl, dev=dev, free_surface=True,
                     receiver_type=receiver_type, **kw)
    t = np.arange(NT, dtype=np.float32) * DT - 0.06
    wav = torch.tensor((1e3 * ricker(t, f=10.0)).astype(np.float32), device=dev)
    src = np.array([[10, 2]])                                # 2 rows below the surface
    rec = np.array([[[ix, iz] for ix in range(20, NX - 4, 8) for iz in (0, 3)]])
    with torch.no_grad():
        return prop(wav, src, rec, models=[m.to(dev) for m in models])


@pytest.mark.parametrize("impl", _impls())
def test_zero_reflectivity_free_surface_matches_constant_density_elastic(impl):
    vp = torch.full((NZ, NX), 2000.0)
    vs = torch.full((NZ, NX), 1000.0)
    zero = torch.zeros((NZ, NX))
    evr = _run(ElasticVRR, impl, [vp, vs, zero, zero, zero, zero], ["px", "pz"])
    ela = _run(Elastic, impl, [vp, vs, torch.ones((NZ, NX))], ["vx", "vz"])
    rel = ((evr - ela).norm() / ela.norm()).item()
    # the old surface treatment gave 1.8 here; fp32 round-off is ~1e-6
    assert rel < 1e-4, f"ElasticVRR vs Elastic under a free surface: rel L2 {rel:.3e}"
