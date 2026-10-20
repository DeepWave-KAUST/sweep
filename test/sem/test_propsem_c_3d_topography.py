"""High-level PropSEM(topography=...) 3-D API on a Hex27 mesh.

Acoustic (AcousticSEM3DDeformed): impl='c' matches impl='eager' on the forward
and dL/dvp, and memory=BoundarySaving() matches the save-all gradient — all
through the PropSEM curved-top-surface build path. Elastic
(ElasticSEM3DDeformed): the eager PropSEM route builds the topography mesh and
runs (impl='c' elastic stays on SEMCudaPropagator, covered by
test_sem_c_elastic3d_deformed).

All comparisons: cos >= 1-1e-4.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.equations.elastic3d_deformed import ElasticSEM3DDeformed
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NY, NX = 25, 21, 29
DH, DT, NT, F0, AB = 20.0, 1.5e-3, 220, 8.0, 3
SRC = np.array([[14, 10, 3]], dtype=np.int64)
_rix = np.arange(4, NX - 4, 2)
REC = np.stack([_rix, np.full_like(_rix, 10), np.full_like(_rix, 4)], -1)[None]


def _ztop(x, y):
    return 60.0 * torch.exp(-(((x - NX * DH * 0.5) / 160.0) ** 2
                              + ((y - NY * DH * 0.5) / 140.0) ** 2))


def _ztop_rows():
    """``_ztop`` on the model columns: the surface row of each, (NY, NX)."""
    Y, X = torch.meshgrid(torch.arange(NY, dtype=torch.float64) * DH,
                          torch.arange(NX, dtype=torch.float64) * DH,
                          indexing="ij")
    return (_ztop(X, Y) / DH).numpy()


def _wav(scale=1.0e6):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 0.15)) ** 2
    return torch.tensor(scale * (1 - 2 * a) * np.exp(-a),
                        dtype=torch.float32, device="cuda")


def _vp():
    z = torch.arange(NZ)[:, None, None]
    x = torch.arange(NX)[None, None, :]
    return (1800 + 8.0 * z + 150 * torch.sin(x / 9.0)
            + 0 * torch.arange(NY)[None, :, None]).float().cuda()


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _run_ac(impl, mem):
    p = PropSEM(AcousticSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=DT, nt=NT,
                n_order=4, free_surface=True,
                abcn=AB * 4, f_dominant=F0, topography=_ztop_rows(), dev="cuda",
                dtype=torch.float32, impl=impl, memory=mem)
    vp = _vp().clone().requires_grad_(True)
    s = p(_wav(), SRC, REC, models=[vp])
    W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
    (W * s).sum().backward()
    return s.detach().cpu().numpy(), vp.grad.detach().cpu().numpy()


def test_propsem_topo3d_acoustic_c_matches_eager():
    se, ge = _run_ac("eager", None)
    sc, gc = _run_ac("c", None)
    assert _cos(sc, se) >= 1 - 1e-4, f"fwd cos={_cos(sc, se):.6f}"
    assert _cos(gc, ge) >= 1 - 1e-4, f"grad cos={_cos(gc, ge):.6f}"


def test_propsem_topo3d_acoustic_boundary_saving():
    _, ge = _run_ac("eager", None)
    _, gc = _run_ac("c", None)
    _, gcb = _run_ac("c", BoundarySaving())
    assert _cos(gcb, gc) >= 1 - 1e-4, f"c-BS vs c-full cos={_cos(gcb, gc):.6f}"
    assert _cos(gcb, ge) >= 1 - 1e-4, f"c-BS vs eager cos={_cos(gcb, ge):.6f}"


def test_propsem_topo3d_elastic_eager_builds_and_runs():
    # abcn=0: ElasticSEM3DDeformed models no C-PML at all -- its own module
    # docstring says so, its build() takes no `cpml`, and there is no 3-D
    # displacement-elastic builder in sweep.sem.cpml. Passing AB here used to
    # pad the mesh for an absorbing layer and then build none, so this ran in
    # a reflecting box while appearing to ask for absorption (PropSEM now
    # refuses a non-zero abcn for it).
    p = PropSEM(ElasticSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=DT, nt=NT,
                n_order=4, free_surface=True,
                abcn=0, f_dominant=F0, topography=_ztop_rows(), dev="cuda",
                dtype=torch.float32, impl="eager")
    vp = _vp()
    vs = (1000 + 4.0 * torch.arange(NZ)[:, None, None]).float().cuda().expand(
        NZ, NY, NX).contiguous()
    rho = (2000 + 3.0 * torch.arange(NZ)[:, None, None]).float().cuda().expand(
        NZ, NY, NX).contiguous()
    o = p(_wav(), SRC, REC, models=[vp, vs, rho],
          source_vector=(1e6, 0.7e6, 0.5e6))
    assert o.shape == (1, NT, REC.shape[1], 3)
    assert torch.isfinite(o).all()
