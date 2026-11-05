"""The 3-D Biot SEM on CUDA (impl='c') against eager, through PropSEM.

BiotVSSEM3DDeformed runs the kernels of ``sem3d/biot3d.cu``: the 2-D Biot step with y,
the flux copies of a node advanced in groups of their number (1 inside an element ...
8 at a corner). Each case compares the thirteen receiver fields and the gradients in
every Biot coefficient with eager (cos >= 1-1e-4):

* a layered porous rock, the C-PML on five sides or an open-pore surface, for each
  kind of source (the explosion, a force, a pore-pressure injection, a solid and a
  fluid velocity);
* water over the porous rock (the fluid-solid faces);
* the viscous memory;
* the checkpointed adjoint against the save-all.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_from_rock
from sweep.equations.biot_params import biot_fluid
from sweep.propagator import Ckpt, Full
from sweep.sem import BiotVSSEM3DDeformed, PropSEM
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NO = 4
SHAPE, DH, DT, NT, F0 = (17, 17, 21), 10.0, 5e-4, 200, 20.0
RECV = ["vx", "vy", "vz", "p", "qx", "qy", "qz", "sxx", "syy", "szz", "sxy", "sxz", "syz"]


def _rock(shape=SHAPE, kappa=1e-10):
    nz = shape[0]
    z = torch.arange(nz, dtype=torch.float32).view(-1, 1, 1).expand(shape)
    deep = (z > nz // 2).float()
    rock = dict(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                k_d=8e9 + 6e9 * deep, mu=6e9 + 5e9 * deep, phi=0.25 - 0.1 * deep,
                kappa=kappa, tortuosity=2.0)
    return [torch.as_tensor(c, dtype=torch.float32).expand(shape).contiguous()
            for c in biot_from_rock(**rock)]


def _ricker(nt, dt, f0):
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float((a / na) @ (b / nb))


def _prop(impl, *, free_surface=False, memory=None, equation_options=None, src_type=None):
    kw = {} if memory is None else {"memory": memory}
    if equation_options:
        kw["equation_options"] = equation_options
    return PropSEM(BiotVSSEM3DDeformed, shape=SHAPE, dh=DH, dt=DT, nt=NT, n_order=NO,
                   f_dominant=F0, free_surface=free_surface, abcn=2 * NO, dev="cuda",
                   dtype=torch.float32, impl=impl, source_type=src_type, receiver_type=RECV,
                   **kw)


def _run(impl, models, src_type, src=(9, 7, 8), **kw):
    p = _prop(impl, src_type=src_type, **kw)
    assert p.impl == impl
    ms = [t.clone().cuda().requires_grad_(True) for t in models]
    # off the source's symmetry planes: no component is a structural zero
    rec = np.stack([np.arange(3, 18, 3), np.full(5, 11), np.full(5, 12)], 1)[None]
    s = p((1e10 * _ricker(NT, DT, F0)).cuda(), np.array([list(src)]), rec, models=ms)
    scale = s.detach().abs().amax(dim=(0, 1, 2), keepdim=True).clamp_min(1e-30)
    W = torch.linspace(0.5, 1.5, s.numel(), device=s.device).reshape(s.shape)
    (W * s / scale).sum().backward()
    return s.detach().cpu(), [t.grad.cpu() for t in ms]


def _check(models, **kw):
    se, ge = _run("eager", models, **kw)
    sc, gc = _run("c", models, **kw)
    for j, nm in enumerate(RECV):
        c = _cos(sc[..., j], se[..., j])
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"
    for k, (e, c) in enumerate(zip(ge, gc)):
        if float(e.abs().max()) == 0.0:
            assert float(c.abs().max()) == 0.0, k
            continue
        cs = _cos(c, e)
        assert cs >= 1 - 1e-4, f"grad of coefficient {k} cos={cs:.6f}"


@pytest.mark.parametrize("src_type", [["sxx", "syy", "szz"], ["fz"], ["p"], ["vx"], ["qz"]],
                         ids=["explosion", "force", "injection", "solid-velocity",
                              "fluid-velocity"])
@pytest.mark.parametrize("free_surface", [False, True], ids=["cpml", "open-pore"])
def test_porous_rock_matches_eager(src_type, free_surface):
    _check(_rock(), src_type=src_type, free_surface=free_surface, memory=Full())


def _layered(mesh, tops, media):
    """GLL-native coefficients, each element wholly one medium (mesh depths)."""
    zc = mesh.z_e.mean((1, 2, 3))
    k = torch.bucketize(zc, torch.tensor(tops[1:], dtype=zc.dtype, device=zc.device))
    E, n = mesh.n_elements, mesh.ngll
    return [torch.stack([torch.as_tensor(float(m[i])) for m in media]).to(zc.device)[k]
            .view(E, 1, 1, 1).expand(E, n, n, n).to(torch.float32).contiguous()
            for i in range(len(media[0]))]


def test_water_over_porous_rock_matches_eager():
    mesh = _prop("eager", free_surface=True, src_type=["p"]).mesh
    poro = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                          k_d=8e9, mu=6e9, phi=0.25, kappa=1e-10, tortuosity=2.0)
    water = biot_fluid(2.25e9, 1000.0)
    models = _layered(mesh, [0.0, 80.0], [water, poro])
    _check(models, src_type=["p"], src=(9, 7, 4), free_surface=True, memory=Full())


def test_viscous_memory_matches_eager():
    _check(_rock(), src_type=["sxx", "syy", "szz"], memory=Full(),
           equation_options={"viscous_q": (5.0, 20.0)})


def test_checkpointed_adjoint_is_the_save_all():
    s_full, g_full = _run("c", _rock(), ["sxx", "syy", "szz"], memory=Full())
    s_ck, g_ck = _run("c", _rock(), ["sxx", "syy", "szz"], memory=Ckpt(chunks=30))
    assert _cos(s_ck, s_full) >= 1 - 1e-6
    for k, (a, b) in enumerate(zip(g_ck, g_full)):
        if float(b.abs().max()) == 0.0:
            continue
        assert _cos(a, b) >= 1 - 1e-6, k
