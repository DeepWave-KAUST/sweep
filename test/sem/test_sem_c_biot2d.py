"""The 2-D Biot SEM on CUDA (impl='c') against eager, through PropSEM.

BiotVSSEM2DDeformed and BiotAnisoPermSEM2DDeformed run the kernels of
``sem2d/biot2d.cu``: the pore pressure injected and pinned, the strong grad p, the
stress divergence, the flux copies at every node advanced with the drag's exact
step, the frame stress, the weak div q, the fluid-solid interface, the viscous
memory, the C-PML on every derivative. Each case compares the eight receiver fields
and the gradients in every Biot coefficient with eager (cos >= 1-1e-4):

* a layered porous rock with a drag fast enough for its exponential to matter, the
  C-PML on three sides, open-pore surface or none, for each kind of source (a
  moment, a force, a pore-pressure injection, a solid and a fluid velocity);
* water over the porous rock over a dry elastic solid (the interfaces);
* the viscous memory, a drag tensor, and the GSLS frame of ViscoBiotVSSEM2DDeformed
  (``Q = 1e9`` there gives BiotVSSEM2DDeformed's records);
* the checkpointed adjoint (PropSEM's default here) against the save-all.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_from_rock
from sweep.equations.biot_params import biot_elastic, biot_fluid
from sweep.propagator import Ckpt, Full
from sweep.sem import (BiotAnisoPermSEM2DDeformed, BiotVSSEM2DDeformed, PropSEM,
                       ViscoBiotVSSEM2DDeformed)
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NO = 4
SHAPE, DH, DT, NT, F0 = (33, 41), 10.0, 5e-4, 360, 12.0
RECV = ["vx", "vz", "p", "qx", "qz", "sxx", "szz", "sxz"]


def _rock(shape=SHAPE, kappa=1e-10):
    nz, nx = shape
    z = torch.arange(nz, dtype=torch.float32)[:, None].expand(nz, nx)
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
    """Cosine of two records, scale-free: the flux of a unit moment is ~1e-16, where an
    absolute guard in the denominator would read as a mismatch."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float((a / na) @ (b / nb))


def _run(impl, models, src_type, *, eq=BiotVSSEM2DDeformed, free_surface=False,
         memory=None, equation_options=None, src=(20, 10), amp=1e10):
    kw = {} if memory is None else {"memory": memory}
    if equation_options:
        kw["equation_options"] = equation_options
    p = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, nt=NT, n_order=NO, f_dominant=F0,
                free_surface=free_surface, abcn=2 * NO, dev="cuda", dtype=torch.float32,
                impl=impl, source_type=src_type, receiver_type=RECV, **kw)
    assert p.impl == impl
    ms = [t.clone().cuda().requires_grad_(True) for t in models]
    rec = np.stack([np.arange(6, 36, 4), np.full(8, 24)], 1)[None]
    s = p((amp * _ricker(NT, DT, F0)).cuda(), np.array([list(src)]), rec, models=ms)
    # a per-field scale, so that every field weighs in the misfit
    scale = s.detach().abs().amax(dim=(0, 1, 2), keepdim=True).clamp_min(1e-30)
    W = torch.linspace(0.5, 1.5, s.numel(), device=s.device).reshape(s.shape)
    (W * s / scale).sum().backward()
    return s.detach().cpu(), [t.grad.cpu() for t in ms]


def _check(case, **kw):
    se, ge = _run("eager", case, **kw)
    sc, gc = _run("c", case, **kw)
    for j, nm in enumerate(RECV):
        c = _cos(sc[..., j], se[..., j])
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"
    for k, (e, c) in enumerate(zip(ge, gc)):
        if float(e.abs().max()) == 0.0:
            assert float(c.abs().max()) == 0.0, k
            continue
        cs = _cos(c, e)
        assert cs >= 1 - 1e-4, f"grad of coefficient {k} cos={cs:.6f}"


@pytest.mark.parametrize("src_type", [["sxx", "szz"], ["fz"], ["p"], ["vx"], ["qz"]],
                         ids=["moment", "force", "injection", "solid-velocity",
                              "fluid-velocity"])
@pytest.mark.parametrize("free_surface", [False, True], ids=["cpml", "open-pore"])
def test_porous_rock_matches_eager(src_type, free_surface):
    _check(_rock(), src_type=src_type, free_surface=free_surface, memory=Full())


def _layered(mesh, tops, media):
    """GLL-native coefficients, each element wholly one medium: ``media[k]`` where the
    element's centre lies between ``tops[k]`` and ``tops[k + 1]`` (mesh depths)."""
    zc = mesh.z_e.mean((1, 2))
    k = torch.bucketize(zc, torch.tensor(tops[1:], dtype=zc.dtype, device=zc.device))
    E, n = mesh.n_elements, mesh.ngll
    return [torch.stack([torch.as_tensor(float(m[i])) for m in media]).to(zc.device)[k]
            .view(E, 1, 1).expand(E, n, n).to(torch.float32).contiguous()
            for i in range(len(media[0]))]


def test_water_over_porous_rock_over_dry_rock_matches_eager():
    """The fluid-solid interface (the water's pressure on the frame, the frame's
    normal velocity in the water's equation) and the sealed porous-dry contact."""
    mesh = PropSEM(BiotVSSEM2DDeformed, shape=SHAPE, dh=DH, dt=DT, nt=NT, n_order=NO,
                   f_dominant=F0, free_surface=True, abcn=2 * NO, dev="cuda",
                   impl="eager").mesh
    poro = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                          k_d=8e9, mu=6e9, phi=0.25, kappa=1e-10, tortuosity=2.0)
    water = biot_fluid(2.25e9, 1000.0)
    dry = biot_elastic(4200.0, 2400.0, 2500.0)
    models = _layered(mesh, [0.0, 80.0, 240.0], [water, poro, dry])
    _check(models, src_type=["p"], src=(20, 4), free_surface=True, memory=Full())


def test_viscous_memory_matches_eager():
    _check(_rock(), src_type=["sxx", "szz"], memory=Full(),
           equation_options={"viscous_q": (5.0, 12.0)})


def test_drag_tensor_matches_eager():
    coef = _rock()
    tensor = coef[:7] + [coef[7], 0.3 * coef[7], 2.0 * coef[7]]
    _check(tensor, src_type=["sxx", "szz"], eq=BiotAnisoPermSEM2DDeformed, memory=Full())


def _q(shape=SHAPE, inf=False):
    nz, nx = shape
    z = torch.arange(nz, dtype=torch.float32)[:, None].expand(nz, nx)
    if inf:              # a grid model of inf interpolates to nan on the GLL nodes
        return [torch.full(shape, 1e9), torch.full(shape, 1e9)]
    return [(30.0 + 1.0 * z).contiguous(), (20.0 + 0.6 * z).contiguous()]


def test_gsls_frame_matches_eager():
    _check(_rock() + _q(), src_type=["sxx", "szz"], eq=ViscoBiotVSSEM2DDeformed,
           memory=Full(), equation_options={"f_ref": F0})


def test_gsls_frame_without_attenuation_is_biot():
    s_v, _ = _run("c", _rock() + _q(inf=True), ["sxx", "szz"], eq=ViscoBiotVSSEM2DDeformed,
                  memory=Full(), equation_options={"f_ref": F0})
    s_b, _ = _run("c", _rock(), ["sxx", "szz"], memory=Full())
    for j in range(len(RECV)):
        assert _cos(s_v[..., j], s_b[..., j]) >= 1 - 1e-8, RECV[j]


def test_checkpointed_adjoint_is_the_save_all():
    s_full, g_full = _run("c", _rock(), ["sxx", "szz"], memory=Full())
    s_ck, g_ck = _run("c", _rock(), ["sxx", "szz"], memory=Ckpt(chunks=50))
    assert _cos(s_ck, s_full) >= 1 - 1e-6
    for k, (a, b) in enumerate(zip(g_ck, g_full)):
        if float(b.abs().max()) == 0.0:
            continue
        assert _cos(a, b) >= 1 - 1e-6, k
