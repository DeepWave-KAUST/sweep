"""memory=Ckpt(chunks=K) on impl='c' for the velocity-stress classes.

The save-all adjoint of the velocity-stress kernels keeps the velocities and the
stretched velocity gradients of every step; the checkpointed one keeps the state at
the start of every K-step chunk and reruns each chunk in the backward
(``sweep.sem.drivers.vs_chunks``). The two must give the same records and
gradients -- up to the kernels' own run-to-run differences (the scatter adds with
atomics) -- for chunks that divide the record, that do not, of one step and of all
but one; the checkpointed run must take less memory; and PropSEM must default to it
on these classes while refusing a chunk where the kernels have none.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import Ckpt, Full
from sweep.sem import PropSEM, SEMPropagator
from sweep.sem.cpml.elastic_vs import build_simple_elastic_cpml
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.equations.elastic3d import ElasticSEM3D
from sweep.sem.equations.elastic_tti_vs3d import ElasticTTISEM3DDeformed
from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
from sweep.sem.equations.elastic_vs_deformed import ElasticVSSEM2DDeformed
from sweep.sem.equations.elastic_vti_vs_deformed import ElasticTTISEM2DDeformed
from sweep.sem.equations.visco_elastic_vs3d import ViscoElasticVSSEM3D
from sweep.sem.equations.visco_elastic_vs_deformed import ViscoElasticVSSEM2DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import build_topography_hex27
from sweep.sem.mesh.quad9 import build_topography_quad9
from sweep.sem.prop_c import (C_CHECKPOINT_CHUNK, SEMCudaPropagator, c_checkpoint_supported,
                              sem_c_available)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NT, DT = 100, 4e-4


def _wav(nt=NT, f0=16.0):
    t = np.arange(nt) * DT
    a = (np.pi * f0 * (t - 0.06)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _case_2d(cls, nex=8, nez=6):
    m = build_topography_quad9(nex, nez, 100.0 * nex, 100.0 * nez,
                               lambda x: 40.0 * torch.exp(-((x - 400.0) / 150.0) ** 2),
                               n_order=4, dtype=torch.float32)
    x, z = m.x_e / 800.0, m.z_e / 600.0
    v = {"vp": 2400.0 + 300.0 * z, "vs": 1250.0 + 150.0 * x, "rho": 2000.0 + 150.0 * z,
         "epsilon": 0.2 + 0.0 * x, "delta": 0.1 + 0.03 * x, "theta": 0.5 + 0.1 * z,
         "Qp": 40.0 + 30.0 * z, "Qs": 25.0 + 20.0 * x}
    names = {ElasticVSSEM2DDeformed: ("vp", "vs", "rho"),
             ElasticTTISEM2DDeformed: ("vp", "vs", "rho", "epsilon", "delta", "theta"),
             ViscoElasticVSSEM2DDeformed: ("vp", "vs", "rho", "Qp", "Qs")}[cls]
    ms = [v[n].to(torch.float32).contiguous().requires_grad_(True) for n in names]
    cp = build_simple_elastic_cpml(m, dt=DT, sides=("left", "right", "bottom"),
                                   pml_thickness_elements=2, c_max=3300.0, f_dominant=16.0)
    kw = {"f_ref": 16.0} if cls is ViscoElasticVSSEM2DDeformed else {}
    eq = cls.build(m, *ms, cpml=cp, **kw)
    src, rec = (350.0, 250.0), [(float(x), 150.0) for x in (150.0, 250.0, 450.0, 650.0)]
    return eq, ms, src, rec, (1.0e6, 0.7e6)


def _case_3d(cls, ne=(5, 4, 4)):
    nex, ney, nez = ne
    if cls is ElasticTTISEM3DDeformed:
        m = build_topography_hex27(nex, ney, nez, 100.0 * nex, 100.0 * ney, 100.0 * nez,
                                   lambda x, y: 30.0 * torch.exp(-((x - 250.0) / 120.0) ** 2),
                                   n_order=4, dtype=torch.float32)
    else:
        m = build_hex8_mesh(nex, ney, nez, 100.0 * nex, 100.0 * ney, 100.0 * nez,
                            n_order=4, dtype=torch.float32)
    x, z = m.x_e / 500.0, m.z_e / 400.0
    v = {"vp": 2400.0 + 300.0 * z, "vs": 1250.0 + 150.0 * x, "rho": 2000.0 + 150.0 * z,
         "epsilon": 0.2 + 0.0 * x, "delta": 0.1 + 0.03 * x, "gamma": 0.15 + 0.0 * x,
         "theta": 0.5 + 0.1 * z, "phi": 0.7 + 0.2 * x, "Qp": 40.0 + 30.0 * z,
         "Qs": 25.0 + 20.0 * x}
    names = {ElasticVSSEM3D: ("vp", "vs", "rho"),
             ElasticTTISEM3DDeformed: ("vp", "vs", "rho", "epsilon", "delta", "gamma",
                                       "theta", "phi"),
             ViscoElasticVSSEM3D: ("vp", "vs", "rho", "Qp", "Qs")}[cls]
    ms = [v[n].to(torch.float32).contiguous().requires_grad_(True) for n in names]
    cp = build_simple_elastic_cpml_3d(m, dt=DT, sides=("left", "right", "front", "back",
                                                       "bottom"),
                                      pml_thickness_elements=1, c_max=3300.0, f_dominant=16.0)
    kw = {"f_ref": 16.0} if cls is ViscoElasticVSSEM3D else {}
    eq = cls.build(m, *ms, cpml=cp, **kw)
    src = (250.0, 250.0, 150.0)
    rec = [(float(x), 150.0, 100.0) for x in (150.0, 250.0, 350.0)]
    return eq, ms, src, rec, (1.0e6, 0.7e6, 0.5e6)


def _run(case, cls, chunk, nt=NT):
    eq, ms, src, rec, payload = case(cls)
    P = SEMCudaPropagator.build(eq, dt=DT, nt=nt, source_xz=[src], receiver_xz=rec,
                                source_kinds=["force"], source_payloads=[payload],
                                pml_guard="off")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    o = P.forward(_wav(nt), ckpt_chunk=chunk)
    W = torch.linspace(0.5, 1.5, o.numel(), device=o.device).reshape(o.shape)
    ((W * o).sum() + 1e-3 * o.square().sum()).backward()
    torch.cuda.synchronize()
    return (o.detach().cpu(), [t.grad.cpu() for t in ms],
            torch.cuda.max_memory_allocated())


CASES = [(_case_2d, ElasticVSSEM2DDeformed), (_case_2d, ElasticTTISEM2DDeformed),
         (_case_2d, ViscoElasticVSSEM2DDeformed), (_case_3d, ElasticVSSEM3D),
         (_case_3d, ElasticTTISEM3DDeformed), (_case_3d, ViscoElasticVSSEM3D)]


@pytest.mark.parametrize("case,cls", CASES, ids=[c.__name__ for _, c in CASES])
@pytest.mark.parametrize("chunk", [25, 7, 1, NT - 1])
def test_checkpointed_adjoint_is_the_save_all(case, cls, chunk):
    o_full, g_full, _ = _run(case, cls, None)
    o_full2, g_full2, _ = _run(case, cls, None)            # the run-to-run floor
    o_ck, g_ck, _ = _run(case, cls, chunk)
    floor = max([_rel(o_full2, o_full)] + [_rel(a, b) for a, b in zip(g_full2, g_full)])
    bar = max(25.0 * floor, 1e-6)
    assert _rel(o_ck, o_full) <= bar, (_rel(o_ck, o_full), floor)
    for k, (a, b) in enumerate(zip(g_ck, g_full)):
        assert _rel(a, b) <= bar, (k, _rel(a, b), floor)


@pytest.mark.parametrize("case,cls", [(lambda c: _case_2d(c, 30, 20), ElasticVSSEM2DDeformed),
                                      (lambda c: _case_3d(c, (8, 8, 8)), ElasticVSSEM3D)],
                         ids=["2d", "3d"])
def test_checkpoints_take_less_memory(case, cls):
    """Meshes large enough that the histories dominate: 600 steps of them against
    19 states and one 32-step chunk."""
    nt = 600
    *_, full = _run(case, cls, None, nt=nt)
    *_, ck = _run(case, cls, 32, nt=nt)
    assert ck < 0.4 * full, (ck / 2**20, full / 2**20)


def test_propsem_defaults_to_checkpoints_on_these_classes(monkeypatch):
    import sweep.sem.prop_base as prop_base
    monkeypatch.setattr(prop_base, "_DEFAULT_MEMORY", None)   # the real default (conftest)
    kw = dict(shape=(21, 33), dh=10.0, dt=DT, nt=NT, n_order=4, free_surface=True,
              abcn=8, f_dominant=16.0, dev="cuda")
    p = PropSEM(ElasticVSSEM2DDeformed, impl="c", **kw)
    assert p.memory_strategy == "ckpt" and p._ckpt_chunk == C_CHECKPOINT_CHUNK
    assert PropSEM(ElasticVSSEM2DDeformed, impl="c", memory=Ckpt(chunks=10),
                   **kw)._ckpt_chunk == 10
    assert PropSEM(ElasticVSSEM2DDeformed, impl="c", memory=Full(),
                   **kw).memory_strategy == "full"
    assert c_checkpoint_supported(ViscoElasticVSSEM3D)
    assert not c_checkpoint_supported(ElasticSEM3D)
    kw3 = dict(shape=(9, 9, 9), dh=10.0, dt=DT, nt=NT, n_order=4, free_surface=True,
               abcn=0, f_dominant=16.0, dev="cuda")
    with pytest.raises(NotImplementedError, match="Ckpt"):
        PropSEM(ElasticSEM3D, impl="c", memory=Ckpt(chunks=10), **kw3)


def test_propsem_checkpointed_gradient_is_the_save_all():
    """Through PropSEM: the default (checkpoints) against memory=Full()."""
    def run(memory):
        p = PropSEM(ElasticVSSEM2DDeformed, shape=(33, 41), dh=10.0, dt=DT, nt=300,
                    n_order=4, free_surface=True, abcn=8, f_dominant=16.0, dev="cuda",
                    impl="c", memory=memory)
        z = torch.arange(33, dtype=torch.float32)[:, None].expand(33, 41)
        ms = [t.contiguous().cuda().requires_grad_(True)
              for t in (2200.0 + 10.0 * z, (2200.0 + 10.0 * z) / 1.8,
                        torch.full((33, 41), 2100.0))]
        w = (1e10 * _wav(300)).cuda()
        s = p(w, np.array([[20, 5]]), np.array([[[x, 6] for x in range(5, 36, 3)]]),
              models=ms)
        s.square().sum().backward()
        return s.detach().cpu(), [t.grad.cpu() for t in ms]
    s_full, g_full = run(Full())
    s_ck, g_ck = run(None)
    assert _rel(s_ck, s_full) < 1e-5
    for a, b in zip(g_ck, g_full):
        assert _rel(a, b) < 1e-4


def test_a_chunk_on_a_route_without_checkpoints_is_refused():
    m = build_hex8_mesh(3, 3, 3, 300.0, 300.0, 300.0, n_order=4, dtype=torch.float32)
    f = lambda v: torch.full(m.x_e.shape, v)
    eq = ElasticSEM3D.build(m, f(2400.0), f(1300.0), f(2000.0))
    P = SEMCudaPropagator.build(eq, dt=DT, nt=20, source_xz=[(150.0, 150.0, 150.0)],
                                receiver_xz=[(100.0, 150.0, 100.0)], source_kinds=["force"],
                                source_payloads=[(0.0, 0.0, 1.0)])
    with pytest.raises(NotImplementedError, match="ckpt_chunk"):
        P.forward(_wav(20), ckpt_chunk=5)
