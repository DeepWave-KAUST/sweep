"""3-D elastic on a DEFORMED Hex27 mesh with TOPOGRAPHY, impl='c' vs eager.

ElasticSEM3DDeformed (full per-node 3x3 inv_J chain rule, bare / natural free
surface) on a curved-top mesh: the CUDA deformed-metric elastic kernels must
match the eager equation on seis_x/y/z and dL/d{vp,vs,rho}, and the CUDA
boundary-saving path (bare reversible ring) must equal the CUDA save-all path.

Force source scaled up (1e6) to lift the gradients off the fp32 noise floor.
All comparisons: cos >= 1-1e-4.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.hex27 import build_topography_hex27
from sweep.sem.equations.elastic3d_deformed import ElasticSEM3DDeformed
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 7, 6, 6
XMAX, YMAX, ZMAX = 700.0, 600.0, 600.0
DT, NT, F0, TSH = 4e-4, 180, 16.0, 0.07
SRC = (XMAX * 0.46, YMAX * 0.42, 250.0)
REC = [(float(x), YMAX * 0.58, 170.0)
       for x in np.linspace(150.0, XMAX - 150.0, 10)]
PAYLOAD = (1.0e6, 0.7e6, 0.5e6)


def _z_top(x, y):
    return 60.0 * torch.exp(-(((x - XMAX * 0.5) / 170.0) ** 2
                              + ((y - YMAX * 0.5) / 150.0) ** 2))


def _mesh():
    return build_topography_hex27(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, _z_top,
                                  n_order=4, dtype=torch.float32)


def _eq(grad=False):
    m = _mesh()
    vp = (2400.0 + 300.0 * (m.z_e / ZMAX)).to(torch.float32).contiguous()
    vs = (1200.0 + 200.0 * (m.x_e / XMAX)).to(torch.float32).contiguous()
    rho = (2000.0 + 150.0 * (m.y_e / YMAX)).to(torch.float32).contiguous()
    if grad:
        vp, vs, rho = (vp.requires_grad_(True), vs.requires_grad_(True),
                       rho.requires_grad_(True))
    return ElasticSEM3DDeformed.build(m, vp, vs, rho), (vp, vs, rho)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _prop(eq, use_cuda):
    return (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
        source_kinds=["force"], source_payloads=[PAYLOAD])


def _grads(eq, models, use_cuda, mem=None):
    vp, vs, rho = models
    o = _prop(eq, use_cuda).forward(_wav(), memory=mem)
    loss = 0.0
    for j in range(o.shape[-1]):                 # every component
        s = o[..., j].T
        W = torch.linspace(0.4 + 0.1 * j, 1.6 - 0.1 * j, s.numel(),
                           device=s.device).reshape(s.shape)
        loss = loss + (W * s).sum()
    loss.backward()
    return vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()


def test_deformed_elastic3d_forward_matches_eager():
    eq, _ = _eq()
    oe = _prop(eq, False).forward(_wav())
    oc = _prop(eq, True).forward(_wav())
    for j, nm in enumerate(('u_x', 'u_y', 'u_z')):  # component order
        c = _cos(oc[..., j].T.cpu(), oe[..., j].T.cpu())
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"


def test_deformed_elastic3d_grads_match_eager():
    ge = _grads(*_eq(grad=True), False)
    gc = _grads(*_eq(grad=True), True)
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"


def test_deformed_elastic3d_rejects_boundary_saving():
    """This path does not implement boundary saving -- it must SAY so.

    ``_forward_elastic3d_deformed`` never wires ``c["bs"]``, so ``memory=
    'boundary'`` was stored and never read. The test that used to live here
    compared ``mem="boundary"`` against ``mem=None`` on this very path and so
    compared a run with ITSELF: measured, the two differ by 8e-07 relative
    while two identical ``mem=None`` runs differ by 1e-06 (the CUDA atomics
    noise floor). It could not have failed.
    """
    import pytest
    from sweep.sem.prop_c import SEMCudaPropagator
    eq, _ = _eq(grad=False)
    prop = _prop(eq, True)
    with pytest.raises(NotImplementedError, match="memory='boundary'"):
        prop.forward(_wav(), memory="boundary")


def test_deformed_elastic3d_c_gradients_match_eager():
    """What the old test actually verified once the self-comparison is dropped:
    the CUDA full-tape gradient against the eager one."""
    gc_full = _grads(*_eq(grad=True), True, mem=None)
    ge = _grads(*_eq(grad=True), False)
    for name, f, e in zip(("vp", "vs", "rho"), gc_full, ge):
        assert _cos(f, e) >= 1 - 1e-4, f"{name} c vs eager cos={_cos(f, e):.6f}"
