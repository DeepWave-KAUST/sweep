"""32-bit connectivity for the SEM C kernels, and the guard that bounds it.

``ibool`` is read once per element-node by every element kernel, so at int64 it
is ~10% of the dominant kernel's memory traffic -- and traffic is what limits
this backend once the step's resident set passes L2. It is passed to the C side
as int32.

The ceiling this introduces is not new: the kernels already address the
(E, ngll, ...) arrays with ``const int off = e * m + tid``, and E*m counts
duplicated nodes (measured 1.56x n_global in 2-D, 1.84x in 3-D), so it overflows
first whatever dtype ibool has. What was missing is that nobody checked it -- on
a large-memory machine the existing int offset would wrap silently and return a
wrong gradient with no error at all.

A real overflow is not constructible (it needs >2e9 nodes, i.e. hundreds of GB),
so the guard is exercised by lowering the threshold rather than by asserting
that a number nobody can reach would be caught.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem.prop_c as PC
from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, DT, NT, F0, ABCN = 33, 49, 20.0, 2e-3, 200, 8.0, 5


def _prop(impl="c", memory=None):
    return PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=4, free_surface=True, abcn=ABCN * 4,
                   memory=memory, f_dominant=F0, dev="cuda",
                   dtype=torch.float32, impl=impl)


def _vp():
    z = np.arange(NZ)[:, None]
    return np.broadcast_to(1800 + 8.0 * z, (NZ, NX)).astype(np.float32).copy()


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")


def _run(impl, memory=None):
    p = _prop(impl, memory)
    src = np.array([[NX // 2, 2]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
    s = p(_wav(), src, rec, models=[vp])
    (s ** 2).sum().backward()
    return s.detach().cpu().numpy(), vp.grad.cpu().numpy()


def _cos(a, b):
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_connectivity_reaches_the_kernels_as_int32():
    eq = _prop()._build_eq(torch.tensor(_vp(), device="cuda"))
    ib = PC._ibool_i32(eq.mesh, torch.device("cuda"))
    assert ib.dtype == torch.int32
    assert int(ib.max()) == eq.mesh.n_global - 1, "connectivity lost values in the cast"
    assert ib.numel() == eq.mesh.ibool.numel()


def test_gradient_unchanged_vs_eager():
    """int32 indexing must not perturb the answer at all."""
    se, ge = _run("eager")
    sc, gc = _run("c")
    assert _cos(sc, se) >= 1 - 1e-6, f"fwd cos={_cos(sc, se):.8f}"
    assert _cos(gc, ge) >= 1 - 1e-6, f"grad cos={_cos(gc, ge):.8f}"


def test_gradient_unchanged_with_boundary_saving():
    _, gf = _run("c", None)
    _, gb = _run("c", BoundarySaving())
    assert _cos(gb, gf) >= 1 - 1e-6, f"grad cos={_cos(gb, gf):.8f}"


def test_guard_rejects_a_mesh_that_would_overflow(monkeypatch):
    """Lower the ceiling under a real mesh rather than build an unbuildable one."""
    eq = _prop()._build_eq(torch.tensor(_vp(), device="cuda"))
    mesh = eq.mesh
    dev = torch.device("cuda")
    PC._ibool_i32(mesh, dev)                      # fine at the real ceiling

    monkeypatch.setattr(PC, "_I32_MAX", mesh.n_global - 1)
    with pytest.raises(ValueError, match="too large for the 32-bit indexing"):
        PC._ibool_i32(mesh, dev)


def test_guard_catches_the_element_local_length_too(monkeypatch):
    """E*m exceeds n_global, so it is the binding limit -- pin that it is checked."""
    eq = _prop()._build_eq(torch.tensor(_vp(), device="cuda"))
    mesh = eq.mesh
    n_local = int(mesh.ibool.numel())
    assert n_local > mesh.n_global, (
        f"element-local length {n_local} should exceed n_global {mesh.n_global}")
    # a ceiling between the two: n_global passes, E*m must still fail
    monkeypatch.setattr(PC, "_I32_MAX", (mesh.n_global + n_local) // 2)
    with pytest.raises(ValueError, match="too large for the 32-bit indexing"):
        PC._ibool_i32(mesh, torch.device("cuda"))
