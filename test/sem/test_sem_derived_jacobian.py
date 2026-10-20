"""The identity the CUDA element kernels rely on to not store w_J.

``sem2d_acoustic_cpml_grads_f`` / ``_grads_T`` no longer read the (E, ngll, ngll)
``w_jac`` array. They rebuild it per node as::

    w_J = w1d[a] * w1d[b] / det(inv_J),   det(inv_J) = ixx*iez - ixz*iex

trading one 4-byte global read per element-node for a few FLOPs -- the right
trade on a kernel that is DRAM-bandwidth-bound at production sizes (measured
-5.6% on grads_f and -6.4% on grads_T at 3.5M GLL nodes). SPECFEM does the same.

This test guards the IDENTITY, not the kernel: nothing else in the codebase
forces ``w_jac`` to stay ``w_outer * det_J`` or ``inv_J`` to stay the inverse of
the same Jacobian, so a future change to the mesh builder could silently break
the kernels while every wave-propagation test still passes. Checked in fp64 on
both a flat mesh and a deformed (topography) one, since the deformed case is
where a wrong Jacobian convention would actually show.
"""
from __future__ import annotations

import pytest
import torch

from sweep.sem.basis.gll import gll_nodes_weights
from sweep.sem.basis.quad9 import evaluate_q2_jacobian
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9

NEX, NEZ, XMAX, ZMAX = 12, 9, 960.0, 720.0


def _flat():
    return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=4, dtype=torch.float64, device="cpu")


def _topo():
    return build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX, n_order=4,
        dtype=torch.float64, device="cpu",
        z_top_fn=lambda x: 60.0 * torch.sin(torch.pi * x / XMAX))


@pytest.mark.parametrize("mesh_fn", [_flat, _topo], ids=["flat", "topography"])
def test_det_J_is_the_reciprocal_of_det_inv_J(mesh_fn):
    mesh = mesh_fn()
    ixx, ixz = mesh.inv_J[..., 0, 0], mesh.inv_J[..., 0, 1]
    iex, iez = mesh.inv_J[..., 1, 0], mesh.inv_J[..., 1, 1]
    derived = 1.0 / (ixx * iez - ixz * iex)
    err = float((derived - mesh.det_J).abs().max() / mesh.det_J.abs().max())
    assert err < 1e-12, f"det_J from inv_J is off by {err:.3e}"


@pytest.mark.parametrize("mesh_fn", [_flat, _topo], ids=["flat", "topography"])
def test_w_jac_is_the_outer_gll_weights_times_det_J(mesh_fn):
    """And that the weight factor is separable, i.e. w1d[a]*w1d[b]."""
    mesh = mesh_fn()
    w1d = mesh.w_ref
    w_outer = w1d[:, None] * w1d[None, :]
    err = float((mesh.w_jac - w_outer.unsqueeze(0) * mesh.det_J).abs().max()
                / mesh.w_jac.abs().max())
    assert err < 1e-12, f"w_jac != w1d (x) w1d * det_J, off by {err:.3e}"


@pytest.mark.parametrize("mesh_fn", [_flat, _topo], ids=["flat", "topography"])
def test_the_exact_expression_the_kernels_evaluate(mesh_fn):
    """End to end, in the kernels' own index convention (a = tid/ngll, b = tid%ngll)."""
    mesh = mesh_fn()
    g = mesh.ngll
    a = torch.arange(g).repeat_interleave(g)
    b = torch.arange(g).repeat(g)
    ixx = mesh.inv_J[..., 0, 0].reshape(mesh.n_elements, -1)
    ixz = mesh.inv_J[..., 0, 1].reshape(mesh.n_elements, -1)
    iex = mesh.inv_J[..., 1, 0].reshape(mesh.n_elements, -1)
    iez = mesh.inv_J[..., 1, 1].reshape(mesh.n_elements, -1)
    w = mesh.w_ref[a] * mesh.w_ref[b] / (ixx * iez - ixz * iex)
    ref = mesh.w_jac.reshape(mesh.n_elements, -1)
    err = float((w - ref).abs().max() / ref.abs().max())
    assert err < 1e-12, f"kernel expression off by {err:.3e}"


def test_the_identity_holds_when_both_cross_terms_are_live():
    """The mesh builders produce a TRIANGULAR Jacobian, so the tests above cannot
    see a sign error in the cross term.

    ``build_axis_aligned_quad9`` gives ixz = iex = 0 and
    ``build_topography_quad9`` bends only the top surface, which makes iex ~ 5e-3
    but leaves ixz ~ 1e-16. With ixz = 0 the product ixz*iex vanishes whatever
    its sign, so ``ixx*iez + ixz*iex`` and even ``ixx*iez`` alone pass on both --
    measured, which is why this case exists. A generic Q2 element with perturbed
    control points has both cross terms O(1e-2) and does discriminate.
    """
    torch.manual_seed(0)
    g = 5
    xi, _ = gll_nodes_weights(4, dtype=torch.float64, device="cpu")
    ex, ez = torch.meshgrid(torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
                            torch.linspace(0.0, 1.5, 3, dtype=torch.float64),
                            indexing="ij")
    ctrl = torch.stack([ex, ez], dim=-1).unsqueeze(0)          # (1, 3, 3, 2)
    ctrl = ctrl + 0.12 * torch.randn_like(ctrl)                # generic deformation
    _, inv_J, det_J = evaluate_q2_jacobian(ctrl, xi, xi)
    ixx, ixz = inv_J[..., 0, 0], inv_J[..., 0, 1]
    iex, iez = inv_J[..., 1, 0], inv_J[..., 1, 1]

    assert float(ixz.abs().max()) > 1e-3, "cross term ixz is not live; test is blind"
    assert float(iex.abs().max()) > 1e-3, "cross term iex is not live; test is blind"

    good = 1.0 / (ixx * iez - ixz * iex)
    assert float((good - det_J).abs().max() / det_J.abs().max()) < 1e-12

    for name, bad in (("ixx*iez + ixz*iex", 1.0 / (ixx * iez + ixz * iex)),
                      ("ixx*iez",           1.0 / (ixx * iez))):
        err = float((bad - det_J).abs().max() / det_J.abs().max())
        assert err > 1e-6, f"a wrong determinant ({name}) would slip through: {err:.3e}"
