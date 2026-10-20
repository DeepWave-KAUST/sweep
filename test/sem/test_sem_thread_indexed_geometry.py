"""The 3-D flat kernels read the quadrature weights per THREAD, so only the
first ngll^3 entries exist.

``sem3d_acoustic_cpml_grads`` / ``_grads_T`` index ``w_jac`` and
``w_j{x,y,z}`` with ``threadIdx.x``, not with the element-node offset, because
on an axis-aligned Hex8 mesh every element's slice is identical. The consts
builders therefore hand them ``ngll^3`` floats instead of ``(E, ngll^3)`` --
measured 30.8 B per global node, 35% of the 3-D consts, that used to be
allocated and never read.

Two things have to hold for that to be lossless, and both are checked here
rather than argued:

* **nothing beyond the first slice is ever read.** Proven by a poison test:
  restore the full ``(E, m)`` layout with everything past the first element
  filled with NaN. A kernel that touched it would return NaN; the record and
  the gradients come back unchanged.
* **every element's slice really was identical**, so the kept slice is the
  value the physics wants. That is a property of the mesh builder, and
  ``_elem_thread_slice`` refuses a mesh where it fails.

The deformed (Hex27) path is the control: its Jacobian is genuinely per node,
it takes the ``*_deformed`` kernels, and its ``w_jac`` must stay full length.
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology, _CompactEq3D
from sweep.sem.equations import AcousticSEM3D, AcousticSEM3DDeformed
from sweep.sem.prop_c import _elem_thread_slice, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NEX, NT = 4, 40.0, 8.0, 12, 16
GEOM = ("w_jac", "w_jx", "w_jy", "w_jz")


def _hill(x, y):
    return 90.0 * torch.exp(-(((x - 960.0) / 500.0) ** 2
                              + ((y - 960.0) / 500.0) ** 2))


def _hill_rows(n):
    """``_hill`` on the model columns: the surface row of each, (n, n)."""
    a = torch.arange(n, dtype=torch.float64) * DH
    Y, X = torch.meshgrid(a, a, indexing="ij")
    return (_hill(X, Y) / DH).numpy()


def _build(deformed=False, memory=None):
    n = NEX * NO + 1
    p = S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D,
                  shape=(n, n, n), dh=DH, dt=6e-4, nt=NT, n_order=NO,
                  free_surface=("top",), abcn=3 * NO,
                  f_dominant=F0, dev="cuda", impl="c", memory=memory,
                  **(dict(topography=_hill_rows(n)) if deformed else {}))
    torch.manual_seed(0)
    vp = 2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")
    return p, vp, n


def _geometry(p):
    """GLOBAL node ids, on the node grid of the model AND its absorbing
    layer: the source mid-model two elements down, receivers after it."""
    nz, ny, nx = p._grid_shape()
    src = (2 * NO * ny + ny // 2) * nx + nx // 2
    rec = torch.arange(src + 5, src + 13, device="cuda")
    t = (torch.arange(NT, dtype=torch.float32, device="cuda") - 5.0) * 6e-4 * F0
    wav = (1.0 - 2.0 * (3.1415926 * t) ** 2) * torch.exp(-(3.1415926 * t) ** 2)
    return src, rec, wav


# --------------------------------------------------------------------------- #
def test_the_flat_geometry_consts_are_one_element_long():
    p, vp, n = _build()
    src, rec, _ = _geometry(p)
    m3 = (NO + 1) ** 3
    for px in (1, 2):
        parts, _ = SEMModelParallel(p, StripTopology(px))._parts3d(
            _CompactEq3D.build(p, vp), int(src), rec)
        for q in parts:
            for k in GEOM:
                assert q.c[k].numel() == m3, (
                    f"px={px}: {k} has {q.c[k].numel()} entries; the flat "
                    f"kernels read {m3} of them")


def test_the_deformed_path_keeps_the_full_per_element_node_array():
    """Control: the curved mesh's Jacobian varies per node, so its w_jac is
    NOT shrinkable and must stay (E, m)."""
    p, vp, n = _build(deformed=True)
    src, rec, _ = _geometry(p)
    eq, _ = p._resolve_model_batch([vp])
    ng = eq.mesh.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[int(src)] = 1.0
    parts, _ = SEMModelParallel(p, StripTopology(2))._parts3d(eq, (pat,), rec)
    m3 = (NO + 1) ** 3
    for q in parts:
        assert q.c["w_jac"].numel() == q.c["n_elements"] * m3


def test_elem_thread_slice_refuses_a_mesh_that_varies_between_elements():
    """The negative control for the check itself: if it accepted a varying
    array, the shrink would silently substitute element 0 everywhere."""
    m = 8
    ok = torch.arange(m, dtype=torch.float32).repeat(4)
    assert torch.equal(_elem_thread_slice(ok, "ok", m),
                       torch.arange(m, dtype=torch.float32))
    bad = ok.clone()
    bad[3 * m + 2] += 1e-6
    with pytest.raises(ValueError, match="varies between elements"):
        _elem_thread_slice(bad, "bad", m)
    with pytest.raises(ValueError, match="not a multiple"):
        _elem_thread_slice(torch.zeros(m + 1), "ragged", m)


def _cos_gap(a, b):
    """1 - cosine, in fp64. The metric this path admits.

    The assembly is a float ``atomicAdd``, so the max relative difference
    between two runs of the SAME code is ~5e-7 here and carries no
    information -- an equality test on it fails for reasons that have nothing
    to do with the change. The cosine gap of the same pairs sits at 1e-16 to
    1e-14, i.e. the round-off of the comparison itself, and moves when the
    physics moves.
    """
    x, y = a.double().reshape(-1), b.double().reshape(-1)
    nx, ny = x.norm(), y.norm()
    if nx == 0 or ny == 0:
        return 0.0 if torch.equal(x, y) else 1.0
    return abs(1.0 - (x @ y / (nx * ny)).item())


@pytest.mark.parametrize("px", [1, 2])
def test_nothing_past_the_first_element_slice_is_ever_read(px):
    """The decisive experiment, in three runs of the same configuration.

    * ``short`` -- what the code does now: ``m`` floats.
    * ``copy``  -- the OLD ``(E, m)`` tiled layout, same values.
    * ``nan``   -- the old layout with every entry past the first element
      replaced by NaN.

    ``nan`` coming back FINITE is the proof: a kernel that indexed those
    entries -- in the forward, the backward, or the boundary-saving reverse
    march -- would return NaN, and nothing downstream can undo one. ``short``
    agreeing with ``copy`` is then the statement that the shrink is what the
    change claims.

    ``copy`` also matters as the control for the comparison: it has the OLD
    allocation footprint, and changing the footprint alone perturbs the atomic
    ordering (measured 8.8e-8 relative), so a numeric bar taken from two
    identically-shaped runs would be too tight for reasons unrelated to the
    values.
    """
    p, vp, n = _build(memory=BoundarySaving())
    src, rec, wav = _geometry(p)

    def run(tail):
        """tail=None keeps the short array; 'copy'/'nan' restore (E, m)."""
        v = vp.clone().requires_grad_(True)
        w = wav.clone().requires_grad_(True)
        ddp = SEMModelParallel(p, StripTopology(px))
        if tail is not None:
            orig = ddp._parts3d

            def patched(eq_g, pattern, receivers):
                parts, mats = orig(eq_g, pattern, receivers)
                for q in parts:
                    e = q.c["n_elements"]
                    for k in GEOM:
                        short = q.c[k]
                        full = torch.empty(e, short.numel(), dtype=short.dtype,
                                           device=short.device)
                        full[:] = float("nan") if tail == "nan" else short
                        full[0] = short
                        q.c[k] = full.reshape(-1) if short.dim() == 1 else full
                return parts, mats

            ddp._parts3d = patched
        out = ddp(w, src, rec, [v])        # boundary-saved: p's memory
        out.pow(2).sum().backward()
        return out.detach(), v.grad, w.grad

    a1, a2 = run(None), run(None)             # A/A: the metric's own floor
    copy, nan = run("copy"), run("nan")
    names = ("record", "d_vp", "d_wavelet")
    for i, nm in enumerate(names):
        assert torch.isfinite(nan[i]).all(), (
            f"px={px}: {nm} is not finite when the tail is NaN -- a kernel "
            "reads past the first element slice, so the shrink loses data")
        bar = max(_cos_gap(a1[i], a2[i]), 1e-12)
        shrink = _cos_gap(a1[i], copy[i])
        unread = _cos_gap(copy[i], nan[i])
        assert shrink <= bar, (
            f"px={px}: {nm} differs by {shrink:.3e} between the m-long array "
            f"and the old (E, m) tile of the same values; A/A is {bar:.3e}")
        assert unread <= bar, (
            f"px={px}: {nm} differs by {unread:.3e} when only the never-read "
            f"tail changes; A/A is {bar:.3e}")
