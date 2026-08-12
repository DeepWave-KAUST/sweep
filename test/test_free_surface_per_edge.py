"""Per-edge free surface (eager, CPU): Acoustic + Elastic 2-D.

Covers the plumbing (``_edges`` normalization + the NotImplementedError guard),
the top-only bit-exactness (the historical ``free_surface=True`` path is
unchanged), and per-edge correctness:

* Acoustic (regular grid): a centred source in a closed 4-edge free-surface box
  gives a receiver record with machine-precision x- and z-mirror symmetry.
* Elastic (staggered grid): gradient consistency (finite-difference vs autograd)
  for per-edge configs, and that a free-surface edge REFLECTS (unlike a PML
  edge).  A staggered free surface is an O(h) boundary condition, so the low vs
  high sides are different-but-valid discretisations and are NOT bit-symmetric;
  correctness is asserted via the adjoint and the reflection, not bit-symmetry.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Elastic
from sweep.equations._edges import (
    normalize_free_surface, normalize_pad, is_top_only_or_none, torch_pad_order,
)
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DH, DT, SO = 10.0, 1.0e-3, 4
DEV = "cpu"


def _wav(nt, f=12.0, delay=0.05):
    t = np.arange(nt, dtype=np.float32) * DT - delay
    return torch.tensor((1e3 * ricker(t, f=f)).astype(np.float32)).to(DEV)


# ---------------------------------------------------------------------------
# _edges normalization
# ---------------------------------------------------------------------------
def test_normalize_free_surface_forms():
    assert normalize_free_surface(True, 2) == (True, False, False, False)
    assert normalize_free_surface(False, 2) == (False, False, False, False)
    assert normalize_free_surface(["top", "left"], 2) == (True, False, True, False)
    assert normalize_free_surface([1, 0, 0, 1], 2) == (True, False, False, True)
    assert normalize_free_surface({"bottom": True}, 2) == (False, True, False, False)
    # 3-D order (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)
    assert normalize_free_surface("back", 3) == (False, False, False, True, False, False)


def test_normalize_pad_and_order():
    assert normalize_pad(50, (True, False, False, False), 2) == (0, 50, 50, 50)
    assert normalize_pad(50, (False,) * 4, 2) == (50, 50, 50, 50)
    assert normalize_pad([9, 50, 30, 40], (True, False, False, False), 2) == (0, 50, 30, 40)
    assert is_top_only_or_none((True, False, False, False))
    assert not is_top_only_or_none((False, True, False, False))
    # axis-major (z_lo,z_hi,x_lo,x_hi) -> torch pad order (x_lo,x_hi,z_lo,z_hi)
    assert torch_pad_order((0, 50, 30, 40), 2) == (30, 40, 0, 50)


def test_invalid_specs_raise():
    with pytest.raises(ValueError):
        normalize_free_surface(["nope"], 2)
    with pytest.raises(ValueError):
        normalize_free_surface([True, False], 2)          # wrong length
    with pytest.raises(ValueError):
        normalize_pad([10, 10, 10], (False,) * 4, 2)      # wrong length


# ---------------------------------------------------------------------------
# Propagator plumbing: default bit-exact layout + guard
# ---------------------------------------------------------------------------
def _ac_prop(fs, abcn=30, shape=(48, 56)):
    eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    return PropTorch(eq, shape=shape, free_surface=fs, abcn=abcn, dh=DH, dt=DT,
                     use_ckpt=False, impl="eager")


def test_default_layout_bit_exact():
    p = _ac_prop(True)
    assert p.fs_faces == (True, False, False, False)
    assert p.pad == (0, 30, 30, 30)
    assert p.padding == (30, 30, 0, 30)          # torch order (x, z)
    assert p.shape == (48 + 30, 56 + 60)
    p0 = _ac_prop(False)
    assert p0.pad == (30, 30, 30, 30)
    assert p0.shape == (48 + 60, 56 + 60)


def test_non_top_per_edge_keeps_top_pml():
    """A per-edge free surface on a NON-top face must not silently add a top free
    surface / drop the top PML.

    Regression: a fold-in meant for topography (which physically implies a top
    free surface) was gated only on ``_image_method_active`` — True for *any*
    per-edge FS — so ``free_surface=['left']`` became ``{top, left}`` with
    ``pad[0]=0``.  The top then reflected (spurious free surface) instead of
    absorbing, injecting boundary reflections.  fs_faces axis order is
    (z_lo=top, z_hi=bottom, x_lo=left, x_hi=right)."""
    for face, idx in [("left", 2), ("right", 3), ("bottom", 1)]:
        p = _ac_prop([face])
        assert not p.fs_faces[0], f"{face}: spurious top free surface (fs_faces={p.fs_faces})"
        assert sum(p.fs_faces) == 1 and p.fs_faces[idx], f"{face}: fs_faces={p.fs_faces}"
        assert p.pad[0] == 30, f"{face}: top lost its PML (pad={p.pad})"
        assert p.pad[idx] == 0, f"{face}: free face should have 0 pad (pad={p.pad})"


def test_per_edge_guard_on_unmigrated_and_3d():
    # Acoustic (migrated) accepts per-edge:
    _ac_prop(["top", "left"])
    # 3-D per-edge is not supported yet:
    with pytest.raises(NotImplementedError):
        eq = Acoustic(spatial_order=SO, device=DEV, backend="torch", dim=3)
        PropTorch(eq, shape=(20, 20, 20), free_surface=["top", "left"],
                  abcn=10, dh=DH, dt=DT, use_ckpt=False, impl="eager")


# ---------------------------------------------------------------------------
# Acoustic per-edge: bit-exact top + closed-box symmetry
# ---------------------------------------------------------------------------
def _ac_run(fs, N, src, rec, nt, abcn=20, vp=2000.0):
    eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    prop = PropTorch(eq, shape=(N, N), free_surface=fs, abcn=abcn, dh=DH, dt=DT,
                     use_ckpt=False, impl="eager")
    return prop(_wav(nt), src, rec, models=[torch.full((N, N), vp)])


def test_acoustic_top_only_bit_exact():
    N = 48
    src = torch.tensor([[N // 2, N // 4]], dtype=torch.int64)
    rx = np.arange(3, N - 3, 5, dtype=np.int64)
    rec = torch.from_numpy(np.stack([rx, np.full_like(rx, 3)], -1)[None])
    a = _ac_run(True, N, src, rec, 100)
    b = _ac_run(["top"], N, src, rec, 100)
    c = _ac_run([True, False, False, False], N, src, rec, 100)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_acoustic_closed_box_symmetry():
    M, nt = 65, 200
    c = M // 2
    src = torch.tensor([[c, c]], dtype=torch.int64)
    d1, d2 = 12, 7
    rec = torch.tensor([[[c - d1, c], [c + d1, c], [c - d2, c], [c + d2, c],
                         [c, c - d1], [c, c + d1], [c, c - d2], [c, c + d2]]], dtype=torch.int64)
    r = _ac_run(["top", "bottom", "left", "right"], M, src, rec, nt, abcn=0)
    tr = r[0, :, :, 0]
    peak = tr.abs().max().item()
    rel = lambda a, b: (a - b).abs().max().item() / peak
    assert peak > 0
    assert max(rel(tr[:, 0], tr[:, 1]), rel(tr[:, 2], tr[:, 3])) < 1e-4   # x-mirror
    assert max(rel(tr[:, 4], tr[:, 5]), rel(tr[:, 6], tr[:, 7])) < 1e-4   # z-mirror


def test_acoustic_single_edge_grad_finite():
    N = 44
    src = torch.tensor([[N // 2, N // 3]], dtype=torch.int64)
    rx = np.arange(3, N - 3, 4, dtype=np.int64)
    rec = torch.from_numpy(np.stack([rx, np.full_like(rx, 3)], -1)[None])
    for fs in (["left"], ["right"], ["bottom"], ["top", "left"]):
        eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
        vp = torch.full((N, N), 2000.0, requires_grad=True)
        prop = PropTorch(eq, shape=(N, N), free_surface=fs, abcn=15, dh=DH, dt=DT,
                         use_ckpt=False, impl="eager")
        r = prop(_wav(120), src, rec, models=[vp])
        (r ** 2).sum().backward()
        assert torch.isfinite(vp.grad).all() and vp.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# Elastic per-edge: bit-exact top + gradient consistency + reflection
# ---------------------------------------------------------------------------
def _el_run(fs, N, src, rec, nt, abcn=18, vp_np=None, need_grad=False, rectype=("vx", "vz")):
    eq = Elastic(spatial_order=SO, device=DEV, backend="torch")
    vp = torch.tensor(vp_np if vp_np is not None else np.full((N, N), 2500.0, np.float32),
                      dtype=torch.float32, requires_grad=need_grad)
    vs = torch.full((N, N), 1400.0)
    rho = torch.full((N, N), 1000.0)
    prop = PropTorch(eq, shape=(N, N), free_surface=fs, abcn=abcn, dh=DH, dt=DT,
                     use_ckpt=False, impl="eager", source_type=["sxx", "szz"],
                     receiver_type=list(rectype))
    return prop(_wav(nt, f=14.0, delay=0.04), src, rec, models=[vp, vs, rho]), vp


def test_elastic_top_only_bit_exact():
    N = 48
    src = torch.tensor([[N // 2, N // 3]], dtype=torch.int64)
    rx = np.arange(4, N - 4, 5, dtype=np.int64)
    rec = torch.from_numpy(np.stack([rx, np.full_like(rx, 4)], -1)[None])
    a, _ = _el_run(True, N, src, rec, 100)
    b, _ = _el_run(["top"], N, src, rec, 100)
    c, _ = _el_run([True, False, False, False], N, src, rec, 100)
    assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.parametrize("fs,tag", [(["bottom"], "bottom"),
                                    (["top", "bottom", "left", "right"], "all4")])
def test_elastic_per_edge_gradient_consistency(fs, tag):
    N, nt = 34, 120
    vp0 = np.full((N, N), 2500.0, np.float32)
    vp0[N // 2:, :] = 2800.0
    src = torch.tensor([[N // 2, N // 2]], dtype=torch.int64)
    rx = np.arange(4, N - 4, 5, dtype=np.int64)
    rec = torch.from_numpy(np.stack([rx, np.full_like(rx, 5)], -1)[None])
    with torch.no_grad():
        obs, _ = _el_run(fs, N, src, rec, nt, vp_np=vp0)
    vp_init = vp0 + 70.0
    syn, vp = _el_run(fs, N, src, rec, nt, vp_np=vp_init, need_grad=True)
    (0.5 * ((syn - obs) ** 2).sum()).backward()
    g = vp.grad.detach().numpy()
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(g).ravel()), g.shape))[0][:3]
    eps = 2.0
    worst = 0.0
    for iz, ix in idx:
        vpp = vp_init.copy(); vpp[iz, ix] += eps
        vpm = vp_init.copy(); vpm[iz, ix] -= eps
        with torch.no_grad():
            lp = 0.5 * ((_el_run(fs, N, src, rec, nt, vp_np=vpp)[0] - obs) ** 2).sum().item()
            lm = 0.5 * ((_el_run(fs, N, src, rec, nt, vp_np=vpm)[0] - obs) ** 2).sum().item()
        fd = (lp - lm) / (2 * eps)
        worst = max(worst, abs(fd - g[iz, ix]) / max(abs(fd), abs(g[iz, ix]), 1e-30))
    assert worst < 5e-2, f"{tag}: FD vs adjoint worst rel {worst:.2e}"


@pytest.mark.parametrize("fs,src_xz,rec_xz,rc", [
    (["bottom"], (32, 10), (32, 16), "vz"),
    (["right"], (10, 32), (16, 32), "vx"),
])
def test_elastic_edge_reflects(fs, src_xz, rec_xz, rc):
    N, nt = 64, 480
    src = torch.tensor([[src_xz[0], src_xz[1]]], dtype=torch.int64)
    rec = torch.tensor([[[rec_xz[0], rec_xz[1]]]], dtype=torch.int64)

    def late(fsx):
        r, _ = _el_run(fsx, N, src, rec, nt, abcn=20, rectype=(rc,))
        return (r[0, 280:, 0, 0] ** 2).sum().item()

    e_pml = late(False)
    e_fs = late(fs)
    assert e_fs > 15 * max(e_pml, 1e-30), f"{fs}: E_fs/E_pml={e_fs / max(e_pml, 1e-30):.1f}"


# --- half-cell image for the staggered +h/2 fields -------------------------
# sxz/syz/vz sit half a cell off the plane the free surface lives on, so their
# image must reflect about ``halo-1/2``:  tau_zx(-h/2) = -tau_zx(+h/2)
# (Kristek, Moczo & Archuleta 2002, Table 1, Eq. 7).  Reflecting about ``halo``
# instead pairs -h/2 with +3h/2.  That mistake is INVISIBLE while the surface row
# is force-zeroed -- the zero absorbs it -- and only shows up as a slow
# instability once the (spurious) zeroing is removed, which is why an end-to-end
# stability test needs a large, long, low-velocity model to catch it.  Asserting
# the image relation directly is exact, instant, and cannot be masked.

def _mirror_rows(fn, halo, odd, n=12):
    """Apply an image-extension to a ramp and return the ghost rows it produced."""
    u = torch.arange(1.0, n + 1.0).reshape(n, 1)
    out = fn(u, halo, odd, -2)
    return out[:halo, 0], u[:, 0]


def test_half_cell_image_pairs_minus_h_over_2_with_plus_h_over_2():
    """Low side: ghost[halo-1] must mirror the FIRST INTERIOR node u[halo]."""
    from sweep.equations._free_surface import (
        extend_top_free_surface,
        extend_top_free_surface_cell_centered,
    )
    halo = 2
    ghost_half, u = _mirror_rows(extend_top_free_surface_cell_centered, halo, True)
    # tau_zx(-h/2) = -tau_zx(+h/2):  the node just above the surface mirrors the
    # node just below it, which for a +h/2 field is u[halo] itself.
    assert ghost_half[-1] == -u[halo], (
        f"half-cell image maps -h/2 to {-ghost_half[-1].item()} but the +h/2 node "
        f"is {u[halo].item()}; the mirror is reflecting about the wrong point")
    # the integer-grid image is the one that gets it wrong for these fields
    ghost_int, _ = _mirror_rows(extend_top_free_surface, halo, True)
    assert ghost_int[-1] == -u[halo + 1], "integer-grid image should pair -h/2 with +3h/2"
    assert not torch.equal(ghost_half, ghost_int), "the two images must differ"


def test_half_cell_image_high_side_starts_one_cell_earlier():
    """High side: flipping the axis reverses the half-cell offset, so the node at
    the surface index is already outside the medium and the ghost block covers one
    more row, reflecting about ``halo+1/2``."""
    from sweep.equations._free_surface import (
        extend_top_free_surface_cell_centered_flipped,
    )
    halo, n = 2, 12
    u = torch.arange(1.0, n + 1.0).reshape(n, 1)
    out = extend_top_free_surface_cell_centered_flipped(u, halo, True, -2)
    assert out.shape == u.shape, "image extension must preserve the shape"
    # u[halo] is air on this side and mirrors the first interior node u[halo+1]
    assert out[halo, 0] == -u[halo + 1, 0], (
        f"high-side image gives {out[halo, 0].item()} at the surface index; "
        f"expected -{u[halo + 1, 0].item()} (mirror of the first interior node)")
    # interior is untouched
    assert torch.equal(out[halo + 1:, 0], u[halo + 1:, 0])
