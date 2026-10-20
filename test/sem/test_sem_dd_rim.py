"""``bs_band_mode='rim'`` on the decomposed and the fluid-solid paths.

Rim mode rings the C-PML band's interface LINE instead of the band AREA. It was
wired for the single-domain acoustic path first (``test_sem_bs_rim.py``); the
two paths pinned here used to hard-code the "full" convention and SILENTLY
ignore the flag -- a caller asking for rim on a strip or on the coupled
propagator got full-band memory and no warning.

Three separate claims, one test each:

* the ring really shrinks -- a perimeter instead of an area;
* the decomposition does not change the bargain: rim at ``px>1`` reproduces rim
  at ``px=1`` to the atomics floor, so a cut sitting inside the band neither
  helps nor hurts;
* the gradient is exact once the band is cropped off, and NOT before. The
  configurations drive energy into the band head-on (source one element off it),
  which is what makes the cropped comparison worth anything --
  ``test_the_configuration_actually_discriminates`` fails if it stops doing so.

ELEMENT rings are deliberately EXEMPT from rim and stay whole-band. The elastic
C-PML memory fields live only inside the band, and zeroing them corrupts the
adjoint entering the INTERIOR, not merely the band gradient (measured, see
``SEMReconState``); dropping them takes even *full* node mode to cos 0.755 on
the fs case below. ``test_fs_rim_keeps_the_element_rings`` pins that they are
untouched, and the fs gradient test is what would catch it if they were not.

Cropping is in ELEMENTS of C-PML thickness, not pixels: the fs models are
element-local, and the band is ``abcn // n_order`` elements thick plus its
one-element rim.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import (AcousticElasticSEM2D, AcousticSEM3D,
                                 ElasticSEM2DDeformed)
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 25.0, 6.0
# --- acoustic 2-D strips: padded, so the band lives in the PAD --------------
NZ, NX, ABCN, PAD = 49, 65, 6, 6 * 4
# nt matters: the bar in test_the_configuration_actually_discriminates is
# an excitation measure, and the band only fills up as the record runs --
# measured uncropped rim cos 0.9998 at nt=300 but 0.9953 at nt=400, and 0.9984
# at nt=400 once the pad was cut from the gradient (it had folded the band's
# error onto the outer ring).
NT, DT = 400, 0.30 * (0.172 * NO * DH) / 4500.0
CROP = 4                      # model pixels: cos 0.99996 at 2, 0.9999999 at 4
# --- coupled: element-local models over the WHOLE mesh, band included, so the
# band overlaps REAL material. FS_NZ x FS_NX is that mesh (the crop below is in
# its coordinates); the model PropSEM adds the band outside is smaller by it --
FS_NZ, FS_NX, FS_NW, FS_ABCN = 49, 97, 8, 3
FS_SHAPE = (FS_NZ - FS_ABCN * NO, FS_NX - 2 * FS_ABCN * NO)
FS_NT, FS_DT = 300, 1.2e-3
# Crop the band PLUS its one-element rim. One element short is not a
# detail: on the elastic case below, K=abcn gives cos 0.997 and
# K=abcn+1 gives exactly 1.0.
FS_CROP_E = FS_ABCN + 1


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def _wav(nt, dt, amp=1.0):
    t = np.arange(nt) * dt
    a = (np.pi * F0 * (t - 1.3 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * amp,
                        device="cuda")


# ============================ acoustic 2-D strips ========================== #
def _dd_run(mode, px, memory):
    """Returns (model gradient (NZ, NX), total band nodes over the strips)."""
    p = S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                  n_order=NO, free_surface=False, abcn=ABCN * NO,
                  memory=BoundarySaving() if memory == "boundary" else None,
                  f_dominant=F0, dev="cuda", impl="c", bs_band_mode=mode)
    z, x = np.arange(NZ)[:, None], np.arange(NX)[None, :]
    vp = (1500 + 3000 * z / (NZ - 1) + 240 * np.sin(x / 9.0)).astype(np.float32)
    m = torch.tensor(vp, device="cuda").requires_grad_(True)
    eq, _ = p._resolve_model_batch([m.detach()])
    nxg = eq.mesh.nex * NO + 1
    node = lambda ix, iz: (PAD + iz) * nxg + (PAD + ix)      # noqa: E731
    src = node(int(0.15 * NX), 4)
    rec = torch.tensor([node(ix, 4) for ix in range(3, NX - 3, 2)]).cuda()
    d = SEMModelParallel(p, StripTopology(px))
    out = d(_wav(NT, DT), src, rec, [m])
    assert torch.isfinite(out).all(), "forward diverged"
    (out ** 2).sum().backward()
    pat = torch.zeros(eq.mesh.n_global, device="cuda")
    pat[int(src)] = 1.0
    parts, _ = d._parts(eq, (pat,), rec)
    nb = sum(int(pt.c["band_idx"].numel()) for pt in parts)
    g = m.grad.detach().clone()
    assert torch.isfinite(g).all(), "gradient not finite"
    del p, d, m, out
    gc.collect()
    torch.cuda.empty_cache()
    return g, nb


_CACHE: dict = {}


def _dd(mode, px, memory):
    key = (mode, px, memory)
    if key not in _CACHE:
        _CACHE[key] = _dd_run(mode, px, memory)
    return _CACHE[key]


def _crop(g, c=CROP):
    return g[c:NZ - c, c:NX - c]


@pytest.mark.parametrize("px", [1, 2])
def test_dd_rim_stores_far_fewer_band_nodes(px):
    """A perimeter, not an area. The strips used to ignore the flag entirely,
    so the count is the part that actually proves the wiring landed."""
    _, full = _dd("full", px, "boundary")
    _, rim = _dd("rim", px, "boundary")
    assert rim * 5 < full, f"px={px}: rim {rim} nodes vs full {full}"


@pytest.mark.parametrize("px", [1, 2])
def test_dd_rim_matches_full_tape_beyond_the_band(px):
    ref, _ = _dd("full", px, None)
    rim, _ = _dd("rim", px, "boundary")
    c = _cos(_crop(rim), _crop(ref))
    assert c >= 1 - 1e-6, f"px={px}: cropped grad cos={c:.8f}"


@pytest.mark.parametrize("px", [1, 2])
def test_dd_full_band_mode_is_still_exact_everywhere(px):
    """The default must not have moved when rim was wired in."""
    ref, _ = _dd("full", px, None)
    full, _ = _dd("full", px, "boundary")
    c = _cos(full, ref)
    assert c >= 1 - 1e-7, f"px={px}: grad cos={c:.8f}"


def test_dd_rim_bargain_does_not_depend_on_px():
    """The cut sits inside the band, so this is not free: both strips must
    agree on band MEMBERSHIP for the shared nodes, which they only do because
    the profile is sliced from one global build (see ``_band_nodes``)."""
    one, _ = _dd("rim", 1, "boundary")
    two, _ = _dd("rim", 2, "boundary")
    c = _cos(one, two)
    assert c >= 1 - 1e-6, f"rim px=1 vs px=2 cos={c:.8f}"


def test_the_configuration_actually_discriminates():
    """If the band never saw the wavefield, rim would agree everywhere and the
    cropped tests above would pass no matter how wrong rim was."""
    ref, _ = _dd("full", 1, None)
    rim, _ = _dd("rim", 1, "boundary")
    c = _cos(rim, ref)
    assert c <= 0.999, f"uncropped rim cos={c:.8f} -- band is not excited"


# ============================== fluid-solid =============================== #
def _fs_run(mode, px, memory):
    p = S.PropSEM(AcousticElasticSEM2D, shape=FS_SHAPE, dh=DH, dt=FS_DT,
                  nt=FS_NT, n_order=NO, free_surface=("top",),
                  abcn=FS_ABCN * NO,
                  memory=BoundarySaving() if memory == "boundary" else None,
                  f_dominant=F0, dev="cuda", impl="c",
                  receiver_type=["p", "vx", "vz"], bs_band_mode=mode)
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < FS_NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    ms = [torch.where(fl, a, b).float().cuda().requires_grad_(True)
          for a, b in ((1500.0, 2300.0), (0.0, 1250.0), (1000.0, 2100.0))]
    eq, _ = p._resolve_model_batch([t.detach() for t in ms])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + (NO + 1)          # one element off the left band
    rec = (torch.arange(4, nxg - 4, 4) + (FS_NW + 1) * nxg).cuda()
    d = SEMModelParallel(p, StripTopology(px))
    out = d(_wav(FS_NT, FS_DT, 1e6), src, rec, ms)
    assert torch.isfinite(out).all(), "forward diverged"
    (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)).backward()
    pats = [torch.zeros(eq.mesh.n_global, device="cuda") for _ in range(3)]
    pats[0][int(src)] = 1.0
    parts, _ = d._parts(eq, tuple(pats), rec)
    counts = (sum(int(pt.c["band_idx"].numel()) for pt in parts),
              sum(int(pt.c["band_node_idx"].numel()) for pt in parts),
              sum(int(pt.c["band_elem_idx"].numel()) for pt in parts))
    ze, xe = eq.mesh.z_e.flatten(1).mean(1), eq.mesh.x_e.flatten(1).mean(1)
    g = [t.grad.detach().clone() for t in ms]
    del p, d, ms, out
    gc.collect()
    torch.cuda.empty_cache()
    return g, counts, (ze, xe)


_FS_CACHE: dict = {}


def _fs(mode, px, memory):
    key = (mode, px, memory)
    if key not in _FS_CACHE:
        _FS_CACHE[key] = _fs_run(mode, px, memory)
    return _FS_CACHE[key]


def _fs_interior(ze, xe, k=FS_CROP_E):
    """Crop only the sides that HAVE a band: the top is a free surface here, so
    cropping it would just throw away interior and make the comparison noisier
    (measured: symmetric crop moves the fs numbers by an order of magnitude)."""
    hi = (FS_NZ - 1 - k * NO) * DH
    lx, hx = k * NO * DH, (FS_NX - 1 - k * NO) * DH
    keep = (ze <= hi) & (xe >= lx) & (xe <= hx)
    assert bool(keep.any()), "crop left no interior -- the grid is too small"
    return keep


def test_fs_rim_shrinks_both_domains_node_rings():
    """Fluid band on nodes, solid band on nodes AND elements; rim is a node
    notion, so the two node rings shrink and the element ring does not."""
    _, (ac_f, el_f, _), _ = _fs("full", 1, "boundary")
    _, (ac_r, el_r, _), _ = _fs("rim", 1, "boundary")
    assert ac_r * 3 < ac_f, f"fluid ring {ac_r} vs {ac_f}"
    assert el_r * 3 < el_f, f"solid node ring {el_r} vs {el_f}"


def test_fs_rim_keeps_the_element_rings():
    """Exempt on purpose -- see the module docstring. If this ever flips,
    ``test_fs_rim_matches_full_tape_beyond_the_band`` is what breaks."""
    _, (_, _, e_full), _ = _fs("full", 1, "boundary")
    _, (_, _, e_rim), _ = _fs("rim", 1, "boundary")
    assert e_rim == e_full > 0, f"element ring {e_rim} vs {e_full}"


@pytest.mark.parametrize("px", [1, 2])
def test_fs_rim_matches_full_tape_beyond_the_band(px):
    """Unlike the acoustic and elastic paths, coupled rim does NOT come all the
    way back to 1.0 -- it plateaus. Measured over the band-aware mask, worst of
    the three models:

    ======  ==============  ===========  ==========
    crop K  full-band BS    rim          elements
    ======  ==============  ===========  ==========
    3       1.000000000     0.9838       162
    4       1.000000000     0.9999780    128
    5       1.000000000     0.9999883     98
    ======  ==============  ===========  ==========

    Full-band BS is EXACTLY 1.0 over the same mask, so the ~2e-5 is rim's own
    residual, not the atomics floor -- and it is the ONE path that plateaus
    instead of converging. The likely source is that the two domains' bands are
    built separately here, so their rims meet at the seabed; that is not
    verified, and the bar records the measurement, not a theory."""
    ref, _, _ = _fs("full", px, None)
    rim, _, (ze, xe) = _fs("rim", px, "boundary")
    keep = _fs_interior(ze, xe)
    for j, name in enumerate(("vp", "vs", "rho")):
        c = _cos(rim[j][keep], ref[j][keep])
        assert c >= 1 - 5e-5, f"px={px} {name}: cropped grad cos={c:.8f}"


@pytest.mark.parametrize("px", [1, 2])
def test_fs_full_band_mode_is_exact_over_the_same_mask(px):
    """What makes the bar above meaningful: BS itself has no error here."""
    ref, _, _ = _fs("full", px, None)
    full, _, (ze, xe) = _fs("full", px, "boundary")
    keep = _fs_interior(ze, xe)
    for j, name in enumerate(("vp", "vs", "rho")):
        c = _cos(full[j][keep], ref[j][keep])
        assert c >= 1 - 1e-8, f"px={px} {name}: cropped grad cos={c:.9f}"


def test_fs_configuration_actually_discriminates():
    ref, _, _ = _fs("full", 1, None)
    rim, _, _ = _fs("rim", 1, "boundary")
    c = min(_cos(rim[j], ref[j]) for j in range(3))
    assert c <= 0.99, f"uncropped fs rim cos={c:.8f} -- band is not excited"


# ============================== the guard ================================= #
def test_rim_is_refused_when_the_band_leaves_no_interior():
    """On a small 3-D grid the C-PML band covers EVERY node, so the rim is
    empty and boundary saving would silently store nothing. Measured on the
    3-D DD test grid: band 5746/5746 nodes, rim 0.

    PropSEM cannot build that grid any more -- it adds the absorbing layer
    outside the model, which always leaves an interior -- so the same mesh
    (6 x 3 x 4 elements, a 2-element C-PML on five faces that overlaps
    across y) is built directly and the strip driver reached through its
    low-level entry, the way SEMCudaPropagator reaches it."""
    from sweep.sem.cpml.acoustic3d import build_cpml3d
    from sweep.sem.mesh.hex8 import build_hex8_mesh
    dh, dt, (nex, ney, nez) = 40.0, 1.2e-3, (6, 3, 4)
    mesh = build_hex8_mesh(nex, ney, nez, nex * NO * dh, ney * NO * dh,
                           nez * NO * dh, NO, device="cuda")
    v = (2500.0 + 200.0 * torch.rand(mesh.n_elements, *(NO + 1,) * 3)).cuda()
    cp = build_cpml3d(mesh, dt=dt,
                      sides=("bottom", "front", "back", "left", "right"),
                      pml_thickness_elements=2, c_max=float(v.max()),
                      f_dominant=8.0)
    eq = AcousticSEM3D.build(mesh, v, dirichlet=("top",), cpml=cp)
    eq.bs_band_mode = "rim"
    ng = eq.mesh.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[ng // 2] = 1.0
    d = SEMModelParallel.for_equation(eq, dt, 32, StripTopology(2))
    with pytest.raises(ValueError, match="nothing to store"):
        d._parts3d(eq, (pat,), torch.tensor([ng // 3]).cuda())


# ============================ elastic 2-D strips ========================== #
# The elastic BS reconstruction is only exact when the band is a large enough
# fraction of the domain and the record is long enough (a pre-existing property,
# unrelated to rim: on 49x97 / abcn=3 / nt=300 even FULL-band BS lands at cos
# 0.35). This configuration is the one where full-band BS is exactly 1.0, so a
# rim failure here cannot be blamed on BS itself.
EL_NZ, EL_NX, EL_ABCN = 65, 129, 5
# the mesh, as for the coupled case; with no free surface the band sits
# outside the model on all four sides
EL_SHAPE = (EL_NZ - 2 * EL_ABCN * NO, EL_NX - 2 * EL_ABCN * NO)
EL_NT, EL_DT = 800, 7e-4
EL_CROP_E = EL_ABCN + 1


def _el_run(mode, px, memory):
    p = S.PropSEM(ElasticSEM2DDeformed, shape=EL_SHAPE, dh=DH, dt=EL_DT,
                  nt=EL_NT, n_order=NO, free_surface=False,
                  abcn=EL_ABCN * NO,
                  memory=BoundarySaving() if memory == "boundary" else None,
                  f_dominant=8.0, dev="cuda", impl="c", bs_band_mode=mode)
    m = p.mesh
    torch.manual_seed(0)
    ms = [(lo + w * torch.rand(m.n_elements, m.ngll, m.ngll)).cuda(
        ).requires_grad_(True)
        for lo, w in ((2500.0, 200.0), (1400.0, 100.0), (2000.0, 100.0))]
    eq, _ = p._resolve_model_batch([t.detach() for t in ms])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + nxg // 8
    rec = (torch.arange(4, nxg - 4, 4) + 4 * NO * nxg).cuda()
    t = np.arange(EL_NT) * EL_DT
    a = (np.pi * 8.0 * (t - 1.3 / 8.0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    d = SEMModelParallel(p, StripTopology(px))
    out = d(w, src, rec, ms)
    assert torch.isfinite(out).all(), "forward diverged"
    (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)).backward()
    pats = [torch.zeros(eq.mesh.n_global, device="cuda") for _ in range(2)]
    pats[0][int(src)] = 1.0
    parts, _ = d._parts(eq, tuple(pats), rec)
    counts = (sum(int(pt.c["band_node_idx"].numel()) for pt in parts),
              sum(int(pt.c["band_elem_idx"].numel()) for pt in parts))
    ze, xe = eq.mesh.z_e.flatten(1).mean(1), eq.mesh.x_e.flatten(1).mean(1)
    g = [t.grad.detach().clone() for t in ms]
    del p, d, ms, out
    gc.collect()
    torch.cuda.empty_cache()
    return g, counts, (ze, xe)


_EL_CACHE: dict = {}


def _el(mode, px, memory):
    key = (mode, px, memory)
    if key not in _EL_CACHE:
        _EL_CACHE[key] = _el_run(mode, px, memory)
    return _EL_CACHE[key]


def _el_interior(ze, xe, k=EL_CROP_E):
    lo, hi = k * NO * DH, (EL_NZ - 1 - k * NO) * DH
    lx, hx = k * NO * DH, (EL_NX - 1 - k * NO) * DH
    keep = (ze >= lo) & (ze <= hi) & (xe >= lx) & (xe <= hx)
    assert bool(keep.any()), "crop left no interior"
    return keep


def test_el_rim_shrinks_node_rings_and_keeps_element_rings():
    _, (n_full, e_full), _ = _el("full", 1, "boundary")
    _, (n_rim, e_rim), _ = _el("rim", 1, "boundary")
    assert n_rim * 5 < n_full, f"node ring {n_rim} vs {n_full}"
    assert e_rim == e_full > 0, f"element ring {e_rim} vs {e_full}"


@pytest.mark.parametrize("px", [1, 2])
def test_el_rim_matches_full_tape_beyond_the_band(px):
    """Exact, not merely close -- and one element of crop short of the rim it
    is not: measured worst-model cos 0.0466 at K=0, 0.99746 at K=abcn,
    1.0000000 at K=abcn+1 and beyond, identically at px=1 and px=2."""
    ref, _, _ = _el("full", px, None)
    rim, _, (ze, xe) = _el("rim", px, "boundary")
    keep = _el_interior(ze, xe)
    for j, name in enumerate(("vp", "vs", "rho")):
        c = _cos(rim[j][keep], ref[j][keep])
        assert c >= 1 - 1e-7, f"px={px} {name}: cropped grad cos={c:.8f}"


@pytest.mark.parametrize("px", [1, 2])
def test_el_full_band_mode_is_exact_over_the_same_mask(px):
    ref, _, _ = _el("full", px, None)
    full, _, (ze, xe) = _el("full", px, "boundary")
    keep = _el_interior(ze, xe)
    for j in range(3):
        c = _cos(full[j][keep], ref[j][keep])
        assert c >= 1 - 1e-8, f"px={px}: cropped grad cos={c:.9f}"


def test_el_configuration_actually_discriminates():
    ref, _, _ = _el("full", 1, None)
    rim, _, _ = _el("rim", 1, "boundary")
    c = min(_cos(rim[j], ref[j]) for j in range(3))
    assert c <= 0.5, f"uncropped elastic rim cos={c:.8f} -- band is not excited"
