"""Rim-only boundary saving:
``PropSEM(memory=BoundarySaving(), bs_band_mode='rim')``.

Boundary saving normally stores the whole C-PML band. The band is an AREA and
its interface is a LINE, so storing only the line cuts the history by 7-12x at
production mesh sizes -- but it stops reproducing the wavefield inside the band,
and the model cells next to the band see that error. (Through the
edge-extended pad it used to fold back onto the outermost model pixels too;
the pad is cut from the gradient now -- test_sem_pad_gradient.py.) These tests
pin both halves of that bargain.

The configuration matters more than anything else here. With the source centred,
a free surface and a short record, the absorbing band barely sees the wavefield
and rim mode agrees with the full tape to 1e-4 EVERYWHERE -- a test built on
that setup passes no matter how wrong the rim reconstruction is. So the
agreement tests run a discriminating case: source at 15% of X so the wavefield
hits the left band head-on, no free surface so the band is 4-sided, and a
1500/4500 contrast to drive strong reflections back into it. On that case the
uncropped cosine is 0.989 on impl='c' (0.90 while the pad was still folded onto
the outer ring), 0.99988 two cells in and 1.0 at eight, which is what makes the
cropped 1.0 meaningful.

``dt`` comes from the SMALLEST GLL spacing inside an element (~0.172 x element),
not from the nominal node spacing: an FD-style CFL diverges into all-NaN here.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import PropSEM
from sweep.sem.boundary_saving import cpml_node_masks
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, F0, ABCN, N_ORDER = 49, 65, 20.0, 8.0, 6, 4
NT, VP_LO, VP_HI, SRC_X_FRAC = 900, 1500.0, 4500.0, 0.15
DT = 0.30 * (0.172 * N_ORDER * DH) / VP_HI     # GLL-limited, with headroom
CROP = 8            # pixels the rim mode is documented to be exact beyond


def _vp():
    z, x = np.arange(NZ)[:, None], np.arange(NX)[None, :]
    return (VP_LO + (VP_HI - VP_LO) * z / (NZ - 1)
            + 0.08 * (VP_HI - VP_LO) * np.sin(x / 9.0)).astype(np.float32)


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def _prop(impl, mode, memory=None):
    return PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=N_ORDER, free_surface=False,
                   abcn=ABCN * N_ORDER, f_dominant=F0, dev="cuda",
                   dtype=torch.float32, impl=impl, memory=memory,
                   bs_band_mode=mode)


def _run(impl, memory, mode):
    p = _prop(impl, mode, memory)
    src = np.array([[int(SRC_X_FRAC * NX), 4]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 4)], -1)[None]
    vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
    s = p(_wav(), src, rec, models=[vp])
    assert torch.isfinite(s).all(), "forward diverged (check the GLL CFL)"
    (s ** 2).sum().backward()
    g = vp.grad.detach().cpu().numpy().astype(np.float64)
    assert np.isfinite(g).all(), "gradient not finite"
    del p, vp, s
    gc.collect()
    torch.cuda.empty_cache()
    return g


def _cos(a, b, crop=0):
    sl = (slice(crop, NZ - crop) if crop else slice(None),
          slice(crop, NX - crop) if crop else slice(None))
    u, v = a[sl].reshape(-1), b[sl].reshape(-1)
    return float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))


# ------------------------------ the bargain -------------------------------- #
@pytest.mark.parametrize("impl", ["c", "eager"])
def test_rim_matches_full_tape_away_from_the_edge(impl):
    ref = _run(impl, Full(), "full")
    rim = _run(impl, BoundarySaving(), "rim")
    c = _cos(rim, ref, CROP)
    assert c >= 1 - 1e-6, f"cropped grad cos={c:.8f}"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_full_band_mode_is_still_exact_everywhere(impl):
    """The default must not have moved: exact including the model edge."""
    ref = _run(impl, Full(), "full")
    full = _run(impl, BoundarySaving(), "full")
    assert _cos(full, ref) >= 1 - 1e-7, f"grad cos={_cos(full, ref):.8f}"


def test_the_configuration_actually_discriminates():
    """Without this the two tests above would pass on a broken rim mode.

    The point of the 4-sided band, the off-centre source and the 1500/4500
    contrast is to put real energy into the absorbing layer. If a future change
    to the setup made the uncropped rim gradient agree too, the cropped
    assertions would no longer prove anything.
    """
    ref = _run("c", Full(), "full")
    rim = _run("c", BoundarySaving(), "rim")
    assert _cos(rim, ref) < 0.99, (
        f"uncropped rim cos={_cos(rim, ref):.6f} -- the band sees too little "
        "energy for this test to discriminate")


# ------------------------------ the saving --------------------------------- #
def test_rim_stores_far_fewer_nodes():
    full_idx = cpml_node_masks(_prop("c", "full", BoundarySaving())._build_eq(
        torch.tensor(_vp(), device="cuda")))[1]
    rim_idx = cpml_node_masks(_prop("c", "rim", BoundarySaving())._build_eq(
        torch.tensor(_vp(), device="cuda")))[1]
    assert rim_idx.numel() < full_idx.numel() / 5, (
        f"rim {rim_idx.numel()} vs band {full_idx.numel()} nodes")


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="bs_band_mode"):
        _prop("c", "ring")
