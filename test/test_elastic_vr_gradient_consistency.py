"""Phase 3 gradient consistency test.

Runs the forward + backward of ElasticVRR twice:
1. impl='eager' with PyTorch autograd (reference gradients)
2. impl='c'     full-mode CUDA backward (candidate)

Compares the 6 model gradients (vp, vs, Rp_x, Rp_z, Rs_x, Rs_z) using
the standard solver_gradient_mode_suite thresholds:
    rel_l2 < 1.5,  cosine > 0.8

Canonical model from MEMORY.md:
    nz=48, nx=56, dh=10.0, dt=1.5e-3, nt=120
    abcn=30, spatial_order=4
    Ricker freq=10Hz delay=0.06s
    Source at (nx//2, nz//4); receivers in horizontal line at z=2 stride 6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations import (
    ElasticVRR,
    compute_vector_reflectivity,
)
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda"

NZ, NX = 48, 56
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
SO = 4
FREQ = 10.0
DELAY = 0.06


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry():
    sources = np.array([[NX // 2, NZ // 4]], dtype=np.int64)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    receivers = np.stack([rec_x, np.full_like(rec_x, 2)], axis=-1)[None, ...]
    return sources, receivers


def _surface_geometry():
    """Shallow source + receivers ON the free-surface row (z=0).

    The Robertsson sigma_xx surface correction only modifies the surface-row
    stress and does NOT propagate into the interior, so the interior (z=2)
    receivers of :func:`_geometry` never exercise the sigma_xx adjoint/imaging
    transpose -- only surface receivers do.
    """
    sources = np.array([[NX // 2, 4]], dtype=np.int64)
    rec_x = np.arange(2, NX - 2, 3, dtype=np.int64)
    receivers = np.stack([rec_x, np.zeros_like(rec_x)], axis=-1)[None, ...]
    return sources, receivers


def _make_prop(impl, free_surface=False, **ckpt_kwargs):
    eq = ElasticVRR(spatial_order=SO, device=DEVICE, backend="torch")
    kw = dict(use_ckpt=False)
    kw.update(ckpt_kwargs)
    return PropTorch(
        eq, shape=(NZ, NX),
        abcn=ABCN, dh=DH, dt=DT,
        impl=impl, free_surface=free_surface,
        eager_options={"use_compile": False} if impl == "eager" else None,
        **kw,
    )


def _layered_models_with_grad():
    z = torch.arange(NZ, device=DEVICE, dtype=torch.float32).view(NZ, 1)
    vp = (1800.0 + 8.0 * z).expand(NZ, NX).contiguous()
    vs = vp / 1.73
    rho = (1000.0 + 5.0 * z).expand(NZ, NX).contiguous().clone()
    rho[NZ // 2 :, :] += 200.0
    Rp_x, Rp_z, Rs_x, Rs_z = compute_vector_reflectivity(vp, vs, rho, h=DH)
    models = [vp.clone(), vs.clone(),
              Rp_x.clone(), Rp_z.clone(), Rs_x.clone(), Rs_z.clone()]
    for m in models:
        m.requires_grad_(True)
    return models


def _rel_l2(a, b):
    return ((a - b).pow(2).sum().sqrt() /
            b.pow(2).sum().sqrt().clamp_min(1e-30)).item()


def _cosine(a, b):
    af = a.flatten()
    bf = b.flatten()
    return (af @ bf / (af.norm() * bf.norm()).clamp_min(1e-30)).item()


def _compute_grads(impl, wavelet, sources, receivers, target, prop_ckpt=None, **prop_kwargs):
    """Build a fresh propagator + models, run forward+backward, return grads."""
    models = _layered_models_with_grad()
    prop = _make_prop(impl, **(prop_ckpt or {}))
    syn = prop(wavelet, sources, receivers, models=models, **prop_kwargs)
    loss = (syn - target).pow(2).sum()
    loss.backward()
    grads = {
        "vp":   models[0].grad.detach().clone(),
        "vs":   models[1].grad.detach().clone(),
        "Rp_x": models[2].grad.detach().clone(),
        "Rp_z": models[3].grad.detach().clone(),
        "Rs_x": models[4].grad.detach().clone(),
        "Rs_z": models[5].grad.detach().clone(),
    }
    return grads, syn.detach()


def test_gradient_consistency_full_mode():
    """impl='c' full-mode backward should produce gradients matching
    impl='eager' (PyTorch autograd) on all 6 model parameters."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wavelet = _wavelet()
    sources, receivers = _geometry()

    # Build a perturbed model as target so the loss is non-trivial.
    target_models = _layered_models_with_grad()
    with torch.no_grad():
        target_models[0] += 30.0  # perturb vp
    target_prop = _make_prop("eager")
    with torch.no_grad():
        target_syn = target_prop(wavelet, sources, receivers, models=target_models)

    # Reference: impl='eager' with autograd
    eager_grads, eager_syn = _compute_grads(
        "eager", wavelet, sources, receivers, target_syn
    )
    # Candidate: impl='c' full backward
    c_grads, c_syn = _compute_grads(
        "c", wavelet, sources, receivers, target_syn
    )

    syn_rel = _rel_l2(c_syn, eager_syn)
    print(f"forward syn rel L2: {syn_rel:.4e}")
    assert syn_rel < 1e-3

    # Compare each model gradient
    results = []
    for name in ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"):
        rel = _rel_l2(c_grads[name], eager_grads[name])
        cos = _cosine(c_grads[name], eager_grads[name])
        results.append((name, rel, cos,
                        eager_grads[name].abs().max().item(),
                        c_grads[name].abs().max().item()))
        print(f"  grad_{name:5s}: rel_l2 = {rel:.4e}  cos = {cos:.4f}  "
              f"eager_max={eager_grads[name].abs().max().item():.4e} "
              f"c_max={c_grads[name].abs().max().item():.4e}")

    # Acceptance thresholds from solver_gradient_mode_suite defaults.
    failed = []
    for name, rel, cos, _, _ in results:
        if not (rel < 1.5 and cos > 0.8):
            failed.append(f"{name}: rel_l2={rel:.4e}, cos={cos:.4f}")
    assert not failed, "Gradient consistency failed on:\n  " + "\n  ".join(failed)


def test_gradient_consistency_boundary_saving():
    """impl='c' boundary-saving backward (the ckpt-style memory-light mode)
    should match eager autograd on all 6 model parameters. The forward
    wavefield is reconstructed in reverse time from saved boundary strips
    rather than stored every step."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wavelet = _wavelet()
    sources, receivers = _geometry()

    target_models = _layered_models_with_grad()
    with torch.no_grad():
        target_models[0] += 30.0
    target_prop = _make_prop("eager")
    with torch.no_grad():
        target_syn = target_prop(wavelet, sources, receivers, models=target_models)

    eager_grads, eager_syn = _compute_grads(
        "eager", wavelet, sources, receivers, target_syn
    )
    # Candidate: impl='c' boundary-saving backward (GPU storage).
    c_grads, c_syn = _compute_grads(
        "c", wavelet, sources, receivers, target_syn, use_boundary_saving=True
    )

    syn_rel = _rel_l2(c_syn, eager_syn)
    print(f"[bs] forward syn rel L2: {syn_rel:.4e}")
    assert syn_rel < 1e-3

    failed = []
    for name in ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"):
        rel = _rel_l2(c_grads[name], eager_grads[name])
        cos = _cosine(c_grads[name], eager_grads[name])
        print(f"  [bs] grad_{name:5s}: rel_l2 = {rel:.4e}  cos = {cos:.4f}")
        if not (rel < 1.5 and cos > 0.8):
            failed.append(f"{name}: rel_l2={rel:.4e}, cos={cos:.4f}")
    assert not failed, "BS gradient consistency failed on:\n  " + "\n  ".join(failed)


def _run_ckpt_case(tag, prop_ckpt):
    wavelet = _wavelet()
    sources, receivers = _geometry()
    target_models = _layered_models_with_grad()
    with torch.no_grad():
        target_models[0] += 30.0
    with torch.no_grad():
        target_syn = _make_prop("eager")(wavelet, sources, receivers, models=target_models)

    eager_grads, _ = _compute_grads("eager", wavelet, sources, receivers, target_syn)
    c_grads, c_syn = _compute_grads(
        "c", wavelet, sources, receivers, target_syn, prop_ckpt=prop_ckpt
    )
    failed = []
    for name in ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"):
        rel = _rel_l2(c_grads[name], eager_grads[name])
        cos = _cosine(c_grads[name], eager_grads[name])
        print(f"  [{tag}] grad_{name:5s}: rel_l2 = {rel:.4e}  cos = {cos:.4f}")
        if not (rel < 1.5 and cos > 0.8):
            failed.append(f"{name}: rel_l2={rel:.4e}, cos={cos:.4f}")
    assert not failed, f"{tag} gradient consistency failed on:\n  " + "\n  ".join(failed)


def test_gradient_consistency_ckpt_chunk():
    """impl='c' chunked-checkpoint backward should match eager autograd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _run_ckpt_case("ckpt_chunk", {"use_ckpt": True, "ckpt_mode": "chunk", "ckpt_chunks": 16})


def test_gradient_consistency_ckpt_recursive():
    """impl='c' recursive-checkpoint backward should match eager autograd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _run_ckpt_case("ckpt_recursive",
                   {"use_ckpt": True, "ckpt_mode": "recursive", "ckpt_num": 5})


@pytest.mark.parametrize("tag,prop_ckpt,fwd_kw", [
    ("full",           {},                                                          {}),
    ("bs",             {},                                                          {"use_boundary_saving": True}),
    ("ckpt_chunk",     {"use_ckpt": True, "ckpt_mode": "chunk", "ckpt_chunks": 16}, {}),
    ("ckpt_recursive", {"use_ckpt": True, "ckpt_mode": "recursive", "ckpt_num": 5}, {}),
])
def test_gradient_consistency_free_surface(tag, prop_ckpt, fwd_kw):
    """All four backward modes (full / boundary-saving / chunk- and recursive-
    checkpoint) must match eager autograd with free_surface=True (image-method
    top surface + sigma_zz = sigma_xz = 0 BC)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wavelet = _wavelet()
    sources, receivers = _geometry()

    target_models = _layered_models_with_grad()
    with torch.no_grad():
        target_models[0] += 30.0
    with torch.no_grad():
        target_syn = _make_prop("eager", free_surface=True)(
            wavelet, sources, receivers, models=target_models)

    # eager autograd reference (mode-independent), free surface ON
    eager_grads, _ = _compute_grads(
        "eager", wavelet, sources, receivers, target_syn,
        prop_ckpt={"free_surface": True})
    # candidate impl='c' for this backward mode, free surface ON
    c_grads, _ = _compute_grads(
        "c", wavelet, sources, receivers, target_syn,
        prop_ckpt={"free_surface": True, **prop_ckpt}, **fwd_kw)

    failed = []
    for name in ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"):
        rel = _rel_l2(c_grads[name], eager_grads[name])
        cos = _cosine(c_grads[name], eager_grads[name])
        print(f"  [{tag}/fs] grad_{name:5s}: rel_l2 = {rel:.4e}  cos = {cos:.4f}")
        if not (rel < 1.5 and cos > 0.8):
            failed.append(f"{name}: rel_l2={rel:.4e}, cos={cos:.4f}")
    assert not failed, (f"[{tag}/fs] free-surface gradient consistency failed:\n  "
                        + "\n  ".join(failed))


@pytest.mark.parametrize("tag,prop_ckpt,fwd_kw", [
    ("full",           {},                                                          {}),
    ("bs",             {},                                                          {"use_boundary_saving": True}),
    ("ckpt_chunk",     {"use_ckpt": True, "ckpt_mode": "chunk", "ckpt_chunks": 16}, {}),
    ("ckpt_recursive", {"use_ckpt": True, "ckpt_mode": "recursive", "ckpt_num": 5}, {}),
])
def test_gradient_consistency_free_surface_surface_receivers(tag, prop_ckpt, fwd_kw):
    """Free-surface gradient consistency recorded ON the surface row (z=0).

    Unlike :func:`test_gradient_consistency_free_surface` (interior z=2
    receivers), this records on the free surface so the loss actually depends
    on the Robertsson sigma_xx surface correction -- this is the geometry that
    exercises the CUDA sigma_xx adjoint (``evr_stress_adjoint_prepare`` surface
    branch) and imaging (``calculate_grad_evr_nobs`` surface material
    derivative).  All four backward modes must match eager autograd.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wavelet = _wavelet()
    sources, receivers = _surface_geometry()
    rt = {"receiver_type": ["sxx"]}   # record the corrected surface stress

    target_models = _layered_models_with_grad()
    with torch.no_grad():
        target_models[0] += 30.0
    with torch.no_grad():
        target_syn = _make_prop("eager", free_surface=True, **rt)(
            wavelet, sources, receivers, models=target_models)

    eager_grads, _ = _compute_grads(
        "eager", wavelet, sources, receivers, target_syn,
        prop_ckpt={"free_surface": True, **rt})
    c_grads, _ = _compute_grads(
        "c", wavelet, sources, receivers, target_syn,
        prop_ckpt={"free_surface": True, **rt, **prop_ckpt}, **fwd_kw)

    failed = []
    for name in ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"):
        rel = _rel_l2(c_grads[name], eager_grads[name])
        cos = _cosine(c_grads[name], eager_grads[name])
        print(f"  [{tag}/fs-surf] grad_{name:5s}: rel_l2 = {rel:.4e}  cos = {cos:.4f}")
        if not (rel < 1.5 and cos > 0.8):
            failed.append(f"{name}: rel_l2={rel:.4e}, cos={cos:.4f}")
    assert not failed, (f"[{tag}/fs-surf] surface-receiver gradient consistency "
                        f"failed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
