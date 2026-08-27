"""Displacement-based elastic TTI (Oh et al. 2020, GJI ggaa295) tests.

CPU: contract, free-surface guard, isotropic rotation invariance (with
eps=eta=0 the Bond-rotated stiffness must not depend on theta), long-run
stability, and finite nonzero eager gradients for all six parameters
(vh, vs, rho, epsilon, eta, theta).

CUDA (canonical 2-D grid 48x56, order 4, abcn 30): forward c-vs-eager,
full backward c-vs-eager on all six parameter gradients, bs/ckpt against
full, and a small multi-parameter FWI smoke that drives the misfit down
with the compiled backend (forward + inversion round trip).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.equations import ElasticTTI2nd, supports_torch_binding
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

MODEL_NAMES = ["vh", "vs", "rho", "epsilon", "eta", "theta"]


def _ricker(t, f):
    arg = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def test_export_contract():
    eq = ElasticTTI2nd(spatial_order=4)
    assert [m.name for m in eq.MODEL_SPECS] == MODEL_NAMES
    assert len(eq.FIELD_SPECS) == 12
    assert eq.prepare_models_for_c is True
    assert eq.default_pml_type == "cpmls"
    assert supports_torch_binding("ElasticTTI2nd") is True


def test_free_surface_raises():
    eq = ElasticTTI2nd(spatial_order=4, device="cpu", backend="torch")
    with pytest.raises(NotImplementedError, match="anisotropic"):
        PropTorch(
            eq, (24, 28), dh=10.0, dt=1e-3, nt=8, abcn=4,
            free_surface=True, device="cpu", impl="eager",
            eager_options=EagerOptions(use_compile=False), use_ckpt=False,
        )


def _cpu_prop(shape, nt, dt, abcn=20):
    eq = ElasticTTI2nd(spatial_order=4, device="cpu", backend="torch")
    return PropTorch(
        eq, shape, dh=10.0, dt=dt, nt=nt, abcn=abcn, device="cpu",
        impl="eager", eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
        source_type=["uz"], receiver_type=["ux", "uz"],
    )


def _cpu_geometry(shape, nt, dt):
    nz, nx = shape
    wavelet = (1e3 * _ricker(np.arange(nt) * dt - 0.08, 12.0)).astype(np.float32)
    src = np.array([[nx // 2, nz // 3]], dtype=np.int64)
    rec_x = np.arange(6, nx - 6, 4, dtype=np.int64)
    rec = np.stack([rec_x, np.full_like(rec_x, nz // 4)], axis=1)[None]
    return wavelet, src, rec


def _const_models(shape, vh, vs, rho, eps, eta, theta, req=False):
    vals = [vh, vs, rho, eps, eta, theta]
    return [torch.full(shape, float(v), requires_grad=req) for v in vals]


def test_eager_iso_rotation_invariance():
    """eps = eta = 0 makes the medium isotropic; the Bond rotation must then
    be the identity for any tilt (C15/C35 vanish identically)."""
    shape, nt, dt = (48, 56), 300, 1e-3
    wavelet, src, rec = _cpu_geometry(shape, nt, dt)
    with torch.no_grad():
        r0 = _cpu_prop(shape, nt, dt)(wavelet, src, rec,
                                      models=_const_models(shape, 2400, 1300, 2000, 0, 0, 0.0))
        r1 = _cpu_prop(shape, nt, dt)(wavelet, src, rec,
                                      models=_const_models(shape, 2400, 1300, 2000, 0, 0, 0.7))
    rel = float((r0 - r1).abs().max() / r0.abs().max())
    assert rel < 5e-5, f"isotropic rotation invariance broken: rel={rel:.3e}"


def test_eager_long_run_stable_and_absorbed():
    """Short-record sanity: the wavetrain leaves and the record decays.

    0.8 s is nowhere near long enough to see the CPML instability this
    equation would otherwise have — that one only passes the direct arrival
    at ~9.6 s. See ``test_mpml_long_record_stays_bounded`` for the test that
    actually pins it.
    """
    shape, nt, dt = (48, 56), 800, 1e-3
    wavelet, src, rec = _cpu_geometry(shape, nt, dt)
    with torch.no_grad():
        r = _cpu_prop(shape, nt, dt)(wavelet, src, rec,
                                     models=_const_models(shape, 2400, 1300, 2000, 0.15, 0.08, 0.4))
    assert torch.isfinite(r).all()
    peak = r.abs().max().item()
    tail = r[:, -nt // 8:].abs().max().item()
    assert tail < 1e-2 * peak, f"late-time energy not decaying: tail/peak={tail/peak:.3e}"


def test_eager_all_six_grads_finite():
    shape, nt, dt = (40, 44), 100, 1e-3
    wavelet, src, rec = _cpu_geometry(shape, nt, dt)
    models = _const_models(shape, 2400, 1300, 2000, 0.12, 0.06, 0.35, req=True)
    record = _cpu_prop(shape, nt, dt)(wavelet, src, rec, models=models)
    assert torch.isfinite(record).all()
    grads = torch.autograd.grad(record.pow(2).mean(), models)
    for name, g in zip(MODEL_NAMES, grads):
        assert torch.isfinite(g).all(), f"grad {name} non-finite"
        assert float(g.norm()) > 0.0, f"grad {name} identically zero"


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        import sweep._C as _C  # noqa: F401
        return hasattr(_C, "elastic_tti_2nd2d_forward")
    except Exception:
        return False


cuda_mark = pytest.mark.skipif(
    not _binding_ready(),
    reason="CUDA + compiled sweep._C with elastic_tti_2nd2d kernels required",
)


def _canonical_setup():
    nz, nx = 48, 56
    dh, dt, nt = 10.0, 1e-3, 300
    abcn = 30
    spatial_order = 4
    shape = (nz, nx)

    src = np.array([[nx // 2, nz // 3]], dtype=np.int64)[None]
    rec_x = np.arange(4, nx - 4, 3, dtype=np.int64)
    rec = np.stack([rec_x, np.full_like(rec_x, nz // 4)], axis=1)[None]

    t = np.arange(nt, dtype=np.float32) * dt - 0.08
    wavelet = torch.tensor((1e3 * _ricker(t, 12.0)).astype(np.float32)).cuda()

    return {
        "shape": shape, "nz": nz, "nx": nx,
        "dh": dh, "dt": dt, "nt": nt, "abcn": abcn,
        "spatial_order": spatial_order,
        "src": src, "rec": rec, "wavelet": wavelet,
    }


def _canonical_models(cfg, req_grad=False, true_variant=False, device="cuda"):
    nz, nx = cfg["nz"], cfg["nx"]
    shape = cfg["shape"]

    def _ramp(top, bottom):
        depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)
        return np.broadcast_to((top + (bottom - top) * depth)[:, None], shape).copy()

    def _box(arr, val):
        out = arr.copy()
        out[nz // 3: (2 * nz) // 3, nx // 4: (3 * nx) // 4] += val
        return out

    vh = _ramp(2000.0, 2600.0)
    vs = vh / 1.9
    rho = _ramp(1000.0, 1200.0)
    # Non-uniform on purpose: a constant Thomsen/angle field makes an
    # anisotropic adjoint test blind to every term that needs structure in the
    # stiffness field (lesson_acoustic_vti_1st_c_grad_defect).
    def _tilt(base, dz, dx):
        z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
        x = np.linspace(-0.5, 0.5, nx, dtype=np.float32)[None, :]
        return np.broadcast_to(base + dz * z + dx * x, shape).astype(np.float32).copy()

    eps = _tilt(0.12, 0.06, 0.03)
    eta = _tilt(0.06, 0.04, 0.02)
    the = _tilt(0.35, 0.18, 0.12)

    if true_variant:
        vh = _box(vh, 200.0)
        vs = _box(vs, 90.0)
        rho = _box(rho, 60.0)
        eps = _box(eps, 0.03)
        eta = _box(eta, 0.02)
        the = _box(the, 0.1)

    return [
        torch.tensor(a, dtype=torch.float32, device=device, requires_grad=req_grad)
        for a in (vh, vs, rho, eps, eta, the)
    ]


def _build_prop(cfg, impl, mode="full", device="cuda"):
    from sweep.propagator.options import (
        CUDAOptions, MemoryOptions, BoundaryOptions, CkptOptions,
    )

    eq = ElasticTTI2nd(spatial_order=cfg["spatial_order"], device=device,
                       backend="torch")
    kwargs = dict(
        source_type=["uz"], receiver_type=["ux", "uz"],
        abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], nt=cfg["nt"],
        device=device, impl=impl,
    )
    if impl == "eager":
        return PropTorch(eq, cfg["shape"], use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False), **kwargs)
    if mode == "full":
        return PropTorch(eq, cfg["shape"], use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **kwargs)
    if mode == "bs":
        return PropTorch(eq, cfg["shape"], cuda_options=CUDAOptions(
            memory=MemoryOptions(strategy="boundary",
                                 boundary=BoundaryOptions(storage="gpu"))), **kwargs)
    if mode == "ckpt":
        return PropTorch(eq, cfg["shape"], cuda_options=CUDAOptions(
            memory=MemoryOptions(strategy="ckpt",
                                 ckpt=CkptOptions(mode="chunk", chunks=10))), **kwargs)
    raise ValueError(mode)


def _observed(cfg):
    prop = _build_prop(cfg, "c", "full")
    with torch.no_grad():
        return prop(cfg["wavelet"], cfg["src"], cfg["rec"],
                    models=_canonical_models(cfg, true_variant=True)).detach()


def _run_grads(cfg, impl, mode, observed):
    device = "cpu" if impl == "eager" else "cuda"
    prop = _build_prop(cfg, impl, mode, device=device)
    models = _canonical_models(cfg, req_grad=True, device=device)
    record = prop(cfg["wavelet"].to(device), cfg["src"], cfg["rec"], models=models)
    loss = (record - observed.to(device)).pow(2).mean()
    grads = torch.autograd.grad(loss, models)
    return [g.detach().double().cpu() for g in grads]


def _metrics(g_ref, g_test):
    a = g_ref.flatten()
    b = g_test.flatten()
    ref_l2 = float(a.norm())
    rel_l2 = float((a - b).norm()) / max(ref_l2, 1e-30)
    cos = float(torch.dot(a, b) / max(float(a.norm() * b.norm()), 1e-30))
    return rel_l2, cos, ref_l2


@cuda_mark
def test_cuda_forward_matches_eager():
    cfg = _canonical_setup()

    def _run(impl):
        device = "cpu" if impl == "eager" else "cuda"
        prop = _build_prop(cfg, impl, device=device)
        with torch.no_grad():
            return prop(cfg["wavelet"].to(device), cfg["src"], cfg["rec"],
                        models=_canonical_models(cfg, device=device)).detach().cpu()

    rec_eager = _run("eager")
    rec_cuda = _run("c")
    assert rec_eager.shape == rec_cuda.shape
    max_diff = float((rec_eager - rec_cuda).abs().max())
    ref = float(rec_eager.abs().max())
    rel = max_diff / max(ref, 1e-30)
    assert rel < 1e-4, f"CUDA vs eager forward divergence: rel={rel:.3e}"


@cuda_mark
def test_cuda_backward_full_matches_eager_all_params():
    cfg = _canonical_setup()
    observed = _observed(cfg)

    grads_eager = _run_grads(cfg, "eager", "full", observed)
    grads_cuda = _run_grads(cfg, "c", "full", observed)

    failures = []
    for name, ge, gc in zip(MODEL_NAMES, grads_eager, grads_cuda):
        rel_l2, cos, ref_l2 = _metrics(ge, gc)
        print(f"[full-vs-eager] {name:8s} rel_l2={rel_l2:.4e} cos={cos:.6f} "
              f"ref_l2={ref_l2:.4e}")
        assert ref_l2 > 0, f"eager grad {name} is identically zero"
        if not (rel_l2 < 1.5 and cos > 0.8):
            failures.append((name, rel_l2, cos))
    assert not failures, f"gradient mismatch: {failures}"


@cuda_mark
@pytest.mark.parametrize("mode", ["bs", "ckpt"])
def test_cuda_backward_mode_matches_full(mode):
    cfg = _canonical_setup()
    observed = _observed(cfg)

    grads_full = _run_grads(cfg, "c", "full", observed)
    grads_mode = _run_grads(cfg, "c", mode, observed)

    failures = []
    for name, gf, gm in zip(MODEL_NAMES, grads_full, grads_mode):
        rel_l2, cos, ref_l2 = _metrics(gf, gm)
        print(f"[{mode}-vs-full] {name:8s} rel_l2={rel_l2:.4e} cos={cos:.6f} "
              f"ref_l2={ref_l2:.4e}")
        assert ref_l2 > 0, f"full grad {name} is identically zero"
        # bs drops the it=0 rho imaging term (family behaviour; ckpt covers
        # it=0 and must stay tight).
        if name == "rho" and mode == "bs":
            ok = rel_l2 < 5e-2 and cos > 0.999
        else:
            ok = rel_l2 < 1e-3 and cos > 0.99999
        if not ok:
            failures.append((name, rel_l2, cos))
    assert not failures, f"{mode} gradient mismatch vs full: {failures}"


@cuda_mark
def test_fwi_smoke_reduces_misfit():
    """Forward + inversion round trip on the compiled backend: 30 Adam steps
    on (vh, vs, epsilon, theta) from the smooth initial model must cut the
    misfit against the boxed true model by well over half."""
    cfg = _canonical_setup()
    observed = _observed(cfg)

    prop = _build_prop(cfg, "c", "bs")
    models = _canonical_models(cfg, req_grad=True)
    # Per-parameter Adam steps sized to each parameter's physical scale;
    # rho/eta stay frozen so the smoke matches the paper's focused strategy.
    lrs = [8.0, 4.0, 0.0, 4e-3, 0.0, 1e-2]  # vh, vs, rho, eps, eta, theta
    groups = [{"params": [m], "lr": lr} for m, lr in zip(models, lrs) if lr > 0]
    opt = torch.optim.Adam(groups)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        record = prop(cfg["wavelet"], cfg["src"], cfg["rec"], models=models)
        loss = (record - observed).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    print(f"[fwi-smoke] loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e} "
          f"ratio={losses[-1]/losses[0]:.4f}")
    assert losses[-1] < 0.5 * losses[0], (
        f"FWI smoke failed to reduce misfit: {losses[0]:.3e} -> {losses[-1]:.3e}"
    )


# ---------------------------------------------------------------------------
# Multiaxial PML — the absorbing layer that keeps this equation stable
# ---------------------------------------------------------------------------

def test_mpml_profiles_contract():
    """The profile transform is the whole fix: ``step`` never sees it."""
    from sweep.equations.elastic_tti_2nd import DEFAULT_MPML_RATIO, mpml_profiles

    nz, nx, dt, width = 40, 56, 1.2e-3, 10

    def prof(n):
        i = np.arange(n)
        frac = np.clip(np.maximum((width - i) / width, (i - (n - 1 - width)) / width), 0, 1)
        sigma = 300.0 * frac ** 2
        alpha = np.pi * 25.0 * (1.0 - frac)
        a = np.exp(-(sigma + alpha) * dt).astype(np.float32)
        a[frac == 0] = 0.0
        b = ((sigma / np.maximum(sigma + alpha, 1e-9)) * (a - 1.0)).astype(np.float32)
        return a, b

    az, bz = prof(nz)
    ax, bx = prof(nx)
    src = [az[None, :, None], bz[None, :, None], az[None, :, None], bz[None, :, None],
           ax[None, None, :], bx[None, None, :], ax[None, None, :], bx[None, None, :]]

    plain = mpml_profiles(src, dt, 0.0)
    mixed = mpml_profiles(src, dt, DEFAULT_MPML_RATIO)
    for got in (plain, mixed):
        assert len(got) == 8
        for t in got:
            assert t.shape == (1, nz, nx), "profiles must come back as 2-D fields"

    # ratio=0 is the plain CPML broadcast, bit for bit -- the compiled kernel
    # has a single index path, so this must not perturb anything.
    assert np.array_equal(plain[0][0], np.broadcast_to(az[:, None], (nz, nx)))
    assert np.array_equal(plain[4][0], np.broadcast_to(ax[None, :], (nz, nx)))

    # The mix damps x where only z is inside the layer (and vice versa): that
    # region is exactly where the plain layer has no x-damping at all.
    z_band, x_interior = 2, nx // 2
    assert plain[4][0, z_band, x_interior] == 0.0
    assert mixed[4][0, z_band, x_interior] != 0.0, "multiaxial mix did not reach the z band"


@cuda_mark
@pytest.mark.parametrize("impl", ["eager", "c"])
def test_mpml_long_record_stays_bounded(impl):
    """12 s on a uniform TTI medium, both backends.

    Plain CPML is unstable for this second-order displacement system — a
    documented limitation (Li & Bou Matar 2010), not a defect of the
    discretisation — and grows out of the PML corners at +34 dB/s. The second
    half of this test keeps ``mpml_ratio=0`` in the picture on purpose: it
    pins the mechanism, so if the underlying CPML ever gains a real fix this
    assertion is the one that says so.
    """
    from sweep.propagator.options import CUDAOptions, MemoryOptions

    n, dh, dt, abcn, freq = 120, 10.0, 1.2e-3, 25, 8.0
    nt = int(round(12.0 / dt))
    eps, eta = 0.20, (0.20 - 0.05) / (1 + 2 * 0.05)

    def _run(ratio):
        eq = ElasticTTI2nd(spatial_order=4, device="cuda", backend="torch",
                           mpml_ratio=ratio)
        kwargs = dict(shape=(n, n), dh=dh, dt=dt, nt=nt, abcn=abcn, dev="cuda",
                      source_type=["uz"], receiver_type=["ux", "uz"],
                      pml_type=eq.default_pml_type)
        prop = (PropTorch(eq, impl="eager", use_ckpt=False,
                          eager_options=EagerOptions(use_compile=False), **kwargs)
                if impl == "eager" else
                PropTorch(eq, impl="c", cuda_options=CUDAOptions(
                    memory=MemoryOptions(strategy="full")), **kwargs))
        full = lambda v: torch.full((n, n), float(v), dtype=torch.float32, device="cuda")
        models = [full(2400.0 * np.sqrt(1 + 2 * eps)), full(1200.0), full(2200.0),
                  full(eps), full(eta), full(np.deg2rad(30.0))]
        src = np.array([[[n // 2, n // 2]]], dtype=np.int64)
        rx = np.arange(15, n - 15, 3, dtype=np.int64)
        rec = np.stack([rx, np.full_like(rx, 20)], axis=1)[None]
        t = np.arange(nt, dtype=np.float32) * dt - 1.2 / freq
        wavelet = (1e3 * _ricker(t, freq)).astype(np.float32)
        with torch.no_grad():
            out = prop(wavelet, src, rec, models=models)
        env = out.abs().amax(dim=(0, 2, 3)).cpu().numpy()
        return out, env, float(env[:int(2.0 / dt)].max())

    out, env, direct = _run(ElasticTTI2nd(spatial_order=4).mpml_ratio)
    late = float(env[int(2.0 / dt):].max())
    print(f"[mpml-{impl}] direct={direct:.3e} late={late:.3e} "
          f"({20 * np.log10(max(late, 1e-38) / direct):.1f} dB)")
    assert torch.isfinite(out).all(), "M-PML record went non-finite"
    assert late < direct, (
        f"late-time energy passed the direct arrival: {late:.3e} vs {direct:.3e}")

    _, env0, direct0 = _run(0.0)
    late0 = float(env0[int(2.0 / dt):].max())
    assert late0 > direct0, (
        "plain CPML no longer blows up here — if that is a real fix, retire "
        "the mpml_ratio default; if it is just this grid, make the test harsher")
