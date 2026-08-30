"""3-D elastic TTI (axis-aligned SG): eager reference checks + CUDA consistency.

CPU part: export contract, iso-parameter equivalence against Elastic3D, and a
short eager run whose gradients must be finite and nonzero for ALL eight raw
parameters (vp0, vs0, rho, epsilon, delta, gamma, theta, phi).

CUDA part (canonical 3-D suite grid nz=24, ny=20, nx=24, dh=10 m, dt=1.5 ms,
nt=120, abcn=30, order 4): forward c-vs-eager, full backward c-vs-eager on all
eight parameter gradients, and bs / ckpt backward against full (same backend,
tight thresholds).  theta / phi backgrounds are nonzero so the Bond-rotation
autograd path is exercised away from its |theta|<1e-7 VTI fallback branch.
"""

import functools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.equations import ElasticTTISG3D, Elastic3D, supports_torch_binding
from sweep.equations.elastic_tti_sg3d import STIFFNESS_KEYS_3D
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

MODEL_NAMES = ["vp0", "vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"]


def _ricker(t, f):
    arg = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


# ---------------------------------------------------------------------------
# CPU: contract + eager reference behaviour
# ---------------------------------------------------------------------------

def test_export_contract():
    eq = ElasticTTISG3D(spatial_order=4)
    assert [m.name for m in eq.MODEL_SPECS] == MODEL_NAMES
    assert len(eq.FIELD_SPECS) == 27
    assert len(STIFFNESS_KEYS_3D) == 21
    assert eq.prepare_models_for_c is True
    assert eq.default_pml_type == "cpmls"
    assert supports_torch_binding("ElasticTTISG3D") is True


def test_free_surface_raises():
    eq = ElasticTTISG3D(spatial_order=4, device="cpu", backend="torch")
    with pytest.raises(NotImplementedError, match="anisotropic"):
        PropTorch(
            eq, (16, 12, 16), dh=10.0, dt=1e-3, nt=8, abcn=4,
            free_surface=True, device="cpu", impl="eager",
            eager_options=EagerOptions(use_compile=False), use_ckpt=False,
        )


def _cpu_prop(eq_cls, shape, nt, dt):
    eq = eq_cls(spatial_order=4, device="cpu", backend="torch")
    return PropTorch(
        eq, shape, dh=10.0, dt=dt, nt=nt, abcn=10, device="cpu",
        impl="eager", eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["vx", "vy", "vz"],
    )


def _cpu_geometry(shape, nt, dt):
    nz, ny, nx = shape
    t = np.arange(nt, dtype=np.float32) * dt
    wavelet = (1e6 * _ricker(t - 0.06, 10.0)).astype(np.float32)
    src = np.array([[nx // 2, ny // 2, nz // 4]], dtype=np.int64)
    rec_x = np.arange(4, nx - 4, 3, dtype=np.int64)
    rec = np.stack([rec_x, np.full_like(rec_x, ny // 2),
                    np.full_like(rec_x, nz // 3)], axis=1)[None]
    return wavelet, src, rec


def test_eager_iso_params_match_elastic3d():
    """With epsilon=delta=gamma=theta=phi=0 the 21-entry constitutive update
    collapses to lambda/mu and must reproduce isotropic Elastic3D up to fp32
    round-off (different summation order of the same terms)."""
    shape, nt, dt = (24, 20, 24), 60, 1.5e-3
    wavelet, src, rec = _cpu_geometry(shape, nt, dt)

    vp = torch.full(shape, 2400.0)
    vs = torch.full(shape, 1200.0)
    rho = torch.full(shape, 2000.0)
    zeros = torch.zeros(shape)

    with torch.no_grad():
        rec_iso = _cpu_prop(Elastic3D, shape, nt, dt)(
            wavelet, src, rec, models=[vp, vs, rho])
        rec_tti = _cpu_prop(ElasticTTISG3D, shape, nt, dt)(
            wavelet, src, rec,
            models=[vp, vs, rho] + [zeros.clone() for _ in range(5)])

    max_diff = float((rec_iso - rec_tti).abs().max())
    ref = float(rec_iso.abs().max())
    assert ref > 0
    assert max_diff / ref < 5e-6, f"iso-equivalence broken: rel={max_diff / ref:.3e}"


def test_eager_forward_and_all_eight_grads_finite():
    shape, nt, dt = (20, 16, 20), 24, 1e-3
    wavelet, src, rec = _cpu_geometry(shape, nt, dt)

    models = [
        torch.full(shape, 2400.0, requires_grad=True),
        torch.full(shape, 1200.0, requires_grad=True),
        torch.full(shape, 2000.0, requires_grad=True),
        torch.full(shape, 0.12, requires_grad=True),
        torch.full(shape, 0.06, requires_grad=True),
        torch.full(shape, 0.05, requires_grad=True),
        torch.full(shape, 0.3, requires_grad=True),
        torch.full(shape, 0.2, requires_grad=True),
    ]
    record = _cpu_prop(ElasticTTISG3D, shape, nt, dt)(wavelet, src, rec, models=models)
    assert torch.isfinite(record).all()
    loss = record.pow(2).mean()
    grads = torch.autograd.grad(loss, models)
    for name, g in zip(MODEL_NAMES, grads):
        assert torch.isfinite(g).all(), f"grad {name} non-finite"
        assert float(g.norm()) > 0.0, f"grad {name} identically zero"


# ---------------------------------------------------------------------------
# CUDA: canonical 3-D consistency suite
# ---------------------------------------------------------------------------

def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        import sweep._C as _C  # noqa: F401
        return hasattr(_C, "elastic_tti_sg3d_forward")
    except Exception:
        return False


cuda_mark = pytest.mark.skipif(
    not _binding_ready(),
    reason="CUDA + compiled sweep._C with elastic_tti_sg3d kernels required",
)


def _canonical_3d_setup():
    nz, ny, nx = 24, 20, 24
    dh, dt, nt = 10.0, 1.5e-3, 120
    abcn = 30
    spatial_order = 4
    shape = (nz, ny, nx)

    src = np.array([[nx // 2, ny // 2, nz // 4]], dtype=np.int64)[None]
    rec_x = np.arange(2, nx - 2, 6, dtype=np.int64)
    rec_y = np.arange(2, ny - 2, 4, dtype=np.int64)
    rx, ry = np.meshgrid(rec_x, rec_y, indexing="xy")
    rec = np.stack([rx.ravel(), ry.ravel(),
                    np.full(rx.size, 2, dtype=np.int64)], axis=-1)[None]

    t = np.arange(nt, dtype=np.float32) * dt - 0.06
    wavelet = torch.tensor((1e6 * _ricker(t, 10.0)).astype(np.float32)).cuda()

    return {
        "shape": shape, "nz": nz, "ny": ny, "nx": nx,
        "dh": dh, "dt": dt, "nt": nt, "abcn": abcn,
        "spatial_order": spatial_order,
        "src": src, "rec": rec, "wavelet": wavelet,
    }


def _canonical_models(cfg, req_grad=False, true_variant=False, device="cuda"):
    """Depth ramps + box anomalies on every parameter; nonzero theta/phi
    backgrounds keep the Bond rotation away from its VTI fallback branch."""
    nz, ny, nx = cfg["nz"], cfg["ny"], cfg["nx"]
    shape = cfg["shape"]

    def _ramp(top, bottom):
        depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)
        col = (top + (bottom - top) * depth).astype(np.float32)
        return np.broadcast_to(col[:, None, None], shape).copy()

    def _box(arr, val):
        out = arr.copy()
        out[nz // 3: (2 * nz) // 3,
            ny // 4: (3 * ny) // 4,
            nx // 4: (3 * nx) // 4] += val
        return out

    def _tilt(base, dz, dx):
        """Depth ramp plus a lateral gradient, so theta/phi vary along the
        Bond rotation's own axes rather than only with depth."""
        z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
        x = np.linspace(-0.5, 0.5, nx, dtype=np.float32)[None, None, :]
        return np.broadcast_to(base + dz * z + dx * x, shape).astype(np.float32).copy()

    vp = _ramp(1800.0, 2400.0)
    vs = vp / 1.73
    rho = _ramp(1000.0, 1200.0)
    # Every anisotropy parameter varies in space.  A constant Thomsen/angle
    # field is the classic way an anisotropic adjoint test goes blind (see
    # lesson_acoustic_vti_1st_c_grad_defect): terms that only survive where the
    # stiffness field has structure drop out of the comparison entirely.
    eps = _tilt(0.08, 0.05, 0.02)
    delta = _tilt(0.04, 0.03, 0.015)
    gam = _tilt(0.05, 0.03, 0.02)
    the = _tilt(0.30, 0.15, 0.10)
    phi = _tilt(0.20, 0.12, 0.08)

    if true_variant:
        vp = _box(vp, 180.0)
        vs = _box(vs, 90.0)
        rho = _box(rho, 60.0)
        eps = _box(eps, 0.03)
        delta = _box(delta, 0.02)
        gam = _box(gam, 0.02)
        the = _box(the, 0.1)
        phi = _box(phi, 0.08)

    return [
        torch.tensor(a, dtype=torch.float32, device=device, requires_grad=req_grad)
        for a in (vp, vs, rho, eps, delta, gam, the, phi)
    ]


def _build_prop(cfg, impl, mode="full", device="cuda",
                src_t=("sxx", "syy", "szz"), rec_t=("vx", "vy", "vz")):
    from sweep.propagator.options import (
        CUDAOptions, MemoryOptions, BoundaryOptions, CkptOptions,
    )

    eq = ElasticTTISG3D(spatial_order=cfg["spatial_order"], device=device,
                        backend="torch")
    kwargs = dict(
        source_type=list(src_t),
        receiver_type=list(rec_t),
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
                                 ckpt=CkptOptions(mode="chunk", chunks=8))), **kwargs)
    raise ValueError(mode)


def _run_grads(cfg, impl, mode, observed, src_t=("sxx", "syy", "szz"),
               rec_t=("vx", "vy", "vz")):
    # The eager autograd reference keeps every intermediate of 120 3-D steps
    # alive (~50 GB on the padded canonical grid) — run it on host RAM; the
    # compiled path stays on CUDA.  fp32 CPU-vs-GPU round-off is far below
    # the comparison thresholds.
    device = "cpu" if impl == "eager" else "cuda"
    prop = _build_prop(cfg, impl, mode, device=device, src_t=src_t, rec_t=rec_t)
    models = _canonical_models(cfg, req_grad=True, device=device)
    wavelet = cfg["wavelet"].to(device)
    record = prop(wavelet, cfg["src"], cfg["rec"], models=models)
    loss = (record - observed.to(device)).pow(2).mean()
    grads = torch.autograd.grad(loss, models)
    return [g.detach().double().cpu() for g in grads]


def _observed(cfg, src_t=("sxx", "syy", "szz"), rec_t=("vx", "vy", "vz")):
    prop = _build_prop(cfg, "c", "full", src_t=src_t, rec_t=rec_t)
    with torch.no_grad():
        return prop(cfg["wavelet"], cfg["src"], cfg["rec"],
                    models=_canonical_models(cfg, true_variant=True)).detach()


def _metrics(g_ref, g_test):
    a = g_ref.flatten()
    b = g_test.flatten()
    ref_l2 = float(a.norm())
    rel_l2 = float((a - b).norm()) / max(ref_l2, 1e-30)
    cos = float(torch.dot(a, b) / max(float(a.norm() * b.norm()), 1e-30))
    return rel_l2, cos, ref_l2


@cuda_mark
def test_cuda_forward_matches_eager():
    cfg = _canonical_3d_setup()

    def _run(impl):
        prop = _build_prop(cfg, impl)
        with torch.no_grad():
            return prop(cfg["wavelet"], cfg["src"], cfg["rec"],
                        models=_canonical_models(cfg)).detach().cpu()

    rec_eager = _run("eager")
    rec_cuda = _run("c")
    assert rec_eager.shape == rec_cuda.shape
    max_diff = float((rec_eager - rec_cuda).abs().max())
    ref = float(rec_eager.abs().max())
    rel = max_diff / max(ref, 1e-30)
    assert rel < 1e-4, f"CUDA vs eager forward divergence: rel={rel:.3e}"


# Source / receiver combinations that exercise the three rho / sign paths of
# the adjoint.  A velocity-receiver-only, stress-source-only test passes even
# with the receiver-cell rho correction and the stress-receiver sign flip both
# missing, which is exactly how those two defects survived here after the
# equation was forked from the pre-PR#62 elastic3d template.
SRC_REC_CASES = [
    (("sxx", "syy", "szz"), ("vx", "vy", "vz")),   # stress source, velocity receivers
    (("sxx", "syy", "szz"), ("sxx", "syy", "szz")),  # stress receivers -> signed adjoint
    (("vz",), ("vx", "vy", "vz")),                 # body force -> source-cell rho term
]


@cuda_mark
@pytest.mark.parametrize("src_t,rec_t", SRC_REC_CASES,
                         ids=lambda v: "".join(x[0] for x in v))
def test_cuda_backward_full_matches_eager_all_params(src_t, rec_t):
    """Every one of the eight raw parameter gradients (velocities, density,
    Thomsen constants AND both angles) must agree between the CUDA adjoint
    and the eager autograd reference, for every source/receiver kind."""
    cfg = _canonical_3d_setup()
    observed = _observed(cfg, src_t, rec_t)

    grads_eager = _run_grads(cfg, "eager", "full", observed, src_t, rec_t)
    grads_cuda = _run_grads(cfg, "c", "full", observed, src_t, rec_t)

    failures = []
    for name, ge, gc in zip(MODEL_NAMES, grads_eager, grads_cuda):
        rel_l2, cos, ref_l2 = _metrics(ge, gc)
        print(f"[full-vs-eager {src_t[0]}/{rec_t[0]}] {name:8s} rel_l2={rel_l2:.4e} "
              f"cos={cos:.6f} ref_l2={ref_l2:.4e}")
        assert ref_l2 > 0, f"eager grad {name} is identically zero"
        # Measured worst case over the three cases is ~2e-5 / cos 1.000000.
        # Keep real margin but stay far tighter than the defects this pins:
        # missing receiver-cell rho correction was rel 0.25 / cos 0.968, and a
        # raw stress-receiver injection is rel 2.0 / cos -1.
        if not (rel_l2 < 2e-3 and cos > 0.9999):
            failures.append((name, rel_l2, cos))
    assert not failures, f"gradient mismatch: {failures}"


@cuda_mark
@pytest.mark.parametrize("mode", ["bs", "ckpt"])
def test_cuda_backward_mode_matches_full(mode):
    """bs / ckpt run the same CUDA kernels as full with a reconstructed (bs)
    or replayed (ckpt) forward field, so their gradients must match the full
    store-all gradients tightly."""
    cfg = _canonical_3d_setup()
    observed = _observed(cfg)

    grads_full = _run_grads(cfg, "c", "full", observed)
    grads_mode = _run_grads(cfg, "c", mode, observed)

    failures = []
    for name, gf, gm in zip(MODEL_NAMES, grads_full, grads_mode):
        rel_l2, cos, ref_l2 = _metrics(gf, gm)
        print(f"[{mode}-vs-full] {name:8s} rel_l2={rel_l2:.4e} cos={cos:.6f} "
              f"ref_l2={ref_l2:.4e}")
        assert ref_l2 > 0, f"full grad {name} is identically zero"
        # The bs reverse loop runs it >= 1 (family behaviour shared with
        # elastic3d / elastic_tti_sg2d), dropping the it=0 rho imaging term
        # a.v(0)*(v[0]-v[1])/rho — nonzero because v[1] != 0.  The stiffness
        # gradients are unaffected (v[0] == 0 makes their it=0 term vanish),
        # and ckpt covers it=0 so it stays bit-exact against full.
        if name == "rho" and mode == "bs":
            ok = rel_l2 < 5e-2 and cos > 0.9995
        else:
            ok = rel_l2 < 1e-3 and cos > 0.999999
        if not ok:
            failures.append((name, rel_l2, cos))
    assert not failures, f"{mode} gradient mismatch vs full: {failures}"


# ---------------------------------------------------------------------------
# Physics: check against the Christoffel equation, not against ourselves
# ---------------------------------------------------------------------------
# Every other test in this file compares one implementation against another, and
# both consume the SAME 21 stiffnesses out of prepare_models -- so a wrong Bond
# rotation, Voigt index or angle convention passes all of them. These compare
# against an analytic reference written independently of sweep.

# ``test`` is a package here, so pytest puts the repo root on sys.path rather
# than this directory; add it so the reference module imports by plain name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

CHRISTOFFEL_MEDIUM = dict(vp0=2400.0, vs0=1200.0, rho=2200.0,
                          epsilon=0.30, delta=0.05, gamma=0.15)


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (25.0, 0.0), (40.0, 45.0),
                                       (90.0, 30.0), (-35.0, 200.0)],
                         ids=lambda v: f"{v:g}")
def test_bond_rotation_matches_independent_tensor_rotation(theta, phi):
    """prepare_models' 21 stiffnesses must equal a rank-4 rotation of the VTI
    tensor done independently.

    The solver builds them with a 6x6 Voigt-space Bond matrix (cheap, but easy to
    get wrong in a way nothing else here would notice). The reference applies
    R R R R : C directly, from the documented meaning of theta and phi, so this
    pins the rotation, the Voigt index map and the angle convention at once.
    """
    from christoffel_reference import rotation, vti_stiffness, vti_tensor, _VOIGT

    m = CHRISTOFFEL_MEDIUM
    eq = ElasticTTISG3D(spatial_order=4, device="cpu")
    tiny = lambda v: torch.full((1,), float(v), dtype=torch.float64)
    prepared = eq.prepare_models(
        [tiny(m["vp0"]), tiny(m["vs0"]), tiny(m["rho"]), tiny(m["epsilon"]),
         tiny(m["delta"]), tiny(m["gamma"]),
         tiny(np.deg2rad(theta)), tiny(np.deg2rad(phi))])
    got = {k: float(v.reshape(-1)[0]) for k, v in
           zip(STIFFNESS_KEYS_3D, prepared[1:])}

    c = vti_tensor(*vti_stiffness(m["vp0"], m["vs0"], m["rho"],
                                  m["epsilon"], m["delta"], m["gamma"]))
    R = rotation(np.deg2rad(theta), np.deg2rad(phi))
    c_rot = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, c)
    ref6 = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    ref6[_VOIGT[i, j], _VOIGT[k, l]] = c_rot[i, j, k, l]

    scale = max(abs(v) for v in got.values())
    bad = []
    for key, value in got.items():
        a, b = (int(key[1]) - 1, int(key[2]) - 1)
        ref = ref6[a, b]
        if abs(value - ref) > 1e-9 * scale:
            bad.append(f"{key}: got {value:.6e} ref {ref:.6e}")
    assert not bad, "Bond rotation disagrees with R R R R : C\n" + "\n".join(bad)


# The wavefront comparison below needs the front to be several wavelengths clear
# of the source, which means a grid far larger than the rest of this file uses
# (~1 min per case). It is the end-to-end check -- rotation AND stencil AND
# source -- so it is kept, but opt-in: SWEEP_TEST_WAVEFRONT=1.
WAVEFRONT_GRID = dict(n=220, dh=10.0, dt=1.2e-3, abcn=16, order=4,
                      freq=8.0, delay=0.12, it_snap=292)

wavefront_mark = pytest.mark.skipif(
    not os.environ.get("SWEEP_TEST_WAVEFRONT"),
    reason="slow end-to-end wavefront check; set SWEEP_TEST_WAVEFRONT=1")


def _qp_snapshot(theta_deg, phi_deg, medium=None):
    """Stress trace of a homogeneous TTI medium at one snapshot, x-z plane."""
    g = WAVEFRONT_GRID
    med = dict(CHRISTOFFEL_MEDIUM, **(medium or {}))
    n, dev = g["n"], "cuda"
    shape = (n, n, n)
    eq = ElasticTTISG3D(spatial_order=g["order"], device=dev)
    solver = PropTorch(eq, shape=shape, dh=g["dh"], dt=g["dt"], dev=dev,
                       nt=g["it_snap"] + 1, abcn=g["abcn"],
                       source_type=["sxx", "syy", "szz"], receiver_type=["vz"],
                       impl="eager", use_ckpt=False,
                       eager_options=EagerOptions(use_compile=False))
    c = n // 2
    src = np.array([[[c, c, c]]], dtype=np.int64)          # (x, y, z)
    t = np.arange(g["it_snap"] + 1, dtype=np.float32) * g["dt"] - g["delay"]
    wav = (1e3 * _ricker(t, g["freq"])).astype(np.float32)

    # The per-cell Bond rotation would materialise (..., 6, 6) fields on the
    # padded grid; the medium is uniform, so rotate once and broadcast.
    tiny = lambda v: torch.full((1, 1, 1), float(v), dtype=torch.float32, device=dev)
    raw = [tiny(med["vp0"]), tiny(med["vs0"]), tiny(med["rho"]),
           tiny(med["epsilon"]), tiny(med["delta"]), tiny(med["gamma"]),
           tiny(np.deg2rad(theta_deg)), tiny(np.deg2rad(phi_deg))]
    models = [m.expand(shape) for m in eq.prepare_models(raw)]
    eq.prepare_models = lambda m: m

    with torch.no_grad():
        _, snaps = solver(wav, src, src, models=models, return_wavefield=True,
                          snapshot_times=[g["it_snap"]])
    a = g["abcn"]
    idx = [eq.wavefields.index(k) for k in ("sxx", "syy", "szz")]
    trace = sum(snaps[0, i, 0, 0] for i in idx)
    cube = trace[a:a + n, a:a + n, a:a + n].cpu().numpy()
    del snaps
    torch.cuda.empty_cache()
    return cube[:, n // 2, :]                              # slice through source


@functools.lru_cache(maxsize=1)
def _isotropic_floor(psi, travel, r_max):
    """Shape rms of the same measurement on an isotropic medium: whatever it
    reports is the noise of the picker on this grid, not an anisotropy error.
    Cached so the parametrised cases do not re-simulate it."""
    from christoffel_reference import pick_front, qp_radius, shape_error

    g = WAVEFRONT_GRID
    flat = dict(epsilon=0.0, delta=0.0, gamma=0.0)
    sl = _qp_snapshot(0.0, 0.0, medium=flat)
    psi = np.asarray(psi)
    pred = qp_radius(psi, travel, **dict(CHRISTOFFEL_MEDIUM, **flat,
                                         theta=0.0, phi=0.0))
    return shape_error(pick_front(sl, g["dh"], psi, r_max_cells=r_max), pred)[1]


@cuda_mark
@wavefront_mark
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (40.0, 45.0)],
                         ids=lambda v: f"{v:g}")
def test_qp_wavefront_matches_christoffel(theta, phi):
    """A point source in a homogeneous medium radiates the group-velocity
    surface, so the picked qP front must have the shape Christoffel predicts.

    Only the ANGULAR variation is asserted. A common radial offset is grid
    dispersion plus the finite-bandwidth bias of an envelope pick and is there
    for an isotropic medium too; the isotropic run is measured here as the floor
    so the threshold is not a guess.
    """
    from christoffel_reference import pick_front, qp_radius, shape_error

    g = WAVEFRONT_GRID
    travel = g["it_snap"] * g["dt"] - g["delay"]
    psi = np.arange(0.0, 360.0, 4.0)
    r_max = 0.38 * g["n"]

    floor = _isotropic_floor(tuple(psi), travel, r_max)

    med = dict(CHRISTOFFEL_MEDIUM, theta=np.deg2rad(theta), phi=np.deg2rad(phi))
    meas = pick_front(_qp_snapshot(theta, phi), g["dh"], psi, r_max_cells=r_max)
    pred = qp_radius(psi, travel, **med)
    bias, rms, dev = shape_error(meas, pred)
    print(f"[christoffel {theta}/{phi}] bias {100 * bias:+.2f}%  "
          f"shape rms {100 * rms:.2f}%  max dev {100 * dev:.2f}%  "
          f"(isotropic floor {100 * floor:.2f}%)")
    assert rms < max(3.0 * floor, 0.01), (
        f"qP front shape is off: rms {100 * rms:.2f}% against a "
        f"{100 * floor:.2f}% measurement floor")

    # Discrimination: the same data against a prediction that drops the
    # anisotropy must fail badly, otherwise the assertion above is vacuous.
    flat = dict(med, epsilon=0.0, delta=0.0, gamma=0.0)
    _, rms_flat, _ = shape_error(meas, qp_radius(psi, travel, **flat))
    assert rms_flat > 8.0 * max(rms, 1e-3), (
        f"the test cannot see the anisotropy: isotropic prediction gives "
        f"rms {100 * rms_flat:.2f}% vs {100 * rms:.2f}%")
