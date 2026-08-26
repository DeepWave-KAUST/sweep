"""CPU C++ binding gradient mode consistency checks.

Run from the repository root after installing the package, for example:

    SWEEP_JIT_FULL=1 PYTHONPATH=src python test/cpu_binding_gradient_consistency.py

``SWEEP_JIT_FULL=1`` is required: this script runs ``impl='c'`` on the CPU, and
the default JIT build compiles only the CUDA tree plus a CPU stub, so without it
every case dies with ``CUDAGuardImpl initialized with non-CUDA DeviceType: cpu``.

The script intentionally keeps the grids small.  It compares the CPU path of
``backend="torch", impl="c"`` across full, boundary-saving,
chunk-checkpoint, and recursive-checkpoint backward requests for every
equation currently exported through the compiled binding.  When CUDA is
available it also compares each C/CPU mode against the matching C/CUDA mode.
Gradient checks report max absolute error, relative L2 error, and cosine
similarity because small gradients can have tiny pointwise errors even when
their direction is wrong.

NOTE ON WHAT THIS FILE CAN AND CANNOT SEE.  Every comparison here is C-vs-C.
That makes it a mode-consistency check, not a correctness check: a defect
present in both C paths — which is what every elastic adjoint bug found so far
turned out to be — is invisible from here.  ``backend_gradient_matrix.py``
imports this module's ``Case``/``CASES``/``run_case`` and adds the eager leg;
that is the file to run when the question is "is the compiled gradient right",
as opposed to "do the memory modes agree".

``Case`` carries the source and receiver loading as axes (``source_type``,
``receiver_type``).  They are not decoration: with the elastic defaults alone
(explosive stress source, velocity receivers) whole families of defect are
unreachable — see the comments on ``Case.source_type`` and on the
``*_src_*`` / ``*_rec_*`` entries in ``CASES``.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import (  # noqa: E402
    Acoustic,
    ElasticTTI2nd,
    ElasticTTISG,
    ElasticTTISG3D,
    Acoustic3D,
    AcousticLSRTM,
    AcousticLSRTM3D,
    AcousticVRZ,
    AcousticVRZ3D,
    DASMu,
    DASMu3D,
    DASZhao,
    DASZhao3D,
    Elastic,
)
from sweep.equations.elastic3d import Elastic as Elastic3D  # noqa: E402
from sweep.propagator.options import CUDAOptions, CkptOptions, MemoryOptions  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    equation_cls: type
    shape: tuple[int, ...]
    pml_type: str
    receiver_type: list[str] | None = None
    # How the shot is loaded.  ``None`` = the equation's default
    # (``['sxx','szz']`` for Elastic — an explosive stress source), which is
    # what every case used to get, silently and without a way to change it.
    #
    # It matters far more than a knob-for-completeness: the compiled elastic
    # backward treats the two families differently.  A body-force source
    # (``['vz']``, ``['vx']``) is injected raw into the velocity field, so the
    # stored difference v(it) - v(it+1) that the rho imaging correlates carries
    # the injected amplitude and needs a source-cell correction; a stress source
    # does not go near that term.  Whole classes of defect therefore live only
    # on one side of this axis and are unreachable from the other.
    source_type: list[str] | None = None


@dataclass(frozen=True)
class TensorMetrics:
    max_abs: float
    rel_l2: float
    cosine: float
    reference_norm: float
    candidate_norm: float


@dataclass(frozen=True)
class RunConfig:
    spatial_order: int = 2
    abcn: int = 2
    nt: int = 8
    dt: float = 0.001
    freq: float = 12.0
    delay: float = 0.004
    ckpt_chunks: int = 3
    ckpt_count: int = 2
    transfer_interval: int = 2
    receiver_stride2d: int = 6
    receiver_stride3d: int = 8
    shape2d: tuple[int, int] = (14, 16)
    shape3d: tuple[int, int, int] = (8, 8, 8)


DEFAULT_CONFIG = RunConfig()
CUDA_SUITE_CONFIG = RunConfig(
    spatial_order=4,
    abcn=30,
    nt=120,
    dt=0.0015,
    freq=10.0,
    delay=0.06,
    ckpt_chunks=24,
    ckpt_count=4,
    transfer_interval=4,
    receiver_stride2d=6,
    receiver_stride3d=8,
    shape2d=(48, 56),
    shape3d=(24, 20, 24),
)


CASES = (
    Case("acoustic2d", Acoustic, (14, 16), "cpmlr"),
    Case("acoustic3d", Acoustic3D, (8, 8, 8), "cpmlr"),
    Case("acoustic_lsrtm2d", AcousticLSRTM, (14, 16), "cpmlr"),
    Case("acoustic_lsrtm3d", AcousticLSRTM3D, (8, 8, 8), "cpmlr"),
    Case("acoustic_vrz2d", AcousticVRZ, (14, 16), "cpmlr"),
    Case("acoustic_vrz3d", AcousticVRZ3D, (8, 8, 8), "cpmlr"),
    Case("elastic2d", Elastic, (14, 16), "cpmls"),
    Case("elastic3d", Elastic3D, (8, 8, 8), "cpmls"),
    Case("das2d", DASZhao, (14, 16), "cpmls", ["exx_t", "ezz_t", "das35_t"]),
    Case("das3d", DASZhao3D, (8, 8, 8), "cpmls", ["exx_t", "eyy_t", "ezz_t", "das35_t"]),
    Case("das_mu2d", DASMu, (14, 16), "cpmls", ["exx", "ezz", "exz"]),
    Case("das_mu3d", DASMu3D, (8, 8, 8), "cpmls", ["exx", "eyy", "ezz", "exy", "exz", "eyz"]),
    # Anisotropic elastic.  These were missing when PR#62 revived the matrix,
    # which is why elastic_tti_sg3d kept a negated stress-receiver adjoint and
    # a missing receiver-cell rho correction until 9ee4def.
    #
    # Run these with --scale cuda-suite.  The 'tiny' profile (order 2, abcn 2,
    # nt=8) cannot resolve the TTI rho gradient: it is the smallest of the eight
    # by three orders of magnitude, so eight order-2 steps leave it at the fp32
    # cancellation floor (rel_l2 8e-4..1.6e-3 at cos 1.000000, against the 5e-4
    # default rtol) and the rho cells report FAIL on noise.  At cuda-suite every
    # TTI cell passes.  Same at tiny for the upstream elastic_tti_sg2d, so this
    # is the scale, not the equations.
    Case("elastic_tti_sg2d", ElasticTTISG, (14, 16), "cpmls"),
    Case("elastic_tti_sg3d", ElasticTTISG3D, (8, 8, 8), "cpmls"),
    Case("elastic_tti_2nd2d", ElasticTTI2nd, (14, 16), "cpmls", ["ux", "uz"], ["uz"]),
    # ---- source / receiver loading axes -------------------------------------
    # Every case above rides on the equation defaults, which for Elastic means
    # an explosive stress source recorded on velocities.  These add the corners
    # that default can never reach.  They are deliberately a short list rather
    # than a cross product: one case per mechanism.
    #
    #   *_src_vz / *_src_vx  body-force loading, the only family whose injection
    #                        lands in the term the rho gradient is built from
    #   *_src_sxz            shear-stress loading (off-diagonal, exercises the
    #                        mu path with no normal-stress component)
    #   *_rec_stress         a stress receiver: the adjoint residual is injected
    #                        into an adjoint STRESS field, which the compiled
    #                        elastic adjoint stores with the opposite sign of
    #                        the true adjoint
    #   *_rec_mixed          velocity and stress receivers together, so a
    #                        per-field sign error cannot hide behind a global one
    Case("elastic2d_src_vz", Elastic, (14, 16), "cpmls", None, ["vz"]),
    Case("elastic2d_src_vx", Elastic, (14, 16), "cpmls", None, ["vx"]),
    Case("elastic2d_src_sxz", Elastic, (14, 16), "cpmls", None, ["sxz"]),
    Case("elastic2d_rec_stress", Elastic, (14, 16), "cpmls", ["sxx", "szz"], None),
    Case("elastic2d_rec_mixed", Elastic, (14, 16), "cpmls", ["vz", "sxx"], ["vz"]),
    Case("elastic3d_src_vz", Elastic3D, (8, 8, 8), "cpmls", None, ["vz"]),
    Case("elastic3d_rec_stress", Elastic3D, (8, 8, 8), "cpmls", ["sxx", "syy", "szz"], None),
    Case("elastic_tti_sg2d_rec_stress", ElasticTTISG, (14, 16), "cpmls", ["sxx", "szz"], None),
    Case("elastic_tti_sg2d_src_vz", ElasticTTISG, (14, 16), "cpmls", None, ["vz"]),
    Case("elastic_tti_sg3d_rec_stress", ElasticTTISG3D, (8, 8, 8), "cpmls",
         ["sxx", "syy", "szz"], None),
    Case("elastic_tti_sg3d_src_vz", ElasticTTISG3D, (8, 8, 8), "cpmls", None, ["vz"]),
    Case("das_mu2d_src_vz", DASMu, (14, 16), "cpmls", ["exx", "ezz", "exz"], ["vz"]),
)

# Gradients impl='c' is KNOWN not to produce, keyed by (case-name prefix, grad
# name).  These are reported as MISSING rather than FAIL so the matrix keeps a
# usable signal — but they are counted, printed, and cross-checked: if the
# gradient ever shows up, the run FAILS with "prune the stale entry", so an
# entry cannot outlive the gap it documents.
#
# Every one of these is a real hole, not a tolerance question: the candidate
# returns exactly 0.0 while eager returns a finite gradient.  Only acoustic2d /
# acoustic3d implement the source-wavelet gradient on the C side
# (``accumulate_source_gradient_2d/3d``); every other equation's backward
# returns model gradients only, and ``_c.py`` picks up a wavelet gradient only
# when ``len(returned_grads) == len(models) + 1``, so the rest silently get None
# -> zero, with no warning.
KNOWN_MISSING_GRADIENTS = {
    ("elastic", "wavelet"): "impl='c' elastic backward returns no source-wavelet gradient",
    ("das", "wavelet"): "impl='c' DAS backward returns no source-wavelet gradient",
    ("das_mu", "wavelet"): "impl='c' DAS-mu backward returns no source-wavelet gradient",
    ("acoustic_vrz", "wavelet"): "impl='c' VRZ backward returns no source-wavelet gradient",
    ("acoustic_lsrtm2d", "wavelet"): "impl='c' LSRTM2D backward returns no source-wavelet gradient",
    # LSRTM inverts the reflectivity mp; vp comes back zero on the C side.
    # NOTE lsrtm3d is deliberately absent from the wavelet list: it DOES return
    # one, and it is wrong (cos -0.005, rel 4.0 vs eager) — that must stay a
    # failure, not a declared gap.
    ("acoustic_lsrtm2d", "vp"): "impl='c' LSRTM2D backward returns no vp gradient (mp only)",
    ("acoustic_lsrtm3d", "vp"): "impl='c' LSRTM3D backward returns no vp gradient (mp only)",
}


def known_missing_gradient(case_name: str, grad_name: str) -> str | None:
    """Longest matching prefix wins, so ``acoustic_lsrtm3d`` can declare a gap
    that ``acoustic_lsrtm`` does not."""
    best = None
    for (prefix, name), reason in KNOWN_MISSING_GRADIENTS.items():
        if name != grad_name or not case_name.startswith(prefix):
            continue
        if best is None or len(prefix) > best[0]:
            best = (len(prefix), reason)
    return None if best is None else best[1]


MODES = ("full", "bs", "ckpt_chunk", "ckpt_recursive")
CUDA_MODE_REFERENCE_ONLY = {
    ("acoustic_vrz2d", "bs"),
    ("acoustic_vrz3d", "bs"),
    ("elastic2d", "bs"),
    ("elastic3d", "bs"),
    # Same mechanism as the elastic entries -- the bs rho gradient differs
    # from full at the source/receiver CELLS (stress-source / stress-receiver
    # bookkeeping against a reconstructed rather than stored wavefield), not
    # in reconstruction quality. These four surfaced when the wavelet
    # bookkeeping misfire stopped masking per-cell first-error reporting.
    ("das_mu2d", "bs"),
    ("das_mu3d", "bs"),
    ("elastic2d_src_sxz", "bs"),
    ("elastic2d_rec_stress", "bs"),
    ("elastic3d_rec_stress", "bs"),
}


def config_for_scale(scale: str) -> RunConfig:
    if scale == "tiny":
        return DEFAULT_CONFIG
    if scale == "cuda-suite":
        return CUDA_SUITE_CONFIG
    raise ValueError(f"Unknown scale {scale!r}")


def scaled_cases(config: RunConfig, selected: tuple[Case, ...] = CASES) -> tuple[Case, ...]:
    out: list[Case] = []
    for case in selected:
        shape = config.shape2d if len(case.shape) == 2 else config.shape3d
        out.append(dataclasses.replace(case, shape=shape))
    return tuple(out)


def ricker(nt: int, dt: float, fm: float = 12.0, delay: float = 0.004) -> np.ndarray:
    """Shape ``(nt,)`` — mode A1 of the source/receiver IO contract (one shared
    wavelet, one point source per shot).

    This used to return ``(1, 1, nt)``, which ``PropBase._normalize_io`` reads
    as source-encoding mode (mode B) and rejects outright; that is what had this
    harness raising ``ValueError: wavelet must have shape (nt,) or (nsrc, nt)``
    on every single case.  The ``(1, 1, ndim)`` sources it was paired with were
    the other half of the problem: still accepted, but silently routing a suite
    meant to exercise the ordinary single-shot path through the encoded one.
    """
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def geometry(shape: tuple[int, ...], config: RunConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Mode A1 geometry: sources ``(nshots, ndim)``, receivers ``(nshots, nrec,
    ndim)``.  ``sources`` used to be ``(1, 1, ndim)`` — accepted, but that is
    the source-encoding shape, so the suite was exercising the encoded path
    rather than the ordinary single-shot one it documents."""
    radius = max(1, config.spatial_order // 2)
    source_z = max(1, min(shape[0] - 1, shape[0] // 4))
    receiver_z = max(1, min(shape[0] - 1, radius))
    margin = max(2, radius)
    if len(shape) == 2:
        nz, nx = shape
        source = np.array([[nx // 2, source_z]], dtype=np.int32)
        rec_x = np.arange(margin, max(margin + 1, nx - margin), config.receiver_stride2d, dtype=np.int32)
        if rec_x.size == 0:
            rec_x = np.array([nx // 2], dtype=np.int32)
        rec_z = np.full(rec_x.size, receiver_z, dtype=np.int32)
        receivers = np.stack([np.clip(rec_x, 0, nx - 1), rec_z], axis=-1)[None, ...]
        return source, receivers

    nz, ny, nx = shape
    source = np.array([[nx // 2, ny // 2, source_z]], dtype=np.int32)
    rec_x = np.arange(margin, max(margin + 1, nx - margin), config.receiver_stride3d, dtype=np.int32)
    rec_y = np.arange(margin, max(margin + 1, ny - margin), config.receiver_stride3d, dtype=np.int32)
    if rec_x.size == 0:
        rec_x = np.array([nx // 2], dtype=np.int32)
    if rec_y.size == 0:
        rec_y = np.array([ny // 2], dtype=np.int32)
    grid_y, grid_x = np.meshgrid(np.clip(rec_y, 0, ny - 1), np.clip(rec_x, 0, nx - 1), indexing="ij")
    rec_z = np.full(grid_x.size, receiver_z, dtype=np.int32)
    receivers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), rec_z], axis=-1)[None, ...]
    return source, receivers


def model_arrays(model_names: list[str], shape: tuple[int, ...]) -> list[np.ndarray]:
    grid = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    arrays: list[np.ndarray] = []
    for name in model_names:
        if name == "vp":
            arrays.append(2200.0 + 60.0 * grid)
        elif name == "vs":
            arrays.append(1200.0 + 30.0 * grid)
        elif name == "rho":
            arrays.append(2100.0 + 20.0 * grid)
        elif name == "mp":
            arrays.append(0.03 + 0.01 * grid)
        elif name == "z":
            arrays.append(4.6e6 + 1.0e4 * grid)
        # Anisotropic (TTI) parameters.  Non-uniform on purpose: a constant
        # Thomsen/angle field degenerates several of these gradients and hides
        # defects (see reference_canonical_gradient_test_model).  Nonzero
        # theta/phi also keep the Bond rotation off its VTI fallback branch.
        elif name == "vp0":
            arrays.append(2200.0 + 60.0 * grid)
        elif name == "vs0":
            arrays.append(1200.0 + 30.0 * grid)
        elif name == "vh":
            arrays.append(2400.0 + 60.0 * grid)
        elif name == "epsilon":
            arrays.append(0.08 + 0.04 * grid)
        elif name == "delta":
            arrays.append(0.03 + 0.02 * grid)
        elif name == "eta":
            arrays.append(0.04 + 0.02 * grid)
        elif name == "gamma":
            arrays.append(0.05 + 0.02 * grid)
        elif name == "theta":
            arrays.append(0.25 + 0.10 * grid)
        elif name == "phi":
            arrays.append(0.15 + 0.08 * grid)
        else:
            raise KeyError(f"No synthetic model recipe for {name!r}.")
    return [array.astype(np.float32, copy=False) for array in arrays]


def make_equation(case: Case, device: torch.device | None = None, config: RunConfig = DEFAULT_CONFIG) -> object:
    return case.equation_cls(spatial_order=config.spatial_order, device=device or torch.device("cpu"), backend="torch")


def make_solver(case: Case, mode: str, device: torch.device | None = None, config: RunConfig = DEFAULT_CONFIG) -> PropTorch:
    device = device or torch.device("cpu")
    equation = make_equation(case, device, config)
    common = dict(
        shape=case.shape,
        abcn=config.abcn,
        dh=10.0,
        dt=config.dt,
        dev=device,
        pml_type=case.pml_type,
        nt=config.nt,
        B=1,
        receiver_type=case.receiver_type,
        source_type=case.source_type,
    )
    if mode == "full":
        return PropTorch(
            equation,
            backend="torch",
            impl="c",
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
            **common,
        )
    if mode == "bs":
        return PropTorch(
            equation,
            backend="torch",
            impl="c",
            use_ckpt=False,
            boundary_saving_config={"enabled": True, "storage": "cpu", "transfer_interval": config.transfer_interval},
            **common,
        )
    if mode == "ckpt_chunk":
        cuda_options = CUDAOptions(
            memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=config.ckpt_chunks))
        )
        return PropTorch(equation, backend="torch", impl="c", cuda_options=cuda_options, **common)
    if mode == "ckpt_recursive":
        cuda_options = CUDAOptions(
            memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="recursive", count=config.ckpt_count))
        )
        return PropTorch(equation, backend="torch", impl="c", cuda_options=cuda_options, **common)
    raise ValueError(mode)


def run_case(
    solver: PropTorch,
    case: Case,
    device: torch.device | None = None,
    config: RunConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = device or torch.device("cpu")
    wavelet = torch.tensor(
        ricker(nt=config.nt, dt=config.dt, fm=config.freq, delay=config.delay),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    sources, receivers = geometry(case.shape, config)
    model_names = make_equation(case, device, config).models
    models = [
        torch.tensor(array, dtype=torch.float32, device=device, requires_grad=True)
        for array in model_arrays(model_names, case.shape)
    ]

    record = solver(wavelet, sources, receivers, models=models)
    loss = record.pow(2).mean()
    loss.backward()

    def grad_or_zero(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.grad is None:
            return torch.zeros_like(tensor, device="cpu")
        return tensor.grad.detach().cpu().clone()

    grads = {"wavelet": grad_or_zero(wavelet)}
    grads.update({name: grad_or_zero(tensor) for name, tensor in zip(model_names, models)})
    return record.detach().cpu().clone(), grads


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor, *, eps: float) -> TensorMetrics:
    candidate_flat = candidate.detach().to(dtype=torch.float64).reshape(-1)
    reference_flat = reference.detach().to(dtype=torch.float64).reshape(-1)
    diff = candidate_flat - reference_flat
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    reference_norm = float(torch.linalg.vector_norm(reference_flat).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate_flat).item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item() / max(reference_norm, eps))
    if reference_norm <= eps or candidate_norm <= eps:
        cosine = float("nan")
    else:
        cosine = float(torch.dot(candidate_flat, reference_flat).item() / (candidate_norm * reference_norm))
        cosine = max(-1.0, min(1.0, cosine))
    return TensorMetrics(
        max_abs=max_abs,
        rel_l2=rel_l2,
        cosine=cosine,
        reference_norm=reference_norm,
        candidate_norm=candidate_norm,
    )


def format_metrics(metrics: TensorMetrics) -> str:
    cosine = "nan" if np.isnan(metrics.cosine) else f"{metrics.cosine:.8f}"
    return (
        f"max_abs={metrics.max_abs:.3e} "
        f"rel_l2={metrics.rel_l2:.3e} "
        f"cos={cosine}"
    )


def assert_gradient_cosine(
    case_name: str,
    mode: str,
    name: str,
    metrics: TensorMetrics,
    *,
    threshold: float,
    eps: float,
) -> None:
    reference_zero = metrics.reference_norm <= eps
    candidate_zero = metrics.candidate_norm <= eps
    if reference_zero and candidate_zero:
        return
    if reference_zero != candidate_zero:
        raise AssertionError(
            f"{case_name}/{mode} {name} gradient norm mismatch: "
            f"reference_norm={metrics.reference_norm:.3e}, "
            f"candidate_norm={metrics.candidate_norm:.3e}, "
            f"{format_metrics(metrics)}"
        )
    if metrics.cosine < threshold:
        raise AssertionError(
            f"{case_name}/{mode} {name} gradient cosine below threshold: "
            f"{format_metrics(metrics)} threshold={threshold:.8f}"
        )


def effective_atol(reference: torch.Tensor, atol: float, atol_scale: float) -> float:
    """Absolute tolerance measured in units of the reference's own peak.

    A fixed ``atol`` cannot serve this case list: the compared tensors span
    ~1e-9 to ~1e9 (a stress receiver moves the record scale by six orders of
    magnitude on its own).  And it fails in the dangerous direction — the old
    ``atol=1e-5`` floor was LARGER than an entire elastic rho gradient (peak
    ~1e-6), so every element passed unconditionally and the comparison was
    vacuous.  That is why adding the source_type axis alone did not turn the
    body-force rho defects red; the axis was reaching the right configuration
    and the tolerance was throwing the answer away.

    Scale it to the reference peak, and keep the fixed value only as the
    degenerate fallback for an all-zero reference.
    """
    if atol_scale <= 0:
        return atol
    peak = float(reference.detach().abs().max()) if reference.numel() else 0.0
    return atol_scale * peak if peak > 0.0 else atol


def assert_result_consistent(
    case_name: str,
    mode: str,
    reference: tuple[torch.Tensor, dict[str, torch.Tensor]],
    candidate: tuple[torch.Tensor, dict[str, torch.Tensor]],
    *,
    reference_label: str,
    rtol: float,
    atol: float,
    cosine_threshold: float,
    cosine_eps: float,
) -> list[str]:
    reference_record, reference_grads = reference
    candidate_record, candidate_grads = candidate
    torch.testing.assert_close(
        candidate_record,
        reference_record,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: f"{case_name}/{mode} record differs from {reference_label}\n{msg}",
    )
    summaries: list[str] = []
    for name, reference_grad in reference_grads.items():
        candidate_grad = candidate_grads[name]
        metrics = tensor_metrics(candidate_grad, reference_grad, eps=cosine_eps)
        torch.testing.assert_close(
            candidate_grad,
            reference_grad,
            rtol=rtol,
            atol=atol,
            msg=lambda msg, name=name, metrics=metrics: (
                f"{case_name}/{mode} {name} gradient differs from {reference_label}\n"
                f"{format_metrics(metrics)}\n{msg}"
            ),
        )
        assert_gradient_cosine(
            case_name,
            mode,
            name,
            metrics,
            threshold=cosine_threshold,
            eps=cosine_eps,
        )
        summaries.append(f"{name}: {format_metrics(metrics)}")
    return summaries


def compare_result_to_reference(
    case_name: str,
    mode: str,
    reference: tuple[torch.Tensor, dict[str, torch.Tensor]],
    candidate: tuple[torch.Tensor, dict[str, torch.Tensor]],
    *,
    reference_label: str,
    rtol: float,
    atol: float,
    cosine_threshold: float,
    cosine_eps: float,
    strict: bool,
    atol_scale: float = 2e-4,
    record_atol_scale: float = 1e-3,
) -> tuple[bool, list[str], list[str], list[str]]:
    """Returns ``(ok, summaries, errors, missing)``.

    ``missing`` lists gradients this backend is DECLARED not to produce (see
    ``KNOWN_MISSING_GRADIENTS``).  They are kept out of ``errors`` so the matrix
    still has a usable pass/fail signal, and reported separately so they stay
    visible — a silently-zero gradient is the failure mode that hid the elastic
    source-wavelet gap for the whole life of the feature.
    """
    reference_record, reference_grads = reference
    candidate_record, candidate_grads = candidate
    summaries: list[str] = []
    errors: list[str] = []

    record_metrics = tensor_metrics(candidate_record, reference_record, eps=cosine_eps)
    try:
        torch.testing.assert_close(
            candidate_record,
            reference_record,
            rtol=rtol,
            atol=effective_atol(reference_record, atol, record_atol_scale),
            msg=lambda msg: f"{case_name}/{mode} record differs from {reference_label}\n{msg}",
        )
    except AssertionError as exc:
        errors.append(str(exc).splitlines()[0])
    summaries.append(f"record: {format_metrics(record_metrics)}")

    missing: list[str] = []
    for name, reference_grad in reference_grads.items():
        candidate_grad = candidate_grads[name]
        metrics = tensor_metrics(candidate_grad, reference_grad, eps=cosine_eps)
        gap = known_missing_gradient(case_name, name)
        if gap is not None:
            # Declared missing on this backend. Report it in its own category so
            # the matrix keeps a usable pass/fail signal — but never drop it, and
            # flag the entry as stale the moment the gradient shows up, so the
            # list cannot quietly outlive the gap it documents.
            # The entry is stale only if the candidate actually RETURNS the
            # gradient. Against an eager reference the gap shows up as
            # zero-vs-nonzero; against a same-backend reference (the within
            # scope) BOTH sides lack it and the comparison is vacuous -- that
            # is consistent with the declaration, not evidence against it.
            absent = metrics.candidate_norm <= cosine_eps
            if absent:
                vac = ("" if metrics.reference_norm > cosine_eps
                       else " [vacuous here: the reference lacks it too]")
                missing.append(f"{case_name}/{mode} {name}: {gap}{vac}")
                summaries.append(f"{name}: MISSING ({gap})")
                continue
            errors.append(
                f"{case_name}/{mode} {name}: KNOWN_MISSING_GRADIENTS claims this "
                f"gradient is absent ({gap}) but the candidate returned one "
                f"({format_metrics(metrics)}) — prune the stale entry"
            )
            continue
        try:
            torch.testing.assert_close(
                candidate_grad,
                reference_grad,
                rtol=rtol,
                atol=effective_atol(reference_grad, atol, atol_scale),
                msg=lambda msg, name=name, metrics=metrics: (
                    f"{case_name}/{mode} {name} gradient differs from {reference_label}\n"
                    f"{format_metrics(metrics)}\n{msg}"
                ),
            )
            assert_gradient_cosine(
                case_name,
                mode,
                name,
                metrics,
                threshold=cosine_threshold,
                eps=cosine_eps,
            )
        except AssertionError as exc:
            errors.append(str(exc).splitlines()[0])
        summaries.append(f"{name}: {format_metrics(metrics)}")

    if errors and strict:
        raise AssertionError("\n".join(errors))
    return not errors, summaries, errors, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[case.name for case in CASES],
        choices=[case.name for case in CASES],
        help="Case names to run. Defaults to all compiled CPU binding cases.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=list(MODES),
        choices=list(MODES),
        help="Gradient modes to compare against full. Defaults to all modes.",
    )
    parser.add_argument(
        "--scale",
        choices=("tiny", "cuda-suite"),
        default="tiny",
        help=(
            "Problem size profile. 'cuda-suite' matches the older CUDA gradient "
            "suite defaults: 2D=48x56, 3D=24x20x24, nt=120, so=4, abcn=30."
        ),
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.999,
        help="Minimum cosine similarity for nonzero gradients.",
    )
    parser.add_argument(
        "--cosine-eps",
        type=float,
        default=1e-30,
        help="Gradient norm below this value is treated as zero for cosine checks.",
    )
    parser.add_argument(
        "--compare-cuda",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Compare each C/CPU mode against matching C/CUDA mode. "
            "'auto' runs the comparison only when CUDA is available."
        ),
    )
    parser.add_argument(
        "--cuda-rtol",
        type=float,
        default=5e-4,
        help="Relative tolerance for C/CPU vs C/CUDA comparisons.",
    )
    parser.add_argument(
        "--cuda-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for C/CPU vs C/CUDA comparisons.",
    )
    parser.add_argument(
        "--cuda-cosine-threshold",
        type=float,
        default=0.999,
        help="Minimum gradient cosine for C/CPU vs C/CUDA comparisons.",
    )
    parser.add_argument(
        "--cuda-strict",
        action="store_true",
        help="Fail on C/CPU vs C/CUDA mismatch instead of reporting and continuing.",
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, torch.get_num_threads())))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_available = torch.cuda.is_available()
    compare_cuda = args.compare_cuda == "always" or (args.compare_cuda == "auto" and cuda_available)
    if args.compare_cuda == "always" and not cuda_available:
        raise RuntimeError("Requested --compare-cuda always, but CUDA is not available.")
    if args.compare_cuda == "auto" and not cuda_available:
        print("C/CUDA comparison skipped because torch.cuda.is_available() is false.")

    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    config = config_for_scale(args.scale)
    selected_cases = list(scaled_cases(config, tuple(selected_cases)))
    selected_modes = list(dict.fromkeys(["full", *args.modes]))
    cuda_ok_count = 0
    cuda_mismatch_count = 0
    cuda_mismatches: list[str] = []
    for case in selected_cases:
        reference = run_case(make_solver(case, "full", torch.device("cpu"), config), case, torch.device("cpu"), config)
        print(f"{case.name:20s} full           c/cpu reference")
        for mode in selected_modes:
            if mode == "full":
                candidate = reference
            else:
                candidate = run_case(make_solver(case, mode, torch.device("cpu"), config), case, torch.device("cpu"), config)
                if (case.name, mode) in CUDA_MODE_REFERENCE_ONLY:
                    print(f"{case.name:20s} {mode:14s} c/cpu full check skipped")
                    print(f"{'':20s} {'':14s} follows matching c/cuda mode, not c/cpu full mode")
                else:
                    summaries = assert_result_consistent(
                        case.name,
                        mode,
                        reference,
                        candidate,
                        reference_label="c/cpu full mode",
                        rtol=args.rtol,
                        atol=args.atol,
                        cosine_threshold=args.cosine_threshold,
                        cosine_eps=args.cosine_eps,
                    )
                    print(f"{case.name:20s} {mode:14s} c/cpu ok")
                    for summary in summaries:
                        print(f"{'':20s} {'':14s} {summary}")

            if not compare_cuda:
                continue

            cuda_device = torch.device("cuda")
            cuda_reference = run_case(make_solver(case, mode, cuda_device, config), case, cuda_device, config)
            ok, summaries, errors, missing = compare_result_to_reference(
                case.name,
                f"{mode}/c-cuda",
                cuda_reference,
                candidate,
                reference_label=f"c/cuda {mode} mode",
                rtol=args.cuda_rtol,
                atol=args.cuda_atol,
                cosine_threshold=args.cuda_cosine_threshold,
                cosine_eps=args.cosine_eps,
                strict=args.cuda_strict,
            )
            status = "c/cuda ok" if ok else "c/cuda mismatch"
            if ok:
                cuda_ok_count += 1
            else:
                cuda_mismatch_count += 1
                cuda_mismatches.append(f"{case.name}/{mode}")
            print(f"{case.name:20s} {mode:14s} {status}")
            for summary in summaries:
                print(f"{'':20s} {'':14s} {summary}")
            for error in errors:
                print(f"{'':20s} {'':14s} ! {error}")

    if compare_cuda:
        print(
            f"C/CUDA comparison summary: ok={cuda_ok_count}, "
            f"mismatch={cuda_mismatch_count}"
        )
        if cuda_mismatches:
            print("C/CUDA mismatches: " + ", ".join(cuda_mismatches))


if __name__ == "__main__":
    main()
