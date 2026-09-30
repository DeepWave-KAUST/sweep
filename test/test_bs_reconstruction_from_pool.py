"""The boundary-saving backward's reconstruction state comes from the propagator.

In boundary-saving mode the backward re-runs the forward backwards from
``u_last_two`` with the saved boundaries injected. The wavefield struct it steps
(the physical fields, plus the carriers the imaging reads) used to be allocated
inside the compiled call. Now ``cuda_layout.reconstruction_nvar`` (an explicit
``bs_reconstruction_nvar``, or the slot table's ``recon`` list -- the DD path's
source of the same count) says how many zeroed grids the propagator hands the
backward as ``forward_wavefields``, and every driver binds them.

Instrument (the same as the other pool tests): wrap ``backward_bs_func``, fill
each handed-over grid with a sentinel right before the call, and require that
every grid was written afterwards -- a driver that still allocated its own
struct would leave every sentinel in place. The absorbing rim is never written
by the reverse stencil, so the criterion is "some cell changed", never "all".
The same wrapper counts the caching allocator's calls across the backward: the
remaining per-call allocations are pinned per equation (RESIDUAL) so a driver
that quietly starts allocating again fails here, not in a memory profile weeks
later.

The second half of the file does the same for the LAST buffers the boundary
saver allocated on its own: the FP32 staging bands a scaled (int8/fp16) store
quantizes through. See the section comment below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

SENTINEL = 12345.0

# key -> (reconstruction grids handed over, allocations the bs backward still
# makes inside C++ -- none since slice 16 (the negated source became a signed
# injection kernel), binding prefix)
# The DAS-mu drivers hand over the elastic fields only: their strain grids are
# dead in the bs reverse loop, and a dead grid is exactly what this test's
# sentinel would expose.
CASES = {
    "acoustic2d": (3, 0, "acoustic2d"),
    "acoustic3d": (3, 0, "acoustic3d"),
    "lsrtm2d": (3, 0, "acoustic_lsrtm2d"),
    "lsrtm3d": (3, 0, "acoustic_lsrtm3d"),
    "vrz2d": (3, 0, "acoustic_vrz2d"),
    "vrz3d": (3, 0, "acoustic_vrz3d"),
    "elastic2d": (7, 0, "elastic2d"),
    "elastic3d": (12, 0, "elastic3d"),
    "das_mu2d": (7, 0, "das_mu2d"),
    "das_mu3d": (12, 0, "das_mu3d"),
    "elastic_tti_sg2d": (11, 0, "elastic_tti_sg2d"),
    "elastic_tti_sg3d": (12, 0, "elastic_tti_sg3d"),
    "das2d": (9, 0, "das2d"),
    "acoustic_vti_1st_2d": (4, 0, "acoustic_vti_1st_2d"),
    "acoustic_vti_1st_3d": (5, 0, "acoustic_vti_1st_3d"),
    "elastic_tti_2nd2d": (6, 0, "elastic_tti_2nd2d"),
}
PARAMS = [pytest.param(k, marks=requires_binding(f"{CASES[k][2]}_forward")) for k in CASES]
# cpu-staged boundaries keep u_last_two on the host; the reconstruction grids
# must still be allocated on the GPU (found the hard way: an illegal address)
STAGED = [pytest.param(k, marks=requires_binding(f"{CASES[k][2]}_forward"))
          for k in ("acoustic2d", "elastic2d", "das2d")]


def _build(key, nt=40, mode="bs_gpu"):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = nt
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", mode, scenario, shape, dev, ns,
                                Path("/tmp"), "reconpool")
    models = suite.tensors_from_models(models_init, grad_flags, dev)
    return solver, wavelet, sources, receivers, models


class _Probe:
    def __init__(self, impl, name):
        self.impl, self.name, self.fn = impl, name, getattr(impl, name)
        self.calls = []
        setattr(impl, name, self)

    def __call__(self, params):
        grids = list(params.forward_wavefields)
        for t in grids:
            t.fill_(SENTINEL)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = self.fn(params)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
        self.calls.append(dict(written=[bool((t != SENTINEL).any()) for t in grids],
                               shapes=[tuple(t.shape) for t in grids], allocs=allocs))
        return out

    def restore(self):
        setattr(self.impl, self.name, self.fn)


@pytest.mark.parametrize("key", PARAMS)
def test_bs_backward_reconstructs_into_the_handed_over_grids(key):
    n, residual, _ = CASES[key]
    solver, wavelet, sources, receivers, models = _build(key)
    impl = getattr(solver, "_backend_impl", solver)
    # warm-up: lazily created CUDA state is not what the allocation count measures
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    for m in models:
        m.grad = None
    probe = _Probe(impl, "backward_bs_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        probe.restore()
    assert len(probe.calls) == 1, f"{key}: backward_bs ran {len(probe.calls)} times"
    call = probe.calls[0]
    assert len(call["written"]) == n, (
        f"{key}: {len(call['written'])} reconstruction grids handed over, expected {n}")
    assert all(call["written"]), (
        f"{key}: reconstruction grids {[i for i, w in enumerate(call['written']) if not w]} "
        "still hold the sentinel after the backward: the driver did not bind them")
    assert call["allocs"] <= residual, (
        f"{key}: the bs backward made {call['allocs']} allocations of its own, "
        f"expected at most {residual}")
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)


@pytest.mark.parametrize("key", STAGED)
def test_bs_backward_with_cpu_staged_boundaries_reconstructs_on_the_gpu(key):
    n = CASES[key][0]
    solver, wavelet, sources, receivers, models = _build(key, mode="bs_cpu")
    impl = getattr(solver, "_backend_impl", solver)
    probe = _Probe(impl, "backward_bs_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        probe.restore()
    call = probe.calls[0]
    assert len(call["written"]) == n and all(call["written"]), (
        f"{key}: cpu-staged bs backward did not reconstruct into the handed-over grids")
    assert any(m.grad is not None and torch.isfinite(m.grad).all() and float(m.grad.abs().max()) > 0
               for m in models)


def test_reconstruction_count_comes_from_the_slot_table_where_one_exists():
    """One authority per count: a class with a slot table derives the
    reconstruction count from ``slots.recon`` (what the DD driver binds), a
    class without one declares ``bs_reconstruction_nvar`` explicitly."""
    from sweep.equations.cuda_layout import CUDALayoutSpec
    from sweep.equations import slot_table

    spec = CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2, slots=slot_table.ACOUSTIC2D)
    assert spec.reconstruction_nvar == slot_table.ACOUSTIC2D.nrecon == 3
    assert CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2).reconstruction_nvar == 0
    assert CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2,
                          bs_reconstruction_nvar=5).reconstruction_nvar == 5


# ---------------------------------------------------------------------------
# Scaled (int8 / fp16) boundary storage: the FP32 staging bands
# ---------------------------------------------------------------------------
# A scaled store never writes the boundary band into its payload directly: the
# FP32 band kernel writes ONE timestep per face, launch_quantize_* compresses
# that into the persistent payload + per-block scale, and the backward's
# launch_dequantize_* expands it back before the restore kernel reads it. Those
# bands were the boundary saver's last torch::zeros -- 4 (2-D) / 6 (3-D) per
# forward AND per backward, on every call, measured with the same allocation
# counter this file already uses. The propagator owns them now
# (``boundary_staging``, shaped by ``Layout.staging_shapes``), so both halves
# are asserted here: the handover, and that the compiled call writes into what
# it was handed. fp32/bf16 storage does not stage at all and must get no bytes.
SCALED_MODES = ("bs_gpu_int8", "bs_gpu_fp16")
UNSCALED_MODES = ("bs_gpu", "bs_gpu_bf16")
STAGING_FACES_2D = 4   # top, bottom, left, right (3-D adds front, back)


class _StagingProbe:
    """Wraps one compiled entry point: records the staging bands it was handed
    and the allocator's call count across the call, and -- when ``sentinel`` --
    fills the bands first, so "the compiled call wrote into it" is provable
    rather than assumed."""

    def __init__(self, impl, name, sentinel=False):
        self.impl, self.name, self.fn = impl, name, getattr(impl, name)
        self.sentinel, self.calls = sentinel, []
        setattr(impl, name, self)

    def __call__(self, params):
        bands = list(params.boundary_staging)
        if self.sentinel:
            for t in bands:
                t.fill_(SENTINEL)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = self.fn(params)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
        self.calls.append(dict(n=len(bands),
                               shapes=[tuple(t.shape) for t in bands],
                               dtypes=[t.dtype for t in bands],
                               devices=[t.device.type for t in bands],
                               ptrs=[t.data_ptr() for t in bands],
                               written=[bool((t != SENTINEL).any()) for t in bands],
                               allocs=allocs))
        return out

    def restore(self):
        setattr(self.impl, self.name, self.fn)


@pytest.mark.parametrize("mode", SCALED_MODES)
@pytest.mark.parametrize("key", [pytest.param("acoustic2d",
                                              marks=requires_binding("acoustic2d_forward"))])
def test_scaled_boundary_storage_stages_through_the_propagators_bands(key, mode):
    solver, wavelet, sources, receivers, models = _build(key, mode=mode)
    impl = getattr(solver, "_backend_impl", solver)
    # warm-up: lazily created CUDA state (and the one-off buffer allocation)
    # is not what the allocation count measures
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    for m in models:
        m.grad = None

    owned = list(impl.boundary_staging)
    assert len(owned) == STAGING_FACES_2D, (
        f"{mode}: the propagator allocated {len(owned)} staging bands, expected "
        f"{STAGING_FACES_2D} (top, bottom, left, right)")

    fwd = _StagingProbe(impl, "forward_func")
    # the sentinel goes on the backward: dequantize_step refills every band in
    # full before each restore, so an overwritten sentinel is proof the
    # compiled call used the propagator's memory and not a private allocation
    bwd = _StagingProbe(impl, "backward_bs_func", sentinel=True)
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        bwd.restore()
        fwd.restore()

    assert len(fwd.calls) == 1 and len(bwd.calls) == 1, (
        f"{mode}: forward ran {len(fwd.calls)}x, backward_bs {len(bwd.calls)}x")
    f, b = fwd.calls[0], bwd.calls[0]

    for name, call in (("forward", f), ("backward_bs", b)):
        assert call["n"] == STAGING_FACES_2D, (
            f"{mode}: the {name} was handed {call['n']} staging bands, expected "
            f"{STAGING_FACES_2D} -- the propagator did not bind them")
        # one timestep, not a ring: 2-D collapses both leading (nvar, nt) axes
        assert all(tuple(sh[:2]) == (1, 1) for sh in call["shapes"]), (
            f"{mode}: {name} staging is not one timestep per face: {call['shapes']}")
        assert all(d == torch.float32 for d in call["dtypes"]), (
            f"{mode}: {name} staging must be float32, got {call['dtypes']}")
        assert all(d == "cuda" for d in call["devices"]), (
            f"{mode}: {name} staging must live on the GPU, got {call['devices']}")
        assert call["ptrs"] == [t.data_ptr() for t in owned], (
            f"{mode}: the {name} was handed bands that are not the propagator's "
            "own -- forward and backward must share one buffer")
        assert call["allocs"] == 0, (
            f"{mode}: the compiled {name} made {call['allocs']} allocations of its "
            f"own; the {STAGING_FACES_2D} staging bands are Python-owned now")

    assert all(b["written"]), (
        f"{mode}: staging bands {[i for i, w in enumerate(b['written']) if not w]} "
        "still hold the sentinel after the backward: dequantize_step wrote "
        "somewhere else, i.e. the driver kept a buffer of its own")
    assert any(m.grad is not None and torch.isfinite(m.grad).all()
               and float(m.grad.abs().max()) > 0 for m in models)


@pytest.mark.parametrize("mode", UNSCALED_MODES)
@pytest.mark.parametrize("key", [pytest.param("acoustic2d",
                                              marks=requires_binding("acoustic2d_forward"))])
def test_unscaled_boundary_storage_allocates_no_staging(key, mode):
    """fp32 and bf16 write the band straight into the payload -- no quantize
    pass, no staging. Allocating bands for them would be pure extra bytes."""
    solver, wavelet, sources, receivers, models = _build(key, mode=mode)
    impl = getattr(solver, "_backend_impl", solver)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    assert impl.boundary_staging == (), (
        f"{mode}: {len(impl.boundary_staging)} staging bands allocated for a "
        "storage dtype that never stages")
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)


def test_staging_shapes_are_the_face_shapes_with_the_time_axes_collapsed():
    """One authority for the geometry: the C++ saver checks the bound bands
    against ``{1, B, width, ny_b, nx_b}`` & co, and Python derives exactly that
    from the same Layout the persistent buffers come from -- so tangent_pad /
    pad / cut_mask cannot drift between the two."""
    from sweep.memory.shape import Layout

    l3 = Layout((60, 50, 40), nvar=2, nt=30, abcn=10, M=4, B=3, width=5, tangent_pad=0)
    assert l3.staging_shapes == (
        (1, 3, l3.width, l3.ny_boundary, l3.nx_boundary),       # top
        (1, 3, l3.width, l3.ny_boundary, l3.nx_boundary),       # bottom
        (1, 3, l3.nz_boundary, l3.width, l3.nx_boundary),       # front
        (1, 3, l3.nz_boundary, l3.width, l3.nx_boundary),       # back
        (1, 3, l3.nz_boundary, l3.ny_boundary, l3.width),       # left
        (1, 3, l3.nz_boundary, l3.ny_boundary, l3.width),       # right
    )
    # 2-D keeps nvar and nt as separate axes, so the collapse takes two of them
    l2 = Layout((60, 40), nvar=2, nt=30, abcn=10, M=4, B=3, width=5, tangent_pad=4)
    assert l2.staging_shapes == (
        (1, 1, 3, l2.width, l2.nx_boundary),                    # top
        (1, 1, 3, l2.width, l2.nx_boundary),                    # bottom
        (1, 1, 3, l2.nz_boundary, l2.width),                    # left
        (1, 1, 3, l2.nz_boundary, l2.width),                    # right
    )
    # one timestep per face: exactly the per-step stride the quantize kernels
    # copy out of the persistent buffer (tangent_pad included, which is the
    # mismatch the saver's TORCH_CHECK exists to catch)
    for face, band in zip(l2.cpu_shapes, l2.staging_shapes):
        assert torch.Size(band).numel() == torch.Size(face[2:]).numel()
