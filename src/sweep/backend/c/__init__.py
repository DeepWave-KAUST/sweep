"""sweep's compiled backend (``impl='c'``): the pure-Python ctypes layer over
the prebuilt ``libsweep_core.so`` cores (csrc/core/capi.h) -- the CUDA core for
CUDA tensors and the host core (the same sources built by g++, :mod:`.hostcore`)
for CPU tensors, chosen per call from ``models[0].device``.

This replaces the compiled pybind shim (csrc/bindings/module.cpp): the same
Python surface -- one function per core entry, the stepped runners, the
boundary session -- reached through the plain C API with ctypes, so after
``pip install`` nothing compiles and no torch version is baked in.  Layout of
the C structs comes from the generated mirror (:mod:`.abi`, contract A); the
behaviour on top (contract B) is split by concern:

* :mod:`.jit` -- where the CUDA core comes from: the shipped core's fit rules,
  the cached local core, the nvcc build, and the ``SWEEP_JIT_FULL`` pybind
  developer path;
* :mod:`.hostcore` -- where the host (CPU) core comes from: the shipped
  ``lib/cpu`` core, the cached local build, the g++ build; which equations it
  leaves out;
* :mod:`.loader` -- ``core_lib(kind)``: dlopen, the C signatures, the ABI guard;
* :mod:`.adapt` -- ``adapt()``: tensor -> Buf and the input validation;
* :mod:`.entries` -- the per-entry functions and the per-call stream scope;
* :mod:`.runners` -- the persistent stepped runners and the boundary session.

Every entry and every runner ``run`` returns None: the drivers allocate
nothing, so what the core writes is in the tensors the caller bound on its
params object, and the caller reads those.  :func:`namespace` is what
``sweep._C`` loads on first use.
"""
from __future__ import annotations

from . import abi as _abi
from .entries import _core_stream_for, _entry_table, _make_entry, visco_acoustic2d_fft_workspace_bytes
from .loader import _load_lock, any_core_lib
from .runners import BackwardRunner, BoundarySession, ForwardRunner, _make_runner_factory

# The equations with a persistent stepped runner in the core; the rest must
# keep ``getattr(_C, f"{name}_forward_runner", None) is None`` so the
# propagator falls back to the per-call stepped path.
RUNNER_EQUATIONS = (
    "acoustic2d", "acoustic3d", "acoustic_vrz2d", "elastic2d", "elastic3d",
    "das_mu2d", "das_mu3d", "elastic_tti_sg2d", "elastic_tti_sg3d", "elastic_vr2d",
)
# Backward runner only: acoustic_vrz3d's phase-aware Vrz3dBackwardBsRunner (its
# forward keeps the per-call stepped path).
BACKWARD_RUNNER_EQUATIONS = ("acoustic_vrz3d",)

_ns = None


def sem_namespace() -> dict:
    """The SEM part of :func:`namespace` -- the kind-3 entries and their input
    classes -- for the compiled developer path, whose shim does not bind them
    (it links the same core, so this reaches the same kernels)."""
    from .sem import INPUTS
    return {k: v for k, v in namespace().items()
            if k.startswith(("sem2d_", "sem3d_")) or k in INPUTS}


def namespace() -> dict:
    """Everything ``sweep._C`` exposes, keyed by its old pybind name (cached;
    built once under the loader's ``_load_lock``).  The entry table is read
    from whichever core loads (both export the same one; the loader checks
    when the second arrives); each entry picks its core per call."""
    global _ns
    if _ns is not None:
        return _ns
    with _load_lock:
        if _ns is not None:
            return _ns
        lib = any_core_lib()
        ns, ids = {}, {}
        for i, (name, kind) in enumerate(_entry_table(lib)):
            ns[name] = _make_entry(lib, i, name, kind)
            ids[name] = i
        runners = [(eq, ("forward", "backward_bs")) for eq in RUNNER_EQUATIONS]
        runners += [(eq, ("backward_bs",)) for eq in BACKWARD_RUNNER_EQUATIONS]
        for eq, suffixes in runners:
            for suffix in suffixes:
                cls = ForwardRunner if suffix == "forward" else BackwardRunner
                entry = f"{eq}_{suffix}"
                if entry not in ids:
                    raise RuntimeError(f"sweep core has no entry {entry}, needed by its stepped runner")
                fname = f"{entry}_runner"
                ns[fname] = _make_runner_factory(cls, ids[entry], fname)
        from .sem import INPUTS as _sem_inputs
        ns.update(_sem_inputs)
        ns.update(
            ForwardInput=_abi.ForwardInput,
            BackwardInput=_abi.BackwardInput,
            BoundarySession=BoundarySession,
            ForwardRunner=ForwardRunner,
            BackwardRunner=BackwardRunner,
            _core_stream_for=_core_stream_for,
            visco_acoustic2d_fft_workspace_bytes=visco_acoustic2d_fft_workspace_bytes,
        )
        _ns = ns
        return ns
