"""sweep's CUDA backend (``impl='c'``): the pure-Python ctypes layer over the
prebuilt ``libsweep_core.so`` (csrc/core/capi.h).

This replaces the compiled pybind shim (csrc/bindings/module.cpp): the same
Python surface -- one function per core entry, the stepped runners, the
boundary session -- reached through the plain C API with ctypes, so after
``pip install`` nothing compiles and no torch version is baked in.  Layout of
the C structs comes from the generated mirror (:mod:`.abi`, contract A); the
behaviour on top (contract B) is split by concern:

* :mod:`.jit` -- where the core comes from: the shipped core's fit rules, the
  cached local core, the nvcc build, and the ``SWEEP_JIT_FULL`` pybind
  developer path;
* :mod:`.loader` -- ``core_lib()``: dlopen, the C signatures, the ABI guard;
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
from .loader import _load_lock, core_lib
from .runners import BackwardRunner, BoundarySession, ForwardRunner, _make_runner_factory

# The equations with a persistent stepped runner in the core; the rest must
# keep ``getattr(_C, f"{name}_forward_runner", None) is None`` so the
# propagator falls back to the per-call stepped path.
RUNNER_EQUATIONS = (
    "acoustic2d", "acoustic3d", "acoustic_vrz2d", "elastic2d", "elastic3d",
    "das_mu2d", "das_mu3d", "elastic_tti_sg2d", "elastic_tti_sg3d", "elastic_vr2d",
)

_ns = None


def namespace() -> dict:
    """Everything ``sweep._C`` exposes, keyed by its old pybind name (cached;
    built once under the loader's ``_load_lock``)."""
    global _ns
    if _ns is not None:
        return _ns
    with _load_lock:
        if _ns is not None:
            return _ns
        lib = core_lib()
        ns, ids = {}, {}
        for i, (name, kind) in enumerate(_entry_table(lib)):
            ns[name] = _make_entry(lib, i, name, kind)
            ids[name] = i
        for eq in RUNNER_EQUATIONS:
            for suffix, cls in (("forward", ForwardRunner), ("backward_bs", BackwardRunner)):
                entry = f"{eq}_{suffix}"
                if entry not in ids:
                    raise RuntimeError(f"sweep core has no entry {entry}, needed by its stepped runner")
                fname = f"{entry}_runner"
                ns[fname] = _make_runner_factory(cls, ids[entry], fname)
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
