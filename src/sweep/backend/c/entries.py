"""The core entries (one function per ``sweep_call`` entry) and the stream scope.

An entry function takes a ForwardInput / BackwardInput, picks the core whose
compute device holds ``models[0]`` (:func:`_core_for`: the CUDA core for CUDA
tensors, the host core for CPU tensors -- one function serves both), packs
the input (:func:`.adapt.adapt`), sets that core's launch stream from torch's
current stream and calls it.  It returns **nothing**: the drivers allocate no
output, so every buffer the core writes is a tensor the caller bound on the
params object -- ``record_out``, ``u_allt_out`` and ``last_two`` for a
forward, ``grads_out``, ``illum_out`` and ``adcig_out`` for a backward -- and
the caller reads those.  The output struct the C API fills
(``ForwardOutputCore`` and friends) only ever describes those same buffers, so
it is handed to the core and dropped.  A failed call is a ``RuntimeError``
with the core's message, never a crash.
"""
from __future__ import annotations

import ctypes
from ctypes import c_void_p

from . import abi as _abi
from .adapt import _is_tensor, adapt
from .loader import _ERR_CAP, _raise_core, any_core_lib, core_lib, kind_for_device

_KIND_FORWARD, _KIND_BACKWARD, _KIND_RTM, _KIND_SEM = 0, 1, 2, 3

# Per entry kind: the input struct ``adapt`` packs and the output struct the
# core fills (dropped: see the module docstring).
_KINDS = {
    _KIND_FORWARD: ("ForwardInputCore", _abi.ForwardOutputCore),
    _KIND_BACKWARD: ("BackwardInputCore", _abi.BackwardOutputCore),
    _KIND_RTM: ("BackwardInputCore", _abi.RTMOutputCore),
}


# ---------------------------------------------------------------------------
# Core selection + stream scope: the core launches on its thread-local stream,
# set per call from torch's current stream so ``with torch.cuda.stream(s)`` is
# honoured; the host core stores and ignores it.
# ---------------------------------------------------------------------------

def _stream_for_models(models) -> int:
    """torch's current stream on ``models[0]``'s device; 0 (the legacy default
    stream) for host models or none.  A single tensor in place of the list is
    refused here, before ``models[0]`` silently means its first row."""
    if _is_tensor(models):
        raise RuntimeError("models takes a list of tensors, not a single tensor; wrap it: [models]")
    if models and models[0].is_cuda:
        import torch
        return torch.cuda.current_stream(models[0].device).cuda_stream
    return 0


def _core_for(models, what: str):
    """``(lib, stream)`` for a call on ``models``: the core whose compute device
    holds ``models[0]`` (CUDA tensors -> the CUDA core, CPU tensors -> the host
    core; anything else is refused by name) and the stream it launches on.  A
    call without models (none bound) goes to whichever core loads."""
    stream = _stream_for_models(models)
    if not models:
        return any_core_lib(), stream
    device = models[0].device
    try:
        kind = kind_for_device(device)
    except RuntimeError as exc:
        raise RuntimeError(f"{what}: models[0] is a {device} tensor; {exc}") from None
    # The core is picked from models[0]; the kernels read every model's raw
    # pointer, and some equations bind them with a count check only, so a
    # model on another device would be dereferenced by the wrong core.
    for i, m in enumerate(models):
        if m.device != device:
            raise RuntimeError(f"{what}: models[{i}] is on {m.device} but models[0] is on {device}; "
                               "all models must live on one device")
    try:
        lib = core_lib(kind)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{what}: models[0] is a {device} tensor and the {kind} core cannot be loaded: {exc}") from None
    return lib, stream


def _stream_for_device(index: int) -> int:
    if index >= 0:
        import torch
        return torch.cuda.current_stream(index).cuda_stream
    return 0


class _StreamScope:
    __slots__ = ("_lib", "_s", "_prev")

    def __init__(self, lib, stream: int):
        self._lib, self._s = lib, stream

    def __enter__(self):
        self._prev = self._lib.sweep_get_stream()
        self._lib.sweep_set_stream(c_void_p(self._s))
        return self

    def __exit__(self, *exc):
        self._lib.sweep_set_stream(c_void_p(self._prev))
        return False


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def _entry_table(lib):
    table = []
    for i in range(lib.sweep_entry_count()):
        name = lib.sweep_entry_name(i)
        if name is None:
            raise RuntimeError(f"sweep core entry table is short at {i}")
        table.append((name.decode(), lib.sweep_entry_kind(i)))
    return table


def _make_entry(lib, entry: int, name: str, kind: int):
    if kind == _KIND_SEM:
        # SEM entries stay on the core the table was read from (the host core
        # builds no SEM; its sweep_sem_call refuses by name).
        from .sem import make_sem_entry
        return make_sem_entry(lib, entry, name, _StreamScope)
    if kind not in _KINDS:
        raise RuntimeError(f"sweep core entry {name} has unknown kind {kind}")
    in_name, out_type = _KINDS[kind]

    def fn(params) -> None:
        # The core is chosen per call: the same ``acoustic2d_forward`` serves
        # CUDA and CPU models.  Both cores export the same table (the loader
        # checks), so ``entry`` indexes either.
        lib, stream = _core_for(params.models, name)
        with _StreamScope(lib, stream):
            core, keep = adapt(in_name, params)
            out = out_type()
            err = ctypes.create_string_buffer(_ERR_CAP)
            # sweep_call looked up per call, not captured: a test may swap it
            # on the library object, and the lookup is nothing next to adapt().
            rc = lib.sweep_call(entry, ctypes.addressof(core), ctypes.addressof(out), err, _ERR_CAP)
            if rc != 0:
                _raise_core(err, name, rc)
        # core/keep/out die with the frame: what the core wrote is in the
        # tensors the caller bound on ``params``.

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = (f"Core entry {entry} ({('forward', 'backward', 'rtm')[kind]}) via libsweep_core "
                  "(the CUDA or the host core, by models[0].device); returns None, the outputs "
                  "are the tensors bound on the params object.")
    return fn


# ---------------------------------------------------------------------------
# Small entries
# ---------------------------------------------------------------------------

def visco_acoustic2d_fft_workspace_bytes(B, nz, nx, device=None) -> int:
    """Work-area bytes of the visco-acoustic spectral step's cuFFT plan for a
    (B, 1, nz, nx) grid on the current device.  0 from a core that does not
    build visco_acoustic2d (the host core)."""
    lib = core_lib(kind_for_device(device)) if device is not None else any_core_lib()
    return int(lib.sweep_visco_fft_workspace_bytes(int(B), int(nz), int(nx)))


def _core_stream_for(params) -> int:
    """Test probe: the stream handle the core launches on for this input.  Asks
    the core already loaded for the models' device, else whichever is loaded:
    the probe is about the stream, it must not pull in (or build) a core."""
    from . import loader
    stream = _stream_for_models(params.models)
    kind = loader.kind_for_device(params.models[0].device) if params.models else None
    lib = loader._libs.get(kind) if kind else None
    if lib is None:
        lib = any_core_lib()
    with _StreamScope(lib, stream):
        return int(lib.sweep_get_stream() or 0)
