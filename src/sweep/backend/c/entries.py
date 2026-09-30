"""The core entries (one function per ``sweep_call`` entry) and the stream scope.

An entry function takes a ForwardInput / BackwardInput, packs it
(:func:`.adapt.adapt`), sets the core's launch stream from torch's current
stream for the models' device and calls the core.  It returns **nothing**:
the drivers allocate no output, so every buffer the core writes is a tensor
the caller bound on the params object -- ``record_out``, ``u_allt_out`` and
``last_two`` for a forward, ``grads_out``, ``illum_out`` and ``adcig_out`` for
a backward -- and the caller reads those.  The output struct the C API fills
(``ForwardOutputCore`` and friends) only ever describes those same buffers, so
it is handed to the core and dropped.  A failed call is a ``RuntimeError``
with the core's message, never a crash.
"""
from __future__ import annotations

import ctypes
from ctypes import c_void_p

from . import abi as _abi
from .adapt import _is_tensor, adapt
from .loader import _ERR_CAP, _raise_core, core_lib

_KIND_FORWARD, _KIND_BACKWARD, _KIND_RTM = 0, 1, 2

# Per entry kind: the input struct ``adapt`` packs and the output struct the
# core fills (dropped: see the module docstring).
_KINDS = {
    _KIND_FORWARD: ("ForwardInputCore", _abi.ForwardOutputCore),
    _KIND_BACKWARD: ("BackwardInputCore", _abi.BackwardOutputCore),
    _KIND_RTM: ("BackwardInputCore", _abi.RTMOutputCore),
}


# ---------------------------------------------------------------------------
# Stream scope: the core launches on its thread-local stream, set per call
# from torch's current stream so ``with torch.cuda.stream(s)`` is honoured.
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


def _launch_stream(params, what: str) -> int:
    """``_stream_for_models`` for a real core call: ``models[0]`` must be a
    CUDA tensor.  The core dereferences every model on the device, so a host
    pointer there is a crash (or silent garbage), not a message."""
    models = params.models
    stream = _stream_for_models(models)
    if models and not models[0].is_cuda:
        raise RuntimeError(
            f"{what}: models[0] is a {models[0].device} tensor; the CUDA core expected "
            "a CUDA tensor (move the models to the device first)")
    return stream


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
    if kind not in _KINDS:
        raise RuntimeError(f"sweep core entry {name} has unknown kind {kind}")
    in_name, out_type = _KINDS[kind]

    def fn(params) -> None:
        with _StreamScope(lib, _launch_stream(params, name)):
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
    fn.__doc__ = (f"Core entry {entry} ({('forward', 'backward', 'rtm')[kind]}) via libsweep_core; "
                  "returns None, the outputs are the tensors bound on the params object.")
    return fn


# ---------------------------------------------------------------------------
# Small entries
# ---------------------------------------------------------------------------

def visco_acoustic2d_fft_workspace_bytes(B, nz, nx) -> int:
    """Work-area bytes of the visco-acoustic spectral step's cuFFT plan for a
    (B, 1, nz, nx) grid on the current device."""
    return int(core_lib().sweep_visco_fft_workspace_bytes(int(B), int(nz), int(nx)))


def _core_stream_for(params) -> int:
    """Test probe: the stream handle the core launches on for this input."""
    lib = core_lib()
    with _StreamScope(lib, _stream_for_models(params.models)):
        return int(lib.sweep_get_stream() or 0)
