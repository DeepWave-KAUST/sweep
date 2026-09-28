"""The stepped runners (the DD time loop) and the boundary session.

A runner is one persistent core handle over a ForwardInput / BackwardInput
packed once at construction: the core struct, everything it points at and the
tensors it describes stay alive for the handle's life, so a segment call
(``run``) costs no re-adaptation.  Like the per-call entries, ``run`` returns
**nothing**: the core writes into the tensors bound on the params object at
construction (``record_out`` / ``u_allt_out`` for a forward runner,
``grads_out`` / ``illum_out`` for a backward one), and the caller reads them.
"""
from __future__ import annotations

import ctypes
from ctypes import c_void_p

from . import abi as _abi
from .adapt import adapt
from .entries import _launch_stream, _stream_for_device, _StreamScope
from .loader import _ERR_CAP, _raise_core, core_lib


class _Runner:
    """One core runner handle.  The core struct, everything it points at and
    the tensors it describes stay alive for the handle's life -- what the
    core writes goes into the buffers bound on the params object at
    construction, whatever the object is later set to."""
    _side = ""          # "forward" | "backward"
    _in_name = ""
    _out_type = None

    def __init__(self, entry_id: int, params):
        self._h = None
        self._lib = lib = core_lib()
        self._params = params
        self._core, self._keep = adapt(self._in_name, params)
        self._run = getattr(lib, f"sweep_{self._side}_runner_run")
        self._device_index = getattr(lib, f"sweep_{self._side}_runner_device_index")
        self._destroy = getattr(lib, f"sweep_{self._side}_runner_destroy")
        h = c_void_p()
        err = ctypes.create_string_buffer(_ERR_CAP)
        with _StreamScope(lib, _launch_stream(params, f"{self._side} runner create")):
            rc = getattr(lib, f"sweep_{self._side}_runner_create")(
                int(entry_id), ctypes.addressof(self._core), ctypes.byref(h), err, _ERR_CAP)
        if rc != 0:
            _raise_core(err, f"{self._side} runner create", rc)
        self._h = h

    def _run_segment(self, a: int, b: int, step_phase: int) -> None:
        h = self._h
        if h is None:
            raise RuntimeError(f"{self._side} runner is closed")
        out = self._out_type()          # filled by the core, dropped: it describes the bound tensors
        err = ctypes.create_string_buffer(_ERR_CAP)
        with _StreamScope(self._lib, _stream_for_device(self._device_index(h))):
            rc = self._run(h, int(a), int(b), int(step_phase), ctypes.addressof(out), err, _ERR_CAP)
        if rc != 0:
            _raise_core(err, f"{self._side} runner run", rc)

    def device_index(self) -> int:
        return -1 if self._h is None else self._device_index(self._h)

    def close(self) -> None:
        h, self._h = self._h, None
        if h is not None and h.value:
            self._destroy(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ForwardRunner(_Runner):
    _side = "forward"
    _in_name = "ForwardInputCore"
    _out_type = _abi.ForwardOutputCore

    def run(self, it_begin: int, it_end: int, step_phase: int) -> None:
        """Steps ``[it_begin, it_end)`` (``step_phase`` as the stepped
        contract spells it); the record lands in the bound ``record_out``."""
        self._run_segment(it_begin, it_end, step_phase)


class BackwardRunner(_Runner):
    _side = "backward"
    _in_name = "BackwardInputCore"
    _out_type = _abi.BackwardOutputCore

    def run(self, bw_it_begin: int, bw_it_end: int, step_phase: int) -> None:
        """The reverse segment ``[bw_it_end, bw_it_begin)``; the gradients
        accumulate into the bound ``grads_out`` (and ``illum_out``)."""
        self._run_segment(bw_it_begin, bw_it_end, step_phase)


def _make_runner_factory(cls, entry_id: int, name: str):
    def factory(params):
        return cls(entry_id, params)
    factory.__name__ = factory.__qualname__ = name
    factory.__doc__ = f"Persistent {cls.__name__} on core entry {entry_id}."
    return factory


class BoundarySession:
    """Persistent boundary-staging session: Python owns it and hands the SAME
    object to every per-step call of a DD time loop, so the copy stream and
    the ring events survive between steps.

    ``handle`` is the ``c_void_p`` the core handed out; ``close()`` destroys it
    exactly once (idempotent, and what ``__del__`` does), after which
    ``handle`` is None and ``finish()`` / ``used()`` -- and ``adapt()`` of an
    input that still names this session -- raise instead of passing NULL to
    the core."""

    def __init__(self):
        self._lib = lib = core_lib()
        self.handle = c_void_p(lib.sweep_session_create())

    @property
    def closed(self) -> bool:
        h = getattr(self, "handle", None)
        return h is None or not h.value

    def _live(self):
        """The handle, or the closed-session error."""
        h = getattr(self, "handle", None)
        if h is None or not h.value:
            raise RuntimeError("BoundarySession is closed")
        return h

    def finish(self) -> None:
        """Let every outstanding copy land and close the current phase."""
        h = self._live()
        err = ctypes.create_string_buffer(_ERR_CAP)
        rc = self._lib.sweep_session_finish(h, err, _ERR_CAP)
        if rc != 0:
            _raise_core(err, "BoundarySession.finish", rc)

    def used(self) -> bool:
        """False when no call site ever bound this session."""
        h = self._live()
        return bool(self._lib.sweep_session_used(h))

    def close(self) -> None:
        """Destroy the core session (once); safe to call again, and on a
        session whose construction failed."""
        h = getattr(self, "handle", None)
        if h is not None and h.value:
            self._lib.sweep_session_destroy(h)      # the very c_void_p the core handed out
        self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
