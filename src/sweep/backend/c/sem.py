"""The SEM entries of the core (kind 3), reached through ``sweep_sem_call``.

They keep the pybind-era calling convention the SEM drivers were written
against: fill a ``SemForwardInput``, call ``sweep._C.sem2d_acoustic_forward(inp)``
-- or ``(inp, n)`` for the DD phase entries, whose step index rides in ``arg``
-- and get back what the pybind function returned.  That was always a set of
fields the caller had bound (``seis``, the ``grad_*`` buffers), so the core
returns nothing and the wrapper hands those fields back (:data:`RETURNS`).

An input writes every assignment straight into its ctypes twin, so a call
hands the core a pointer instead of re-packing the struct: the DD phase entries
run several times per time step on structs of up to ~230 fields, and packing
per call cost 180-600 us against a step whose kernels take tens of us.  A
tensor's Buf is described once and kept under a weak reference, so the
per-step field rotations of the time loops are one memmove each, and a tensor
lives exactly as long as a field (or the caller) holds it -- as with pybind:
a cache that kept released buffers alive raised the boundary-saving peaks
above the save-all tape's.  Which struct an entry takes comes from the core itself
(``sweep_entry_struct``), so the two cannot drift.
"""
from __future__ import annotations

import ctypes
import operator
import weakref
from ctypes import c_int64

from . import abi as _abi
from .adapt import _check_sem_dtype, _fill_buf, _is_tensor
from .loader import _ERR_CAP, _raise_core

_BUF_SIZE = ctypes.sizeof(_abi.Buf)

# The tuple each pybind function returned: entry -> the bound fields, in order.
# Entries not listed returned None (the DD phase entries write in place).
RETURNS = {
    "sem2d_acoustic_forward": ('seis',),
    "sem2d_acoustic_backward": ('grad_inv_M', 'grad_inv_rho', 'grad_wavelet'),
    "sem2d_elastic_forward": ('seis_x', 'seis_z'),
    "sem2d_elastic_backward": ('grad_lam', 'grad_mu', 'grad_lam2mu', 'grad_inv_M', 'grad_wavelet'),
    "sem2d_acoustic_cpml_forward": ('seis',),
    "sem2d_acoustic_cpml_backward": ('grad_inv_M', 'grad_inv_rho', 'grad_kappa', 'grad_inv_Mg', 'grad_wavelet'),
    "sem2d_elastic_cpml_forward": ('seis_x', 'seis_z'),
    "sem2d_elastic_cpml_backward": ('grad_lam', 'grad_mu', 'grad_lam2mu', 'grad_inv_M', 'grad_w_rho_J', 'grad_wavelet'),
    "sem2d_fluid_solid_forward": ('seis_chi', 'seis_x', 'seis_z'),
    "sem2d_fluid_solid_backward": ('grad_ac_inv_rho', 'grad_ac_inv_M', 'grad_ac_kappa', 'grad_el_lam', 'grad_el_mu', 'grad_el_lam2mu', 'grad_el_inv_M', 'grad_el_w_rho_J', 'grad_wavelet'),
    "sem3d_acoustic_forward": ('seis',),
    "sem3d_acoustic_backward": ('grad_c2_jx', 'grad_c2_jy', 'grad_c2_jz', 'grad_inv_M', 'grad_wavelet'),
    "sem3d_elastic_forward": ('seis_x', 'seis_y', 'seis_z'),
    "sem3d_elastic_backward": ('grad_lam', 'grad_mu', 'grad_lam2mu', 'grad_inv_M', 'grad_wavelet'),
    "sem3d_elastic_vs_forward": ('seis_x', 'seis_y', 'seis_z'),
    "sem3d_elastic_vs_backward": ('grad_lam', 'grad_mu', 'grad_lam2mu', 'grad_inv_M', 'grad_wavelet'),
}

# The pybind docstrings, kept on the wrappers.
DOCS = {
    "sem2d_acoustic_forward": 'SEM 2D acoustic-potential forward (CUDA only, fp32)',
    "sem2d_acoustic_backward": 'SEM 2D acoustic-potential backward (adjoint, save-all; CUDA only, fp32)',
    "sem2d_elastic_forward": 'SEM 2D elastic forward (CUDA only, fp32)',
    "sem2d_elastic_backward": 'SEM 2D elastic backward (adjoint, save-all; CUDA only, fp32)',
    "sem2d_acoustic_cpml_forward": 'SEM 2D acoustic-potential forward with C-PML (CUDA only, fp32)',
    "sem2d_acoustic_cpml_dd_grads": 'SEM2D DD stepped forward, assembly phase',
    "sem2d_acoustic_cpml_dd_step": 'SEM2D DD stepped forward, update phase',
    "sem2d_acoustic_cpml_dd_bwd_pre": 'SEM2D DD stepped backward, pre/assembly phase',
    "sem2d_acoustic_cpml_dd_bwd_mid": 'SEM2D DD stepped backward, transpose phase',
    "sem2d_acoustic_cpml_dd_bwd_end": 'SEM2D DD stepped backward, combine phase',
    "sem2d_elastic_cpml_dd_asm": 'SEM2D elastic DD stepped forward, assembly phase',
    "sem2d_elastic_cpml_dd_step": 'SEM2D elastic DD stepped forward, update phase',
    "sem2d_elastic_cpml_dd_bwd_march_asm": 'SEM2D elastic DD reverse march, assembly phase (bs)',
    "sem2d_elastic_cpml_dd_bwd_march_fin": 'SEM2D elastic DD reverse march, update phase (bs)',
    "sem2d_elastic_cpml_dd_bwd_pre": 'SEM2D elastic DD stepped backward, transpose phase',
    "sem2d_elastic_cpml_dd_bwd_end": 'SEM2D elastic DD stepped backward, combine phase',
    "sem2d_fluid_solid_dd_asm": 'SEM2D fluid-solid DD stepped phase asm',
    "sem2d_fluid_solid_dd_mid": 'SEM2D fluid-solid DD stepped phase mid',
    "sem2d_fluid_solid_dd_step": 'SEM2D fluid-solid DD stepped phase step',
    "sem2d_fluid_solid_dd_bwd_a": 'SEM2D fluid-solid DD stepped phase bwd_a',
    "sem2d_fluid_solid_dd_bwd_b": 'SEM2D fluid-solid DD stepped phase bwd_b',
    "sem2d_fluid_solid_dd_bwd_c": 'SEM2D fluid-solid DD stepped phase bwd_c',
    "sem2d_fluid_solid_dd_bwd_d": 'SEM2D fluid-solid DD stepped phase bwd_d',
    "sem2d_fluid_solid_dd_bwd_e": 'SEM2D fluid-solid DD stepped phase bwd_e',
    "sem2d_fluid_solid_dd_bwd_recon": 'SEM2D fluid-solid DD boundary-saving re-anchor + ring restore',
    "sem2d_fluid_solid_dd_bwd_march_asm": 'SEM2D fluid-solid DD boundary-saving march assembly',
    "sem2d_fluid_solid_dd_bwd_march_fin": 'SEM2D fluid-solid DD boundary-saving march finish',
    "sem2d_acoustic_cpml_backward": 'SEM 2D acoustic-potential backward with C-PML (adjoint; CUDA only, fp32)',
    "sem2d_elastic_cpml_forward": 'SEM 2D elastic forward with C-PML (CUDA only, fp32)',
    "sem2d_elastic_cpml_backward": 'SEM 2D elastic backward with C-PML (adjoint; CUDA only, fp32)',
    "sem2d_fluid_solid_forward": 'SEM 2D coupled fluid-solid forward (CUDA only, fp32)',
    "sem2d_fluid_solid_backward": 'SEM 2D coupled fluid-solid backward (adjoint; CUDA only, fp32)',
    "sem3d_acoustic_forward": 'SEM 3D acoustic forward (Hex8, CUDA only, fp32)',
    "sem3d_acoustic_backward": 'SEM 3D acoustic backward (adjoint, save-all; CUDA only, fp32)',
    "sem3d_acoustic_cpml_dd_asm": 'SEM3D acoustic C-PML DD stepped phase asm',
    "sem3d_acoustic_cpml_dd_step": 'SEM3D acoustic C-PML DD stepped phase step',
    "sem3d_acoustic_cpml_dd_bwd_pre": 'SEM3D acoustic C-PML DD stepped phase bwd_pre',
    "sem3d_acoustic_cpml_dd_bwd_mid": 'SEM3D acoustic C-PML DD stepped phase bwd_mid',
    "sem3d_acoustic_cpml_dd_bwd_end": 'SEM3D acoustic C-PML DD stepped phase bwd_end',
    "sem3d_elastic_forward": 'SEM 3D elastic forward (CUDA only, fp32)',
    "sem3d_elastic_backward": 'SEM 3D elastic backward (adjoint, save-all; CUDA only, fp32)',
    "sem3d_elastic_vs_forward": 'SEM 3D velocity-stress elastic forward (CUDA only, fp32)',
    "sem3d_elastic_vs_backward": 'SEM 3D velocity-stress elastic backward (adjoint; CUDA only, fp32)',
}


class _LiveInput:
    """A SEM input whose fields live in its ctypes twin (``_struct``)."""
    _struct = None
    _fields = ()
    _kinds = {}
    _offsets = {}

    def __init__(self):
        core = self._struct()
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_base", ctypes.addressof(core))
        object.__setattr__(self, "_desc", {})     # id(tensor) -> (weakref to it, its packed Buf)
        object.__setattr__(self, "_device", -1)   # the first CUDA tensor's device: the call's stream
        for name, _, default in self._fields:
            setattr(self, name, default)

    def __setattr__(self, name, value):
        kind = self._kinds.get(name)
        if kind is None:
            # A typo'd field would otherwise be accepted and silently ignored
            # (the pybind class raised).  Underscore names are bookkeeping.
            if name.startswith("_"):
                object.__setattr__(self, name, value)
                return
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        if kind == "Buf":
            self._put_buf(name, value)
        elif kind == "int":
            # What pybind did: an integer (not a float) that fits the C int --
            # ctypes would wrap 2**31 to -2**31 without a word.
            value = operator.index(value)
            if not -2 ** 31 <= value < 2 ** 31:
                raise OverflowError(f"{type(self).__name__}.{name} = {value} does not fit a C int")
            setattr(self._core, name, value)
        else:
            value = float(value)
            setattr(self._core, name, value)
        object.__setattr__(self, name, value)

    def _put_buf(self, name, t):
        dst = self._base + self._offsets[name]
        if t is None:
            ctypes.memset(dst, 0, _BUF_SIZE)
            return
        k = id(t)
        hit = self._desc.get(k)
        if hit is None or hit[0]() is not t:      # new, or an id a dead tensor left behind
            if not _is_tensor(t):
                raise TypeError(f"{type(self).__name__}.{name} takes a tensor, not a {type(t).__name__}")
            _check_sem_dtype(type(self).__name__, name, t)
            b = _abi.Buf()
            _fill_buf(b, t, any_width=True)
            if len(self._desc) > 2 * len(self._offsets) + 8:
                for j in [j for j, (r, _) in self._desc.items() if r() is None]:
                    del self._desc[j]
            hit = self._desc[k] = (weakref.ref(t), bytes(b))
            if self._device < 0 and t.is_cuda:
                object.__setattr__(self, "_device", t.device.index)
        ctypes.memmove(dst, hit[1], _BUF_SIZE)

    def __repr__(self):
        set_ = [f"{n}=Tensor<{tuple(v.shape)}>" for n, k, _ in self._fields
                if k == "Buf" and (v := self.__dict__.get(n)) is not None]
        return f"{type(self).__name__}({', '.join(set_)})"


def _live_class(name: str):
    struct = getattr(_abi, name + "Core")
    fields = _abi.FIELDS[name + "Core"]
    return type(name, (_LiveInput,), {
        "__module__": __name__, "__slots__": (),
        "__doc__": f"SEM input for the core's {name}Core (core/sem_input_core.h).",
        "_struct": struct, "_fields": tuple(fields),
        "_kinds": {n: k for n, k, _ in fields},
        "_offsets": {n: getattr(struct, n).offset for n, k, _ in fields if k == "Buf"}})


#: The SEM input classes under their pybind names, exported on sweep._C.
INPUTS = {n: _live_class(n) for n in _abi.SEM_INPUTS}


def make_sem_entry(lib, entry: int, name: str, stream_scope):
    raw = lib.sweep_entry_struct(entry)
    if raw is None:
        raise RuntimeError(f"sweep core entry {name} is kind 3 but names no input struct")
    cls = INPUTS.get(raw.decode().removesuffix("Core"))
    if cls is None:
        raise RuntimeError(f"sweep core entry {name} takes {raw.decode()}, which sweep.backend.c.abi "
                           "does not mirror; regenerate it from this tree")
    fields = RETURNS.get(name, ())

    def fn(params, arg: int = 0):
        if type(params) is not cls:
            raise TypeError(f"{name} takes a {cls.__name__}, not a {type(params).__name__}")
        stream = 0
        if params._device >= 0:
            import torch
            stream = torch.cuda.current_stream(params._device).cuda_stream
        err = ctypes.create_string_buffer(_ERR_CAP)
        with stream_scope(lib, stream):
            rc = lib.sweep_sem_call(entry, params._base, c_int64(int(arg)), err, _ERR_CAP)
        if rc != 0:
            _raise_core(err, name, rc)
        if not fields:
            return None
        if len(fields) == 1:
            return getattr(params, fields[0])
        return tuple(getattr(params, f) for f in fields)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = DOCS.get(name, f"SEM core entry {entry} via libsweep_core.")
    return fn
