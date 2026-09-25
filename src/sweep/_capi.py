"""The ctypes shim over ``libsweep_core.so`` (csrc/core/capi.h).

This replaces the compiled pybind shim (csrc/bindings/module.cpp): the same
Python surface -- one function per core entry, the stepped runners, the
boundary session -- reached through the plain C API with ctypes, so after
``pip install`` nothing compiles and no torch version is baked in.  The only
things asked of a tensor are ``data_ptr()``, ``shape``, ``stride()``,
``dtype``, ``device``, ``element_size()`` and ``_version``, which every torch
has.

Layout of the C structs comes from the generated ``sweep._core_abi`` (contract
A); this module is the behaviour on top (contract B): the tensor -> Buf
adapter (the twin of cuda/common/buf_torch.h ``buf_of`` and
cuda/common/adapt_inputs.h ``InputArena``), the output mapping back to the
tensors the propagator bound (``to_torch``), the per-call stream scope, and
the ABI guard that refuses a core whose struct layout is not the mirror's.

Nothing here owns device memory: a Buf is a view of a tensor the caller keeps
alive, and every host-side array a struct points at sits in the ``keepalive``
list returned next to it.  The host copies the drivers read and the Buf arrays
of the list fields are cached on the params object (``_capi_host_cache``, see
``adapt``) so a per-call path does not redo a device->host sync for an index
tensor nobody touched.  Only ``core_lib()`` touches the filesystem; the module
imports without torch and without a GPU.
"""
from __future__ import annotations

import ctypes
import math
import threading
from ctypes import POINTER, c_char_p, c_float, c_int, c_int32, c_int64, c_size_t, c_void_p

from . import _core_abi as _abi

_ERR_CAP = 4096          # the message buffer every fallible C call fills
_BUF_MAX_DIMS = 8        # csrc/core/buf.h BUF_MAX_DIMS, mirrored by Buf.sizes_
_KIND_FORWARD, _KIND_BACKWARD, _KIND_RTM = 0, 1, 2

# Byte width each storage tag implies (core/buf.h buf_dtype_element_size);
# buf_of() refuses a tensor whose element_size() disagrees, so a float64 or
# int64 buffer never reaches the core mislabelled.
_TAG_WIDTH = {_abi.FP32: 4, _abi.FP16: 2, _abi.BF16: 2, _abi.INT8: 1}

# The equations with a persistent stepped runner in the core; the rest must
# keep ``getattr(_C, f"{name}_forward_runner", None) is None`` so the
# propagator falls back to the per-call stepped path.
RUNNER_EQUATIONS = (
    "acoustic2d", "acoustic3d", "acoustic_vrz2d", "elastic2d", "elastic3d",
    "das_mu2d", "das_mu3d", "elastic_tti_sg2d", "elastic_tti_sg3d", "elastic_vr2d",
)

# The attribute adapt() caches under on a params object (the generated input
# classes admit underscore names; an object that refuses them is not cached).
_CACHE_ATTR = "_capi_host_cache"

_lib = None
_ns = None
_dtype_tags = None
# Guards the first load (core_lib) and the namespace build (namespace): both
# happen on a propagator's first use, which several threads may reach at once
# (per-GPU worker threads, a DataLoader worker).  Re-entrant because
# namespace() holds it while it calls core_lib().
_load_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Loading the core
# ---------------------------------------------------------------------------

def _declare(lib, path: str = "<core>") -> None:
    """argtypes/restypes for every C function, so ctypes never guesses a width
    (a size_t/int64 return read as int truncates silently on the wrong guess).
    A core missing one of them predates this shim's ABI: refused by name."""
    def sig(name, argtypes, restype):
        try:
            f = getattr(lib, name)
        except AttributeError:
            raise RuntimeError(
                f"{path} has no {name}: the core predates this shim's C API "
                f"(ABI {_abi.ABI_VERSION}); rebuild it from this tree") from None
        f.argtypes = argtypes
        f.restype = restype

    sig("sweep_core_abi_version", [], c_int)
    sig("sweep_entry_count", [], c_int)
    sig("sweep_entry_name", [c_int], c_char_p)
    sig("sweep_entry_kind", [c_int], c_int)
    sig("sweep_call", [c_int, c_void_p, c_void_p, c_char_p, c_int], c_int)
    for side in ("forward", "backward"):
        sig(f"sweep_{side}_runner_create", [c_int, c_void_p, POINTER(c_void_p), c_char_p, c_int], c_int)
        sig(f"sweep_{side}_runner_run", [c_void_p, c_int, c_int, c_int, c_void_p, c_char_p, c_int], c_int)
        sig(f"sweep_{side}_runner_device_index", [c_void_p], c_int)
        sig(f"sweep_{side}_runner_destroy", [c_void_p], None)
    sig("sweep_session_create", [], c_void_p)
    sig("sweep_session_finish", [c_void_p, c_char_p, c_int], c_int)
    sig("sweep_session_used", [c_void_p], c_int)
    sig("sweep_session_destroy", [c_void_p], None)
    sig("sweep_set_stream", [c_void_p], None)
    sig("sweep_get_stream", [], c_void_p)
    sig("sweep_visco_fft_workspace_bytes", [c_int64, c_int64, c_int64], c_size_t)
    sig("sweep_layout_count", [], c_int64)
    sig("sweep_layout_probe", [c_int64], c_int64)


def _probe_name(i: int) -> str:
    """What LAYOUT_PROBE[i] measures, for an error message."""
    name, member = _abi.LAYOUT_PROBE[i]
    if name in getattr(_abi, "ENUMS", {}):
        return f"enumerator {name}::{member}"
    if member is None:
        return f"sizeof({name})"
    return f"offsetof({name}, {member})"


def _check_abi(lib, path: str) -> None:
    """The ABI guard.  The version number catches a deliberate break; the
    layout probe catches the silent one -- a field added to a core struct
    without regenerating the mirror would otherwise shift every later field
    and the core would read garbage from a struct that looked right, and a
    reordered BoundaryDtype would mis-tag every boundary buffer.  The mirror's
    side of every probe entry is ``_core_abi.ctypes_layout()``, so an entry
    kind this module never heard of (the enum probes) still compares."""
    got = lib.sweep_core_abi_version()
    if got != _abi.ABI_VERSION:
        raise RuntimeError(
            f"sweep core ABI {got} != the shim's ABI {_abi.ABI_VERSION} ({path}); "
            "the core and sweep._core_abi were built from different trees")
    probe = _abi.LAYOUT_PROBE
    expect = list(_abi.ctypes_layout())
    n = int(lib.sweep_layout_count())
    if n != len(probe) or len(expect) != len(probe):
        raise RuntimeError(
            f"sweep core layout probe has {n} entries, the shim's mirror has "
            f"{len(probe)} (ctypes_layout answers {len(expect)}) ({path}); regenerate "
            "sweep._core_abi or rebuild the core from the same tree")
    for i in range(len(probe)):
        val = int(lib.sweep_layout_probe(i))
        if val != expect[i]:
            name, member = probe[i]
            raise RuntimeError(
                f"sweep core/shim layout mismatch at probe {i} ({name}, {member}): "
                f"{_probe_name(i)} expected {expect[i]}, got {val} ({path}); regenerate "
                "the mirror or rebuild the core from the same tree")


def core_lib():
    """The loaded core (cached, one per process).

    Thread-safe: the first load (dlopen, ``_declare``, the ABI guard) runs
    under ``_load_lock`` and later callers get the cached ``ctypes.CDLL``,
    which is safe to share across threads -- ctypes releases the GIL around
    every core call, and the only per-thread state the core keeps (its launch
    stream, ``sweep_set_stream``) is thread-local on the C side.

    Opened ``RTLD_GLOBAL``.  That is not what makes cuFFT or the driver
    resolve: the core links cudart statically under ``-fvisibility=hidden``
    (``nm -D`` on the shipped core exports no ``cuda*`` symbol at all, and its
    NEEDED list is ``libcufft.so.11`` plus the system libraries -- no
    libcudart, no libcuda), so cuFFT comes through NEEDED/RUNPATH (or the
    ``_preload_cufft`` seat) and the driver through cudart's own dlopen of
    ``libcuda.so.1``, both inside the core's own scope whichever mode it is
    opened with.  What ``RTLD_GLOBAL`` does is put the core's ``sweep_*``
    exports in the process-global lookup scope, so anything loaded after it
    that references them by name -- the ``SWEEP_JIT_FULL`` pybind shim
    (linked ``-lsweep_core``) in the same process, a ``ctypes.CDLL(None)``
    lookup, a profiler's hook -- binds to this one copy.
    """
    global _lib
    if _lib is not None:
        return _lib
    with _load_lock:
        if _lib is not None:
            return _lib
        from . import _jit
        path = str(_jit.core_path())
        # cuFFT: the core's RUNPATH reaches the pip nvidia-cufft wheel only
        # from an installed layout (a custom SWEEP_CORE, a moved
        # site-packages do not).  Seat the wheel's copy first (silent when
        # there is none); a toolkit is looked for only if the load then fails
        # on libcufft, so the common path never probes for nvcc.
        _jit._preload_cufft(None)
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            if "libcufft" not in str(exc):
                raise
            _jit._preload_cufft(_jit._find_cuda_home())
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _declare(lib, path)
        _check_abi(lib, path)
        _lib = lib
        return lib


def _raise_core(err, what: str, rc: int):
    msg = err.value.decode("utf-8", "replace")
    raise RuntimeError(msg or f"{what} failed (rc={rc}) without a message")


# ---------------------------------------------------------------------------
# tensor -> Buf (the twin of buf_torch.h / adapt_inputs.h)
# ---------------------------------------------------------------------------

def _tags():
    # torch is imported here, not at module import: the tag table is the only
    # torch object the adapter needs, and it is built once.
    global _dtype_tags
    if _dtype_tags is None:
        import torch
        _dtype_tags = {torch.uint8: _abi.INT8, torch.float16: _abi.FP16, torch.bfloat16: _abi.BF16}
    return _dtype_tags


def _is_tensor(x) -> bool:
    return hasattr(x, "data_ptr") and hasattr(x, "shape")


def _fill_buf(b, t) -> None:
    """Describe ``t`` into ``b``, a ZEROED Buf (fresh struct memory): None stays
    the undefined Buf (defined_ False, everything zero), exactly buf_of()."""
    if t is None:
        return
    shape = t.shape
    nd = len(shape)
    if nd > _BUF_MAX_DIMS:
        raise RuntimeError(
            f"buf_of: tensor has {nd} dimensions but Buf carries at most {_BUF_MAX_DIMS}; "
            "raise BUF_MAX_DIMS in csrc/core/buf.h.")
    tag = _tags().get(t.dtype, _abi.FP32)
    es = t.element_size()
    if es != _TAG_WIDTH[tag]:
        raise RuntimeError(
            f"buf_of: a {t.dtype} tensor is {es} bytes wide but the boundary storage tag "
            f"implies {_TAG_WIDTH[tag]}; the boundary layer stores float32, float16, "
            "bfloat16 or uint8.")
    dev = t.device
    cuda = dev.type == "cuda"
    b.defined_ = True
    b.ndim_ = nd
    b.numel_ = math.prod(shape)
    b.elem_size_ = es
    b.dtype_ = tag
    b.is_cuda_ = cuda
    b.device_ = int(dev.index) if cuda else -1
    b.data_ = t.data_ptr()
    # Sizes and strides verbatim, the nonzero stride of a 0-size dim included
    # (a DD cut face; Buf::stride documents why it must not be recomputed).
    strides = t.stride()
    sizes_, strides_ = b.sizes_, b.strides_
    for i in range(nd):
        sizes_[i] = shape[i]
        strides_[i] = strides[i]


def _host_int32(t):
    import torch
    return t.to("cpu", torch.int32).contiguous()


def _host_int32_of_list(what: str, xs):
    """A plain list/tuple (or anything iterable of ints) as the int32 host
    tensor the driver reads; a scalar is refused rather than made 0-d."""
    import torch
    try:
        xs = list(xs)
    except TypeError:
        raise RuntimeError(f"{what}: expected a tensor or a list of ints, got {type(xs).__name__}") from None
    return torch.as_tensor([int(x) for x in xs], dtype=torch.int32)


# -- the per-params cache ---------------------------------------------------
#
# The per-call path (every non-runner entry, and the cpu/disk-staged DD steps
# the persistent runner contract excludes) hands the same params object to
# adapt() call after call.  Its host copies (source_field_indices,
# receiver_field_indices, checkpoint_steps: each a device->host sync) and the
# ctypes Buf arrays of its list fields are kept on the object under
# _CACHE_ATTR, keyed by field name, and reused only while the source is the
# SAME tensor object (identity; the cache holds a reference, so the id cannot
# be recycled) with an unchanged ``_version`` (every in-place op bumps it,
# metadata-only ones like resize_/transpose_/set_ included, and a view shares
# its base's counter).  A replaced tensor is a different object: a miss.  Not
# locked: a params object is one call site's, not shared between threads.

def _cache_of(obj):
    """The cache dict on ``obj``; None when the object refuses new attributes
    (the adapter then just works uncached)."""
    c = getattr(obj, _CACHE_ATTR, None)
    if c is None:
        c = {}
        try:
            setattr(obj, _CACHE_ATTR, c)
        except (AttributeError, TypeError):
            return None
    return c


def _host_copy(cache, name: str, t):
    """The int32 host copy of tensor ``t`` for field ``name``: cached as
    (source tensor, its _version, the copy)."""
    ver = getattr(t, "_version", None)
    if cache is not None:
        hit = cache.get(name)
        if hit is not None and hit[0] is t and hit[1] == ver:
            return hit[2]
    h = _host_int32(t)
    if cache is not None:
        cache[name] = (t, ver, h)
    return h


def _buf_array(cache, what: str, name: str, ts):
    """The ``Buf * n`` array describing the tensors ``ts`` (None entries stay
    undefined Bufs): cached as (key, array, the tensors), the key being each
    tensor's (id, data_ptr, _version)."""
    key = []
    for i, t in enumerate(ts):
        if t is None:
            key.append(None)
        elif _is_tensor(t):
            key.append((id(t), t.data_ptr(), getattr(t, "_version", None)))
        else:
            raise RuntimeError(f"{what}.{name}[{i}] is a {type(t).__name__}, not a tensor")
    key = tuple(key)
    if cache is not None:
        hit = cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
    arr = (_abi.Buf * len(ts))()
    for i, t in enumerate(ts):
        _fill_buf(arr[i], t)
    if cache is not None:
        cache[name] = (key, arr, list(ts))
    return arr


def _session_handle(what: str, name: str, val):
    """The raw core pointer for a BoundarySession* field: a BoundarySession, a
    raw handle (int / c_void_p), or None.  A closed session is refused here --
    the core would otherwise be handed NULL for a session the caller thinks
    is live."""
    if val is None:
        return None
    if isinstance(val, BoundarySession) or hasattr(val, "handle"):
        h = val.handle
        h = h.value if isinstance(h, c_void_p) else h
        if not h:
            raise RuntimeError(f"{what}.{name}: BoundarySession is closed")
        return int(h)
    h = val.value if isinstance(val, c_void_p) else val
    return int(h) if h else None


def adapt(struct_name: str, obj):
    """Build the core struct for ``obj`` (a ForwardInput / BackwardInput).

    Returns ``(core_struct, keepalive)``: every host array the struct points
    at, every host copy the drivers read (``source_field_indices`` and friends,
    the ``.to(cpu)`` InputArena made once at entry) and every tensor described
    lives in ``keepalive`` -- the struct must not outlive that list.  The host
    copies and the Buf arrays are also cached on ``obj`` (``_capi_host_cache``,
    see above) and reused while their sources are unchanged.
    """
    core = getattr(_abi, struct_name)()
    keep = []
    cache = _cache_of(obj)
    for name, kind, default in _abi.FIELDS[struct_name]:
        val = getattr(obj, name, default)
        if kind == "Buf":
            if val is not None and name in _abi.HOST_BUF:
                # A tensor the drivers read on the host: an int32 host copy
                # (what InputArena made); a plain list is made one directly.
                if _is_tensor(val):
                    val = _host_copy(cache, name, val)
                else:
                    val = _host_int32_of_list(f"{struct_name}.{name}", val)
            if val is not None:
                keep.append(val)
            _fill_buf(getattr(core, name), val)
        elif kind == "BufList":
            if val is None:
                ts = []
            elif _is_tensor(val):
                raise RuntimeError(
                    f"{struct_name}.{name} takes a list of tensors, not a single "
                    f"{type(val).__name__}; wrap it: [{name}]")
            else:
                ts = list(val)
            n = len(ts)
            if n:
                arr = _buf_array(cache, struct_name, name, ts)
                keep.append(arr)
                keep.extend(ts)
                setattr(core, name, _abi.BufList(p=ctypes.cast(arr, POINTER(_abi.Buf)), n=n))
        elif kind == "IntSpan":
            if name in _abi.HOST_INT_SPAN and _is_tensor(val):
                # A tensor the drivers read on the host: one int32 host copy
                # (what InputArena::host_ints_of made), cached; empty -> n=0.
                if math.prod(val.shape) > 0:
                    h = _host_copy(cache, name, val)
                    keep.append(h)
                    setattr(core, name, _abi.IntSpan(
                        p=ctypes.cast(c_void_p(h.data_ptr()), POINTER(c_int32)),
                        n=math.prod(h.shape)))
            else:
                xs = [int(x) for x in (val if val is not None else ())]
                if xs:
                    arr = (c_int32 * len(xs))(*xs)
                    keep.append(arr)
                    setattr(core, name, _abi.IntSpan(p=ctypes.cast(arr, POINTER(c_int32)), n=len(xs)))
        elif kind == "FloatSpan":
            xs = [float(x) for x in (val if val is not None else ())]
            if xs:
                arr = (c_float * len(xs))(*xs)
                keep.append(arr)
                setattr(core, name, _abi.FloatSpan(p=ctypes.cast(arr, POINTER(c_float)), n=len(xs)))
        elif kind == "CStrList":
            bs = [s if isinstance(s, bytes) else str(s).encode()
                  for s in (val if val is not None else ())]
            if bs:
                arr = (c_char_p * len(bs))(*bs)
                keep.append(bs)
                keep.append(arr)
                setattr(core, name, _abi.CStrList(p=ctypes.cast(arr, POINTER(c_char_p)), n=len(bs)))
        elif kind in ("int", "unsigned int", "int64_t"):
            setattr(core, name, int(val))
        elif kind == "bool":
            setattr(core, name, bool(val))
        elif kind in ("float", "double"):
            setattr(core, name, float(val))
        elif kind == "BoundarySession*":
            setattr(core, name, _session_handle(struct_name, name, val))
        else:
            raise RuntimeError(f"sweep._core_abi: field {struct_name}.{name} has unknown kind {kind!r}")
    return core, keep


# ---------------------------------------------------------------------------
# Output Buf -> the tensor the propagator bound (adapt_inputs.h to_torch)
# ---------------------------------------------------------------------------

def _tensor_of(b, singles, lists, what: str):
    """The drivers allocate nothing, so an output Buf is one of the bound
    tensors: find it by pointer identity.  Undefined -> None."""
    if not b.defined_:
        return None
    ptr = b.data_ or 0
    for t in singles:
        if t is not None and t.data_ptr() == ptr:
            return t
    for lst in lists:
        for t in lst:
            if t is not None and t.data_ptr() == ptr:
                return t
    raise RuntimeError(f"{what}: the driver handed back a buffer the propagator did not bind")


def _forward_candidates(p):
    return (p.u_allt_out, p.last_two, p.record_out)


def _backward_candidates(p):
    return (list(p.grads_out or ()), list(p.illum_out or ()), p.adcig_out)


def _map_forward(out, cands):
    u_allt, last_two, record = cands
    return (_tensor_of(out.wavefield, (u_allt,), (), "ForwardOutput.wavefield"),
            _tensor_of(out.last_two, (last_two,), (), "ForwardOutput.last_two"),
            _tensor_of(out.record, (record,), (), "ForwardOutput.record"))


def _map_backward(out, cands):
    grads_out, illum_out, adcig_out = cands
    items = out.grads.items
    grads = [_tensor_of(items[i], (), (grads_out,), "BackwardOutput.grads") for i in range(out.grads.n)]
    # The leading [] is the legacy checkpoints list the pybind tuple carried.
    return ([], grads,
            _tensor_of(out.source_illumination, (), (illum_out,), "BackwardOutput.source_illumination"),
            _tensor_of(out.receiver_illumination, (), (illum_out,), "BackwardOutput.receiver_illumination"),
            _tensor_of(out.adcig, (adcig_out,), (), "BackwardOutput.adcig"))


def _map_rtm(out, cands):
    _, illum_out, adcig_out = cands
    return (_tensor_of(out.source_illumination, (), (illum_out,), "RTMOutput.source_illumination"),
            _tensor_of(out.receiver_illumination, (), (illum_out,), "RTMOutput.receiver_illumination"),
            _tensor_of(out.adcig, (adcig_out,), (), "RTMOutput.adcig"))


# The mapping on its own, for a test that builds an output struct by hand.
def map_forward_outputs(out, params):
    """``ForwardOutputCore`` -> (wavefield, last_two, record) against ``params``."""
    return _map_forward(out, _forward_candidates(params))


def map_backward_outputs(out, params):
    """``BackwardOutputCore`` -> ([], grads, source_illum, receiver_illum, adcig)."""
    return _map_backward(out, _backward_candidates(params))


def map_rtm_outputs(out, params):
    """``RTMOutputCore`` -> (source_illum, receiver_illum, adcig)."""
    return _map_rtm(out, _backward_candidates(params))


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

_KINDS = {
    _KIND_FORWARD: ("ForwardInputCore", _abi.ForwardOutputCore, _forward_candidates, _map_forward),
    _KIND_BACKWARD: ("BackwardInputCore", _abi.BackwardOutputCore, _backward_candidates, _map_backward),
    _KIND_RTM: ("BackwardInputCore", _abi.RTMOutputCore, _backward_candidates, _map_rtm),
}


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
    in_name, out_type, candidates, map_out = _KINDS[kind]

    def fn(params):
        with _StreamScope(lib, _launch_stream(params, name)):
            core, keep = adapt(in_name, params)
            out = out_type()
            err = ctypes.create_string_buffer(_ERR_CAP)
            # sweep_call looked up per call, not captured: a test may swap it
            # on the library object, and the lookup is nothing next to adapt().
            rc = lib.sweep_call(entry, ctypes.addressof(core), ctypes.addressof(out), err, _ERR_CAP)
            if rc != 0:
                _raise_core(err, name, rc)
            return map_out(out, candidates(params))   # core/keep die with the frame

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = f"Core entry {entry} ({('forward', 'backward', 'rtm')[kind]}) via libsweep_core."
    return fn


# ---------------------------------------------------------------------------
# Stepped runners (the DD time loop) and the boundary session
# ---------------------------------------------------------------------------

class _Runner:
    """One core runner handle.  The core struct, everything it points at and
    the tensors it describes stay alive for the handle's life; the output
    candidates are the ones bound at construction, which is what the core
    writes into whatever the params object is later set to."""
    _side = ""          # "forward" | "backward"
    _in_name = ""
    _out_type = None
    _candidates = None
    _map_out = None

    def __init__(self, entry_id: int, params):
        self._h = None
        self._lib = lib = core_lib()
        self._params = params
        self._core, self._keep = adapt(self._in_name, params)
        self._cands = self._candidates(params)
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

    def _run_segment(self, a: int, b: int, step_phase: int):
        h = self._h
        if h is None:
            raise RuntimeError(f"{self._side} runner is closed")
        out = self._out_type()
        err = ctypes.create_string_buffer(_ERR_CAP)
        with _StreamScope(self._lib, _stream_for_device(self._device_index(h))):
            rc = self._run(h, int(a), int(b), int(step_phase), ctypes.addressof(out), err, _ERR_CAP)
        if rc != 0:
            _raise_core(err, f"{self._side} runner run", rc)
        return self._map_out(out, self._cands)

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
    _candidates = staticmethod(_forward_candidates)
    _map_out = staticmethod(_map_forward)

    def run(self, it_begin: int, it_end: int, step_phase: int):
        return self._run_segment(it_begin, it_end, step_phase)


class BackwardRunner(_Runner):
    _side = "backward"
    _in_name = "BackwardInputCore"
    _out_type = _abi.BackwardOutputCore
    _candidates = staticmethod(_backward_candidates)
    _map_out = staticmethod(_map_backward)

    def run(self, bw_it_begin: int, bw_it_end: int, step_phase: int):
        return self._run_segment(bw_it_begin, bw_it_end, step_phase)


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


# ---------------------------------------------------------------------------
# Small entries and the namespace
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


def namespace() -> dict:
    """Everything ``sweep._C`` exposes, keyed by its old pybind name (cached;
    built once under ``_load_lock``)."""
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
