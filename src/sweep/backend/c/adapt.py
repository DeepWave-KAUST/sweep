"""tensor -> Buf: the core struct for a ForwardInput / BackwardInput.

The twin of cuda/common/buf_torch.h ``buf_of`` and cuda/common/adapt_inputs.h
``InputArena``, in Python.  The only things asked of a tensor are
``data_ptr()``, ``shape``, ``stride()``, ``dtype``, ``device`` and
``element_size()``, which every torch has.

Nothing here owns device memory: a Buf is a view of a tensor the caller keeps
alive, and every host-side array a struct points at -- the Buf arrays of the
list fields, the int32 host copies of the tensors the drivers read on the host
(``source_field_indices`` and friends, what ``InputArena`` made once at entry)
-- is made per call and sits in the ``keepalive`` list returned next to the
struct.  The module imports without torch and without a GPU.
"""
from __future__ import annotations

import ctypes
import math
from ctypes import POINTER, c_char_p, c_float, c_int32, c_void_p

from . import abi as _abi

_BUF_MAX_DIMS = 8        # csrc/core/buf.h BUF_MAX_DIMS, mirrored by Buf.sizes_

# Byte width each storage tag implies (core/buf.h buf_dtype_element_size);
# _fill_buf refuses a tensor whose element_size() disagrees, so a float64 or
# int64 buffer never reaches the core mislabelled.
_TAG_WIDTH = {_abi.FP32: 4, _abi.FP16: 2, _abi.BF16: 2, _abi.INT8: 1}

_dtype_tags = None


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


_HOST_CACHE_ATTR = "_capi_host_copies"


def _host_int32_cached(obj, name: str, t):
    """``_host_int32(t)`` reused across calls on the same params object while
    the tensor is the same object with the same version counter.  The DD time
    loop hands the same indices to every step; each fresh copy is a device
    sync, and two syncs per step drained the GPU queue on the per-call path
    (measured: +170 us/step).  In-place writes bump ``_version``, a new tensor
    fails the identity check, and holding the reference keeps its storage from
    being recycled under the same id.  Objects that refuse the attribute get no
    cache."""
    ver = getattr(t, "_version", None)
    cache = getattr(obj, _HOST_CACHE_ATTR, None)
    if cache is None:
        try:
            cache = {}
            setattr(obj, _HOST_CACHE_ATTR, cache)
        except (AttributeError, TypeError):
            return _host_int32(t)
    hit = cache.get(name)
    if hit is not None and hit[0] is t and hit[1] == ver:
        return hit[2]
    h = _host_int32(t)
    cache[name] = (t, ver, h)
    return h


def _host_int32(t):
    """The int32 host copy of ``t`` the drivers read (what InputArena made;
    a CPU int32 contiguous tensor is its own copy)."""
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


def _session_handle(what: str, name: str, val):
    """The raw core pointer for a BoundarySession* field: a ``BoundarySession``
    (anything carrying a ``handle``), a raw handle (int / c_void_p), or None.
    A closed session is refused here -- the core would otherwise be handed NULL
    for a session the caller thinks is live."""
    if val is None:
        return None
    if hasattr(val, "handle"):
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
    each a device->host sync made here, per call) and every tensor described
    lives in ``keepalive`` -- the struct must not outlive that list.

    Validation happens here, by field name: a single tensor where a list is
    due, a non-tensor inside a list, a scalar for a host index table and a
    closed boundary session are refused before the core can dereference them.
    """
    core = getattr(_abi, struct_name)()
    keep = []
    for name, kind, default in _abi.FIELDS[struct_name]:
        val = getattr(obj, name, default)
        if kind == "Buf":
            if val is not None and name in _abi.HOST_BUF:
                # A tensor the drivers read on the host: an int32 host copy
                # (what InputArena made); a plain list is made one directly.
                if _is_tensor(val):
                    val = _host_int32_cached(obj, name, val)
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
                # The ``Buf * n`` array the BufList points into (None entries
                # stay undefined Bufs); the tensors ride in the keepalive too.
                arr = (_abi.Buf * n)()
                for i, t in enumerate(ts):
                    if t is not None and not _is_tensor(t):
                        raise RuntimeError(f"{struct_name}.{name}[{i}] is a {type(t).__name__}, not a tensor")
                    _fill_buf(arr[i], t)
                keep.append(arr)
                keep.extend(ts)
                setattr(core, name, _abi.BufList(p=ctypes.cast(arr, POINTER(_abi.Buf)), n=n))
        elif kind == "IntSpan":
            if name in _abi.HOST_INT_SPAN and _is_tensor(val):
                # A tensor the drivers read on the host: one int32 host copy
                # (what InputArena::host_ints_of made); empty -> n=0.
                if math.prod(val.shape) > 0:
                    h = _host_int32_cached(obj, name, val)
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
            raise RuntimeError(f"sweep.backend.c.abi: field {struct_name}.{name} has unknown kind {kind!r}")
    return core, keep
