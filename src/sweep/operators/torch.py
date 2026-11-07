import importlib

import torch
import torch.nn.functional as F


def _resolve_spacing_for_axis(h, axis, ndim):
    if isinstance(h, torch.Tensor):
        if h.ndim == 0:
            return h
        spatial_ndim = h.shape[0]
    elif isinstance(h, (tuple, list)):
        spatial_ndim = len(h)
    else:
        return h

    normalized_axis = axis if axis >= 0 else ndim + axis
    spatial_axis = normalized_axis - (ndim - spatial_ndim)
    if spatial_axis < 0 or spatial_axis >= spatial_ndim:
        raise ValueError(
            f"Axis {axis} is incompatible with spacing of length {spatial_ndim} for tensor ndim={ndim}."
        )
    return h[spatial_axis]


def _to_nchw(u):
    if u.ndim == 2:
        return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
    if u.ndim == 3:
        return u.unsqueeze(1), lambda x: x.squeeze(1)
    if u.ndim == 4:
        return u, lambda x: x
    raise ValueError(f"Expected 2D/3D/4D input for gradient kernel, got shape {tuple(u.shape)}")


def _to_ncdhw(u):
    if u.ndim == 3:
        return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
    if u.ndim == 4:
        return u.unsqueeze(1), lambda x: x.squeeze(1)
    if u.ndim == 5:
        return u, lambda x: x
    raise ValueError(f"Expected 3D/4D/5D input for 3D gradient kernel, got shape {tuple(u.shape)}")


# ---- compiled-path stencils -------------------------------------------------
# Under torch.compile a conv is an opaque cuDNN call: Inductor cannot fuse it
# with the pointwise update around it, so every derivative round-trips a full
# field through memory and its backward keeps the conv inputs alive.  The same
# zero-padded cross-correlation written as a sum of shifted slices fuses into
# the step kernel (A100/V100 across the eager equations at order 4: forward
# 1.2-6x, full-tape gradient 1.1-10x faster and 1.3-4x less memory; 3-D, shot
# batches and high orders gain most).  Uncompiled eager keeps the conv -- there
# every slice is its own launch and the conv wins -- with a backward that keeps
# only the kernel (``_StencilConv`` below).
#
# The slice form needs the taps as Python floats, which a traced graph cannot
# read off a tensor, so ``register_stencil`` records them on the kernel tensor
# when the kernel is built.  An unregistered kernel keeps the conv: it reads the
# class-level ``None`` below.  The read is a plain attribute access because
# Dynamo before torch 2.5 breaks the graph on ``getattr(tensor, name, None)``,
# and there the slices never ran.
_TAPS_ATTR = "_sweep_taps"
setattr(torch.Tensor, _TAPS_ATTR, None)

# A/B switch: False keeps the conv under torch.compile as well.
SLICE_STENCILS_UNDER_COMPILE = True

def _resolve_is_compiling(torch_mod=torch):
    """The "is Dynamo tracing this?" probe: ``torch.compiler.is_compiling``, or
    on older 2.x (2.1 has ``torch.compiler`` without it) ``torch._dynamo``'s.
    Without the fallback every compile-only path -- these slice stencils, the
    source scatter -- silently stayed off there."""
    fn = getattr(getattr(torch_mod, "compiler", None), "is_compiling", None)
    if fn is None:
        try:
            fn = getattr(importlib.import_module(torch_mod.__name__ + "._dynamo"), "is_compiling", None)
        except ImportError:
            fn = None
    return fn or (lambda: False)


_is_compiling = _resolve_is_compiling()


def register_stencil(kernel):
    """Record the non-zero taps of a conv kernel ``(K, 1, *k)`` on the tensor.

    The taps are ``((offset, ...), coeff)`` pairs, offsets relative to the
    kernel centre; the ``K`` output channels are summed, matching
    :func:`apply_kernels`.  Kernels with an even extent are left alone (their
    zero-padded conv does not preserve the shape).  Returns ``kernel``.
    """
    if not isinstance(kernel, torch.Tensor) or kernel.ndim not in (4, 5) or kernel.shape[1] != 1:
        return kernel
    if _is_compiling():
        # Reading the taps off a traced tensor would break the graph; the
        # kernel keeps the conv instead.
        return kernel
    spatial = tuple(kernel.shape[2:])
    if any(s % 2 == 0 for s in spatial):
        return kernel
    k = kernel.detach().to("cpu", torch.float64).sum(dim=0)[0]
    centre = [s // 2 for s in spatial]
    taps = tuple(
        (tuple(i - c for i, c in zip(idx, centre)), float(k[tuple(idx)]))
        for idx in torch.nonzero(k).tolist()
    )
    if taps:
        setattr(kernel, _TAPS_ATTR, taps)
    return kernel


def _compiled_taps(kernel):
    if not (SLICE_STENCILS_UNDER_COMPILE and _is_compiling()) or not isinstance(kernel, torch.Tensor):
        return None
    return kernel._sweep_taps   # _TAPS_ATTR, spelled out: see above


def _apply_taps(u, taps):
    """Zero-padded, shape-preserving cross-correlation of the trailing dims of
    ``u`` with ``taps`` -- ``F.conv{2,3}d(u, kernel, padding=k//2)`` as shifted
    slices -- with the transpose as its derivative (``_Taps``)."""
    return _Taps.apply(u, taps)


def _mirrored(taps):
    """The taps of the transpose of ``_correlate(., taps)``: the offsets negated."""
    return tuple((tuple(-o for o in off), c) for off, c in taps)


class _Taps(torch.autograd.Function):
    """``_correlate`` with ``_correlate`` as its derivative.

    The zero-padded correlation ``out[x] = sum_o c_o u[x + o]`` (``u`` zero off
    the grid) is linear, and its transpose is the same correlation with the
    offsets negated: ``du[y] = sum_o c_o g[y - o]``, ``g`` zero off the grid.
    Differentiating the slices instead gives the same sum, but as a
    ``slice_backward`` per tap into a zero buffer of the padded shape, which
    Inductor cannot fuse with the step's kernels (one extra full-grid write and
    read per tap group).  The transpose computed as a correlation is gathered
    on the field's own grid and fuses like the forward.  It is the exact
    transpose, so the gradient moves by float32 round-off only (the sum runs
    over the taps in another order), and it is itself a ``_Taps``: second
    derivatives go through the same form.  torch.func's vmap is generated from
    the forward.  There is no forward-mode rule: Dynamo breaks the graph on an
    autograd.Function with a custom ``jvp``, and the op runs compiled only."""

    generate_vmap_rule = True

    @staticmethod
    def forward(u, taps):
        return _correlate(u, taps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.taps = inputs[1]

    @staticmethod
    def backward(ctx, grad):
        return _Taps.apply(grad, _mirrored(ctx.taps)), None


def _correlate(u, taps):
    """The correlation of ``_apply_taps``, by shifted slices of ``u`` padded
    with zeros."""
    nd = len(taps[0][0])
    pads = [max(abs(off[d]) for off, _ in taps) for d in range(nd)]
    pad_arg = []
    for p in reversed(pads):
        pad_arg += [p, p]
    up = F.pad(u, pad_arg) if any(pads) else u
    size = u.shape[-nd:]
    out = None
    for off, c in taps:
        sl = tuple(slice(p + o, p + o + n) for p, o, n in zip(pads, off, size))
        term = up[(Ellipsis,) + sl] * c
        out = term if out is None else out + term
    return out


def _zero_halo_where(out, pads):
    """Out-of-place :func:`_zero_halo` / :func:`_zero_halo_3d` (an in-place
    write would cut the fusion)."""
    nd = len(pads)
    keep = None
    for d, p in enumerate(pads):
        if p <= 0:
            continue
        n = out.shape[out.ndim - nd + d]
        idx = torch.arange(n, device=out.device)
        shape = [1] * nd
        shape[d] = n
        m = ((idx >= p) & (idx < n - p)).view(shape)
        keep = m if keep is None else keep & m
    if keep is None:
        return out
    return torch.where(keep, out, torch.zeros((), dtype=out.dtype, device=out.device))


def _zero_halo(out, padding):
    if isinstance(padding, int):
        pad_z = pad_x = padding
    else:
        pad_z, pad_x = padding
    if pad_z > 0:
        out[..., :pad_z, :] = 0
        out[..., -pad_z:, :] = 0
    if pad_x > 0:
        out[..., :, :pad_x] = 0
        out[..., :, -pad_x:] = 0
    return out


def _zero_halo_3d(out, padding):
    if isinstance(padding, int):
        pad_z = pad_y = pad_x = padding
    else:
        pad_z, pad_y, pad_x = padding
    if pad_z > 0:
        out[..., :pad_z, :, :] = 0
        out[..., -pad_z:, :, :] = 0
    if pad_y > 0:
        out[..., :, :pad_y, :] = 0
        out[..., :, -pad_y:, :] = 0
    if pad_x > 0:
        out[..., :, :, :pad_x] = 0
        out[..., :, :, -pad_x:] = 0
    return out


# ---- uncompiled stencils ----------------------------------------------------
# A conv's autograd node saves its input, which only the kernel's gradient
# reads.  A stencil's kernel is a constant, so that copy is dead weight: one
# field per derivative per step, alive until the backward (2-D Acoustic kept 5
# fields a step, 2-D Elastic 13).  ``_StencilConv`` keeps the kernel alone.  Its
# backward is the transposed correlation with the same kernel and padding --
# the call the conv's own backward makes for the input -- so the record and
# every gradient stay bitwise the conv's.  ``gradient``'s halo zeroing and
# ``apply_kernels``' channel sum run inside it, which makes a derivative one
# autograd node instead of a conv and four CopySlices (each a full-field copy
# in the backward).  Forward mode (``jvp``) and second derivatives work; a
# second derivative is a plain conv with the kernel, where the conv's own also
# ran cuDNN's weight-gradient kernels.  Compiled steps take the slice form above
# and never get here.  The price is host time: a Python autograd node costs
# ~25 us a call more than the conv's (RTX 6000 Ada), which a host-bound 2-D step
# shows -- the staggered-grid elastic family ran 12-14% slower at 1000x1000
# there, the acoustic one 7-11% faster (fewer nodes); on a V100 every equation
# measured stayed within -5% and +0.1%.

# A/B switch: False runs the uncompiled stencils as plain convs again.
STENCIL_BACKWARD_KEEPS_KERNEL_ONLY = True


def _stencil(u, kernel, padding, halo=None, channel_sum=False):
    """``F.conv{2,3}d(u, kernel, padding=padding)``; its channels summed when
    ``channel_sum``, then its ``halo`` zeroed in place."""
    out = (F.conv2d if kernel.ndim == 4 else F.conv3d)(u, kernel, padding=padding)
    if channel_sum and out.shape[1] != 1:
        out = out.sum(dim=1, keepdim=True)
    if halo is not None:
        out = (_zero_halo if kernel.ndim == 4 else _zero_halo_3d)(out, halo)
    return out


class _StencilConv(torch.autograd.Function):
    """:func:`_stencil` with a constant ``kernel``, keeping nothing but the kernel."""

    @staticmethod
    def forward(ctx, u, kernel, padding, halo, channel_sum):
        # An undefined output gradient stays None, as it does for the conv,
        # instead of zeros pushed through a transposed conv.
        ctx.set_materialize_grads(False)
        # An attribute, not save_for_backward: the kernel is a constant -- no
        # version to check, nothing for a checkpoint to recompute -- and a
        # saved tensor costs a pack and an unpack per call.
        ctx.kernel, ctx.padding, ctx.halo, ctx.channel_sum = kernel, padding, halo, channel_sum
        return _stencil(u, kernel, padding, halo, channel_sum)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None, None
        kernel = ctx.kernel
        if kernel.dtype != grad.dtype:      # autocast: the forward ran in the cast dtype
            kernel = kernel.to(grad.dtype)
        if ctx.halo is not None:
            grad = (_zero_halo if kernel.ndim == 4 else _zero_halo_3d)(grad.clone(), ctx.halo)
        if ctx.channel_sum and kernel.shape[0] != 1:
            grad = grad.expand(-1, kernel.shape[0], *grad.shape[2:])
        conv_t = F.conv_transpose2d if kernel.ndim == 4 else F.conv_transpose3d
        return conv_t(grad, kernel, padding=ctx.padding), None, None, None, None

    @staticmethod
    def jvp(ctx, u_t, *_):
        return None if u_t is None else _stencil(u_t, ctx.kernel, ctx.padding, ctx.halo, ctx.channel_sum)


def _conv_stencil(u, kernel, padding, halo=None, channel_sum=False):
    """:func:`_stencil`, through :class:`_StencilConv` when a backward will
    run (``u`` differentiated, ``kernel`` constant, not traced)."""
    if (u.requires_grad and STENCIL_BACKWARD_KEEPS_KERNEL_ONLY and torch.is_grad_enabled()
            and not kernel.requires_grad and not _is_compiling()):
        return _StencilConv.apply(u, kernel, padding, halo, channel_sum)
    return _stencil(u, kernel, padding, halo, channel_sum)


def separable_d2_2d(u, k1d, hz=1.0, hx=1.0):
    """Separable per-axis 2nd derivatives of a 2-D wavefield.

    Naming: ``d2`` = second derivative (∂²); ``2d`` = 2-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_2d` for the shortcut.

    Implementation: two separable 1-D ``conv2d`` calls, one per axis.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, nx)``.
        k1d: 1-D kernel of length ``2M+1`` (or a tuple ``(kz, kx)`` of
            pre-shaped 4-D kernels for the cached path).
        hz, hx: Grid spacings along z and x.

    Returns:
        ``(d2u_dz2, d2u_dx2)`` — two tensors of the same shape as ``u``.
    """
    if isinstance(k1d, tuple):
        kz, kx = k1d
        pad = max(kz.shape[-3], kx.shape[-1]) // 2
        tz, tx = _compiled_taps(kz), _compiled_taps(kx)
        if tz is not None and tx is not None:
            return _apply_taps(u, tz) / (hz*hz), _apply_taps(u, tx) / (hx*hx)
    else:
        kz = k1d[None, None, :, None]  # (1,1,k,1)
        kx = k1d[None, None, None, :]  # (1,1,1,k)
        pad = k1d.shape[-1] // 2
    lapx = _conv_stencil(u, kx, (0, pad)) / (hx*hx)
    lapz = _conv_stencil(u, kz, (pad, 0)) / (hz*hz)
    return lapz, lapx

def separable_d2_3d(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """Separable per-axis 2nd derivatives of a 3-D wavefield.

    Naming: ``d2`` = second derivative (∂²); ``3d`` = 3-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_3d` for the shortcut.

    Implementation: three separable 1-D ``conv3d`` calls, one per axis.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, ny, nx)``.
        k1d: 1-D kernel of length ``2M+1`` (or a tuple ``(kz, ky, kx)``
            of pre-shaped 5-D kernels for the cached path).
        hz, hy, hx: Grid spacings along z, y, and x.

    Returns:
        ``(d2u_dz2, d2u_dy2, d2u_dx2)`` — three tensors of the same
        shape as ``u``.
    """
    if isinstance(k1d, tuple):
        kz, ky, kx = k1d
        pad = max(kz.shape[-3], ky.shape[-2], kx.shape[-1]) // 2
        taps = [_compiled_taps(k) for k in (kz, ky, kx)]
        if all(t is not None for t in taps):
            return tuple(_apply_taps(u, t) / (h * h) for t, h in zip(taps, (hz, hy, hx)))
    else:
        pad = k1d.shape[-1] // 2
        kz = k1d.view(1, 1, -1, 1, 1)
        ky = k1d.view(1, 1, 1, -1, 1)
        kx = k1d.view(1, 1, 1, 1, -1)

    lapz = _conv_stencil(u, kz, (pad, 0, 0)) / (hz * hz)
    lapy = _conv_stencil(u, ky, (0, pad, 0)) / (hy * hy)
    lapx = _conv_stencil(u, kx, (0, 0, pad)) / (hx * hx)

    return lapz, lapy, lapx

# Plain Python — no @torch.jit.script. TorchScript and Dynamo are separate
# JITs and a @torch.jit.script function is opaque to Dynamo: every PD call
# inside the equation step becomes a graph break, which neutralises
# torch.compile on Elastic / DAS (their step calls these kernels 8+ times).
# Measured on RTX 6000 Ada, removing the decorator also makes eager Elastic
# ~6% faster, so the decorator was net cost even without torch.compile.
def apply_kernels(u, kernels):
    # u: (B, 1, H, W). kernels: (1, 1, kh, kw) or (K, 1, kh, kw).
    taps = _compiled_taps(kernels)
    if taps is not None:
        return _apply_taps(u, taps)
    _, _, KH, KW = kernels.shape
    return _conv_stencil(u, kernels, (KH // 2, KW // 2), channel_sum=True)


def apply_kernels_3d(u, kernels):
    # u: (B, 1, D, H, W). kernels: (1, 1, kD, kH, kW) or (K, 1, kD, kH, kW).
    taps = _compiled_taps(kernels)
    if taps is not None:
        return _apply_taps(u, taps)
    _, _, KD, KH, KW = kernels.shape
    return _conv_stencil(u, kernels, (KD // 2, KH // 2, KW // 2), channel_sum=True)


def laplacian_2d(u, k1d, hz=1.0, hx=1.0):
    """Isotropic 2-D Laplacian via separable kernels: ``d2z + d2x``.

    Convenience wrapper around :func:`separable_d2_2d` for callers that
    only need the scalar Laplacian ``∇²u`` (the common isotropic-acoustic
    case). Anisotropic equations should keep using
    :func:`separable_d2_2d` and combine the two components themselves.
    """
    d2z, d2x = separable_d2_2d(u, k1d, hz=hz, hx=hx)
    return d2z + d2x


def laplacian_3d(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """Isotropic 3-D Laplacian via separable kernels: ``d2z + d2y + d2x``.

    Convenience wrapper around :func:`separable_d2_3d`; see
    :func:`laplacian_2d` for the rationale.
    """
    d2z, d2y, d2x = separable_d2_3d(u, k1d, hz=hz, hy=hy, hx=hx)
    return d2z + d2y + d2x


def gradient(u, h, axis, kernels=None):
    h_axis = _resolve_spacing_for_axis(h, axis, u.ndim)
    if kernels is not None:
        if axis not in kernels:
            raise ValueError(f"No gradient kernel configured for axis={axis}.")
        kernel = kernels[axis]
        taps = _compiled_taps(kernel)
        if kernel.ndim == 4:
            padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            u_nchw, restore = _to_nchw(u)
            if taps is not None:
                return restore(_zero_halo_where(_apply_taps(u_nchw, taps) / h_axis, padding))
            out = _conv_stencil(u_nchw, kernel / h_axis, padding, halo=padding)
        elif kernel.ndim == 5:
            padding = (kernel.shape[-3] // 2, kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            u_ncdhw, restore = _to_ncdhw(u)
            if taps is not None:
                return restore(_zero_halo_where(_apply_taps(u_ncdhw, taps) / h_axis, padding))
            out = _conv_stencil(u_ncdhw, kernel / h_axis, padding, halo=padding)
        else:
            raise ValueError(f"Expected 2D or 3D gradient kernel, got shape {tuple(kernel.shape)}")
        return restore(out)
    return torch.gradient(u, spacing=h_axis, dim=axis)[0]
