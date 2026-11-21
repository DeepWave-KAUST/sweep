"""Domain decomposition of the eager propagator, by autograd alone.

The eager step computes on whole arrays, with zero padding at the array's
edge.  A tile therefore holds its OWNED cells of the runtime grid (pads
included) plus ``H`` GHOST cells on each cut side; after one step the
outermost ``R`` cells of the array are wrong (``R`` is the step's reach), so
with ``H >= K * R`` the owned cells stay exact for ``K`` steps, and every
``K`` steps the ghosts are replaced by the neighbour's owned cells.  That
exchange is an ordinary differentiable op (:class:`HaloSwap`): out of place,
its backward the transpose of the copy (the ghosts' gradient goes back to the
owner and is added there; the overwritten ghost inputs get zero).  Nothing
else is written for the gradient: a tile's rollout is the single-domain
rollout on a smaller array, and ``loss.backward()`` gives the model and the
wavelet gradients as it does on one card.

Cuts: x strips in 2-D; an x/y tile grid in 3-D (``MeshTopology(py, px)``).
A rank exchanges all its sides at once (:class:`_TileExchange`): the x
faces, the y faces and the corners, each a block from one neighbour, the
messages in one group, two full-field copies per exchange whatever the
number of cut axes.  (The single-process oracle exchanges along x over all
rows, then along y over all columns -- the corners then come through the y
neighbour's x ghosts -- which fills the same cells.)

Nothing here is equation-specific.  An equation is admitted when
  * its step is local: the probed reach fits the narrowest tile
    (:func:`step_reach`);
  * every grid-shaped tensor it holds can be cut: any tensor on the equation,
    or in a sub-object of it, whose last dim is the runtime nx (or, in 3-D,
    whose second-to-last is the runtime ny) is sliced per tile (the CPML
    profiles, a topography's rows per column, a new equation's own profile),
    see :func:`tile_propagator`;
  * the self-check passes (:func:`admit`): 2K steps from a random state on
    tiles, with one exchange, against the global grid.  What the slicer
    missed -- a grid-shaped numpy array, an ``nx - 1`` staggered array padded
    at use, an operation that is not local -- fails the tile step or the
    comparison, and decomposition is refused with the reason.

Which fields are exchanged is probed too (:func:`exchange_plan`), per axis:
the K-step VJP of the outputs on one side of mid-grid against each input
field says whether that field's ghosts along the axis reach an owned cell
within K steps.  A CPML memory field of the z strips does not (its update
reads along z), nor does a second-order scheme's previous field at K = 1; the
acoustic equation at K = 1 exchanges one field of six, the staggered elastic
one 7 of 15.

:class:`EagerModelParallel` runs one tile per rank (built by
``ModelParallel(prop, mesh)`` when ``prop`` is eager);
:func:`decomposed_forward` runs every tile in one process (the exchange a
``torch.cat`` of slices), the oracle the tests compare against.
"""

from __future__ import annotations

import copy
import warnings

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as ckpt_torch

from sweep._compile import SettledSteps, init_cuda_before_checkpoint
from sweep.parallel._topology import MeshTopology
from sweep.parallel.routing import assemble_tile_records
from sweep.propagator._second_order import has_tangent, refuse_forward_ad
from sweep.receivers.torch import ReceiverTorch
from sweep.sources.torch import SourceTorch
from sweep.utils.torch import EdgePadding

__all__ = ["EagerModelParallel", "FusedHaloSwap", "HaloSwap", "Link", "admit",
           "decomposed_forward", "exchange_plan", "step_reach", "tile_propagator"]

#: The cut axes: name -> tensor dim (wavefields are ``(B, 1, nz, [ny,] nx)``,
#: coordinates ``(x, [y,] z)``) and coordinate index.
AXES = {"x": (-1, 0), "y": (-2, 1)}


def _impl(prop):
    """The eager backend of a ``PropTorch`` facade (or ``prop`` itself)."""
    return getattr(prop, "_backend_impl", prop)


def _is_eager(prop):
    from sweep.propagator._torch_eager import _PropTorchEager
    return isinstance(_impl(prop), _PropTorchEager)


def _sl(t, dim, lo, hi):
    """``t`` narrowed to ``[lo, hi)`` along ``dim`` (a view)."""
    return t.narrow(dim, lo, hi - lo)


# ---------------------------------------------------------------- the probes
def _direct_convolutions():
    """cuDNN / oneDNN off: an FFT or Winograd convolution leaves ~1e-20 where
    a tap is zero, which a dependency probe reads as a dependency (the 3-D
    acoustic step probed a reach of 6 instead of 2 in float64).  The direct
    kernels keep an exact zero."""
    return torch.backends.cudnn.flags(enabled=False), torch.backends.mkldnn.flags(enabled=False)


def _reach_of(grads, dim, c, side):
    """Per input field, how many cells across mid-grid ``c`` along ``dim`` its
    nonzero gradient extends, for outputs taken on ``side`` of ``c``."""
    out = []
    for gr in grads:
        w = 0
        if gr is not None:
            hit = (gr != 0).movedim(dim, -1).flatten(0, -2).any(0).nonzero().flatten()
            if len(hit):
                w = int(hit.max()) + 1 - c if side == "left" else c - int(hit.min())
        out.append(max(w, 0))
    return out


def _probe(prop, models_rt, K, seed, axis="x"):
    """Per field, the cells across mid-grid along ``axis`` its ghosts reach in
    K steps (max over both directions): the VJP of K steps' outputs on one
    side of mid-grid against the inputs, with exact zeros.  Not a
    perturbation test: in float32 a perturbation of the stress does not
    change the velocity at all (dt / (rho h) ~ 5e-8 is below the rounding of
    an O(1) velocity), and that test measured the elastic reach as 2 instead
    of 3.  A staggered scheme is not symmetric either, so both directions
    are probed."""
    prop = _impl(prop)
    dim = AXES[axis][0]
    g = torch.Generator(device=prop.dev).manual_seed(seed)
    shape = (1, 1) + tuple(prop._runtime_shape())
    c = shape[dim] // 2
    h = prop._equation_spacing
    dt_ = models_rt[0].dtype
    width = [0] * len(prop.wavefield_names)
    for side in ("left", "right"):
        cudnn, mkldnn = _direct_convolutions()           # one entry each
        with torch.enable_grad(), cudnn, mkldnn:
            s0 = [torch.randn(shape, generator=g, device=prop.dev, dtype=dt_).requires_grad_()
                  for _ in prop.wavefield_names]
            st = s0
            for _ in range(K):
                st = prop._plain_step_func(*st, *models_rt, prop.dt, h, None)
            loss = 0
            for o in st:
                part = _sl(o, dim, 0, c) if side == "left" else _sl(o, dim, c, o.shape[dim])
                loss = loss + (part * torch.rand(part.shape, generator=g, device=prop.dev,
                                                 dtype=o.dtype)).sum()
            grads = torch.autograd.grad(loss, s0, allow_unused=True)
        width = [max(a, b) for a, b in zip(width, _reach_of(grads, dim, c, side))]
    return width


def step_reach(prop, models_rt, seed=0, axis="x"):
    """Cells one step carries information along ``axis``, max over the
    fields and both directions: acoustic M, staggered elastic 2M - 1
    (M = so / 2)."""
    return max(_probe(prop, models_rt, 1, seed, axis))


def exchange_plan(prop, models_rt, K, seed=2, axis="x"):
    """``(fields, H)``: the indices of the state tensors whose ghosts along
    ``axis`` reach an owned cell within ``K`` steps, and the halo width that
    holds them."""
    width = _probe(prop, models_rt, K, seed, axis)
    return [j for j, w in enumerate(width) if w > 0], max(width)


# ------------------------------------------------------- equation slicing
_MODULE_DICTS = ("_modules", "_buffers", "_parameters")


def _cut_tensor(t, cuts):
    """``t`` cut along every axis whose runtime length it carries: ``cuts``
    lists ``(dim, lo, hi, n)``."""
    for dim, lo, hi, n in cuts:
        if t.dim() >= -dim and t.shape[dim] == n and n > 1:
            t = _sl(t, dim, lo, hi)
    return t


def _slice_obj(obj, cuts, found, seen, root=False):
    """A copy of ``obj`` with every grid-shaped tensor cut (:func:`_cut_tensor`).
    Sweep objects and ``nn.Module``s (the root -- the equation, which may be a
    user's class defined anywhere -- always) are copied shallowly and walked:
    derivative operators, CPML helpers, free-surface helpers.  Tensors that
    are not grid-shaped are shared.  ``found`` collects the shapes of the
    sliced tensors."""
    if torch.is_tensor(obj):
        new = _cut_tensor(obj, cuts)
        if new is not obj:
            found.append(tuple(obj.shape))
        return new
    if isinstance(obj, list):
        return [_slice_obj(o, cuts, found, seen) for o in obj]
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(_slice_obj(o, cuts, found, seen) for o in obj)
    if isinstance(obj, dict):
        return {k: _slice_obj(v, cuts, found, seen) for k, v in obj.items()}
    if root or (hasattr(obj, "__dict__") and (type(obj).__module__.startswith("sweep.")
                                              or isinstance(obj, torch.nn.Module))):
        if id(obj) in seen:
            return seen[id(obj)]
        new = copy.copy(obj)
        seen[id(obj)] = new
        for k, v in list(vars(obj).items()):
            if k in _MODULE_DICTS:            # nn.Module dicts a shallow copy would share
                new.__dict__[k] = {kk: _slice_obj(vv, cuts, found, seen) for kk, vv in v.items()}
                continue
            nv = _slice_obj(v, cuts, found, seen)
            if nv is not v:
                new.__dict__[k] = nv
        return new
    return obj


def tile_propagator(prop, cuts):
    """A shallow copy of the eager backend whose equation is cut to the tile
    (``cuts``: ``(dim, lo, hi)`` per cut axis, runtime indices), with its own
    step functions built over it (and its own compiled-step cache: Dynamo
    guards on the equation object).  Returns ``(copy, shapes of the sliced
    tensors)``."""
    prop = _impl(prop)
    shape = (1, 1) + tuple(prop._runtime_shape())
    cuts = [(dim, lo, hi, shape[dim]) for dim, lo, hi in cuts]
    found = []
    eq = _slice_obj(prop.equation, cuts, found, {}, root=True)
    p = copy.copy(prop)
    for d in _MODULE_DICTS:
        if d in p.__dict__:
            p.__dict__[d] = dict(p.__dict__[d])
    p.equation = eq
    p._steps_funcs, p._step_alone = {}, None
    p._plain_step_func = p._build_step_func()
    p._plain_steps_func = p._build_steps_func(p._plain_step_func)
    return p, found


# ------------------------------------------------------------- geometry
class _Cut:
    """One cut axis of the tile grid: the runtime cells each of its ``parts``
    tiles owns, and -- after admission -- the reach, the halo and the fields
    exchanged along it."""

    def __init__(self, axis, n_rt, ranges, phys=None):
        self.axis, self.dim, self.coord = axis, AXES[axis][0], AXES[axis][1]
        self.n, self.ranges, self.parts = n_rt, ranges, len(ranges)
        self.phys = phys if phys is not None else (0, n_rt)   # runtime span of the physical cells
        self.R = self.H = None
        self.fields = ()

    def extent(self, i):
        """``(lo, own_lo, own_hi, hi)`` of tile ``i`` along this axis: the
        owned range with ``H`` ghost cells on each cut side."""
        a, b = self.ranges[i]
        return (a - self.H if i > 0 else 0, a, b, b + self.H if i < self.parts - 1 else self.n)

    def phys_extent(self, i):
        """``(offset, length)`` of the PHYSICAL cells tile ``i`` owns."""
        a, b = self.ranges[i]
        p0, p1 = self.phys
        return max(a, p0) - p0, min(b, p1) - max(a, p0)

    # Compiled-step width the balanced extents are rounded to: on a V100 the
    # 3-D acoustic step is ~10% faster at widths divisible by 4 (139 cells:
    # 1.53 ms, 140: 1.37 ms), the 2-D one ~5%; the elastic step does not care.
    ALIGN = 4

    def balance(self):
        """Re-cut so every tile computes the same number of cells along this
        axis: its owned cells -- the edge tiles' include the absorbing pad --
        plus ``H`` ghost cells per cut side (``H`` known after admission's
        probe).  The edge tiles then own fewer physical cells than the inner
        ones.  Extents are whole multiples of ``ALIGN`` cells but for one tile
        (the step runs ~10% slower on some widths that are not, see
        ``ALIGN``), so they can differ by up to ``2 * ALIGN - 1`` cells."""
        P, H = self.parts, self.H
        if P == 1:
            return
        A = self.ALIGN
        chunks, left = divmod(self.n + 2 * H * (P - 1), A)
        base, rem = divmod(chunks, P)
        # the spare chunks to the inner tiles first, then the edges; the
        # leftover (< ALIGN) cells to the last tile with no spare chunk
        order = list(range(1, P - 1)) + [0, P - 1]
        ext = [base * A] * P
        for i in order[:rem]:
            ext[i] += A
        ext[order[rem]] += left
        ranges, a = [], 0
        for i in range(P):
            b = a + ext[i] - H * ((i > 0) + (i < P - 1))
            ranges.append((a, b))
            a = b
        assert a == self.n
        uniform, self.ranges = self.ranges, ranges
        for i in range(P):
            if self.phys_extent(i)[1] < 1:
                self.ranges = uniform
                raise NotImplementedError(
                    f"domain decomposition: balance=True leaves tile {i} along {self.axis} "
                    f"no physical cell (its absorbing pad alone exceeds its share): fewer tiles "
                    f"or balance=False")


def tile_grid(prop, py, px):
    """The cut axes of a ``py x px`` tile grid (``[x]`` in 2-D, ``[x, y]`` in
    3-D), the physical axes split as :meth:`MeshTopology.local_extent` splits
    them, the edge tiles taking the pads of their side."""
    prop = _impl(prop)
    phys = tuple(prop._shape_phys)
    rt = tuple(prop._runtime_shape())
    off = prop._runtime_coord_offset()                    # (x, [y,] z)
    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=py * px, rank=0)
    loc, _ = topo.local_extent(phys)
    cuts = []
    for axis, parts in (("x", px), ("y", py)):
        if axis == "y" and len(phys) == 2:
            continue
        dim, ci = AXES[axis]
        n_loc, n_rt, o = loc[dim], rt[dim], int(off[ci])
        ranges = [(0 if i == 0 else o + i * n_loc, n_rt if i == parts - 1 else o + (i + 1) * n_loc)
                  for i in range(parts)]
        cuts.append(_Cut(axis, n_rt, ranges, phys=(o, o + phys[dim])))
    return cuts


def _tile_index(cuts, flat):
    """Tile ``flat`` (= ``yi * px + xi``) -> its index along each cut axis."""
    px = cuts[0].parts
    return [flat % px] + ([flat // px] if len(cuts) > 1 else [])


def _rect(cuts, idx):
    """Per cut axis, ``(lo, own_lo, own_hi, hi)`` of the tile at ``idx``."""
    return [c.extent(i) for c, i in zip(cuts, idx)]


def _cut_models(models_rt, cuts, rect):
    """The tile's part of grid-shaped runtime models; others shared."""
    spec = [(c.dim, lo, hi, c.n) for c, (lo, _, _, hi) in zip(cuts, rect)]
    return [_cut_tensor(m, spec) for m in models_rt]


# ------------------------------------------------------------ admission
def _check_backward_additive(prop, global_rt, s0, cuts, g, rtol=2e-5):
    """The backward of one step must be additive in the cotangent: the tiles
    back-propagate their owned pieces of it separately and the exchange's
    backward adds them up.  Autograd's VJP is linear, so this holds for every
    step that is differentiated by autograd; it fails for a hand-written
    backward that is nonlinear in the cotangent (the qP equations'
    ``s_gradient='adjoint'``, which takes S_n from the adjoint field's own
    gradient) -- whose DD record would be right and its gradient not."""
    h = prop._equation_spacing
    st = [x.clone().requires_grad_() for x in s0]
    ms = [m.detach().clone().requires_grad_() if m.is_floating_point() else m for m in global_rt]
    leaves = st + [m for m in ms if m.requires_grad]
    with torch.enable_grad():
        out = prop._plain_step_func(*st, *ms, prop.dt, h, None)
        outs = [o for o in out if torch.is_tensor(o) and o.requires_grad]
        vs = [torch.randn(o.shape, generator=g, device=o.device, dtype=o.dtype) for o in outs]
        c = cuts[0]
        mask = torch.zeros(outs[0].shape[c.dim], device=prop.dev, dtype=outs[0].dtype)
        mask[: c.ranges[0][1]] = 1                     # the first tile's side of the first cut
        shape = [1] * outs[0].dim()
        shape[c.dim] = -1
        mask = mask.view(shape)

        def vjp(cot):
            return torch.autograd.grad(outs, leaves, cot, retain_graph=True, allow_unused=True)

        whole = vjp(vs)
        left = vjp([v * mask for v in vs])
        right = vjp([v * (1 - mask) for v in vs])
    for i, (w, a, b) in enumerate(zip(whole, left, right)):
        if w is None:
            continue
        a = torch.zeros_like(w) if a is None else a
        b = torch.zeros_like(w) if b is None else b
        err = float((a + b - w).abs().max() / w.abs().max().clamp_min(1e-30))
        if not err <= rtol:
            raise NotImplementedError(
                f"domain decomposition: the step's backward is not additive in the cotangent "
                f"(input {i}: {err:.1e} relative) -- a hand-written backward that is not the "
                f"transpose of the step (e.g. the qP equations' s_gradient='adjoint'); the "
                f"tiles' gradients would not add up to the single domain's. Use an "
                f"autograd-differentiated mode (s_gradient='frozen' or 'exact' for qP).")


def admit(prop, padded_models, cuts, K, rtol=1e-5, seed=1, balance=False):
    """Decide what to exchange and check that the equation runs on tiles.

    ``padded_models``: the global models on the runtime grid, batch layout
    applied, not yet prepared.  Fills each cut's ``R`` (the one-step reach),
    ``H`` (the halo) and ``fields`` (the state tensors exchanged along it,
    :func:`exchange_plan`).  Raises ``NotImplementedError`` with the reason.

    Self-check: 2K steps from a random state on the global grid, against
    every tile -- each cut from the same state, run K steps with its own
    prepared models, its ghosts then set as the exchange sets them (the
    exchanged fields' from the global state, the others' to RANDOM values:
    what they hold must not matter; along x first, then y, so the corners are
    the y pass's), then K steps more; owned cells within ``rtol`` (relative),
    not bitwise: cuDNN may pick another algorithm for another array shape.
    Transient memory: a few global-grid fields per state tensor.

    ``balance``: once the halo is known, re-cut every axis so the tiles
    compute equally many cells (:meth:`_Cut.balance`); the self-check then
    runs on the balanced tiles.
    """
    prop = _impl(prop)
    if prop._strips is not None:
        raise NotImplementedError(
            "domain decomposition of the eager propagator does not support the CPML "
            "strips layout (cpml_strips=True, the 3-D default): the strips are separate "
            "tensors the cut does not pass through. Build the propagator with "
            "cpml_strips=False.")
    shape = (1, 1) + tuple(prop._runtime_shape())
    global_rt = prop._prepare_runtime_models([m.detach() for m in padded_models])
    for c in cuts:
        try:
            c.R = step_reach(prop, global_rt, axis=c.axis)
            c.fields, c.H = exchange_plan(prop, global_rt, K, axis=c.axis)
        except Exception as e:
            raise NotImplementedError(
                f"domain decomposition: probing the step's reach along {c.axis} failed: "
                f"{type(e).__name__}: {e}") from e
        if c.R >= c.n // 2 - 1:          # the probe (at mid-grid) saturated
            raise NotImplementedError(
                f"domain decomposition: the step reaches across half the grid along "
                f"{c.axis} (R={c.R}, n={c.n}): it is not local, a tile would hold its "
                f"neighbours whole")
        if balance:
            c.balance()
        narrow = min(b - a for a, b in c.ranges)
        if c.R == 0 or c.H > narrow:
            raise NotImplementedError(
                f"domain decomposition: the halo H={c.H} along {c.axis} (exchange every "
                f"{K} steps, reach R={c.R}) does not fit the narrowest tile's {narrow} "
                f"owned cells; fewer tiles or a smaller exchange_steps")
    g = torch.Generator(device=prop.dev).manual_seed(seed)
    dt_ = global_rt[0].dtype
    s0 = [torch.randn(shape, generator=g, device=prop.dev, dtype=dt_) for _ in prop.wavefield_names]
    h = prop._equation_spacing
    step = prop._plain_step_func
    _check_backward_additive(prop, global_rt, s0, cuts, g)
    ntiles = 1
    for c in cuts:
        ntiles *= c.parts
    with torch.no_grad():
        mid = s0
        for _ in range(K):
            mid = step(*mid, *global_rt, prop.dt, h, None)
        ref = mid
        for _ in range(K):
            ref = step(*ref, *global_rt, prop.dt, h, None)
        for flat in range(ntiles):
            rect = _rect(cuts, _tile_index(cuts, flat))
            spec = [(c.dim, lo, hi) for c, (lo, _, _, hi) in zip(cuts, rect)]
            cut4 = [(c.dim, lo, hi, c.n) for c, (lo, _, _, hi) in zip(cuts, rect)]
            where = " x ".join(f"{c.axis}[{lo}, {hi})" for c, (lo, _, _, hi) in zip(cuts, rect))
            try:
                p, _ = tile_propagator(prop, spec)
                ms = p._prepare_runtime_models(
                    _cut_models([m.detach() for m in padded_models], cuts, rect))
                st = [_cut_tensor(t, cut4) for t in s0]
                for _ in range(K):
                    st = p._plain_step_func(*st, *ms, prop.dt, h, None)
                st = [x.clone() for x in st]
                for c, (lo, a, b, hi) in zip(cuts, rect):          # the exchange, x then y
                    others = [q for q in cut4 if q[0] != c.dim]
                    for j, x in enumerate(st):
                        for gl, gh in ((lo, a), (b, hi)):
                            if gh <= gl:
                                continue
                            ghost = _sl(x, c.dim, gl - lo, gh - lo)
                            if j in c.fields:
                                ghost.copy_(_cut_tensor(_sl(mid[j], c.dim, gl, gh), others))
                            else:
                                ghost.copy_(1e3 * torch.randn(ghost.shape, generator=g,
                                                              device=x.device, dtype=x.dtype))
                for _ in range(K):
                    st = p._plain_step_func(*st, *ms, prop.dt, h, None)
            except Exception as e:                  # a grid-shaped value the slicer missed
                raise NotImplementedError(
                    f"domain decomposition: the step of tile {where} failed -- a grid-shaped "
                    f"value the cut could not pass through? {type(e).__name__}: {e}") from e
            own = [(c.dim, a - lo, b - lo, hi - lo) for c, (lo, a, b, hi) in zip(cuts, rect)]
            own_g = [(c.dim, a, b, c.n) for c, (_, a, b, _) in zip(cuts, rect)]
            for j, (x, y) in enumerate(zip(st, ref)):
                x, y = _cut_tensor(x, own), _cut_tensor(y, own_g)
                scale = y.abs().max().clamp_min(1e-30)
                err = float((x - y).abs().max() / scale)
                if not err <= rtol:
                    raise NotImplementedError(
                        f"domain decomposition: tile {where} disagrees with the global grid "
                        f"on field {prop.wavefield_names[j]} by {err:.2e} (relative) after "
                        f"2 x {K} steps: the step is not local, or holds a grid-shaped value "
                        f"the cut did not pass through")


# --------------------------------------------------------- the exchange
class Link:
    """One tile's two cut sides along one axis: ``H`` ghost cells, owned
    cells ``[a, b)`` of the tile's array along ``dim``, the neighbour ranks
    (None at the grid's edge), and the exchange's static buffers, made once
    per (fields, shape, dtype).

    ``transport``: ``'nccl'`` -- NCCL called directly
    (:mod:`sweep.parallel.nccl_direct`), CUDA only; ``'dist'`` --
    ``torch.distributed`` P2P (any backend, the CPU path); ``'nocomm'`` --
    timing only, recv <- send, no messages (wrong results)."""

    def __init__(self, dim, H, a, b, left, right, *, transport, group=None):
        self.dim, self.H, self.a, self.b = dim, H, a, b
        self.left, self.right, self.group = left, right, group
        self.sides = [(k, q) for k, q in ((0, left), (1, right)) if q is not None]
        self.transport = transport
        self._bufs = {}

    def buffers(self, like, nf):
        """``(send, recv, messages)``, each buffer ``(2, nf, *like.shape)``
        with ``H`` along ``dim``: side 0 = low, 1 = high.  Static: refilled
        every exchange."""
        key = (nf, tuple(like.shape), like.dtype)
        got = self._bufs.get(key)
        if got is None:
            sh = list(like.shape)
            sh[self.dim] = self.H
            send = torch.empty((2, nf, *sh), dtype=like.dtype, device=like.device)
            recv = torch.empty_like(send)
            msgs = None
            if self.transport == "nccl":
                from sweep.parallel.nccl_direct import Messages
                msgs = Messages([(q, send[k].view(-1), recv[k].view(-1)) for k, q in self.sides])
            got = self._bufs[key] = (send, recv, msgs)
        return got

    def exchange(self, send, recv, msgs):
        if self.transport == "nccl":
            msgs.run()
        elif self.transport == "nocomm":
            recv.copy_(send)
        else:
            ops = []
            for k, q in self.sides:
                ops.append(dist.P2POp(dist.isend, send[k], q, group=self.group))
                ops.append(dist.P2POp(dist.irecv, recv[k], q, group=self.group))
            for req in dist.batch_isend_irecv(ops):
                req.wait()


class HaloSwap(torch.autograd.Function):
    """Out-of-place halo exchange along one axis of the tensors handed in
    (the fields the exchange plan selected), in as few launches as possible:
    the fields are stacked into one tensor, both sides' edge cells packed into
    the static send buffer in one op, the messages sent (NCCL on the current
    stream), the received ghosts and the owned cells concatenated in one op,
    and the result unbound into the fields again (views).

    Forward: ghost cells <- the neighbour's owned edge cells.  Backward, the
    transpose of that copy: the ghosts' gradient goes back to the neighbour,
    which adds it to its owned edge cells; our own ghost inputs get zero
    (forward overwrote them).  Every rank runs the same sequence of
    exchanges, so the backward's messages pair up across ranks in order."""

    @staticmethod
    def forward(ctx, link, *fields):
        ctx.link = link
        dim, H, a, b = link.dim, link.H, link.a, link.b
        S = torch.stack(fields)                                   # (nf, B, 1, nz, [ny,] nx)
        send, recv, msgs = link.buffers(fields[0], len(fields))
        if len(link.sides) == 2:
            torch.stack((_sl(S, dim, a, a + H), _sl(S, dim, b - H, b)), out=send)
        else:
            k = link.sides[0][0]
            send[k].copy_(_sl(S, dim, a, a + H) if k == 0 else _sl(S, dim, b - H, b))
        link.exchange(send, recv, msgs)
        parts = [_sl(S, dim, a, b)]
        if link.left is not None:
            parts.insert(0, recv[0])
        if link.right is not None:
            parts.append(recv[1])
        out = torch.cat(parts, dim=dim) if len(parts) > 1 else _sl(S, dim, a, b).clone()
        return tuple(out.unbind(0))

    @staticmethod
    def backward(ctx, *grads):
        link = ctx.link
        dim, H, a, b = link.dim, link.H, link.a, link.b
        ref = next(g for g in grads if g is not None)
        G = torch.stack([torch.zeros_like(ref) if g is None else g for g in grads])
        n = G.shape[dim]
        send, recv, msgs = link.buffers(ref, len(grads))
        if len(link.sides) == 2:
            torch.stack((_sl(G, dim, 0, a), _sl(G, dim, b, n)), out=send)   # our ghosts
        else:
            k = link.sides[0][0]
            send[k].copy_(_sl(G, dim, 0, a) if k == 0 else _sl(G, dim, b, n))
        link.exchange(send, recv, msgs)
        pad = [0, 0] * (-dim - 1) + [a, n - b]                   # zeros on our ghosts
        gi = torch.nn.functional.pad(_sl(G, dim, a, b), pad)
        if link.left is not None:
            _sl(gi, dim, a, a + H).add_(recv[0])
        if link.right is not None:
            _sl(gi, dim, b - H, b).add_(recv[1])
        return (None, *gi.unbind(0))


class _TileExchange:
    """One tile's halo exchange over ALL its cut sides at once: the x faces,
    the y faces and, in 3-D, the corners, each a block from one neighbour,
    all the messages in one group.  A block carries the fields whose ghosts
    matter across that side -- an x face the fields exchanged along x, a y
    face those along y, a corner those in both sets (a corner is both an x
    ghost and a y ghost; a field in one set only has nothing to collect
    there) -- so the faces hold owned rows / columns only and the regions
    never overlap.  Static send / receive buffers per block.

    Two full-field copies per exchange, forward and backward each one (the
    stack of the fields), whatever the number of cut axes: the sequential
    per-axis :class:`HaloSwap` made two per axis.
    """

    def __init__(self, cuts, rect, mesh, transport, group=None):
        self.transport, self.group = transport, group
        self.union = sorted(set().union(*[set(c.fields) for c in cuts]))
        pos = {j: i for i, j in enumerate(self.union)}
        own = {c.dim: (a - lo, b - lo) for c, (lo, a, b, hi) in zip(cuts, rect)}
        self.nd = len(cuts)
        # blocks: (peer, field positions, send region, recv region); a region
        # is {dim: (lo, hi)} on the tile's array
        self.blocks = []
        axes = {c.axis: c for c in cuts}

        def region(side_of):
            """``{dim: (lo, hi)}``: per cut axis, the low ghost / the owned
            range / the high ghost (``side_of[axis]`` -1 / 0 / +1), or the
            owned edge the neighbour on that side receives (``edge=True``)."""
            out = {}
            for c in cuts:
                a, b = own[c.dim]
                out[c.dim] = {-1: (0, a), 0: (a, b), 1: (b, b + c.H if b + c.H else b)}[side_of[c.axis]]
            return out

        def edge(side_of):
            out = {}
            for c in cuts:
                a, b = own[c.dim]
                out[c.dim] = {-1: (a, a + c.H), 0: (a, b), 1: (b - c.H, b)}[side_of[c.axis]]
            return out

        sides = [-1, 1]
        for c in cuts:                                    # faces
            for d in sides:
                peer = mesh.neighbour_rank(c.axis, d)
                if peer is None:
                    continue
                side_of = {ax: 0 for ax in axes}
                side_of[c.axis] = d
                fields = [pos[j] for j in c.fields]
                self.blocks.append((peer, fields, edge(side_of), region(side_of)))
        if self.nd == 2:                                  # corners (3-D)
            both = [pos[j] for j in cuts[0].fields if j in set(cuts[1].fields)]
            for dx in sides:
                for dy in sides:
                    if not both:
                        continue
                    xn, yn = mesh.neighbour_rank("x", dx), mesh.neighbour_rank("y", dy)
                    if xn is None or yn is None:
                        continue
                    peer = mesh.rank_at(mesh.shot_group, mesh.yi + dy, mesh.xi + dx)
                    side_of = {"x": dx, "y": dy}
                    self.blocks.append((peer, both, edge(side_of), region(side_of)))
        self._bufs = {}

    @staticmethod
    def _view(O, reg):
        for dim, (lo, hi) in reg.items():
            O = _sl(O, dim, lo, hi)
        return O

    def buffers(self, like):
        """Per block ``(sel, send, recv)``, and the messages: static."""
        key = (tuple(like.shape), like.dtype)
        got = self._bufs.get(key)
        if got is None:
            items = []
            for peer, fields, sreg, rreg in self.blocks:
                sh = [len(fields)] + list(like.shape)
                for dim, (lo, hi) in sreg.items():
                    sh[dim] = hi - lo
                send = torch.empty(sh, dtype=like.dtype, device=like.device)
                recv = torch.empty_like(send)
                sel = torch.as_tensor(fields, device=like.device)
                items.append((peer, sel, send, recv, sreg, rreg))
            msgs = None
            if self.transport == "nccl":
                from sweep.parallel.nccl_direct import Messages
                msgs = Messages([(q, sd.view(-1), rc.view(-1)) for q, _, sd, rc, _, _ in items])
            got = self._bufs[key] = (items, msgs)
        return got

    def exchange(self, items, msgs):
        if self.transport == "nccl":
            msgs.run()
        elif self.transport == "nocomm":
            for _, _, sd, rc, _, _ in items:
                rc.copy_(sd)
        else:
            ops = []
            for q, _, sd, rc, _, _ in items:
                ops.append(dist.P2POp(dist.isend, sd, q, group=self.group))
                ops.append(dist.P2POp(dist.irecv, rc, q, group=self.group))
            for req in dist.batch_isend_irecv(ops):
                req.wait()


class FusedHaloSwap(torch.autograd.Function):
    """:class:`_TileExchange` as a differentiable op on the union of the
    exchanged fields.  Forward: the fields stacked (the output), each
    block's owned edge packed and sent, the received blocks written into the
    ghost regions in place.  Backward, the transpose: the ghost regions'
    gradient sent back and added to the neighbour's edge, our ghost inputs'
    gradient zero."""

    @staticmethod
    def forward(ctx, ex, *fields):
        ctx.ex = ex
        O = torch.stack(fields)
        items, msgs = ex.buffers(fields[0])
        for _, sel, send, _, sreg, _ in items:
            send.copy_(ex._view(O, sreg).index_select(0, sel))
        ex.exchange(items, msgs)
        for _, sel, _, recv, _, rreg in items:
            ex._view(O, rreg).index_copy_(0, sel, recv)
        return tuple(O.unbind(0))

    @staticmethod
    def backward(ctx, *grads):
        ex = ctx.ex
        ref = next(g for g in grads if g is not None)
        G = torch.stack([torch.zeros_like(ref) if g is None else g for g in grads])
        items, msgs = ex.buffers(ref)
        for _, sel, send, _, _, rreg in items:              # our ghosts' gradient goes back
            ghost = ex._view(G, rreg)
            send.copy_(ghost.index_select(0, sel))
            ghost.index_fill_(0, sel, 0)
        ex.exchange(items, msgs)
        for _, sel, _, recv, sreg, _ in items:
            ex._view(G, sreg).index_add_(0, sel, recv)
        return (None, *G.unbind(0))


class _SumGradOverRanks(torch.autograd.Function):
    """Identity whose backward all-reduces the gradient over the ranks: the
    wavelet's gradient, a sum of every tile's part, complete on each rank
    (as the single-domain run gives it) instead of a per-rank partial sum.
    Every rank reaches this node once, after its rollout's backward, so the
    collective pairs up."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        dist.all_reduce(g)
        return g


# ---------------------------------------------------------- the rollout
class _Tile:
    """One tile's rollout set-up: its propagator copy, models, sources,
    receivers, the receivers it owns (global indices) and its state."""


def _mask_wavelet(wavelet, n, inside):
    """The wavelet as ``(n, nt)``, ``n`` the per-step sample count (shots, or
    sources of an encoded shot), its rows for sources outside the tile zero."""
    nt = wavelet.shape[-1]
    w = wavelet.reshape(-1, nt)
    if w.shape[0] == 1 and n > 1:
        w = w.expand(n, nt)
    if w.shape[0] != n:
        raise ValueError(f"wavelet has {w.shape[0]} rows for {n} per-step samples")
    return w * inside.to(w.dtype)[:, None]


def _setup_tile(prop, cuts, rect, B, enc, wavelet, src_rt, rec_rt, models_ext, tile_prop=None):
    """``_Tile`` for the tile ``rect`` (per cut axis ``(lo, own_lo, own_hi,
    hi)``); ``models_ext`` its extended runtime models (ghosts filled), batch
    layout applied, not yet prepared."""
    t = _Tile()
    t.cuts, t.rect = cuts, rect
    if tile_prop is None:
        tile_prop = tile_propagator(prop, [(c.dim, lo, hi) for c, (lo, _, _, hi) in zip(cuts, rect)])[0]
    t.p = tile_prop
    t.models = t.p._prepare_runtime_models(models_ext)
    sw = [B, 1] + list(prop._runtime_shape())
    for c, (lo, _, _, hi) in zip(cuts, rect):
        sw[c.dim] = hi - lo
    sw = tuple(sw)
    stencil = getattr(prop.equation, "source_receiver_stencil", None)
    # sources: every one inside the tile's extended box is injected (ghosts
    # included: with K > 1 what is injected there reaches the owned cells);
    # the others move to an owned cell with a zero wavelet
    inside = torch.ones(src_rt.shape[:-1], dtype=torch.bool, device=src_rt.device)
    own = torch.ones(rec_rt.shape[:-1], dtype=torch.bool, device=rec_rt.device)
    t_src, t_rec = src_rt.clone(), rec_rt.clone()
    safe = torch.zeros_like(src_rt[0, 0])
    safe[-1] = prop.coord_offset[-1]
    for c, (lo, a, b, hi) in zip(cuts, rect):
        sx, rx = src_rt[..., c.coord], rec_rt[..., c.coord]
        inside &= (sx >= lo) & (sx < hi)
        own &= (rx >= a) & (rx < b)                        # half-open: one owner
        t_src[..., c.coord] -= lo
        t_rec[..., c.coord] -= lo
        safe[c.coord] = a - lo
    t_src = torch.where(inside.unsqueeze(-1), t_src, safe)
    t.wavelet = _mask_wavelet(wavelet, inside.numel(), inside.reshape(-1))
    t.src = SourceTorch(t_src, sw, prop.dev, enc, False, spread_kernel=stencil)
    if not bool((own == own[:1]).all()):
        raise NotImplementedError(
            "domain decomposition of the eager propagator needs the same receivers in "
            "every shot's tile: the receiver coordinates differ between shots across a "
            "cut. Pad the receiver array so each shot's spread covers the same cells.")
    t.rec_idx = own[0].nonzero().flatten()
    t_rec = t_rec[:, t.rec_idx] if len(t.rec_idx) else torch.zeros_like(rec_rt[:, :1]) + safe
    t.rec = ReceiverTorch(t_rec, gather_kernel=stencil, shape=sw)
    t.state = [torch.zeros(sw, device=prop.dev, dtype=models_ext[0].dtype) for _ in prop.wavefield_names]
    return t


def _rollout(t, nt, K, exchange):
    """A tile's time loop -- the single-domain rollout of its propagator copy
    (full tape, or chunk checkpointing as ``_rollout_ckpt``), with
    ``exchange(t, state) -> state`` after every ``K`` steps.  Returns the
    record of the tile's own receivers, ``(B, nt, nrec, len(receiver_type))``."""
    p = t.p
    wsteps = t.wavelet.movedim(-1, 0).clone(memory_format=torch.contiguous_format).unbind(0)
    records = p._records(t.models, wsteps)
    k = p._resolved_steps_per_call(records)
    if K % k:
        raise ValueError(
            f"exchange_steps={K} must be a multiple of the compiled call's {k} steps "
            f"(EagerOptions.steps_per_call)")
    p._steps_fn(k)
    hoist = p._hoists_invariants()

    def run(state, start, end, advance):
        blocks, s = [], start
        while s < end:
            state, blk, n = advance(state, s, end)
            blocks.append(blk)
            s += n
            if s % K == 0 and s < nt:
                state = exchange(t, state)
        return state, blocks

    if not (p.use_ckpt and records):
        advance = p._stepper(t.models, wsteps, t.src, t.rec, nt, False, SettledSteps(), k,
                             invariants={} if hoist else None, records=records)
        _, blocks = run(t.state, 0, nt, advance)
        return p._cat_blocks(blocks)[:, :, : len(t.rec_idx)]

    from sweep.propagator._torch_eager import CKPT_PLAIN_WARMUP
    chunk = -(-int(max(1, p.ckpt_chunks)) // k) * k
    settled = SettledSteps()
    state, blocks, start = t.state, [], 0
    if p.use_compile and CKPT_PLAIN_WARMUP:
        advance = p._stepper(t.models, wsteps, t.src, t.rec, nt, False, settled, k, records=records)
        compiled = False
        while start < min(nt, 8 + 2 * k) and not compiled:
            state, blk, n = advance(state, start, nt)
            compiled = n == k and settled.first is not None
            blocks.append(blk)
            start += n
            if start % K == 0 and start < nt:
                state = exchange(t, state)
    if p.use_compile:
        init_cuda_before_checkpoint()
    nw, nm = len(state), len(t.models)
    for start_t in range(start, nt, chunk):

        def checkpoint_chunk(*inputs, start_t=start_t):
            st = list(inputs[:nw])
            ms = list(inputs[nw:nw + nm])
            advance = p._stepper(ms, wsteps, t.src, t.rec, nt, False, settled, k,
                                 invariants={} if hoist else None, records=records)
            st, blks = run(st, start_t, min(start_t + chunk, nt), advance)
            return tuple(st), p._cat_blocks(blks)

        state, rec = ckpt_torch(checkpoint_chunk, *state, *t.models, use_reentrant=False)
        state = list(state)
        blocks.append(rec)
    return torch.cat(blocks, dim=1)[:, :, : len(t.rec_idx)]


def _prepare_global(prop, wavelet, sources, receivers, models, kwargs):
    """What every tile set-up needs from the global propagator: IO mode,
    runtime coordinates, the wavelet and the models on the device."""
    prop = _impl(prop)
    if prop._custom_gradients:
        raise NotImplementedError(
            "domain decomposition of the eager propagator does not support registered "
            "custom gradients (register_gradient): the imaging condition runs on the "
            "single-domain tape only.")
    if prop.memory_strategy == "boundary":
        raise NotImplementedError(
            "domain decomposition of the eager propagator does not support boundary "
            "saving: the tiles run the full tape or chunk checkpointing. Build the "
            "propagator with memory=Full() or memory=Ckpt(...).")
    _, B, _, _, enc = prop._normalize_io(wavelet, sources, receivers)
    kwargs = dict(kwargs)
    kwargs.setdefault("fd_pad", prop._runtime_fd_pad())
    kwargs.setdefault("shape", prop._runtime_shape())
    prop.init_abc(**kwargs)
    src_rt = prop._as_device_tensor(sources, dtype=torch.long)
    if src_rt.dim() == 2:
        src_rt = src_rt.unsqueeze(1)
    src_rt = src_rt + prop.coord_offset
    rec_rt = prop._as_device_tensor(receivers, dtype=torch.long) + prop.coord_offset
    models = models if models is not None else prop.parameters()
    models = [prop._as_device_tensor(m, dtype=torch.float32) for m in models]
    wavelet = prop._as_device_tensor(wavelet, dtype=torch.float32)
    return prop, B, enc, wavelet, src_rt, rec_rt, models


def _pad_global(prop, models, B):
    """Global physical models onto the runtime grid, batch layout applied."""
    padded = [EdgePadding.apply(m, prop._runtime_padding()) for m in models]
    return prop._runtime_batch_layout(padded, B)


# ---------------------------------------------------- single process
def decomposed_forward(prop, wavelet, sources, receivers, models=None, *, px=2, py=1,
                       exchange_steps=1, balance=False, **kwargs):
    """Every tile of a ``py x px`` decomposition in this process, the
    exchange a concatenation of slices: the record ``(B, nt, nrec,
    len(receiver_type))`` in the single-domain layout, differentiable in the
    models and the wavelet.  The oracle the tests compare the single-domain
    run against (and, with gloo, the rank path); not a way to use several
    GPUs."""
    prop, B, enc, wavelet, src_rt, rec_rt, models = _prepare_global(
        prop, wavelet, sources, receivers, models, kwargs)
    K = int(exchange_steps)
    cuts = tile_grid(prop, py, px)
    padded = _pad_global(prop, models, B)
    admit(prop, padded, cuts, K, balance=balance)
    ntiles = py * px if len(cuts) > 1 else px
    tiles = []
    for flat in range(ntiles):
        rect = _rect(cuts, _tile_index(cuts, flat))
        tiles.append(_setup_tile(prop, cuts, rect, B, enc, wavelet, src_rt, rec_rt,
                                 _cut_models(padded, cuts, rect)))
    nt = wavelet.shape[-1]
    # The tiles advance in lockstep, K steps at a time, on the full tape (the
    # rank path's ``_rollout`` has the checkpointed loop; here every tile is
    # in one process, and one exchange needs every tile's state at once).
    states = [t.state for t in tiles]
    steppers = []
    for t in tiles:
        p = t.p
        wsteps = t.wavelet.movedim(-1, 0).clone(memory_format=torch.contiguous_format).unbind(0)
        records = p._records(t.models, wsteps)
        k = p._resolved_steps_per_call(records)
        if K % k:
            raise ValueError(f"exchange_steps={K} must be a multiple of the compiled call's {k} steps")
        p._steps_fn(k)
        steppers.append(p._stepper(t.models, wsteps, t.src, t.rec, nt, False, SettledSteps(), k,
                                   invariants={} if p._hoists_invariants() else None, records=records))
        t.blocks = []
    s = 0
    while s < nt:
        end = min(s + K, nt)
        for i, t in enumerate(tiles):
            u = s
            while u < end:
                states[i], blk, n = steppers[i](states[i], u, end)
                t.blocks.append(blk)
                u += n
        s = end
        if s < nt:
            for ax in range(len(cuts)):                   # x, then y
                states = _cat_exchange(tiles, states, cuts, ax)
    recs = [t.p._cat_blocks(t.blocks)[:, :, : len(t.rec_idx)] for t in tiles]
    rec = torch.cat(recs, dim=2)
    perm = torch.cat([t.rec_idx for t in tiles]).argsort()
    return rec[:, :, perm]


def _cat_exchange(tiles, states, cuts, ax):
    """Ghost cells along cut axis ``ax`` <- the neighbour's owned cells, for
    the fields exchanged along it, every tile (``torch.cat`` of slices:
    autograd supplies the adjoint)."""
    c = cuts[ax]
    px = cuts[0].parts
    stride = 1 if ax == 0 else px
    new = []
    for i, t in enumerate(tiles):
        lo, a, b, hi = t.rect[ax]
        fields = []
        for j, f in enumerate(states[i]):
            if j not in c.fields:                 # its ghosts reach no owned cell
                fields.append(f)
                continue
            parts = [_sl(f, c.dim, a - lo, b - lo)]
            if a > lo:
                L = tiles[i - stride]
                lo_L = L.rect[ax][0]
                parts.insert(0, _sl(states[i - stride][j], c.dim, lo - lo_L, a - lo_L))
            if hi > b:
                Rt = tiles[i + stride]
                lo_R = Rt.rect[ax][0]
                parts.append(_sl(states[i + stride][j], c.dim, b - lo_R, hi - lo_R))
            fields.append(torch.cat(parts, dim=c.dim) if len(parts) > 1 else parts[0])
        new.append(fields)
    return new


# ------------------------------------------------------- one tile per rank
class EagerModelParallel:
    """One tile per rank of the eager propagator -- x strips in 2-D, an x/y
    tile grid in 3-D -- built by ``ModelParallel(prop, mesh)`` when ``prop``
    is ``impl='eager'``.

        prop = PropTorch(eq, impl="eager", shape=(nz, nx), ..., dev=f"cuda:{rank}")
        mesh = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
        ddp  = ModelParallel(prop, mesh, exchange_steps=8)
        syn  = ddp(wavelet, sources_global, receivers_global, models=[vp])
        loss = 0.5 * (syn - obs[:, :, ddp.own_receivers]).pow(2).sum()
        loss.backward()                      # vp.grad: this rank's tile; wavelet.grad

    The same call and the same semantics as the stepped ``ModelParallel``.
    ``models`` are either the GLOBAL physical models, replicated on every
    rank -- the gradient is then global-shaped with THIS rank's tile filled
    and zero elsewhere, all-reduce to assemble -- or this rank's TILE of them
    (as :meth:`MeshTopology.local_extent` cuts the axes; its gradient is the
    tile's, complete).  Either way a rank's part is the derivative of the
    GLOBAL misfit (the sum of the ranks' misfits over the receivers they own)
    with respect to its tile's model.  The wavelet's gradient, which the
    stepped class does not return, is complete on every rank.  The record
    holds the receivers this tile owns (:attr:`own_receiver_indices`);
    :meth:`gather_record` assembles the global record on rank 0.

    ``exchange_steps`` (``K``): how many steps a tile runs between two halo
    exchanges, a multiple of the compiled call's steps
    (``EagerOptions.steps_per_call``); default two calls in 2-D, one in 3-D.
    More steps between exchanges amortise the exchange's host time (on V100
    the one-shot 2-D acoustic weak-scaling efficiency goes 79 -> 87 -> 90 %
    for K = 4, 8, 16 on 2 GPUs) at the price of a wider halo, ``H = K * R``
    cells per cut side computed twice, and of more fields to exchange; in
    3-D, where the step is GPU-bound, one step a call is the cheaper side.  ``halo``: ``'nccl'`` (default; NCCL called directly
    on CUDA, ``torch.distributed`` on a CPU) or ``'dist'``.

    ``balance`` (default False: the axes are split as the stepped class splits
    them, :meth:`MeshTopology.local_extent`): cut so every tile computes the
    same number of cells -- an edge tile carries the absorbing pad, an inner
    tile a halo on both sides, and the slowest tile sets the pace.  The edge
    tiles then own fewer physical cells; the cut depends on the halo, known
    after the first call's admission, so ``balance=True`` takes the GLOBAL
    models (tile models would have to be cut before it is known).
    :meth:`local_extent` gives this rank's physical tile.

    Memory: the tiles run the full tape or chunk checkpointing as the wrapped
    propagator does (boundary saving is refused); the admission self-check
    allocates a few global-grid fields per state tensor, once.  Refused:
    boundary saving, CPML strips (``cpml_strips=True``), registered custom
    gradients, per-edge free surfaces other than the top, ``return_wavefield``.
    """

    def __init__(self, prop, mesh: MeshTopology, *, halo: str = "nccl", exchange_steps=None,
                 overlap=None, balance: bool = False):
        if not _is_eager(prop):
            raise TypeError("EagerModelParallel wraps an impl='eager' propagator")
        if overlap is not None:
            raise NotImplementedError(
                "overlap= is the stepped drivers' switch (impl='c'): the eager tiles have no "
                "communication/computation overlap. Leave it unset; exchange_steps= is the "
                "eager knob.")
        p = _impl(prop)
        if mesh.shot_groups != 1 or mesh.py * mesh.px != mesh.world_size:
            raise NotImplementedError(
                "domain decomposition of the eager propagator takes one tile grid over all "
                "ranks (MeshTopology(py, px, shot_groups=1, world_size=py * px, ...))")
        if p.ndim == 2 and mesh.py != 1:
            raise ValueError("a 2-D grid is cut along x only: MeshTopology(py=1, ...)")
        if halo not in ("nccl", "dist"):
            raise ValueError(f"halo must be 'nccl' or 'dist' for the eager propagator; got {halo!r}")
        fs = getattr(p, "fs_faces", None)
        if fs is not None:
            from sweep.equations._edges import edge_names, is_top_only_or_none
            if not is_top_only_or_none(tuple(fs)):
                free = [n for n, on in zip(edge_names(p.ndim), fs) if on]
                raise NotImplementedError(
                    f"domain decomposition does not support a per-edge free surface; got "
                    f"free faces {free}. Only a top free surface (or none) can be decomposed.")
        if exchange_steps is None:
            # 2-D: two compiled calls -- the step is host-bound and the
            # exchange's host time is what the interval amortises (V100,
            # 1000^2 one shot: 79 -> 87 % for one -> two calls).  3-D: one
            # call -- the step is GPU-bound, and a longer interval widens the
            # halo on two axes and adds fields to exchange (K=2 exchanges the
            # acoustic equation's six fields over a 144^2 instead of 136^2
            # footprint per 128^2: 53 % against 96 % weak efficiency).
            k = p._resolved_steps_per_call(True)
            exchange_steps = 2 * k if p.ndim == 2 else k
        if not isinstance(balance, bool):
            raise TypeError(f"balance must be True or False, got {balance!r}")
        self.balance = balance
        if int(exchange_steps) < 1 or int(exchange_steps) != exchange_steps:
            raise ValueError(f"exchange_steps must be a positive integer; got {exchange_steps!r}")
        self.prop, self.mesh, self.halo, self.K = prop, mesh, halo, int(exchange_steps)
        self.rank, self.world = mesh.rank, mesh.world_size
        self.dev = p.dev
        self._transport = "nccl" if halo == "nccl" and self.dev.type == "cuda" else "dist"
        if self._transport == "nccl":
            from sweep.parallel.nccl_direct import library
            if library() is None:
                self._transport = "dist"
                warnings.warn("eager DD: libnccl.so.2 cannot be opened; the halo exchange "
                              "goes through torch.distributed", stacklevel=2)
        self._admitted = {}            # (K, shape, dtype, B) -> (cuts, tile prop)
        self._links = {}
        self.own_receivers = None

    # ------------------------------------------------------------- forward
    def forward(self, wavelet, sources_global, receivers_global, models=None, **kwargs):
        """This rank's record, ``(B, nt, n_own, len(receiver_type))``, of the
        receivers it owns (:attr:`own_receiver_indices`); differentiable."""
        if kwargs.pop("return_wavefield", False):
            raise NotImplementedError("domain decomposition does not return wavefield snapshots")
        if has_tangent(wavelet, *(models or ())):
            refuse_forward_ad("Domain decomposition (eager)", "its halo exchange carries no tangent")
        prop, B, enc, wavelet, src_rt, rec_rt, models = _prepare_global(
            self.prop, wavelet, sources_global, receivers_global, models, kwargs)
        global_models = self._models_are_global(prop, models)
        if self.balance and not global_models:
            raise ValueError(
                "balance=True takes the global models: the balanced cut depends on the halo, "
                "which the first call's admission measures, so tile models cannot be cut to "
                "it beforehand")
        key = (self.K, tuple(prop._runtime_shape()), models[0].dtype, B)
        if key not in self._admitted:
            cuts = tile_grid(prop, self.mesh.py, self.mesh.px)
            padded = (_pad_global(prop, models, B) if global_models
                      else self._gather_padded(prop, models, B))
            admit(prop, padded, cuts, self.K, balance=self.balance)
            rect = _rect(cuts, self._my_index(cuts))
            self._admitted[key] = (cuts, tile_propagator(
                prop, [(c.dim, lo, hi) for c, (lo, _, _, hi) in zip(cuts, rect)])[0])
        cuts, tile_prop = self._admitted[key]
        rect = _rect(cuts, self._my_index(cuts))
        self._cuts = cuts
        self.reach = {c.axis: c.R for c in cuts}
        self.halo_width = {c.axis: c.H for c in cuts}
        self.exchanged_fields = {c.axis: [prop.wavefield_names[j] for j in c.fields] for c in cuts}
        links = [self._link(c, lo, a, b) for c, (lo, a, b, hi) in zip(cuts, rect)]
        ex = self._exchange(cuts, rect)
        if global_models:
            # this rank's tile of the replicated models: its gradient lands on
            # the tile's cells of the global leaf, zero elsewhere, the ghost
            # cells' part routed to their owner by the model halo exchange --
            # the stepped ModelParallel's semantics (all-reduce to assemble)
            ext = [c.phys_extent(i) for c, i in zip(cuts, self._my_index(cuts))]
            spec = [(c.dim, o, o + n, prop._shape_phys[c.dim]) for c, (o, n) in zip(cuts, ext)]
            models = [_cut_tensor(m, spec) for m in models]
        models_ext = self._extend_tile_models(prop, models, B, cuts, links)
        active = [link for link in links if link.sides]
        if active and wavelet.requires_grad and torch.is_grad_enabled():
            wavelet = _SumGradOverRanks.apply(wavelet)
        t = _setup_tile(prop, cuts, rect, B, enc, wavelet, src_rt, rec_rt, models_ext, tile_prop=tile_prop)
        self.own_receivers = t.rec_idx.cpu()
        if not active:                                                # a world of one
            return _rollout(t, wavelet.shape[-1], self.K, lambda t_, s: s)

        def exchange(t_, state):
            state = list(state)
            new = FusedHaloSwap.apply(ex, *[state[j] for j in ex.union])
            for j, f in zip(ex.union, new):
                state[j] = f
            return state

        return _rollout(t, wavelet.shape[-1], self.K, exchange)

    __call__ = forward

    def local_extent(self):
        """``(local_shape, offsets)`` of this rank's PHYSICAL tile, as
        :meth:`MeshTopology.local_extent` returns them: the depth axis whole,
        offsets ``(0, [oy,] ox)``.  With ``balance=True`` the cut is known
        after the first call."""
        phys = tuple(_impl(self.prop)._shape_phys)
        if not self.balance:
            return self.mesh.local_extent(phys)
        if getattr(self, "_cuts", None) is None:
            raise RuntimeError("balance=True: the cut is made by the first call's admission; "
                               "local_extent() is known after it")
        loc, offs = list(phys), [0] * len(phys)
        for c, i in zip(self._cuts, self._my_index(self._cuts)):
            offs[c.dim], loc[c.dim] = c.phys_extent(i)
        return tuple(loc), tuple(offs)

    def _my_index(self, cuts):
        return [self.mesh.xi] + ([self.mesh.yi] if len(cuts) > 1 else [])

    # ------------------------------------------------------------- models
    def _models_are_global(self, prop, models):
        phys = tuple(prop._shape_phys)
        loc, _ = self.mesh.local_extent(phys)
        nd = len(phys) - 1                                 # the cut axes: x, and y in 3-D
        got = {tuple(int(s) for s in m.shape[-nd:]) for m in models}
        if got == {phys[-nd:]}:
            return True
        if got == {tuple(loc[-nd:])}:
            return False
        axes = "x" if nd == 1 else "(y, x)"
        raise ValueError(
            f"models must all be the global ({phys[-nd:]} along {axes}) or all this rank's "
            f"tile ({tuple(loc[-nd:])}); got {sorted(got)}")

    def _tile_padding(self, prop, cuts):
        """``EdgePadding`` pads for this rank's tile: the runtime pad on the
        outer faces, ``H`` on the cut faces (replicated, then overwritten by
        the model halo exchange).  Torch pad order: x_lo, x_hi, (y_lo, y_hi,)
        z_lo, z_hi."""
        pad = list(prop._runtime_padding())
        for c in cuts:
            k = 0 if c.axis == "x" else 2
            if self.mesh.neighbour_rank(c.axis, -1) is not None:
                pad[k] = c.H
            if self.mesh.neighbour_rank(c.axis, +1) is not None:
                pad[k + 1] = c.H
        return tuple(pad)

    def _extend_tile_models(self, prop, models, B, cuts, links):
        """This rank's tile models onto its extended runtime grid: padded,
        the ghost cells then taken from the neighbours (differentiably:
        their gradient goes back to the owner), along x then y."""
        padded = [EdgePadding.apply(m, self._tile_padding(prop, cuts)) for m in models]
        padded = prop._runtime_batch_layout(padded, B)
        shapes = {tuple(m.shape) for m in padded}
        if len(shapes) != 1:
            raise NotImplementedError("tile models of different shapes under DD")
        for link in links:
            if link.sides:
                padded = list(HaloSwap.apply(link, *padded))
        return padded

    def _gather_padded(self, prop, models, B):
        """The global padded models from every rank's tile, for the admission
        self-check (a one-time gather; the tiles' outer pads concatenate to
        the global runtime grid)."""
        pad = list(prop._runtime_padding())
        for axis, k in (("x", 0), ("y", 2)):
            if axis == "y" and prop.ndim == 2:
                continue
            if self.mesh.neighbour_rank(axis, -1) is not None:
                pad[k] = 0
            if self.mesh.neighbour_rank(axis, +1) is not None:
                pad[k + 1] = 0
        mine = [EdgePadding.apply(m.detach(), tuple(pad)).cpu() for m in models]
        parts = [None] * self.world
        dist.all_gather_object(parts, mine)
        full = []
        for i in range(len(models)):
            rows = []
            for yi in range(self.mesh.py):
                row = [parts[self.mesh.rank_at(0, yi, xi)][i] for xi in range(self.mesh.px)]
                rows.append(torch.cat(row, dim=-1))
            full.append((torch.cat(rows, dim=-2) if len(rows) > 1 else rows[0]).to(prop.dev))
        return prop._runtime_batch_layout(full, B)

    # ------------------------------------------------------------- links
    def _exchange(self, cuts, rect):
        key = tuple((c.axis, c.H, tuple(c.fields), a - lo, b - lo) for c, (lo, a, b, hi) in zip(cuts, rect))
        if key not in self._links:
            self._links[key] = _TileExchange(cuts, rect, self.mesh, self._transport)
        return self._links[key]

    def _link(self, c, lo, a, b):
        key = (c.axis, c.H, a - lo, b - lo)
        if key not in self._links:
            self._links[key] = Link(c.dim, c.H, a - lo, b - lo,
                                    self.mesh.neighbour_rank(c.axis, -1),
                                    self.mesh.neighbour_rank(c.axis, +1), transport=self._transport)
        return self._links[key]

    # ------------------------------------------------------------- records
    @property
    def own_receiver_indices(self):
        """Global receiver indices whose traces this rank's record carries,
        as a tuple (the stepped ``ModelParallel``'s spelling; empty before
        any call).  :attr:`own_receivers` is the same as a tensor."""
        return () if self.own_receivers is None else tuple(self.own_receivers.tolist())

    def gather_record(self, tile_record):
        """The global record ``(B, nt, nrec, C)`` on rank 0 (None elsewhere),
        from every rank's tile record."""
        if self.world == 1:
            return tile_record
        payload = (self.own_receivers.tolist(), tile_record.detach().cpu())
        gathered = [None] * self.world if self.rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if self.rank != 0:
            return None
        # assemble_tile_records places along axis -2; the canonical record's
        # receivers are axis 2 of 4, so move them last and back
        moved = [(idx, rc.movedim(2, -1).movedim(-1, -2)) for idx, rc in gathered]
        full = assemble_tile_records([(idx, rc) for idx, rc in moved])
        return None if full is None else full.movedim(-2, 2)
