"""One-call domain-decomposed propagator (model-parallel, multi-GPU).

Wraps the per-step stepped driver + NCCL halo exchange (proven bitwise in
``test/dd_nccl_*.py``) behind a single object that behaves like the
single-domain ``PropTorch`` — same call signature, and the forward is
AUTOGRAD-TRANSPARENT, so a plain ``loss.backward()`` gives you the model
gradient (no manual adjoint). Callers never hand-slice the model, fill halos,
remap source/receiver coordinates, or drive the time loop:

    mesh = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    prop = PropTorch(eq, shape=(nz, nx), dh=10., dt=dt, nt=nt, abcn=20,
                     source_type=["h1"], receiver_type=["h1"], dev=dev)
    ddp  = ModelParallel(prop, mesh)                   # decompose across the mesh
    vp   = vp_tile.requires_grad_()                    # this rank's model tile
    syn  = ddp(wavelet, sources_global, receivers_global, models=[vp])
    loss = 0.5 * (syn - obs_tile).pow(2).sum()
    loss.backward()                                    # -> vp.grad (this tile)
    full = ddp.gather_record(syn)                      # rank-0 assembled record

(For an arbitrary adjoint source instead of an L2 misfit, use
``syn.backward(gradient=adjoint_source_tile)``.)

What it does internally (x-cut decomposition, v1):
  * ``MeshTopology.local_extent`` -> this rank's tile shape / x-offset;
  * a ONE-TIME NCCL model-halo exchange fills each tile's cut-side pad with the
    true neighbour material, so the user supplies only its own tile (no global
    model needed) and never edge-replicates by hand;
  * ``partition_global_coords`` keeps only this tile's sources/receivers
    (shifted to tile-local; a zero-amplitude dummy keeps nsrc/nrec >= 1);
  * the equation family selects the exchange protocol — acoustic advances one
    full step then exchanges ``u_now`` (M wide); elastic uses the half-step
    protocol (phase-1 velocity, exchange v; phase-2 stress + tail, exchange s);
  * the backward replays boundary-saving reconstruction with the per-step
    lambda/recon halo exchanges and ``cut_face_mask`` set, then returns each
    tile's INTERIOR model gradient (cut-side pad gradients belong to the
    neighbour and are dropped — proven correct in test_dd_*_backward_two_tile).

v1 scope: x-cut (``py == 1``), acoustic2d/3d + elastic2d/3d, fixed acquisition
geometry per instance (capture once; later ``forward`` calls rebind models +
wavelet).  Free surface supported.  Boundary-saving storage/dtype are inherited
from the wrapped prop's memory config (``PropTorch(memory=MemoryOptions(
boundary=BoundaryOptions(storage=..., storage_dtype=...)))``) — e.g. int8 to
shrink the boundary ring for finer grids; defaults to gpu/fp32.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from sweep.parallel._topology import MeshTopology
from sweep.parallel.mesh import ModelParallelMesh
from sweep.parallel.routing import partition_global_coords
from sweep.propagator.torch import PropTorch
from sweep.propagator._stepped import (
    SteppedBackwardRunner,
    SteppedBindingRunner,
    acoustic_adj_pairs,
    acoustic_psi_pairs,
)

X_LO_BIT, X_HI_BIT = 1, 2
Y_LO_BIT, Y_HI_BIT = 16, 32

# Per-family wavefield-list geometry (see acoustic.h / elastic.h ::bind and
# sweep/propagator/_stepped.py).  ``nv``/``nphys``/``nrecon`` are 2-D, 3-D.
_FAMILIES = {
    "acoustic": dict(nwf=(9, 12), nphys=None, nv=None, nrecon=(3, 3),
                     half_step=False),
    "elastic": dict(nwf=(15, 36), nphys=(5, 9), nv=(2, 3), nrecon=(7, 12),
                    half_step=True),
}


def _family_of(equation) -> str:
    name = type(equation).__name__.lower()
    if "elastic" in name:
        return "elastic"
    if "acoustic" in name:
        return "acoustic"
    raise ValueError(
        f"ModelParallel v1 supports acoustic/elastic only, got {type(equation).__name__}"
    )


class _DDForward(torch.autograd.Function):
    """Differentiable bridge so a domain-decomposed forward composes with plain
    autograd — ``loss.backward()`` on a misfit of the returned record populates
    each model tensor's ``.grad``, identical to single-domain ``PropTorch``.

    forward runs the DD stepped forward (no autograd inside — gradients flow
    through the C++/NCCL stepped calls, not the tape); backward runs the DD
    adjoint (:meth:`ModelParallel._run_adjoint`) with the incoming record-gradient as
    the adjoint source. The ``.grad`` matches the SHAPE of the model you passed:

      * a per-tile model (``local_shape``) gets its tile's interior gradient —
        the canonical model-parallel case (each rank owns + optimises its tile);
      * a replicated global model (``global_shape``) gets a global-shaped grad
        with THIS rank's tile filled (``all_reduce`` across ranks to assemble
        the full gradient; only valid when the global model fits on each GPU).
    """

    @staticmethod
    def forward(ctx, ddp, wavelet, sources, receivers, *models):
        ctx.ddp = ddp
        ctx.shapes = [tuple(m.shape) for m in models]
        # autograd.Function.forward runs with grad DISABLED; the first call's
        # one-time _capture builds a transient prop graph and backward()s it to
        # grab the bp params, so re-enable grad around the call. The detached
        # models keep the DD forward itself off the tape (gradients flow through
        # _DDForward.backward, not autograd); the stepped loop has its own
        # no_grad. Without this, a first forward on the autograd path crashes
        # ("element 0 does not require grad").
        with torch.enable_grad():
            rec = ddp.forward(wavelet, sources, receivers,
                              models=[m.detach() for m in models])
        return rec.detach().clone()

    @staticmethod
    def backward(ctx, grad_record):
        ddp = ctx.ddp
        tile_grads = ddp._run_adjoint(grad_record.contiguous())
        out = []
        for shp, g in zip(ctx.shapes, tile_grads):
            if shp == ddp.global_shape:        # replicated global model
                full = g.new_zeros(shp)
                gt = g.reshape(ddp.local_shape)
                if ddp.ndim == 2:
                    full[:, ddp.x0:ddp.x0 + ddp.nxp] = gt
                else:
                    full[:, ddp.y0:ddp.y0 + ddp.nyp,
                         ddp.x0:ddp.x0 + ddp.nxp] = gt
                out.append(full)
            else:                              # per-tile model
                out.append(g.reshape(shp))
        # grads align to (ddp, wavelet, sources, receivers, *models)
        return (None, None, None, None) + tuple(out)


class ModelParallel:
    """Run a single-domain :class:`PropTorch` decomposed across GPUs (model
    parallel / domain decomposition) — a strategy wrapper in the spirit of
    ``torch.nn.parallel.DistributedDataParallel``::

        prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, nt=nt, abcn=abcn, ...)
        ddp  = ModelParallel(prop, mesh)        # mesh = MeshTopology(py, px, ...)
        syn  = ddp(wavelet, sources, receivers, models=[vp])   # same call as prop
        loss = 0.5 * (syn - obs).pow(2).sum(); loss.backward()  # autograd-transparent

    ``prop`` specifies the GLOBAL problem (equation, grid spacing, nt, abcn,
    spatial order, source/receiver types, free surface, PML, B) — the propagator
    you would build to run on one GPU if the model fit.  ``ModelParallel`` reads
    that spec and builds per-tile solvers with cut-aware padding internally: a
    model-parallel grid cannot reuse one global prop's symmetric pad, so ``prop``
    is a config carrier, not the compute object (constructing it is cheap — the
    big buffers are allocated lazily at forward, which only the per-tile solvers
    do).  ``mesh`` is the :class:`MeshTopology` (x-cut ``py=1``; 3-D may add a
    y-cut ``py>1`` — see :func:`sweep.parallel.balanced_grid`)."""

    def __init__(self, prop, mesh: MeshTopology) -> None:
        # Read the global-problem spec off the single-domain propagator.
        # dh/dt are stored as buffer tensors (dh per-axis); recover plain Python
        # for the per-tile prop (scalar dh when the spacing is uniform).
        self._global_prop = prop
        equation = prop.equation
        global_shape = prop._shape_phys
        if torch.is_tensor(prop.dh):
            _dh = prop.dh.flatten().tolist()
            dh = _dh[0] if len(set(_dh)) == 1 else _dh
        else:
            dh = prop.dh
        dt = float(prop.dt) if torch.is_tensor(prop.dt) else prop.dt
        nt, abcn, B = prop.nt, prop.abcn, prop.B
        spatial_order = prop.equation.so
        source_type, receiver_type = prop.source_type, prop.receiver_type
        free_surface, pml_type = prop.free_surface, prop.pml_type
        dev = prop.dev
        model_parallel = mesh
        # Boundary-saving storage/dtype are inherited from the wrapped prop's
        # memory config (set the normal way via
        # ``PropTorch(memory=MemoryOptions(boundary=BoundaryOptions(...)))``), so
        # compressing (fp16/bf16/int8) or offloading the DD boundary ring to fit
        # finer grids is a first-class API choice. Defaults (gpu/fp32) reproduce
        # the original v1 behaviour.
        _bcfg = getattr(prop, "boundary_saving_config", None) or {}
        self._bstorage = _bcfg.get("storage", "gpu")
        self._bdtype = _bcfg.get("storage_dtype", "fp32")

        self.topo = model_parallel
        self.global_shape = tuple(int(s) for s in global_shape)
        self.ndim = len(self.global_shape)
        if self.ndim not in (2, 3):
            raise ValueError("global_shape must be 2-D or 3-D")
        self.equation = equation
        self.family = _family_of(equation)
        self.dev = dev
        self.nt = int(nt)
        self.abcn = int(abcn)
        self.so = int(spatial_order)
        self.M = self.so // 2
        self.pad = self.abcn + self.M
        self.free_surface = bool(free_surface)
        self.world = self.topo.world_size
        self.rank = self.topo.rank
        self._st = dict(_FAMILIES[self.family])

        self.local_shape, self.offsets = self.topo.local_extent(self.global_shape)
        self.x0 = self.offsets[-1]                       # global x-origin of tile
        self.nxp = self.local_shape[-1]
        if self.ndim == 3:
            self.y0 = self.offsets[1]
            self.nyp = self.local_shape[1]
        # interior offsets (self.lo / self.hi / self.lo_y / self.hi_y) are
        # derived from self.prop.padding after the propagator is built, so they
        # follow the propagator's (possibly cut-aware, asymmetric) pad.

        # cut-face mask + active halo axes: a cut exists wherever a neighbour is
        # present.  x-cut (px>1) and, for 3-D, y-cut (py>1) -> 2x2 when both.
        self.cut_mask = 0
        axes = []
        if self.topo.neighbour_rank("x", -1) is not None:
            self.cut_mask |= X_LO_BIT
        if self.topo.neighbour_rank("x", +1) is not None:
            self.cut_mask |= X_HI_BIT
        if self.topo.px > 1:
            axes.append("x")
        if self.ndim == 3:
            if self.topo.neighbour_rank("y", -1) is not None:
                self.cut_mask |= Y_LO_BIT
            if self.topo.neighbour_rank("y", +1) is not None:
                self.cut_mask |= Y_HI_BIT
            if self.topo.py > 1:
                axes.append("y")
        self.axes = tuple(axes)

        pml = pml_type or ("cpmls" if self.family == "elastic" else "cpmlr")
        self.prop = PropTorch(
            equation, backend="torch", impl="c", shape=self.local_shape, dev=dev,
            dh=dh, dt=dt, source_type=list(source_type),
            receiver_type=list(receiver_type), abcn=abcn,
            free_surface=self.free_surface, pml_type=pml, nt=nt, B=B,
            use_ckpt=False,
            boundary_saving_config={
                "enabled": True,
                # inherited from the wrapped prop (PropTorch memory= API);
                # gpu/fp32 by default, or fp16/bf16/int8 / cpu for finer grids.
                "storage": self._bstorage,
                "storage_dtype": self._bdtype,
                "transfer_interval": 1, "pinned_memory": False},
            model_parallel=self.topo,
        )

        # Per-side interior offsets in the runtime (shape_cuda) frame. The prop
        # allocates a cut-aware pad — cut faces carry only the M halo, edge
        # faces carry abcn+M — so read its per-side low pad instead of assuming
        # the symmetric ``abcn+M``.  self.prop.padding is in F.pad order:
        # (x_lo, x_hi, [y_lo, y_hi,] z_lo, z_hi); add M for the stencil halo.
        ppad = self.prop.padding
        self.lo = ppad[0] + self.M
        self.hi = self.lo + self.nxp
        if self.ndim == 3:
            self.lo_y = ppad[2] + self.M
            self.hi_y = self.lo_y + self.nyp

        # process group + halo exchangers (one per direction-set, reused)
        self.mesh = (ModelParallelMesh(grid=(self.topo.py, self.topo.px))
                     if self.world > 1 else None)
        self._fwd_halo = None
        self._bwd_halo = None
        self._model_halo = None
        self._halo_sl_cache = {}     # field.ndim -> cut-axis crop slice tuple

        # comm/compute overlap (acoustic forward): a dedicated comm stream runs
        # step's halo exchange while step's interior computes. Eligible
        # only for x-face cuts (the phase-split forward emits x cut strips) AND
        # when no source sits in a cut strip (checked per-call in forward).
        self._comm_stream = None
        self._comm_evt = None
        self._overlap_ok = (self.world > 1 and self.cut_mask != 0
                            and (self.cut_mask & ~0x3) == 0)

        self._captured = False
        self._geom_key = None     # (src,rec,wavelet) bytes of the live geometry
        self._nwf = self._st["nwf"][0 if self.ndim == 2 else 1]
        self._nrecon = self._st["nrecon"][0 if self.ndim == 2 else 1]
        if self.family == "elastic":
            self._nv = self._st["nv"][0 if self.ndim == 2 else 1]
            self._nphys = self._st["nphys"][0 if self.ndim == 2 else 1]

    # ------------------------------------------------------------------ utils
    def _halo(self, attr):
        if self.world == 1:
            return None
        from sweep.parallel.fast_halo import FastHaloSet
        cur = getattr(self, attr)
        if cur is None:
            cur = FastHaloSet(self.mesh, self.M, self.axes)
            setattr(self, attr, cur)
        return cur

    def _halo_view(self, field):
        """Crop the field to owned±M in each CUT axis (plus-stencil: corners
        unread, so x and y halos exchange independently). The slice tuple is
        loop-invariant (lo/hi/M fixed once captured), so build it once per
        field ndim and reuse it for every step's exchange."""
        sl = self._halo_sl_cache.get(field.ndim)
        if sl is None:
            s = [slice(None)] * field.ndim
            if self.topo.px > 1:
                s[-1] = slice(self.lo - self.M, self.hi + self.M)
            if self.ndim == 3 and self.topo.py > 1:
                s[-2] = slice(self.lo_y - self.M, self.hi_y + self.M)
            sl = tuple(s)
            self._halo_sl_cache[field.ndim] = sl
        return field[sl]

    def _exchange(self, halo, tensor):
        if halo is not None:
            halo.exchange(self._halo_view(tensor))

    def _exchange_group(self, halo, tensors):
        """Halo-exchange a group of fields in ONE batched P2P (elastic velocity
        / stress groups). Collapses ``len(tensors)`` separate NCCL rounds into a
        single isend/irecv+wait — the per-step latency win for the multi-field
        elastic protocol (acoustic exchanges a single field, so it uses
        :meth:`_exchange`/the overlap path instead)."""
        if halo is not None:
            halo.exchange_group([self._halo_view(t) for t in tensors])

    def _src_away_from_cuts(self, sg) -> bool:
        """True when no source sits within M of an x-cut line (k*nxp). The
        forward injects the source in phase 2 — after phase 1's cut strips have
        been exchanged — so a source IN a strip would cross the cut one step
        late. Comm/compute overlap is only correct (bit-exact vs serial) when
        every source is clear of the strips; otherwise we fall back to serial.
        ``sg`` is the global source-coord tensor (B, nsrc, ndim), x at index 0."""
        cuts = [k * self.nxp for k in range(1, self.topo.px)]
        if not cuts:
            return True
        xs = sg.reshape(-1, self.ndim)[:, 0].tolist()
        return all(abs(int(x) - c) > self.M for x in xs for c in cuts)

    def _slice_tile(self, model):
        """Accept a global (Nz,[Ny,]Nx) or already-tiled (Nz,[Ny,]nyp,nxp) array."""
        arr = model.detach().cpu().numpy() if torch.is_tensor(model) else np.asarray(model)
        if tuple(arr.shape) == self.global_shape:
            if self.ndim == 2:
                return arr[:, self.x0: self.x0 + self.nxp].copy()
            return arr[:, self.y0: self.y0 + self.nyp,
                       self.x0: self.x0 + self.nxp].copy()
        if tuple(arr.shape) == self.local_shape:
            return arr.copy()
        raise ValueError(
            f"model shape {arr.shape} matches neither global {self.global_shape} "
            f"nor tile {self.local_shape}"
        )

    def _repad_runtime_model(self, rt: torch.Tensor, tile: np.ndarray) -> None:
        """Write a physical tile into the runtime-padded model ``rt`` in place,
        edge-replicating the pad (FS-aware z), then leaving the cut-side x pad
        for the model-halo exchange to overwrite.  Matches the propagator's
        edge padding (np.pad(edge), proven bitwise for the NCCL checks)."""
        lo_x, hi_x = self.lo, self.hi
        ztop = self.M if self.free_surface else self.pad
        t = torch.as_tensor(tile, device=rt.device, dtype=rt.dtype)
        nz = tile.shape[0]
        if self.ndim == 2:
            t = t.reshape(1, 1, nz, self.nxp)
            rt.zero_()
            rt[..., ztop:ztop + nz, lo_x:hi_x] = t
            # x edges over interior-z band, then z edges over full width
            rt[..., ztop:ztop + nz, :lo_x] = rt[..., ztop:ztop + nz, lo_x:lo_x + 1]
            rt[..., ztop:ztop + nz, hi_x:] = rt[..., ztop:ztop + nz, hi_x - 1:hi_x]
            rt[..., :ztop, :] = rt[..., ztop:ztop + 1, :]
            rt[..., ztop + nz:, :] = rt[..., ztop + nz - 1: ztop + nz, :]
        else:
            ny = tile.shape[1]
            lo_y, hi_y = self.lo_y, self.hi_y
            t = t.reshape(1, 1, nz, ny, self.nxp)
            rt.zero_()
            rt[..., ztop:ztop + nz, lo_y:hi_y, lo_x:hi_x] = t
            zi, yi = slice(ztop, ztop + nz), slice(lo_y, hi_y)
            # x edges (z,y interior) -> y edges (z interior, full x) -> z edges (full)
            rt[..., zi, yi, :lo_x] = rt[..., zi, yi, lo_x:lo_x + 1]
            rt[..., zi, yi, hi_x:] = rt[..., zi, yi, hi_x - 1:hi_x]
            rt[..., zi, :lo_y, :] = rt[..., zi, lo_y:lo_y + 1, :]
            rt[..., zi, hi_y:, :] = rt[..., zi, hi_y - 1:hi_y, :]
            rt[..., :ztop, :, :] = rt[..., ztop:ztop + 1, :, :]
            rt[..., ztop + nz:, :, :] = rt[..., ztop + nz - 1: ztop + nz, :, :]

    def _set_models(self, model_tiles: List[np.ndarray]):
        """Rebind both forward and backward params' runtime models from physical
        tiles: edge re-pad, then one NCCL model-halo exchange to fill cut pads."""
        mhalo = self._halo("_model_halo")
        for rt_f, rt_b, tile in zip(self.fp.models, self.bp.models, model_tiles):
            self._repad_runtime_model(rt_f, tile)
            self._exchange(mhalo, rt_f)
            rt_b.copy_(rt_f)

    # --------------------------------------------------------------- capture
    def _capture(self, wavelet, loc_src, loc_rec, tiles):
        impl = self.prop._backend_impl
        cap = {}
        f_orig, b_orig = impl.forward_func, impl.backward_bs_func

        # The capture's public forward/backward must carry the cut-face mask,
        # exactly like the real stepped runs (see forward()/gradient()). The
        # cut-aware pad makes an interior tile thin on its cut faces (M halo, no
        # abcn PML); without the mask the kernel treats a cut face as a full
        # abcn+M PML, so phys_y1 - phys_y0 (the boundary-save extent) goes
        # NEGATIVE and the compact boundary kernel launches with a <= 0 grid ->
        # CUDA "invalid configuration argument". This only bites when a tile is
        # thin enough to flip the sign — py>=3 (both y-faces cut) is the first
        # such case; x stays safe only because its per-tile interior is wider.
        def fwrap(p):
            p.cut_face_mask = self.cut_mask
            out = f_orig(p); cap["fp"] = p; cap["fraw"] = out; return out

        def bwrap(p):
            p.cut_face_mask = self.cut_mask
            out = b_orig(p); cap["bp"] = p; return out

        impl.forward_func, impl.backward_bs_func = fwrap, bwrap
        models = [torch.tensor(t, device=self.dev, requires_grad=True) for t in tiles]
        syn = self.prop(wavelet, loc_src, loc_rec, models=models)
        rec = syn[0] if isinstance(syn, (tuple, list)) else syn
        rec.sum().backward()
        impl.forward_func, impl.backward_bs_func = f_orig, b_orig

        self.fp, self.bp = cap["fp"], cap["bp"]
        self.f_func, self.b_func = f_orig, b_orig
        self._cuda_ndim = cap["fraw"][2].ndim

        L = list(self.fp.wavefields)
        if not L:
            L = [torch.zeros_like(self.fp.models[0]) for _ in range(self._nwf)]
        self.L_fwd = L
        self.L_adj = list(self.bp.adjoint_wavefields)
        if not self.L_adj:
            self.L_adj = [torch.zeros_like(self.bp.models[0]) for _ in range(self._nwf)]
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(self._nrecon)]
        # acoustic grads_out = [grad_wavelet, *model_grads] (size = models + 1);
        # elastic has no wavelet grad (size = models).
        if self.family == "acoustic":
            self.gbufs = ([torch.zeros_like(self.bp.forward_source)]
                          + [torch.zeros_like(m) for m in self.bp.models])
        else:
            self.gbufs = [torch.zeros_like(m) for m in self.bp.models]
        self.bp.grads_out = self.gbufs
        # acoustic backward produces source/receiver illumination; elastic none
        if self.family == "acoustic":
            self.illum = [torch.zeros_like(self.bp.models[0]),
                          torch.zeros_like(self.bp.models[0])]
        else:
            self.illum = []
        self.bp.illum_out = self.illum
        self.record = torch.zeros_like(cap["fraw"][2])
        self.fp.record_out = self.record
        # Cache the canonical wavelet shape + the per-tile source/receiver counts
        # so per-shot _set_geometry can validate and reshape. The wavelet lives on
        # the forward params as write-only ``fp.source``; the SAME wavelet is also
        # exposed (readable) as ``bp.forward_source``, which is what we read here.
        fsrc = getattr(self.bp, "forward_source", None)
        self._fs_shape = tuple(fsrc.shape) if fsrc is not None else None
        self._cap_ls_shape = tuple(np.asarray(loc_src).shape)
        self._cap_lr_shape = tuple(np.asarray(loc_rec).shape)
        # Fail loud NOW (on every path, not just multi-shot) if the C bindings
        # ever rename a geometry field that _set_geometry writes.
        for obj, who, fields in (
            (self.fp, "ForwardInput", ("sources_loc", "receivers_loc", "source")),
            (self.bp, "BackwardInput",
             ("forward_sources_loc", "adjoint_sources_loc", "forward_source")),
        ):
            missing = [f for f in fields if not hasattr(obj, f)]
            if missing:
                raise AttributeError(
                    f"ModelParallel._set_geometry expects {who} fields {missing}, "
                    f"absent on this build — the C param bindings changed; update "
                    f"_set_geometry's field names.")
        self._captured = True

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`forward` so a ``ModelParallel`` can be invoked like
        the single-domain ``PropTorch`` (``ddp(...)`` == ``ddp.forward(...)``).
        ModelParallel is a composition wrapper, not an ``nn.Module``, so it does
        not get ``__call__`` for free."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------- geometry
    def _runtime_coords(self, loc):
        """Physical per-tile coords (1, n, ndim) -> runtime int32 on device,
        applying the SAME per-axis low-side shift as ``_c.py`` (PML low pad +
        stencil halo M, cut-aware via ``self.prop.padding``)."""
        pad, M = self.prop.padding, self.M
        out = np.array(loc, dtype=np.int32).copy()
        out[..., 0] += pad[0] + M
        if self.ndim == 3:
            out[..., 1] += pad[2] + M
        out[..., -1] += pad[-2] + M
        return torch.as_tensor(out, dtype=torch.int32, device=self.dev)

    def _set_geometry(self, ls, lr, wav):
        """Re-apply this call's source/receiver/wavelet onto the captured params
        WITHOUT a full re-capture. The C++ params store geometry as plain coord
        arrays plus the wavelet, so multi-shot only needs those fields swapped;
        the wavefield/record/grad buffers are reused untouched. ``ls``/``lr``
        already encode per-tile ownership (dummy + zero wavelet off-tile).
        Forward and backward params use different field NAMES for the same
        quantities (see probe): fp.source/sources_loc/receivers_loc vs
        bp.forward_source/forward_sources_loc/adjoint_sources_loc."""
        # The captured fp/bp coord arrays + record buffers are sized for the FIRST
        # shot's per-tile source/receiver counts; a later shot with a different
        # count would silently mis-shape the wavelet reshape or overrun the record.
        # Require a fixed per-tile (#sources, #receivers) across shots — fail loud.
        if ls.shape != self._cap_ls_shape or lr.shape != self._cap_lr_shape:
            raise ValueError(
                f"ModelParallel multi-shot: per-tile source/receiver count changed "
                f"since capture (src {self._cap_ls_shape}->{tuple(ls.shape)}, "
                f"rec {self._cap_lr_shape}->{tuple(lr.shape)}). Re-geometry needs a "
                f"fixed per-tile layout (same #sources/shot and a fixed receiver "
                f"spread); the captured buffers are sized for the first shot.")
        src_rt = self._runtime_coords(ls)        # (1, npts, ndim)
        rec_rt = self._runtime_coords(lr)        # (1, nrec, ndim)
        fs = None
        if self._fs_shape is not None:
            fs = torch.as_tensor(wav, dtype=torch.float32, device=self.dev).reshape(self._fs_shape)
        # forward params
        self.fp.sources_loc = src_rt
        self.fp.receivers_loc = rec_rt
        if fs is not None:
            self.fp.source = fs
        # backward params (reconstruct forward from source; adjoint injected at
        # the receivers -> adjoint_sources_loc carries the receiver coords)
        self.bp.forward_sources_loc = src_rt
        self.bp.adjoint_sources_loc = rec_rt
        if fs is not None:
            self.bp.forward_source = fs

    # --------------------------------------------------------------- forward
    def forward(self, wavelet, sources_global, receivers_global, models):
        """Run the DD forward; return this rank's tile record (raw CUDA layout).

        ``models`` is a list of global or already-tiled physical arrays, OR
        ``None`` to REUSE the model already edge-padded and halo-exchanged by a
        previous forward.  An FWI epoch fires many shots through the SAME model
        (only the source moves), so re-padding the runtime model and running the
        NCCL model-halo collective on every shot is pure waste; pass ``models``
        on the first shot of the epoch and ``models=None`` for the rest to skip
        it.  (This is explicit on purpose — the propagator never guesses whether
        an in-place optimiser step changed the model, which would risk silently
        running on a stale model.)

        AUTOGRAD: if any model tensor ``requires_grad``, the returned record is
        differentiable — ``loss.backward()`` on a misfit populates each model's
        ``.grad`` (== :meth:`gradient` of the residual), exactly like the
        single-domain ``PropTorch`` autograd path. With no grad-requiring model
        it stays on the fast ``torch.no_grad`` stepped path; the explicit
        :meth:`gradient` adjoint API remains for manual control."""
        if (models is not None
                and any(torch.is_tensor(m) and m.requires_grad for m in models)):
            if not all(torch.is_tensor(m) for m in models):
                raise TypeError(
                    "ModelParallel autograd forward needs every model as a tensor "
                    "when any requires grad (got a mix of tensor/non-tensor).")
            return _DDForward.apply(
                self, wavelet, sources_global, receivers_global, *models)
        sg = self._prepare_call(wavelet, sources_global, receivers_global, models)
        fhalo = self._halo("_fwd_halo")
        with torch.no_grad():
            if self.family == "acoustic":
                self._forward_loop_acoustic(fhalo, sg)
            else:
                self._forward_loop_elastic(fhalo)

        # A tile owning no real receivers carries only a dummy receiver; its
        # record is meaningless.  Zero it so a residual/adjoint derived from it
        # injects nothing on this tile — otherwise the dummy's recorded value
        # becomes a spurious adjoint source that corrupts the gradient near the
        # cut (matters for y-cut/2x2, where receivers at a fixed y leave whole
        # tile rows without receivers; x-cut with x-spread receivers never hit
        # it).
        if not self._own_rec_idx:
            self.record.zero_()
        # hand the DD-consistent ring to the backward params
        self.bp.boundary_gpu = list(self.fp.boundary_gpu)
        self.bp.u_last_two = self.fp.last_two
        return self.record

    def _prepare_call(self, wavelet, sources_global, receivers_global, models):
        """Per-call forward setup shared by every step loop: slice the model (or
        reuse it for ``models=None``), partition the source/receiver coords to
        this tile, (re)capture or re-apply the geometry, rebind the model, set
        the cut-face mask, and zero the forward buffers. Returns the GLOBAL
        source tensor ``sg`` (the acoustic overlap path needs it to rule out a
        source sitting in a cut strip)."""
        if models is None:
            if not self._captured:
                raise RuntimeError(
                    "ModelParallel.forward(models=None) reuses the previously set "
                    "model, but no forward has run yet — pass models on the first "
                    "call.")
            tiles = None
        else:
            tiles = [self._slice_tile(m) for m in models]
        sg = torch.as_tensor(np.asarray(sources_global), dtype=torch.int64)
        rg = torch.as_tensor(np.asarray(receivers_global), dtype=torch.int64)
        loc_s, mask_s = partition_global_coords(sg, self.topo, self.global_shape)
        loc_r, mask_r = partition_global_coords(rg, self.topo, self.global_shape)
        self._owns_src = bool(mask_s.any())
        self._own_rec_idx = torch.nonzero(mask_r[0], as_tuple=False).flatten().tolist()

        dummy = [1, 1] if self.ndim == 2 else [1, 1, 1]
        if self._owns_src:
            ls = loc_s[mask_s].reshape(1, -1, self.ndim).to(torch.int32).numpy()
            # Wavelet handling for the ENCODED SUPERSHOT case. A shared (nt,)
            # wavelet broadcasts to every owned source (unchanged). A PER-SOURCE
            # encoded wavelet (nsrc, nt) — B virtual sources each with its own
            # signed wavelet — must be subset to THIS tile's owned sources so it
            # aligns with `ls` (already subset by mask_s). Subset BEFORE the host
            # copy to avoid materialising the full (B, nt) array on CPU.
            if torch.is_tensor(wavelet):
                wsel = (wavelet[mask_s[0].to(wavelet.device)]
                        if wavelet.ndim == 2 else wavelet)
                wav = wsel.detach().cpu().numpy().astype(np.float32)
            else:
                wav = np.asarray(wavelet, dtype=np.float32)
                if wav.ndim == 2:
                    wav = wav[mask_s[0].cpu().numpy()]
            if wav.ndim == 2 and wav.shape[0] != ls.shape[1]:
                raise ValueError(
                    "ModelParallel encoded supershot: per-tile wavelet rows "
                    f"{wav.shape[0]} != owned sources {ls.shape[1]}")
        else:
            ls = np.array([[dummy]], dtype=np.int32)
            wav = np.zeros(self.nt, dtype=np.float32)
        if self._own_rec_idx:
            lr = loc_r[0, mask_r[0]].reshape(1, -1, self.ndim).to(torch.int32).numpy()
        else:
            lr = np.array([[dummy]], dtype=np.int32)

        geom_key = (ls.tobytes(), lr.tobytes(), wav.tobytes())
        if not self._captured:
            self._capture(wav, ls, lr, tiles)
            self._geom_key = geom_key
        elif geom_key != self._geom_key:
            # source/receiver/wavelet changed since capture (multi-shot) — swap
            # them on the captured params; fixed geometry hits the cache (no-op).
            self._set_geometry(ls, lr, wav)
            self._geom_key = geom_key
        # Re-pad the runtime model + run the NCCL model-halo only when the caller
        # supplied a (possibly updated) model; models=None reuses the buffers.
        if tiles is not None:
            self._set_models(tiles)
        # ALL forward kernels are now cut-aware (in_pml uses phys_x0/x1 or
        # && !cut_*): the forward needs the mask too, else cut-side interior
        # cells fall into the zero-coeff PML branch and drift in the last ulp
        # against the single-domain reference (the asymmetric-pad invariant).
        self.fp.cut_face_mask = self.cut_mask
        for t in self.L_fwd:
            t.zero_()
        self.record.zero_()
        return sg

    def _forward_loop_acoustic(self, fhalo, sg):
        """Acoustic forward time loop. Uses true comm/compute overlap (phase-1
        cut strips exchanged async on a comm stream while phase-2 interior
        computes, then compute waits for comm) when eligible — x-face cuts only
        and no source in a cut strip — else a serial step-then-exchange loop.
        Both are bit-identical (the overlap is a pure reordering)."""
        runner = SteppedBindingRunner(
            self.f_func, self.fp, self.L_fwd, acoustic_psi_pairs(self.ndim))
        if self._overlap_ok and self._src_away_from_cuts(sg):
            if self._comm_stream is None:
                self._comm_stream = torch.cuda.Stream()
                self._comm_evt = torch.cuda.Event()
            comm, evt = self._comm_stream, self._comm_evt
            compute = torch.cuda.current_stream()
            for it in range(self.nt):
                runner.run_phase(it + 1, 1)
                evt.record()
                un = self._halo_view(runner.u_next)
                with torch.cuda.stream(comm):
                    comm.wait_event(evt)
                    fhalo.exchange_start(un)        # copy-send + P2P (no wait)
                runner.run_phase(it + 1, 2)         # interior, overlaps P2P
                with torch.cuda.stream(comm):
                    fhalo.exchange_finish(un)       # wait P2P + copy-recv
                compute.wait_stream(comm)
        else:
            for it in range(self.nt):
                runner.run_to(it + 1)
                self._exchange(fhalo, runner.u_now)

    def _forward_loop_elastic(self, fhalo):
        """Elastic forward time loop: phase-1 velocity update + batched velocity-
        halo exchange, phase-2 stress update + batched stress-halo exchange.
        Elastic slots don't rotate, so the field lists are fixed across steps."""
        runner = SteppedBindingRunner(
            self.f_func, self.fp, self.L_fwd, psi_pairs=(), u_blocks=())
        vel = [self.L_fwd[f] for f in range(self._nv)]
        stress = [self.L_fwd[f] for f in range(self._nv, self._nphys)]
        for it in range(self.nt):
            runner.run_phase(it + 1, 1)
            self._exchange_group(fhalo, vel)
            runner.run_phase(it + 1, 2)
            self._exchange_group(fhalo, stress)

    # -------------------------------------------------------------- gradient
    def _run_adjoint(self, adjoint_source_tile):
        """Internal VJP: run the DD backward for an adjoint source (raw CUDA
        layout, this tile's receivers) and return ``[grad_model_tile, ...]``
        (interior). Drives :class:`_DDForward.backward`; users get gradients via
        autograd (``loss.backward()`` / ``record.backward(gradient=adjoint)``),
        not by calling this directly."""
        if not self._captured:
            raise RuntimeError("forward() must run before the adjoint")
        self.bp.adjoint_source = torch.as_tensor(
            adjoint_source_tile, device=self.dev, dtype=torch.float32)
        for t in self.L_adj + self.recon + self.gbufs + self.illum:
            t.zero_()
        self.bp.cut_face_mask = self.cut_mask
        bhalo = self._halo("_bwd_halo")
        nv = self._nv if self.family == "elastic" else None

        with torch.no_grad():
            if self.family == "acoustic":
                br = SteppedBackwardRunner(
                    self.b_func, self.bp, self.L_adj, self.recon,
                    adj_pairs=acoustic_adj_pairs(self.ndim))
                for it in range(self.nt - 1, -1, -1):
                    br.run_segment(it + 1, it)
                    if it == 0:
                        break
                    self._exchange(bhalo, br.lambda_now)
                    self._exchange(bhalo, br.recon_u_now)
            else:
                br = SteppedBackwardRunner(
                    self.b_func, self.bp, self.L_adj, self.recon,
                    adj_pairs=(), adj_u_blocks=(), recon_u_blocks=())
                # fixed elastic slots -> precompute each phase's exchange group
                # (adjoint + recon fields) and batch into one P2P per phase
                ph1 = ([self.L_adj[f] for f in range(nv)]
                       + [self.recon[f] for f in range(nv, self._nphys)])
                ph2 = ([self.L_adj[f] for f in range(nv, self._nphys)]
                       + [self.recon[f] for f in range(nv)])
                for it in range(self.nt - 1, 0, -1):     # elastic BS floor it==1
                    br.run_phase(it + 1, it, 1)
                    self._exchange_group(bhalo, ph1)
                    br.run_phase(it + 1, it, 2)
                    self._exchange_group(bhalo, ph2)

        # crop the runtime model grad to the physical tile interior (z is
        # FS-aware: top pad = M under free surface; cut-side x-pad grad belongs
        # to the neighbour and is dropped — proven in test_dd_*_backward).
        ztop = self.M if self.free_surface else self.pad
        nz = self.global_shape[0]
        if self.ndim == 2:
            interior = (..., slice(ztop, ztop + nz), slice(self.lo, self.hi))
        else:
            interior = (..., slice(ztop, ztop + nz),
                        slice(self.lo_y, self.hi_y), slice(self.lo, self.hi))
        # grads_out slot 0 is grad_wavelet for acoustic; model grads are the rest
        model_grads = self.gbufs[1:] if self.family == "acoustic" else self.gbufs
        out = [g[interior].clone() for g in model_grads]
        # Shot-parallel: with shot_groups>1 each group ran a DIFFERENT shot on
        # the SAME tile, so the FWI gradient (a sum over shots) needs the per-
        # tile gradients summed across the shot process group (the shot_groups
        # ranks sharing this (yi,xi) tile). One all_reduce per model leaves every
        # rank holding the shot-summed tile gradient. Pure model-parallel
        # (shot_groups==1) skips it.
        if (self.topo.shot_groups > 1 and self.mesh is not None
                and self.mesh.shot_pg is not None):
            import torch.distributed as dist
            for g in out:
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.mesh.shot_pg)
        return out

    # ---------------------------------------------------------------- gather
    def gather_record(self, tile_record):
        """Assemble the global record on rank 0 (returns None on other ranks)."""
        if self.world == 1:
            return tile_record
        import torch.distributed as dist
        payload = (self._own_rec_idx, tile_record.detach().cpu())
        gathered = [None] * self.world
        dist.gather_object(payload, gathered if self.rank == 0 else None, dst=0)
        if self.rank != 0:
            return None
        # place each tile's receiver columns at their global index
        ncomp_axis = tile_record.ndim
        full = None
        nrec_global = max(max(idx) for idx, _ in gathered if idx) + 1
        for idx, rc in gathered:
            if not idx:
                continue
            if full is None:
                shape = list(rc.shape)
                shape[-2] = nrec_global
                full = torch.zeros(shape, dtype=rc.dtype)
            for j, gi in enumerate(idx):
                full[..., gi, :] = rc[..., j, :]
        return full
