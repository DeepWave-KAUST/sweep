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

import os
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
    # LSRTM: two coupled acoustic fields in one wavefield list (background +
    # scattered), so nwf and nrecon are the acoustic numbers doubled.  Both
    # fields advance together in one step and BOTH get their halo exchanged.
    "lsrtm": dict(nwf=(18, 24), nphys=None, nv=None, nrecon=(6, 6),
                  half_step=False),
}


# Which equations ModelParallel can actually run, and their wavefield family.
#
# The criterion is NOT the class name. DD advances the solver one step at a
# time and exchanges halos between steps, so the equation's CUDA forward AND
# backward have to honour the stepped range (``p.it_begin`` / ``p.it_end``).
# Only these do -- see csrc/cuda/equations/{acoustic2d, acoustic3d,
# acoustic_vrz3d, elastic2d, elastic3d}/{forward,backward}.cu.
#
# This used to be a substring match on the class name, which accepted every
# ``Acoustic*`` / ``Elastic*`` variant in the library. The ones without a
# stepped forward do not fail on the way in: their time loop is a plain
# ``for (it = 0; it < p.nt; ++it)``, so a "run one step" call runs the WHOLE
# record and DD then exchanges halos of a wavefield that already reached nt.
# Silently wrong, which is worse than unsupported. ``AcousticVRZ`` (2-D) is
# the clearest case -- its 3-D sibling IS stepped, so the name gives no hint.
#
# Adding an equation here means all three of: its forward.cu and backward.cu
# implement the stepped range; its wavefield list matches the family geometry
# in ``_FAMILIES``; and a DD-vs-single parity test covers it (test/dd_corner_*).
_DD_EQUATIONS = {
    "Acoustic": "acoustic",             # csrc/cuda/equations/acoustic2d
    "Acoustic3D": "acoustic",           # csrc/cuda/equations/acoustic3d
    "AcousticVRZ3D": "acoustic",        # csrc/cuda/equations/acoustic_vrz3d
    "Elastic": "elastic",               # elastic2d and elastic3d -- the 3-D
                                        # class is also named ``Elastic``,
                                        # exported as ``Elastic3D``
    # csrc/cuda/equations/acoustic_lsrtm3d -- forward is stepped; the backward
    # is NOT yet, so gradients under DD raise instead of running the whole
    # record per stepped call (see _run_adjoint).  The 2-D AcousticLSRTM is
    # not stepped at all and stays unsupported.
    "AcousticLSRTM3D": "lsrtm",
}


def _family_of(equation) -> str:
    # Walk the MRO so a subclass of a supported equation still works. Every
    # equation in the library derives straight from First/SecondOrderEquation,
    # never from a sibling, so this cannot smuggle in an unsupported one.
    for klass in type(equation).__mro__:
        family = _DD_EQUATIONS.get(klass.__name__)
        if family is not None:
            return family
    raise NotImplementedError(
        f"domain decomposition does not support {type(equation).__name__}. It "
        f"needs an equation whose CUDA forward and backward implement the "
        f"stepped range (it_begin/it_end): "
        f"{', '.join(sorted(_DD_EQUATIONS))} -- Elastic covers 2-D and 3-D, "
        f"and note AcousticVRZ3D is stepped while the 2-D AcousticVRZ is not. "
        f"An equation without it would not raise, it would run the full record "
        f"on every stepped call."
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
        # ``prop.free_surface`` is only a bool.  It used to BE the whole boundary
        # spec ("is the top face free?"), but with dev's per-edge free surface it
        # is merely ``any(fs_faces)``, so passing it down would silently turn a
        # bottom/left/right free surface into a TOP one on every tile -- wrong
        # PML layout, wrong C-side fs bitmask, wrong image mirror, and a wrong
        # ``ztop`` in this file's own slicing.  Per-edge free surfaces are not
        # supported under domain decomposition (a cut face must never carry a
        # free surface), so refuse rather than degrade the physics silently.
        _fs_faces = getattr(prop, "fs_faces", None)
        if _fs_faces is not None:
            from sweep.equations._edges import edge_names, is_top_only_or_none
            if not is_top_only_or_none(tuple(_fs_faces)):
                _free = [n for n, on in zip(edge_names(prop.ndim), _fs_faces) if on]
                raise NotImplementedError(
                    "domain decomposition does not support a per-edge free "
                    f"surface; got free faces {_free}. Only a top free surface "
                    "(or none) can be decomposed -- any other face would have to "
                    "be reconciled against the tile cut faces."
                )
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
        # Boundary tail truncation (BoundaryOptions.tail_steps) is inherited
        # like storage/dtype.  Every tile reads the SAME value off the wrapped
        # prop's config, which is what keeps the truncated reverse loop's stop
        # step globally consistent across ranks (see _run_adjoint).
        self._btail = int(_bcfg.get("tail_steps") or 0)

        self.topo = model_parallel
        self.global_shape = tuple(int(s) for s in global_shape)
        self.ndim = len(self.global_shape)
        if self.ndim not in (2, 3):
            raise ValueError("global_shape must be 2-D or 3-D")
        self.equation = equation
        self.family = _family_of(equation)
        # VRZ is an "acoustic"-family variant (variable density).  Its CUDA
        # forward kernels now implement the phased (comm/compute overlap) path
        # (acoustic_vrz3d/forward.cu), so VRZ is eligible for the same forward
        # overlap as acoustic3d.  The BACKWARD stays serial for the whole
        # acoustic family (only elastic has a phased backward), so VRZ's
        # backward still uses the step-then-exchange loop.
        self._is_vrz = "vrz" in type(equation).__name__.lower()
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

        # Inherit the wrapped prop's formulation verbatim. There is deliberately
        # no fallback: PropBase.__init__ already resolved a None pml_type to
        # ``equation.default_pml_type`` before we read it, so ``pml_type`` here
        # is always a concrete string. The old ``or ("cpmls" if elastic else
        # "cpmlr")`` was therefore unreachable, and the rule it encoded was
        # wrong anyway — it guessed the formulation from a substring of the
        # class name (``_family_of``), which puts AcousticVTI1st in the
        # "acoustic" family and would have handed it 'cpmlr' when its staggered
        # step unpacks the 8 profiles of 'cpmls'. The equation's own
        # ``default_pml_type`` is the only thing that knows.
        pml = pml_type
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
                # tail truncation shrinks each tile's boundary ring to the
                # last tail_steps steps (None = full length); the C++ side
                # indexes it in shifted saved-step coordinates either way.
                "tail_steps": self._btail or None,
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
        self._halo_sl_cache = {}     # (field.ndim, axis) -> crop slice tuple

        # comm/compute overlap (acoustic + VRZ forward): a dedicated comm stream
        # runs step's halo exchange while step's interior computes. Eligible
        # only for x-face cuts (the phase-split forward emits x cut strips) AND
        # when no source sits in a cut strip (checked per-call in forward).
        # SWEEP_DD_DISABLE_OVERLAP=1 forces the serial step-then-exchange path —
        # the bit-exact reference for validating the overlap, and a production
        # safety escape hatch.
        self._comm_stream = None
        self._comm_evt = None
        _disable_overlap = os.environ.get("SWEEP_DD_DISABLE_OVERLAP", "") not in ("", "0")
        self._overlap_ok = (self.world > 1 and self.cut_mask != 0
                            and (self.cut_mask & ~0x3) == 0
                            and not _disable_overlap)

        self._captured = False
        # Two-stage capture. ``bp`` -- and every adjoint-side buffer derived from
        # it -- stays None until a gradient is actually asked for, so a caller's
        # no_grad first call allocates NO adjoint wavefields and NO boundary
        # ring (no_grad means no_grad). ``_need_adjoint`` is the sticky signal,
        # set by forward() at the only point that still sees the caller's
        # un-detached models; _prepare_call promotes the capture on it.
        self.bp = None
        self._need_adjoint = False
        self._model_dtype = None  # pinned by the first capture; see _capture
        self._geom_key = None     # (src,rec,wavelet) bytes of the live geometry
        self._nwf = self._st["nwf"][0 if self.ndim == 2 else 1]
        self._nrecon = self._st["nrecon"][0 if self.ndim == 2 else 1]
        if self.family == "elastic":
            self._nv = self._st["nv"][0 if self.ndim == 2 else 1]
            self._nphys = self._st["nphys"][0 if self.ndim == 2 else 1]

    # ------------------------------------------------------------------ utils
    def _halo(self, attr):
        """One FastHaloSet PER CUT AXIS (dict keyed by axis, in self.axes
        order). Each axis exchanges its own view (see :meth:`_halo_view`), so
        the sets stay per-axis; a shared multi-axis set would key exchangers
        by a single view's data_ptr and cross-wire the axis strips."""
        if self.world == 1:
            return None
        from sweep.parallel.fast_halo import FastHaloSet
        cur = getattr(self, attr)
        if cur is None:
            cur = {ax: FastHaloSet(self.mesh, self.M, (ax,))
                   for ax in self.axes}
            setattr(self, attr, cur)
        return cur

    def _halo_view(self, field, ax):
        """Crop the field to owned±M along the EXCHANGE axis only; every other
        axis keeps its FULL runtime extent. The perpendicular global-PML pad
        cells are live (the in_pml branch updates them every step and their
        stencil reads the cut halo), so the exchange strips must cover them.
        Cropping the non-exchange axis to owned±M (the old single-view code)
        left the (cut face x perpendicular PML band) halo cells permanently
        zero — a hard reflecting wall at each cut/model-edge corner. Only
        2-axis meshes (2x2+) ever cropped a perpendicular axis, which is why
        every single-axis split was bit-exact while PYxPX>1 diverged."""
        sl = self._halo_sl_cache.get((field.ndim, ax))
        if sl is None:
            s = [slice(None)] * field.ndim
            if ax == "x":
                s[-1] = slice(self.lo - self.M, self.hi + self.M)
            else:
                s[-2] = slice(self.lo_y - self.M, self.hi_y + self.M)
            sl = tuple(s)
            self._halo_sl_cache[(field.ndim, ax)] = sl
        return field[sl]

    def _exchange(self, halo, tensor):
        if halo is not None:
            for ax, hs in halo.items():
                hs.exchange(self._halo_view(tensor, ax))

    def _exchange_group(self, halo, tensors):
        """Halo-exchange a group of fields in ONE batched P2P per cut axis
        (elastic velocity / stress groups). Collapses ``len(tensors)`` separate
        NCCL rounds into one isend/irecv+wait per axis — the per-step latency
        win for the multi-field elastic protocol (acoustic exchanges a single
        field, so it uses :meth:`_exchange`/the overlap path instead)."""
        if halo is not None:
            for ax, hs in halo.items():
                hs.exchange_group([self._halo_view(t, ax) for t in tensors])

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
        tiles: edge re-pad, then one NCCL model-halo exchange to fill cut pads.

        Unconditionally off the tape, like the stepped loops: these are runtime
        buffer writes and the DD gradient flows through ``_DDForward.backward``,
        never through them. The no_grad also pins the grad MODE these buffers
        are touched in, which matters because ``fast_halo`` caches an exchanger
        (and the strip VIEWS it holds) per ``data_ptr``, built once on first
        use. Let the views be built under one mode and copy_'d into under the
        other and autograd refuses the mix: "a view was created in no_grad mode
        and its base ... has been modified inplace with grad mode enabled".
        """
        mhalo = self._halo("_model_halo")
        # Before the adjoint capture there is no bp to mirror into. Build an
        # explicit None-per-model list rather than zip()ing an empty one: zip
        # would silently truncate to ZERO iterations, skipping the re-pad and
        # the NCCL model halo entirely -- wrong physics, no error.
        bmodels = (list(self.bp.models) if self.bp is not None
                   else [None] * len(model_tiles))
        with torch.no_grad():
            for rt_f, rt_b, tile in zip(self.fp.models, bmodels, model_tiles):
                self._repad_runtime_model(rt_f, tile)
                self._exchange(mhalo, rt_f)
                if rt_b is not None:
                    rt_b.copy_(rt_f)

    # --------------------------------------------------------------- capture
    def _capture(self, wavelet, loc_src, loc_rec, tiles, need_adjoint):
        # ``need_adjoint`` False = FORWARD-ONLY capture: the probe runs under
        # no_grad with non-grad models, so _c.py computes requires_backward
        # False, forces use_boundary_saving off and allocates NEITHER the
        # adjoint wavefields NOR the boundary ring. Only fp is captured; bp
        # stays None until _prepare_call promotes (see there).
        #
        # torch.no_grad() around the first call is fine either way (forward-only
        # never needs grad; the adjoint capture lifts it), but inference_mode is
        # a one-way door: enable_grad does NOT lift it, and -- for BOTH capture
        # kinds -- the params captured here would hold inference tensors that
        # reject the in-place writes _set_models does on every later call. Say
        # so now rather than let either failure surface far from the cause.
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "ModelParallel's first call runs a one-time capture whose C "
                "params must stay writable across later calls, which "
                "torch.inference_mode() forbids (and the adjoint capture also "
                "differentiates a probe forward). Make the first call outside "
                "inference_mode (torch.no_grad() is fine), or warm the "
                "propagator up with one call before entering inference_mode.")
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

        # Allocate the probe's models BEFORE the try: this is the one full-tile
        # allocation in _capture, i.e. where a real run OOMs, and nothing is
        # installed yet when it does.
        # Reuse the dtype the FIRST capture settled on. The steady-state path is
        # dtype-tolerant (_repad_runtime_model casts each tile to the runtime
        # buffer's dtype), so a caller may hand a float64 model to any call
        # after the first; a promotion re-capture that inherited that dtype
        # would build a float64 probe and the C kernel would reject it
        # ("expected scalar type Float but found Double"). Pinning the dtype
        # makes a promoted instance land in exactly the state it would have
        # reached had it been gradient-capable from its first call. None on the
        # first capture, so that one still infers from the tile as before.
        models = [torch.tensor(t, device=self.dev, dtype=self._model_dtype,
                               requires_grad=need_adjoint) for t in tiles]
        self._model_dtype = models[0].dtype
        # The ADJOINT probe is differentiated to grab the backward params, so it
        # needs grad MODE even when the caller has switched it off — but it only
        # ever runs once a gradient has actually been asked for, so the textbook
        # first call ("generate observed data", which nobody wraps in grad) takes
        # the forward-only branch instead and keeps the caller's no_grad intact.
        # (_DDForward.forward re-enables grad for the same reason: an
        # autograd.Function's forward also runs with grad disabled.) The probe's
        # graph is local to this call; only the C param objects are kept.
        #
        # The kernel swap lives INSIDE the try so any raise — including a
        # half-done swap — still restores. Left installed, fwrap/bwrap would
        # route every later stepped call through these closures, and since
        # _captured stays False a retry would read them back as its "originals"
        # and nest one more layer per failure.
        #
        # Cost worth knowing: the ADJOINT probe allocates the adjoint wavefields
        # and boundary ring, so the first GRADIENT call is heavier than a bare
        # forward. One-time and inherent to capturing the backward params, not a
        # leak. On the forward-only branch bwrap never fires (no backward runs),
        # which is exactly why cap holds no "bp" there. b_func still comes from
        # b_orig, which needs no probe at all.
        try:
            impl.forward_func, impl.backward_bs_func = fwrap, bwrap
            if need_adjoint:
                with torch.enable_grad():
                    syn = self.prop(wavelet, loc_src, loc_rec, models=models)
                    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
                    rec.sum().backward()
            else:
                with torch.no_grad():
                    self.prop(wavelet, loc_src, loc_rec, models=models)
        finally:
            impl.forward_func, impl.backward_bs_func = f_orig, b_orig

        self.fp, self.bp = cap["fp"], cap.get("bp")
        self.f_func, self.b_func = f_orig, b_orig
        self._cuda_ndim = cap["fraw"][2].ndim

        L = list(self.fp.wavefields)
        if not L:
            L = [torch.zeros_like(self.fp.models[0]) for _ in range(self._nwf)]
        self.L_fwd = L
        if self.bp is not None:
            self._bind_adjoint_buffers()
        else:
            # Forward-only capture: every adjoint-side buffer is exactly what a
            # no_grad caller must NOT pay for. Empty lists (not absent
            # attributes) so _run_adjoint's zero-loop and _set_geometry stay
            # well-typed; _run_adjoint refuses outright until promotion.
            self.L_adj, self.recon = [], []
            self.coupling, self.adj_coeffs = [], []
            self.gbufs, self.illum = [], []
        self.record = torch.zeros_like(cap["fraw"][2])
        self.fp.record_out = self.record
        # Cache the canonical wavelet shape + the per-tile source/receiver counts
        # so per-shot _set_geometry can validate and reshape. Read it off
        # ``fp.source``, which a forward-only capture also has: the binding is
        # ``def_readwrite`` (csrc/bindings/module.cpp), not write-only, and it is
        # the SAME tensor object bp exposes as ``forward_source`` (_c.py assigns
        # both from one ``wavelet`` local, and .contiguous() on an already
        # contiguous tensor returns self), so the shape is identical either way.
        fsrc = getattr(self.fp, "source", None)
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
            if obj is None:       # forward-only capture: no bp to validate yet
                continue
            missing = [f for f in fields if not hasattr(obj, f)]
            if missing:
                raise AttributeError(
                    f"ModelParallel._set_geometry expects {who} fields {missing}, "
                    f"absent on this build — the C param bindings changed; update "
                    f"_set_geometry's field names.")
        self._captured = True

    def _bind_adjoint_buffers(self):
        """Allocate + bind every adjoint-side buffer on the captured ``bp``.

        Called from :meth:`_capture` ONLY when the backward params exist, i.e.
        once a gradient has actually been asked for.  A forward-only capture
        must leave all of this unallocated -- together with the C-side adjoint
        wavefields and boundary ring that ``requires_backward=False`` already
        suppresses, that is the whole memory win of the lazy split."""
        self.L_adj = list(self.bp.adjoint_wavefields)
        if not self.L_adj:
            self.L_adj = [torch.zeros_like(self.bp.models[0]) for _ in range(self._nwf)]
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(self._nrecon)]
        # VRZ variable-density gradient is a spatial divergence of the coupling
        # field c/e = lambda*vp*grad(p), so under DD the divergence at a cut seam
        # needs the neighbour's c/e.  Materialise the six coupling buffers and bind
        # them to bp.adjoint_workspace (the phased VRZ backward reads c/e from
        # there) so _run_adjoint can halo-exchange them between the build and
        # divergence sub-steps.  Plain acoustic's pointwise u_tt*lambda gradient
        # needs no such exchange, so it keeps self.coupling empty.
        if self._is_vrz:
            self.coupling = [torch.zeros_like(self.bp.models[0]) for _ in range(6)]
            # C0/Cx/Cy/Cz: the fused adjoint's transpose fast-path reads these coeffs
            # over [ix-M,ix+M] -> into the cut halo.  They are model-only (constant
            # within a backward), so build once + halo-exchange once (before the reverse
            # loop) rather than per step.  Without their exchanged halo, cut-adjacent
            # physical cells (now on the cut-aware fast-path) read a 0 coeff halo -> the
            # adjoint (hence gradient) drifts at the source.
            self.adj_coeffs = [torch.zeros_like(self.bp.models[0]) for _ in range(4)]
            self.bp.adjoint_workspace = self.coupling + self.adj_coeffs  # [0-5]=c/e, [6-9]=C0,Cx,Cy,Cz
        else:
            self.coupling = []
            self.adj_coeffs = []
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
        arrays plus the wavelet, so multi-shot only needs those fields swapped
        plus the two COUNT-sized buffers (record ~ nrec, grad-wavelet ~ nsrc)
        refreshed; every model-sized buffer (wavefields, recon, model grads,
        illum) AND the boundary ring are reused untouched. ``ls``/``lr`` already
        encode per-tile ownership (dummy + zero wavelet off-tile). Forward and
        backward params use different field NAMES for the same quantities (see
        probe): fp.source/sources_loc/receivers_loc vs
        bp.forward_source/forward_sources_loc/adjoint_sources_loc.

        Per-tile source/receiver COUNTS change every iter in the encoded OBN
        path (re-seeded shared shots partition differently across tiles), so we
        reallocate ONLY the two small count-sized buffers here rather than
        forcing a full re-capture. The forced-recapture path reallocated every
        model-sized buffer + the boundary ring each iter and leaked ~one
        adjoint-wavefield set per iter (the old buffers stayed pinned through
        the transient capture graph); reusing them in place is both leak-free
        and ~1 fwd+bwd cheaper per iter."""
        # Receiver count changed -> reallocate the (N, nrec, nt) record buffer
        # (nrec is axis -2 for single- AND multi-channel records) and rebind it
        # as the stepped forward's output. Cheap: nrec*nt floats.
        if lr.shape != self._cap_lr_shape:
            new_rec_shape = list(self.record.shape)
            new_rec_shape[-2] = int(lr.shape[1])
            self.record = torch.zeros(
                new_rec_shape, dtype=self.record.dtype,
                device=self.record.device)
            self.fp.record_out = self.record
            self._cap_lr_shape = tuple(lr.shape)
        # Source count changed -> forward_source is (B, nsrc, nt), so the
        # grad-wavelet buffer grads_out[0] (== zeros_like(forward_source)) must
        # be resized to match, else the C adjoint's per-source write to
        # grad_wavelet overruns. Model grads (grads_out[1:]) stay model-sized.
        # Elastic has no grad-wavelet slot / no forward_source (_fs_shape None).
        if ls.shape != self._cap_ls_shape:
            if self._fs_shape is not None:
                self._fs_shape = (self._fs_shape[0], int(ls.shape[1]),
                                  self._fs_shape[2])
                # gbufs only exists once the adjoint has been captured; a
                # forward-only instance still needs the _fs_shape update above
                # (it reshapes this shot's wavelet into fp.source).
                if self.family == "acoustic" and self.bp is not None:
                    self.gbufs[0] = torch.zeros(
                        self._fs_shape, dtype=self.gbufs[0].dtype,
                        device=self.gbufs[0].device)
                    self.bp.grads_out = self.gbufs
            self._cap_ls_shape = tuple(ls.shape)
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
        # the receivers -> adjoint_sources_loc carries the receiver coords).
        # Skipped before the adjoint capture: promotion re-captures with the
        # LIVE geometry, so bp is born correct rather than patched up here.
        if self.bp is not None:
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

        AUTOGRAD: if any model tensor ``requires_grad`` AND grad mode is on, the
        returned record is differentiable — ``loss.backward()`` on a misfit
        populates each model's ``.grad`` (== :meth:`gradient` of the residual),
        exactly like the single-domain ``PropTorch`` autograd path. Otherwise it
        stays on the forward-only stepped path; the explicit :meth:`gradient`
        adjoint API remains for manual control.

        MEMORY: the forward-only path allocates neither the adjoint wavefields
        nor the nt-scaled boundary ring — the capture that binds them is
        deferred until a gradient is first asked for. So wrapping a modelling /
        observed-data / line-search forward in ``torch.no_grad()`` is not
        cosmetic: it is how you avoid paying for gradient machinery you never
        use. An instance promoted to gradient-capable keeps that machinery for
        its lifetime (``use_boundary_saving`` is frozen at capture), so build a
        separate ``ModelParallel`` for pure-forward work if you want it to stay
        light."""
        # ``torch.is_grad_enabled()`` is part of the predicate because
        # ``requires_grad`` is a TENSOR property, blind to grad mode: without it
        # a caller who writes ``with torch.no_grad(): ddp(..., models=[vp])``
        # around the inversion leaf — a line-search trial, a QC forward on the
        # current iterate, observed data through a model that happens to carry
        # requires_grad — would still take the autograd path and pay for the
        # whole nt-scaled boundary ring and the adjoint wavefields it explicitly
        # opted out of. no_grad means no_grad, exactly as it does for every
        # other torch op.
        if (models is not None and torch.is_grad_enabled()
                and any(torch.is_tensor(m) and m.requires_grad for m in models)):
            if not all(torch.is_tensor(m) for m in models):
                raise TypeError(
                    "ModelParallel autograd forward needs every model as a tensor "
                    "when any requires grad (got a mix of tensor/non-tensor).")
            # The ONLY frame that still sees the caller's UN-detached models:
            # _DDForward.forward re-enters forward() with models detached and
            # autograd.Function.forward runs with grad off, so one frame down
            # this fact is unrecoverable. Must stay INSIDE this branch -- set at
            # function top, the reentrant call would recompute False from the
            # detached models and erase it. Sticky on purpose: once an instance
            # has been asked for a gradient, a later no_grad forward must not
            # downgrade it back to a forward-only capture.
            self._need_adjoint = True
            return _DDForward.apply(
                self, wavelet, sources_global, receivers_global, *models)
        sg = self._prepare_call(wavelet, sources_global, receivers_global, models)
        fhalo = self._halo("_fwd_halo")
        with torch.no_grad():
            if self.family == "acoustic":
                self._forward_loop_acoustic(fhalo, sg)
            elif self.family == "lsrtm":
                self._forward_loop_lsrtm(fhalo)
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
        # hand the DD-consistent ring to the backward params (a forward-only
        # capture has neither: fp.boundary_gpu is [] and fp.last_two is the
        # empty placeholder, because use_boundary_saving was never on)
        if self.bp is not None:
            self.bp.boundary_gpu = list(self.fp.boundary_gpu)
            self.bp.u_last_two = self.fp.last_two
        # Clone: ``self.record`` is a live buffer that the NEXT call zeroes in
        # _prepare_call, so handing it out aliases every shot of an observed-data
        # loop to one tensor that goes to zero on the following iteration. Until
        # now the autograd path masked this for grad-carrying models (it returns
        # a detached clone), but no_grad calls are routed here on purpose, so the
        # buffer must not escape. The record is small next to a wavefield.
        return self.record.clone()

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
            self._capture(wav, ls, lr, tiles, self._need_adjoint)
            self._geom_key = geom_key
        elif self._need_adjoint and self.bp is None:
            # LAZY ADJOINT PROMOTION: the first capture was forward-only and a
            # gradient is now wanted. It has to happen HERE -- before THIS
            # call's forward -- not at backward time: the boundary ring is
            # written BY the forward (see the save_forward_* calls in the CUDA
            # forwards) and fp.use_boundary_saving was frozen False when the
            # forward-only fp was captured, so a bp grafted on afterwards would
            # reconstruct from a ring nobody ever wrote (and stepped+BS
            # TORCH_CHECKs a non-empty boundary_gpu). Re-capture with THIS
            # call's geometry so _geom_key stays honest and _set_geometry is
            # not needed. Fires at most once per instance (bp is set after).
            if tiles is None:
                raise RuntimeError(
                    "ModelParallel's first grad-requiring call must pass "
                    "models=[...]: the forward-only first capture allocated no "
                    "adjoint buffers, so models=None has nothing to reuse.")
            # fast_halo keys exchangers by data_ptr and each one holds strip
            # VIEWS of the tensor it was built on -- a strong ref to that base.
            # The re-capture replaces fp.models and L_fwd, so a kept cache would
            # pin the superseded buffers for the process lifetime. Rebuilding is
            # rank-local (it only constructs P2POp objects, no collective) and
            # numerically inert (a pure strip copy-send/copy-recv).
            # _bwd_halo is still None -- no adjoint has run on this instance.
            self._fwd_halo = self._model_halo = None
            # Release the forward-only capture's forward state BEFORE the probe
            # allocates its own, so promotion never holds two full forward sets
            # at once. _captured goes False first so an OOM here leaves a clean
            # "capture again next call" state rather than a half-built one.
            self.fp, self.L_fwd, self.record = None, [], None
            self._captured = False
            self._capture(wav, ls, lr, tiles, True)
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
            fx = fhalo["x"]                # _overlap_ok => axes == ("x",)
            for it in range(self.nt):
                runner.run_phase(it + 1, 1)
                evt.record()
                un = self._halo_view(runner.u_next, "x")
                with torch.cuda.stream(comm):
                    comm.wait_event(evt)
                    fx.exchange_start(un)           # copy-send + P2P (no wait)
                runner.run_phase(it + 1, 2)         # interior, overlaps P2P
                with torch.cuda.stream(comm):
                    fx.exchange_finish(un)          # wait P2P + copy-recv
                compute.wait_stream(comm)
        else:
            for it in range(self.nt):
                runner.run_to(it + 1)
                self._exchange(fhalo, runner.u_now)

    def _forward_loop_lsrtm(self, fhalo):
        """LSRTM forward time loop: one coupled step (background + scattered
        advance together), then exchange the halo of BOTH ``u_now`` fields.

        Exchanging only the background would leave the scattered field's cut
        halo stale, and the coupling ``mp * vp^2 * lap(bg)`` would inject that
        staleness into the recorded scattered data.  Serial step-then-exchange
        only: the phase-split overlap path is acoustic-2D/3D-specific.
        """
        from sweep.propagator._stepped import (
            lsrtm_psi_pairs, lsrtm_u_blocks, u_now_slot)

        u_blocks = lsrtm_u_blocks(self.ndim)
        runner = SteppedBindingRunner(
            self.f_func, self.fp, self.L_fwd,
            psi_pairs=lsrtm_psi_pairs(self.ndim), u_blocks=u_blocks)
        for it in range(self.nt):
            runner.run_to(it + 1)
            # u_now slots rotate per step; recompute both after the step.
            self._exchange_group(
                fhalo, [self.L_fwd[u_now_slot(runner.k, b)] for b in u_blocks])

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
        if self.family == "lsrtm":
            # acoustic_lsrtm3d/backward.cu still runs a plain [nt-1, 0] loop, so a
            # stepped call would replay the WHOLE record and DD would exchange halos
            # of an adjoint that already reached t=0 -- silently wrong gradients.
            # Refuse until the backward honours bw_it_begin/bw_it_end.
            raise NotImplementedError(
                "domain decomposition supports AcousticLSRTM3D for FORWARD modelling "
                "only: its CUDA backward is not stepped yet (bw_it_begin/bw_it_end), "
                "so DD gradients would be silently wrong. Run the gradient on a "
                "single device, or use an equation whose backward is stepped "
                "(Acoustic, Acoustic3D, AcousticVRZ3D, Elastic).")
        if not self._captured:
            raise RuntimeError("forward() must run before the adjoint")
        if self.bp is None:
            raise RuntimeError(
                "ModelParallel has only a forward-only capture: the last "
                "forward ran with no grad-requiring model, so no boundary ring "
                "was saved and there is nothing to reconstruct the forward "
                "wavefield from. Re-run that forward with a requires_grad "
                "model (the autograd path promotes the capture automatically) "
                "before asking for the adjoint.")
        self.bp.adjoint_source = torch.as_tensor(
            adjoint_source_tile, device=self.dev, dtype=torch.float32)
        for t in self.L_adj + self.recon + self.gbufs + self.illum + self.coupling + self.adj_coeffs:
            t.zero_()
        self.bp.cut_face_mask = self.cut_mask
        bhalo = self._halo("_bwd_halo")
        nv = self._nv if self.family == "elastic" else None

        with torch.no_grad():
            if self._is_vrz:
                # VRZ phased backward (Fix A): advance+recon -> exchange lambda,p
                # -> build coupling c/e from the POST-exchange lambda,p -> exchange
                # c/e -> divergence/accumulate.  The c/e exchange gives the gradient
                # divergence the neighbour's coupling values at the cut seam
                # (acoustic's pointwise gradient needs no such exchange).  VRZ
                # rotates the psi pairs only (swap_pml), like the forward recon.
                br = SteppedBackwardRunner(
                    self.b_func, self.bp, self.L_adj, self.recon,
                    adj_pairs=acoustic_psi_pairs(self.ndim))
                # Pre-loop: build the adjoint coeffs C0/Cx/Cy/Cz once (model-only,
                # constant within a backward) and halo-exchange them once, so the fused
                # adjoint's transpose fast-path reads valid coeffs in the cut halo at
                # every reverse step (phase 1).  step_phase 4 = coeff build only.
                br.run_vrz_phase(self.nt, self.nt - 1, 4)
                self._exchange_group(bhalo, self.adj_coeffs)
                for it in range(self.nt - 1, 0, -1):    # step 0 contributes no grad
                    br.run_vrz_phase(it + 1, it, 1)     # advance adjoint + recon
                    self._exchange(bhalo, br.lambda_now)
                    self._exchange(bhalo, br.recon_u_now)
                    br.run_vrz_phase(it + 1, it, 2)     # build c/e (POST-exchange lambda,p)
                    self._exchange_group(bhalo, self.coupling)
                    br.run_vrz_phase(it + 1, it, 3)     # divergence -> grad += (once)
            elif self.family == "acoustic":
                # Plain acoustic doubles psi AND zeta in the fused adjoint
                # (swap_aux), needing the wider adj-pairs over a 15-field list.
                # VRZ (variable density) doubles only psi in the adjoint
                # (swap_pml), so its 12-field adjoint rotates just the psi pairs
                # -- exactly like the forward recon.  adjoint_extra_nvar (the
                # zeta double-buffer) is the discriminator: acoustic sets it to
                # 3, VRZ leaves it 0.  Using adj_pairs for VRZ indexes past the
                # 12-field list -> IndexError in rotate_wavefield_roles.
                _adj_extra = getattr(getattr(self.equation, "cuda_layout", None),
                                     "adjoint_extra_nvar", 0)
                _adj_pairs = (acoustic_adj_pairs(self.ndim) if _adj_extra
                              else acoustic_psi_pairs(self.ndim))
                br = SteppedBackwardRunner(
                    self.b_func, self.bp, self.L_adj, self.recon,
                    adj_pairs=_adj_pairs)
                # Boundary tail truncation: with tail_steps = K the strips
                # cover forward steps [nt-K, nt-1] and the restore at reverse
                # step ``it`` consumes the strip of step ``it - 1``, so the
                # reverse loop stops at bs_it0 + 1 (same bound as the C++
                # monolithic driver).  ``stop`` is derived from nt and the
                # captured tail — both identical on every rank — so all tiles
                # cease their lockstep halo exchanges at the same step; no
                # rank can be left waiting.  stop == 0 (tail off or >= nt)
                # reproduces the historical loop verbatim.
                _tail = int(getattr(self.bp, "boundary_tail_steps", 0) or 0)
                _bs_it0 = max(0, self.nt - _tail) if _tail > 0 else 0
                stop = _bs_it0 + 1 if _bs_it0 > 0 else 0
                for it in range(self.nt - 1, stop - 1, -1):
                    br.run_segment(it + 1, it)
                    if it == stop:
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
                # The injections run as their own sub-phase (step_phase 3),
                # at the same op position monolithic runs them; the split
                # exists purely so an exchange can sit between the injections
                # and the phase-1 kernels. A body-force source writes recon
                # VELOCITY and a stress receiver writes adjoint STRESS -- both
                # are ph2 fields, which phase 1 READS across the cut, so
                # without that exchange the neighbour's halo is one injection
                # stale at every reverse step. When neither is in play the
                # strips are untouched since the previous ph2 exchange and
                # the ship is skipped: the default combination (stress
                # source, velocity receivers) keeps the two-exchange step.
                _vel = ("vx", "vy", "vz")
                inj_cross = (any(t in _vel for t in self.prop.source_type)
                             or any(t not in _vel for t in self.prop.receiver_type))
                for it in range(self.nt - 1, 0, -1):     # elastic BS floor it==1
                    br.run_phase(it + 1, it, 3)          # injections(it)
                    if inj_cross:
                        self._exchange_group(bhalo, ph2)
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

    @property
    def own_receiver_indices(self):
        """Global receiver indices (caller's receiver order) whose traces this
        rank's tile record carries — set by the last forward's geometry, empty
        before any call.  Ownership is a partition of the global receiver list
        across the tile grid, so per-rank misfits over these traces sum to the
        global misfit (see the dd_fwi examples' partition assert)."""
        return tuple(getattr(self, "_own_rec_idx", ()) or ())

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
