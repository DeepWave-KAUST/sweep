import inspect

import numpy as np
from sweep.parallel.pml import dd_cut_face_mask, dd_cut_pad
from sweep.core.arguments import (
    merge_legacy_boundary_kwargs,
    refuse_free_surface_if_anisotropic,
    resolve_boundary_spec,
    normalise_spacing,
    resolve_device,
)
from sweep.core import geometry, validation
from sweep.core import topography as topography_
from sweep.equations.fields import build_field_index, format_field_specs
from sweep.equations._edges import (
    normalize_free_surface,
    normalize_pad,
    is_top_only_or_none,
    torch_pad_order,
    fs_faces_to_c_bitmask,
)
from sweep.propagator.options import BOUNDARY_DEFAULTS, CKPT_DEFAULTS, PROP_DEFAULTS

class PropBase:

    # Which equation flag gates an image-method topography staircase on this
    # impl; the compiled backend overrides it with the _c spelling.
    _IMAGE_TOPO_FLAG = "supports_image_topography"

    def __init__(self,
                 equation,
                 shape,
                 source_type=None,
                 receiver_type=None,
                 abcn=PROP_DEFAULTS.abcn,
                 free_surface=PROP_DEFAULTS.free_surface,
                 topography=None,
                 topo_method='auto',
                 dh=PROP_DEFAULTS.dh,
                 dt=PROP_DEFAULTS.dt,
                 dev=PROP_DEFAULTS.dev,
                 device=None,
                 use_ckpt=PROP_DEFAULTS.use_ckpt,
                 ckpt_chunks=CKPT_DEFAULTS.chunks,
                 ckpt_mode=CKPT_DEFAULTS.mode,
                 ckpt_num=CKPT_DEFAULTS.count,
                 ckpt_storage=CKPT_DEFAULTS.storage,
                 ckpt_pinned_memory=CKPT_DEFAULTS.pinned_memory,
                 pml_type=None,
                 nt=PROP_DEFAULTS.nt,
                 B=PROP_DEFAULTS.batch_size,
                 allow_growth=PROP_DEFAULTS.allow_growth,
                 boundary_saving_config=None,
                 boundary_buffer=None,
                 **kwargs):
        """Base class for the Propagator

        Args:
            equation (class): The wave equation class from sweep.equations
            shape (tupel or list): The shape of the model
            source_type (list, optional): List of strings for the source type. Defaults to [].
            receiver_type (list, optional): List of strings for the receiver type. Defaults to [].
            abcn (int, optional): The number of layers of absorbing boundary conditions. Defaults to 50.
            free_surface (bool, optional): If the model has a free surface. Defaults to False.
            topography (array_like, optional): Irregular free-surface
                topography — a 1-D integer array of length ``nx_phys``
                giving the per-column surface row index in the physical
                grid (``0`` = top of physical domain).  When given, a
                free surface is **implicit** — you do NOT need to set
                ``free_surface=True``.  ``topo_method`` selects the
                discretisation (image method vs APM); the propagator's
                PML layout is auto-set to match.  Defaults to ``None``.
            topo_method (str, optional): Which surface scheme to use
                when ``topography`` is given.  One of:

                * ``'auto'`` (default) — pick ``'apm'`` if the equation
                  declares ``supports_apm=True`` (currently
                  :class:`Elastic`/:class:`ElasticAPM`), else ``'image'``
                  (vacuum / Robertsson staircase).
                * ``'image'`` — staircase image method.  Acoustic uses
                  Mittet 2002 vacuum cells; Elastic uses Robertsson 1996
                  odd-parity stress mirror.  Sets ``free_surface=True``
                  internally (top PML suppressed).
                * ``'apm'`` — Cao & Chen 2018 parameter-modified method
                  (elastic only).  Sets ``free_surface=False`` internally
                  (full PML, including top).  Best long-time stability
                  on rough staircase topography.

                Ignored when ``topography is None``.  Legacy:
                ``free_surface=True + topography`` (without
                ``topo_method``) still selects image method, with a
                ``DeprecationWarning``.
            dh (float or sequence, optional): Grid spacing in model-axis order.
                For 2D use ``(dz, dx)`` and for 3D use ``(dz, dy, dx)``.
                Defaults to 10..
            dt (float, optional): Time step (seconds). Defaults to 0.002.
            dev (str, optional): Deprecated alias for ``device``. Defaults to None.
            device (str | torch.device, optional): The device to run the simulation on.
                When None, the equation's device is used. Preferred over ``dev``.
            use_ckpt (bool | None, optional): Legacy request for / exclusion of
                the checkpointing mode.  The gradient-memory mode is a
                three-way choice (full / boundary / ckpt) resolved by
                ``options.resolve_memory_strategy``; prefer
                ``memory=Full()`` / ``BoundarySaving(...)`` / ``Ckpt(...)``.
                None (default) picks
                the backend default: 'boundary' for impl='c', 'ckpt' for
                eager/jax.
            ckpt_chunks (int, optional): The number of time steps to chunk for checkpointing. Defaults to 100.
            ckpt_mode (str, optional): Checkpointing mode. "chunk" stores periodic checkpoints and
                replays each chunk, while "recursive" stores a fixed number of checkpoints and
                recursively recomputes intermediate states. Defaults to "chunk".
            ckpt_num (int, optional): Number of persistent checkpoints to save when
                ckpt_mode="recursive". Defaults to 0.
            ckpt_storage (str, optional): Store CUDA checkpoints on "gpu" or "cpu".
                CPU storage uses host memory to reduce device-memory pressure.
            ckpt_pinned_memory (bool, optional): Use pinned host memory when
                ckpt_storage="cpu". Defaults to True for CPU checkpoint storage.
            pml_type (str, optional): The type of PML to use. **You almost
                never need to set this** — leave it ``None`` and the propagator
                falls back to ``equation.default_pml_type``, which is the only
                CPML formulation each equation ships. The kwarg exists for
                advanced experiments (e.g. ``Acoustic1st`` accepts ``'spml'``
                in addition to its default ``'cpmls'``). Possible string
                values across the codebase: ``'cpmlr'``, ``'cpmls'``,
                ``'spml'``. Defaults to None.
            nt (int, optional): The number of time steps. Defaults to -1, which means it will be determined by the length of the source time function.
            B (int, optional): The batch size for the simulation. Defaults to 1.
            allow_growth (bool, optional): Whether to allow GPU memory growth. Defaults to True.
            boundary_saving_config (dict, optional): Configuration for boundary saving. Defaults to None, which means boundary saving is disabled. If provided, it should be a dictionary with the following keys:
                - enabled (bool): Whether to enable boundary saving. If True, the boundary wavefields will be saved and transferred to CPU for checkpointing. Defaults to False.
                - storage (str): Where to store the boundary wavefields. Options are 'gpu' and 'cpu'. If 'gpu', the boundary wavefields will be stored in GPU memory. If 'cpu', the boundary wavefields will be transferred to CPU memory. Defaults to 'gpu'.
                - transfer_interval (int): The interval (in time steps) at which to transfer the boundary wavefields to CPU memory if storage is 'cpu'. For example, if transfer_interval is 10, then every 10 time steps the boundary wavefields will be transferred to CPU memory. Defaults to 1.
                - pinned_memory (bool): Whether to use pinned memory for the boundary wavefields when storage is 'cpu'. Using pinned memory can speed up the transfer between GPU and CPU. Defaults to False.
                - disk_async_read (bool): Whether to read disk boundary chunks asynchronously during backward when storage is 'disk'. Defaults to False.
        """
        
        self.equation = equation
        if pml_type is None:
            # Each WaveEquation subclass declares its own default_pml_type;
            # see e.g. ElasticTTISG → 'cpmls', most acoustics → 'cpmlr'.
            pml_type = equation.default_pml_type
        if getattr(self.equation, 'setup_pml', None):
            self.equation.setup_pml(pml_type)
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.wavefield_specs = list(getattr(equation, "field_specs", []))
        self._wavefield_spec_index = build_field_index(self.wavefield_specs)
        self.shape = shape
        self.ndim = len(shape)
        # Physical (pre-pad) shape: self.shape is overwritten with the padded
        # runtime shape below, so retain it for a strategy wrapper (e.g.
        # sweep.parallel.ModelParallel) that reads a built propagator's
        # global-problem spec. (dh/dt are already registered as buffers later.)
        self._shape_phys = tuple(int(s) for s in shape)
        self.dev = resolve_device(device, dev, equation)
        # ---- Per-edge boundary spec (free surface + PML thickness) ----------
        # ``free_surface`` and ``abcn`` accept the historical scalar/bool forms
        # as well as per-edge specs; both normalise to canonical axis-major
        # tuples ``(z_lo, z_hi, [y_lo, y_hi,] x_lo, x_hi)``.  ``free_surface=True``
        # with a scalar ``abcn`` reproduces the old top-only layout bit-for-bit.
        self._abcn_arg = abcn
        self.fs_faces, self.pad, self.abcn = resolve_boundary_spec(
            free_surface, abcn, self.ndim, equation, topography,
            normalize_free_surface=normalize_free_surface,
            normalize_pad=normalize_pad,
            is_top_only_or_none=is_top_only_or_none,
        )
        # Resolve topo_method + free_surface BEFORE PML padding is
        # computed.  Two separate flags come out:
        #   ``self.free_surface``           — physical: model has a free
        #     surface (any topo + ``True`` flag).  This is the user-facing
        #     attribute.
        #   ``self._image_method_active``   — implementation: CUDA / Python
        #     kernels should use the image-method PML layout (top PML
        #     suppressed) AND the odd-parity z-derivative mirror at the
        #     surface row.  ``True`` only when method == 'image'.
        # APM implements the free surface via per-cell modulus
        # modifications and keeps full PML on all four sides, so it has
        # ``free_surface=True`` (physical) but ``_image_method_active=False``
        # (no image-method mirror / no top-PML suppression).
        self._topo_method, self.free_surface, self._image_method_active = (
            self._resolve_topo_method(
                topography=topography,
                topo_method=topo_method,
                free_surface=any(self.fs_faces),
            )
        )
        # ``topography=`` implies a free surface even with free_surface=False --
        # image method or APM alike, both of them isotropic constructions.
        refuse_free_surface_if_anisotropic(
            equation, self.free_surface, "implied by topography=")
        # A topography STAIRCASE (not a flat free surface) needs the equation
        # to actually apply ``_topo_rows_runtime`` / the kernel-side rows;
        # accepting it otherwise silently models a flat surface.  The flag name
        # is per-impl (``_IMAGE_TOPO_FLAG``): e.g. 3-D Elastic honours the
        # rows on eager but not in its CUDA kernels.
        if topography is not None and self._topo_method == 'image' \
                and not getattr(equation, self._IMAGE_TOPO_FLAG, False):
            raise NotImplementedError(
                f"topography= with the image method is not implemented by "
                f"{type(equation).__name__} on this impl "
                f"({self._IMAGE_TOPO_FLAG} is False)."
            )
        # ``_resolve_topo_method`` can turn the TOP free surface on implicitly
        # (topography= implies an image-method free surface even with
        # free_surface=False).  Fold that back into the canonical fs_faces/pad so
        # the per-edge padding layout suppresses the top PML accordingly.  (APM
        # topography returns _image_method_active=False and keeps full PML, so
        # this correctly does not fire.)  Gate on ``topography is not None``:
        # topography is the ONLY reason to add a top free surface the user didn't
        # ask for — a plain per-edge request like ``free_surface=['left']`` must
        # NOT get a spurious top free surface (which would drop the top PML).
        if topography is not None and self._image_method_active and not self.fs_faces[0]:
            self.fs_faces = (True,) + tuple(self.fs_faces[1:])
            self.pad = normalize_pad(self._abcn_arg, self.fs_faces, self.ndim)
        self._dh, self._grid_spacing = normalise_spacing(dh, self.ndim, shape)
        self._dt = float(dt)
        # None = unspecified; the torch entry points always pass a resolved
        # bool (see options.resolve_memory_strategy).  Direct PropBase / JAX
        # construction keeps the historical checkpointing default.  Held raw
        # here because the strategy needs the boundary config too, which is not
        # built yet -- it is derived once, below.
        _ckpt_requested = True if use_ckpt is None else bool(use_ckpt)
        self.ckpt_chunks = ckpt_chunks
        self.ckpt_mode = ckpt_mode
        self.ckpt_num = ckpt_num
        self.ckpt_storage, self.ckpt_pinned_memory = self._normalize_checkpoint_config(
            ckpt_storage,
            ckpt_pinned_memory,
        )
        self.pml_type = pml_type

        self.nt = nt
        self.B = B
        self.allow_growth = allow_growth
        # NOT warned here: by this point ``boundary_saving_config`` is the
        # INTERNAL wire format -- PropTorch translates memory=BoundarySaving()
        # into exactly this dict, so warning here would warn about the new API.
        # The warning belongs at the entry point the caller crosses.
        boundary_saving_config = merge_legacy_boundary_kwargs(
            kwargs, boundary_saving_config)

        self.boundary_saving_config = self._normalize_boundary_saving_config(boundary_saving_config)
        # ONE piece of state for the gradient-memory mode. ``use_ckpt`` used to
        # be an independent flag that happened to agree with the boundary
        # config; the two could drift, and a drift is silent -- the run measures
        # one strategy and reports the other.
        self._memory_strategy = (
            "ckpt" if _ckpt_requested
            else ("boundary" if self.boundary_saving_config.get("enabled") else "full"))
        self.transfer_interval = self.boundary_saving_config["transfer_interval"]
        self.boundary_on_cpu = (self.boundary_saving_config["storage"] == "cpu")
        self.use_pinned_memory = self.boundary_saving_config["pinned_memory"]
        self._abc_cache_key = None

        # Optional sweep.parallel.ModelParallelMesh; when set, init_abc routes
        # through rank-local PML widths and source/receiver / model tile work
        # is performed in subclasses. None = single-rank behaviour (unchanged).
        self.model_parallel = kwargs.pop('model_parallel', None)
        if kwargs:
            # Every named option and every legacy spelling has been consumed by
            # now, so anything left is a typo -- swallowing it silently turns
            # e.g. free_surfce=True into a flat-surface run that looks fine.
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: "
                f"{sorted(kwargs)}")

        # Keep the equation object aware of geometry-dependent boundary
        # behavior.  ``equation.free_surface`` is the image-method-layout
        # flag (used by the Python eager step to decide whether to apply
        # the top-row image mirror).  APM equations set free_surface=False
        # internally because their kernels don't engage the image mirror;
        # the per-cell category handles the FS BC.
        self.equation.free_surface = self._image_method_active
        # Per-edge free-surface faces (canonical axis-major bool tuple).  Migrated
        # equations (Acoustic / Elastic 2-D) read this; legacy equations ignore it
        # and keep using the top-only ``free_surface`` bool above.
        self.equation.fs_faces = self.fs_faces
        # C-side bitmask (SolverContext axis order 0=z,1=y,2=x) for impl='c'.
        self._fs_faces_c = fs_faces_to_c_bitmask(self.fs_faces, self.ndim)
        self.equation.abcn = self.abcn
        if getattr(self.equation, "pd", None) is not None and hasattr(self.equation.pd, "set_spacing"):
            self.equation.pd.set_spacing(self._grid_spacing)

        self.source_type = self._resolve_field_types(source_type, role="source")
        self.receiver_type = self._resolve_field_types(receiver_type, role="receiver")

        # PML / free-surface layout, PER EDGE.  Each face's pad is its PML width
        # (``self.pad``, axis-major, with free-surface faces forced to 0 — their
        # halo holds the image mirror); the stencil halo is added later in
        # ``_runtime_padding``.  ``self.padding`` is in torch pad order (last
        # spatial axis first), as every consumer has always assumed.  For the
        # top-only default this reproduces ``padding_z=(0, abcn)`` bit-for-bit.
        #
        # A model-parallel mesh contributes a SECOND, independent reason for a
        # 0 pad: a cut (neighbour-facing) face gets no PML, only the stencil
        # halo, because :class:`HaloExchange` supplies those cells.  That is the
        # cut-aware compact padding — interior tiles no longer waste an ``abcn``
        # pad on every split face.  ``build_rank_pml_widths`` returns the SAME
        # axis-major ``[z_lo, z_hi, (y_lo, y_hi,) x_lo, x_hi]`` layout as
        # ``self.pad``, so the two combine face-by-face with ``min``: a face is
        # thin if EITHER the per-edge layout says so (free surface, or an
        # explicitly thinner pad) or the mesh says it is cut.  Without a mesh
        # ``self.pad`` is left exactly as ``normalize_pad`` produced it.
        self.pad = dd_cut_pad(self.model_parallel, self.pad, self.abcn, self.ndim)

        # ---- sigma=0 buffer between the physical box and the PML ------------
        # ``self.pml_pad`` is the damping-ramp width per face (what the CPML
        # profiles, the boundary Layout and the C side see); ``self.pad`` is
        # the tensor pad = ramp + buffer, and drives the runtime shape, the
        # coordinate shift and the gradient crop.  The buffer cells are plain
        # interior cells whose gradient is cropped away, so a boundary-saving
        # shell placed at THEIR outer edge is never read by the gradient of a
        # physical cell: the storage-noise rim lands in the buffer.  Faces
        # without a ramp (free surface, DD cut) get no buffer.
        self.pml_pad = tuple(self.pad)
        self.boundary_buffer = self._resolve_boundary_buffer(boundary_buffer, topography)
        self.pad = tuple(p + self.boundary_buffer if p > 0 else 0 for p in self.pml_pad)

        self.padding_z = (self.pad[0], self.pad[1])
        self.padding = torch_pad_order(self.pad, self.ndim)
        self.shape_nopad = tuple([w+2*self.equation.so for w in self.shape])
        self.shape = tuple(
            self.shape[ax] + self.pad[2*ax] + self.pad[2*ax + 1]
            for ax in range(self.ndim)
        )
        self.shape_cuda = tuple([s+self.equation.so for s in self.shape])

        # DD cut-face bitmask (x_lo=1, x_hi=2, y_lo=16, y_hi=32; z is never
        # split in v1). Passed to the boundary Layout so its phys_* bounds
        # match this (possibly asymmetric) pad. 0 = single domain.
        self._dd_cut_mask = dd_cut_face_mask(self.model_parallel, self.ndim)

        # Topography is processed AFTER self.shape is PML-padded so the
        # runtime-coord conversion can compute the final padded surface row.
        self._process_topography(topography)

        self._set_call_signature()

    def _resolve_topo_method(self, *, topography, topo_method, free_surface):
        """Resolve topo method, physical free-surface state, and image-method
        layout flag.

        New semantics (post-refactor):

        * ``free_surface=True`` (no topo)  → flat free surface (image method).
        * ``free_surface=False`` (no topo) → no free surface, full PML.
        * ``topography=`` given             → free surface is ON regardless
          of the ``free_surface`` flag; method auto-selects (APM if
          supported, otherwise image) unless the user explicitly passes
          ``topo_method='image'`` or ``'apm'``.
        * ``topo_method='apm'`` without ``topography``                 → error.

        Returns
        -------
        (method, free_surface_physical, image_method_active) :
            (str | None, bool, bool)

        ``method``                — ``'image'``, ``'apm'``, or ``None``.
        ``free_surface_physical`` — user-facing flag: does the model have a
                                    free surface at all?
        ``image_method_active``   — implementation flag: should kernels use
                                    image-method PML layout (top PML
                                    suppressed) and the odd-parity
                                    z-derivative mirror?  Only true when
                                    ``method == 'image'``.
        """
        return topography_.resolve_topo_method(
            topography=topography, topo_method=topo_method,
            free_surface=free_surface,
            supports_apm=bool(getattr(self.equation, 'supports_apm', False)),
            # is_curvilinear is an INSTANCE attribute with no class-level
            # default (unlike supports_apm), so the getattr default is what
            # every non-curvilinear equation relies on.
            is_curvilinear=bool(getattr(self.equation, 'is_curvilinear', False)),
            equation_name=type(self.equation).__name__,
        )

    def _process_topography(self, topography):
        """Validate and store irregular free-surface topography.

        ``topography=`` takes a 1-D ``(nx_phys,)`` integer array of
        per-column surface row indices.  The matching 2-D air mask is
        derived internally for the APM path; the image-method path uses
        only the 1-D form.  Method dispatch follows ``self._topo_method``
        (set in :meth:`_resolve_topo_method`):

        ============== =================================================
        ``_topo_method``  Backend
        ============== =================================================
        ``'image'``      Image-method / vacuum staircase
                         (Mittet 2002 / Robertsson 1996).
                         Uses ``_topo_rows_runtime``.
        ``'apm'``        APM (Cao & Chen 2018) — elastic only.
                         Uses ``_apm_air_mask_runtime``.
        ============== =================================================

        Curvilinear equations (``equation.is_curvilinear``) consume the
        per-column elevation independently of the dispatch above.
        """
        # Reset all topo-related attributes.
        self.topography = None
        self._topo_rows_runtime = None
        self.equation.topography = None
        self.equation._topo_rows_runtime = None
        self.equation._apm_air_mask_runtime = None

        if topography is None:
            # Curvilinear equations need an identity-metric grid even
            # for flat topo, otherwise ``step`` raises.
            if getattr(self.equation, "is_curvilinear", False):
                self._attach_curvilinear_metrics(
                    topography=None, halo=self.equation.so // 2
                )
            return

        if self.ndim not in (2, 3):
            raise NotImplementedError(
                "topography support is currently limited to 2-D and 3-D "
                f"propagators (got ndim={self.ndim})."
            )

        import torch

        topo_input = torch.as_tensor(topography)

        # Resolve physical extents from the runtime shape + image-method-
        # layout flag (image suppresses top PML; APM has PML on both top
        # and bottom).  ``self.shape`` is laid out as (nz, nx) for 2-D and
        # (nz, ny, nx) for 3-D — z is always axis 0.
        if self._image_method_active:
            nz_phys = self.shape[0] - self.abcn
        else:
            nz_phys = self.shape[0] - 2 * self.abcn
        if self.ndim == 2:
            nx_phys = self.shape[1] - 2 * self.abcn
            ny_phys = None
            phys_extent = (nz_phys, nx_phys)
        else:
            ny_phys = self.shape[1] - 2 * self.abcn
            nx_phys = self.shape[2] - 2 * self.abcn
            phys_extent = (nz_phys, ny_phys, nx_phys)

        # Normalise input → (topo_row_phys, air_mask_phys) tuple.  Each
        # method below picks the form it needs.  Shapes:
        #   2-D : topo_row_phys (nx_phys,)              air_mask (nz, nx)
        #   3-D : topo_row_phys (ny_phys, nx_phys)      air_mask (nz, ny, nx)
        topo_row_phys, air_mask_phys = self._canonicalise_topography(
            topo_input, *phys_extent
        )

        # Curvilinear path: always uses 1-D row.  Boundary-fitted grid
        # ignores the air_mask form entirely.
        if getattr(self.equation, "is_curvilinear", False):
            self.topography = topo_row_phys
            self.equation.topography = topo_row_phys
            self._attach_curvilinear_metrics(
                topography=topo_row_phys.cpu().numpy(),
                halo=self.equation.so // 2,
            )
            return

        # Dispatch on the topo method resolved at construction.
        if self._topo_method == 'image':
            self._populate_image_method_topography(topo_row_phys)
        elif self._topo_method == 'apm':
            self._populate_apm_topography(air_mask_phys)
        else:
            # _resolve_topo_method should never let us get here with
            # topography != None and method == None, but guard anyway.
            raise RuntimeError(
                f"Internal error: topography given but _topo_method is "
                f"{self._topo_method!r}"
            )

    def _canonicalise_topography(self, topo_input, *phys_extent):
        return topography_.canonicalise_topography(topo_input, *phys_extent)

    def _populate_image_method_topography(self, topo_row_phys):
        """Set the runtime surface rows for the image-method / vacuum path.

        The build is in :func:`sweep.core.topography.build_image_method_topo_rows`;
        the four assignments stay here because they are this propagator taking
        ownership of the equation's runtime binding -- see that module's note on
        the mutation boundary.
        """
        topo_runtime = topography_.build_image_method_topo_rows(
            topo_row_phys,
            abcn=self.abcn,
            halo=self.equation.so // 2,
            # truthiness `or`, not `is not None` -- preserved verbatim
            device=getattr(self.equation, "device", None) or self.dev,
        )
        self.topography = topo_row_phys
        self._topo_rows_runtime = topo_runtime
        self.equation.topography = topo_row_phys
        self.equation._topo_rows_runtime = topo_runtime

    def _populate_apm_topography(self, air_mask_phys):
        """Set the APM air mask on the equation.

        The build is in :func:`sweep.core.topography.build_apm_air_mask`; the
        three assignments stay here. Note ``self.topography`` keeps the
        PHYSICAL mask while the equation gets the padded one, and that the
        physical mask lives on the INPUT's device -- CPU for a numpy
        topography, even on a CUDA run.
        """
        air_mask_padded = topography_.build_apm_air_mask(
            air_mask_phys,
            abcn=self.abcn,
            halo=self.equation.so // 2,
            device=getattr(self.equation, "device", None) or self.dev,
        )
        self.topography = air_mask_phys
        self.equation.topography = air_mask_phys
        self.equation._apm_air_mask_runtime = air_mask_padded

    def _attach_curvilinear_metrics(self, topography, halo):
        """Build a :class:`CurvilinearGrid` from ``topography`` (physical
        nx-long array, ``None`` for flat) and attach padded metric
        tensors to ``self.equation``."""
        from sweep.utils.curvilinear import CurvilinearGrid

        if self.ndim != 2:
            raise NotImplementedError(
                "Curvilinear path only supports 2-D propagators (got "
                f"ndim={self.ndim})."
            )

        nz_phys = self.shape[0] - (self.abcn if self._image_method_active else 2 * self.abcn)
        nx_phys = self.shape[1] - 2 * self.abcn
        device = getattr(self.equation, "device", None) or self.dev

        grid = CurvilinearGrid(
            topography=topography,
            nz_phys=nz_phys,
            nx_phys=nx_phys,
            dh=float(self._grid_spacing[-1]),
            device="cpu",   # build on CPU; move to device after padding
        )

        # Pad metrics to runtime shape using edge replication. The
        # runtime shape is ``self.shape + 2 * halo`` per axis, with
        # padding pattern ``(pad_x_left, pad_x_right, pad_z_top, pad_z_bottom)``.
        if self._image_method_active:
            pad_z_top, pad_z_bot = halo, self.abcn + halo
        else:
            pad_z_top = pad_z_bot = self.abcn + halo
        pad_x_left = pad_x_right = self.abcn + halo
        pad_runtime = (pad_x_left, pad_x_right, pad_z_top, pad_z_bot)
        padded = grid.padded_metrics(pad_runtime)

        if device is not None:
            try:
                padded = {k: v.to(device) for k, v in padded.items()}
            except (RuntimeError, TypeError):
                pass

        self.equation.set_curvilinear_metrics(
            alpha=padded["alpha"],
            metric_pηη=padded["metric_pηη"],
            metric_pη=padded["metric_pη"],
            d_eta=grid.d_eta,
        )
        # Also expose α_xi, α_eta, β for elastic; safely ignored by
        # acoustic which doesn't need them.
        self.equation._curv_beta = padded["beta"]
        self.equation._curv_alpha_xi = padded["alpha_xi"]
        self.equation._curv_alpha_eta = padded["alpha_eta"]
        # ``h_prime`` is the 1-D surface slope along ξ, padded to the
        # runtime x-extent. Elastic uses it for the rotated free-surface
        # traction BC on a curved surface (acoustic ignores it).
        self.equation._curv_h_prime = padded["h_prime"]
        self._curvilinear_grid = grid

    def _default_field_types(self, role):
        attr = "default_source_fields" if role == "source" else "default_receiver_fields"
        defaults = getattr(self.equation, attr, None)
        if defaults:
            return list(defaults)
        return [self.wavefield_names[0]]

    def _resolve_field_types(self, kinds, role):
        resolved = PropBase._default_field_types(self, role) if not kinds else list(kinds)
        attr = "supports_source" if role == "source" else "supports_receiver"
        output = []
        for name in resolved:
            spec = self._wavefield_spec_index.get(name)
            if spec is None:
                available = [
                    spec for spec in self.wavefield_specs
                    if getattr(spec, attr, False)
                ]
                role_name = f"{role}_type"
                raise ValueError(
                    f"Unknown {role_name} entry '{name}'. Available {role_name} values:\n"
                    f"{format_field_specs(available)}"
                )
            if not getattr(spec, attr, False):
                role_name = f"{role}_type"
                raise ValueError(
                    f"Field '{name}' resolves to '{spec.name}', but `{spec.name}` is not valid for {role_name}."
                )
            output.append(spec.name)
        return output

    @property
    def use_ckpt(self):
        """Whether the gradient-memory strategy is checkpointing.

        Read-only on purpose. It was a writable flag, and six places flipped it
        after construction while the boundary config stayed as it was, so the
        object could hold two answers to one question. Change the strategy with
        :attr:`memory_strategy`, which says which one it is becoming.
        """
        return self._memory_strategy == "ckpt"

    @use_ckpt.setter
    def use_ckpt(self, value):
        # Python's own message for a getter-only property ("property has no
        # setter") does not say what to write instead, and this one is reached
        # from code that predates the read-only change.
        raise AttributeError(
            "use_ckpt is read-only: it reports the gradient-memory strategy "
            "rather than setting it. Write `memory_strategy = "
            f"{'ckpt' if value else 'full'!r}` (or 'boundary') instead, which "
            "moves the boundary configuration with it."
        )

    @property
    def memory_strategy(self):
        """'full' | 'boundary' | 'ckpt' -- the single source for the mode."""
        return self._memory_strategy

    @memory_strategy.setter
    def memory_strategy(self, value):
        self._set_memory_strategy(value)

    def _set_memory_strategy(self, strategy, *, reason=None):
        """Move to another gradient-memory strategy, deliberately.

        ``reason`` is for the caller's benefit at the call site, not stored: it
        makes ``_set_memory_strategy('boundary', reason='APM has no ckpt
        backward')`` read as what it is, where ``use_ckpt = False`` left the
        reader to work out what it fell back TO.
        """
        if strategy not in ("full", "boundary", "ckpt"):
            raise ValueError(
                f"memory strategy must be 'full', 'boundary' or 'ckpt', got {strategy!r}")
        self._memory_strategy = strategy

    def _disable_ckpt(self, reason):
        """Drop out of checkpointing, keeping boundary saving if it is enabled.

        ``use_ckpt = False`` did not say where it landed. It landed here.
        """
        if self._memory_strategy != "ckpt":
            return
        self._set_memory_strategy(
            "boundary" if self.boundary_saving_config.get("enabled") else "full",
            reason=reason)

    def _set_call_signature(self):
        forward = getattr(type(self), "forward", None)
        if forward is None:
            return
        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError):
            return
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name == "self":
            signature = signature.replace(parameters=parameters[1:])
        self.__signature__ = signature

    # Shape validation lives in sweep.core.validation as a pure function of the
    # three shapes; this stays as the binding that supplies self.ndim.
    def _shape_tuple(self, value):
        return validation.shape_tuple(value)

    def _normalize_io(self, wavelet, sources, receivers):
        return validation.normalize_io(wavelet, sources, receivers, self.ndim)

    def _normalize_boundary_saving_config(self, config):
        default = {
            "enabled": BOUNDARY_DEFAULTS.enabled,
            "storage": BOUNDARY_DEFAULTS.storage,
            "transfer_interval": BOUNDARY_DEFAULTS.transfer_interval,
            "pinned_memory": BOUNDARY_DEFAULTS.pinned_memory,
            "disk_dir": BOUNDARY_DEFAULTS.disk_dir,
            "ring_buffers": BOUNDARY_DEFAULTS.ring_buffers,
            "disk_async_read": BOUNDARY_DEFAULTS.disk_async_read,
            "tail_steps": None,
        }

        if config is None:
            config = {}
        if "boundary_disk_async_read" in config:
            config = dict(config)
            config["disk_async_read"] = config.pop("boundary_disk_async_read")

        merged = default.copy()
        merged.update(config)

        if merged["storage"] not in {"gpu", "cpu", "disk"}:
            raise ValueError("boundary_saving_config['storage'] must be 'gpu', 'cpu', or 'disk'")

        if merged.get("tail_steps") is not None:
            tail = merged["tail_steps"]
            if not isinstance(tail, int) or isinstance(tail, bool) or tail < 1:
                raise ValueError("boundary_saving_config['tail_steps'] must be a positive int or None")

        if merged["storage"] == "gpu":
            merged["transfer_interval"] = BOUNDARY_DEFAULTS.gpu_transfer_interval
            merged["pinned_memory"] = BOUNDARY_DEFAULTS.gpu_pinned_memory
            merged["disk_dir"] = None
            merged["ring_buffers"] = BOUNDARY_DEFAULTS.gpu_ring_buffers
            merged["disk_async_read"] = BOUNDARY_DEFAULTS.disk_async_read

        if merged["storage"] == "cpu":
            merged["disk_dir"] = None
            merged["disk_async_read"] = BOUNDARY_DEFAULTS.disk_async_read
            if merged["transfer_interval"] is None:
                merged["transfer_interval"] = BOUNDARY_DEFAULTS.cpu_transfer_interval
            if merged["ring_buffers"] is None:
                merged["ring_buffers"] = BOUNDARY_DEFAULTS.cpu_ring_buffers
            if merged["pinned_memory"] is None:
                merged["pinned_memory"] = BOUNDARY_DEFAULTS.cpu_pinned_memory

        if merged["storage"] == "disk":
            merged["pinned_memory"] = False
            if merged["transfer_interval"] is None:
                if merged["disk_async_read"]:
                    merged["transfer_interval"] = (
                        BOUNDARY_DEFAULTS.disk_async_transfer_interval_2d
                        if self.ndim == 2
                        else BOUNDARY_DEFAULTS.disk_async_transfer_interval_3d
                    )
                else:
                    merged["transfer_interval"] = BOUNDARY_DEFAULTS.disk_transfer_interval
            if merged["ring_buffers"] is None:
                if merged["disk_async_read"]:
                    merged["ring_buffers"] = BOUNDARY_DEFAULTS.disk_async_ring_buffers
                else:
                    merged["ring_buffers"] = (
                        BOUNDARY_DEFAULTS.disk_ring_buffers_2d
                        if self.ndim == 2
                        else BOUNDARY_DEFAULTS.disk_ring_buffers_3d
                    )
            if merged["disk_async_read"] and merged["ring_buffers"] < BOUNDARY_DEFAULTS.disk_async_ring_buffers:
                merged["ring_buffers"] = BOUNDARY_DEFAULTS.disk_async_ring_buffers

        if merged["transfer_interval"] < 1:
            raise ValueError("boundary_saving_config['transfer_interval'] must be >= 1")
        if merged["ring_buffers"] < 1:
            raise ValueError("boundary_saving_config['ring_buffers'] must be >= 1")

        return merged

    def _normalize_checkpoint_config(self, storage, pinned_memory):
        if storage not in {"gpu", "cpu"}:
            raise ValueError("ckpt_storage must be 'gpu' or 'cpu'")
        if storage == "gpu":
            if pinned_memory:
                raise ValueError("ckpt_pinned_memory is only valid when ckpt_storage='cpu'")
            return "gpu", False
        if pinned_memory is None:
            pinned_memory = CKPT_DEFAULTS.cpu_pinned_memory
        return "cpu", bool(pinned_memory)

    def resolve_boundary_saving_config(self, override=None, use_boundary_saving=None):
        config = self.boundary_saving_config.copy()
        if override is not None:
            config = self._normalize_boundary_saving_config({**config, **override})
        if use_boundary_saving is not None:
            config["enabled"] = bool(use_boundary_saving)
        return config

    _INIT_ABC_KEYS = frozenset({"fd_pad", "shape", "max_vel", "pml_freq"})

    def _resolve_boundary_buffer(self, requested, topography):
        """Cells of sigma=0 buffer on every PML face; see ``BOUNDARY_BUFFER_REACH``.

        ``None`` = the equation's need (REACH*M+1, 0 for a pointwise-imaging
        equation), applied under EVERY gradient-memory strategy and impl so that
        full, checkpoint and boundary-saving runs -- and eager vs compiled --
        solve the same discretised problem (the buffer moves the PML outward,
        which shifts the gradient by ~1e-2 relative on a small grid; only
        boundary saving needs the buffer, but a mode-dependent grid would make
        the modes disagree).  An explicit value is honoured, but a
        boundary-saving run below the equation's need is refused: the
        alternative is a shell narrower than the imaging stencil, i.e. the
        outermost physical cells imaging cells that were never restored.
        """
        M = self.equation.so // 2
        reach = int(getattr(self.equation, "BOUNDARY_BUFFER_REACH", 0) or 0)
        needed = reach * M + 1 if reach > 0 else 0
        if requested is None:
            k = needed
        else:
            if isinstance(requested, bool) or int(requested) != requested or int(requested) < 0:
                raise ValueError(f"boundary_buffer must be a non-negative int, got {requested!r}")
            k = int(requested)
        if k > 0 and topography is not None:
            raise NotImplementedError(
                "boundary_buffer (sigma=0 buffer for boundary saving) is not "
                "supported together with topography= yet.")
        if self._memory_strategy == "boundary" and k < needed:
            raise ValueError(
                f"{type(self.equation).__name__} boundary saving needs a buffer of at "
                f"least {needed} cells (its imaging stencil reaches {reach}M beyond the "
                f"boundary shell); got boundary_buffer={k}. Leave boundary_buffer=None "
                "for the default, or use memory=Full()/Ckpt().")
        return k

    def init_abc(self, **kwargs):
        # forward()'s residual kwargs land here on both impls, so this is the
        # one place a call-time typo (pml_freqs=...) can be caught instead of
        # silently keeping the default.
        unknown = set(kwargs) - self._INIT_ABC_KEYS
        if unknown:
            raise TypeError(
                f"forward() got unexpected keyword arguments: {sorted(unknown)}; "
                f"recognised extras are {sorted(self._INIT_ABC_KEYS)}")
        _padding = [self.equation.so // 2, self.equation.so // 2] * self.ndim
        fd_pad = tuple(kwargs.get('fd_pad', _padding))
        shape = tuple(kwargs.get('shape', self.shape))

        # ``self.pad`` is the single source of truth for per-face PML widths,
        # layout ``[z_low, z_high, (y_low, y_high,) x_low, x_high]``.  It already
        # carries BOTH reasons a face can be thin: a per-edge free surface, and
        # (with a model-parallel mesh) a cut face — interior-facing sides connect
        # to neighbour tiles via HaloExchange and must NOT be absorbed.  Deriving
        # the widths here a second time would drop the per-edge free-surface
        # faces and disagree with the actual padding layout.
        #
        # ``rank_coord`` stays in the cache key: two ranks can share a pad tuple
        # while sitting at different mesh coords.
        rank_coord = (self.model_parallel.coord
                      if self.model_parallel is not None else None)

        abc_key = (
            self.pml_type,
            tuple(self.pml_pad),  # per-edge damping-ramp widths, axis-major (FS/cut faces = 0)
            self.equation.so,
            fd_pad,
            self._dt,
            tuple(self._grid_spacing),
            kwargs.get('max_vel', 4500.0),
            kwargs.get('pml_freq', 25.0),
            shape,
            rank_coord,
        )

        if abc_key != self._abc_cache_key:
            self.equation.init_abc(
                    type=self.pml_type,
                    pml_width=list(abc_key[1]),
                    accuracy=self.equation.so,
                    fd_pad=list(fd_pad),
                    dt=self._dt,
                    grid_spacing=list(self._grid_spacing),
                    max_vel=kwargs.get('max_vel', 4500.0),
                    dtype=np.float32,
                    pml_freq=kwargs.get('pml_freq', 25.0),
                    shape=shape
            )
            self._abc_cache_key = abc_key
        
    def crop(self, data):
        """Crop the data to the original shape

        Args:
            data (np.ndarray): The data to be cropped

        Returns:
            np.ndarray: The cropped data
        """
        # Remove each face's PML pad, recovering the physical model.  Free-surface
        # faces have pad 0 (their halo is handled elsewhere), so nothing is cropped
        # there — reproducing the old image ``data[..., 0:-abcn, abcn:-abcn]``.
        return geometry.crop_to_physical(data, self.pad, self.ndim)

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]

    # Grid geometry lives in sweep.core.geometry as plain functions of the
    # numbers; these stay as the thin bindings that supply them from self.
    def _runtime_fd_halo(self):
        return geometry.fd_halo(self.equation.so)

    def _runtime_shape(self):
        return geometry.runtime_shape(self.shape, self._runtime_fd_halo())

    def _runtime_padding(self):
        return geometry.runtime_padding(self.padding, self._runtime_fd_halo())

    def _runtime_fd_pad(self):
        return geometry.runtime_fd_pad(self._runtime_fd_halo(), self.ndim)

    def _runtime_coord_offset(self):
        return geometry.runtime_coord_offset(
            self.pad, self._runtime_fd_halo(), self.ndim)

    def _runtime_crop_slices(self):
        return geometry.runtime_crop_slices(self._runtime_fd_halo(), self.ndim)

    def _crop_runtime_halo(self, data):
        return geometry.crop_runtime_halo(
            data, self._runtime_fd_halo(), self.ndim)

    def _spatial_pad_pairs(self, flat_padding):
        return geometry.spatial_pad_pairs(flat_padding)
