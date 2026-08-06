from collections.abc import Sequence
import inspect

import numpy as np
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
                 full_mode=PROP_DEFAULTS.full_mode,
                 boundary_saving_config=None,
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
            use_ckpt (bool, optional): Use checkpointing to save memory. Defaults to True.
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
        if device is not None and dev is not None and device != dev:
            import warnings
            warnings.warn(
                "Both 'device' and 'dev' were passed to the propagator; using 'device'. "
                "'dev' is deprecated and will be removed in a future release.",
                DeprecationWarning, stacklevel=2,
            )
        resolved_device = device if device is not None else dev
        if resolved_device is None:
            # Inherit from the equation, which is the source of truth: its
            # operators (laplace kernels etc.) were already built on this device.
            resolved_device = getattr(equation, 'device', None)
        self.dev = resolved_device
        # ---- Per-edge boundary spec (free surface + PML thickness) ----------
        # ``free_surface`` and ``abcn`` accept the historical scalar/bool forms
        # as well as per-edge specs; both normalise to canonical axis-major
        # tuples ``(z_lo, z_hi, [y_lo, y_hi,] x_lo, x_hi)``.  ``free_surface=True``
        # with a scalar ``abcn`` reproduces the old top-only layout bit-for-bit.
        self.fs_faces = normalize_free_surface(free_surface, self.ndim)
        self._abcn_arg = abcn
        self.pad = normalize_pad(abcn, self.fs_faces, self.ndim)
        # ``self.abcn`` stays a representative uniform PML width for the legacy
        # readers (topography / curvilinear) that assume one — those paths are
        # guarded to the top-only configuration just below.
        self.abcn = abcn if isinstance(abcn, int) and not isinstance(abcn, bool) else max(self.pad + (0,))
        # Per-edge free surface (anything other than top-only or none), and
        # per-edge PML thickness, are a staged feature: only equations that opt
        # in (``supports_per_edge_free_surface``) handle them, 2-D only, and not
        # with topography.  Fail loud rather than silently degrade to top-only.
        _extended_boundary = (not is_top_only_or_none(self.fs_faces)) or not isinstance(abcn, int)
        if _extended_boundary:
            if self.ndim != 2:
                raise NotImplementedError(
                    "per-edge free surface / per-edge PML thickness is currently "
                    f"2-D only; got a {self.ndim}-D propagator (free_surface="
                    f"{free_surface!r}, abcn={abcn!r})."
                )
            if not getattr(equation, "supports_per_edge_free_surface", False):
                raise NotImplementedError(
                    f"{type(equation).__name__} does not support a per-edge free "
                    "surface or per-edge PML thickness yet (only top-only "
                    "free_surface=True/False with a scalar abcn). Supported: "
                    "Acoustic, Elastic (2-D)."
                )
            if topography is not None:
                raise NotImplementedError(
                    "per-edge free surface cannot be combined with topography= yet."
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
        if np.isscalar(dh):
            self._dh = float(dh)
            self._grid_spacing = tuple([self._dh] * self.ndim)
        else:
            if not isinstance(dh, Sequence) or isinstance(dh, (str, bytes)):
                raise TypeError(
                    "dh must be a float or a sequence ordered like shape "
                    "(2D: (dz, dx), 3D: (dz, dy, dx))."
                )
            if len(dh) != self.ndim:
                raise ValueError(
                    f"dh must have length {self.ndim} to match shape {shape}, "
                    f"got {len(dh)}."
                )
            self._grid_spacing = tuple(float(v) for v in dh)
            self._dh = float(self._grid_spacing[-1])
        self._dt = float(dt)
        self.use_ckpt = use_ckpt
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
        self.full_mode = full_mode
        legacy_boundary_config = {}
        if "transfer_interval" in kwargs:
            legacy_boundary_config["transfer_interval"] = kwargs.pop("transfer_interval")
        if "boundary_on_cpu" in kwargs:
            legacy_boundary_config["storage"] = "cpu" if kwargs.pop("boundary_on_cpu") else "gpu"
        if "use_pinned_memory" in kwargs:
            legacy_boundary_config["pinned_memory"] = kwargs.pop("use_pinned_memory")
        if "boundary_disk_async_read" in kwargs:
            legacy_boundary_config["disk_async_read"] = kwargs.pop("boundary_disk_async_read")
        if boundary_saving_config is None:
            boundary_saving_config = legacy_boundary_config or None
        else:
            boundary_saving_config = {**legacy_boundary_config, **boundary_saving_config}

        self.boundary_saving_config = self._normalize_boundary_saving_config(boundary_saving_config)
        self.transfer_interval = self.boundary_saving_config["transfer_interval"]
        self.boundary_on_cpu = (self.boundary_saving_config["storage"] == "cpu")
        self.use_pinned_memory = self.boundary_saving_config["pinned_memory"]
        self._abc_cache_key = None

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
        self.padding_z = (self.pad[0], self.pad[1])
        self.padding = torch_pad_order(self.pad, self.ndim)
        self.shape_nopad = tuple([w+2*self.equation.so for w in self.shape])
        self.shape = tuple(
            self.shape[ax] + self.pad[2*ax] + self.pad[2*ax + 1]
            for ax in range(self.ndim)
        )
        self.shape_cuda = tuple([s+self.equation.so for s in self.shape])

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
        valid_methods = {'auto', 'image', 'apm'}
        if topo_method not in valid_methods:
            raise ValueError(
                f"topo_method must be one of {sorted(valid_methods)}; "
                f"got {topo_method!r}"
            )

        has_topo = topography is not None
        supports_apm = bool(getattr(self.equation, 'supports_apm', False))

        # Curvilinear equations have their own topo path (boundary-fitted
        # grid via metric tensors); ``topo_method`` doesn't apply.  Honour
        # the user's ``free_surface`` flag verbatim and skip method
        # resolution.
        is_curvilinear = bool(getattr(self.equation, 'is_curvilinear', False))
        if is_curvilinear:
            fs = bool(free_surface) or has_topo
            return None, fs, fs

        # ---- No topography ----
        if not has_topo:
            if free_surface:
                # Flat free surface — only image method supports this
                # configuration.  APM is meaningless without per-cell
                # categories from a topo mask.
                if topo_method == 'apm':
                    raise ValueError(
                        "topo_method='apm' requires a topography= mask; "
                        "for a flat free surface use topo_method='image' "
                        "(or omit it)."
                    )
                return 'image', True, True
            # No free surface at all.
            return None, False, False

        # ---- Topography given: free surface is implicit ----
        # If the user also passed free_surface=False, we still turn it on
        # (topo implies FS); if they passed True, that matches.  No
        # warning needed — the new semantics make the combination
        # unambiguous.
        if topo_method == 'auto':
            method = 'apm' if supports_apm else 'image'
        elif topo_method == 'apm':
            if not supports_apm:
                raise ValueError(
                    f"topo_method='apm' requires equation.supports_apm=True; "
                    f"{type(self.equation).__name__} only supports the image "
                    f"method (use topo_method='image' or omit topo_method)."
                )
            method = 'apm'
        else:  # 'image'
            method = 'image'

        image_method_active = (method == 'image')
        return method, True, image_method_active

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
        """Validate the per-column surface row array and derive the matching
        air mask.

        Shapes (dispatched on ``len(phys_extent)``):

        * 2-D propagator (``phys_extent = (nz_phys, nx_phys)``):
          ``topo_input`` must be 1-D ``(nx_phys,)``.  Returns
          ``(topo_row_phys, air_mask_phys)`` with shapes
          ``(nx_phys,)`` and ``(nz_phys, nx_phys)``.

        * 3-D propagator (``phys_extent = (nz_phys, ny_phys, nx_phys)``):
          ``topo_input`` must be 2-D ``(ny_phys, nx_phys)``.  Returns
          ``(topo_row_phys, air_mask_phys)`` with shapes
          ``(ny_phys, nx_phys)`` and ``(nz_phys, ny_phys, nx_phys)``.

        ``topo_row_phys`` is ``int64``; ``air_mask_phys`` is ``float32``
        (1.0 above the surface, 0.0 at and below).
        """
        import torch

        if len(phys_extent) == 2:
            nz_phys, nx_phys = phys_extent
            if topo_input.ndim != 1:
                raise ValueError(
                    f"2-D topography must be 1-D ``(nx_phys,)`` (surface row "
                    f"index per physical column); got shape "
                    f"{tuple(topo_input.shape)}.  Non-single-valued geometries "
                    f"(overhangs, caves) are not supported by the standard "
                    f"staircase path."
                )
            if topo_input.shape[0] != nx_phys:
                raise ValueError(
                    f"topography length {topo_input.shape[0]} != physical nx "
                    f"({nx_phys})"
                )
            topo_row_phys = topo_input.to(torch.long)
            if (topo_row_phys < 0).any() or (topo_row_phys >= nz_phys).any():
                raise ValueError(
                    f"topography values must satisfy 0 <= row < {nz_phys}; "
                    f"got range [{int(topo_row_phys.min())}, "
                    f"{int(topo_row_phys.max())}]"
                )
            iz = torch.arange(nz_phys, device=topo_row_phys.device).view(-1, 1)
            air_mask_phys = (iz < topo_row_phys.view(1, -1)).to(torch.float32)
            return topo_row_phys, air_mask_phys

        # 3-D branch.
        nz_phys, ny_phys, nx_phys = phys_extent
        if topo_input.ndim != 2:
            raise ValueError(
                f"3-D topography must be 2-D ``(ny_phys, nx_phys)`` (surface "
                f"row index per (iy, ix) physical column); got shape "
                f"{tuple(topo_input.shape)}.  Overhangs / caves are not "
                f"supported."
            )
        if tuple(topo_input.shape) != (ny_phys, nx_phys):
            raise ValueError(
                f"3-D topography shape {tuple(topo_input.shape)} != "
                f"(ny_phys, nx_phys) = ({ny_phys}, {nx_phys})"
            )
        topo_row_phys = topo_input.to(torch.long)
        if (topo_row_phys < 0).any() or (topo_row_phys >= nz_phys).any():
            raise ValueError(
                f"topography values must satisfy 0 <= row < {nz_phys}; "
                f"got range [{int(topo_row_phys.min())}, "
                f"{int(topo_row_phys.max())}]"
            )
        iz = torch.arange(nz_phys, device=topo_row_phys.device).view(-1, 1, 1)
        air_mask_phys = (iz < topo_row_phys.view(1, ny_phys, nx_phys)).to(
            torch.float32
        )
        return topo_row_phys, air_mask_phys

    def _populate_image_method_topography(self, topo_row_phys):
        """Set ``self._topo_rows_runtime`` for the image-method /
        vacuum-staircase path.  Translates physical row indices to
        runtime-grid coordinates and replicate-pads the horizontal
        axes (x for 2-D; y and x for 3-D) through PML + stencil halo so
        the surface stays continuous through the absorbing boundary.

        Result shapes:
          2-D : 1-D ``(nx_phys + 2*(abcn+halo),)`` ``int32``.
          3-D : 2-D ``(ny_phys + 2*(abcn+halo), nx_phys + 2*(abcn+halo))``
                ``int32``.
        """
        import torch
        import torch.nn.functional as F

        halo = self.equation.so // 2
        # Runtime z layout (free_surface=True):
        #   [0, halo)              top stencil halo (image)
        #   [halo, halo + nz_phys) physical interior
        #   [halo + nz_phys, ...)  bottom PML + bottom halo
        topo_z = topo_row_phys + halo
        pad_each = self.abcn + halo

        # int32, not int64: ``_c.py`` reads ``data_ptr<int>()`` and a dtype
        # cast there would create a temporary whose GPU memory is reused
        # before the async CUDA kernels finish reading it.
        if topo_row_phys.ndim == 1:
            # 2-D propagator: 1-D row per ix.
            topo_runtime = F.pad(
                topo_z.to(torch.float32).view(1, 1, -1),
                (pad_each, pad_each),
                mode="replicate",
            ).view(-1).to(torch.int32)
        else:
            # 3-D propagator: 2-D row per (iy, ix).  Pad x then y via a
            # single F.pad call (right, left, top, bottom).
            topo_runtime = F.pad(
                topo_z.to(torch.float32).view(1, 1, *topo_z.shape),
                (pad_each, pad_each, pad_each, pad_each),
                mode="replicate",
            ).squeeze(0).squeeze(0).to(torch.int32)

        device = getattr(self.equation, "device", None) or self.dev
        if device is not None:
            try:
                topo_runtime = topo_runtime.to(device=device)
            except (RuntimeError, TypeError):
                pass

        # Defensive: ensure CPU→GPU copy is complete before any forward
        # kernel reads ``topo_rows[..., ix]``.  Without this we've seen ~30%
        # non-determinism in CUDA forward results.
        if topo_runtime.device.type == "cuda":
            torch.cuda.synchronize(topo_runtime.device)

        self.topography = topo_row_phys
        self._topo_rows_runtime = topo_runtime
        self.equation.topography = topo_row_phys
        self.equation._topo_rows_runtime = topo_runtime

    def _populate_apm_topography(self, air_mask_phys):
        """Set ``self.equation._apm_air_mask_runtime`` for the APM path.

        Replicate-pads the air mask through PML + stencil halo so the
        surface stays continuous through the absorbing boundary.

        Shapes:

        * 2-D propagator: ``air_mask_phys`` is ``(nz_phys, nx_phys)``;
          output is ``(nz_phys + 2*pad, nx_phys + 2*pad)``.
        * 3-D propagator: ``air_mask_phys`` is
          ``(nz_phys, ny_phys, nx_phys)``; output is
          ``(nz_phys + 2*pad, ny_phys + 2*pad, nx_phys + 2*pad)``,
          replicate-padded on all 6 sides.
        """
        import torch
        import torch.nn.functional as F

        halo = self.equation.so // 2
        pad_each = self.abcn + halo

        if air_mask_phys.ndim == 2:
            nz_phys, nx_phys = air_mask_phys.shape
            air_mask_padded = F.pad(
                air_mask_phys.view(1, 1, nz_phys, nx_phys),
                (pad_each, pad_each, pad_each, pad_each),
                mode="replicate",
            ).view(nz_phys + 2 * pad_each, nx_phys + 2 * pad_each)
        elif air_mask_phys.ndim == 3:
            nz_phys, ny_phys, nx_phys = air_mask_phys.shape
            # F.pad on (N=1, C=1, D, H, W) accepts a 6-tuple
            # (W_left, W_right, H_left, H_right, D_left, D_right).
            air_mask_padded = F.pad(
                air_mask_phys.view(1, 1, nz_phys, ny_phys, nx_phys),
                (pad_each, pad_each, pad_each, pad_each, pad_each, pad_each),
                mode="replicate",
            ).view(
                nz_phys + 2 * pad_each,
                ny_phys + 2 * pad_each,
                nx_phys + 2 * pad_each,
            )
        else:
            raise ValueError(
                f"air_mask_phys must be 2-D or 3-D, got ndim={air_mask_phys.ndim}"
            )

        device = getattr(self.equation, "device", None) or self.dev
        if device is not None:
            try:
                air_mask_padded = air_mask_padded.to(device=device)
            except (RuntimeError, TypeError):
                pass

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

    def _shape_tuple(self, value):
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.shape(value)
        return tuple(int(dim) for dim in shape)

    def _normalize_io(self, wavelet, sources, receivers):
        """Validate user-facing shapes for ``wavelet`` / ``sources`` / ``receivers``.

        The propagator accepts three input modes:

        - **A1**: ``wavelet=(nt,)``, ``sources=(nshots, ndim)``,
          ``receivers=(nshots, nrec, ndim)`` — naive multi-shot, shared wavelet.
        - **A2**: ``wavelet=(nshots, nt)``, ``sources=(nshots, ndim)``,
          ``receivers=(nshots, nrec, ndim)`` — naive multi-shot, per-shot wavelet.
        - **B**:  ``wavelet=(nt,)`` or ``(nsrc, nt)``,
          ``sources=(1, nsrc, ndim)``, ``receivers=(1, nrec, ndim)`` —
          source encoding (single super-shot, ``nsrc`` superposed point sources).

        ``receivers`` must always be 3-D; shared receiver arrays should be
        pre-broadcast/repeated to ``(B, nrec, ndim)`` by the user.

        Returns
        -------
        mode : {'A1', 'A2', 'B'}
        batch_size : int
            Internal batch dim (``nshots`` for A, ``1`` for B).
        nsrc_per_shot : int
            Number of point sources per shot (``1`` for A, ``nsrc`` for B).
        is_encoded : bool
            ``True`` iff ``mode == 'B'``.
        """
        ws = self._shape_tuple(wavelet)
        ss = self._shape_tuple(sources)
        rs = self._shape_tuple(receivers)
        ndim = self.ndim

        if len(rs) != 3 or rs[-1] != ndim:
            raise ValueError(
                f"receivers must have shape (B, nrec, {ndim}); got {rs}. "
                "Pre-broadcast/repeat per-shot if you previously passed a "
                "shared (nrec, dim) array."
            )
        nrec = rs[1]

        if len(ss) == 2:
            if ss[-1] != ndim:
                raise ValueError(
                    f"sources must have shape (nshots, {ndim}); got {ss}."
                )
            nshots = ss[0]
            if rs[0] != nshots:
                raise ValueError(
                    f"receivers batch ({rs[0]}) must match sources nshots "
                    f"({nshots}) in naive multi-shot mode."
                )
            if len(ws) == 1:
                return 'A1', nshots, 1, nrec, False
            if len(ws) == 2:
                if ws[0] != nshots:
                    raise ValueError(
                        f"wavelet must have shape (nshots={nshots}, nt); got {ws}."
                    )
                return 'A2', nshots, 1, nrec, False
            raise ValueError(
                "wavelet must have shape (nt,) [shared] or (nshots, nt) "
                f"[per-shot] in naive multi-shot mode; got {ws}."
            )

        if len(ss) == 3:
            if ss[0] != 1 or ss[-1] != ndim:
                raise ValueError(
                    "sources in source-encoding mode must have shape "
                    f"(1, nsrc, {ndim}); got {ss}."
                )
            nsrc = ss[1]
            if rs[0] != 1:
                raise ValueError(
                    "receivers batch must be 1 in source-encoding mode; "
                    f"got {rs[0]}."
                )
            if len(ws) == 1:
                return 'B', 1, nsrc, nrec, True
            if len(ws) == 2:
                if ws[0] != nsrc:
                    raise ValueError(
                        f"wavelet must have shape (nt,) or (nsrc={nsrc}, nt) "
                        f"in source-encoding mode; got {ws}."
                    )
                return 'B', 1, nsrc, nrec, True
            raise ValueError(
                "wavelet must have shape (nt,) or (nsrc, nt) in "
                f"source-encoding mode; got {ws}."
            )

        raise ValueError(
            f"sources must have shape (nshots, {ndim}) [naive multi-shot] "
            f"or (1, nsrc, {ndim}) [source encoding]; got {ss}."
        )

    def _normalize_boundary_saving_config(self, config):
        default = {
            "enabled": BOUNDARY_DEFAULTS.enabled,
            "storage": BOUNDARY_DEFAULTS.storage,
            "transfer_interval": BOUNDARY_DEFAULTS.transfer_interval,
            "pinned_memory": BOUNDARY_DEFAULTS.pinned_memory,
            "disk_dir": BOUNDARY_DEFAULTS.disk_dir,
            "ring_buffers": BOUNDARY_DEFAULTS.ring_buffers,
            "disk_async_read": BOUNDARY_DEFAULTS.disk_async_read,
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

    def init_abc(self, **kwargs):
        _padding = [self.equation.so // 2, self.equation.so // 2] * self.ndim
        fd_pad = tuple(kwargs.get('fd_pad', _padding))
        shape = tuple(kwargs.get('shape', self.shape))
        abc_key = (
            self.pml_type,
            tuple(self.pad),  # per-edge PML widths, axis-major (FS faces = 0)
            self.equation.so,
            fd_pad,
            self._dt,
            tuple(self._grid_spacing),
            kwargs.get('max_vel', 4500.0),
            kwargs.get('pml_freq', 25.0),
            shape,
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
        
        if getattr(self.equation, 'need_init', False):
            self.equation.init(self.shape, self.dev, self._dh)

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
        slices = [Ellipsis]
        for ax in range(self.ndim):
            lo = self.pad[2*ax]
            hi = self.pad[2*ax + 1]
            slices.append(slice(lo, -hi if hi > 0 else None))
        return data[tuple(slices)]

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]

    def _runtime_fd_halo(self):
        return self.equation.so // 2

    def _runtime_shape(self):
        halo = self._runtime_fd_halo()
        if halo <= 0:
            return self.shape
        return tuple(s + 2 * halo for s in self.shape)

    def _runtime_padding(self):
        halo = self._runtime_fd_halo()
        if halo <= 0:
            return self.padding
        return tuple(p + halo for p in self.padding)

    def _runtime_fd_pad(self):
        halo = self._runtime_fd_halo()
        return [halo, halo] * self.ndim

    def _runtime_coord_offset(self):
        halo = self._runtime_fd_halo()
        # Physical origin -> padded-grid origin: each axis' LOW-side pad plus the
        # stencil halo, in torch/reverse-axis order (last entry is z) to match
        # ``self.padding``.  Free-surface low faces have pad 0, so e.g. a top FS
        # gives z-offset ``halo`` — reproducing the old image-method rule.
        return tuple(self.pad[2*ax] + halo for ax in reversed(range(self.ndim)))

    def _runtime_crop_slices(self):
        halo = self._runtime_fd_halo()
        if halo <= 0:
            return (slice(None),) * self.ndim
        return tuple(slice(halo, -halo) for _ in range(self.ndim))

    def _crop_runtime_halo(self, data):
        halo = self._runtime_fd_halo()
        if halo <= 0:
            return data
        return data[(...,) + self._runtime_crop_slices()]

    def _spatial_pad_pairs(self, flat_padding):
        pairs = [(flat_padding[2 * i], flat_padding[2 * i + 1]) for i in range(len(flat_padding) // 2)]
        return tuple(reversed(pairs))
