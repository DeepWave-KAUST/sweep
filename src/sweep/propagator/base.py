from collections.abc import Sequence
import inspect

import numpy as np
from sweep.equations.fields import build_field_index, format_field_specs
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
        self.abcn = abcn
        # Resolve topo_method + effective free_surface layout BEFORE PML
        # padding is computed — the method dictates the layout.  When
        # ``topography is None`` the user's ``free_surface`` flag is
        # honoured unchanged.
        self._topo_method, self.free_surface = self._resolve_topo_method(
            topography=topography,
            topo_method=topo_method,
            free_surface=free_surface,
        )
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

        # Keep the equation object aware of geometry-dependent boundary behavior.
        self.equation.free_surface = self.free_surface
        self.equation.abcn = self.abcn
        if getattr(self.equation, "pd", None) is not None and hasattr(self.equation.pd, "set_spacing"):
            self.equation.pd.set_spacing(self._grid_spacing)

        self.source_type = self._resolve_field_types(source_type, role="source")
        self.receiver_type = self._resolve_field_types(receiver_type, role="receiver")

        if self.free_surface:
            self.padding_z = (0, self.abcn)
            shape_z = self.shape[0] + self.abcn
        else:
            self.padding_z = (self.abcn, self.abcn)
            shape_z = self.shape[0] + 2*self.abcn

        self.padding = (self.abcn,) * 2*(self.ndim-1) + self.padding_z
        self.shape_nopad = tuple([w+2*self.equation.so for w in self.shape])
        self.shape = (shape_z,) + tuple(s+2*self.abcn for s in self.shape[1:])
        self.shape_cuda = tuple([s+self.equation.so for s in self.shape])

        # Topography is processed AFTER self.shape is PML-padded so the
        # runtime-coord conversion can compute the final padded surface row.
        self._process_topography(topography)

        self._set_call_signature()

    def _resolve_topo_method(self, *, topography, topo_method, free_surface):
        """Resolve the topographic discretisation method and the effective
        ``free_surface`` layout flag.

        Returns
        -------
        (method, free_surface_effective) : (str | None, bool)
            ``method`` is one of ``'image'``, ``'apm'``, or ``None`` (no
            topography).  ``free_surface_effective`` is the boolean used
            to lay out PML padding.
        """
        valid_methods = {'auto', 'image', 'apm'}
        if topo_method not in valid_methods:
            raise ValueError(
                f"topo_method must be one of {sorted(valid_methods)}; "
                f"got {topo_method!r}"
            )

        # No topography — user's free_surface stays as-is, no method.
        if topography is None:
            return None, bool(free_surface)

        supports_apm = bool(getattr(self.equation, 'supports_apm', False))

        # Curvilinear equations have their own topo path (boundary-fitted
        # grid via metric tensors); ``topo_method`` doesn't apply.  Honour
        # the user's ``free_surface`` flag verbatim and skip method
        # resolution.
        is_curvilinear = bool(getattr(self.equation, 'is_curvilinear', False))
        if is_curvilinear:
            return None, bool(free_surface)

        # Legacy back-compat path: ``topo_method='auto'`` + explicit
        # ``free_surface=True`` used to mean "image method".  Honour it,
        # but emit a DeprecationWarning steering users toward
        # ``topo_method=`` for explicit selection.
        if topo_method == 'auto' and free_surface is True:
            import warnings
            warnings.warn(
                "Passing `free_surface=True` along with `topography=` is "
                "deprecated.  When `topography` is given, a free surface "
                "is implicit — use `topo_method='image'` (or omit it) to "
                "select the image / vacuum staircase, or `topo_method="
                "'apm'` for the Cao & Chen 2018 parameter-modified path.",
                DeprecationWarning, stacklevel=4,
            )
            method = 'image'
        elif topo_method == 'auto':
            # Sensible default: APM for equations that support it, else image.
            method = 'apm' if supports_apm else 'image'
        else:
            method = topo_method   # 'image' or 'apm'

        # Validate method against equation capability.
        if method == 'apm' and not supports_apm:
            raise ValueError(
                f"topo_method='apm' requires equation.supports_apm=True; "
                f"{type(self.equation).__name__} only supports the image "
                f"method (use topo_method='image' or omit topo_method)."
            )

        # The method dictates the grid layout:
        #   'image' → top PML suppressed (top halo holds image mirror)
        #   'apm'   → full PML on every side (no special top treatment)
        free_surface_effective = (method == 'image')
        return method, free_surface_effective

    def _process_topography(self, topography):
        """Validate and store irregular free-surface topography.

        ``topography=`` takes a 1-D ``(nx_phys,)`` integer array of
        per-column surface row indices.  The matching 2-D air mask is
        derived internally for the APM path; the image-method path uses
        only the 1-D form.  Method dispatch follows ``self._topo_method``
        (set in :meth:`_resolve_topo_method`):

        ============================ ===============================
        ``free_surface`` (constructor)  Method
        ============================ ===============================
        ``True``                       Image-method / vacuum
                                       staircase (Stage 1 for
                                       :class:`Acoustic`, Stage 2
                                       Robertsson 1996 for
                                       :class:`Elastic`).  Uses
                                       ``_topo_rows_runtime``.
        ``False``                      APM (Cao & Chen 2018,
                                       :class:`Elastic` only).
                                       Uses ``_apm_air_mask_runtime``.
        ============================ ===============================

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

        if self.ndim != 2:
            raise NotImplementedError(
                "topography support is currently limited to 2-D propagators "
                f"(got ndim={self.ndim}). 3-D will be added in Stage 1b."
            )

        import torch

        topo_input = torch.as_tensor(topography)

        # Resolve physical (nz_phys, nx_phys) from the runtime shape +
        # ``free_surface`` flag (image method skips top PML; APM has PML
        # on both top and bottom).
        if self.free_surface:
            nz_phys = self.shape[0] - self.abcn
        else:
            nz_phys = self.shape[0] - 2 * self.abcn
        nx_phys = self.shape[1] - 2 * self.abcn

        # Normalise input → (topo_row_phys, air_mask_phys) tuple.  Each
        # method below picks the form it needs.
        topo_row_phys, air_mask_phys = self._canonicalise_topography(
            topo_input, nz_phys, nx_phys
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

    def _canonicalise_topography(self, topo_input, nz_phys, nx_phys):
        """Validate a 1-D ``(nx_phys,)`` per-column surface row array and
        derive the matching 2-D air mask.

        Returns ``(topo_row_phys, air_mask_phys)`` where ``topo_row_phys``
        is ``long`` shape ``(nx_phys,)`` and ``air_mask_phys`` is
        ``float32`` shape ``(nz_phys, nx_phys)``, with rows strictly
        above ``topo_row_phys[ix]`` marked air (=1.0).
        """
        import torch

        if topo_input.ndim != 1:
            raise ValueError(
                f"topography must be a 1-D array of length nx (surface row "
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
        # Derive the 2-D air mask used by APM.  Image-method ignores it.
        iz = torch.arange(nz_phys, device=topo_row_phys.device).view(-1, 1)
        air_mask_phys = (iz < topo_row_phys.view(1, -1)).to(torch.float32)
        return topo_row_phys, air_mask_phys

    def _populate_image_method_topography(self, topo_row_phys):
        """Set ``self._topo_rows_runtime`` for the image-method /
        vacuum-staircase path.  Translates physical row indices to
        runtime-grid coordinates and pads the x-axis through PML +
        stencil halo so the surface stays continuous through the
        absorbing boundary."""
        import torch
        import torch.nn.functional as F

        halo = self.equation.so // 2
        # Runtime z layout (free_surface=True):
        #   [0, halo)              top stencil halo (image)
        #   [halo, halo + nz_phys) physical interior
        #   [halo + nz_phys, ...)  bottom PML + bottom halo
        topo_z = topo_row_phys + halo

        pad_each = self.abcn + halo
        topo_runtime = F.pad(
            topo_z.to(torch.float32).view(1, 1, -1),
            (pad_each, pad_each),
            mode="replicate",
        ).view(-1).to(torch.long)

        device = getattr(self.equation, "device", None) or self.dev
        if device is not None:
            try:
                topo_runtime = topo_runtime.to(device=device)
            except (RuntimeError, TypeError):
                pass

        self.topography = topo_row_phys
        self._topo_rows_runtime = topo_runtime
        self.equation.topography = topo_row_phys
        self.equation._topo_rows_runtime = topo_runtime

    def _populate_apm_topography(self, air_mask_phys):
        """Set ``self.equation._apm_air_mask_runtime`` for the APM path.
        Replicate-pads the 2-D air_mask through PML + stencil halo on
        all four sides so the surface stays continuous through the
        absorbing boundary."""
        import torch
        import torch.nn.functional as F

        nz_phys, nx_phys = air_mask_phys.shape
        halo = self.equation.so // 2
        pad_each = self.abcn + halo
        air_mask_padded = F.pad(
            air_mask_phys.view(1, 1, nz_phys, nx_phys),
            (pad_each, pad_each, pad_each, pad_each),
            mode="replicate",
        ).view(nz_phys + 2 * pad_each, nx_phys + 2 * pad_each)

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

        nz_phys = self.shape[0] - (self.abcn if self.free_surface else 2 * self.abcn)
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
        if self.free_surface:
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

    def _auto_detect_source_encoding(self, wavelet, sources, receivers):
        wavelet_shape = self._shape_tuple(wavelet)
        sources_shape = self._shape_tuple(sources)
        receivers_shape = self._shape_tuple(receivers)

        if len(wavelet_shape) != 3 or len(sources_shape) != 3 or len(receivers_shape) != 3:
            return False

        if wavelet_shape[0] != 1 or sources_shape[0] != 1 or receivers_shape[0] != 1:
            return False

        if sources_shape[-1] != self.ndim or receivers_shape[-1] != self.ndim:
            return False

        if wavelet_shape[1] != sources_shape[1]:
            return False

        return True

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
            tuple([self.abcn if not self.free_surface else 0] + (2**self.ndim-1) * [self.abcn]),
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
        if self.free_surface:
            return data[..., 0:-self.abcn, self.abcn:-self.abcn]
        else:
            s = slice(self.abcn, -self.abcn)
            return data[(...,) + (s,) * self.ndim]

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
        if halo <= 0:
            offset = [self.abcn] * self.ndim
            if self.free_surface:
                offset[-1] = 0
            return tuple(offset)

        offset = [self.abcn + halo] * self.ndim
        if self.free_surface:
            offset[-1] = halo
        return tuple(offset)

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
