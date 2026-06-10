"""Eager (pure-PyTorch) boundary-saving setup for the eager propagator.

Split out of ``_PropTorchEager`` as a mixin: reverse-driver dispatch, per-rollout
state init, and the boundary-saving time loop.  Mixed into ``_PropTorchEager``
(resolved via MRO); the per-step physics + adjoint live in ``_eager_boundary_saving``.
"""
import torch


class _EagerBoundarySavingMixin:
    """Eager boundary-saving methods for ``_PropTorchEager``."""

    def enable_eager_boundary_saving(self, flag=True, mode=None, self_check=True,
                                     check_tol=None, storage_dtype="fp32", storage="gpu"):
        """Route the eager rollout through pure-PyTorch boundary saving (see
        ``_eager_boundary_saving``) instead of the full autograd tape.  Mutually
        exclusive with ``use_ckpt``.  ``mode`` forces a reverse driver
        ('swap2nd' | 'substep'); None auto-dispatches.

        ``self_check`` (default True) runs a one-time interior forward-
        consistency probe on the first backward step: if the reverse driver does
        not invert the step (``func(reconstructed S_i)`` differs from ``S_{i+1}``
        in the lossless interior by rel-L2 > ``check_tol``), the propagator
        raises instead of returning a wrong gradient.  Set ``self_check=False``
        to skip the probe once an equation has been validated.

        ``storage_dtype`` ('fp32' | 'fp16' | 'bf16' | 'int8') compresses the
        per-step boundary ring only (compute and the seed frame stay FP32) —
        same split as the CUDA path.  ``storage`` ('gpu' | 'cpu') keeps the ring
        buffer on device or offloads it to host RAM — the ring is the dominant
        eager-BS GPU cost, so 'cpu' trades device memory for per-step H2D/D2H
        copies.  ``check_tol=None`` picks a tolerance that clears the storage
        quantization floor while still catching a broken reverse (O(0.1-1)
        error)."""
        self._eager_bs = bool(flag)
        self._eager_bs_mode = mode
        self._eager_bs_self_check = bool(self_check)
        self._eager_bs_storage_dtype = storage_dtype
        self._eager_bs_storage = storage
        if check_tol is None:
            check_tol = {"fp32": 1e-2, "fp16": 3e-2, "bf16": 6e-2, "int8": 1.5e-1}.get(storage_dtype, 1e-2)
        self._eager_bs_check_tol = float(check_tol)

    def _eager_bs_reverse_mode(self):
        """Pick (and validate) the exact reverse-reconstruction driver for eager
        boundary saving, dispatching on the equation's time scheme.  Both drivers
        reuse the forward physics — there is no per-equation adjoint::

            2nd-order             -> 'swap2nd'  (reuses func with time levels swapped)
            1st-order + substeps  -> 'substep'  (reuses interior_substeps reversed)

        1st-order equations *without* ``interior_substeps`` are unsupported (no
        approximate fallback).  ``enable_eager_boundary_saving(mode=...)`` can
        force a mode.  Raises if the forced/derived mode is unavailable, or unsafe
        for the current BC (a free surface the substeps would silently drop).
        """
        from sweep.equations.base import SecondOrderEquation
        is_2nd = isinstance(self.equation, SecondOrderEquation)
        has_substeps = callable(getattr(self.equation, "interior_substeps", None))
        forced = getattr(self, "_eager_bs_mode", None)
        if forced is not None:
            from sweep.propagator._eager_boundary_saving import _REVERSE_DRIVERS
            if forced not in _REVERSE_DRIVERS:
                raise ValueError(
                    f"Unknown eager boundary-saving reverse mode {forced!r}; "
                    f"registered drivers: {sorted(_REVERSE_DRIVERS)}."
                )
            if forced == "substep" and not has_substeps:
                raise ValueError(
                    f"reverse mode 'substep' needs {type(self.equation).__name__}"
                    ".interior_substeps()."
                )
            reverse_mode = forced
        elif is_2nd:
            reverse_mode = "swap2nd"
        elif has_substeps:
            reverse_mode = "substep"
        else:
            raise NotImplementedError(
                "Eager boundary saving for the 1st-order equation "
                f"'{type(self.equation).__name__}' requires an "
                "interior_substeps() hook (the exact reverse driver). Add one "
                "(reusing the equation's step core), or use impl='c' / chunk "
                "checkpointing (use_ckpt=True)."
            )
        # The substep reconstruction carries the free-surface BC only if the
        # equation's interior_substeps do (declared via supports_bs_free_surface);
        # the ring geometry also assumes full PML.  swap2nd is always FS-safe
        # (reuses ``func``).  Refuse rather than return a wrong near-surface grad.
        if (
            reverse_mode == "substep"
            and getattr(self, "free_surface", False)
            and not getattr(self.equation, "supports_bs_free_surface", False)
        ):
            raise NotImplementedError(
                "Eager boundary saving with a free surface is not supported "
                f"for {type(self.equation).__name__} (its interior_substeps "
                "would drop the free-surface BC). Options: free_surface=False, "
                "impl='c', or chunk checkpointing (use_ckpt=True). Equations "
                "whose interior_substeps carry the free-surface BC (e.g. "
                "Elastic) set supports_bs_free_surface=True."
            )
        return reverse_mode

    def _init_eager_bs(self, wavefield, runtime_models, wavelet, src, nt):
        """Set up the per-rollout state for eager (pure-PyTorch) boundary saving
        and return ``(state, base_cfg, multi_receiver)``.

        Builds the ring geometry, resolves the reverse driver, pre-allocates the
        reconstruction storage (CUDA-style contiguous buffers) seeded with the
        initial frame, prepares the (optionally compiled) per-step physics
        callables, and assembles ``base_cfg`` — the part of the per-step config
        that does not change across time.  The caller passes the varying
        ``step``/``time_index`` to ``_BoundarySaveStep.apply``.  Reconstruction
        state is per-rollout (no class globals).
        """
        from sweep.propagator._eager_boundary_saving import (
            ReconState, _interior_index_tuple, _ring_index_tuples, save_step,
        )
        from sweep.equations.base import SecondOrderEquation

        multi_receiver = len(self.receiver_indices) > 1
        halo = self._runtime_fd_halo()
        offsets = tuple(self.abcn + halo for _ in range(self.ndim))
        shape = tuple(wavefield[0].shape)
        rings = _ring_index_tuples(shape, self.ndim, offsets, halo)
        interior_idx = _interior_index_tuple(shape, self.ndim, offsets)
        nwf = len(self.wavefield_names)
        nm = len(runtime_models)
        reverse_mode = self._eager_bs_reverse_mode()

        cpml_indices = [
            k for k, name in enumerate(self.wavefield_names)
            if getattr(self._wavefield_spec_index.get(name), "boundary_related", False)
        ]
        phys_indices = [k for k in range(nwf) if k not in cpml_indices]
        # 2nd-order wavefields are stored as consecutive (now, prev) physical
        # pairs.  swap2nd reconstructs EVERY pair — so a multi-field equation
        # (e.g. LSRTM: background h1/h2 + scattered sh1/sh2) recovers its full
        # physical state, not only field 0.  A single-field equation has one pair
        # (the original path).  ring_fields keys off the equation *type*, not the
        # reverse mode, so a forced mode cannot change which fields are stored.
        is_2nd = isinstance(self.equation, SecondOrderEquation)
        second_order_pairs = [
            (phys_indices[k], phys_indices[k + 1])
            for k in range(0, len(phys_indices) - 1, 2)
        ]
        ring_fields = [now for now, _ in second_order_pairs] if is_2nd else phys_indices

        # Pre-allocate the whole ring storage as big contiguous buffers (CUDA-
        # style), then write each step into a slice — no per-step allocation.
        # Seed step 0 with the initial frame.
        store_dtype = getattr(self, "_eager_bs_storage_dtype", "fp32")
        storage_loc = getattr(self, "_eager_bs_storage", "gpu")
        storage_device = wavefield[0].device if storage_loc == "gpu" else torch.device("cpu")
        state = ReconState(nt, rings, shape, len(ring_fields), store_dtype,
                           wavefield[0].device, storage_device,
                           compute_dtype=wavefield[0].dtype)
        for pos, f in enumerate(ring_fields):
            save_step(state, 0, pos, wavefield[f])

        # Per-step physics callables, built ONCE and reused across every step + in
        # backward.  With use_compile=True the inner step is torch.compile'd; the
        # custom autograd.Function and the time loop stay eager (Dynamo treats the
        # Function as an opaque boundary), so only the per-step kernels are fused.
        bs_func = self._maybe_compile(self.equation.func)
        bs_substeps = (
            [self._maybe_compile(ss) for ss in self.equation.interior_substeps()]
            if reverse_mode == "substep" else None
        )

        # Constant per-step config; step/time_index pass to apply() as args.
        base_cfg = {
            "equation": self.equation, "dt": self.dt, "h": self._equation_spacing,
            "nwf": nwf, "nm": nm, "nt": nt,
            "state": state, "models": runtime_models, "src": src,
            "source_indices": self.source_indices, "wavelet": wavelet,
            "reverse": reverse_mode, "ring_fields": ring_fields,
            "cpml_indices": cpml_indices, "phys_indices": phys_indices,
            "pairs": second_order_pairs, "interior_idx": interior_idx,
            "self_check": getattr(self, "_eager_bs_self_check", True),
            "check_tol": getattr(self, "_eager_bs_check_tol", 1e-2),
            "func": bs_func, "substeps": bs_substeps,
        }
        return state, base_cfg, multi_receiver

    def _rollout_eager_bs(self, wavefield, runtime_models, wavelet, src, rec,
                          record, receivers, nt, adj):
        """Time loop under eager (pure-PyTorch) boundary saving: each physics step
        is wrapped in a ``_BoundarySaveStep`` autograd.Function that stores only
        the boundary ring (source injection + receiver sampling stay normal ops —
        their gather/scatter backward is value-free, so no full wavefield is
        retained).  Per-rollout setup lives in ``_init_eager_bs``; the loop passes
        the varying step/time straight to ``_BoundarySaveStep.apply``.  Writes the
        record in place, seeds the reconstruction with the final frame, and
        returns the final wavefield list.
        """
        from sweep.propagator._eager_boundary_saving import _BoundarySaveStep
        state, base_cfg, multi_receiver = self._init_eager_bs(
            wavefield, runtime_models, wavelet, src, nt
        )
        for i in range(nt):
            time = i if not adj else nt - i - 1
            wavefield = list(_BoundarySaveStep.apply(base_cfg, i, time, *wavefield, *runtime_models))
            for source_idx in self.source_indices:
                wavefield[source_idx] = src(wavefield[source_idx], wavelet[..., time])
            if multi_receiver:
                record[:, i, :, :] = rec.sample_fields([wavefield[idx] for idx in self.receiver_indices])
            else:
                receiver_idx = self.receiver_indices[0]
                record[:, i, :, 0] = rec(wavefield[receiver_idx]).view(*receivers.shape[:-1])
        # Seed the reconstruction with the final full frame (detached).
        state.frame = [w.detach() for w in wavefield]
        return wavefield

