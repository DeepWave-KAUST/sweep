"""Turning what the caller passed into the canonical internal form.

`PropBase.__init__` accepts arguments in several historical spellings -- a
deprecated `dev` alongside `device`, a scalar or per-axis `dh`, boundary options
as either loose keywords or a config dict. Resolving each is a pure function of
its inputs, so it belongs here rather than as another twenty lines of a
360-line constructor: these can be tested without building a propagator, and a
propagator is an expensive thing to build.

Distinct from :mod:`sweep.core.validation`, which is about the *call* signature
(wavelet, sources, receivers). This module is about the *constructor*.

Distinct from :mod:`sweep.core.geometry`, which is arithmetic on an
already-resolved grid. Spacing lands here because normalising it is argument
handling; what the grid then does with it is geometry.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence


def resolve_device(device, dev, equation):
    """Pick the device, honouring `device` over the deprecated `dev`.

    Falls back to the equation's own device, which is the source of truth: its
    operators (laplace kernels and friends) were already built there, so
    disagreeing with it would mean a silent cross-device copy per step.

    The warning fires only when the two arguments actually DISAGREE. Passing the
    same device twice is redundant, not a mistake, and warning about it would
    train callers to ignore the warning.
    """
    if device is not None and dev is not None and device != dev:
        warnings.warn(
            "Both 'device' and 'dev' were passed to the propagator; using 'device'. "
            "'dev' is deprecated and will be removed in a future release.",
            DeprecationWarning, stacklevel=3,
        )
    resolved = device if device is not None else dev
    if resolved is None:
        resolved = getattr(equation, 'device', None)
    return resolved


def normalise_spacing(dh, ndim, shape=None):
    """`dh` -> `(dh_scalar, grid_spacing)`, both in shape order.

    Returns the per-axis tuple and the scalar the rest of the code still uses.
    The scalar is the LAST axis (x), not the first and not a mean: on a
    non-uniform grid the old code took `_grid_spacing[-1]`, and anything else
    would change every stencil that still reads the scalar.

    The str/bytes guard below is currently UNREACHABLE and kept deliberately:
    `np.isscalar("10.0")` is True, so a numeric string takes the scalar branch
    and `float()` accepts it -- `dh="10.0"` has always behaved as `dh=10.0`.
    That is pre-existing behaviour, preserved here rather than tightened,
    because this refactor's contract is that nothing changes. The guard stays
    because it becomes live the moment the scalar test stops absorbing strings.
    """
    import numpy as np

    if np.isscalar(dh):
        scalar = float(dh)
        return scalar, tuple([scalar] * ndim)

    if not isinstance(dh, Sequence) or isinstance(dh, (str, bytes)):
        raise TypeError(
            "dh must be a float or a sequence ordered like shape "
            "(2D: (dz, dx), 3D: (dz, dy, dx))."
        )
    if len(dh) != ndim:
        raise ValueError(
            f"dh must have length {ndim} to match shape {shape}, "
            f"got {len(dh)}."
        )
    grid_spacing = tuple(float(v) for v in dh)
    return float(grid_spacing[-1]), grid_spacing


# Loose keyword -> key in the boundary-saving config dict. Some are renames,
# `boundary_on_cpu` is a bool that became a two-valued enum.
_LEGACY_BOUNDARY_KEYS = {
    "transfer_interval": ("transfer_interval", lambda v: v),
    "boundary_on_cpu": ("storage", lambda v: "cpu" if v else "gpu"),
    "use_pinned_memory": ("pinned_memory", lambda v: v),
    "boundary_disk_async_read": ("disk_async_read", lambda v: v),
}


def merge_legacy_boundary_kwargs(kwargs, boundary_saving_config):
    """Fold the pre-config-dict boundary keywords into `boundary_saving_config`.

    Consumes the legacy keys from `kwargs` (popping them, so an unrecognised
    leftover still surfaces as an unexpected keyword) and returns the merged
    config, or None when neither form was used.

    **An explicit config wins.** `{**legacy, **explicit}` is the order the
    constructor has always used, and it is the right way round: a caller who
    passes `boundary_saving_config` is stating intent, while a stray legacy
    keyword is usually left over from older code. Reversing it would let the
    deprecated spelling silently override the current one.
    """
    legacy = {}
    for old_key, (new_key, convert) in _LEGACY_BOUNDARY_KEYS.items():
        if old_key in kwargs:
            legacy[new_key] = convert(kwargs.pop(old_key))
    if boundary_saving_config is None:
        return legacy or None
    return {**legacy, **boundary_saving_config}
