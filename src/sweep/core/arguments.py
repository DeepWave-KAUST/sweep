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

    Consumes the legacy keys from `kwargs` (popping them; PropBase.__init__
    rejects whatever is left over) and returns the merged config, or None when
    neither form was used.

    **An explicit config wins.** `{**legacy, **explicit}` is the order the
    constructor has always used, and it is the right way round: a caller who
    passes `boundary_saving_config` is stating intent, while a stray legacy
    keyword is usually left over from older code. Reversing it would let the
    deprecated spelling silently override the current one.
    """
    legacy = {}
    for old_key, (new_key, convert) in _LEGACY_BOUNDARY_KEYS.items():
        if old_key in kwargs:
            warn_deprecated_spelling(
                f"the loose {old_key}= keyword",
                f"memory=BoundarySaving({new_key}=...)")
            legacy[new_key] = convert(kwargs.pop(old_key))
    if boundary_saving_config is None:
        return legacy or None
    return {**legacy, **boundary_saving_config}


def refuse_free_surface_if_anisotropic(equation, requested, how):
    """Refuse a free surface on an equation that has no correct one.

    Called TWICE, from two different places, and both are needed: once on the
    explicit request (``free_surface=`` as a bool or a per-edge list), and once
    more after the topography method has been resolved, because ``topography=``
    implies a free surface even with ``free_surface=False`` and that is only
    known once ``resolve_topo_method`` has run. The second call is the one that
    was missing -- an anisotropic equation given a topography sailed straight
    past the first check.

    ``how`` names which of the two fired, so the message says what the caller
    actually did rather than just what it could not have.
    """
    if not requested or getattr(equation, "supports_free_surface", True):
        return
    raise NotImplementedError(
        f"{type(equation).__name__} does not support a free surface "
        f"({how}): the anisotropic stress-free boundary condition "
        "couples through the stiffness tensor and is not the isotropic "
        "image method this solver implements. Use free_surface=False "
        "(absorbing top) or an isotropic equation."
    )


def resolve_boundary_spec(free_surface, abcn, ndim, equation, topography,
                          *, normalize_free_surface, normalize_pad,
                          is_top_only_or_none):
    """`(free_surface, abcn)` -> `(fs_faces, pad, abcn_scalar)`, or a refusal.

    The normalisers are injected rather than imported so this module stays free
    of a dependency on the equations package, which imports propagator code.

    Three capability checks live here, and all three REFUSE rather than degrade:

    * an anisotropic equation with any free surface at all -- the anisotropic
      stress-free condition couples through the stiffness tensor and is not the
      isotropic image condition these solvers implement, so accepting the
      request would produce confident, wrong surface physics;
    * a per-edge free surface or per-edge PML in 3-D, or on an equation that has
      not opted in;
    * a per-edge free surface combined with topography.

    The `isinstance(abcn, bool)` exclusion below is belt-and-braces: `normalize_pad`
    already rejects a bool `abcn` outright, so this branch is never the guard that
    fires. It stays because `isinstance(True, int)` is True, and losing both at
    once would silently turn `abcn=True` into a 1-cell PML.

    `abcn_scalar` is a REPRESENTATIVE uniform width kept for the legacy readers
    (topography, curvilinear) that assume one number. It is the max over faces
    when the pad is per-edge -- those readers are guarded to the top-only
    configuration, so the representative value is never the one that matters.
    """
    fs_faces = normalize_free_surface(free_surface, ndim)
    refuse_free_surface_if_anisotropic(equation, any(fs_faces), "free_surface=")
    pad = normalize_pad(abcn, fs_faces, ndim)
    abcn_scalar = (abcn if isinstance(abcn, int) and not isinstance(abcn, bool)
                   else max(pad + (0,)))

    # Per-edge anything is a staged feature. `not isinstance(abcn, int)` catches
    # a per-edge PML spec even when the free surface is plain top-only.
    if (not is_top_only_or_none(fs_faces)) or not isinstance(abcn, int):
        if ndim != 2:
            raise NotImplementedError(
                "per-edge free surface / per-edge PML thickness is currently "
                f"2-D only; got a {ndim}-D propagator (free_surface="
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
    return fs_faces, pad, abcn_scalar


def validate_memory_strategy(use_checkpoint, use_boundary_saving,
                             boundary_tail_steps):
    """Refuse gradient-memory combinations that would silently pick one path.

    Both refusals exist because the failure they prevent is SILENT:

    * checkpointing and boundary saving together -- the compiled wrapper picks
      the checkpoint backward, so the boundary-saving request would simply be
      ignored. Historically ckpt just won; making it an error keeps the
      three-way choice (full / boundary / ckpt) an explicit one.
    * `tail_steps` with checkpointing -- truncated backward is implemented only
      in the boundary-saving backward, and the checkpoint path the wrapper would
      choose ignores the truncation, producing a full-length gradient that looks
      entirely plausible.
    """
    if use_checkpoint and use_boundary_saving:
        raise ValueError(
            "boundary saving and checkpointing are both enabled; the "
            "gradient-memory mode is a three-way choice (full/boundary/"
            "ckpt) -- pass memory=Full() / BoundarySaving(...) / Ckpt(...), or "
            "disable "
            "one of use_ckpt/boundary_saving_config.")
    if boundary_tail_steps and use_checkpoint:
        raise NotImplementedError(
            "tail_steps requires the boundary-saving backward; pass "
            "use_ckpt=False (or memory=BoundarySaving()).")


# ===========================================================================
# DEPRECATED SPELLINGS -- delete this section, and the functions marked with
# `_deprecated`, in one go.
#
# Removal criterion, so that "is it safe yet?" is checkable rather than a
# judgement call years from now:
#
#   1. sweep-tasks' YAML emitter writes only the flat `kind:` form
#      (yaml_io.py, currently emits `strategy:` + a same-named sub-block);
#   2. no YAML under the campaign directories still uses `strategy:` with a
#      same-named sub-block (22 files did at the time of writing);
#   3. `test/test_memory_legacy_spellings.py` -- which exists only to pin what
#      this section accepts -- has no remaining callers to protect.
#
# When all three hold: delete this section, the legacy branches of
# `as_memory_strategy`, and that test file. The test file is the checklist;
# deleting it is the confirmation.
#
# The stored experiment YAML is the reason this is a deprecation rather than a
# removal. Those files record what was actually run, so they are read, not
# rewritten.
# ===========================================================================

_WARNED_SPELLINGS = set()


def warn_deprecated_spelling(what: str, instead: str, *, stacklevel: int = 3):
    """Warn once per process per spelling.

    Once-per-spelling rather than once-per-call because the alternative is a
    line of noise per propagator construction, and a warning that scrolls is a
    warning that gets filtered out wholesale -- taking the ones that matter with
    it. ``DeprecationWarning`` is hidden by default in library code but pytest
    surfaces it, which is where the remaining callers actually are.
    """
    if what in _WARNED_SPELLINGS:
        return
    _WARNED_SPELLINGS.add(what)
    warnings.warn(
        f"{what} is deprecated; use {instead} instead. "
        "The old spelling still works and is still read.",
        DeprecationWarning, stacklevel=stacklevel,
    )
