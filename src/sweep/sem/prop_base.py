"""PropBase-compatible torch propagator for the SEM backend.

An adapter that drives the SEM solver with the *same constructor and call* as
the FD ``sweep.propagator.PropTorch``::

    prop = PropSEM(AcousticSEM2D, shape=(nz, nx), dh=DH, dt=DT,
                   free_surface=True)
    seis = prop(wavelet, sources, receivers, models=[vp_grid])   # (nshots, nt, nrec, 1)

    # 3-D: same call, a (nz, ny, nx) grid model.
    prop = PropSEM(AcousticSEM3D, shape=(nz, ny, nx), dh=DH, dt=DT)
    seis = prop(wavelet, sources, receivers, models=[vp_grid])

It builds on :class:`SEMPropagator` and shares PropTorch's option types (the
free-surface edge spec and ``Full`` / ``BoundarySaving`` / ``Ckpt``).

Conventions matched to ``PropTorch``
------------------------------------
* ``shape`` is the model; the absorbing layer -- ``abcn`` grid cells, whole
  elements -- is added outside it on every side without a free surface, and
  the model is edge-extended into it. Its gradient stops at the model edge,
  as FD's ``EdgePadding`` does.
* ``free_surface`` -- ``True`` (the top), edge names, a per-edge mask or a
  ``{edge: bool}`` dict; default ``False``. ``topography`` -- the surface row
  of every column, ``(nx,)`` / ``(ny, nx)``, fractional allowed -- puts one on
  top, and the curved mesh follows it.
* ``sources`` / ``receivers`` — grid indices, column order ``[ix, iz]``
  in 2-D and ``[ix, iy, iz]`` in 3-D (same as the FD backends). ``sources`` is
  ``(nshots, ndim)`` (one source per shot) or ``(1, nsrc, ndim)`` (source
  encoding); ``receivers`` is ``(nshots, nrec, ndim)`` or ``(1, nrec, ndim)``.
  Indices are converted to physical metres via ``i*dh``.
* ``wavelet`` — ``(nt,)`` shared across shots, ``(nshots, nt)`` per-shot, or (in
  source-encoding mode) ``(nsrc, nt)`` per superposed source. Its length is
  ``nt``; the constructor's ``nt`` is optional and, when given, checked.
* ``models`` — FD-style grid tensor ``(nz, nx)`` / ``(nz, ny, nx)``. It is
  multilinearly and *differentiably* interpolated to the SEM GLL nodes, so
  ``loss.backward()`` returns ``d loss / d model`` on the grid — directly
  comparable to the FD gradient. A GLL-native ``(E, ngll, ...)`` tensor is also
  accepted and used as-is.
* returns the seismogram tensor ``(nshots, nt, nrec, ncomp)`` -- the SAME
  layout as the FD ``PropTorch``, with ``ncomp = 1`` for acoustic. With
  ``nshots > 1`` the
  shots run as a real ``torch.vmap`` batch (one batched kernel per step), not a
  Python loop; a single shot uses the plain single-shot path. Gradients flow
  through the batch.
* ``source_type`` -- the FD names as well as the equation's own: on the 2-D
  elastics ``["vz"]`` is a vertical force and ``["sxx", "szz"]`` an explosion,
  as on ``PropTorch`` (see :func:`_resolve_vector_source`).
* ``return_wavefield=True`` (eager) -- ``(seis, snapshots)``, the snapshots
  ``(nsnap, nfield, nshots, 1, *grid)`` on the grid of the model and its
  absorbing layer, PropTorch's layout: the SEM field evaluated at every grid
  point through its element's basis, zero above a topography.
* ``memory`` -- ``Full()`` (the default), ``BoundarySaving(storage_dtype=...)``
  or ``Ckpt(chunks=...)``, given to the constructor.
* ``impl`` -- ``None`` (``'auto'``) runs the CUDA kernels where the equation has
  them and the device is CUDA, eager otherwise; ``'eager'`` / ``'c'`` choose.

Scope
-----
2-D: constant-density acoustic on a flat Quad4 mesh (:class:`AcousticSEM2D`) or a
Quad9 deformed mesh that follows a ``topography``
(:class:`AcousticSEM2DDeformed`), and the 2-D elastics. 3-D: acoustic on Hex8 /
Hex27 meshes and the 3-D elastics. The same
``solver(wavelet, sources, receivers, models)`` call drives all of them.
Autograd gradients (no manual ``adj``); boundary saving + gradient
checkpointing supported.
"""

from __future__ import annotations

import dataclasses
import itertools
import numbers

import torch
import torch.utils.checkpoint

from .mesh import MESH_BUILDERS
from .mesh.hex8 import hex8_layout
from .cpml.acoustic import build_cpml
from .cpml.acoustic3d import build_cpml3d
from .equations.acoustic2d import AcousticSEM2D
from .equations.acoustic3d import AcousticSEM3D
from .equations.acoustic3d_deformed import AcousticSEM3DDeformed
from .equations.acoustic_potential2d import AcousticPotentialSEM2D
from .fields import (MESH_NDIM, format_field_specs, mesh_kind, model_names,
                     resolve_field_types)
from .propagator import SEMPropagator, receiver_gather



# Hex27 (deformed / topography) 3-D equations — the only 3-D ones accepting
# ``topography``; on a flat mesh they reduce to their axis-aligned twins.




# Side names in ``shape`` axis order: 2-D shape is (z, x), 3-D is (z, y, x), and
# each axis has a low and a high side. ``top`` is z-low, which is why it is the
# side that is not padded when it carries the free surface.
_SIDES = {2: (("top", 0, 0), ("bottom", 0, 1),
              ("left", 1, 0), ("right", 1, 1)),
          3: (("top", 0, 0), ("bottom", 0, 1),
              ("front", 1, 0), ("back", 1, 1),
              ("left", 2, 0), ("right", 2, 1))}


def _outside_model(raw, n):
    """Which nodes lie OUTSIDE the model grid ``[0, n-1]`` along one axis,
    from their grid coordinate ``raw`` (before the clamp).

    By position, so a node on the model edge is inside on BOTH elements that
    share it -- the rule the node-grid strip interpolation (dd.py) needs, and
    hence the one that keeps eager and impl='c' gradients equal. The
    tolerance only has to absorb round-off of an edge node's coordinate: the
    nearest GLL node off the edge is a sizeable fraction of a cell away
    (0.69 of one at n_order=4).
    """
    tol = 1e-3
    return (raw < -tol) | (raw > (n - 1) + tol)


def _resolve_free_surface(free_surface, ndim, topography):
    """``free_surface`` as PropTorch takes it -> the tuple of side names that
    carry one.

    A bool (``True`` is the top), an edge name or names, a per-edge mask in
    the edge order, or an ``{edge: bool}`` dict -- PropTorch's own parser. A
    topography is a free surface on top, as it is there. The solver itself
    knows no sides -- a free surface is a node set the step zeroes, or the
    weak form's natural condition on a side with no absorbing layer.
    """
    from sweep.equations._edges import normalize_free_surface
    faces = normalize_free_surface(free_surface, ndim)
    sides = [nm for (nm, _, _), on in zip(_SIDES[ndim], faces) if on]
    if topography and "top" not in sides:
        sides.insert(0, "top")
    return tuple(sides)


def _resolve_abcn(abcn, n_order):
    """``abcn`` -> the absorbing layer in grid cells: one width for every side
    without a free surface, in whole elements; ``None`` is ten elements."""
    if abcn is None:
        return 10 * n_order
    if isinstance(abcn, (tuple, list, dict)):
        raise NotImplementedError(
            f"PropSEM takes one abcn for every absorbing side; got {abcn!r}.")
    if isinstance(abcn, bool) or not isinstance(abcn, numbers.Integral):
        raise TypeError(f"abcn is a number of grid cells; got {abcn!r}")
    abcn = int(abcn)
    if abcn < 0:
        raise ValueError(f"abcn must be >= 0; got {abcn}")
    if abcn % n_order:
        up = -(-abcn // n_order) * n_order
        raise ValueError(
            f"abcn={abcn} is not a whole number of elements: the absorbing "
            f"layer is made of elements of n_order={n_order} cells, so abcn "
            f"must be a multiple of {n_order} (e.g. {up}).")
    return abcn


def _resolve_topography(topography, shape):
    """``topography`` -> the surface rows, float64 on the host: one per
    column, ``(nx,)`` in 2-D and ``(ny, nx)`` in 3-D, PropTorch's form.
    Fractional rows are kept: a curved mesh needs no staircase."""
    if topography is None:
        return None
    if callable(topography):
        raise TypeError(
            "topography is the surface row of every column, as on PropTorch "
            "-- an (nx,) array, (ny, nx) in 3-D, fractional rows allowed -- "
            "not a function: sample it, e.g. z_top(np.arange(nx) * dh) / dh.")
    t = torch.as_tensor(topography).detach().to(device="cpu", dtype=torch.float64)
    want = tuple(shape[1:])
    if tuple(t.shape) != want:
        raise ValueError(
            f"topography needs one surface row per column, shape {want}; "
            f"got {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("topography holds non-finite rows")
    if float(t.min()) < 0 or float(t.max()) >= shape[0] - 1:
        raise ValueError(
            f"topography rows must lie in [0, {shape[0] - 1}) -- inside the "
            f"model, above its last row; got [{float(t.min()):g}, "
            f"{float(t.max()):g}]")
    return t


class _SurfaceRows:
    """The ``z_top`` a curved mesh is built from: the topography rows in
    metres, linear between columns -- so exact on them, which is where an
    even ``n_order`` puts the mesh's control nodes -- and flat across the
    absorbing layer. A class rather than a closure, so a propagator stays
    picklable."""

    def __init__(self, rows, dh, origin):
        self.z = rows * float(dh)                  # metres, (nx,) or (ny, nx)
        self.dh = float(dh)
        self.origin = tuple(origin)                # model's first (y,) x column

    def _axis(self, p, origin, n):
        c = (p.double() / self.dh - origin).clamp(0.0, n - 1)
        i0 = c.floor().long().clamp(max=n - 1)
        return i0, (i0 + 1).clamp(max=n - 1), c - i0

    def __call__(self, x, y=None):
        z = self.z.to(x.device)
        x0, x1, fx = self._axis(x, self.origin[-1], z.shape[-1])
        if y is None:
            out = z[x0] + fx * (z[x1] - z[x0])
        else:
            y0, y1, fy = self._axis(y, self.origin[0], z.shape[0])
            lo = z[y0, x0] + fx * (z[y0, x1] - z[y0, x0])
            hi = z[y1, x0] + fx * (z[y1, x1] - z[y1, x0])
            out = lo + fy * (hi - lo)
        return out.to(x.dtype)


def _resolve_device(device, dev):
    """``device`` or ``dev``, as PropTorch takes them; the CPU when neither."""
    from sweep.core.arguments import resolve_device
    return resolve_device(device, dev, None) or "cpu"


def _lagrange(nodes, x):
    """The Lagrange basis on ``nodes`` at the points ``x``: ``(len(x),
    len(nodes))``, exactly 1/0 at a node."""
    out = torch.ones(x.numel(), nodes.numel(), dtype=torch.float64)
    for j in range(nodes.numel()):
        for k in range(nodes.numel()):
            if k != j:
                out[:, j] *= (x - nodes[k]) / (nodes[j] - nodes[k])
    return out


def _c_shared_wavelet(w):
    """The C backends take one ``(nt,)`` wavelet per shot. Raised from inside
    each per-shot loop, never ahead of it: the shots before this one have
    already had their PML placement checked, and that order is contractual."""
    if w.dim() != 1:
        raise NotImplementedError(
            "impl='c' supports a shared (nt,) wavelet per shot "
            f"(got {tuple(w.shape)}); per-source wavelets are not wired yet.")
    return w


# --------------------------------------------------------------------------- #
# FD source names on the vector (elastic) equations. PropTorch injects a source
# by adding the wavelet to the named fields every step: into a velocity that is
# a body force along its axis, into a stress a moment-tensor component whose
# RATE is the wavelet. A SEM elastic equation has neither as a source field of
# its own -- its sources are the weak-form loads on u / v -- so the names are
# translated into the load that does the same thing.
# --------------------------------------------------------------------------- #
_FD_FORCE = {2: ("vx", "vz"), 3: ("vx", "vy", "vz")}
_FD_STRESS = {2: ("sxx", "szz", "sxz"),
              3: ("sxx", "syy", "szz", "sxy", "sxz", "syz")}
# 2-D (Mxx, Mzz, Mxz) slot of each stress name, the order moment_tensor_stamp
# takes; sxz drives Mxz = Mzx, the one shear component
_MOMENT_SLOT = {"sxx": 0, "szz": 1, "sxz": 2}


def _resolve_vector_source(equation, source_type, ndim):
    """``source_type`` on an elastic equation -> ``(names, force_axes, moment)``.

    Three vocabularies, all FD-shaped lists of names:

    * the equation's own source fields (``u_z``, ``v_x``, ...) and the FD
      velocity names (``vx``, ``vz``, ``vy``): a body force, one unit component
      per name -- so ``["vx", "vz"]`` is a diagonal force, not an explosion;
    * the FD stress names (``sxx``, ``szz``, ``sxz``; 2-D only): the moment
      tensor with a unit entry per name -- ``["sxx", "szz"]`` is FD's explosion.

    Returns the validated names, the force axes (``None`` for a moment
    source) and ``(Mxx, Mzz, Mxz)`` (``None`` for a force). A force and a
    moment cannot be mixed: they take the wavelet differently (see
    :class:`_Acq`), and one call drives one time function.
    """
    own = [s.name for s in equation.field_specs if s.supports_source]
    if not source_type:
        names = resolve_field_types(equation, None, "source")
        return names, frozenset(own.index(n) for n in names), None
    fd_force, fd_stress = _FD_FORCE[ndim], _FD_STRESS[ndim]
    name = equation.__name__
    out, axes, moment = [], {}, [0.0, 0.0, 0.0]
    for want in source_type:
        if want in out:
            raise ValueError(f"duplicate source_type entry '{want}'")
        if want in own or want in fd_force:
            ax = own.index(want) if want in own else fd_force.index(want)
            if ax in axes:
                raise ValueError(
                    f"source_type entries '{axes[ax]}' and '{want}' are the "
                    f"same force component on {name}")
            axes[ax] = want
        elif want in fd_stress:
            if ndim == 3:
                raise NotImplementedError(
                    f"stress (moment-tensor) source_type '{want}' is 2-D only: "
                    f"the 3-D point stamp carries no basis derivatives yet. "
                    f"Use a force ({', '.join(own)} or {', '.join(fd_force)}).")
            moment[_MOMENT_SLOT[want]] = 1.0
        else:
            avail = [s for s in equation.field_specs if s.supports_source]
            raise ValueError(
                f"Unknown source_type entry '{want}' for {name}. Available "
                f"source_type values:\n{format_field_specs(avail)}"
                f"\nFD names: {', '.join(fd_force)} (force), "
                f"{', '.join(fd_stress)} (stress, 2-D)")
        out.append(want)
    if axes and any(moment):
        raise ValueError(
            f"source_type {list(source_type)} mixes a force with a stress "
            f"source: the force takes the wavelet as itself, the stress as its "
            f"rate (as FD does), so they need two time functions. Run them as "
            f"two calls and add the records.")
    if any(moment):
        return out, None, tuple(moment)
    return out, frozenset(axes), None


def _resolve_snapshot_steps(nt, return_wavefield, snapshot_times,
                            snapshot_interval):
    """The steps to copy the wavefield at -- ``PropTorch``'s rule: the named
    ``snapshot_times``, every ``snapshot_interval``-th step, or every step;
    ``None`` when no wavefield is asked for."""
    if not return_wavefield:
        return None
    if snapshot_times is not None and snapshot_interval is not None:
        raise ValueError("Use either snapshot_times or snapshot_interval, not both.")
    if snapshot_times is not None:
        steps = sorted({int(t) for t in snapshot_times})
    elif snapshot_interval is not None:
        if int(snapshot_interval) < 1:
            raise ValueError("snapshot_interval must be >= 1.")
        steps = list(range(0, nt, int(snapshot_interval)))
    else:
        steps = list(range(nt))
    bad = [t for t in steps if not 0 <= t < nt]
    if bad:
        raise ValueError(
            f"Snapshot time(s) {bad} outside the valid range [0, {nt - 1}].")
    return steps


# --------------------------------------------------------------------------- #
# The unknown-option hint table. Module level because it is a constant of the
# signature, not of a call: a **kwargs catch-all silently swallowed ANY unknown
# option on both backends -- including a misspelt memory= (which then ran the
# full tape while the caller believed boundary saving was on).
# --------------------------------------------------------------------------- #
_OPTION_HINTS = {
    "record_field_every": " (PropSEM takes return_wavefield=True with "
                          "snapshot_interval= or snapshot_times=, as PropTorch "
                          "does)",
    "memory": " (a constructor option, as on PropTorch: PropSEM(..., "
              "memory=BoundarySaving(...)) or memory=Ckpt(chunks=...))",
    "bs_store_dtype": " (the constructor's memory=BoundarySaving("
                      "storage_dtype=...))",
    "bs_store_dtype_elem": " (a constructor option)",
    "checkpoint_chunk": " (the constructor's memory=Ckpt(chunks=...))",
}

# Constructor keywords that were renamed or removed, with what replaces them:
# a TypeError naming the replacement instead of Python's bare one.
_CTOR_HINTS = {
    "abcn_elements": "abcn= in grid cells (abcn_elements * n_order)",
    "model_padding": "nothing: the absorbing layer is always added outside "
                     "the model, as on PropTorch",
}


def _reject_unsupported(self, kwargs, *, compile_step, adj,
                        return_wavefield, models,
                        snapshot_times=None, snapshot_interval=None):
    """The argument rejections of one ``_forward_fields`` call, IN ORDER.

    The order is contract: each raise has (or could have) a test pinning its
    exception type and message, and two of them fire on the same call --
    ``impl='c'`` + ``compile_step`` + ``adj=True`` must keep raising the
    compile_step ``NotImplementedError``, not the adj one. Kept as one
    straight-line helper for exactly that reason: a dict-driven table would
    make the winner depend on iteration order.
    """
    if kwargs:
        det = "; ".join(k + _OPTION_HINTS.get(k, "") for k in sorted(kwargs))
        raise TypeError(
            f"PropSEM.forward got unknown option(s): {det}. Every "
            f"supported option is a named parameter -- check the "
            f"signature; a typo here used to be silently ignored.")
    if self._use_c and compile_step:
        raise NotImplementedError(
            "compile_step compiles the eager eq.step; impl='c' runs a "
            "compiled CUDA loop already and has no step to wrap. It used "
            "to be accepted and silently ignored.")
    if adj:
        raise NotImplementedError(
            "PropSEM produces gradients via autograd; adj=True is not supported.")
    if return_wavefield:
        if self._use_c:
            raise NotImplementedError(
                "return_wavefield needs impl='eager': the CUDA kernels run the "
                "whole time loop inside one call and keep no intermediate "
                "state to copy out.")
        if self._memory == "boundary":
            raise ValueError(
                "return_wavefield is not available with memory=BoundarySaving() "
                "-- the forward state is discarded and reconstructed.")
        if self._ckpt_chunk:
            raise ValueError(
                "return_wavefield runs the shots one by one, where Ckpt (a "
                "knob of the vmapped multi-shot path) would be silently "
                "ignored.")
    elif snapshot_times is not None or snapshot_interval is not None:
        # PropTorch ignores these without return_wavefield; here they are
        # refused, since a caller who passed them expects snapshots back
        raise ValueError(
            "snapshot_times / snapshot_interval need return_wavefield=True.")
    # `not models` on a multi-element tensor raises RuntimeError first -- that
    # is why a bare-tensor `models` never reaches the tensor branches of
    # _resolve_model_batch / _dd3d_route. Keep the expression as written.
    if not models or len(models) != len(self.model_names):
        raise ValueError(f"this equation needs models={self.model_names}.")


# --------------------------------------------------------------------------- #
# One call's acquisition. Built once so the two backends cannot disagree on
# any of it -- the shot count, the source coordinates in metres, which wavelet
# each shot gets and the receiver batch are read by the DD route, the CUDA
# route and the eager factory alike, and a second normalisation is how they
# would drift. The dd._StripCtx placement and shape (slots, one __init__ that
# validates, no compute).
# --------------------------------------------------------------------------- #
class _Acq:
    """Sources, receivers and wavelet of one ``_forward_fields`` call,
    normalised into ``(nshots, shot_src, wav_per_shot, receivers)``.

    Source acquisition mirrors the FD ``PropTorch`` (``_normalize_io``):

    * **A** ``sources=(nshots, nd)`` -- naive multi-shot, 1 source/shot;
      ``wavelet=(nt,)`` shared [A1] or ``(nshots, nt)`` per-shot [A2].
    * **B** ``sources=(1, nsrc, nd)`` -- source encoding, 1 super-shot of
      ``nsrc`` superposed sources; ``wavelet=(nt,)`` shared or ``(nsrc, nt)``
      per-source, threaded down 2-D and injected as
      ``sum_k stamp_k(x) * wavelet[k, n]`` (single super-shot wavefield).

    ``wav_per_shot`` for A1 is ``[wavelet] * nshots`` -- the SAME object per
    shot, not a comprehension: ``_forward_c`` reads that object identity to
    choose the shared ``(nt,)`` C entry over the stacked ``(B, nt)`` one.
    """

    # No back-reference to the propagator and no spare copies of the inputs:
    # everything a caller reads is below, and dead state on a context object
    # is how two readers start disagreeing about which copy is current.
    __slots__ = ("sv", "receivers", "nshots", "shot_src", "wav_per_shot")

    def __init__(self, prop, wavelet, sources, receivers, source_vector):
        nd = prop.ndim
        # elastic body-force direction (default unit vertical force, z = last).
        # FIRST, before any coercion: a bad source_vector override must keep
        # raising ahead of a bad sources/receivers shape, as it does today.
        self.sv = prop._source_vector(source_vector)

        sources = torch.as_tensor(sources)
        receivers = torch.as_tensor(receivers)
        if not torch.is_tensor(wavelet):
            wavelet = torch.as_tensor(wavelet)
        # sources/receivers stay where they are -- _coords reads them host-side
        wavelet = wavelet.to(device=prop.dev, dtype=prop.dtype)
        if prop._src_moment is not None:
            # A stress source_type means what it means on PropTorch, which adds
            # the wavelet to the stress every step: the wavelet is the moment
            # RATE. The moment-tensor load is driven by the moment, its time
            # integral -- without it the record is one time derivative off
            # FD's. Done here, once, so every backend sees the same series.
            wavelet = torch.cumsum(wavelet, dim=-1) * prop.dt

        if sources.dim() == 2 and sources.shape[1] == nd:              # mode A
            nshots = sources.shape[0]
            shot_src = [[prop._coords(sources[b])] for b in range(nshots)]
            if wavelet.dim() == 1:
                wav_per_shot = [wavelet] * nshots
            elif wavelet.dim() == 2 and wavelet.shape[0] == nshots:
                wav_per_shot = [wavelet[b] for b in range(nshots)]
            else:
                raise ValueError(
                    f"multi-shot wavelet must be (nt,) or (nshots={nshots}, nt); "
                    f"got {tuple(wavelet.shape)}")
        elif sources.dim() == 3 and sources.shape[0] == 1 and sources.shape[2] == nd:
            nshots, nsrc = 1, sources.shape[1]                         # mode B
            shot_src = [[prop._coords(sources[0, k]) for k in range(nsrc)]]
            if wavelet.dim() == 1 or (wavelet.dim() == 2 and wavelet.shape[0] == nsrc):
                wav_per_shot = [wavelet]      # 1-D shared; (nsrc, nt) per-source
            else:
                raise ValueError(
                    f"source-encoding wavelet must be (nt,) or (nsrc={nsrc}, nt); "
                    f"got {tuple(wavelet.shape)}")
        else:
            raise ValueError(
                f"sources must be (nshots, {nd}) [multi-shot] or (1, nsrc, {nd}) "
                f"[source encoding]; got {tuple(sources.shape)}")
        self.nshots, self.shot_src, self.wav_per_shot = (
            nshots, shot_src, wav_per_shot)

        if receivers.dim() != 3 or receivers.shape[-1] != nd:
            raise ValueError(
                f"receivers must be (B, nrec, {nd}); got {tuple(receivers.shape)}")
        if receivers.shape[0] == 1 and nshots > 1:
            # a VIEW: every shot then shares one receiver row, which is what
            # makes _forward_c's row dedupe (and so its shot batch) fire
            receivers = receivers.expand(nshots, -1, -1)
        elif receivers.shape[0] != nshots:
            raise ValueError(
                f"receivers batch {receivers.shape[0]} must be 1 or match "
                f"nshots {nshots}.")
        self.receivers = receivers

    def check_pershot(self, pershot_eqs):
        """Per-shot models must carry one equation per shot."""
        if pershot_eqs is not None and len(pershot_eqs) != self.nshots:
            raise ValueError(
                f"per-shot models carry B={len(pershot_eqs)} but the sources "
                f"define nshots={self.nshots}; they must match (or drop the "
                f"model batch axis to share one model).")



class PropSEM:
    """SEM propagator with the FD ``PropTorch`` constructor and call.

    The options PropTorch has mean what they mean there: ``shape`` is the
    model and ``abcn`` (grid cells, whole elements: a multiple of
    ``n_order``; default ten elements) the absorbing layer added outside it;
    ``free_surface`` takes PropTorch's edge spec (default ``False``) and
    ``topography`` its per-column surface rows; ``memory`` is ``Full()``,
    ``BoundarySaving(storage_dtype=...)`` or ``Ckpt(chunks=...)``; ``impl``
    ``None`` is ``'auto'``; ``nt`` may be left to the wavelet; ``device`` or
    ``dev``. The rest is the SEM's own: the equation class and ``n_order``,
    ``f_dominant`` (the C-PML's frequency), ``density`` (the fixed density of
    :class:`AcousticPotentialSEM2D`), ``bs_band_mode`` and
    ``bs_store_dtype_elem`` (what boundary saving keeps of the C-PML band,
    and at what precision its element histories on ``impl='c'``).
    """

    def __init__(
        self,
        equation=AcousticSEM2D,
        *,
        shape,
        dh,
        dt,
        nt=None,
        n_order: int = 4,
        abcn=None,
        free_surface=False,
        topography=None,
        memory=None,
        f_dominant: float = 10.0,
        impl=None,
        device=None,
        dev=None,
        dtype: torch.dtype = torch.float32,
        density: float = 1000.0,
        bs_band_mode: str = "full",
        bs_store_dtype_elem=None,
        source_type=None,
        receiver_type=None,
        **kwargs,
    ):
        if kwargs:
            det = "; ".join(k + (f" (now {_CTOR_HINTS[k]})" if k in _CTOR_HINTS else "")
                            for k in sorted(kwargs))
            raise TypeError(f"PropSEM got unknown option(s): {det}")
        if bs_band_mode not in ("full", "rim"):
            raise ValueError(
                f"bs_band_mode must be 'full' or 'rim', got {bs_band_mode!r}")
        # What boundary saving stores of the C-PML region: the whole band
        # ("full", exact everywhere) or only its interface line ("rim", a
        # perimeter instead of an area -- 7-12x less history). "rim" is NOT a
        # drop-in: measured exact from ~8 model pixels in, the error sitting in
        # the model cells next to the band (the band itself is no longer in
        # the gradient: _interp_to_gll). See
        # sweep.sem.boundary_saving.cpml_node_masks.
        self.bs_band_mode = bs_band_mode
        shape = tuple(int(s) for s in shape)
        self.n_order = int(n_order)
        self.abcn = _resolve_abcn(abcn, self.n_order)
        # the C-PML builders and the DD strips count the layer in elements
        self.abcn_elements = self.abcn // self.n_order
        self._resolve_equation(equation, shape, self.abcn_elements,
                               source_type, receiver_type)

        if any((n - 1) % self.n_order for n in shape):
            raise ValueError(
                f"every (n-1) in shape {shape} must be divisible by "
                f"n_order = {self.n_order} to tile the grid exactly.")

        self.equation = equation
        self.shape = shape                        # the model grid
        self.dh = float(dh)
        self.dt = float(dt)
        # the wavelet's length, set by every call; a given nt is checked
        self._nt_given = None if nt is None else int(nt)
        self.nt = self._nt_given
        self.topography = _resolve_topography(topography, shape)
        # resolved to a tuple of side names; an empty tuple is falsey, so the
        # places that ask "is there a free surface at all" keep working
        self.free_surface = _resolve_free_surface(
            free_surface, self.ndim, self.topography is not None)
        self.f_dominant = float(f_dominant)
        self.dev = _resolve_device(device, dev)
        self.dtype = dtype

        self._select_backend(equation, impl, self.dev, density)
        self._resolve_memory(memory, bs_store_dtype_elem)
        if self._use_c:
            # from the ORDER, not from the mesh: PropSEM.mesh is lazy on
            # purpose (the DD ranks must never materialise the global one --
            # test_sem_dd_consts_contract pins that), and the bound is on
            # ngll**ndim, which needs neither
            from .prop_c import check_block_size
            check_block_size(self.n_order, self.ndim)

        self._construct_mesh(equation, shape, self.n_order, self.topography,
                             dtype, self.dev)

    def _resolve_memory(self, memory, bs_store_dtype_elem):
        """``memory`` as PropTorch takes it -> the three run knobs the paths
        read: ``_memory`` (``None`` or ``'boundary'``), ``_bs_store`` (the
        ring's storage dtype; ``None`` keeps the compute precision) and
        ``_ckpt_chunk`` (steps per checkpointed chunk). What PropSEM cannot
        do is refused, never dropped."""
        from sweep.propagator.options import (BoundarySaving, Ckpt, Full,
                                              as_memory_strategy)
        m = as_memory_strategy(memory)
        self.memory = Full() if m is None else m
        self._memory = self._bs_store = self._ckpt_chunk = None

        def unsupported(obj, keep):
            return [f.name for f in dataclasses.fields(obj)
                    if f.name not in keep and getattr(obj, f.name) != f.default]

        if isinstance(m, BoundarySaving):
            bad = unsupported(m, ("storage_dtype",))
            if bad:
                raise NotImplementedError(
                    f"PropSEM's boundary saving keeps its ring on the device "
                    f"and takes storage_dtype only; BoundarySaving("
                    f"{', '.join(b + '=...' for b in bad)}) is not supported.")
            self._memory = "boundary"
            # 'fp32', the default, is "no compression": the compute dtype
            self._bs_store = None if m.storage_dtype == "fp32" else m.storage_dtype
        elif isinstance(m, Ckpt):
            bad = unsupported(m, ("chunks",))
            if bad:
                raise NotImplementedError(
                    f"PropSEM checkpoints in chunks of steps on the device and "
                    f"takes chunks only; Ckpt({', '.join(b + '=...' for b in bad)}) "
                    f"is not supported.")
            if self._use_c:
                raise NotImplementedError(
                    "Ckpt chunks the eager autograd tape; impl='c' keeps the "
                    "whole history or saves the boundary (BoundarySaving()), "
                    "and has no tape to chunk.")
            self._ckpt_chunk = int(m.chunks)
        if self.bs_band_mode != "full" and self._memory != "boundary":
            raise ValueError(
                f"bs_band_mode={self.bs_band_mode!r} sets what boundary saving "
                f"keeps of the C-PML band; it needs memory=BoundarySaving().")
        if bs_store_dtype_elem is not None:
            if self._memory != "boundary" or not self._use_c:
                raise ValueError(
                    "bs_store_dtype_elem is the precision of the element "
                    "histories impl='c' boundary saving keeps; it needs "
                    "impl='c' and memory=BoundarySaving().")
        self._bs_store_elem = bs_store_dtype_elem

    def _resolve_equation(self, equation, shape, abcn_elements, source_type, receiver_type):
        """Derive and validate everything PropSEM needs from the equation:
        mesh family vs shape, the model list, what build()'s signature can
        carry, the calling convention, and the field-role selections.
        Raises at construction for anything the first forward would
        otherwise turn into a confusing error."""
        self.ndim = len(shape)
        # Support is a PROPERTY of the equation, not membership of a list here.
        # There used to be five hand-kept tuples (_ACOUSTIC_2D, _ACOUSTIC,
        # _ELASTIC_2D, _ELASTIC_3D, _DEFORMED_3D) answering overlapping
        # questions, and an equation missing from one of them failed in a
        # different way than missing from another. Two derived facts replace
        # them: which mesh family build() takes, and which models it wants.
        kind = mesh_kind(equation)
        if MESH_NDIM[kind] != self.ndim:
            raise ValueError(
                f"{equation.__name__} builds on a {kind} mesh "
                f"({MESH_NDIM[kind]}-D); shape {shape} is {self.ndim}-D.")
        # Positional models of build(), MINUS the ones PropSEM supplies
        # itself: the acoustic potential form takes (vp, rho) but ``density=``
        # exists precisely so callers pass models=[vp] and rho is a constant
        # the adapter fills in. Deriving the raw signature here would be more
        # faithful to build() and would break that contract.
        self._supplied = {"rho"} if equation is AcousticPotentialSEM2D else set()
        self.model_names = [n for n in model_names(equation)
                            if n not in self._supplied]
        # PropSEM drives build() one generic way: positional models, plus
        # `dirichlet` for the acoustic forms and a single `cpml`. An equation
        # whose signature needs something else (the fluid-solid form wants
        # cpml_ac AND cpml_el) cannot be driven through it. Detect that from
        # the signature HERE, so it is a clear refusal at construction rather
        # than a TypeError from inside the first forward.
        import inspect as _inspect
        _kw = {q.name for q in
               _inspect.signature(equation.build).parameters.values()
               if q.kind is q.KEYWORD_ONLY}
        _split = set(getattr(equation, "cpml_builders", None) or ())
        _unsupported = (_kw - {"cpml", "dirichlet", "domain", "free_top"}
                        - _split)
        if _unsupported:
            raise NotImplementedError(
                f"{equation.__name__}.build takes keyword argument(s) "
                f"{sorted(_unsupported)} that PropSEM does not supply. Use "
                f"SEMPropagator directly for this equation.")
        if abcn_elements and "cpml" not in _kw and not _split:
            raise NotImplementedError(
                f"{equation.__name__}.build takes no `cpml` argument, so it "
                f"cannot carry PropSEM's absorbing layer. Pass "
                f"abcn=0 for a bare reflecting box, or use "
                f"SEMPropagator directly.")

        # "acoustic" here means the FD-facing calling convention: ONE source
        # field, so a scalar source and a single receiver component. Read off
        # field_specs rather than a name-based tuple -- AcousticPotentialSEM2D
        # takes (vp, rho) and is still acoustic in this sense.
        self.is_acoustic = sum(
            f.supports_source for f in equation.field_specs) == 1

        # Which fields the sources drive and the receivers sample, by name --
        # the FD ``source_type`` / ``receiver_type`` convention.
        _stamp = getattr(equation, "propsem_source", None)
        n_src = sum(f.supports_source for f in equation.field_specs)
        # the vector equations (a force with one component per axis) also
        # take the FD names; see _resolve_vector_source
        self._src_force_axes, self._src_moment = None, None
        if not self.is_acoustic and _stamp is None and n_src == self.ndim:
            (self.source_type, self._src_force_axes,
             self._src_moment) = _resolve_vector_source(
                 equation, source_type, self.ndim)
        else:
            self.source_type = resolve_field_types(equation, source_type,
                                                   "source")
        if (_stamp is not None and source_type is not None
                and self.source_type != list(equation.default_source_type)):
            raise NotImplementedError(
                f"{equation.__name__} injects its PropSEM source as "
                f"{_stamp[0]!r} into {equation.default_source_type}; other "
                f"source_type selections would be positionally mis-mapped by "
                f"the stamp layout. Use SEMPropagator with explicit "
                f"source_kinds/source_payloads.")
        self.receiver_type = resolve_field_types(equation, receiver_type,
                                                 "receiver")
        # Derived receiver components (p = -chi_ddot, vx = u_x_dot, ...):
        # the LOW-LEVEL propagators record the BASE fields; the requested
        # components are produced from those traces on the way out (see
        # _apply_derived_receivers). _recv_base is the deduped base-field
        # list handed down; _recv_plan maps each requested component to its
        # base column and transform.
        derived = getattr(equation, "derived_receiver_types", {})
        self._recv_base, self._recv_plan = [], []
        for nm in self.receiver_type:
            base, op = derived.get(nm, (nm, None))
            if base not in self._recv_base:
                self._recv_base.append(base)
            self._recv_plan.append((self._recv_base.index(base), op))
        self._recv_identity = (
            self._recv_base == self.receiver_type
            and all(op is None for _, op in self._recv_plan))
        # what return_wavefield=True copies out: the fields the receivers
        # record from (the BASE fields behind a derived p / vx), in that order
        self.snapshot_fields = list(self._recv_base)
        self.snapshot_times_last = ()

    def _select_backend(self, equation, impl, dev, density):
        """impl=None / 'auto' / 'eager' / 'c', as on PropTorch: the CUDA
        backend serves exactly the equations its dispatch implements, read
        from prop_c.C_SUPPORTED, and 'auto' takes it wherever it can run.
        ``self.impl`` is the one chosen."""
        # impl='c': route through the CUDA SEM backend. It implements the 2-D
        # AcousticPotentialSEM2D (variable-density potential form); vp is the FWI
        # model and ``density`` is the fixed density. impl='eager' and impl='c'
        # use the SAME equation so their forward + gradient match to fp32.
        impl = "auto" if impl is None else impl
        if impl not in ("auto", "eager", "c"):
            raise ValueError(
                f"impl must be None, 'auto', 'eager' or 'c'; got {impl!r}")
        self._use_c = False
        self._c_density = float(density)
        if impl in ("c", "auto"):
            # Every equation the CUDA dispatch in prop_c handles. It used to be
            # only the acoustic three, so five equations HAD a CUDA kernel that
            # the solver could not reach -- callers had to drop to
            # SEMCudaPropagator and assemble the call themselves, and the same
            # PropSEM call meant different things on the two backends.
            from .prop_c import C_SUPPORTED, sem_c_available
            if equation in C_SUPPORTED:
                if sem_c_available() and str(dev).startswith("cuda"):
                    self._use_c = True
                elif impl == "c":
                    raise RuntimeError(
                        "impl='c' needs CUDA (dev='cuda') and a sweep core "
                        "built with the SEM entry points.")
            elif impl == "c":
                raise NotImplementedError(
                    f"impl='c' has no CUDA kernel for {equation.__name__}; "
                    f"available: {sorted(c.__name__ for c in C_SUPPORTED)}. "
                    f"Use impl='eager'.")
        self.impl = "c" if self._use_c else "eager"

    def _construct_mesh(self, equation, shape, n_order, topography, dtype, dev):
        """Per-side padding (a free surface is never padded) and the mesh,
        flat or topography-fitted, from the equation's mesh family."""
        # As on the FD ``PropTorch``: ``shape`` is the model and the
        # ``abcn``-cell absorbing layer is added OUTSIDE it on every absorbing
        # side. Sources/receivers are model coordinates, shifted into the
        # padded grid internally; the velocity is edge-extended into the pad.
        #
        # Axis order in ``shape`` is (z, [y,] x).
        pad = self.abcn_elements
        elems_int = [(n - 1) // n_order for n in shape]          # per shape-axis
        # A side carrying the free surface is NOT padded -- there is no absorbing
        # layer there -- so the pad is resolved per side, not per axis.
        low_pad = [pad] * self.ndim
        high_pad = [pad] * self.ndim
        for nm, ax, hi in _SIDES[self.ndim]:
            if nm in self.free_surface:
                (high_pad if hi else low_pad)[ax] = 0
        elems_pad = [e + lo + hi
                     for e, lo, hi in zip(elems_int, low_pad, high_pad)]
        self._pad0 = [lo * n_order for lo in low_pad]            # interior origin
        self._elems_pad = elems_pad
        maxes = [ep * n_order * self.dh for ep in elems_pad]     # physical extents

        # Which mesh family, and flat or topography-fitted: both come from the
        # equation's own build() signature (fields.mesh_kind) and the presence
        # of ``topography``. This replaced two parallel if-chains -- one per
        # dimension -- that tested equation identity against hand-kept tuples.
        kind = mesh_kind(equation)
        flat_builder, topo_builder = MESH_BUILDERS[kind]
        if topography is not None and topo_builder is None:
            raise ValueError(
                f"topography needs a curved-element mesh; "
                f"{equation.__name__} builds on a {kind} mesh. Use the "
                f"'Deformed' variant of this equation.")
        # Both dimensions build the same call; 3-D only has a y axis to add.
        # ``shape`` and ``elems_pad`` are (z, [y,] x), so index 0 is z and -1
        # is x in either.
        common = dict(nez=elems_pad[0], nex=elems_pad[-1], x_max=maxes[-1],
                      n_order=n_order, dtype=dtype, device=dev)
        if self.ndim == 3:
            common.update(ney=elems_pad[1], y_max=maxes[1])
        if topography is None:
            build = (flat_builder, dict(z_max=maxes[0], **common))
        else:
            # the model's first column (and row of columns, in 3-D) sits at
            # the lateral pads: the surface is extended flat across them
            z_top = _SurfaceRows(topography, self.dh, self._pad0[1:])
            build = (topo_builder,
                     dict(z_bottom=maxes[0], z_top_fn=z_top, **common))
        # The mesh is built on first access (the ``mesh`` property below), not
        # here: its per-element-node arrays are 50.7 B/node, and a domain-
        # decomposition rank never needs them -- it reads ``hex8_layout()``
        # and builds only its own strip. Same object, same numbers, later.
        # A topography mesh is built right away regardless, so an inverted
        # element is still refused at construction and not at the first
        # forward.
        self._mesh_kind = kind
        self._mesh_build = build
        self._mesh = None
        self._layout = None
        # interpolation tables are built on first use (see _interp_to_gll);
        # the GLL shape is free and some callers read it directly
        self._g_i0 = None
        n_el = 1
        for e in elems_pad:
            n_el *= e
        self._gll_shape = (n_el,) + (n_order + 1,) * self.ndim
        if topography is not None:
            self.mesh

    @property
    def mesh(self):
        """The full mesh, built on first access and then shared -- the same
        object on every read, so ``id(self.mesh)``-keyed caches behave as
        they did when the constructor built it."""
        m = self._mesh
        if m is None:
            builder, kw = self._mesh_build
            m = self._mesh = builder(**kw)
        return m

    @mesh.setter
    def mesh(self, m):
        self._mesh = m
        self._layout = None          # the layout follows the mesh (below)

    def hex8_layout(self):
        """The mesh's O(nx + ny + nz) description -- axes and scalars, no
        arrays -- for a consumer that must not materialise the mesh (the DD
        strips). Only the flat hex8 family has one; every other family keeps
        the real mesh.

        Follows the MESH, not the constructor: while no mesh exists it is
        derived from the stored build arguments (so a DD rank never builds
        the mesh), and once one exists -- built lazily, or assigned by the
        caller -- it is that mesh's own layout, so the compact DD path and
        the fused path can never describe two different geometries."""
        if self._mesh_kind != "hex8":
            raise TypeError(
                f"hex8_layout() is for the flat hex8 family; this propagator "
                f"builds a {self._mesh_kind} mesh")
        lay = self._layout
        if lay is None:
            m = self._mesh
            if m is None:
                _, kw = self._mesh_build
                lay = hex8_layout(**kw)
            else:
                lay = getattr(m, "layout", None)
                if lay is None:
                    raise TypeError(
                        "prop.mesh was replaced by a mesh that carries no "
                        "Hex8Layout, so the compact DD path cannot describe "
                        "it; build the replacement with build_hex8_mesh")
            self._layout = lay
        return lay

    # ------------------------------------------------------------------ #
    # Differentiable grid -> GLL multilinear interpolation (2-D bilinear /
    # 3-D trilinear, dispatched generically over ``ndim``).
    # ------------------------------------------------------------------ #
    def _build_grid_to_gll(self):
        # Per shape-axis GLL physical coords (order z, [y,] x), mapped to INTERIOR
        # grid coords: subtract the interior origin, clamp to [0, n-1] so pad nodes
        # read the nearest model boundary (velocity edge-extended into the PML).
        coords_e = ([self.mesh.z_e, self.mesh.x_e] if self.ndim == 2
                    else [self.mesh.z_e, self.mesh.y_e, self.mesh.x_e])
        self._g_i0, self._g_f = [], []
        outside = None
        for ax, ce in enumerate(coords_e):
            n = self.shape[ax]
            raw = ce.reshape(-1) / self.dh - self._pad0[ax]
            off = _outside_model(raw, n)
            outside = off if outside is None else outside | off
            c = raw.clamp(0, n - 1)
            i0 = c.floor().long().clamp(0, n - 2)
            self._g_i0.append(i0)
            self._g_f.append((c - i0).to(self.dtype))
        # the pad's element-nodes, which _interp_to_gll cuts out of the
        # gradient; None when there is no pad (abcn=0, or no absorbing side)
        self._g_pad = outside if bool(outside.any()) else None
        self._gll_shape = self.mesh.x_e.shape

    def _interp_to_gll(self, grid: torch.Tensor) -> torch.Tensor:
        # Lazy: the tables are (E*ngll^dim) x 3 axes of int64 + float32 --
        # measured 70 B/node, more than the mesh itself -- and the DD path
        # interpolates per strip and never calls this, so a rank that only
        # runs DD never pays for them.
        if getattr(self, "_g_i0", None) is None:
            self._build_grid_to_gll()
        ng = self.mesh.ngll
        gll_native = (self.mesh.n_elements,) + (ng,) * self.ndim
        if tuple(grid.shape) == gll_native:
            return grid                              # already GLL-native
        if tuple(grid.shape) != self.shape:
            raise ValueError(
                f"model shape {tuple(grid.shape)} is neither the grid "
                f"{self.shape} nor the GLL shape {gll_native}.")
        out = None
        for corner in itertools.product((0, 1), repeat=self.ndim):
            w, idx = 1.0, []
            for ax, c in enumerate(corner):
                f = self._g_f[ax]
                w = w * ((1.0 - f) if c == 0 else f)
                idx.append(self._g_i0[ax] + c)
            term = w * grid[tuple(idx)]
            out = term if out is None else out + term
        if self._g_pad is not None and out.requires_grad:
            # The pad reads the clamped model edge, so through the interpolant
            # the whole C-PML band's sensitivity would land on the outermost
            # ring of cells -- measured 24% of the gradient's L2 there, against
            # 4.2% for FD, whose EdgePadding backward drops the pad instead.
            # Same here: the pad's values are a constant of the model. The
            # forward values are untouched.
            out = torch.where(self._g_pad, out.detach(), out)
        return out.reshape(self._gll_shape)

    def _absorbing_sides(self):
        return tuple(nm for nm, _, _ in _SIDES[self.ndim]
                     if nm not in self.free_surface)

    def _resolve_model_batch(self, models):
        """Split ``models`` into equation(s): shared or per-shot.

        A model entry with a leading shot axis -- ``(B, *shape)`` instead of
        ``shape`` -- gives every shot its own material model; entries without
        one are broadcast, so ``models=[vp_B, rho]`` mixes freely. Returns
        ``(eq, pershot_eqs)`` where ``pershot_eqs`` is ``None`` in the shared
        case (then ``eq`` is THE equation) or the per-shot list (then ``eq``
        is its first element, which carries the shared geometry).

        Two batch-wide conventions, both needed so impl='c' (one batched
        kernel sequence) and eager (a per-shot loop) return the same numbers:

        * ONE C-PML profile for the whole batch, built with the batch-max vp
          (the profile is per-mesh in the batched kernels, so eager must not
          give each shot its own damping strength);
        * gradients of a BROADCAST model are summed over shots -- the same
          semantics autograd gives a tensor reused across a loop.

        Also stashes ``self._c_materials`` for the impl='c' batched fast path
        (stacked ``(B, ...)`` material arrays, Bm == B) -- ``None`` whenever
        the equation cannot take them batched.
        """
        if torch.is_tensor(models):
            models = [models]
        nd1 = self.ndim + 1
        ps = lambda m: torch.is_tensor(m) and m.dim() == nd1 \
            and tuple(m.shape[1:]) == self.shape
        self._c_materials = None
        if models is None or not any(ps(m) for m in models):
            return self._build_eq(models), None
        sizes = {m.shape[0] for m in models if ps(m)}
        if len(sizes) != 1:
            raise ValueError(
                f"per-shot models disagree on the batch size: {sorted(sizes)}")
        B = sizes.pop()
        c_max = float(torch.as_tensor(models[0]).max().detach())   # batch-max vp
        eqs = [self._build_eq([m[b] if ps(m) else m for m in models],
                              c_max=c_max) for b in range(B)]
        if self._use_c:
            from .prop_c import sem_c_model_batchable
            if sem_c_model_batchable(eqs[0]):
                self._c_materials = tuple(
                    torch.stack(ts) for ts in zip(
                        *[(e._inv_rho, e._inv_M, e.kappa_node) for e in eqs]))
        return eqs[0], eqs

    def _build_eq(self, models, c_max=None):
        eq = self._build_eq_raw(models, c_max=c_max)
        eq.bs_band_mode = self.bs_band_mode
        return eq

    def _cpml_cached(self, tag, make, *key):
        """Memoise one C-PML profile on an EXACT key.

        The profile was rebuilt on every forward call, and it is the single
        largest cost in ``_resolve_model_batch``: measured 7.16 ms of 11.46 at
        275k nodes and 41.97 of 54.50 at 2.1M (62-77%), while the whole of
        ``_resolve_model_batch`` is roughly half of a DD forward call at
        nt=60. It is also why domain decomposition did not scale -- the rebuild
        is O(n_global) and every rank repeats it, so it is a serial fraction
        that more GPUs cannot shrink.

        The profile depends only on ``(mesh, dt, sides, thickness, f_dominant,
        c_max)``. The key is those values EXACTLY -- no tolerance on ``c_max``
        -- so a changed model still rebuilds and the result is bit-identical to
        building every time. What this buys is the shot loop: one FWI iteration
        calls forward once per shot with the same model, so the profile is
        built once and reused for the rest.
        """
        k = (tag, id(self.mesh), float(self.dt), tuple(key))
        if getattr(self, "_cpml_key", None) == k:
            return self._cpml_val
        val = make()
        self._cpml_key, self._cpml_val = k, val
        return val

    def _build_eq_raw(self, models, c_max=None):
        if torch.is_tensor(models):                  # accept a single vp tensor
            models = [models]
        grids = [self._interp_to_gll(m.to(device=self.dev, dtype=self.dtype))
                 for m in models]
        if c_max is None:
            c_max = float(grids[0].max().detach())   # vp drives the PML profile
        sides = self._absorbing_sides()
        if self.is_acoustic:
            dirichlet = self.free_surface
            if self.abcn_elements == 0:      # bare (a 0-thick CPML divides by 0)
                cpml = None
            else:
                _bld = build_cpml if self.ndim == 2 else build_cpml3d
                cpml = self._cpml_cached(
                    "acoustic",
                    lambda: _bld(self.mesh, dt=self.dt, sides=sides,
                                 pml_thickness_elements=self.abcn_elements,
                                 c_max=c_max, f_dominant=self.f_dominant),
                    sides, self.abcn_elements, c_max, self.f_dominant)
            if self.equation is AcousticPotentialSEM2D:   # vp + constant density
                rho = torch.full_like(grids[0], self._c_density)
                return AcousticPotentialSEM2D.build(
                    self.mesh, grids[0], rho, dirichlet=dirichlet, cpml=cpml)
            return self.equation.build(self.mesh, grids[0],
                                       dirichlet=dirichlet, cpml=cpml)
        # elastic: vp/vs/rho and a NATURAL free surface -- traction-free is what
        # the weak form does on a side with no absorbing layer, so there is no
        # dirichlet set to build. ``free_surface`` therefore acts only by
        # removing that side from ``sides``.
        # One elastic route for 2-D and 3-D alike. Equations that never take
        # a C-PML declare no cpml_builder and no ``cpml`` keyword; the
        # construction-time signature check already refused abcn > 0
        # for them, so the builder branches are only reached where they exist.
        import inspect as _inspect
        kw = {}
        if "dirichlet" in _inspect.signature(self.equation.build).parameters:
            # the fluid-solid form carries a pressure-release surface on its
            # FLUID side; route the same free_surface the acoustic branch uses
            kw["dirichlet"] = self.free_surface
        if self.abcn_elements:
            split = getattr(self.equation, "cpml_builders", None)
            if split:                        # e.g. fluid-solid: cpml_ac + cpml_el
                built = self._cpml_cached(
                    "split",
                    lambda: {n: b(self.mesh, dt=self.dt, sides=sides,
                                  pml_thickness_elements=self.abcn_elements,
                                  c_max=c_max, f_dominant=self.f_dominant)
                             for n, b in split.items()},
                    sides, self.abcn_elements, c_max, self.f_dominant,
                    tuple(split))
                kw.update(built)
            else:
                kw["cpml"] = self._cpml_cached(
                    "single",
                    lambda: self.equation.cpml_builder(
                        self.mesh, dt=self.dt, sides=sides,
                        pml_thickness_elements=self.abcn_elements,
                        c_max=c_max, f_dominant=self.f_dominant),
                    sides, self.abcn_elements, c_max, self.f_dominant)
        return self.equation.build(self.mesh, *grids, **kw)

    # receiver field indices (1 for acoustic, 3 components for elastic)
    def _recv_idxs(self, eq):
        """Field indices the seismogram components come from -- the BASE
        fields backing ``receiver_type`` (derived components like ``p`` /
        ``vx`` are produced from these traces on the way out)."""
        by_name = {s.name: i for i, s in enumerate(eq.field_specs)}
        return [by_name[n] for n in self._recv_base]

    # ------------------------------------------------------------------ #
    # Derived receiver components: matched-stencil time differences of the
    # recorded base traces. Pointwise-in-space relations (p = -chi_ddot,
    # v = u_dot) commute with the linear receiver gather, so differentiating
    # the TRACE equals recording the derived quantity -- and the central
    # stencils below are exactly the leapfrog's own discrete derivatives.
    # Fields are quiescent before t=0, so the left pad is a true zero; the
    # final sample uses a one-sided difference.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trace_ddt(x, dt):
        """du/dt along dim 1 of ``(nshots, nt, nrec, ...)`` traces."""
        z = torch.zeros_like(x[:, :1])
        pad = torch.cat([z, x, x[:, -1:]], 1)          # u(-dt)=0, u(nt)~u(nt-1)
        return (pad[:, 2:] - pad[:, :-2]) / (2.0 * dt)

    @staticmethod
    def _trace_neg_d2dt2(x, dt):
        """-d2u/dt2 along dim 1 (the pressure from a chi trace)."""
        z = torch.zeros_like(x[:, :1])
        pad = torch.cat([z, x, x[:, -1:]], 1)
        return -(pad[:, 2:] - 2.0 * x + pad[:, :-2]) / (dt * dt)

    def _apply_derived_receivers(self, out):
        """(nshots, nt, nrec, n_base) -> (nshots, nt, nrec, n_requested)."""
        if self._recv_identity:
            return out                    # bit-untouched default path
        comps = []
        for col, op in self._recv_plan:
            x = out[..., col]
            if op == "velocity":
                x = self._trace_ddt(x, self.dt)
            elif op == "pressure":
                x = self._trace_neg_d2dt2(x, self.dt)
            elif op is not None:
                raise AssertionError(f"unknown derived receiver op {op!r}")
            comps.append(x)
        return torch.stack(comps, -1)

    def _default_source(self):
        """Is the source the equation's default -- the unit vertical force on
        the elastics, the one source field elsewhere? (What a bare node id
        means to the DD driver.)"""
        if self._src_moment is not None:
            return False
        if self._src_force_axes is None:      # scalar / equation-declared
            return True
        return self._src_force_axes == frozenset({self.ndim - 1})

    def _source_vector(self, override=None):
        """Body-force direction implied by ``source_type``, or an override.

        ``source_type=["u_z"]`` (or FD's ``["vz"]``) means "drive the z
        displacement", i.e. a vertical force. Naming several force components
        drives them together with equal weight -- a diagonal force; FD's
        explosion is the stress pair ``["sxx", "szz"]``, a moment tensor, for
        which there is no force vector (``None``). A per-call
        ``source_vector`` wins over a force ``source_type``, since it can
        express directions no field list can (e.g. 30 deg).
        """
        nd = self.ndim
        if self._src_moment is not None:
            if override is not None:
                raise ValueError(
                    f"source_vector sets a body-force direction, but "
                    f"source_type={self.source_type} is a stress (moment-"
                    f"tensor) source; give a force source_type to use it.")
            return None
        if override is not None:
            return tuple(float(c) for c in override)
        if self._src_force_axes is not None:
            return tuple(1.0 if a in self._src_force_axes else 0.0
                         for a in range(nd))
        names = [s.name for s in self.equation.field_specs if s.supports_source]
        return tuple(1.0 if n in self.source_type else 0.0 for n in names[:nd])

    # ------------------------------------------------------------------ #
    # interior index -> physical coords (metres) in the padded grid.
    # 2-D index row [ix, iz] -> (x, z); 3-D [ix, iy, iz] -> (x, y, z).
    # ------------------------------------------------------------------ #
    def _coords(self, idx_row) -> tuple:
        # idx_row is in FD column order (x first, z last); ``self._pad0`` /
        # ``self.shape`` are in (z, [y,] x) order — reverse to align.
        rev_pad = self._pad0[::-1]                # (x, [y,] z)
        return tuple((float(idx_row[k]) + rev_pad[k]) * self.dh
                     for k in range(self.ndim))

    # ------------------------------------------------------------------ #
    # forward — PropTorch-compatible signature
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, wavelet, sources, receivers, models=None, **kw):
        """See :meth:`_forward_fields` for the full option reference. This
        wrapper adds the derived receiver components: requested ``p`` /
        ``vx`` / ``vy`` / ``vz`` are produced from the recorded base traces
        (p = -chi_ddot, v = u_dot) so callers get FD-aligned pressure /
        particle-velocity data without converting anything themselves."""
        out = self._forward_fields(wavelet, sources, receivers,
                                   models=models, **kw)
        if isinstance(out, tuple):                       # return_wavefield
            seis, snaps = out
            return self._apply_derived_receivers(seis), snaps
        return self._apply_derived_receivers(out)

    def _set_nt(self, wavelet):
        """This call's ``nt``: the wavelet's length, which a ``nt`` given to
        the constructor must match. A new length drops the strip driver,
        whose consts carry the old one."""
        shape = getattr(wavelet, "shape", None)
        n = int((shape if shape is not None else torch.as_tensor(wavelet).shape)[-1])
        if self._nt_given is not None and n != self._nt_given:
            raise ValueError(
                f"the wavelet has {n} samples but this propagator was built "
                f"with nt={self._nt_given}")
        if n != self.nt:
            self.nt = n
            self._dd1 = None

    def _forward_fields(self, wavelet, sources, receivers, models=None,
                        adj=False, return_wavefield=False,
                        compile_step=False, source_vector=None,
                        pml_guard="warn", snapshot_times=None,
                        snapshot_interval=None, **kwargs):
        """``return_wavefield=True`` (eager only) returns ``(seis, snapshots)``
        as ``PropTorch`` does, with the same ``snapshot_times`` /
        ``snapshot_interval`` rule (default: every step) and the same layout:
        ``snapshots`` is ``(nsnap, nfield, nshots, 1, *grid)``, detached and on
        the host, the grid being the model and its absorbing layer; ``nfield``
        runs over :attr:`snapshot_fields` (the fields the receivers record
        from) and the steps are :attr:`snapshot_times_last`. Each grid point
        takes the SEM field through its element's basis (zero above a
        topography). The shots then run one by one.

        The gradient memory is the constructor's ``memory``:
        ``BoundarySaving()`` runs the backward pass with boundary-saving
        wavefield reconstruction (peak memory ~flat in ``nt``), at the cost of
        one extra forward per step in backward; seismograms unchanged, model
        gradient exact in the lossless interior. ``Full()`` keeps the full
        autograd tape; ``Ckpt(chunks=k)`` checkpoints it in ``k``-step chunks.

        ``compile_step=True`` ``torch.compile``\\ s the per-step graph.

        ``models`` is ``[vp]`` for acoustic and ``[vp, vs, rho]`` for elastic.
        ``source_vector`` (elastic only) is the body-force direction; default is
        a unit vertical force (a stress ``source_type`` refuses it). The
        seismogram is ``(nshots, nt, nrec, ncomp)``
        for both, matching the FD ``PropTorch``: ``ncomp = 1`` for acoustic and
        the receiver-field components for elastic.

        Source acquisition mirrors the FD ``PropTorch`` (three modes, picked by
        ``sources``/``wavelet`` shape):

        * **A1** ``wavelet=(nt,)``, ``sources=(nshots, nd)`` — multi-shot, shared
          wavelet (vmap-batched over shots).
        * **A2** ``wavelet=(nshots, nt)``, ``sources=(nshots, nd)`` — multi-shot,
          per-shot wavelet.
        * **B**  ``wavelet=(nt,)`` or ``(nsrc, nt)``, ``sources=(1, nsrc, nd)``,
          ``receivers=(1, nrec, nd)`` — source encoding: one super-shot of
          ``nsrc`` superposed point sources. ``(nsrc, nt)`` gives each source its
          own time function (polarity / amplitude / time-shift encoding); the
          gradient is the true cross-talk source-encoded gradient. Output is
          ``(1, nt, nrec, ncomp)``.

        Which path a call takes (the eager rows are ``_forward_eager``'s):

        ===========================  ==================================
        3-D acoustic C-PML on CUDA   ``_forward_c_dd3d`` (one strip,
                                     before any global equation)
        other equations on CUDA      ``_forward_c``: batched when
                                     ``_c_batchable``, else its loop
        eager, boundary saving       ``_forward_bs_single``, that
                                     looped, or ``_forward_bs_batched``
        eager, full tape             ``props[0].forward``, that looped,
                                     or ``_run_batched``
        ===========================  ==================================
        """
        _reject_unsupported(self, kwargs, compile_step=compile_step, adj=adj,
                            return_wavefield=return_wavefield, models=models,
                            snapshot_times=snapshot_times,
                            snapshot_interval=snapshot_interval)
        self._set_nt(wavelet)
        snap_steps = _resolve_snapshot_steps(self.nt, return_wavefield,
                                             snapshot_times, snapshot_interval)
        acq = _Acq(self, wavelet, sources, receivers, source_vector)

        # 3-D acoustic on CUDA: the single-strip DD driver, BEFORE any global
        # equation is built (that build is what it saves) -- see _forward_c_dd3d
        if self._use_c and self._dd3d_route(models):
            # NOTE bs_store_dtype_elem is not forwarded: this route rings
            # nodes only, and the element knob is refused there
            return self._forward_c_dd3d(acq.shot_src, acq.receivers,
                                        acq.wav_per_shot, models, pml_guard)

        eq, pershot_eqs = self._resolve_model_batch(models)
        acq.check_pershot(pershot_eqs)

        if self._use_c:                                  # CUDA SEM backend
            # NOTE pml_guard is not forwarded: the C route locates its
            # stamps through build_point_stamp, which has no placement
            # guard (the eager SEMPropagator.build does). Deliberate.
            return self._forward_c(eq, acq.shot_src, acq.receivers,
                                   acq.wav_per_shot, memory=self._memory,
                                   store_dtype=self._bs_store,
                                   store_dtype_elem=self._bs_store_elem,
                                   sv=acq.sv, pershot_eqs=pershot_eqs)

        # Eager: build the per-shot propagators, then run them. The build
        # locates every point stamp and runs the PML placement guard, so it
        # stays AHEAD of the runner choice -- a source in the absorbing band
        # must be reported before a checkpoint/nshots quarrel is.
        props = self._build_props(eq, pershot_eqs, acq, pml_guard)
        if snap_steps is not None:
            return self._forward_snapshots(props, acq, snap_steps,
                                           compile_step=compile_step)
        return self._forward_eager(eq, pershot_eqs, props, acq,
                                   memory=self._memory,
                                   compile_step=compile_step,
                                   checkpoint_chunk=self._ckpt_chunk,
                                   store_dtype=self._bs_store)

    # ------------------------------------------------------------------ #
    # The eager propagator factory: acquisition -> one SEMPropagator per shot
    # ------------------------------------------------------------------ #
    def _rec_row(self, receivers, b):
        """Shot ``b``'s receiver row in metres. The interior->padded shift
        lives in ``_coords``; this is the row loop that stood open-coded in
        the eager build, in ``_forward_c``'s stamp cache and in
        ``_forward_c_dd3d``."""
        return [self._coords(receivers[b, j])
                for j in range(receivers.shape[1])]


    def _eager_source_stamp(self, sv):
        """The ``(kind, payload)`` the EAGER build stamps at every source point,
        or ``None`` for the acoustic single-source-field case (where
        ``SEMPropagator.build``'s own default -- scalar, unit -- is what the
        caller wants and passing it explicitly would only be a second copy of it).

        NOTE this is NOT the same expression as ``_forward_c``'s: the eager build
        first honours an equation-declared ``propsem_source`` (fluid-solid: a
        scalar injected into chi), where impl='c' hard-codes scalar/force and
        relies on its positional zip to land the stamp. They agree for every
        equation that is both C-supported and stamp-carrying; unifying them is a
        behaviour change on the C side and is deliberately NOT done here.
        """
        if self.is_acoustic:
            return None
        stamp = getattr(self.equation, "propsem_source", None)
        if stamp is not None:
            kind, payload = stamp         # equation-declared injection (arity pinned)
            return kind, payload
        if self._src_moment is not None:
            return "moment_tensor", self._src_moment
        return "force", sv


    def _build_props(self, eq, pershot_eqs, acq, pml_guard):
        """One :class:`SEMPropagator` per shot, built from the acquisition.

        Rebuilt on EVERY call by design: ``build`` is where the point stamps are
        located and where ``pml_guard`` runs, and both have to see this call's
        coordinates and this call's equation (a cache keyed on less than that is
        how a reused propagator would answer with the previous model's mesh).
        """
        stamp = self._eager_source_stamp(acq.sv)
        props = []
        for b in range(acq.nshots):
            nsrc = len(acq.shot_src[b])
            extra = ({} if stamp is None else
                     dict(source_kinds=[stamp[0]] * nsrc,
                          source_payloads=[stamp[1]] * nsrc))
            props.append(SEMPropagator.build(
                pershot_eqs[b] if pershot_eqs is not None else eq,
                dt=self.dt, nt=self.nt,
                source_xz=acq.shot_src[b],
                receiver_xz=self._rec_row(acq.receivers, b),
                receiver_type=self._recv_base,
                pml_guard=pml_guard, **extra))
        return props

    def _forward_snapshots(self, props, acq, steps, *, compile_step=False):
        """``return_wavefield=True``: each shot's full-tape forward with its
        snapshots -> ``(seis, snaps)``, seis as ``_forward_eager`` returns it
        and snaps ``(nsnap, nfield, nshots, 1, *grid)`` on the host.

        One shot at a time, the per-shot-model loop's form: the vmapped batch
        keeps no per-step state to copy out, and snapshots are for looking at,
        not for speed. The seismogram is the same number either way (the loop
        and the vmap are bit-identical; see _run_batched).
        """
        seis, snaps = [], []
        for b in range(acq.nshots):
            s, w = props[b].forward(acq.wav_per_shot[b],
                                    compile_step=bool(compile_step),
                                    snapshot_times=steps)
            seis.append(s)
            w = w.detach().cpu()                # (nsnap, nfield, n_global)
            snaps.append(self._on_grid(w.reshape(-1, w.shape[-1]))
                         .reshape(w.shape[:2] + self._grid_shape()))
        self.snapshot_times_last = tuple(steps)
        return torch.stack(seis, 0), torch.stack(snaps, 2).unsqueeze(3)

    # ------------------------------------------------------------------ #
    # GLL -> grid for return_wavefield: every grid point of the model and its
    # absorbing layer takes the SEM field through its element's basis
    # ------------------------------------------------------------------ #
    def _grid_shape(self):
        """The snapshot grid, (z, [y,] x): the model and its absorbing layer,
        the grid PropTorch's snapshots are on."""
        return tuple(e * self.n_order + 1 for e in self._elems_pad)

    def _grid_points(self):
        """``(elem, wz, col, col_w)`` over the snapshot grid's points in
        (z, [y,] x) order: each point's element, its basis weights along z
        (0 above a topography) and its lateral column, whose basis weights
        over an element face are ``col_w[col]``.

        The lateral axes are uniform in every mesh PropSEM builds, so a
        column's element and local coordinate follow from its index. Along z
        the element rows are found on the mesh's own element faces -- flat,
        or graded under a topography, where z is linear in the element's eta
        on every column -- so this is exact for both. Cached per mesh.
        """
        cached = getattr(self, "_grid_pts", None)
        mesh = self.mesh
        if cached is not None and cached[0] is mesh:
            return cached[1]
        from .basis.gll import gll_nodes_weights
        no, ng = self.n_order, self.n_order + 1
        nodes = gll_nodes_weights(no, dtype=torch.float64)[0]
        grid = self._grid_shape()

        def lateral(n_pts, n_el):
            i = torch.arange(n_pts)
            e = (i // no).clamp(max=n_el - 1)
            return e, _lagrange(nodes, 2.0 * (i - e * no).double() / no - 1.0)

        ex, wx = lateral(grid[-1], self._elems_pad[-1])
        if self.ndim == 2:
            col_e, col_w = ex, wx                              # (Ncol,), (Ncol, ng)
        else:
            ey, wy = lateral(grid[1], self._elems_pad[1])
            col_e = (ey[:, None] * self._elems_pad[-1] + ex[None, :]).reshape(-1)
            col_w = (wy[:, None, :, None] * wx[None, :, None, :]).reshape(
                col_e.numel(), ng * ng)
        n_lat = 1
        for e in self._elems_pad[1:]:
            n_lat *= e
        nez = self._elems_pad[0]
        z_e = mesh.z_e.detach().cpu().double().reshape(mesh.n_elements, ng, -1)
        top, bot = z_e[:, 0], z_e[:, -1]                       # (E, ng_lat)
        z0 = torch.stack([(top[ez * n_lat + col_e] * col_w).sum(-1)
                          for ez in range(nez)], 1)            # (Ncol, nez)
        z1 = torch.stack([(bot[ez * n_lat + col_e] * col_w).sum(-1)
                          for ez in range(nez)], 1)
        zq = (torch.arange(grid[0], dtype=torch.float64) * self.dh).expand(
            col_e.numel(), grid[0]).contiguous()               # (Ncol, Nz)
        # the element whose rows hold the point; only the surface gets a
        # tolerance (float32 mesh coordinates) -- inside, a point a hair above
        # an element edge belongs to the element above it
        row = torch.searchsorted(z0, zq, right=True) - 1
        air = zq < z0[:, :1] - 1e-4 * self.dh
        row = row.clamp(0, nez - 1)
        a, b = z0.gather(1, row), z1.gather(1, row)
        eta = (2.0 * (zq - a) / (b - a) - 1.0).clamp(-1.0, 1.0)
        wz = _lagrange(nodes, eta.reshape(-1)).reshape(eta.shape + (ng,))
        wz[air] = 0.0
        elem = row * n_lat + col_e[:, None]
        # (Ncol, Nz) -> the grid's (z, column) order
        tables = (elem.T.reshape(-1), wz.transpose(0, 1).reshape(-1, ng),
                  torch.arange(col_e.numel()).repeat(grid[0]), col_w)
        self._grid_pts = (mesh, tables)
        return tables

    def _on_grid(self, vals):
        """GLL values ``(K, n_global)`` -> ``(K, npoints)`` at the snapshot
        grid's points, in chunks of points. A product and a sum over the
        element's nodes, not a matrix product: every point then sums in the
        same order however many snapshots come along, so a snapshot is the
        same bits whichever others were asked for."""
        elem, wz, col, col_w = self._grid_points()
        ng = self.n_order + 1
        ib = self.mesh.ibool.detach().cpu().long().reshape(-1, ng, col_w.shape[1])
        out = vals.new_zeros(vals.shape[0], elem.numel())
        step = max(1, (1 << 24) // max(1, vals.shape[0] * ib[0].numel()))
        for s in range(0, elem.numel(), step):
            sl = slice(s, s + step)
            w = (wz[sl, :, None] * col_w[col[sl], None, :]).to(vals.dtype)
            out[:, sl] = (vals[:, ib[elem[sl]]] * w).sum((-2, -1))
        return out

    # ------------------------------------------------------------------ #
    # The one eager entry: built props + mode -> seismogram
    # ------------------------------------------------------------------ #
    def _forward_eager(self, eq, pershot_eqs, props, acq, *, memory=None,
                       compile_step=False, checkpoint_chunk=None,
                       store_dtype=None):
        """Run the built props under one memory mode -> ``(nshots, nt, nrec,
        ncomp)``. Reads like ``prop_c.sem_c_forward_shot_batch``: the props carry
        the acquisition, the keywords carry the mode, and the body is the runner
        table.

        Four runners, in the order the conditions are tested:

        ==========================  ================================================
        memory='boundary'           single shot -> ``_forward_bs_single``
                                    per-shot models -> the same, looped (the
                                    vmapped rollout closure-captures ONE
                                    equation's coefficients)
                                    shared model, nshots>1 -> ``_forward_bs_batched``
        memory=None/'full'          single shot -> ``props[0].forward`` (no vmap
                                    overhead)
                                    per-shot models -> that, looped
                                    shared model, nshots>1 -> ``_run_batched``
                                    (vmapped; ``checkpoint_chunk`` chunks its tape)
        ==========================  ================================================

        ``checkpoint_chunk`` reaches exactly one of them, so the other two
        full-tape branches REJECT it rather than accept and drop it. The two
        rejections keep their historical (and differing) exception types.
        """
        compile_step = bool(compile_step)
        recv_idxs = self._recv_idxs(eq)
        nshots = acq.nshots

        if memory == "boundary":
            if nshots == 1:                          # single shot: no vmap overhead
                return self._forward_bs_single(
                    eq, props[0], acq.wav_per_shot[0], recv_idxs,
                    compile_step=compile_step, store_dtype=store_dtype).unsqueeze(0)
            if pershot_eqs is not None:
                # per-shot models: the vmapped rollout closure-captures ONE
                # equation's coefficients, so batch by looping instead.
                return torch.stack([self._forward_bs_single(
                    pershot_eqs[b], props[b], acq.wav_per_shot[b], recv_idxs,
                    compile_step=compile_step,
                    store_dtype=store_dtype) for b in range(nshots)], 0)
            return self._forward_bs_batched(eq, props, acq, recv_idxs,
                                            store_dtype=store_dtype)

        if nshots == 1:                              # single shot: no vmap overhead
            if checkpoint_chunk:
                # It only reaches _run_batched, which this branch never calls,
                # so it used to be accepted and dropped. Say so rather than let
                # a caller believe the tape is being chunked.
                raise ValueError(
                    "Ckpt only applies to the multi-shot (vmapped) path; this "
                    "call has nshots=1, where it would be a no-op. Use "
                    "memory=BoundarySaving() to bound single-shot memory.")
            # (nt, nrec, ncomp) from the low level -> (1, nt, nrec, ncomp)
            return props[0].forward(acq.wav_per_shot[0],
                                    compile_step=compile_step).unsqueeze(0)
        if pershot_eqs is not None:
            # per-shot models: vmap's mapped-step closure captures ONE
            # equation's coefficient tensors, so run the shots as a loop. A
            # broadcast (shared) model still reaches the vmapped path below.
            if checkpoint_chunk:
                raise NotImplementedError(
                    "Ckpt applies to the vmapped shared-model path; per-shot "
                    "models run as a loop, where it would be silently ignored.")
            return torch.stack([props[b].forward(acq.wav_per_shot[b],
                                                 compile_step=compile_step)
                                for b in range(nshots)], 0)
        return self._run_batched(eq, props, acq.wav_per_shot, recv_idxs,
                                 chunk_size=checkpoint_chunk,
                                 compile_step=compile_step)

    # ------------------------------------------------------------------ #
    # The receiver gather, in ONE place: it was written out verbatim in
    # _forward_bs_single, the batched BS block and _run_batched (and reads the
    # same two things every time -- the global node ids behind a stamp's element
    # and the stamp's Lagrange weights).
    # ------------------------------------------------------------------ #
    def _gather_rows(self, mesh, prop):
        """One shot's ``(rec_G, rec_W)`` -- see ``propagator.receiver_gather``,
        which this and the C consts builder share."""
        return receiver_gather(mesh, prop.receivers)


    def _on_dev(self, G, W):
        return G.to(self.dev), W.to(device=self.dev, dtype=self.dtype)


    def _gather_batch(self, mesh, props):
        """The same, stacked over shots: ``(B, nrec, ngll^d)``."""
        G, W = zip(*(self._gather_rows(mesh, p) for p in props))
        return self._on_dev(torch.stack(G), torch.stack(W))

    # ------------------------------------------------------------------ #
    # The four eager runners
    # ------------------------------------------------------------------ #
    def _forward_bs_single(self, eq, prop, wav, recv_idxs, *,
                           compile_step=False, store_dtype=None):
        """Single-shot boundary-saved rollout -> ``(nt, nrec, ncomp)``."""
        from .boundary_saving import run_boundary_saved
        mesh = eq.mesh
        if wav.dim() == 2:                          # FD mode-B per-source wavelet
            source_spec = prop._build_per_source_patterns(self.dev, self.dtype)
        else:
            source_spec = prop._build_source_patterns(mesh.n_global, self.dev,
                                                      self.dtype)
        rec_G, rec_W = self._on_dev(*self._gather_rows(mesh, prop))
        out = run_boundary_saved(eq, source_spec, wav, rec_G, rec_W,
                                 self.dt, self.nt, compile_step=compile_step,
                                 store_dtype=store_dtype, recv_idx=recv_idxs)
        return torch.stack(out, -1).transpose(0, 1)


    def _forward_bs_batched(self, eq, props, acq, recv_idxs, *, store_dtype=None):
        """Multi-shot boundary-saved rollout: ONE vmapped reverse march over all
        shots -> ``(nshots, nt, nrec, ncomp)``. Shared model only -- the mapped
        closure captures one equation's coefficients (the per-shot-model case
        loops :func:`_forward_bs_single` instead).

        ``compile_step`` is deliberately absent: ``run_boundary_saved_batched``
        does not take it (compile(vmap) of the BS step is a separate lever), and
        accepting it here would be the silent no-op the option guards exist to
        stop.
        """
        from .boundary_saving import run_boundary_saved_batched
        mesh = eq.mesh
        pats = [pr._build_source_patterns(mesh.n_global, self.dev, self.dtype)
                for pr in props]
        src_pats = [torch.stack([pl[f] for pl in pats])
                    for f in range(len(pats[0]))]                 # per source field
        rec_G, rec_W = self._gather_batch(mesh, props)
        wav_B = torch.stack(acq.wav_per_shot)                     # (B, nt)
        out = run_boundary_saved_batched(
            eq, src_pats, wav_B, rec_G, rec_W, self.dt, self.nt,
            store_dtype=store_dtype, recv_idx=recv_idxs)
        return torch.stack(out, -1).transpose(1, 2)


    def _run_batched(self, eq, props, wav_per_shot, recv_idxs, *,
                     chunk_size=None, compile_step=False):
        """vmap-batched multi-shot time loop — runs all shots in parallel.

        ``torch.vmap`` vectorises the *single-shot* ``eq.step`` over the shot
        dimension with no kernel changes: the dynamic wavefields are mapped
        (``in_dims=0``) and the constant coefficients are closure-captured.
        Bit-for-bit identical to the per-shot loop; autograd flows to the model.

        ``compile_step=True`` wraps the vmapped step in :func:`torch.compile`.
        """
        dev, dtype, dt, nt = self.dev, self.dtype, self.dt, self.nt
        B = len(props)
        mesh = eq.mesh
        n_global = mesh.n_global

        pat_list = [p._build_source_patterns(n_global, dev, dtype) for p in props]
        patterns = [torch.stack([pl[f] for pl in pat_list])
                    for f in range(len(pat_list[0]))]            # (B, n_global)
        rec_G, rec_W = self._gather_batch(mesh, props)           # (B, nrec, ngll^d)
        wav_B = torch.stack(wav_per_shot)                        # (B, nt)

        st0 = eq.init_state(dt, dev, dtype)                      # tuple
        dyn = tuple(f.unsqueeze(0).repeat(B, *([1] * f.dim())) for f in st0)
        vstep = torch.vmap(lambda fields, src: eq.step(fields, src, dt),
                           in_dims=(0, 0))
        if compile_step:
            vstep = torch.compile(vstep)

        ar = torch.arange(B, device=dev)[:, None, None]

        def advance(n0, k, dyn_in):
            d = dyn_in
            chunk = []
            for j in range(k):
                sval = wav_B[:, n0 + j]
                src = [pat * sval[:, None] for pat in patterns]
                d = vstep(d, src)
                comps = [(d[ri][ar, rec_G] * rec_W).sum(-1) for ri in recv_idxs]
                chunk.append(torch.stack(comps, -1))           # (B, nrec, ncomp)
            return d, torch.stack(chunk, dim=1)                # (B, k, nrec, ncomp)

        if not chunk_size:                            # full graph (short nt)
            _, seis = advance(0, nt, dyn)
            return seis                                    # (B, nt, nrec, ncomp)

        n_dyn = len(st0)

        def chunk_fn(n0, k, *dyn_in):
            d, sc = advance(n0, k, dyn_in)
            return (*d, sc)

        parts, n = [], 0
        while n < nt:
            k = min(chunk_size, nt - n)
            out = torch.utils.checkpoint.checkpoint(
                chunk_fn, n, k, *dyn, use_reentrant=False)
            dyn = out[:n_dyn]
            parts.append(out[-1])
            n += k
        return torch.cat(parts, dim=1)                     # (B, nt, nrec, ncomp)

    # ------------------------------------------------------------------ #
    # 3-D acoustic on CUDA: one driver, the strip one
    # ------------------------------------------------------------------ #
    def _dd3d_route(self, models):
        """The 3-D acoustic C-PML families run on the DD driver with a single
        strip. Measured against the fused single-domain driver it replaced
        (129^3, nt=60, RTX 6000 Ada): backward 15-22% faster, peak memory
        43-52% smaller -- the fused path built the global equation (its
        (E, ngll^3) material copies and the model interpolation whose autograd
        backward was 7% of device time) where the strip driver interpolates on
        the node grid and never holds anything (E, ngll^3)-sized. Per-shot
        models (a leading batch axis) keep the equation path."""
        if self.equation not in (AcousticSEM3D, AcousticSEM3DDeformed):
            return False
        if self.abcn_elements <= 0:
            return False
        m = models[0] if isinstance(models, (list, tuple)) else models
        return (torch.is_tensor(m) and tuple(m.shape) == tuple(self.shape)
                and (not isinstance(models, (list, tuple)) or len(models) == 1))

    def _stamp3d(self, x, y, z):
        """A point's element and Lagrange weights. On the flat hex8 mesh from
        the layout (no global mesh is ever built on this route); on a
        deformed one from the mesh, which topography builds anyway."""
        from .propagator import build_point_stamp, PointStamp3D, lagrange_basis_at
        if self.equation is AcousticSEM3DDeformed:
            return build_point_stamp(self.mesh, x, y, z)
        lay = self.hex8_layout()
        if not (0.0 <= x <= lay.x_max and 0.0 <= y <= lay.y_max and 0.0 <= z <= lay.z_max):
            raise ValueError(f"point ({x}, {y}, {z}) outside domain")
        ex = min(int(x / lay.dx_e), lay.nex - 1)
        ey = min(int(y / lay.dy_e), lay.ney - 1)
        ez = min(int(z / lay.dz_e), lay.nez - 1)
        return PointStamp3D(
            elem_id=(ez * lay.ney + ey) * lay.nex + ex,
            w_gamma=lagrange_basis_at(lay.xi, 2.0 * (z - ez * lay.dz_e) / lay.dz_e - 1.0),
            w_eta=lagrange_basis_at(lay.xi, 2.0 * (y - ey * lay.dy_e) / lay.dy_e - 1.0),
            w_xi=lagrange_basis_at(lay.xi, 2.0 * (x - ex * lay.dx_e) / lay.dx_e - 1.0))

    # ------------------------------------------------------------------ #
    # The C routes: reproduced unchanged EXCEPT for the _rec_row call that falls
    # out of the factory (marked below). Nothing else in them moves -- the stamp
    # caches, the identity test that gates the shot batch and the (kind, payload)
    # derivation are all the C path's own contract.
    # ------------------------------------------------------------------ #
    def _forward_c_dd3d(self, shot_src, receivers, wav_per_shot, models,
                        pml_guard):
        """Per shot: stamp the sources and receivers, run the single-strip DD
        driver, stack to ``(nshots, nt, nrec, 1)``. Stamps are cached on the
        (source, receiver) key like the fused path's point stamps were."""
        from types import SimpleNamespace
        from .dd import SEMModelParallel, StripTopology, _Stamps
        from .propagator import _check_pml_placement
        dev = torch.device("cuda")
        dd = getattr(self, "_dd1", None)
        if dd is None:
            dd = self._dd1 = SEMModelParallel(self, StripTopology(1))
        cache = getattr(self, "_dd_stamps", None)
        if cache is None:
            cache = self._dd_stamps = {}

        def stamps(pts):
            key = tuple(tuple(map(float, xz)) for xz in pts)
            hit = cache.get(key)
            if hit is None:
                hit = cache[key] = _Stamps.of([self._stamp3d(*xz) for xz in pts], dev)
            return hit

        lay = self.hex8_layout() if self.equation is AcousticSEM3D else self.mesh
        guard_eq = SimpleNamespace(mesh=lay, cpml=SimpleNamespace(
            sides=tuple(self._absorbing_sides()),
            pml_thickness=self.abcn_elements * max(float(lay.dx_e), float(lay.dy_e),
                                                   float(lay.dz_e))))
        seis = []
        for b, srcs in enumerate(shot_src):
            w = wav_per_shot[b]
            _c_shared_wavelet(w)
            rec_xz = self._rec_row(receivers, b)                      # was inline
            _check_pml_placement(guard_eq, srcs, rec_xz, pml_guard)
            out = dd(w, stamps(srcs), stamps(rec_xz), models)   # (nt, nrec)
            seis.append(out.unsqueeze(-1))
        return torch.stack(seis, 0)                                 # (nshots, nt, nrec, 1)


    def _c_batchable(self, eq, nsh, same_rec, wav_per_shot, memory, pershot_eqs):
        """Does this call take the shot batch? The five conditions of the
        "Shot batching on impl='c'" section of docs/user-guide/sem.md, in one
        place. ``nsh`` and ``same_rec`` are derived from the stamp cache, so
        this runs AFTER it, never before.

        The batchable test is looked up on the module here rather than imported
        at the top: three shot-batch tests force the per-shot loop by patching
        ``prop_c.sem_c_shot_batchable``.
        """
        from . import prop_c
        return (nsh > 1 and memory in (None, "boundary") and same_rec
                and prop_c.sem_c_shot_batchable(eq)
                # per-shot models ride the batch only where the kernels take
                # them (Bm == B, acoustic); elsewhere fall to the per-shot loop
                and (pershot_eqs is None or self._c_materials is not None)
                and all(w.dim() == 1 for w in wav_per_shot))


    def _forward_c(self, eq, shot_src, receivers, wav_per_shot, memory=None,
                   store_dtype=None, store_dtype_elem=None, sv=None,
                   pershot_eqs=None):
        """CUDA SEM backend: one :class:`SEMCudaPropagator` per shot (save-all
        forward + adjoint), stacked to ``(nshots, nrec, nt)``. The autograd
        Function chains gradients through ``build()`` to the (interpolated) vp,
        then ``_interp_to_gll`` chains to the grid model.

        The point stamps (source/receiver mesh locate + basis evaluation) are
        pure geometry and are CACHED across calls — rebuilding them cost
        ~2.7 ms/receiver (per-point mesh search with GPU syncs) and dominated
        the per-shot runtime before the CUDA loop even started."""
        self._c_cache_stamps(shot_src, receivers)
        nsh = len(shot_src)
        same_rec = all(s is self._c_rec_stamps[0] for s in self._c_rec_stamps)
        if self._c_batchable(eq, nsh, same_rec, wav_per_shot, memory, pershot_eqs):
            return self._forward_c_batched(eq, nsh, wav_per_shot, sv, memory,
                                           store_dtype, store_dtype_elem)
        seis = []
        for b in range(nsh):
            _c_shared_wavelet(wav_per_shot[b])
            prop = self._c_prop(eq if pershot_eqs is None else pershot_eqs[b],
                                b, sv)
            seis.append(prop.forward(wav_per_shot[b], memory=memory,
                                     bs_store_dtype=store_dtype,
                                     bs_store_dtype_elem=store_dtype_elem))
        return torch.stack(seis, 0)                          # (nshots, nt, nrec, ncomp)

    def _c_cache_stamps(self, shot_src, receivers):
        """Fill ``_c_src_stamps`` / ``_c_rec_stamps`` for this geometry, reusing
        them when the geometry has not moved."""
        from .propagator import build_moment_stamp, build_point_stamp
        src_key = tuple(tuple(map(float, xz)) for s in shot_src for xz in s)
        rec_rows = [tuple(receivers[b].reshape(-1).tolist())
                    for b in range(receivers.shape[0])]
        key = (src_key, tuple(rec_rows))
        if getattr(self, "_c_stamp_key", None) == key:
            return
        # a moment tensor is stamped in every element at an edge (EdgeStamp),
        # exactly as the eager build does it
        stamp = build_moment_stamp if self._src_moment is not None else build_point_stamp
        self._c_src_stamps = [[stamp(self.mesh, *xz) for xz in s] for s in shot_src]
        row_cache = {}                        # dedupe expanded receiver rows
        self._c_rec_stamps = []
        for b, row in enumerate(rec_rows):
            if row not in row_cache:
                row_cache[row] = [build_point_stamp(self.mesh, *xz)
                                  for xz in self._rec_row(receivers, b)]
            self._c_rec_stamps.append(row_cache[row])
        self._c_stamp_key = key

    def _c_prop(self, eq, b, sv):
        """Shot ``b``'s :class:`SEMCudaPropagator`, from the cached stamps.

        ONE place derives (kind, payload). The eager path derives them from
        ``source_type``; impl='c' used to hard-code a scalar source, which is
        why it could only ever serve the acoustic equations. Both the batched
        entry and the per-shot loop come here, so batched == loop stamp for
        stamp -- they used to each carry the derivation with a comment saying
        it matched the other. (fs: "force" + the source_type vector lands the
        unit stamp on chi via the positional zip.)
        """
        from .prop_c import SEMCudaPropagator
        srcs = self._c_src_stamps[b]
        if self.is_acoustic:
            kind, payload = "scalar", (1.0,)
        elif self._src_moment is not None:
            # the kernels only ever see the assembled per-field patterns, so a
            # moment tensor needs nothing from them that a force does not
            kind, payload = "moment_tensor", self._src_moment
        else:
            kind, payload = "force", sv
        return SEMCudaPropagator(
            eq=eq, dt=self.dt, nt=self.nt, sources=srcs,
            receivers=self._c_rec_stamps[b],
            source_kinds=[kind] * len(srcs),
            source_payloads=[payload] * len(srcs),
            receiver_type=self._recv_base)

    def _forward_c_batched(self, eq, nsh, wav_per_shot, sv, memory,
                           store_dtype, store_dtype_elem):
        """B shots in ONE kernel sequence instead of B serial loops.

        The backend is launch-bound, so B x nt separate launches are exactly
        the overhead worth amortising. ``_c_batchable`` has already checked the
        conditions -- among them that every shot's receiver row is the SAME
        stamp object, which is why ``_c_prop`` may read ``_c_rec_stamps[b]``
        here as it does in the loop. ``memory='boundary'`` batches too: the
        band ring is (nt, B, nband), so large-B fwd+bwd stays affordable where
        the (nt, B, ng) save-all histories would not.
        """
        from .prop_c import sem_c_forward_shot_batch
        props = [self._c_prop(eq, b, sv) for b in range(nsh)]
        if all(w is wav_per_shot[0] for w in wav_per_shot):
            wav = wav_per_shot[0]                                  # (nt,) shared
        else:
            wav = torch.stack(list(wav_per_shot))                  # (B, nt)
        out = sem_c_forward_shot_batch(props, wav,
                                       materials=self._c_materials,
                                       memory=memory,
                                       bs_store_dtype=store_dtype,
                                       bs_store_dtype_elem=store_dtype_elem)
        if out.dim() == 3:                    # single-component acoustic
            out = out.unsqueeze(-1)           # (B, nrec, nt, 1)
        return out.transpose(1, 2)            # (B, nt, nrec, ncomp)

    __call__ = forward
