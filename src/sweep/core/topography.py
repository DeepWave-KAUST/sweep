"""Irregular free-surface topography: policy and runtime-grid construction.

Pulled out of the propagator base class, but only the parts that are genuinely
computations. The ASSIGNMENTS stay in ``PropBase``, deliberately -- see below.

Why the mutation boundary is where it is
----------------------------------------
Three of these methods used to write runtime state onto the *equation* object
(``equation._topo_rows_runtime``, ``equation._apm_air_mask_runtime``), which
looks like exactly the coupling this refactor set out to remove: an equation is
supposed to be a specification, not a per-propagator blackboard.

It cannot be removed here, because that state is the only channel that exists.
Eager kernels read it off ``self`` at call time -- ``Acoustic.func`` branches on
``getattr(self, "_topo_rows_runtime", None)``, ``Elastic.func`` dispatches its
*entire step* to the APM branch on ``_apm_air_mask_runtime`` -- and the compiled
path reads the same attributes off ``self.equation``. Neither ``func`` nor
``interior_substeps`` takes a surface argument, and the boundary-saving reverse
driver calls the latter with no propagator in scope. Making the equation a real
specification means changing ``func``'s signature across every equation class
plus both reverse drivers plus the compiled binding -- a different refactor, and
not one that lands under a bit-exact gate in one step.

So: this module computes, ``PropBase`` installs. In particular the *reset* --
clearing all three attributes on every construction -- stays in the propagator,
because it is an ownership claim rather than bookkeeping. It is what stops
``Prop(eq, topography=hill)`` followed by ``Prop(eq)`` from leaving the second
propagator quietly running on the first one's surface, and no gate in the tree
builds two propagators on one equation, so losing it would ship green.

Two parameters that look wrong and are not
------------------------------------------
* **``abcn``, not ``pad``.** Every extent here uses the scalar ``abcn``, even
  though a per-edge free surface or a DD cut face makes the actual pad on a
  given face something else. That is correct as written: under domain
  decomposition the propagator shrinks ``pad`` on cut faces while ``abcn`` stays
  at full width, and the topography extents are defined against the latter.
  Passing ``pad`` "because it is more accurate" changes DD-with-topography
  output.
* **``device`` resolved with ``or``.** ``getattr(equation, "device", None) or
  dev`` is a truthiness test, not ``is not None``. It agrees with the stricter
  form for every value seen today; it is reproduced verbatim rather than tidied.
"""
from __future__ import annotations


def resolve_topo_method(*, topography, topo_method, free_surface,
                        supports_apm, is_curvilinear, equation_name):
    """Resolve ``(method, free_surface, image_method_active)``.

    Pure policy over four flags. ``image_method_active`` is not a synonym for
    "there is a free surface": it selects the *layout*, suppressing the top PML
    band, which the compiled kernels' free-surface bitmask and the PML pad both
    key off.
    """
    has_topo = topography is not None

    # Curvilinear equations have their own topography path (a boundary-fitted
    # grid via metric tensors), so ``topo_method`` does not apply. Honour the
    # caller's free_surface verbatim and skip method resolution.
    if is_curvilinear:
        fs = bool(free_surface) or has_topo
        return None, fs, fs

    if not has_topo:
        if free_surface:
            # A flat free surface is the image method by definition; APM needs
            # per-cell categories, which only a topography mask provides.
            if topo_method == 'apm':
                raise ValueError(
                    "topo_method='apm' requires a topography= mask; "
                    "for a flat free surface use topo_method='image' "
                    "(or omit it)."
                )
            return 'image', True, True
        return None, False, False

    # Topography given: the free surface is implicit. free_surface=False is
    # overridden rather than warned about -- topography implies a surface, so
    # the combination is not ambiguous.
    if topo_method == 'auto':
        method = 'apm' if supports_apm else 'image'
    elif topo_method == 'apm':
        if not supports_apm:
            raise ValueError(
                f"topo_method='apm' requires equation.supports_apm=True; "
                f"{equation_name} only supports the image "
                f"method (use topo_method='image' or omit topo_method)."
            )
        method = 'apm'
    else:                       # 'image'
        method = 'image'

    return method, True, (method == 'image')


def physical_extent(shape, *, abcn, ndim, image_method_active):
    """Runtime (PML-padded) shape -> physical extent.

    z is asymmetric under the image method: it suppresses the TOP PML band only,
    so z loses ``abcn`` once rather than twice. x (and y) always lose it twice.
    """
    nz_phys = shape[0] - (abcn if image_method_active else 2 * abcn)
    if ndim == 2:
        return (nz_phys, shape[1] - 2 * abcn)
    return (nz_phys, shape[1] - 2 * abcn, shape[2] - 2 * abcn)


def canonicalise_topography(topo_input, *phys_extent):
    """Validate the surface-row array and derive the matching air mask.

    * 2-D (``phys_extent = (nz_phys, nx_phys)``): ``topo_input`` is 1-D
      ``(nx_phys,)``; returns rows ``(nx_phys,)`` and mask
      ``(nz_phys, nx_phys)``.
    * 3-D (``phys_extent = (nz_phys, ny_phys, nx_phys)``): ``topo_input`` is 2-D
      ``(ny_phys, nx_phys)``; returns rows of the same shape and mask
      ``(nz_phys, ny_phys, nx_phys)``.

    Rows are ``int64``; the mask is **float32**, not bool -- ``F.pad`` in
    replicate mode rejects bool, and the APM classifier branches on the tensor
    having a ``device``. The mask is also born on the INPUT's device, which for
    a numpy topography is the CPU even on a CUDA run; moving it here would
    change what the caller stores.

    ``*phys_extent`` is a star-arg on purpose: the 2-D/3-D branch is chosen by
    its LENGTH. Rewriting it as keywords would let a stray ``ny=None`` select
    the wrong branch in silence.
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


def build_image_method_topo_rows(topo_row_phys, *, abcn, halo, device):
    """Physical surface rows -> the runtime int32 row array the kernels index.

    Runtime z layout under a free surface::

        [0, halo)                top stencil halo (holds the image mirror)
        [halo, halo + nz_phys)   physical interior
        [halo + nz_phys, ...)    bottom PML + bottom halo

    Three things here are load-bearing and must not be tidied:

    * **int32, not int64.** The compiled side reads ``data_ptr<int>()``. A dtype
      cast at the call site instead would create a temporary whose GPU memory is
      recycled before the async kernels have finished reading it; and widening
      the dtype turns a wrong number into a pointer reinterpretation.
    * **float32 -> replicate pad -> int32.** ``F.pad(mode="replicate")`` does not
      accept integer tensors. Replicate copies values, so the round trip is
      exact; substituting an integer pad routine would look equivalent and
      silently change the dtype contract above.
    * **The CUDA synchronize.** It guarantees the host-to-device copy has landed
      before any forward kernel indexes ``topo_rows[..., ix]``. Without it this
      showed roughly 30% non-determinism in the CUDA forward. Dropping it does
      not produce a red gate -- it produces a *flaky* one.

    The 3-D pad tuple is in REVERSED axis order relative to the array
    (``(x_lo, x_hi, y_lo, y_hi)`` over a ``(ny, nx)`` view). Every width is
    equal, so a transposition here is invisible unless ``ny != nx``.
    """
    import torch
    import torch.nn.functional as F

    topo_z = topo_row_phys + halo
    pad_each = abcn + halo

    if topo_row_phys.ndim == 1:
        topo_runtime = F.pad(
            topo_z.to(torch.float32).view(1, 1, -1),
            (pad_each, pad_each),
            mode="replicate",
        ).view(-1).to(torch.int32)
    else:
        topo_runtime = F.pad(
            topo_z.to(torch.float32).view(1, 1, *topo_z.shape),
            (pad_each, pad_each, pad_each, pad_each),
            mode="replicate",
        ).squeeze(0).squeeze(0).to(torch.int32)

    if device is not None:
        try:
            topo_runtime = topo_runtime.to(device=device)
        except (RuntimeError, TypeError):
            pass

    if topo_runtime.device.type == "cuda":
        torch.cuda.synchronize(topo_runtime.device)
    return topo_runtime


def build_apm_air_mask(air_mask_phys, *, abcn, halo, device):
    """Physical air mask -> the runtime mask, replicate-padded on every face.

    Stays float32 throughout: ``F.pad(mode="replicate")`` rejects bool, and the
    APM classifier downstream branches on the tensor having a ``device``.

    The 3-D pad tuple is in REVERSED axis order -- ``F.pad`` on a
    ``(1, 1, D, H, W)`` view takes ``(W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)``,
    i.e. x, y, z against an array laid out ``(nz, ny, nx)``. All six widths are
    equal here, so a transposed tuple is invisible unless the extents differ;
    that is what makes it worth stating rather than leaving to be re-derived.
    """
    import torch
    import torch.nn.functional as F

    pad_each = abcn + halo

    if air_mask_phys.ndim == 2:
        nz_phys, nx_phys = air_mask_phys.shape
        air_mask_padded = F.pad(
            air_mask_phys.view(1, 1, nz_phys, nx_phys),
            (pad_each, pad_each, pad_each, pad_each),
            mode="replicate",
        ).view(nz_phys + 2 * pad_each, nx_phys + 2 * pad_each)
    elif air_mask_phys.ndim == 3:
        nz_phys, ny_phys, nx_phys = air_mask_phys.shape
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

    if device is not None:
        try:
            air_mask_padded = air_mask_padded.to(device=device)
        except (RuntimeError, TypeError):
            pass
    return air_mask_padded
