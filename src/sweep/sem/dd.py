"""SEM model-domain decomposition (1-D x strips) on the impl='c' backend.

The SEM mirror of the FD ``sweep.parallel`` stack, phase 1: the acoustic
C-PML CUDA path, cut into x strips, one rank (or one sequential partition in
the single-process harness) per strip.

Why the C loop must be STEPPED here: the fused driver runs all ``nt`` steps
inside one C++ call, but a cut needs an interface-column sum after each
step's assembly. So DD runs the loop in Python over split-phase C entry
points (the FD stepped pattern):

    forward : dd_grads (assembly) -> exchange 6 fields -> dd_step (update)
    backward: dd_bwd_pre (grads_f recompute) -> exchange 6
              -> dd_bwd_mid (step_T + grads_T)  -> exchange 3
              -> dd_bwd_end (combine + wavelet grad)

Exchange semantics: SEM assembles by SUMMING interface contributions (FD
halos are copy semantics), so the acceptance bar is tolerance-based on GPU
and the backward needs two DEDUP rules the eager autograd version never
did:

* the adjoint exchange makes each rank's interface adjoint the COMPLETE
  single-domain value, so the per-node material gradients (kappa, inv_M)
  come out complete on BOTH sides of a cut -- the non-owned interface
  column is zeroed before the gradients ride the slice chain back to the
  global model (ownership: an interface column belongs to the RIGHT strip,
  matching the half-open receiver rule);
* the wavelet gradient is a global scalar reduction per step, so it reads
  an ownership-masked source pattern (``src_pattern_own``).

Setup follows the slice-don't-rebuild rule: masses, ``kappa_node``, the
C-PML profile AND the source pattern come from ONE global build,
column-sliced per strip -- a per-strip profile rebuild would grow absorbing
bands at every internal cut (the mirror of FD's "PML hole"; the forward
asserts the cut columns are PML-free), and a setup-time interface sum of
the sliced pattern would DOUBLE a source sitting exactly on a cut (both
copies already carry the complete value; caught by the src-on-cut
falsifier). Cut-column receivers are registered on BOTH strips but owned by
one -- see ``_Part``.

Modes: without ``torch.distributed`` one process steps every strip (the
FD PLAN.md D5 correctness harness -- exchange is a tensor add); under
``torchrun`` each rank owns one strip and the exchange is a batched
isend/irecv pair (self-adjoint, so the reverse loop reuses it verbatim).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from .drivers.acoustic3d import (_SEMAcoustic3DCpmlDDC, _AC3D_RINGS,  # noqa: F401
                                 _AC3D_TAPES)
from .cdrive import EL_CPML_FIELDS, EL_CPML_TRIPLES
from .drivers.acoustic2d import _SEMAcousticCpmlDDC
from .drivers.elastic2d import _SEMElasticCpmlDDC
from .drivers.fluid_solid import _SEMFluidSolidDDC

from .equations.acoustic_potential2d import AcousticPotentialSEM2D
from .cpml._axis import damping_profile as _damping_profile_1d
from .cpml.acoustic3d import _ab_coeffs as _cpml_ab_coeffs
from .equations.acoustic3d import AcousticSEM3D
from .equations.acoustic3d_deformed import AcousticSEM3DDeformed
from .equations.acoustic_elastic2d import AcousticElasticSEM2D
from .equations.elastic2d_deformed import ElasticSEM2DDeformed
from .mesh.quad9 import build_axis_aligned_quad9


@dataclass(frozen=True)
class _Family:
    """One equation family's fixed facts, looked up once in ``__init__``.

    These stood as four booleans re-tested in priority order in seven places:
    the consts builder, the Function, how many source patterns the family
    takes and which one a point source drives, how many record components come
    back, whether the compact 3-D source applies, whether ``bs_store_dtype_elem``
    does, and whether an internal cut can be checked for C-PML. One row per
    equation means adding a family is a row plus its consts method and its
    Function -- not seven if-chains kept in step by hand.
    """

    ndim: int
    ncomp: int          # record components -- and, equally, source patterns
    src_field: int      # which pattern a point source lands in
    elem_rings: bool    # has ELEMENT boundary histories (bs_store_dtype_elem)
    node_cpml: bool     # has a node-level b_x the cut check can read
    deformed: bool      # curvilinear: per-element-node metric geometry
    consts: str         # SEMModelParallel method building one strip's consts
    fn: type            # the autograd Function stepping the strips


#                          ndim ncomp src  elem   node   deformed
_FAMILIES = {
    AcousticPotentialSEM2D: _Family(2, 1, 0, False, True,  False,
                                    "_consts_ac2d", _SEMAcousticCpmlDDC),
    # elastic: a unit VERTICAL force by default (PropSEM's source_vector
    # convention), and element-local C-PML coefficients, so an internal cut
    # element carries no PML by construction and there is no b_x to check
    ElasticSEM2DDeformed:   _Family(2, 2, 1, True,  False, False,
                                    "_consts_el2d", _SEMElasticCpmlDDC),
    # fluid-solid: the source drives chi, the equation's own convention
    AcousticElasticSEM2D:   _Family(2, 3, 0, True,  True,  False,
                                    "_consts_fs",   _SEMFluidSolidDDC),
    AcousticSEM3D:          _Family(3, 1, 0, False, True,  False,
                                    "_consts_ac3d", _SEMAcoustic3DCpmlDDC),
    AcousticSEM3DDeformed:  _Family(3, 1, 0, False, True,  True,
                                    "_consts_ac3d", _SEMAcoustic3DCpmlDDC),
}


@dataclass(frozen=True)
class StripTopology:
    """1-D x-strip rank grid (the ``py == 1`` row of FD's ``MeshTopology``).

    ``px`` strips over the element columns; z is never split. Pure
    arithmetic -- no ``torch.distributed`` calls.
    """

    px: int
    rank: int = 0

    def __post_init__(self):
        if self.px < 1:
            raise ValueError(f"px must be >= 1, got {self.px}")
        if not (0 <= self.rank < self.px):
            raise ValueError(f"rank {self.rank} outside [0, {self.px})")

    @property
    def is_edge_lo(self) -> bool:
        return self.rank == 0

    @property
    def is_edge_hi(self) -> bool:
        return self.rank == self.px - 1

    def local_elems(self, nex: int) -> tuple[int, int]:
        """Element-column range ``[ex0, ex1)`` (uniform tiles, the FD v1
        rule: pad the domain to a multiple of ``px`` yourself)."""
        if nex % self.px:
            raise ValueError(
                f"nex={nex} is not divisible by px={self.px}; pad the domain "
                f"to a multiple (uniform tiles, as in the FD v1 rule)")
        w = nex // self.px
        return self.rank * w, (self.rank + 1) * w


def _sendrecv_sum(x, peer):
    """out = x + the peer's x (both neighbours call it): the SEM interface
    SUM. Self-adjoint, so the reverse loop calls the same function."""
    import torch.distributed as dist
    buf = torch.empty_like(x)
    for req in dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, x.contiguous(), peer),
             dist.P2POp(dist.irecv, buf, peer)]):
        req.wait()
    return x + buf


@dataclass
class _Part:
    """One strip's C-side plumbing.

    Receiver plumbing distinguishes REGISTRATION from OWNERSHIP. A receiver
    on a cut column sits on a shared DOF whose two copies must BOTH receive
    the adjoint source in the backward (the mirror of the forward's
    source-pattern setup exchange) -- registering it only on the owning
    strip leaves the neighbour's interface adjoint short of the injection
    and corrupts the gradient (measured: vp rel 0.94 with a receiver on the
    cut). So every strip registers every receiver in its CLOSED column
    range, while output/ownership stays half-open and unique.
    """

    c: dict                       # per-strip consts (the _cpml_consts keys)
    gidx_spec: tuple              # (n_rows, nx_g, x0, x1c) -- see `gidx`
    if_lo: torch.Tensor | None    # local indices of the cut columns
    if_hi: torch.Tensor | None
    own: torch.Tensor             # 0/1 ownership mask (right strip owns a cut)
    own_rows: torch.Tensor        # registered rows this strip OWNS
    slot: torch.Tensor            # global record row per owned receiver
    dup_hi_rows: torch.Tensor     # registered rows on the hi cut (owned by
                                  # the right neighbour; dist grad_syn recv)
    own_lo_rows: torch.Tensor     # owned rows on the lo cut (dist send)

    @property
    def gidx(self):
        """local node -> global node, rebuilt on demand.

        It used to be a stored ``n_local`` int64 tensor, held from setup
        through the whole backward -- **8 B per global node per rank**
        (measured 301.2 -> 293.0 at nex=48, px=1) -- and read by nothing in
        ``src``: only by two tests, which this property keeps working
        unchanged. The strip's numbering is the row-major formula, so four
        ints describe it exactly.

        Test-facing: this ALLOCATES on every access. Do not put it in a loop.
        """
        n_rows, nxg, x0, x1c = self.gidx_spec
        dev = self.own.device
        rows = torch.arange(n_rows, device=dev)
        cols = torch.arange(x0, x1c + 1, device=dev)
        return (rows[:, None] * nxg + cols[None, :]).reshape(-1)


class _DDExchange:
    """Interface-column sum over a list of parts (single-process) or via
    P2P with the two neighbours (distributed)."""

    def __init__(self, parts, dist_mode):
        self.parts = parts
        self.dist = dist_mode
        self._bufs = {}          # cached staging buffers, see block()
        self._flats = {}         # cached flat cut indices, see block()

    def _flat(self, par, iface):
        """``iface`` in every row of a ``(K, ng)`` parent, as ONE 1-D index
        into its flattened storage -- exactly the entries ``par[:, iface]``
        reads, in that order.

        Cached per ``(K, ng, iface)``: the index is a pure function of the
        block's shape and the cut, both fixed for the whole run, and building
        it costs an allocation on a path that runs three times a step.
        """
        K, ng = par.shape
        key = (K, ng, id(iface))
        idx = self._flats.get(key)
        if idx is None:
            rows = torch.arange(K, device=iface.device) * ng
            idx = self._flats[key] = (rows[:, None] + iface[None, :]).reshape(-1)
        return idx

    def block(self, parents, flat=None):
        """Exchange a whole BLOCK of fields in one gather and one scatter --
        ``parents[i]`` is part i's block.

        This is the fast path, and the reason it exists is measured: the
        per-field :meth:`many` spends ~14 us per index kernel, so an
        acoustic 6-field exchange cost 204 us of which only 36 us was the
        actual P2P (ORIX H100, 2.4 kB). Indexing the parent once collapses
        12 kernels into 2. Callers that allocate the exchanged fields as
        rows of one tensor (as the C structs' buffers already are) get this
        for free.

        A block whose rows are all ``n_global`` long comes as a ``(K, ng)``
        tensor and the cut is the same column set in every row. A
        HETEROGENEOUS block -- some rows on an axis slab, so shorter -- comes
        as its 1-D storage plus ``flat``, where ``flat[i]`` is part i's
        ``(lo, hi)`` pair of flat offsets inside it (``None`` for a side with
        no cut). Both cost one index_select and one index_copy_: indexing
        FLAT is what makes the mixed-length case as cheap as the uniform one.
        """
        def side(i, lo_hi):
            """Part i's flat index for its lo (0) or hi (1) cut."""
            if flat is not None:
                return flat[i][lo_hi]
            p = self.parts[i]
            iface = p.if_lo if lo_hi == 0 else p.if_hi
            return None if iface is None else self._flat(parents[i], iface)

        # view, not reshape: a non-contiguous parent would reshape to a COPY
        # and the summed interface would be written into the copy and lost.
        # Every caller allocates its block as one buffer, so this holds.
        buf = [p.view(-1) for p in parents]
        if self.dist:
            import torch.distributed as dist
            r = dist.get_rank()
            for phase in (0, 1):
                for lo_hi, peer in ((0, r - 1), (1, r + 1)):
                    idx = side(0, lo_hi)
                    if idx is None or (min(r, peer) % 2) != phase:
                        continue
                    # staging buffers are cached per (length, side): a fresh
                    # pair per exchange cost two allocations on a path that
                    # runs 3x per time step
                    key = (int(idx.numel()), peer)
                    st = self._bufs.get(key)
                    if st is None:
                        st = (torch.empty(idx.numel(), dtype=buf[0].dtype,
                                          device=buf[0].device),
                              torch.empty(idx.numel(), dtype=buf[0].dtype,
                                          device=buf[0].device))
                        self._bufs[key] = st
                    send, recv = st
                    torch.index_select(buf[0], 0, idx, out=send)
                    for req in dist.batch_isend_irecv(
                            [dist.P2POp(dist.isend, send, peer),
                             dist.P2POp(dist.irecv, recv, peer)]):
                        req.wait()
                    send.add_(recv)
                    buf[0].index_copy_(0, idx, send)
            return parents
        for i in range(len(self.parts) - 1):
            hi, lo = side(i, 1), side(i + 1, 0)
            tot = buf[i][hi] + buf[i + 1][lo]
            buf[i].index_copy_(0, hi, tot)
            buf[i + 1].index_copy_(0, lo, tot)
        return parents

    def many(self, field_lists):
        """Exchange SEVERAL fields in one shot: ``field_lists[k][i]`` is
        field k of part i. Under ``dist`` the interface slices are packed
        into ONE isend/irecv pair instead of one per field, which is what
        keeps the fluid-solid step (7 + 4 fields) from paying seven
        latencies."""
        if not self.dist:
            for fl in field_lists:
                self(fl)
            return field_lists
        import torch.distributed as dist
        p, r = self.parts[0], dist.get_rank()
        K = len(field_lists)
        for phase in (0, 1):
            for iface, peer in ((p.if_lo, r - 1), (p.if_hi, r + 1)):
                if iface is None or (min(r, peer) % 2) != phase:
                    continue
                n = iface.numel()
                send = torch.empty(K * n, dtype=torch.float32,
                                   device=field_lists[0][0].device)
                for k, fl in enumerate(field_lists):
                    send[k * n:(k + 1) * n] = fl[0][iface]
                recv = torch.empty_like(send)
                for req in dist.batch_isend_irecv(
                        [dist.P2POp(dist.isend, send, peer),
                         dist.P2POp(dist.irecv, recv, peer)]):
                    req.wait()
                tot = send + recv
                for k, fl in enumerate(field_lists):
                    fl[0].index_copy_(0, iface, tot[k * n:(k + 1) * n])
        return field_lists

    def __call__(self, fields):
        """``fields[i]`` is part i's tensor in chi layout; summed in place
        on every internal interface column, returned."""
        if self.dist:
            import torch.distributed as dist
            (p,), (f,) = self.parts, fields
            r = dist.get_rank()
            for phase in (0, 1):
                for iface, peer in ((p.if_lo, r - 1), (p.if_hi, r + 1)):
                    if iface is None or (min(r, peer) % 2) != phase:
                        continue
                    f.index_copy_(0, iface, _sendrecv_sum(f[iface], peer))
            return fields
        for i in range(len(self.parts) - 1):
            a, b = self.parts[i], self.parts[i + 1]
            s = fields[i][a.if_hi] + fields[i + 1][b.if_lo]
            fields[i].index_copy_(0, a.if_hi, s)
            fields[i + 1].index_copy_(0, b.if_lo, s)
        return fields


class _CompactEq3D:
    """The flat 3-D acoustic DD path's source of truth, holding nothing
    global-sized.

    A rank used to build the whole `AcousticSEM3D` (108.7 B/node on top of the
    mesh) and slice it per strip. The first compact source removed the
    equation but still materialised the material and the mass globally and
    left the propagator's interpolation tables (70 B/node, larger than the
    mesh) in place -- and on V100 sixteen ranks then hit exactly the wall eight
    did (26.8 vs 27.2 GB at 513^3): everything left in a rank's peak was
    px-invariant.

    This object stores the caller's model tensor, scalars, the GLL weights and
    the three 1-D axis coordinates -- the propagator's `Hex8Layout`, which is
    what `build_hex8_mesh` tiles into the mesh, so the mesh itself is never
    materialised on a DD rank (its arrays were the last px-invariant term,
    50.7 B/node: `ibool` 15.1, `x_e/y_e/z_e` 22.8, `x/y/z_global` 12).
    Everything a strip needs is produced per strip inside `_parts3d`:

    * material -- a strip-local trilinear interpolation on coordinates read
      from the layout's axes through the strip's own connectivity (the global
      `x_e` is `x_axis[ibool % nx]`, exactly), then the same last-writer
      scatter;
    * `c_max` -- each strip's max of its interpolant, max-reduced (exact);
    * inverse mass -- the repeated addition of one value per node, which is
      what `index_add_` does on this mesh because the GLL weights are
      symmetric, so it is bitwise the equation's;
    * geometry factors -- one element's slice tiled;
    * free-surface pins -- from the strip mesh;
    * C-PML -- 1-D axis values from `build_cpml3d`'s own formulas
      (`finish_cpml`, once `c_max` is known).

    Bitwise: every consts entry and the materials equal the sliced global
    equation (`test_sem_dd_consts_contract.py`). Gradients: a cut-plane node has
    8 element copies globally but 4 on its owning strip, so the strip's scatter
    under-counts the model gradient there by 2 -- `_CutMultiplicity` restores
    it. Topography keeps the real equation (its profile is not separable and
    its metrics are genuinely per-element-node).
    """


    @classmethod
    def build(cls, prop, vp_model):
        # the layout, not the mesh: reading `prop.mesh` here would build the
        # global mesh on every rank, which is exactly what this path removes
        lay = prop.hex8_layout()
        self = cls.__new__(cls)
        self.mesh = lay        # answers nex/ney/nez, extents, n_global, device
        mesh = lay
        no = prop.n_order
        nxg, nyg, nzg = lay.nx_global, lay.ny_global, lay.nz_global

        # -- NOTHING global-sized is built here. The material, the mass, the
        #    geometry factors and the free-surface pins are produced per STRIP
        #    inside _parts3d from what is stored below; what remains global is
        #    the model itself (4 B/node, the caller's tensor) and scalars.
        #    Measured before this: the build's transients alone peaked at
        #    85 B/node and the prop's interpolation tables cost 70 B/node,
        #    neither shrinking with px -- 16 ranks hit the same wall as 8.
        self.model = vp_model                      # (nz, ny, nx) grid, as given
        self.dtype = prop.dtype
        self.dh = float(prop.dh)
        self.grid_shape = tuple(prop.shape)
        self.pad0 = list(prop._pad0)               # per axis (z, y, x)
        self.n_order = no
        self.free_surface = tuple(prop.free_surface)
        # c_max = max over ALL GLL nodes of the interpolated model. Each strip
        # computes its own max in _parts3d and they are max-reduced -- max is
        # order-independent, so the result equals the global g.max() exactly.
        self.c_max = None                          # set by _parts3d
        self.jac = float(mesh.jacobian_det)
        self.w_ref = mesh.w_ref                    # (ngll,), float32
        self._inv_half_dx = 2.0 / mesh.dx_e
        self._inv_half_dy = 2.0 / mesh.dy_e
        self._inv_half_dz = 2.0 / mesh.dz_e
        self._d_e = (float(mesh.dx_e), float(mesh.dy_e), float(mesh.dz_e))
        self._D = mesh.deriv                       # (ngll, ngll), tiny
        # 1-D axis coordinates: the mesh's x_global is a meshgrid of the
        # layout's x_axis, so `x_global.reshape(nz, ny, nx)[0, 0, :]` IS
        # x_axis bitwise, and the mesh need not exist to read it
        self._axis_coord = (lay.x_axis, lay.y_axis, lay.z_axis)
        self._dims = (nxg, nyg, nzg)
        self._prop = prop

        self._compact_axes = None
        self.cpml = None                           # set by finish_cpml
        return self

    def finish_cpml(self, c_max):
        """Build the 1-D C-PML axis values once the global c_max is known."""
        from types import SimpleNamespace
        prop, mesh, dev = self._prop, self.mesh, self.mesh.device
        self.c_max = c_max = float(c_max)
        ax_x, ax_y, ax_z = self._axis_coord
        # -- C-PML on the 1-D axes: build_cpml3d's own formulas on the same
        #    coordinates (its defaults reflection_target=1e-3, poly=2.0) --
        import numpy as np
        sides = prop._absorbing_sides()
        thick = prop.abcn_elements
        alpha_max = float(np.pi * prop.f_dominant)
        axes = {}
        for pref, coord, hi, d_e, lo_s, hi_s, db_key in (
                ("x", ax_x, mesh.x_max, mesh.dx_e, "left", "right", "dbx_dx"),
                ("y", ax_y, mesh.y_max, mesh.dy_e, "front", "back", "dby_dy"),
                ("z", ax_z, mesh.z_max, mesh.dz_e, "top", "bottom", "dbz_dz")):
            L = thick * d_e
            smax = -np.log(1e-3) * (2.0 + 1) * c_max / (2.0 * L)
            coord = coord.contiguous()
            sig, alp = _damping_profile_1d(
                coord, 0.0, float(hi), lo_s in sides, hi_s in sides,
                pml_thickness=L, sigma_max=float(smax),
                alpha_max=alpha_max, poly=2.0)
            a1, b1 = _cpml_ab_coeffs(sig, alp, float(prop.dt))
            db = np.gradient(b1.detach().cpu().numpy(),
                             coord.detach().cpu().numpy())
            axes["a_" + pref] = a1.to(self.dtype).contiguous()
            axes["b_" + pref] = b1.to(self.dtype).contiguous()
            axes[db_key] = torch.as_tensor(db, dtype=self.dtype, device=dev)
        self._compact_axes = axes
        # forward's PML-hole check needs the global profile peak; on the
        # separable profile that IS the axis peak
        self.cpml = SimpleNamespace(b_x=axes["b_x"])
        return self


class _CutMultiplicity(torch.autograd.Function):
    """Identity forward; backward scales the gradient per node.

    `vp2[ibool] = c2` is how both the equation and the strip turn the
    per-element-node interpolant into a node field, and its backward hands the
    node gradient to EVERY element copy of that node -- 8 copies for an
    interior node, and the copies are identical, so the model gradient is the
    node gradient times the multiplicity. A node on a cut plane has 8 copies
    globally but only 4 on the strip that owns it (the x-split halves its
    x-neighbours), so the strip's scatter under-counts it by exactly 2.
    Measured before this: px=2 with the source on the cut, vp gradient
    rel 0.499 against the fused path; px=3 (cuts away from the source) 4e-4.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        (scale,) = ctx.saved_tensors
        return g * scale, None


def _free_surface_local(ml, sides, dev):
    """The strip's own free-surface node list (sorted, local numbering). The
    strip holds the full top face for its x-range, and local numbering is
    monotone in global numbering, so this equals _local_subset(global)."""
    from .mesh.boundaries import free_surface_node_idx
    return free_surface_node_idx(ml, sides, dev).to(torch.int64)


class _Stamps:
    """Point sources / receivers for the 3-D drivers as element ids plus the
    Lagrange weights over that element's ``ngll^3`` nodes -- what
    :class:`~sweep.sem.propagator.PointStamp3D` carries, batched. A stamp
    lies in exactly one element, so under DD it belongs to exactly one strip:
    no cut-plane registration or ownership rules, unlike node-id receivers.
    This is how :class:`PropSEM` (grid positions between GLL nodes) reaches
    the strip drivers; DD users keep passing global node ids."""

    __slots__ = ("elem", "w")

    def __init__(self, elem, w):
        self.elem, self.w = elem, w

    @classmethod
    def of(cls, stamps, dev):
        elem = torch.tensor([int(s.elem_id) for s in stamps], dtype=torch.long, device=dev)
        w = (torch.stack([s.gather_weights() for s in stamps]).to(dev, torch.float32)
             if stamps else torch.zeros(0, 0, device=dev))
        return cls(elem, w)

    @property
    def n(self):
        return int(self.elem.numel())


def _hex8_elem_nodes(elem, nex, ney, nez, no, dev):
    """Global node ids ``(k, ngll^3)`` of elements ``elem`` on the row-major
    grid -- :func:`~sweep.sem.mesh.hex8.hex8_ibool`'s formula for a handful
    of elements, without materialising the ``(E, ngll^3)`` table."""
    ngll = no + 1
    nxg, nyg = nex * no + 1, ney * no + 1
    ex = elem % nex
    ey = (elem // nex) % ney
    ez = elem // (nex * ney)
    k = torch.arange(ngll, device=dev)
    gz = ez[:, None, None, None] * no + k[None, :, None, None]
    gy = ey[:, None, None, None] * no + k[None, None, :, None]
    gx = ex[:, None, None, None] * no + k[None, None, None, :]
    return ((gz * nyg + gy) * nxg + gx).reshape(elem.numel(), -1)


def _strip_source_pattern(src_node, nxg, nxl, x0, x1c, n_rows, dev):
    """This strip's slice of the unit pattern at GLOBAL node ``src_node`` --
    what ``pattern[gidx]`` gives, without the n_global-sized pattern ever
    existing. A source on a cut plane lands on BOTH strips, as the slice
    would (the no-exchange rule for patterns in ``_parts`` relies on it)."""
    # The n_global vector this replaces raised IndexError on EVERY rank for
    # an out-of-range node. Slicing per strip would raise only on the strip
    # whose x-range holds gx and hand every other strip a silent all-zero
    # pattern -- under dist the survivors then block on the first interface
    # exchange instead of failing. So the check is rank-independent and
    # keeps the old Python-index semantics (a negative node wraps).
    n_global = n_rows * nxg
    if not -n_global <= src_node < n_global:
        raise IndexError(f"source node {src_node} is out of bounds for "
                         f"n_global {n_global}")
    if src_node < 0:
        src_node += n_global
    v = torch.zeros(n_rows * nxl, dtype=torch.float32, device=dev)
    gx = src_node % nxg
    if x0 <= gx <= x1c:
        v[(src_node // nxg) * nxl + (gx - x0)] = 1.0
    return v


def _strip_interp_tables(src, x0, x1c):
    """Index/fraction tables for THIS strip's NODE GRID -- 1-D, per axis.

    The trilinear interpolant is separable and the strip's nodes are a regular
    (nz_g, ny_g, nx_l) grid, so the tables carry ``nz_g + ny_g + nx_l``
    numbers, not ``E * ngll^3`` of them. Per element-node they were the
    largest single term a rank held: **70 B per global node** in three int64
    index arrays and three float32 fraction arrays, the same tables the
    propagator used to hold globally.

    The coordinates are the global layout's own axes -- the mesh's ``x_e`` is
    ``x_global[ibool]`` and ``x_global`` is a meshgrid of ``x_axis``, so a
    node's coordinate IS its axis entry, bitwise -- and the clamp / floor /
    fraction arithmetic is the propagator's ``_build_grid_to_gll``, unchanged.
    So is its pad rule: the third table is, per axis, which nodes lie outside
    the model, whose gradient ``_strip_interp_c2`` drops.
    """
    from .prop_base import _outside_model
    dh, pad0, shape = src.dh, src.pad0, src.grid_shape
    lay = src.mesh
    coords = [lay.z_axis, lay.y_axis, lay.x_axis[x0:x1c + 1]]   # z, y, x
    i0s, fs, offs = [], [], []
    for ax, ce in enumerate(coords):
        n = shape[ax]
        raw = ce.reshape(-1) / dh - pad0[ax]
        offs.append(_outside_model(raw, n))
        c = raw.clamp(0, n - 1)
        i0 = c.floor().long().clamp(0, n - 2)
        i0s.append(i0)
        fs.append((c - i0).to(src.dtype))
    # resolved here, where the tables are cached, so a call never syncs on it
    return i0s, fs, (offs if any(bool(o.any()) for o in offs) else None)


def _strip_interp_c2(src, tables, dims_l):
    """Interpolate the grid model onto THIS strip's nodes, once per node.

    Bitwise what the per-element-node path produced. Every node's value is
    the same eight products accumulated in the same corner order, and the
    weight is the same ``(w_z * w_y) * w_x`` association -- the factors depend
    on one axis each, so broadcasting them multiplies the identical floats.
    The old path then did ``vp2[ibool] = c2``, a last-writer scatter over the
    ``k`` element copies of each node; all ``k`` carry the same value, so the
    node grid needs no scatter at all.

    The corner gather narrows the STRIP axis first
    (``index_select(2, ix)`` before dim 1 and dim 0). That is not a style
    choice: selecting z first would make the intermediate the full domain's
    (nz_l, ny_m, nx_m), a px-invariant transient, and the whole saving of
    this function would go with it.

    Returns ``(vp2 on the strip's local nodes, this strip's max of vp)``.
    """
    grid = src.model.to(device=src._prop.dev, dtype=src.dtype)
    (iz, iy, ix), (fz, fy, fx), offs = tables
    import itertools
    out = None
    for cz, cy, cx in itertools.product((0, 1), repeat=3):
        wz = (1.0 - fz) if cz == 0 else fz
        wy = (1.0 - fy) if cy == 0 else fy
        wx = (1.0 - fx) if cx == 0 else fx
        w = (wz[:, None, None] * wy[None, :, None]) * wx[None, None, :]
        term = w * (grid.index_select(2, ix + cx)
                        .index_select(1, iy + cy)
                        .index_select(0, iz + cz))
        out = term if out is None else out + term
    if offs is not None and out.requires_grad:
        # the pad is a constant of the model, as in PropSEM._interp_to_gll
        oz, oy, ox = offs
        pad = oz[:, None, None] | oy[None, :, None] | ox[None, None, :]
        out = torch.where(pad, out.detach(), out)
    g = out.reshape(-1)
    assert g.numel() == dims_l, (
        f"strip node grid is {g.numel()} nodes, mesh says {dims_l}")
    return g * g, g.max()


def _strip_eids(nex, ney, nez, ex0, ex1, dev):
    """This strip's element ids: whole x-runs ``[ex0, ex1)`` of every (ez, ey).

    ``e = (ez*ney + ey)*nex + ex``, so the ids are one outer product rather
    than the ``ney*nez`` separate ``torch.arange`` calls this replaces -- each
    of those was a kernel launch and a Python round trip, measured **45% of
    `_parts3d`**, and `_parts3d` runs on every forward call on every rank.
    Same ids in the same order, so nothing downstream changes. ``ney=1``
    serves the 2-D path, where ``e = ez*nex + ex``.
    """
    base = (torch.arange(nez, device=dev)[:, None] * ney
            + torch.arange(ney, device=dev)[None, :]).reshape(-1) * nex
    return (base[:, None]
            + torch.arange(ex0, ex1, device=dev)[None, :]).reshape(-1)


def _node_share_count(no, rows, cols, nyg, dims_g):
    """How many elements of the GLOBAL mesh touch each of this strip's nodes.

    ``k = 1, 2, 4 or 8``: a node is shared across an axis exactly when it is
    an element boundary (``i % n_order == 0``) that is not a domain edge. Two
    places need it and must not disagree -- the inverse mass assembles ``k``
    identical contributions, and the model gradient reaches ``k`` element
    copies of the node.
    """
    nxg, nyg_, nzg = dims_g
    iz = torch.div(rows, nyg, rounding_mode="floor")
    iy = rows % nyg
    share = lambda i, n: (((i % no) == 0) & (i > 0) & (i < n - 1))
    return share(cols, nxg), share(iy, nyg_), share(iz, nzg)


def _share_product(sx, sy, sz, dtype=torch.int64):
    """``k`` on the strip's (row, col) grid from the three per-axis flags."""
    return ((sx[None, :].to(dtype) + 1) * (sy[:, None].to(dtype) + 1)
            * (sz[:, None].to(dtype) + 1))


def _strip_inv_mass(src, rows, cols, nyg):
    """This strip's inverse mass, bitwise what the global `index_add_` gives.

    On an axis-aligned mesh every element contributes ((wx*wy)*wz)*jac at a
    node, and the GLL weights are symmetric (w[0] == w[n]), so all k
    contributions at a node are the SAME float and the assembly is a
    repeated addition of one value -- order-free. k = 2 per axis on which
    the node is an interior element boundary, else 1. Reproducing the
    repeated addition reproduces the atomics exactly; a product of 1-D
    assembled weights would be ~2 ulp off.
    """
    no = src.n_order
    w = src.w_ref
    ix = cols                                       # global x node index
    iz = torch.div(rows, nyg, rounding_mode="floor")
    iy = rows % nyg
    wx, wy, wz = w[ix % no], w[iy % no], w[iz % no]
    sx, sy, sz = _node_share_count(no, rows, cols, nyg, src._dims)
    # ((wx*wy)*wz)*jac in the same association as w3 * jacobian_det
    # element-block dims are (z, y, x), so build() computes
    # (w[jz]*w[jy])*w[jx] -- reproduce that association exactly
    v = ((wz[:, None] * wy[:, None]) * wx[None, :]) * src.jac
    k = _share_product(sx, sy, sz)                   # 1, 2, 4 or 8
    m = torch.zeros_like(v)
    acc = torch.zeros_like(v)
    for step in range(1, 9):                        # repeated float32 adds
        acc = acc + v
        m = torch.where(k == step, acc, m)
    return (1.0 / m).reshape(-1).contiguous()


def _axis_cpml_ok(mesh_g, cp_g, nxg, nyg, nzg):
    """Is this profile separable, i.e. does each axis array vary along its own
    axis only? Exactly true on an axis-aligned mesh (measured spread 0.000e+00)
    and FALSE under topography, where the node coordinates move and `b_z` picks
    up a 1.2e-2 relative spread across x and y. Checked, not assumed."""
    for name, ax in (("b_x", 2), ("dbx_dx", 2), ("b_y", 1), ("dby_dy", 1),
                     ("b_z", 0), ("dbz_dz", 0), ("a_x", 2), ("a_y", 1),
                     ("a_z", 0)):
        v = getattr(cp_g, name, None)
        if v is None or v.numel() != nzg * nyg * nxg:
            return False
        w = v.reshape(nzg, nyg, nxg)
        red = [i for i in range(3) if i != ax]
        if float((w.amax(dim=red) - w.amin(dim=red)).abs().max()) != 0.0:
            return False
    return True


def _axis_profiles(cp_g, nxg, nyg, nzg):
    """The nine C-PML arrays reduced to their 1-D axis values.

    Each is stored per NODE (n_global floats) but carries only nx+ny+nz
    independent numbers -- measured 6.3 MiB of storage for 4.6 KiB of
    information at 65^3, a 1370x redundancy, and the C-PML group is 36.0 of the
    158.5 B/node a rank holds for the global equation. Reducing here lets a
    strip materialise its own block without the global one ever existing.
    """
    out = {}
    for name, ax in (("a_x", 2), ("b_x", 2), ("dbx_dx", 2),
                     ("a_y", 1), ("b_y", 1), ("dby_dy", 1),
                     ("a_z", 0), ("b_z", 0), ("dbz_dz", 0)):
        w = getattr(cp_g, name).reshape(nzg, nyg, nxg)
        out[name] = (w[0, 0, :] if ax == 2 else
                     w[0, :, 0] if ax == 1 else w[:, 0, 0]).contiguous()
    return out


def _strip_cpml_from_axes(axes, rows, cols, nyg):
    """Materialise a strip's C-PML block from the 1-D axis values.

    ``rows`` indexes the widened (z, y) axis and ``cols`` the strip's x nodes,
    the same pair the slicing path gathers with, so the result is bit-identical
    to ``global_array.reshape(-1)[rows[:, None] * nxg + cols]`` -- and the
    global array never has to exist. Pinned by
    ``test_strip_cpml_matches_the_sliced_global_one``.
    """
    iz, iy = torch.div(rows, nyg, rounding_mode="floor"), rows % nyg
    n_r, n_c = rows.numel(), cols.numel()
    out = {}
    for name in ("a_x", "b_x", "dbx_dx"):
        out[name] = axes[name][cols].expand(n_r, n_c).reshape(-1).contiguous()
    for name in ("a_y", "b_y", "dby_dy"):
        out[name] = axes[name][iy][:, None].expand(n_r, n_c).reshape(-1).contiguous()
    for name in ("a_z", "b_z", "dbz_dz"):
        out[name] = axes[name][iz][:, None].expand(n_r, n_c).reshape(-1).contiguous()
    return out


def _assert_strip_ibool(ml, mesh_g, eids, gidx):
    """The strip mesh is REBUILT (for its local node numbering) while every
    physical quantity is SLICED from the global build. That only composes if
    the rebuilt connectivity agrees with the global one restricted to this
    strip -- i.e. if global node ``gidx[j]`` is exactly the strip's node ``j``.
    Rebuilding with a different element family, or a mesh whose numbering
    depends on its coordinates, would break it silently and corrupt every
    sliced field at once, so it is checked instead of documented.

    With a ``Hex8Layout`` in place of the global mesh there is no global
    ``ibool`` to compare against: the reference is then the numbering FORMULA
    the layout stands for (``hex8_ibool``), evaluated on this strip's elements
    and compared in GLOBAL numbering through ``gidx`` -- nothing n_global-sized
    is built, which is the point of the layout."""
    from .mesh.hex8 import Hex8Layout
    if isinstance(mesh_g, Hex8Layout):
        no, nex, ney = mesh_g.n_order, mesh_g.nex, mesh_g.ney
        nxg, nyg = mesh_g.nx_global, mesh_g.ny_global
        got = ml.ibool.reshape(ml.n_elements, -1)
        e = eids.to(got.device)
        ez = torch.div(e, nex * ney, rounding_mode="floor")
        ey = torch.div(e, nex, rounding_mode="floor") % ney
        ex = e % nex
        k = torch.arange(no + 1, device=got.device)
        gz = ez[:, None] * no + k[None, :]
        gy = ey[:, None] * no + k[None, :]
        gx = ex[:, None] * no + k[None, :]
        want = ((gz[:, :, None, None] * nyg + gy[:, None, :, None]) * nxg
                + gx[:, None, None, :]).reshape(e.numel(), -1)
        have = gidx.to(got.device)[got]
        if not torch.equal(want, have):
            bad = int((want != have).sum())
            raise AssertionError(
                f"rebuilt strip ibool disagrees with the layout's numbering "
                f"formula in {bad} of {want.numel()} entries -- the strip's "
                "local numbering composed with gidx is not the global "
                "row-major numbering, so every gathered field is misaligned")
        return
    ng = int(mesh_g.n_global)
    g2l = torch.full((ng,), -1, dtype=torch.long, device=gidx.device)
    g2l[gidx] = torch.arange(gidx.numel(), device=gidx.device)
    want = g2l[mesh_g.ibool.reshape(mesh_g.n_elements, -1)[eids]]
    got = ml.ibool.reshape(ml.n_elements, -1).to(want.device, want.dtype)
    if int((want < 0).sum()):
        raise AssertionError(
            "strip elements reference global nodes outside the strip slice")
    if not torch.equal(want, got):
        bad = int((want != got).sum())
        raise AssertionError(
            f"rebuilt strip ibool disagrees with the sliced global one in "
            f"{bad} of {want.numel()} entries -- the rebuild and the slice "
            "are not the same mesh, so every sliced field is misaligned")


def _band_nodes(mesh, *profiles, mode="full"):
    """This strip's C-PML band: every node of every element that touches a
    node with a non-zero profile coefficient (``mode='full'``), or only the
    non-PML nodes of that set -- the interface LINE (``mode='rim'``, 7-12x
    fewer nodes, exact from a few model pixels in; see ``cpml_node_masks``).
    ``profiles`` is however many per-axis coefficients the equation has (2-D:
    b_x, b_z, dbx, dbz; 3-D adds the y pair).

    The mode is read ONCE per decomposition from the propagator, so the strips
    cannot disagree about it -- which the composition argument below needs just
    as much as it needs the sliced profile.

    What makes boundary saving compose with the decomposition is NOT that a
    cut avoids the band -- it usually does not. The rim reaches a cut through
    the OTHER axes (the bottom band in 2-D, the y and bottom bands in 3-D),
    and on test-sized grids it swallows most of the domain: 90% of nodes in
    2-D with 17 of 25 cut-column nodes in-band, 100% in 3-D. What holds is
    that the profile is sliced from ONE global build, so the two strips
    sharing a cut compute IDENTICAL membership for it: every shared node is
    either restored on both sides from rings holding the same (exchanged)
    forward values, or recomputed on both sides by the same pointwise march.
    Never one of each -- which is what a per-strip profile rebuild would
    cause, and why the sliced-profile assertion guards this too."""
    ng = mesh.n_global
    dev = profiles[0].device
    active = torch.zeros(ng, dtype=torch.bool, device=dev)
    for v in profiles:
        active |= v != 0
    from .boundary_saving import band_from_active
    band, _ = band_from_active(
        active, mesh.ibool.reshape(mesh.n_elements, -1).to(dev), mode)
    return torch.nonzero(band).reshape(-1).contiguous()


def _patch_dup_grad_syn(parts, gs, nt, stride=1):
    """Cut-column receivers are REGISTERED on both strips but the loss only
    reads the OWNED copy, so autograd hands the duplicate row a zero
    gradient. Copy the owner's row over it -- the adjoint injection must be
    identical on both copies of a shared DOF, or the interface adjoint
    diverges (measured on the acoustic path: vp gradient off by rel 0.94).
    ``stride`` is the number of record tensors per part (1 acoustic / 2
    elastic)."""
    try:
        import torch.distributed as dist
        dd = dist.is_available() and dist.is_initialized()
    except Exception:
        dd = False
    if dd:
        import torch.distributed as dist
        p, r = parts[0], dist.get_rank()
        for k in range(stride):
            g = gs[k]
            ops, buf = [], None
            if p.own_lo_rows.numel():
                ops.append(dist.P2POp(dist.isend,
                                      g[p.own_lo_rows].contiguous(), r - 1))
            if p.dup_hi_rows.numel():
                buf = torch.empty(p.dup_hi_rows.numel(), nt,
                                  dtype=torch.float32, device=g.device)
                ops.append(dist.P2POp(dist.irecv, buf, r + 1))
            if ops:
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
            if buf is not None:
                g.index_copy_(0, p.dup_hi_rows, buf)
        return gs
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if not a.dup_hi_rows.numel():
            continue
        for k in range(stride):
            gs[stride * i + k].index_copy_(0, a.dup_hi_rows,
                                           gs[stride * (i + 1) + k][b.own_lo_rows])
    return gs


def _ifc_subset(eq_g, ex0, ex1, no, nxg, gidx, dev):
    """This strip's interface quadrature points, in LOCAL numbering.

    A point belongs to the strip that owns its FACE (faces sit inside one
    element column, exactly like elements), so a shared node on a cut column
    collects a partial sum on each side and the column exchange completes it
    -- the same rule the element assembly follows. Slicing by the point's own
    column instead would double-count the cut node.
    """
    ifc = eq_g.ifc
    nodes = ifc.node_idx.reshape(ifc.node_idx.shape[0], -1)   # (n_face, ngll)
    face_col0 = (nodes.min(dim=1).values % nxg) // no          # element column
    keep = torch.nonzero((face_col0 >= ex0) & (face_col0 < ex1)).reshape(-1)
    if keep.numel() == 0:
        z = torch.zeros(0, device=dev)
        return (z.to(torch.int64), z.float(), z.float(), z.float())
    ng_max = int(gidx.max()) + 1
    lut = torch.full((max(ng_max, int(nodes.max()) + 1),), -1,
                     dtype=torch.int64, device=gidx.device)
    lut[gidx] = torch.arange(gidx.numel(), device=gidx.device)
    sel_nodes = nodes[keep].reshape(-1).to(gidx.device)
    loc = lut[sel_nodes]
    assert int((loc < 0).sum()) == 0, "interface face crosses a strip boundary"
    flat = lambda v: v.reshape(v.shape[0], -1)[keep].reshape(-1)
    return (loc.contiguous().to(dev),
            flat(ifc.weight).to(dev, torch.float32).contiguous(),
            flat(ifc.nx).to(dev, torch.float32).contiguous(),
            flat(ifc.nz).to(dev, torch.float32).contiguous())


def _pin_mask_global(eq_g):
    """(n_global,) 0/1 rigid-wall mask of the GLOBAL elastic C-PML: 0 on the
    outer PML edge where u is pinned. Sliced per strip like every other
    per-node quantity."""
    ng = eq_g.mesh.n_global
    pin = torch.ones(ng, dtype=torch.float32,
                     device=eq_g.mesh.ibool.device)
    idx = eq_g.cpml.outer_edge_idx
    if idx.numel():
        pin[idx.to(pin.device)] = 0.0
    return pin


def _pin_band_global(eq_g, node_in_pml=None, mode="full"):
    """(n_global,) 0/1 marker of the GLOBAL elastic band nodes (the ring the
    reverse march cannot recompute), computed on the GLOBAL mesh and sliced
    per strip. ``node_in_pml`` overrides ``eq_g.cpml.node_in_pml`` for the
    coupled path, whose ``cpml`` is an (acoustic, elastic) pair."""
    ng = eq_g.mesh.n_global
    v = torch.zeros(ng, dtype=torch.float32, device=eq_g.mesh.ibool.device)
    nodes = eq_g.cpml.node_in_pml if node_in_pml is None else node_in_pml
    mesh = eq_g.mesh
    active = torch.zeros(ng, dtype=torch.bool, device=v.device)
    nodes = nodes.to(v.device)
    if nodes.dtype == torch.bool and nodes.numel() == ng:
        active |= nodes
    else:                                    # element-shaped marker
        active[mesh.ibool[nodes != 0]] = True
    from .boundary_saving import band_from_active
    band, _ = band_from_active(
        active, mesh.ibool.reshape(mesh.n_elements, -1), mode)
    v[band] = 1.0
    return v


def _elastic_band_mask(ml, eq_g, eids, node_in_pml=None):
    """This strip's per-element "touches the C-PML band" flag, in STRIP order.

    Sliced from the global mesh like every other per-element quantity, so two
    strips cannot disagree about an element (they never share one anyway --
    elements are partitioned, only nodes are shared)."""
    mesh = eq_g.mesh
    nodes = (eq_g.cpml.node_in_pml if node_in_pml is None
             else node_in_pml).to(mesh.ibool.device)
    ng = mesh.n_global
    active = torch.zeros(ng, dtype=torch.bool, device=mesh.ibool.device)
    if nodes.dtype == torch.bool and nodes.numel() == ng:
        active |= nodes
    else:
        active[mesh.ibool[nodes != 0]] = True
    elem_nodes = mesh.ibool.reshape(mesh.n_elements, -1)
    return active[elem_nodes].any(1)[eids]            # global -> strip order


def _elastic_band_elems(ml, eq_g, eids, node_in_pml=None):
    """This strip's band ELEMENTS (local indices) of the elastic ring.

    No ``mode``: element rings are exempt from ``bs_band_mode='rim'`` and always
    cover the whole band. The six C-PML memory fields live only inside it, and
    zeroing them corrupts the adjoint entering the INTERIOR once the band is
    strongly excited, not just the band gradient (see ``SEMReconState``)."""
    return torch.nonzero(_elastic_band_mask(ml, eq_g, eids, node_in_pml)
                         ).reshape(-1).to(device="cuda", dtype=torch.int64).contiguous()


def _local_subset(global_idx, gidx, dev):
    """Map GLOBAL node indices onto this strip's local numbering, dropping
    the ones it does not hold. Used for the rigid-wall pin list: a global
    outer-edge node on a cut column exists on both strips and both pin it
    to zero, which is idempotent."""
    if global_idx.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=dev)
    ng_max = int(gidx.max()) + 1
    lut = torch.full((max(ng_max, int(global_idx.max()) + 1),), -1,
                     dtype=torch.int64, device=gidx.device)
    lut[gidx] = torch.arange(gidx.numel(), device=gidx.device)
    loc = lut[global_idx.to(gidx.device)]
    return loc[loc >= 0].contiguous().to(dev)


# --------------------------------------------------------------------------- #
# strips: what every equation family shares. A strip is the same object for
# every family -- its element ids, node columns, cut planes, ownership mask
# and receiver registration depend on the mesh dims and the topology only --
# and the families differ solely in the consts dict and the material tuple
# they hand the C entries. So _parts builds the strips ONCE and hands each to
# a per-family consts builder.
# --------------------------------------------------------------------------- #
class _StripCtx:
    """What one ``_parts`` call shares across its strips: the global dims and
    numbering, the devices, the source-pattern form and the rank set. Built
    once per call so the strips cannot disagree on any of it -- the band mode
    in particular: both sides of a cut must give the shared nodes the SAME
    band membership (see ``_band_nodes``)."""

    __slots__ = ("dd", "eq", "p", "mesh", "d3", "compact", "no", "bmode",
                 "nex", "ney", "nez", "nxg", "nyg", "nzg", "dev", "mdev",
                 "pattern", "src_node", "slots", "rows", "ranks", "axes", "vp2")

    def __init__(self, dd, eq_g, pattern_g):
        if torch.is_tensor(pattern_g):
            pattern_g = (pattern_g,)          # the acoustic single-field form
        self.dd, self.eq, self.p, self.mesh = dd, eq_g, dd.prop, eq_g.mesh
        self.d3 = dd.fam.ndim == 3
        self.compact = isinstance(eq_g, _CompactEq3D)
        no = self.no = self.p.n_order
        self.bmode = getattr(self.p, "bs_band_mode", "full")
        mesh = self.mesh
        self.nex, self.nez = mesh.nex, mesh.nez
        # 2-D is the 3-D numbering with a single y node -- e = ez*nex + ex,
        # g = iz*nxg + ix -- so ney = 1 and nyg = 1 (NOT no + 1)
        self.ney = mesh.ney if self.d3 else 1
        self.nxg, self.nzg = self.nex * no + 1, self.nez * no + 1
        self.nyg = self.ney * no + 1 if self.d3 else 1
        self.dev = torch.device("cuda")
        # the compact source carries a layout (no arrays, so no ibool to
        # take a device from); an equation carries the mesh
        self.mdev = mesh.device if self.compact else mesh.ibool.device
        self.pattern = pattern_g
        # a bare node index means "slice the unit pattern per strip": the
        # n_global-sized pattern is then never built (the compact 3-D form)
        self.src_node = pattern_g if isinstance(pattern_g, int) else None
        if self.src_node is not None and not self.d3:
            raise TypeError("a bare source node index is the 3-D compact "
                            "form; the 2-D strips take (n_global,) patterns")
        self.slots = (no + 1) ** (3 if self.d3 else 2)   # receiver stamp width
        self.rows = torch.arange(self.nzg * self.nyg, device=self.dev)
        self.ranks = [dd.topo.rank] if dd._dist else range(dd.topo.px)
        self.axes = None              # 3-D C-PML axis tables (_prep_ac3d)
        self.vp2 = {}                 # compact: rank -> strip vp^2 (pre-pass)

    def strip_mesh(self, nex_l):
        """The strip mesh, rebuilt for its LOCAL numbering only and cached
        (``_strip_mesh``; the kwargs ARE the cache key)."""
        mesh, no = self.mesh, self.no
        x_max = float(mesh.x_max) * nex_l / self.nex
        if not self.d3:
            return self.dd._strip_mesh(
                build_axis_aligned_quad9,
                nex=nex_l, nez=self.nez, x_max=x_max,
                z_max=float(mesh.z_max), n_order=no,
                dtype=mesh.x_global.dtype, device=mesh.ibool.device)
        from .mesh.hex8 import Hex8Mesh, build_hex8_mesh
        from .mesh.hex27 import build_axis_aligned_hex27
        # rebuild with the GLOBAL mesh's OWN element family: the GLL
        # numbering happens to coincide for axis-aligned hex8 and hex27, and
        # relying on that coincidence is how a mesh-family change would
        # silently corrupt ibool. The compact source is hex8 by construction.
        build = (build_hex8_mesh
                 if (self.compact or isinstance(mesh, Hex8Mesh))
                 else build_axis_aligned_hex27)
        kw = dict(nex=nex_l, ney=self.ney, nez=self.nez, x_max=x_max,
                  y_max=float(mesh.y_max), z_max=float(mesh.z_max),
                  n_order=no, dtype=torch.float32, device=self.mdev)
        if build is build_hex8_mesh:
            # connectivity and counts are all a 3-D strip reads of its mesh;
            # the coordinate arrays would be 35/px B/node of ballast, and an
            # int64 ibool a further 15.5/px next to the int32 copy the
            # kernels take (_ibool_i32 then casts nothing and shares storage)
            kw["geometry"] = False
            kw["ibool32"] = True
        return self.dd._strip_mesh(build, **kw)


@dataclass
class _Strip:
    """One strip before any equation family has touched it: its extent, its
    rebuilt mesh, the maps into the global build, the cut planes with their
    ownership, and the receiver registration -- everything a consts builder
    gathers with, so the families cannot drift apart on any of it."""

    r: int
    t: StripTopology
    ex0: int
    ex1: int
    nxl: int                      # local x nodes
    x0: int                       # global x node columns held: CLOSED [x0, x1c]
    x1c: int
    ml: object                    # the strip mesh
    eids: torch.Tensor            # global element ids, strip order (mdev)
    cols: torch.Tensor            # global x node columns (cuda)
    gidx: torch.Tensor            # local node -> global node (cuda)
    gm: torch.Tensor              # gidx on the global arrays' device
    if_lo: torch.Tensor | None    # local ids of the cut planes
    if_hi: torch.Tensor | None
    own: torch.Tensor             # 0/1 mask, zero on if_hi (right owns a cut)
    reg: torch.Tensor             # registered receivers (global record rows)
    own_rows: torch.Tensor
    dup_hi_rows: torch.Tensor
    own_lo_rows: torch.Tensor
    rec_G: torch.Tensor
    rec_W: torch.Tensor
    n_rec: int
    _on: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self._on[self.gidx.device] = self.gidx
        self._on[self.gm.device] = self.gm

    def index(self, device):
        """``gidx`` on ``device``: a global array is gathered on ITS device (a
        low-level propagator may hold its equation on the host)."""
        g = self._on.get(device)
        if g is None:
            g = self._on[device] = self.gidx.to(device)
        return g

    # -- gathers -------------------------------------------------------- #
    def sl(self, v):
        """A global per-NODE array on this strip, contiguous fp32 on cuda."""
        from .prop_c import _f32
        return _f32(v.reshape(-1)[self.index(v.device)])

    def el(self, v):
        """A global per-ELEMENT array (leading dim E) on this strip. The
        subset is taken FIRST: on the compact source per-element arrays are
        expand views, and reshaping one wholesale would materialise (E, m)."""
        from .prop_c import _f32
        return _f32(v[self.eids].reshape(-1))

    def pattern(self, ctx, k):
        """Source field ``k`` on this strip. A bare node index slices the
        unit pattern directly: a source on a cut lands on BOTH strips, as
        the slice of the global vector would."""
        if ctx.src_node is not None:
            return _strip_source_pattern(ctx.src_node, ctx.nxg, self.nxl,
                                         self.x0, self.x1c, ctx.rows.numel(),
                                         ctx.dev)
        from .prop_c import _f32
        v = ctx.pattern[k]
        return _f32(v.reshape(-1)[self.index(v.device)])

    # -- the consts every family carries -------------------------------- #
    def rec(self):
        from .prop_c import _i64
        return {"rec_G": _i64(self.rec_G.contiguous()),
                "rec_W": self.rec_W.contiguous()}

    def tail(self, ctx):
        return {"dt": float(ctx.p.dt), "nt": ctx.p.nt, "ngll": ctx.no + 1,
                "n_elements": self.ml.n_elements, "n_global": self.ml.n_global,
                "n_rec": self.n_rec}

    def part(self, ctx, c):
        return _Part(c, (ctx.rows.numel(), ctx.nxg, self.x0, self.x1c),
                     self.if_lo, self.if_hi, self.own, self.own_rows,
                     self.reg[self.own_rows], self.dup_hi_rows,
                     self.own_lo_rows)

    def release(self):
        """Drop the index maps once the consts are built: they are n_local
        int64 each, and ``_Part`` rebuilds ``gidx`` from four ints."""
        self.gidx = self.gm = self.cols = self.eids = None
        self._on.clear()


def _build_strip(ctx, r, receivers):
    no, dev, rows = ctx.no, ctx.dev, ctx.rows
    t = StripTopology(ctx.dd.topo.px, r)
    ex0, ex1 = t.local_elems(ctx.nex)
    nex_l = ex1 - ex0
    nxl = nex_l * no + 1
    x0, x1c = ex0 * no, ex1 * no
    ml = ctx.strip_mesh(nex_l)
    eids = _strip_eids(ctx.nex, ctx.ney, ctx.nez, ex0, ex1, ctx.mdev)
    cols = torch.arange(x0, x1c + 1, device=dev)
    gidx = (rows[:, None] * ctx.nxg + cols[None, :]).reshape(-1)
    if ctx.d3:
        # the rebuild is trusted for its numbering and every physical field
        # is SLICED through gidx -- that composes only if the two agree, so
        # it is checked before anything reads through the map
        _assert_strip_ibool(ml, ctx.mesh, eids, gidx)
    if_lo = None if t.is_edge_lo else rows * nxl
    if_hi = None if t.is_edge_hi else rows * nxl + (nxl - 1)
    own = torch.ones(rows.numel() * nxl, dtype=torch.float32, device=dev)
    if if_hi is not None:
        own[if_hi] = 0.0              # a cut belongs to the RIGHT strip
    recv = _strip_receivers(ctx, t, ml, ex0, ex1, receivers)
    return _Strip(r, t, ex0, ex1, nxl, x0, x1c, ml, eids, cols, gidx,
                  gidx.to(ctx.mdev), if_lo, if_hi, own, *recv)


def _strip_receivers(ctx, t, ml, ex0, ex1, receivers):
    """Registration vs ownership (see ``_Part``): node-id receivers register
    over the CLOSED column range [x0, x1c] and are owned half-open (the
    right strip owns a cut, matching ``own``); a point stamp lies in ONE
    element, so for stamps registration IS ownership and no cut rule
    applies."""
    dev, m, no = ctx.dev, ctx.slots, ctx.no
    nex_l, nxl = ex1 - ex0, (ex1 - ex0) * no + 1
    x0, x1c = ex0 * no, ex1 * no
    if isinstance(receivers, _Stamps):
        if not ctx.d3:
            raise TypeError("stamped receivers: 3-D acoustic only")
        ex_r = receivers.elem % ctx.nex
        reg = torch.nonzero((ex_r >= ex0) & (ex_r < ex1)).reshape(-1)
        own_rows = torch.arange(reg.numel(), device=dev)
        none = torch.zeros(0, dtype=torch.long, device=dev)
        e_loc = (receivers.elem[reg] // ctx.nex) * nex_l + (ex_r[reg] - ex0)
        rec_G = ml.ibool.reshape(ml.n_elements, m)[
            e_loc.to(ml.ibool.device)].to(dev, torch.long)
        rec_W = receivers.w[reg].contiguous()
        return reg, own_rows, none, none, rec_G, rec_W, int(reg.numel())
    gx = receivers % ctx.nxg
    reg = torch.nonzero((gx >= x0) & (gx <= x1c)).reshape(-1)
    gxr = gx[reg]
    is_own = (gxr < x1c) | torch.tensor(t.is_edge_hi, device=dev)
    own_rows = torch.nonzero(is_own).reshape(-1)
    dup_hi_rows = torch.nonzero(~is_own).reshape(-1)
    own_lo_rows = torch.nonzero(is_own & (gxr == x0)
                                & torch.tensor(not t.is_edge_lo,
                                               device=dev)).reshape(-1)
    lrec = (receivers[reg] // ctx.nxg) * nxl + (gxr - x0)
    rec_G = lrec[:, None].expand(-1, m) if lrec.numel() else \
        torch.zeros(0, m, dtype=torch.long, device=dev)
    rec_W = torch.zeros(max(int(lrec.numel()), 0), m,
                        dtype=torch.float32, device=dev)
    if lrec.numel():
        rec_W[:, 0] = 1.0             # a one-node stamp
    return reg, own_rows, dup_hi_rows, own_lo_rows, rec_G, rec_W, \
        int(lrec.numel())


def _band_support(c, axes):
    """Each memory field rings on its own axis's support (it is exactly zero
    elsewhere); under rim every one of these is empty, correctly."""
    from .boundary_saving import band_support_idx
    for ax in axes:
        c[f"band_idx_{ax}"] = band_support_idx(c["band_idx"], c[f"b_{ax}"])




def _elastic_cpml_consts(c, cp, el):
    """The element-local elastic C-PML coefficients, (E, ngll, ngll) each: the
    element subset is the whole slicing story -- internal cut ELEMENTS carry
    no PML because the global profile has none there."""
    for nm in EL_CPML_FIELDS:
        c[nm] = el(getattr(cp, nm))
    for nm in EL_CPML_TRIPLES:
        triple = getattr(cp, nm)
        for j in range(3):
            c[f"{nm}_{j}"] = el(triple[j])


class SEMModelParallel:
    """DD wrapper over the impl='c' SEM acoustic C-PML propagator.

    ``prop`` is a built :class:`~sweep.sem.prop_base.PropSEM` used as the
    CONFIG CARRIER (mirroring FD ``ModelParallel(prop, mesh)``); the compute
    runs on the stepped CUDA entry points regardless of ``prop``'s impl.
    Scope: ``AcousticPotentialSEM2D``, ``ElasticSEM2DDeformed``,
    ``AcousticElasticSEM2D``, ``AcousticSEM3D`` and ``AcousticSEM3DDeformed``
    -- all C-PML, one shot per call, records at global-node receivers. The
    equation whitelist is exact (``is``, not ``issubclass``) on purpose: each
    family needs its own stepped entry points and consts builder, so a
    near-miss must fail loudly rather than fall through to a sibling's path.

    The two 3-D acoustic families share ``_parts3d`` and the same stepped
    kernels -- the deformed one only swaps which geometry the strip consts
    carry (per element-node ``w_J`` plus the nine inverse-Jacobian metrics,
    instead of the flat mesh's per-thread ``w_jac`` and ``1/2h`` scalars).
    That works because the strip mesh is rebuilt for its ibool TOPOLOGY only
    and every geometric quantity is sliced from the global build -- the same
    rule that keeps the C-PML profile intact across a cut.
    """

    @classmethod
    def for_equation(cls, eq, dt, nt, topo=None):
        """A wrapper over a BUILT 3-D acoustic equation, for callers that
        hold one and no PropSEM (the low-level propagator): the config the
        strips need is read off the equation, and there is no model to
        resolve, so drive it with :meth:`run`. One strip unless ``topo``
        says otherwise."""
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            equation=type(eq), n_order=int(eq.mesh.n_order), dt=dt, nt=nt,
            abcn_elements=int(getattr(eq, "cpml", None) is not None),
            bs_band_mode=getattr(eq, "bs_band_mode", "full"), shape=None)
        return cls(cfg, topo or StripTopology(1))

    def __init__(self, prop, topo: StripTopology):
        if prop.equation not in _FAMILIES:
            raise NotImplementedError(
                "SEM DD covers "
                + ", ".join(e.__name__ for e in _FAMILIES)
                + f" (all C-PML); got {prop.equation.__name__}")
        self.fam = _FAMILIES[prop.equation]
        if prop.abcn_elements <= 0:
            raise NotImplementedError("SEM DD phase 1 expects a C-PML rim "
                                      "(abcn > 0)")
        if not torch.cuda.is_available():
            raise RuntimeError("SEM DD runs the impl='c' stepped loop: CUDA "
                               "is required")
        self.prop = prop
        self.topo = topo
        # px == 1 has no cuts and therefore no exchanges: it is a plain
        # local object even inside a torchrun process (rank 0 uses exactly
        # this to build the fused single-domain reference).
        self._dist = False
        if topo.px > 1:
            try:
                import torch.distributed as dist
                self._dist = dist.is_available() and dist.is_initialized()
            except Exception:
                pass
        if self._dist:
            import torch.distributed as dist
            if dist.get_world_size() != topo.px:
                raise ValueError(
                    f"world_size {dist.get_world_size()} != px {topo.px}")
            if dist.get_rank() != topo.rank:
                raise ValueError("topo.rank must equal dist.get_rank()")

    def _strip_mesh(self, build, **kw):
        """Build one strip's mesh, once.

        Measured: rebuilding it was **99.7%** of `_parts*` (10.26 of 10.29 ms at
        nex=24, px=2), and `_parts*` runs on every forward call on every rank.
        Nothing else in there was close -- the C-PML slices are 2%, the source
        pattern rounds to 0, `_band_nodes` 4% and the ibool assertion 2%.

        The strip mesh depends only on its element counts, extents, order,
        dtype and device -- not on the model, the source or the receivers -- so
        unlike the C-PML profile (keyed on `c_max`, so it rebuilds when the
        model moves) this one hits on every call after the first, in every
        usage pattern. It is only read afterwards: `_ibool_i32`,
        `_assert_strip_ibool` and the element-count fields never write to it.
        """
        key = (build.__name__,) + tuple(sorted(
            (k, str(v) if isinstance(v, (torch.device, torch.dtype)) else v)
            for k, v in kw.items()))
        cache = getattr(self, "_mesh_cache", None)
        if cache is None:
            cache = self._mesh_cache = {}
        ml = cache.get(key)
        if ml is None:
            ml = cache[key] = build(**kw)
        return ml

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _parts(self, eq_g, pattern_g, receivers):
        """The strips of ``eq_g`` for this rank set, with their consts and the
        differentiable materials (``mats[i*k:(i+1)*k]`` pairs with
        ``parts[i]``, ``k`` = 1 / 3 / 5 / 8 for 3-D / acoustic / elastic /
        fluid-solid).

        ``pattern_g``: a tuple of (n_global,) source patterns in source-field
        order, a bare tensor as the single-field form, or (3-D) a bare
        GLOBAL node index, in which case each strip slices its own unit
        pattern. ``receivers``: global node ids, or a :class:`_Stamps` (3-D).
        """
        ctx = _StripCtx(self, eq_g, pattern_g)
        strips = [_build_strip(ctx, r, receivers) for r in ctx.ranks]
        if self.fam.ndim == 3:
            self._prep_ac3d(ctx, strips)   # needs ALL strips: c_max is a reduce
        consts = getattr(self, self.fam.consts)
        out, mats = [], []
        for s in strips:
            c, mat = consts(ctx, s)
            out.append(s.part(ctx, c))
            mats += list(mat)
            s.release()
        # NO setup exchange for the source pattern: it is a GLOBAL slice, so
        # both copies of a cut node already carry the complete value. (An
        # interface sum here DOUBLES a source sitting exactly on a cut --
        # forward off by 100%, caught by the src-on-cut falsifier. A sum
        # would only be right if each strip assembled its own point stamps,
        # which are element-local and single-sided.)
        return out, mats

    def _parts3d(self, eq_g, pattern_g, receivers):
        """The 3-D name: ``run`` dispatches on it and a test wraps it on
        the instance."""
        return self._parts(eq_g, pattern_g, receivers)

    def _cache(self, name):
        d = getattr(self, name, None)
        if d is None:
            d = {}
            setattr(self, name, d)
        return d

    # -- 3-D acoustic (flat / deformed / compact) ------------------------ #
    @staticmethod
    def _global_yz_planes(ctx):
        """``{'y': mask, 'z': mask}`` of the planes the GLOBAL profile damps,
        or ``None`` when the strips cannot disagree.

        They cannot when the profile is separable (``ctx.axes``): ``b_y`` is
        then a function of ``iy`` alone, so every strip reads the same planes.
        Under topography it is not, and they can -- see ``cpml_slab_tables``.
        """
        if ctx.axes is not None:
            return None
        from .slab import plane_mask
        cp, dev = ctx.eq.cpml, ctx.dev
        dims = (ctx.nxg, ctx.nyg, ctx.nzg)
        return {ax: plane_mask(getattr(cp, f"b_{ax}").to(dev), ax, dims,
                               getattr(cp, f"db{ax}_d{ax}").to(dev))
                for ax in ("y", "z")}

    def _prep_ac3d(self, ctx, strips):
        """The once-per-call part of the 3-D path: the C-PML axis tables,
        and on the compact source the material pre-pass that yields c_max.

        The nine C-PML arrays are stored per NODE but vary along one axis
        each on an axis-aligned mesh (36.0 of the 158.5 B/node a rank held
        for the global equation), so a strip materialises its block from 1-D
        tables and the global one never exists. Separability is CHECKED,
        not assumed: under topography the nodes move and it is false, and
        that case slices per node instead.
        """
        eq_g, dev = ctx.eq, ctx.dev
        if not ctx.compact:
            cp_g = eq_g.cpml
            ctx.axes = ({k: v.to(dev) for k, v in
                         _axis_profiles(cp_g, ctx.nxg, ctx.nyg, ctx.nzg).items()}
                        if _axis_cpml_ok(ctx.mesh, cp_g, ctx.nxg, ctx.nyg, ctx.nzg)
                        else None)
            return
        # each strip's interpolant and max; the max-reduce over strips (and
        # over ranks) IS the global g.max(), max being order-free
        tcache, gmax = self._cache("_interp_cache"), None
        for s in strips:
            tkey = (s.r, self.topo.px, ctx.nex, ctx.ney, ctx.nez, ctx.no, eq_g.dh,
                    tuple(eq_g.pad0), eq_g.grid_shape, str(eq_g.dtype))
            tables = tcache.get(tkey)
            if tables is None:
                tables = tcache[tkey] = _strip_interp_tables(eq_g, s.x0, s.x1c)
            vp2_l, mx = _strip_interp_c2(eq_g, tables, s.ml.n_global)
            ctx.vp2[s.r] = vp2_l
            gmax = mx if gmax is None else torch.maximum(gmax, mx)
        if self._dist:
            import torch.distributed as dist
            gmax = gmax.clone()
            dist.all_reduce(gmax, op=dist.ReduceOp.MAX)
        c_max = float(gmax.detach())
        acache = self._cache("_axes_cache")
        akey = (c_max, ctx.nex, ctx.ney, ctx.nez, ctx.no, eq_g._d_e,
                float(eq_g._prop.dt), eq_g._prop.abcn_elements,
                float(eq_g._prop.f_dominant), tuple(eq_g._prop._absorbing_sides()))
        hit = acache.get(akey)
        if hit is None:
            # six host syncs (np.gradient) per call otherwise; the key is
            # exact on c_max, so a moved model rebuilds and a shot loop reuses
            eq_g.finish_cpml(c_max)
            hit = acache[akey] = (eq_g._compact_axes, eq_g.cpml)
        else:
            eq_g.c_max, (eq_g._compact_axes, eq_g.cpml) = c_max, hit
        ctx.axes = eq_g._compact_axes

    def _geom_ac3d(self, ctx, s):
        """The geometry consts of the three 3-D variants. The flat kernels
        index w_jac / w_j{x,y,z} by THREAD id (every element's slice is
        identical on the axis-aligned mesh), so they get ONE element's m
        values -- the (E, m) tile was 30.8 B/node no kernel ever read. The
        deformed kernels read w_J and the nine metrics per ELEMENT-NODE, so
        the flat path's per-thread invariant does not hold there and must
        not be asserted; w_j{x,y,z} and 1/2h go unused."""
        from .prop_c import _elem_thread_slice, _f32, metric_consts
        eq_g, dev, m3, E = ctx.eq, ctx.dev, ctx.slots, s.ml.n_elements
        if self.fam.deformed:
            empty = torch.empty(0, device=dev)
            return {"w_jac": s.el(eq_g._w_J).reshape(E, m3),
                    "w_jx": empty, "w_jy": empty, "w_jz": empty,
                    "ihx": 0.0, "ihy": 0.0, "ihz": 0.0,
                    # the shortcut rides with the streams: see metric_consts
                    **metric_consts(eq_g, lambda v: s.el(v).reshape(E, m3))}
        if ctx.compact:
            # the same floats AcousticSEM3D.build makes: w3 = (wx*wy)*wz,
            # then * jac and * (2/d)^2 * jac -- one element's slice
            w = eq_g.w_ref
            w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
            jac = eq_g.jac
            dx, dy, dz = eq_g._d_e
            flat = lambda t: _f32(t.reshape(-1))
            return {"w_jac": flat(w3 * jac),
                    "w_jx": flat(w3 * ((2.0 / dx) ** 2 * jac)),
                    "w_jy": flat(w3 * ((2.0 / dy) ** 2 * jac)),
                    "w_jz": flat(w3 * ((2.0 / dz) ** 2 * jac)),
                    "ihx": float(eq_g._inv_half_dx),
                    "ihy": float(eq_g._inv_half_dy),
                    "ihz": float(eq_g._inv_half_dz)}
        # the uniformity the thread indexing rests on is checked, not trusted
        thr = lambda v, nm: _f32(_elem_thread_slice(v, nm, m3))
        return {"w_jac": thr(eq_g._w_jac, "w_jac"),
                "w_jx": thr(eq_g._w_jx_noc2, "w_jx"),
                "w_jy": thr(eq_g._w_jy_noc2, "w_jy"),
                "w_jz": thr(eq_g._w_jz_noc2, "w_jz"),
                "ihx": float(eq_g._inv_half_dx),
                "ihy": float(eq_g._inv_half_dy),
                "ihz": float(eq_g._inv_half_dz)}

    def _consts_ac3d(self, ctx, s):
        from .prop_c import _f32, _ibool_i32
        from .slab import band_node_mask, cpml_slab_tables
        eq_g, dev, rows, cols, nyg = ctx.eq, ctx.dev, ctx.rows, s.cols, ctx.nyg
        compact, m3 = ctx.compact, ctx.slots
        c = {
            "ibool": _ibool_i32(s.ml, dev).reshape(s.ml.n_elements, m3),
            "D": _f32(eq_g._D),
            **self._geom_ac3d(ctx, s),
            # compact: the repeated-addition assembly, bitwise the global
            # index_add_ (symmetric GLL weights, so k identical terms)
            "inv_M": (_strip_inv_mass(eq_g, rows, cols, nyg) if compact
                      else s.sl(eq_g._inv_M)),
            "src_pattern": s.pattern(ctx, 0),
            **s.rec(),
            "dirichlet_idx": (_free_surface_local(s.ml, eq_g.free_surface, dev)
                              if compact
                              else _local_subset(eq_g.dirichlet_idx, s.gidx, dev)),
            "deformed": int(self.fam.deformed),
            **s.tail(ctx),
        }
        if ctx.axes is not None:
            c.update(_strip_cpml_from_axes(ctx.axes, rows, cols, nyg))
        else:
            for nm in ("a_x", "b_x", "dbx_dx", "a_y", "b_y", "dby_dy",
                       "a_z", "b_z", "dbz_dz"):
                c[nm] = s.sl(getattr(eq_g.cpml, nm))
        # the BS ring on the STRIP mesh from the SLICED profile: a cut may
        # sit inside it and still compose (see _band_nodes)
        c["band_idx"] = _band_nodes(s.ml, c["b_x"], c["b_y"], c["b_z"],
                                    c["dbx_dx"], c["dby_dy"], c["dbz_dz"],
                                    mode=ctx.bmode)
        _band_support(c, "xyz")
        # a middle strip has no x slab: nsx = 0. The y/z plane sets come from
        # the GLOBAL profile when it is the one being sliced, so every strip
        # agrees about a shared plane -- lap_d is exchanged on its slab and
        # the two sides of a cut must correspond entry for entry.
        c.update(cpml_slab_tables(c, self._global_yz_planes(ctx)))
        c["band_node"] = band_node_mask(c)   # the node kernels' fast-path test
        if compact:
            # The old per-element-node path handed the node gradient to every
            # element copy of the node (k of them on this strip) and
            # _CutMultiplicity supplied the factor 2 a cut was missing (8
            # copies globally, 4 here). On the node grid there is one copy,
            # so the whole factor is applied here -- and it is exactly the
            # GLOBAL share count, verified element-wise to equal the old
            # (strip scatter) x (cut correction) at px = 1, 2 and 4. Summing
            # k identical fp32 values and multiplying by k agree bitwise for
            # k in {1, 2, 4, 8}: all powers of two.
            mult = _share_product(
                *_node_share_count(ctx.no, rows, cols, nyg, eq_g._dims),
                dtype=torch.float32).reshape(-1).to("cuda")
            mat = _CutMultiplicity.apply(ctx.vp2[s.r].to("cuda", torch.float32),
                                         mult)
        else:
            mat = eq_g.vp_squared_node.reshape(-1)[s.gm].to("cuda", torch.float32)
        return c, (mat,)

    # -- 2-D acoustic ---------------------------------------------------- #
    def _consts_ac2d(self, ctx, s):
        from .prop_c import _f32, _i64, _ibool_i32
        eq_g, cp_g, dev = ctx.eq, ctx.eq.cpml, ctx.dev
        # Throwaway strip build: it supplies ONLY the element geometry (_D,
        # _w_J, _ixx, _ixz, _iex, _iez), _mask_e and dirichlet_idx. Its w_J
        # and inv_J are the axis-aligned rebuild's own floats (not bitwise
        # the global build's element subset), so they stay here. Every
        # assembled / global per-node field is SLICED from the global build
        # instead: masses need both sides of a cut, and the profile must
        # not grow PML at internal cuts.
        eq_l = AcousticPotentialSEM2D.build(
            s.ml, eq_g.vp_e[s.eids].detach(), eq_g.rho_e[s.eids].detach(),
            cpml=None, dirichlet=eq_g.dirichlet)
        prof = {nm: s.sl(getattr(cp_g, nm))
                for nm in ("a_x", "b_x", "dbx_dx", "a_z", "b_z", "dbz_dz")}
        c = {
            "ibool": _ibool_i32(s.ml, dev), "D": _f32(eq_l._D),
            "wJ": _f32(eq_l._w_J), "w1d": _f32(s.ml.w_ref),
            "ixx": _f32(eq_l._ixx), "ixz": _f32(eq_l._ixz),
            "iex": _f32(eq_l._iex), "iez": _f32(eq_l._iez),
            "mask_e": _f32(eq_l._mask_e.reshape(-1)),
            "inv_Mg": s.sl(eq_g._inv_M_geo),
            **prof,
            "src_pattern": s.pattern(ctx, 0),
            **s.rec(),
            "dirichlet_idx": (_i64(eq_l.dirichlet_idx)
                              if eq_l.dirichlet_idx.numel() else
                              torch.zeros(0, dtype=torch.int64, device=dev)),
            "band_idx": _band_nodes(s.ml, prof["b_x"], prof["b_z"],
                                    prof["dbx_dx"], prof["dbz_dz"],
                                    mode=ctx.bmode),
            **s.tail(ctx),
        }
        _band_support(c, "xz")
        # differentiable: element subset + per-node slices, their autograd
        # chains reaching the caller's model
        mat = (eq_g._inv_rho[s.eids].to("cuda", torch.float32),
               eq_g._inv_M[s.gm].to("cuda", torch.float32),
               eq_g.kappa_node[s.gm].to("cuda", torch.float32))
        return c, mat

    # -- 2-D elastic ----------------------------------------------------- #
    def _consts_el2d(self, ctx, s):
        from .prop_c import _f32, _i64, _ibool_i32
        eq_g, cp, dev = ctx.eq, ctx.eq.cpml, ctx.dev
        E, m2 = ctx.nex * ctx.nez, ctx.slots
        # throwaway strip build: supplies ONLY _D, _w_J, _ixx, _ixz, _iex,
        # _iez and _mask_e (see _consts_ac2d for why it stays)
        eq_l = ElasticSEM2DDeformed.build(
            s.ml, eq_g.vp_e[s.eids].detach(), eq_g.vs_e[s.eids].detach(),
            eq_g.rho_e[s.eids].detach(), cpml=None)
        # a profile field that is not element-shaped is passed whole
        el = lambda v: (s.el(v.reshape(E, -1)) if v.numel() == E * m2 else _f32(v))
        c = {
            "ibool": _ibool_i32(s.ml, dev), "D": _f32(eq_l._D),
            "wJ": _f32(eq_l._w_J),
            "ixx": _f32(eq_l._ixx), "ixz": _f32(eq_l._ixz),
            "iex": _f32(eq_l._iex), "iez": _f32(eq_l._iez),
            "mask_e": _f32(eq_l._mask_e.reshape(-1)),
            "pat_x": s.pattern(ctx, 0), "pat_z": s.pattern(ctx, 1),
            **s.rec(),
            "pin_mask": s.sl(_pin_mask_global(eq_g)),
            "outer_edge_idx": _local_subset(cp.outer_edge_idx, s.gidx, dev),
            # BS rings sliced from the GLOBAL band, never rebuilt on the strip
            "band_node_idx": _i64(torch.nonzero(
                s.sl(_pin_band_global(eq_g, mode=ctx.bmode)) > 0).reshape(-1)),
            "band_elem_idx": _elastic_band_elems(s.ml, eq_g, s.eids),
            "elem_band": _elastic_band_mask(s.ml, eq_g, s.eids).to(
                device="cuda", dtype=torch.int32).contiguous(),
            **s.tail(ctx),
        }
        _elastic_cpml_consts(c, cp, el)
        mat = (eq_g._lam[s.eids].to("cuda", torch.float32),
               eq_g._mu[s.eids].to("cuda", torch.float32),
               eq_g._lam2mu[s.eids].to("cuda", torch.float32),
               eq_g._inv_M[s.gm].to("cuda", torch.float32),
               eq_g._w_rho_J[s.eids].to("cuda", torch.float32))
        return c, mat

    # -- 2-D fluid-solid ------------------------------------------------- #
    def _consts_fs(self, ctx, s):
        from .prop_c import _f32, _i64, _ibool_i32
        eq_g, dev = ctx.eq, ctx.dev
        ac, el_ = eq_g.ac, eq_g.el
        cp_ac, cp_el = eq_g.cpml[0], eq_g.cpml[1]
        E = ctx.nex * ctx.nez
        # no throwaway build here: the coupled equation's geometry comes from
        # the GLOBAL element subset (the two sub-equations share the mesh);
        # a field is reshaped to (E, ...) first so a flat one still subsets
        # per element
        el = lambda v: s.el(v.reshape(E, -1))
        ig, iw, inx, inz = _ifc_subset(eq_g, s.ex0, s.ex1, ctx.no, ctx.nxg,
                                       s.gidx, dev)
        c = {
            "ibool": _ibool_i32(s.ml, dev), "D": _f32(ac._D),
            "wJ": el(ac._w_J), "w1d": _f32(s.ml.w_ref),
            "ixx": el(ac._ixx), "ixz": el(ac._ixz),
            "iex": el(ac._iex), "iez": el(ac._iez),
            "ac_mask_e": el(ac._mask_e.reshape(-1)),
            "el_mask_e": el(el_._mask_e.reshape(-1)),
            "dirichlet_idx": _local_subset(ac.dirichlet_idx, s.gidx, dev),
            "ifc_idx": ig, "ifc_w": iw, "ifc_nx": inx, "ifc_nz": inz,
            "pat_chi": s.pattern(ctx, 0),
            "pat_x": s.pattern(ctx, 1), "pat_z": s.pattern(ctx, 2),
            **s.rec(),
            "ac_inv_Mg": s.sl(ac._inv_M_geo),
            "outer_edge_idx": _local_subset(cp_el.outer_edge_idx, s.gidx, dev),
            "has_cpml": True,
            # fluid band on nodes, solid band on nodes AND elements (its
            # histories are element-local) -- all sliced from the global build
            "band_node_idx": _i64(torch.nonzero(
                s.sl(_pin_band_global(eq_g, cp_el.node_in_pml, mode=ctx.bmode)) > 0
            ).reshape(-1)),
            "band_elem_idx": _elastic_band_elems(s.ml, eq_g, s.eids,
                                                 cp_el.node_in_pml),
            "ckpt_every": int(getattr(ctx.p.equation,
                                      "bs_frame_checkpoint_every", 50)),
            **s.tail(ctx),
        }
        for nm in ("a_x", "b_x", "dbx_dx", "a_z", "b_z", "dbz_dz"):
            c[nm] = s.sl(getattr(cp_ac, nm))
        c["band_idx"] = _band_nodes(s.ml, c["b_x"], c["b_z"], c["dbx_dx"],
                                    c["dbz_dz"], mode=ctx.bmode)
        _elastic_cpml_consts(c, cp_el, el)
        mat = (ac._inv_rho[s.eids].to("cuda", torch.float32),   # _FS_MATS order
               ac._inv_M[s.gm].to("cuda", torch.float32),
               ac.kappa_node[s.gm].to("cuda", torch.float32),
               el_._lam[s.eids].to("cuda", torch.float32),
               el_._mu[s.eids].to("cuda", torch.float32),
               el_._lam2mu[s.eids].to("cuda", torch.float32),
               el_._inv_M[s.gm].to("cuda", torch.float32),
               el_._w_rho_J[s.eids].to("cuda", torch.float32))
        return c, mat

    def forward(self, wavelet, source, receivers, models):
        """One shot; ``source``/``receivers`` are GLOBAL node indices of the
        row-major global grid. Returns ``(nt, nrec)`` -- fully assembled in
        single-process mode, own-columns-only under ``dist``.

        The gradient memory is the propagator's own. ``memory=
        BoundarySaving()`` stores each strip's C-PML band ring instead of its
        save-all tape; the two memory levers compose because both strips give
        a shared cut the same band membership (see :func:`_band_nodes`). Its
        ``storage_dtype`` picks the ring's storage precision -- ``'fp32'``,
        ``'bf16'`` (2x) or ``'int8'`` (4x, one amax scale per field and step).
        The propagator's ``bs_store_dtype_elem`` overrides it for the ELEMENT
        rings, which the elastic and fluid-solid paths carry; on the elastic
        path they are 82% of the ring memory and quantise ~70x better than the
        node rings, so ``storage_dtype='fp32', bs_store_dtype_elem='int8'`` is
        the setting that actually pays there. Refused on paths that ring nodes
        only."""
        p = self.prop
        p._set_nt(wavelet)
        # Flat 3-D acoustic: build the compact source instead of the global
        # equation. Same numbers (the consts contract test pins bit-identity),
        # ~8 B/node persistent instead of 108.7 on top of the shared mesh --
        # which is both the DD serial fraction and the reason eight GPUs only
        # bought 1.42x the model size. Anything it cannot represent (per-shot
        # models, topography, 2-D, elastic, fs) falls through to the equation.
        m1 = models[0] if isinstance(models, (list, tuple)) and len(models) == 1 \
            else (models if torch.is_tensor(models) else None)
        if (self.fam.ndim == 3 and not self.fam.deformed and torch.is_tensor(m1)
                and tuple(m1.shape) == tuple(p.shape)):
            eq_g = _CompactEq3D.build(p, m1)
        else:
            eq_g, _ = p._resolve_model_batch(models)
        return self.run(eq_g, wavelet, source, receivers, p._memory,
                        p._bs_store, p._bs_store_elem)

    # ------------------------------------------------------------------ #
    # run(), in the six phases its docstring names
    # ------------------------------------------------------------------ #
    def _bs_knobs(self, memory, store, store_elem):
        """``(bs, node ring kind, element ring kind)`` -- with the three
        refusals, which must fire before anything is built."""
        from .prop_c import _band_store
        bs = memory == "boundary"
        kind, _ = _band_store(store)                  # raises on a bad name
        kind_e, _ = _band_store(store if store_elem is None else store_elem)
        if (kind or kind_e) and not bs:
            raise ValueError("bs_store_dtype needs memory='boundary'")
        if store_elem is not None and not self.fam.elem_rings:
            raise ValueError(
                "bs_store_dtype_elem only applies to rings with ELEMENT "
                "histories (elastic, fluid-solid); this path rings nodes only")
        return bs, kind, kind_e

    def _source_patterns(self, eq_g, source, dev):
        """The source in the form ``_parts*`` wants: a tuple of ``(n_global,)``
        patterns in source-field order, or -- on the compact 3-D equation --
        the bare global node id, from which each strip slices its own unit
        pattern so nothing n_global-sized is built on the rank.

        The four caller forms in one place: a ready-made tuple, point
        :class:`_Stamps`, a compact-equation node id, a plain node id.
        """
        if isinstance(source, tuple):
            return source                       # patterns built by the caller
        if isinstance(source, _Stamps):
            # point stamps (PropSEM's route): one summed pattern of Lagrange
            # weights on the global node grid -- 4 B/node, px-invariant, and
            # only the stamps' own elements are ever numbered
            if self.fam.ndim != 3:
                raise NotImplementedError("stamped sources: 3-D acoustic only")
            mesh = eq_g.mesh
            g = (_hex8_elem_nodes(source.elem, mesh.nex, mesh.ney, mesh.nez,
                                  self.prop.n_order, dev)
                 if isinstance(eq_g, _CompactEq3D)
                 else mesh.ibool[source.elem.to(mesh.ibool.device)]
                 .reshape(source.n, -1).to(dev, torch.long))
            v = torch.zeros(mesh.n_global, dtype=torch.float32, device=dev)
            v.index_add_(0, g.reshape(-1), source.w.reshape(-1))
            return (v,)
        if isinstance(eq_g, _CompactEq3D):
            return int(source)
        # A node id stands for the equation's DEFAULT unit source. A PropSEM
        # built with another source_type (a horizontal force, a moment
        # tensor) would have it replaced here without a word.
        default = getattr(self.prop, "_default_source", None)
        if default is not None and not default():
            raise NotImplementedError(
                f"a node-id source is the equation's default unit source, but "
                f"this PropSEM has source_type={self.prop.source_type}; pass "
                f"the source patterns as a tuple instead.")
        # one field per record component, the point source in src_field: the
        # acoustic potential, the elastic force (vertical), fluid-solid's chi
        ng, mdev = eq_g.mesh.n_global, eq_g.mesh.ibool.device
        pats = [torch.zeros(ng, dtype=torch.float32, device=mdev)
                for _ in range(self.fam.ncomp)]
        pats[self.fam.src_field][int(source)] = 1.0
        return tuple(pats)

    def _receiver_ids(self, receivers, dev):
        """``(receivers, nrec)``: point :class:`_Stamps` pass through, node ids
        become one cuda int64 row."""
        if isinstance(receivers, _Stamps):
            if self.fam.ndim != 3:
                raise NotImplementedError("stamped receivers: 3-D acoustic only")
            return receivers, receivers.n
        rec = torch.as_tensor(receivers, device=dev, dtype=torch.long)
        return rec, rec.numel()

    def _assert_cuts_are_pml_free(self, eq_g, parts):
        """Internal interfaces must carry no C-PML -- the mirror of FD's "PML
        hole". The profile is SLICED from the global mesh, never rebuilt; a
        rebuild grows a fresh absorbing band at every internal cut. 3-D carries
        the same node-level ``b_x``, so a cut PLANE is checked exactly like a
        2-D cut column.

        The scale is the GLOBAL profile's peak: a strip lying wholly outside
        the band has no scale of its own, and normalising a residue by a
        residue reads as 100%.

        Relative, not ``== 0.0``: on a DEFORMED mesh the profile is built from
        curved coordinates, so a node exactly on the band edge lands at float
        residue rather than a clean zero -- measured 9.4e-16 against a 0.0995
        peak, i.e. 9.4e-15 relative, where the FIRST genuinely in-band column
        is 7.7e-3 relative. Twelve orders of margin between the two, so the
        tolerance changes nothing about what this catches: a rebuilt profile
        puts a full-strength band on the cut, not a rounding residue.

        Skipped where the family has no node profile to read (elastic: its
        C-PML coefficients are element-local, so an internal cut element
        carries no PML by construction).
        """
        if not self.fam.node_cpml:
            return
        cp = getattr(eq_g, "cpml", None)
        bx = getattr(cp, "b_x", None) if cp is not None else None
        peak = max(float(bx.abs().max()) if bx is not None else 0.0, 1e-30)
        for pp in parts:
            for iface in (pp.if_lo, pp.if_hi):
                if iface is None:
                    continue
                cut = float(pp.c["b_x"][iface].abs().max())
                assert cut <= 1e-12 * peak, (
                    f"internal interface carries C-PML ({cut:.3e}, "
                    f"{cut / peak:.1e} of the global peak) -- "
                    "the profile must be sliced from the GLOBAL mesh, "
                    "never rebuilt")

    def _strip_record(self, parts, seis, nrec, dev):
        """The strips' own-row seismograms scattered into one global
        ``(nt, nrec)`` -- ``(nt, nrec, ncomp)`` where the family records more
        than one component."""
        ncomp = self.fam.ncomp
        record = torch.zeros(nrec, self.prop.nt, ncomp, dtype=torch.float32,
                             device=dev)
        for i, pp in enumerate(parts):
            if not pp.slot.numel():
                continue
            for j in range(ncomp):
                record[pp.slot, :, j] = seis[ncomp * i + j][pp.own_rows]
        out = record.transpose(0, 1)                    # (nt, nrec, ncomp)
        return out if ncomp > 1 else out[..., 0]

    def run(self, eq_g, wavelet, source, receivers, memory=None,
            bs_store_dtype=None, bs_store_dtype_elem=None):
        """One shot on a BUILT equation ``eq_g`` -- what :meth:`forward` does
        after resolving the model, and how the low-level
        :class:`~sweep.sem.prop_c.SEMCudaPropagator` (which holds a built
        equation and no PropSEM) reaches the strip driver.

        ``source`` is a global node id, a :class:`_Stamps`, or a tuple of
        ``(n_global,)`` patterns in source-field order; ``receivers`` global
        node ids or a :class:`_Stamps`. Returns ``(nt, nrec)``
        (``(nt, nrec, ncomp)`` on the elastic and fluid-solid paths).

        Refuse -> normalise -> build the strips -> check the cuts -> step ->
        gather, one method each, in that order: the refusals have to precede
        every allocation, and the cut check has to precede the first step.
        """
        bs, kind, kind_e = self._bs_knobs(memory, bs_store_dtype,
                                          bs_store_dtype_elem)
        dev = torch.device("cuda")
        wav = torch.as_tensor(wavelet, device=dev, dtype=torch.float32)
        pats = self._source_patterns(eq_g, source, dev)
        rec, nrec = self._receiver_ids(receivers, dev)
        parts, mats = (self._parts3d(eq_g, pats, rec) if self.fam.ndim == 3
                       else self._parts(eq_g, pats, rec))
        for pp in parts:
            pp.c.update(bs=bs, bs_kind=kind, bs_kind_e=kind_e)
        self._assert_cuts_are_pml_free(eq_g, parts)
        seis = self.fam.fn.apply(wav, parts, _DDExchange(parts, self._dist),
                                 bs, *mats)
        return self._strip_record(parts, seis, nrec, dev)

    __call__ = forward

    def gather_record(self, record):
        """Assemble the global record on rank 0 (dist mode); identity in
        single-process mode. Mirrors FD ``ModelParallel.gather_record``."""
        if not self._dist:
            return record
        import torch.distributed as dist
        out = [None] * dist.get_world_size()
        dist.gather_object(record.cpu(),
                           out if dist.get_rank() == 0 else None, dst=0)
        if dist.get_rank() != 0:
            return None
        full = torch.zeros_like(record.cpu())
        for part in out:
            full = full + part
        return full
