"""CUDA (``impl='c'``) propagator for sweep.sem — acoustic-potential (Phase 0/1).

Mirrors the FD ``impl='c'`` split: a compiled propagator that reuses ALL of the eager
:class:`SEMPropagator`'s host-side setup (mesh build, source/receiver Lagrange stamps,
source patterns, PML guard) and overrides only the time loop, dispatching it to the
``sweep._C.sem2d_acoustic_*`` device kernels over Python-allocated buffers (the project's
hard rule: every wavefield/output buffer is allocated here, not in the ``.cu``).

Autograd: the eager ``build()`` runs OUTSIDE the autograd Function (with vp_e/rho_e
requiring grad), so ``eq._inv_M`` / ``eq._inv_rho`` carry the model graph. The Function
takes those (cast to cuda-fp32) + the wavelet as differentiable inputs and returns their
gradients from the adjoint kernel; PyTorch then chains inv_M/inv_rho through ``build()``
back to vp_e/rho_e. fp32 only; single backward (no double-backward). CPML/elastic/3D are
later phases — this fails fast on anything it does not cover.
"""

from __future__ import annotations

import functools

import torch

from .propagator import SEMPropagator, receiver_gather
from .equations.acoustic_elastic2d import AcousticElasticSEM2D
from .equations.acoustic_potential2d import AcousticPotentialSEM2D
from .equations.acoustic3d import AcousticSEM3D
from .equations.acoustic3d_deformed import AcousticSEM3DDeformed
from .equations.elastic3d import ElasticSEM3D
from .equations.elastic3d_deformed import ElasticSEM3DDeformed
from .equations.elastic_vs3d import ElasticVSSEM3D
from .equations.elastic2d_deformed import ElasticSEM2DDeformed
from .cdrive import (AC_CPML_FIELDS, EL_CPML_FIELDS, EL_CPML_TRIPLES,
                     METRIC_NAMES)
from .drivers.acoustic2d import _SEMAcousticCpmlC
from .drivers.elastic2d import _SEMElasticCpmlC
from .drivers.fluid_solid import _SEMFluidSolidC
from .drivers.bare import (_SEMAcousticC, _SEMElasticC, _SEMElastic3DC,
                           _SEMElasticVS3DC, _VS3D_CPML)
# The autograd Functions live in sweep.sem.drivers (one module per equation
# family, filled through sweep.sem.cdrive); they are re-exported here for
# the callers and tests that import them from prop_c. The bar for moving one
# was the previous tree's input structs, field for field, plus the gate.


def sem_c_available() -> bool:
    """True when the compiled ``sweep._C`` binding with the SEM entry points is usable."""
    try:
        import sweep._C as _C
        return hasattr(_C, "sem2d_acoustic_forward")
    except Exception:
        return False


def sem_c_shot_batchable(eq) -> bool:
    """Equations whose impl='c' kernels take the shot-batch axis (B > 1).

    Today: the three 2-D C-PML paths -- acoustic-potential (the FWI
    workhorse), the coupled fluid-solid (the OBN workhorse -- ~15-20 launches
    per step made its per-shot loop the most launch-bound of all) and the
    standalone elastic. The other paths keep the per-shot loop, which is
    correct, just unamortised.
    """
    return (isinstance(eq, (AcousticPotentialSEM2D, AcousticElasticSEM2D,
                            ElasticSEM2DDeformed))
            and getattr(eq, "cpml", None) is not None)


def sem_c_model_batchable(eq) -> bool:
    """Shot-batchable equations that ALSO take per-shot materials (Bm == B).

    The fluid-solid batch shares one model by design (its Function has no Bm),
    so it is shot-batchable but not model-batchable -- per-shot models there
    fall back to the per-shot loop.
    """
    return (isinstance(eq, AcousticPotentialSEM2D)
            and getattr(eq, "cpml", None) is not None)


def _fs_batch_consts(props, memory):
    """Batched fluid-solid consts: shared geometry/receivers from ``props[0]``,
    the three source patterns stacked ``(B, ng)``, ``B``/``bs``/band keys set.
    Shared by :func:`sem_c_forward_shot_batch` and the CUDA-graph runner so the
    two batched entry points cannot drift. ``B == 1`` (a single prop) keeps the
    historical single-shot layout up to a leading length-1 pattern axis, which
    the C driver reads identically. Sets ``bs`` only (no store kinds): the
    batched coupled path has always run fp32 rings -- see _stamp_bs."""
    p0 = props[0]
    c = _stack_patterns(p0._fluid_solid_consts(), props,
                        ("pat_chi", "pat_x", "pat_z"))
    c["B"] = len(props)
    c["bs"] = memory == "boundary"
    if p0.eq.cpml is not None:             # see _forward_fluid_solid
        c.update(p0._fluid_solid_bands())
    return c


def sem_c_forward_shot_batch(props, wavelet, materials=None, memory=None,
                             bs_store_dtype=None, bs_store_dtype_elem=None):
    """Run B same-receiver shots of the acoustic C-PML path in ONE batched
    CUDA call. ``props`` are the per-shot :class:`SEMCudaPropagator`\\ s (they
    contribute only their source patterns; geometry, receivers and the C-PML
    profile come from ``props[0]`` and must be shared -- the caller guarantees
    identical receiver rows). ``wavelet`` is ``(nt,)`` shared or ``(B, nt)``
    per-shot. ``materials`` overrides the (inv_rho, inv_M, kappa) triple with
    per-shot stacks ``(B, ...)`` -- the Bm == B case; ``None`` reads the
    single shared model off the equation (Bm == 1, the degenerate
    "same model, different sources" batch). ``memory='boundary'`` runs the
    backward on the batched C-PML band ring -- (nt, B, nband) histories
    instead of the (nt, B, ng) save-all fields, which is what makes LARGE
    batches affordable in fwd+bwd. Returns ``(B, n_rec, nt)``.

    The coupled fluid-solid and standalone elastic C-PML paths batch too
    (same conditions), except the model is ALWAYS shared (``materials`` must
    be None -- one model serves every shot of the FWI inner loop) and the
    return is ``(B, n_rec, nt, ncomp)`` with the components selected and
    ordered by ``props[0].receiver_type`` (fs: chi/u_x/u_z; elastic:
    u_x/u_z), exactly like the per-shot ``forward``.
    """
    p0 = props[0]
    eq = p0.eq
    B = len(props)
    t = lambda x: x.to("cuda", torch.float32)
    wav = wavelet.to("cuda", torch.float32)
    flags = lambda: _bs_flags(memory, bs_store_dtype,
                              bs_store_dtype if bs_store_dtype_elem is None
                              else bs_store_dtype_elem)
    if isinstance(eq, ElasticSEM2DDeformed):
        if materials is not None:
            raise NotImplementedError(
                "per-shot materials are not wired for the elastic batch "
                "(its kernels always share one model across the batch)")
        c = _stack_patterns(p0._elastic_cpml_consts(), props, ("pat_x", "pat_z"))
        c.update(B=B, **flags())
        comps = _SEMElasticCpmlC.apply(
            t(eq._lam), t(eq._mu), t(eq._lam2mu), t(eq._inv_M),
            t(eq._w_rho_J), wav, c)
        return _select_components(p0, comps)
    if isinstance(eq, AcousticElasticSEM2D):
        if materials is not None:
            raise NotImplementedError(
                "per-shot materials are not wired for the fluid-solid batch "
                "(the coupled path always shares one model across the batch)")
        c = _fs_batch_consts(props, memory)
        ac, el = eq.ac, eq.el
        comps = _SEMFluidSolidC.apply(
            t(ac._inv_rho), t(ac._inv_M), t(ac.kappa_node),
            t(el._lam), t(el._mu), t(el._lam2mu), t(el._inv_M),
            t(el._w_rho_J), wav, c)
        return _select_components(p0, comps)
    c = _stack_patterns(p0._cpml_consts(), props, ("src_pattern",))
    c.update(B=B, **flags())
    if materials is None:
        c["Bm"] = 1
        inv_rho, inv_M, kappa = t(eq._inv_rho), t(eq._inv_M), t(eq.kappa_node)
    else:
        c["Bm"] = B
        inv_rho, inv_M, kappa = (t(x) for x in materials)
    return _SEMAcousticCpmlC.apply(inv_rho, inv_M, kappa, wav, c)


def _select_components(prop, comps):
    """``(B, n_rec, nt, ncomp)`` in ``receiver_type`` order -- the batched
    twin of :meth:`SEMCudaPropagator._assemble` (no (nt, nrec) transpose)."""
    recv = [sp.name for sp in prop.eq.field_specs if sp.supports_receiver]
    at = {nm: i for i, nm in enumerate(recv)}
    rtype = prop.receiver_type or recv
    return torch.stack([comps[at[nm]] for nm in rtype], -1)


def _f32(t):
    """To CUDA fp32, contiguous -- the cast every consts builder used to
    define as a local lambda (12 identical copies; i64 had 9)."""
    return t.to(device="cuda", dtype=torch.float32).contiguous()


def _i64(t):
    return t.to(device="cuda", dtype=torch.int64).contiguous()


def _memo_consts(fn):
    """Cache a constants dict on the propagator instance (returns a shallow copy
    so callers may freely add keys).

    These constants are pure geometry / acquisition -- ibool, the metric terms,
    the flattened C-PML profile, the source patterns and receiver gathers. They
    do NOT depend on the model tensors, which reach the autograd Function as
    separate arguments. Rebuilding them on every forward call was pure waste,
    and worse: the C-PML band indices go through ``torch.nonzero``, which has to
    report its output size to the host. That HOST SYNC on every call also makes
    the C++ time loop impossible to capture in a CUDA graph
    (cudaErrorStreamCaptureInvalidated).
    """
    key = "_memo_" + fn.__name__

    @functools.wraps(fn)
    def wrap(self):
        c = getattr(self, key, None)
        if c is None:
            c = fn(self)
            setattr(self, key, c)
        return dict(c)

    return wrap


# Boundary-saving ring storage precision. The ring is O(nt * nband) and is the
# dominant BS allocation, so this is the main memory lever after rim mode.
# int8 additionally needs a (n_fields, nt) fp32 amax table, which the C side
# fills on save and reads on restore.
_BAND_KIND = {None: 0, torch.float32: 0, "fp32": 0,
              torch.bfloat16: 1, "bf16": 1,
              torch.int8: 2, "int8": 2}
_BAND_DTYPE = {0: torch.float32, 1: torch.bfloat16, 2: torch.int8}


def _has_element_rings(eq):
    """Do this equation's boundary rings carry ELEMENT histories?

    Only the elastic and coupled paths do; the acoustic ones ring nodes
    only, so a separate element store dtype would be a silent no-op there
    and is refused instead.
    """
    from .equations import AcousticElasticSEM2D, ElasticSEM2DDeformed
    return isinstance(eq, (ElasticSEM2DDeformed, AcousticElasticSEM2D))


def _band_store(store):
    """``store_dtype`` -> ``(kind, torch dtype)``."""
    if store not in _BAND_KIND:
        raise ValueError(
            f"bs_store_dtype must be None/float32/bfloat16/int8; got {store!r}")
    k = _BAND_KIND[store]
    return k, _BAND_DTYPE[k]


# Ring layout of the acoustic C-PML memory fields: True stores psi_d / zeta_d
# on their own axis's support (band_idx_d), False on the union band the way
# every driver did before. A module switch rather than a user knob: it exists
# so a test can build both layouts in one process and compare the stored
# bytes -- production has no reason to ask for the larger one.
_BS_RING_SPLIT = True

# Storage of the 3-D acoustic C-PML memory fields: True keeps psi_d / zeta_d /
# a_d*psi_d (and the backward's restores) on their axis's index SLAB -- see
# sweep/sem/slab.py -- False on every node, the way the drivers did before.
# Same test-facing role as _BS_RING_SPLIT: both layouts in one process. The
# drivers fall back to the full layout by themselves wherever a save-all tape
# is taken (the tape copies full arrays) or the rings are not split.
_PSI_SLAB = True

# The 3-D save-all backward: True saves w_sum^n each forward step and reads
# it back (no forward state is rebuilt: no a*psi, no assembly, no zeta, and
# no g9 exchange under DD; the seven u/psi/zeta tapes are not taken at all),
# False keeps the seven tapes and recomputes w_sum^k as before. Same
# test-facing role as the two switches above.
_WSUM_TAPE = True

# The 2-D acoustic save-all backward, the same trade one dimension down: True
# saves w_sum^n each forward step and reads row k back, so the reverse step
# rebuilds nothing (no assembly, no zeta, no g6 exchange under DD) and the
# forward takes two tapes (chi + w_sum) instead of five. The chi tape STAYS
# either way -- 2-D images inv_rho from chi^k, which 3-D does not, which is
# also why this is its own switch: the two paths have different residues and
# roll back independently.
_WSUM_TAPE_2D = True

# The 3-D acoustic C-PML families (flat and topography) have ONE driver: the
# strip driver in sweep.sem.dd, run with a single strip by PropSEM.forward
# (prop_base._forward_c_dd3d) and by SEMCudaPropagator.forward
# (_forward_acoustic3d_cpml below). The fused single-domain Function that
# used to live here duplicated its allocation, tape, ring, slab and rotation
# logic and was slower and heavier (it built the global equation); it is gone.


def _band_alloc(kind, n_fields, nt, nband, dev, B=1):
    """``(make_one_history, scale_table)`` for a BS ring of ``n_fields``.

    ``B > 1`` sizes the ring for a shot batch: histories ``(nt, B, nband)``
    and the int8 scale table ``(n_fields, nt, B)`` -- one quantisation scale
    per (field, step, SHOT), so one shot's amplitude cannot set another's
    quantisation step. ``B == 1`` keeps the historical shapes exactly.
    """
    dt = _BAND_DTYPE[kind]
    # mk(nb) sizes ONE field's ring: with the per-field split, psi_d / zeta_d
    # ring on their own axis's support and only u / chi on the union band
    def mk(nb=nband):
        shape = (nt, B, nb) if B > 1 else (nt, nb)
        return torch.zeros(*shape, dtype=dt, device=dev)
    scs = (n_fields, nt, B) if B > 1 else (n_fields, nt)
    sc = (torch.zeros(*scs, dtype=torch.float32, device=dev)
          if kind == 2 else torch.empty(0, dtype=torch.float32, device=dev))
    return mk, sc


_I32_MAX = 2 ** 31 - 1


def _ibool_i32(mesh, dev):
    """``mesh.ibool`` as int32 for the C kernels, with the range checked.

    The element-local connectivity is read once per element-node by every
    element kernel, so at int64 it is ~10% of the dominant kernel's memory
    traffic -- and that traffic is what limits the backend once the step's
    resident set passes L2. int32 halves it.

    The int32 ceiling this introduces is NOT new: the kernels already index the
    (E, ngll, ...) arrays with ``const int off = e * m + tid``, and E*m counts
    duplicated nodes, so it exceeds n_global (measured 1.56x in 2-D, 1.84x in
    3-D) and therefore overflows FIRST. What was missing is that neither limit
    was checked -- on a large-memory machine the existing int offset would wrap
    silently and produce a wrong gradient with no error. Hence the guard, which
    is worth having independently of the dtype.

    Head-room today: at ~244 B per node of step state, a 48 GB card holds about
    2.0e8 nodes, i.e. E*m ~ 3.1e8 = 14% of int32.
    """
    n_global = int(mesh.n_global)
    n_local = int(mesh.ibool.numel())          # E * ngll^d, the int-addressed one
    if n_global > _I32_MAX or n_local > _I32_MAX:
        raise ValueError(
            f"mesh too large for the 32-bit indexing in the SEM C kernels: "
            f"n_global={n_global:,}, E*ngll^d={n_local:,}, int32 "
            f"max={_I32_MAX:,}. The kernels address the element-local arrays "
            f"with an int offset, so this limit holds whatever dtype ibool has.")
    return mesh.ibool.to(device=dev, dtype=torch.int32).contiguous()


def _elem_thread_slice(t, name, m):
    """The ``m`` quadrature weights the 3-D flat kernels actually read.

    ``sem3d_acoustic_cpml_grads`` / ``_grads_T`` index these arrays with the
    THREAD id rather than the element-node offset, because on the axis-aligned
    Hex8 mesh every element's slice is identical (the Jacobian is a scalar and
    Gx/Gy/Gz are ``(2/d)^2 |J|``). So of an ``(E, m)`` array the kernels read
    ``m`` values and never look at the other ``(E-1)*m`` -- measured **30.8 B
    per global node** across the four of them, 35% of the 3-D consts,
    allocated and never read. Passing the slice hands them the same floats.

    That every element's slice really is identical is a property of the mesh
    builder, not of the kernels, so it is CHECKED here rather than assumed: a
    non-uniform Hex8 would otherwise silently get element 0's weights, which
    is what the kernels have always done. The comparison runs against an
    ``expand`` VIEW, so the check itself allocates nothing.
    """
    v = t.reshape(-1)
    if v.numel() % m:
        raise ValueError(
            f"{name} has {v.numel()} entries, not a multiple of ngll^3 = {m}")
    rows = v.reshape(-1, m)
    head = v[:m]
    if rows.shape[0] > 1 and not torch.equal(rows, head.expand(rows.shape[0], m)):
        raise ValueError(
            f"{name} varies between elements; the 3-D acoustic C-PML kernels "
            "read it per thread and assume every element's slice is identical. "
            "A deformed mesh must take the *_deformed kernels instead.")
    return head.contiguous()


# Which forward method runs an equation, with and without a C-PML. MOST
# DERIVED FIRST -- the deformed classes subclass their flat twins, so the row
# order is what keeps a Hex27 mesh off the Hex8 path. ``None`` means the pair
# does not exist yet and the lookup raises.
#
# The bare 3-D acoustic pair is ``None`` on purpose: its kernels
# (sem3d_acoustic_forward/backward) bake c² into the element integral, i.e.
# solve div(c² grad u), which flips the sign of every reflection coefficient.
# The eager classes now solve c² Laplace u; the kernels have not been redone.
_C_ROUTES = (
    #  equation                with C-PML                      bare
    (AcousticPotentialSEM2D, "_forward_acoustic_cpml",   "_forward_acoustic"),
    (ElasticSEM2DDeformed,   "_forward_elastic_cpml",    "_forward_elastic"),
    (AcousticElasticSEM2D,   "_forward_fluid_solid",     "_forward_fluid_solid"),
    (ElasticVSSEM3D,         "_forward_elastic_vs3d",    "_forward_elastic_vs3d"),
    (ElasticSEM3DDeformed,   "_forward_elastic3d_deformed",
                             "_forward_elastic3d_deformed"),
    (ElasticSEM3D,           None,                       "_forward_elastic3d"),
    (AcousticSEM3DDeformed,  "_forward_acoustic3d_cpml", None),
    (AcousticSEM3D,          "_forward_acoustic3d_cpml", None),
)

# Equations the CUDA dispatch handles: the table's own keys. PropSEM reads
# THIS to decide whether impl='c' is available. It used to be a second
# hand-written tuple that a test kept in step by regex-scraping the dispatch
# for `isinstance(eq, X)`; there is only one list now, so drift is not a thing
# that can happen rather than a thing that is checked.
C_SUPPORTED = tuple(kind for kind, _, _ in _C_ROUTES)


def c_route(equation, cpml: bool):
    """The forward method ``equation`` takes with / without a C-PML, or None
    when that pair has no kernel (the first matching row, as in dispatch)."""
    for kind, with_cpml, bare in _C_ROUTES:
        if issubclass(equation, kind):
            return with_cpml if cpml else bare
    return None


# --------------------------------------------------------------------------- #
# consts: what every family's dict starts from, and the shared gathers
# --------------------------------------------------------------------------- #
# What the C structs read of the two 2-D C-PML profiles, by name. The elastic
# triples flatten to ``<nm>_<j>``; every entry is (E, ngll, ngll) or node
# shaped and is cast whole. The names live in cdrive so the strip builders in
# dd.py and the drivers' fill lists read the same tuple.


def _acoustic_profile(cp, cast=_f32):
    """The six node-shaped acoustic C-PML coefficients, ``cast`` each."""
    return {nm: cast(getattr(cp, nm)) for nm in AC_CPML_FIELDS}


def _elastic_profile(cp, cast=_f32):
    """The flattened ElasticCPMLProfile: ten fields plus six triples as
    ``<nm>_0..2`` (28 keys). One copy of the loop that _elastic_cpml_consts
    and _fluid_solid_consts each carried verbatim."""
    c = {nm: cast(getattr(cp, nm)) for nm in EL_CPML_FIELDS}
    for nm in EL_CPML_TRIPLES:
        triple = getattr(cp, nm)
        for j in range(3):
            c[f"{nm}_{j}"] = cast(triple[j])
    return c


def _dirichlet_idx(geo, dev):
    """``dirichlet_idx`` as int64 on cuda, with the empty fallback the
    kernels need as a real (0,) tensor rather than a None."""
    return (_i64(geo.dirichlet_idx) if geo.dirichlet_idx.numel()
            else torch.zeros(0, dtype=torch.int64, device=dev))


def _bs_flags(memory, store, store_elem):
    """The three boundary-saving flags every ring-capable C driver reads.
    ``store_elem`` is the already-resolved element-ring dtype (callers decide
    whether None means "follow ``store``")."""
    return {"bs": memory == "boundary",
            "bs_kind": _band_store(store)[0],
            "bs_kind_e": _band_store(store_elem)[0]}


def _stack_patterns(c, props, names):
    """A copy of ``c`` whose per-shot pattern keys are ``(B, ng)`` stacks, one
    row per prop. ``B == 1`` keeps a leading length-1 axis, which the C
    drivers read identically to the single-shot layout."""
    c = dict(c)
    ng = c["n_global"]
    dev = props[0].eq.mesh.ibool.device
    pats = [p._build_source_patterns(ng, dev, torch.float32) for p in props]
    for j, nm in enumerate(names):
        c[nm] = torch.stack([_f32(pl[j]) for pl in pats]).contiguous()
    return c


# The structural test runs on the CONTROL NODES and demands EXACTLY zero drift.
# A separable map stores literally the same float down a column, so the drift is
# 0.0 bitwise -- measured on the real propagator meshes at nex 8, 16, 32 and 48,
# spans 1280 to 7680 m. Any tolerance here would be relative to the span, so the
# same real curvature slips under it on a big enough domain (a 0.6 m bend over
# 9 km is 7e-8 of span); exactly-zero has no such scale. A mesh that is columnar
# in intent but carries ULP noise in x simply keeps the full 3x3 path, which is
# the safe direction. inv_J cannot serve at all: its structural zeros are fp32
# round-off from inverting the Jacobian and grow with the domain (2e-6 of the
# matrix at nex=8, 2.5e-5 at nex=48), so a fixed threshold on THEM switches the
# shortcut off silently on a big enough mesh instead.
_NOISE_FACTOR = 100.0     # how far above this mesh's own metric round-off a
                          # "constant" may drift before it counts as graded


def _columnar_metric(mesh, I):
    """The columnar shortcut of ``I = inv_J``, or ``{}`` when the mesh has none.

    ``build_topography_hex27`` maps x(xi), y(eta), z(xi, eta, gamma): the x and
    y control positions each depend on one index alone -- literally the same
    float down a column -- so J is lower triangular, inv_J has four
    identically-zero components (ixy, ixz, iex, iez) and two mesh-wide
    constants (ixx, iey), and
    the deformed kernels can read four (E, m) streams instead of ten -- 1.25x
    of them, measured.

    Derived here and never taken on a caller's word: a mesh built from general
    control nodes fails and keeps the full 3x3 path, which is always correct.
    Replacing the fp32 round-off of a structural zero with the exact zero is a
    small accuracy gain, so the two paths are NOT bitwise.
    """
    c = mesh.control_xyz                        # (E, 3, 3, 3, 3), axes zeta/eta/xi
    if float(c[..., :2].abs().max()) == 0.0:    # a degenerate mesh: say nothing
        return {}
    drift = lambda t, dims: max(float((t.amax(dim=d) - t.amin(dim=d)).abs().max())
                                for d in dims)
    # x must not move along eta or gamma, nor y along xi or gamma -- at all
    if max(drift(c[..., 0], (1, 2)), drift(c[..., 1], (1, 3))) != 0.0:
        return {}
    # The two survivors travel as scalars, so they must be constant across the
    # WHOLE mesh -- a columnar map with graded x spacing is not. For a mesh that
    # passed the test above, inv_J's four zeros are pure round-off, so they
    # measure this mesh's own metric noise and calibrate what "constant" means
    # here; a real grading sits orders above it.
    Ixx, Ixy, Ixz, Iex, Iey, Iez = I[:6]
    scale = max(float(t.abs().max()) for t in I)
    eps = float(torch.finfo(Ixx.dtype).eps)
    noise = max(max(float(t.abs().max()) for t in (Ixy, Ixz, Iex, Iez)) / scale, eps)
    for t in (Ixx, Iey):
        own = float(t.abs().max())
        if own == 0.0 or float(t.max() - t.min()) > _NOISE_FACTOR * noise * own:
            return {}
    return {"columnar": 1, "col_ixx": float(Ixx.mean()),
            "col_iey": float(Iey.mean())}


def metric_consts(eq, reshape):
    """The nine inv_J streams under the names the deformed kernels read, plus
    the columnar shortcut when the mesh has one.

    Both construction sites go through here -- :meth:`_Base.metric3d` for the
    bare deformed entries and :func:`sweep.sem.dd._geom_ac3d` for the C-PML
    ones -- because a site that builds the streams and forgets the shortcut
    loses it in silence: the struct field defaults to the (correct, slower)
    full 3x3 path, so nothing crashes and nothing is wrong. That is exactly
    what happened when only one of the two was taught about it.
    """
    return {**{nm: reshape(v) for nm, v in zip(METRIC_NAMES, eq._I)},
            **_columnar_metric(eq.mesh, eq._I)}



# Every element kernel launches one block per element with one thread per
# element-node, so the element node count IS the block size and CUDA's 1024
# limit is a bound on the order. Without this the launch returns
# cudaErrorInvalidConfiguration, and since the C entries check
# cudaGetLastError only after the whole nt loop (e.g. sem3d/forward.cu:101),
# the user sees a bare "invalid configuration argument" attributed to nothing,
# nt steps after the cause. Measured: n_order=9 runs (m=1000), n_order=10 does
# not (m=1331).
_MAX_BLOCK = 1024


def check_block_size(n_order, ndim):
    """Refuse an order whose element does not fit in one CUDA block.

    Takes the order and the dimension rather than a mesh: this runs at
    construction, and ``PropSEM.mesh`` is lazy on purpose.
    """
    ngll = n_order + 1
    m = ngll ** ndim
    if m > _MAX_BLOCK:
        top = int(round(_MAX_BLOCK ** (1.0 / ndim))) - 1
        raise ValueError(
            f"n_order={n_order} gives {ngll}^{ndim} = {m} nodes per element, "
            f"and impl='c' launches one THREAD per element node, so it cannot "
            f"exceed CUDA's {_MAX_BLOCK} threads per block. Use n_order <= "
            f"{top} in {ndim}-D, or impl='eager'.")


class _ConstsBase:
    """The family-agnostic part of a consts dict, for ONE builder call.

    Everything here is geometry or acquisition. The blocks are built lazily
    on first use, so a builder pays only for what it spreads; a memoised
    builder keeps its dict through ``_memo_consts`` exactly as before, an
    unmemoised one hands fresh tensors per forward exactly as before. The
    object is a local of the builder and holds no reference to the
    propagator, so a propagator is still freed by refcount.
    """

    __slots__ = ("eq", "mesh", "dev", "ng", "E", "m", "tail",
                 "_prop", "_ibool", "_pats", "_pat", "_rec")

    def __init__(self, prop):
        eq = prop.eq
        mesh = eq.mesh
        self.eq, self.mesh = eq, mesh
        self.dev = torch.device("cuda")
        self.ng, self.E = mesh.n_global, mesh.n_elements
        # (E, ngll, ngll) in 2-D, (E, ngll, ngll, ngll) in 3-D: the receiver
        # stamp width and the flat element size follow the connectivity rank
        self.m = mesh.ngll ** (mesh.ibool.dim() - 1)
        self.tail = {"dt": float(prop.dt), "nt": prop.nt, "ngll": mesh.ngll,
                     "n_elements": self.E, "n_global": self.ng,
                     "n_rec": len(prop.receivers)}
        self._prop = prop            # read by the lazy blocks below only
        self._ibool = self._pats = self._rec = None
        self._pat = {}

    # -- every family --------------------------------------------------- #
    @property
    def ibool(self):
        if self._ibool is None:
            self._ibool = _ibool_i32(self.mesh, self.dev)
        return self._ibool

    def ibool_flat(self):
        """The 3-D kernels take ``(E, m)``; a view of the contiguous int32."""
        return self.ibool.reshape(self.E, self.m)

    def D(self, geo):
        return _f32(geo._D)

    def pattern(self, k):
        """Source field ``k`` on cuda fp32 (the patterns are assembled once on
        the MESH device, which may be the host, and cast per field on demand:
        the families read 1, 2 or 3 of them)."""
        t = self._pat.get(k)
        if t is None:
            if self._pats is None:
                self._pats = self._prop._build_source_patterns(
                    self.ng, self.mesh.ibool.device, torch.float32)
            t = self._pat[k] = _f32(self._pats[k])
        return t

    def rec(self):
        if self._rec is None:
            _, rec_G, rec_W = self._prop._recv_gather(self.m)
            self._rec = {"rec_G": rec_G, "rec_W": rec_W}
        return self._rec

    # -- 2-D families --------------------------------------------------- #
    def geom2d(self, geo):
        """D, w_J, the 1-D GLL weights and the four inverse-Jacobian terms of
        the element geometry ``geo`` (the equation, or the coupled
        equation's acoustic half, which owns the shared mesh geometry)."""
        return {"D": self.D(geo), "wJ": _f32(geo._w_J), "w1d": _f32(self.mesh.w_ref),
                "ixx": _f32(geo._ixx), "ixz": _f32(geo._ixz),
                "iex": _f32(geo._iex), "iez": _f32(geo._iez)}

    def dirichlet(self, geo):
        return _dirichlet_idx(geo, self.dev)

    # -- 3-D families --------------------------------------------------- #
    def flat3(self, t):
        """An (E, ...) per-element-node array as ``(E, m)`` cuda fp32."""
        return _f32(t.reshape(self.E, self.m))

    def ex3(self, t):
        """Like ``flat3`` but an ``(ngll, ngll, ngll)`` array shared by every
        element (the axis-aligned Hex8 quadrature) is broadcast first."""
        E, m = self.E, self.m
        return _f32(t.expand(E, -1, -1, -1).reshape(E, m) if t.dim() == 3
                    else t.reshape(E, m))

    def metric3d(self, E, m):
        """The flattened w_J + 3x3 inverse Jacobian (E, m) of a Hex27 mesh, in
        the ``eq._I`` order the deformed kernels index by, plus the columnar
        shortcut when the mesh has one (see :func:`_columnar_metric`)."""
        r = lambda t: _f32(t.reshape(E, m))
        return {"wJ": r(self.eq._w_J), **metric_consts(self.eq, r)}

    def elastic3d_geom(self):
        """The bare Hex8 elastic quadrature: w3 and |J| per element-node plus
        the six ``4/(d_a d_b)`` stiffness scalings."""
        eq, mesh, E, m = self.eq, self.mesh, self.E, self.m
        ngll = mesh.ngll
        jac = mesh.jacobian_det
        if not torch.is_tensor(jac):                  # uniform axis-aligned |J|
            jac = torch.full((1, 1, 1, 1), float(jac), dtype=torch.float32)
        jac_e = (jac if jac.dim() == 4 and jac.shape[0] == E else
                 jac.expand(E, ngll, ngll, ngll))
        return {"w3": _f32(eq._w3.expand(E, -1, -1, -1).reshape(E, m)),
                "jac": _f32(jac_e.reshape(E, m)),
                "Gxx": float(4.0 / mesh.dx_e ** 2),
                "Gyy": float(4.0 / mesh.dy_e ** 2),
                "Gzz": float(4.0 / mesh.dz_e ** 2),
                "Gxy": float(4.0 / (mesh.dx_e * mesh.dy_e)),
                "Gxz": float(4.0 / (mesh.dx_e * mesh.dz_e)),
                "Gyz": float(4.0 / (mesh.dy_e * mesh.dz_e))}

    def vs3d_geom(self):
        """The velocity-stress Hex8 quadrature: the three half-cell weight
        arrays and the ``1/(2h)`` scalars."""
        eq = self.eq
        return {"wHx": self.ex3(eq._wHx), "wHy": self.ex3(eq._wHy),
                "wHz": self.ex3(eq._wHz),
                "ihx": float(eq._inv_half_dx), "ihy": float(eq._inv_half_dy),
                "ihz": float(eq._inv_half_dz)}


class SEMCudaPropagator(SEMPropagator):
    """``impl='c'`` propagator (acoustic-potential + elastic; forward + adjoint)."""

    def _recv_gather(self, m):
        """``(nrec, rec_G, rec_W)`` with the zero-receiver fallback.

        This exact block existed 8 times (verified byte-identical up to the
        2-D/3-D ``m``); the gather itself is pure geometry shared by every
        consts builder.
        """
        mesh = self.eq.mesh
        nrec = len(self.receivers)
        if nrec > 0:
            G, W = receiver_gather(mesh, self.receivers)
            rec_G, rec_W = _i64(G), _f32(W)
        else:
            rec_G = torch.zeros((0, m), dtype=torch.int64, device="cuda")
            rec_W = torch.zeros((0, m), dtype=torch.float32, device="cuda")
        return nrec, rec_G, rec_W

    def _base(self):
        """The family-agnostic consts pieces for ONE builder call."""
        return _ConstsBase(self)

    # -- 2-D acoustic-potential ----------------------------------------- #
    @_memo_consts
    def _consts(self):
        """Bare acoustic-potential: geometry + dirichlet + one source field."""
        b, eq = self._base(), self.eq
        return {
            "ibool": b.ibool, **b.geom2d(eq),
            "mask_e": _f32(eq._mask_e.reshape(-1)),
            "dirichlet_idx": b.dirichlet(eq),
            "src_pattern": b.pattern(0),
            **b.rec(), **b.tail,
        }

    def forward(self, source_wavelet: torch.Tensor, memory=None,
                bs_store_dtype=None, bs_store_dtype_elem=None, **kwargs):
        """Seismogram ``(nt, nrec, ncomp)`` -- the same contract as the eager
        :meth:`SEMPropagator.forward`. The private ``_forward_*`` dispatch
        targets below return one tensor per receiver-capable field, in
        ``field_specs`` order; ``receiver_type`` then selects and orders the
        output components."""
        for k in ("record_field_every", "snapshot_times"):
            if kwargs.get(k) is not None:
                raise NotImplementedError(
                    f"{k}: the CUDA kernels run the whole time loop in one "
                    f"call and keep no intermediate state; use the eager "
                    f"SEMPropagator for snapshots.")
        return self._assemble(self._forward_components(
            source_wavelet, memory=memory, bs_store_dtype=bs_store_dtype,
            bs_store_dtype_elem=bs_store_dtype_elem))

    def _assemble(self, comps):
        recv = [sp.name for sp in self.eq.field_specs if sp.supports_receiver]
        at = {nm: i for i, nm in enumerate(recv)}
        rtype = self.receiver_type or recv
        return torch.stack([comps[at[nm]] for nm in rtype], -1).transpose(0, 1)

    # Which impl='c' forward paths actually consume ``memory='boundary'``.
    # Derived from which methods wire c["bs"]: _forward_acoustic_cpml,
    # _forward_elastic_cpml and _forward_acoustic3d_cpml (both 3-D acoustic
    # families; their bare pair is refused in _C_ROUTES). The others run
    # the full tape regardless, and asking for boundary saving on them used to
    # be silently ignored, so a caller believed they were saving memory and
    # reconstructing the wavefield when they were not.
    #
    # Note the 2-D pair is CPML-ONLY: the bare AcousticPotentialSEM2D path is
    # inlined in _forward_components and never sets the flag.
    _BS_ANY_CPML = (AcousticSEM3D, AcousticSEM3DDeformed)
    _BS_NEEDS_CPML = (AcousticPotentialSEM2D, ElasticSEM2DDeformed,
                      AcousticElasticSEM2D)

    def _assert_memory_supported(self, memory):
        if memory != "boundary":
            return
        eq = self.eq
        has_cpml = getattr(eq, "cpml", None) is not None
        if isinstance(eq, self._BS_ANY_CPML):
            return
        if isinstance(eq, self._BS_NEEDS_CPML) and has_cpml:
            return
        why = ("without a C-PML (the bare path never wires the flag)"
               if isinstance(eq, self._BS_NEEDS_CPML) else "at all")
        raise NotImplementedError(
            f"memory='boundary' is not implemented for {type(eq).__name__} on "
            f"impl='c' {why}. It used to be accepted and ignored, which looked "
            f"like it worked. Use impl='eager' (boundary saving is implemented "
            f"there for every equation) or memory=None.")

    def _forward_acoustic(self, wav):
        """Bare 2-D acoustic potential: no C-PML, so no memory fields."""
        eq, dev = self.eq, torch.device("cuda")
        seis = _SEMAcousticC.apply(eq._inv_M.to(device=dev, dtype=torch.float32),
                                   eq._inv_rho.to(device=dev, dtype=torch.float32),
                                   wav, self._consts())
        return (seis,)

    def _check_forward(self, source_wavelet, memory, bs_store_dtype,
                       bs_store_dtype_elem):
        """Record the run's memory knobs on ``self`` (the forward methods read
        them through ``_stamp_bs``) and refuse what this backend cannot do.
        Returns the wavelet as cuda fp32."""
        self._assert_memory_supported(memory)
        self._c_memory = memory
        self._c_bs_store = bs_store_dtype
        # element rings may be stored coarser than the node rings: see
        # "Ring precision" in docs/user-guide/sem.md for why that is the
        # useful split. None -> follow bs_store_dtype.
        self._c_bs_store_elem = (bs_store_dtype if bs_store_dtype_elem is None
                                 else bs_store_dtype_elem)
        if bs_store_dtype_elem is not None and not _has_element_rings(self.eq):
            raise ValueError(
                "bs_store_dtype_elem only applies to rings with ELEMENT "
                "histories (the elastic and coupled fluid-solid paths); "
                f"{type(self.eq).__name__} has node rings only")
        if source_wavelet.dim() != 1 or source_wavelet.shape[0] != self.nt:
            raise ValueError(f"needs a shared (nt={self.nt},) wavelet; "
                             f"got {tuple(source_wavelet.shape)}")
        if not torch.cuda.is_available():
            raise RuntimeError("SEM impl='c' requires CUDA.")
        return source_wavelet.to(device=torch.device("cuda"), dtype=torch.float32)

    def _forward_components(self, source_wavelet: torch.Tensor, memory=None,
                            bs_store_dtype=None, bs_store_dtype_elem=None,
                            **kwargs):
        """One shot through the family's forward method -- the (C-PML, bare)
        pair of the FIRST row of ``_C_ROUTES`` the equation is an instance of.
        The rows are ordered most-derived first, so a deformed equation never
        reaches its flat twin's row."""
        wav = self._check_forward(source_wavelet, memory, bs_store_dtype,
                                  bs_store_dtype_elem)
        cpml = getattr(self.eq, "cpml", None)
        for kind, with_cpml, bare in _C_ROUTES:
            if isinstance(self.eq, kind):
                name = with_cpml if cpml is not None else bare
                if name is None:
                    raise NotImplementedError(
                        f"{type(self.eq).__name__} "
                        f"{'with' if cpml is not None else 'without'} a C-PML "
                        f"has no impl='c' kernel; use impl='eager'"
                        + (" or abcn > 0" if issubclass(kind, AcousticSEM3D)
                           else "") + ".")
                return getattr(self, name)(wav)
        raise NotImplementedError(
            f"SEM impl='c' supports acoustic-potential, elastic, coupled "
            f"fluid-solid and 3-D acoustic (got {type(self.eq).__name__}).")

    @_memo_consts
    def _cpml_consts(self):
        """Acoustic-potential C-PML: the bare consts + the geometric inverse
        mass, the node profile and the boundary-saving band indices."""
        b, eq = self._base(), self.eq
        cp = eq.cpml
        from .boundary_saving import cpml_node_masks, band_support_idx
        # torch.nonzero -> host sync; memoised so a graph capture never sees it
        band_idx = _i64(cpml_node_masks(eq)[1])
        return {
            "band_idx": band_idx,
            # each memory field's ring index: the band nodes where ITS
            # coefficient is non-zero (it is exactly zero elsewhere). Filtered
            # on the RAW profile, not the fp32 cast: test_sem_bs_ring_split's
            # sorted-subset contract rests on that.
            "band_idx_x": band_support_idx(band_idx, cp.b_x),
            "band_idx_z": band_support_idx(band_idx, cp.b_z),
            "ibool": b.ibool, **b.geom2d(eq),
            "mask_e": _f32(eq._mask_e.reshape(-1)),
            "inv_Mg": _f32(eq._inv_M_geo),
            **_acoustic_profile(cp),
            "src_pattern": b.pattern(0),
            **b.rec(),
            "dirichlet_idx": b.dirichlet(eq),
            **b.tail,
        }

    def _stamp_bs(self, c):
        """Stamp the three boundary-saving flags onto a consts dict.

        Seven forward paths need them and six carried a verbatim copy of this
        block; the seventh -- fluid-solid -- set only ``bs`` and dropped both
        precision flags, so `_SEMFluidSolidC.forward`'s
        ``int(c.get("bs_kind", 0))`` silently fell back to fp32 rings and
        ``bs_store_dtype`` delivered none of its documented 2x/4x on the one
        path where the ELEMENT rings dominate. Measured, by injecting the
        dropped flag: fluid-solid boundary-saved peak 432.8 MB -> 243.9 (bf16)
        / 151.6 (int8), i.e. the rings are ~87% of that peak.

        ``_c_bs_store_elem`` is already resolved at the entry point (it is
        ``bs_store_dtype`` when the caller left the element knob unset), so
        the getattr defaults here only cover a call that precedes it.
        """
        store = getattr(self, "_c_bs_store", None)
        c.update(_bs_flags(getattr(self, "_c_memory", None), store,
                           getattr(self, "_c_bs_store_elem", store)))
        return c

    def _forward_acoustic_cpml(self, wav):
        """Acoustic-potential C-PML forward + adjoint (P3a/P3b) via autograd
        Function. ``memory='boundary'`` switches the save-all histories to the
        CPML-band ring (exact reconstruction; gradient identical)."""
        eq = self.eq
        c = self._stamp_bs(self._cpml_consts())
        inv_rho = eq._inv_rho.to("cuda", torch.float32)
        inv_M = eq._inv_M.to("cuda", torch.float32)
        kappa = eq.kappa_node.to("cuda", torch.float32)
        seis = _SEMAcousticCpmlC.apply(inv_rho, inv_M, kappa, wav, c)
        return (seis,)

    # -- 2-D elastic ----------------------------------------------------- #
    @_memo_consts
    def _elastic_consts(self):
        """Bare elastic: geometry + the two displacement source fields."""
        b, eq = self._base(), self.eq
        return {
            "ibool": b.ibool, **b.geom2d(eq),
            "mask_e": _f32(eq._mask_e.reshape(-1)),
            "pat_x": b.pattern(0), "pat_z": b.pattern(1),
            **b.rec(), **b.tail,
        }

    def _forward_elastic(self, wav):
        eq = self.eq
        c = self._elastic_consts()
        lam = eq._lam.to("cuda", torch.float32)
        mu = eq._mu.to("cuda", torch.float32)
        lam2mu = eq._lam2mu.to("cuda", torch.float32)
        inv_M = eq._inv_M.to("cuda", torch.float32)
        seis_x, seis_z = _SEMElasticC.apply(lam, mu, lam2mu, inv_M, wav, c)
        return (seis_x, seis_z)

    @_memo_consts
    def _elastic_cpml_consts(self):
        """Elastic consts + the flattened ElasticCPMLProfile (P3e), the
        rigid-wall pin and the boundary-saving bands. Extends the memoised
        ``_elastic_consts`` copy, so the two caches share their tensors."""
        eq = self.eq
        cp = eq.cpml
        dev = torch.device("cuda")
        outer_edge_idx = _i64(cp.outer_edge_idx)
        # the same rigid-wall nodes as a 0/1 mask, so the backward can apply the
        # pin inside a fused per-node kernel instead of a separate scatter launch
        pin = torch.ones(eq.mesh.n_global, dtype=torch.float32, device=dev)
        if outer_edge_idx.numel():
            pin[outer_edge_idx] = 0.0
        from .boundary_saving import (cpml_band_element_mask, cpml_band_elements,
                                      cpml_node_masks)
        return {
            **self._elastic_consts(),
            **_elastic_profile(cp),
            "outer_edge_idx": outer_edge_idx, "pin_mask": pin,
            "band_node_idx": _i64(cpml_node_masks(eq)[1]),   # host sync: memoised
            "band_elem_idx": _i64(cpml_band_elements(eq)),
            # (E,) int32: does this element touch the C-PML band? The element
            # kernels launch one block per element, so an interior block can
            # take a fast path with no divergence at all -- outside the band
            # every C-PML coefficient is exactly 0 (and the stretch exactly 1),
            # measured, so the terms it skips are exact zeros.
            "elem_band": cpml_band_element_mask(eq).to(dev, torch.int32),
        }

    def _forward_elastic_cpml(self, wav):
        eq = self.eq
        c = self._stamp_bs(self._elastic_cpml_consts())
        lam = eq._lam.to("cuda", torch.float32)
        mu = eq._mu.to("cuda", torch.float32)
        lam2mu = eq._lam2mu.to("cuda", torch.float32)
        inv_M = eq._inv_M.to("cuda", torch.float32)
        w_rho_J = eq._w_rho_J.to("cuda", torch.float32)
        seis_x, seis_z = _SEMElasticCpmlC.apply(
            lam, mu, lam2mu, inv_M, w_rho_J, wav, c)
        return (seis_x, seis_z)

    # -- 3-D acoustic ---------------------------------------------------- #
    @_memo_consts
    def _acoustic3d_consts(self):
        """3-D constants: flat ibool/rec (m = ngll^3), no metric tensors."""
        b, eq = self._base(), self.eq
        return {
            "ibool": b.ibool_flat(), "D": b.D(eq),
            "dirichlet_idx": b.dirichlet(eq),
            "src_pattern": b.pattern(0),
            **b.rec(), **b.tail,
        }

    def _forward_acoustic3d_cpml(self, wav):
        """Both 3-D acoustic C-PML families (flat and topography) run on the
        strip driver of :mod:`sweep.sem.dd` with one strip -- the same route
        :class:`PropSEM` takes. The source is this propagator's summed pattern
        (kinds and payloads applied), the receivers its point stamps."""
        from .dd import SEMModelParallel, _Stamps
        eq = self.eq
        dd = getattr(self, "_dd1", None)
        if dd is None:
            dd = self._dd1 = SEMModelParallel.for_equation(eq, self.dt, self.nt)
        pat = self._build_source_patterns(eq.mesh.n_global, eq.mesh.ibool.device,
                                          torch.float32)[0]
        rec = _Stamps.of(self.receivers, torch.device("cuda"))
        out = dd.run(eq, wav, (_f32(pat),), rec, memory=self._c_memory,
                     bs_store_dtype=self._c_bs_store)          # (nt, nrec)
        return (out.transpose(0, 1).contiguous(),)

    # -- 3-D elastic ----------------------------------------------------- #
    def _elastic3d_consts(self):
        """Bare Hex8 elastic: w3 / |J| / G** drive the stiffness. Not memoised
        (it never was)."""
        b, eq = self._base(), self.eq
        return {
            "ibool": b.ibool_flat(), "D": b.D(eq),
            **b.elastic3d_geom(),
            "pat_x": b.pattern(0), "pat_y": b.pattern(1), "pat_z": b.pattern(2),
            **b.rec(), **b.tail,
        }

    def _elastic3d_deformed_consts(self):
        """Hex27 topography elastic (ElasticSEM3DDeformed): the full per-node
        metric drives the deformed stiffness/imaging; w3 / jac / G** are
        placeholders the deformed kernels never read (w3 and jac alias wJ)."""
        b, eq = self._base(), self.eq
        metric = b.metric3d(b.E, b.m)
        return {
            "ibool": b.ibool_flat(), "D": b.D(eq),
            "w3": metric["wJ"], "jac": metric["wJ"],
            "pat_x": b.pattern(0), "pat_y": b.pattern(1), "pat_z": b.pattern(2),
            **b.rec(),
            "Gxx": 0.0, "Gyy": 0.0, "Gzz": 0.0, "Gxy": 0.0, "Gxz": 0.0, "Gyz": 0.0,
            "deformed": 1, **metric,
            **b.tail,
        }

    def _forward_elastic3d(self, wav):
        return self._run_elastic3d(self._elastic3d_consts(), wav)

    def _forward_elastic3d_deformed(self, wav):
        return self._run_elastic3d(self._elastic3d_deformed_consts(), wav)

    def _run_elastic3d(self, c, wav):
        eq = self.eq
        E, m = c["n_elements"], c["ngll"] ** 3
        t = lambda x: x.to("cuda", torch.float32).reshape(E, m)
        sx, sy, sz = _SEMElastic3DC.apply(
            t(eq._lam), t(eq._mu), t(eq._lam2mu),
            eq._inv_M.to("cuda", torch.float32), wav, c)
        return (sx, sy, sz)

    def _elastic_vs3d_consts(self):
        """Velocity-stress Hex8 elastic, with the per-axis C-PML coefficients
        (broadcast to (E, m) like the quadrature) when the equation has one;
        ``has_cpml`` tells the driver whether to fill or hand empties."""
        b, eq = self._base(), self.eq
        c = {
            "ibool": b.ibool_flat(), "D": b.D(eq),
            **b.vs3d_geom(),
            "pat_x": b.pattern(0), "pat_y": b.pattern(1), "pat_z": b.pattern(2),
            **b.rec(),
            "has_cpml": eq.cpml is not None,
            **b.tail,
        }
        if eq.cpml is not None:     # the driver's own fill list, so they cannot drift
            c.update({nm: b.ex3(getattr(eq.cpml, nm)) for nm in _VS3D_CPML})
        return c

    def _forward_elastic_vs3d(self, wav):
        eq = self.eq
        c = self._elastic_vs3d_consts()
        E, m = c["n_elements"], c["ngll"] ** 3
        t = lambda x: x.to("cuda", torch.float32).reshape(E, m)
        sx, sy, sz = _SEMElasticVS3DC.apply(
            t(eq._lam), t(eq._mu), t(eq._lam2mu),
            eq._inv_M.to("cuda", torch.float32), wav, c)
        return (sx, sy, sz)

    # -- 2-D coupled fluid-solid ----------------------------------------- #
    def _fluid_solid_consts(self):
        """Shared geometry (from the acoustic sub-equation, which owns the
        shared mesh geometry) + both domains' masks + interface quadrature +
        (when present) both C-PML profiles, cast to cuda-fp32 (P4). Not
        memoised (it never was): the dict is fresh per call; the band indices
        live in ``_fluid_solid_bands``."""
        b, eq = self._base(), self.eq
        ac, el, ifc = eq.ac, eq.el, eq.ifc
        c = {
            "ibool": b.ibool, **b.geom2d(ac),
            "ac_mask_e": _f32(ac._mask_e.reshape(-1)),
            "el_mask_e": _f32(el._mask_e.reshape(-1)),
            "dirichlet_idx": b.dirichlet(ac),
            "ifc_idx": _i64(ifc.node_idx.reshape(-1)),
            "ifc_w": _f32(ifc.weight.reshape(-1)),
            "ifc_nx": _f32(ifc.nx.reshape(-1)),
            "ifc_nz": _f32(ifc.nz.reshape(-1)),
            "pat_chi": b.pattern(0), "pat_x": b.pattern(1),   # chi, u_x, u_z
            "pat_z": b.pattern(2),
            **b.rec(),
            "has_cpml": eq.cpml is not None,
            **b.tail,
        }
        if eq.cpml is not None:
            cp_ac, cp_el = eq.cpml
            c.update({
                "ac_inv_Mg": _f32(ac._inv_M_geo),
                **_acoustic_profile(cp_ac),
                **_elastic_profile(cp_el),
                "outer_edge_idx": _i64(cp_el.outer_edge_idx),
            })
        return c

    @_memo_consts
    def _fluid_solid_bands(self):
        """Both domains' C-PML bands for boundary saving.

        ``cpml_node_masks`` cannot serve here: the fluid-solid ``eq.cpml`` is
        the (ac_profile, el_profile) TUPLE, so its getattr probing comes back
        empty -- which is exactly why the EAGER coupled BS runs ring-less
        (zero-memory reverse + anchors) and is validated only on the bare box.
        The C path saves real rings instead: acoustic band from the node
        profile's b/db coefficients, elastic band from ``node_in_pml``, each
        widened by its one-element rim -- or, under ``bs_band_mode='rim'``,
        that rim ALONE, for both domains' node rings.

        The solid's ELEMENT ring is exempt from ``'rim'`` and always covers the
        whole band: its six memory fields live only inside the C-PML, and
        zeroing them corrupts the adjoint entering the INTERIOR once the band
        is strongly excited -- not just the band gradient (see
        ``SEMReconState``). So rim buys the node rings only.
        """
        eq, mesh = self.eq, self.eq.mesh
        mode = getattr(eq, "bs_band_mode", "full")
        if mode not in ("full", "rim"):
            raise ValueError(
                f"bs_band_mode must be 'full' or 'rim', got {mode!r}")
        cp_ac, cp_el = eq.cpml
        ngm = mesh.n_global
        elem_nodes = mesh.ibool.reshape(mesh.n_elements, -1)

        def band(coefs):
            active = torch.zeros(ngm, dtype=torch.bool, device=mesh.ibool.device)
            for v in coefs:
                v = v.to(mesh.ibool.device)
                if tuple(v.shape) == (ngm,):
                    active |= v != 0
                else:                        # element-local, ibool-shaped
                    active[mesh.ibool[v != 0]] = True
            # 'rim' thins the NODE ring only: the solid's ELEMENT ring
            # stays full-band, for the reason the docstring gives
            from .boundary_saving import band_from_active
            b, has = band_from_active(active, elem_nodes, mode)
            return (torch.nonzero(b).reshape(-1),
                    torch.nonzero(has).reshape(-1), has)

        ac_nodes, _, _ = band([cp_ac.b_x, cp_ac.b_z, cp_ac.dbx_dx, cp_ac.dbz_dz])
        el_nodes, el_elems, el_has = band([cp_el.node_in_pml])
        return {
            "band_idx": _i64(ac_nodes),
            "band_node_idx": _i64(el_nodes),
            "band_elem_idx": _i64(el_elems),
            # the shared elastic element kernels' interior fast-path flag: the
            # coupled path launches the SAME two kernels as the elastic one and
            # its solid domain has the same band geometry, so it takes the same
            # fast path with no new argument to make
            "elem_band": el_has.to(torch.device("cuda"), torch.int32),
            "ckpt_every": int(getattr(type(eq), "bs_frame_checkpoint_every", 50)),
        }

    def _forward_fluid_solid(self, wav):
        eq = self.eq
        c = self._stamp_bs(self._fluid_solid_consts())
        # The band indices are geometry of the ABSORBING LAYER, so the
        # condition is "is there one", not "is this boundary saving" -- the
        # elastic path already builds them that way (_elastic_cpml_consts is
        # reached only with a C-PML, and builds them unconditionally there).
        # They cost one memoised host sync and only the ring layout reads them
        # today, but an element flag or a band-only history added later would
        # read `band_elem_idx` in tape mode and find the empty tensor this
        # used to leave there. Empty means "no band", so the layer would
        # silently stop absorbing: no error, no NaN, just reflections.
        # The bare box has no cpml and no band, which is not the same thing.
        # Pinned by test_the_band_indices_are_built_in_both_memory_modes.
        if eq.cpml is not None:
            c.update(self._fluid_solid_bands())
        ac, el = eq.ac, eq.el
        t = lambda x: x.to("cuda", torch.float32)
        seis_chi, seis_x, seis_z = _SEMFluidSolidC.apply(
            t(ac._inv_rho), t(ac._inv_M), t(ac.kappa_node),
            t(el._lam), t(el._mu), t(el._lam2mu), t(el._inv_M), t(el._w_rho_J),
            wav, c)
        return (seis_chi, seis_x, seis_z)


def make_sem_propagator(eq, *, impl: str = "eager", **build_kwargs) -> SEMPropagator:
    """Minimal ``impl`` switch. ``'eager'`` → :class:`SEMPropagator`; ``'c'`` →
    :class:`SEMCudaPropagator` (compiled binding); ``'auto'`` → ``'c'`` when available
    and the equation is supported (acoustic-potential, no CPML), else ``'eager'``.
    Full :class:`PropSEM`-level wiring is a later step."""
    cls = SEMPropagator
    if impl == "c":
        if not sem_c_available():
            raise RuntimeError("impl='c' needs the compiled sweep._C SEM binding.")
        cls = SEMCudaPropagator
    elif impl == "auto":
        supported = isinstance(eq, (AcousticPotentialSEM2D, ElasticSEM2DDeformed))
        if sem_c_available() and supported:      # bare + CPML both implemented
            cls = SEMCudaPropagator
    elif impl != "eager":
        raise ValueError(f"impl must be 'eager', 'c', or 'auto'; got {impl!r}")
    return cls.build(eq, **build_kwargs)
