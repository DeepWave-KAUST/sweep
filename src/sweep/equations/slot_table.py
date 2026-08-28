"""Declarative description of an equation's CUDA wavefield bind order.

The order in which a compiled equation's wavefield tensors are bound
(``tensors[i++]`` in each ``*WavefieldTensor::bind()``) is load-bearing
positional knowledge that lives in C++ and, until this module, was hand-copied
into three separate Python places:

* ``propagator/_stepped.py`` -- ``ACOUSTIC{2,3}D_{PSI,ADJ}_PAIRS``, the index
  tuples the host-side buffer-role rotation swaps between stepped calls;
* ``parallel/dd_propagator.py`` -- ``_FAMILIES``, the per-family
  ``nwf``/``nphys``/``nv``/``nrecon`` counts;
* ``equations/cuda_layout.py`` -- ``pml_slot_axes`` / ``checkpoint_slot_axes``,
  positional tuples that are already *in* bind order but say so only implicitly.

Declaring the order once and deriving all three removes the class of bug where
one copy is updated and another is not -- which is not hypothetical: a wrong
pair index silently mis-binds a CPML double buffer, and nothing fails loudly.

Fact vs policy
--------------
A slot's ``axis`` is a **fact** -- which derivative that CPML memory variable
accumulates, readable from the kernel that writes it. Whether the runtime
*slab-allocates* that slot is a **policy**, and the two must not be conflated:
``_c.py`` allocates aux slots as per-axis slabs only when ``pml_slot_axes`` is
non-None, and ten equations deliberately leave it None and get full grids. If
``pml_slot_axes`` were derived from ``axis`` unconditionally, declaring a table
for one of those ten would silently change its allocation. Hence
:attr:`SlotTable.aux_storage`: the derived tuple stays ``None`` unless the
equation opts in, which reproduces today's behaviour exactly.

Scope
-----
This describes the **wavefield list** only. It is deliberately not also the
eager field vector, nor the source/receiver field-id space: those are different
orderings of overlapping names (for ``Elastic3D``, field id 19 is ``m_szzz``
while bind index 19 is ``m_sxxy``). The table makes "which namespace is this
integer in?" answerable, not moot.

``AcousticLSRTM``/``3D`` are intentionally left without a table: their lists are
two back-to-back copies of the acoustic layout, bound as two independent
structs, which a flat slot tuple cannot express. They keep the legacy path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal[
    "u",                                 # rotating time-level buffer (u_prev/now/next)
    "vel", "stress", "phys",             # non-rotating physical state
    "pml_psi", "pml_zeta", "pml_mem",    # CPML auxiliaries
    "dbuf",                              # double-buffer shadow of another slot
]

PHYS_ROLES = frozenset({"u", "vel", "stress", "phys"})
U_BLOCK = 3                              # u_prev / u_now / u_next


@dataclass(frozen=True)
class Slot:
    """One entry of an equation's ``tensors[i++]`` bind sequence."""

    name: str                            # C++ member, "_t" suffix stripped
    role: Role
    axis: str | None = None              # 'x'|'y'|'z' differencing axis
    dbuf_of: str | None = None           # for role == "dbuf": the slot it shadows
    adjoint_only: bool = False           # bound in adjoint_wavefields only
    reserved: bool = False               # bound + checkpointed, read by no kernel


@dataclass(frozen=True)
class SlotTable:
    """Positional truth for ONE equation's CUDA wavefield list.

    ``slots`` is literally the maximal ``tensors[i++]`` sequence; the forward
    list is the prefix with ``adjoint_only`` dropped, and the adjoint list is
    the whole thing.
    """

    slots: tuple[Slot, ...]
    #: names of the reconstruction (boundary-saving replay) list, in bind order.
    #: NOT derivable from ``slots``: the elastic carries (``fvx_prev``, ...) are
    #: separate locals spliced onto the same Python list, absent from bind().
    #: These are the exact strings ``elastic2d/backward.cu`` prints when the
    #: length is wrong, so a drift test can compare names, not just a count.
    recon: tuple[str, ...] = ()
    #: "full" reproduces today's allocation; "slab" opts into per-axis aux slabs.
    aux_storage: Literal["full", "slab"] = "full"

    # -- forward / adjoint lists -------------------------------------------
    def _fwd(self) -> tuple[Slot, ...]:
        return tuple(s for s in self.slots if not s.adjoint_only)

    @property
    def n_forward(self) -> int:
        return len(self._fwd())

    @property
    def n_adjoint(self) -> int:
        return len(self.slots)

    @property
    def nrecon(self) -> int:
        return len(self.recon)

    # -- counts that cuda_layout currently declares by hand ------------------
    @property
    def base_nvar(self) -> int:
        return sum(1 for s in self._fwd() if s.role in PHYS_ROLES)

    @property
    def pml_nvar(self) -> int:
        return self.n_forward - self.base_nvar

    @property
    def adjoint_extra_nvar(self) -> int:
        return self.n_adjoint - self.n_forward

    @property
    def pml_slot_axes(self) -> tuple[str | None, ...] | None:
        """Per-axis slab tags for the forward aux slots, or None (full grids).

        ``None`` unless the equation opts into ``aux_storage="slab"`` -- see the
        fact/policy note in the module docstring.
        """
        if self.aux_storage != "slab":
            return None
        return tuple(s.axis for s in self._fwd() if s.role not in PHYS_ROLES)

    # -- index sets the DD driver needs -------------------------------------
    @property
    def phys_idx(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self._fwd()) if s.role in PHYS_ROLES)

    @property
    def vel_idx(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self._fwd()) if s.role == "vel")

    @property
    def stress_idx(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self._fwd()) if s.role == "stress")

    # -- buffer-role rotation ------------------------------------------------
    @property
    def u_blocks(self) -> tuple[int, ...]:
        """Start indices of each rotating 3-tensor time-level block.

        ``()`` for the first-order staggered families, whose fields all update
        in place at a fixed slot -- which is what ``_stepped.py`` wants passed
        as ``u_blocks=()``.
        """
        idx = [i for i, s in enumerate(self._fwd()) if s.role == "u"]
        return tuple(idx[i] for i in range(0, len(idx), U_BLOCK))

    def pairs(self, *, adjoint: bool) -> tuple[tuple[int, int], ...]:
        """``(slot, shadow)`` index pairs the per-step swap exchanges.

        Ordered by the SHADOW index, which is what reproduces the 3-D adjoint
        tuple's non-monotone first components ``(3,4,7,5,6,8)``.
        """
        pool = self.slots if adjoint else self._fwd()
        by_name = {s.name: i for i, s in enumerate(pool)}
        out = [(by_name[s.dbuf_of], i)
               for i, s in enumerate(pool)
               if s.role == "dbuf" and s.dbuf_of in by_name]
        out.sort(key=lambda p: p[1])
        return tuple(out)


def slot_table_of(equation):
    """The declared :class:`SlotTable` for ``equation``, or None (legacy path)."""
    return getattr(getattr(equation, "cuda_layout", None), "slots", None)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _u3() -> list[Slot]:
    return [Slot("u_prev", "u"), Slot("u_now", "u"), Slot("u_next", "u")]


#: ``AcousticWavefieldTensor::bind()`` (csrc/cuda/common/acoustic.h) -- 2-D
#: maximal list. The forward binds 9 (psi double buffer); the fused adjoint
#: binds 11, adding the zeta shadows.
ACOUSTIC2D = SlotTable(
    slots=tuple(_u3() + [
        Slot("psix", "pml_psi", "x"), Slot("psiz", "pml_psi", "z"),
        Slot("zetax", "pml_zeta", "x"), Slot("zetaz", "pml_zeta", "z"),
        Slot("psixn", "dbuf", "x", dbuf_of="psix"),
        Slot("psizn", "dbuf", "z", dbuf_of="psiz"),
        Slot("zetaxn", "dbuf", "x", dbuf_of="zetax", adjoint_only=True),
        Slot("zetazn", "dbuf", "z", dbuf_of="zetaz", adjoint_only=True),
    ]),
    recon=("u_prev", "u_now", "u_next"),
    aux_storage="slab",
)

#: 3-D sibling. Note the y slots are inserted MID-LIST (after the 2-D psi/zeta
#: quartet, before the psi shadows), which is why the derived 3-D adjoint pair
#: tuple is non-monotone in its first components.
ACOUSTIC3D = SlotTable(
    slots=tuple(_u3() + [
        Slot("psix", "pml_psi", "x"), Slot("psiz", "pml_psi", "z"),
        Slot("zetax", "pml_zeta", "x"), Slot("zetaz", "pml_zeta", "z"),
        Slot("psiy", "pml_psi", "y"), Slot("zetay", "pml_zeta", "y"),
        Slot("psixn", "dbuf", "x", dbuf_of="psix"),
        Slot("psizn", "dbuf", "z", dbuf_of="psiz"),
        Slot("psiyn", "dbuf", "y", dbuf_of="psiy"),
        Slot("zetaxn", "dbuf", "x", dbuf_of="zetax", adjoint_only=True),
        Slot("zetazn", "dbuf", "z", dbuf_of="zetaz", adjoint_only=True),
        Slot("zetayn", "dbuf", "y", dbuf_of="zetay", adjoint_only=True),
    ]),
    recon=("u_prev", "u_now", "u_next"),
    aux_storage="slab",
)

#: Variable-density VRZ shares the acoustic bind order but has NO fused adjoint,
#: so it stops at the psi shadows -- which is exactly why ``pairs(adjoint=True)``
#: equals ``pairs(adjoint=False)`` for it, and why the DD driver's
#: ``adjoint_extra_nvar`` discriminator becomes unnecessary.
ACOUSTIC_VRZ2D = SlotTable(
    slots=ACOUSTIC2D.slots[:9], recon=("u_prev", "u_now", "u_next"))
ACOUSTIC_VRZ3D = SlotTable(
    slots=ACOUSTIC3D.slots[:12], recon=("u_prev", "u_now", "u_next"))


def _elastic_mem(prefixes, axes) -> list[Slot]:
    return [Slot(f"m_{p}{a}", "pml_mem", a) for p in prefixes for a in axes]


#: ``ElasticWavefieldTensor::bind()`` -- 5 physical + 10 CPML memory variables.
#: No shadows and no rotating block: every field updates in place at a fixed
#: slot, so ``pairs()`` and ``u_blocks`` both come out empty, matching the
#: explicit ``()``s the DD driver passes today.
ELASTIC2D = SlotTable(
    slots=tuple([
        Slot("vx", "vel"), Slot("vz", "vel"),
        Slot("sxx", "stress"), Slot("szz", "stress"), Slot("sxz", "stress"),
    ] + _elastic_mem(("vx", "vz", "sxx", "szz", "sxz"), ("x", "z"))),
    recon=("vx", "vz", "sxx", "szz", "sxz", "fvx_prev", "fvz_prev"),
    aux_storage="slab",
)

ELASTIC3D = SlotTable(
    slots=tuple([
        Slot("vx", "vel"), Slot("vy", "vel"), Slot("vz", "vel"),
        Slot("sxx", "stress"), Slot("syy", "stress"), Slot("szz", "stress"),
        Slot("sxy", "stress"), Slot("sxz", "stress"), Slot("syz", "stress"),
    ] + _elastic_mem(("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz"),
                     ("x", "y", "z"))),
    recon=("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz",
           "fvx_prev", "fvy_prev", "fvz_prev"),
    aux_storage="slab",
)
