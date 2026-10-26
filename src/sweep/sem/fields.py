"""Field specification for SEM equations — the single declarative source of
truth for which evolving fields an equation carries, in what order, and each
field's role. Mirrors ``sweep.equations.fields.FieldSpec`` (the FD backend).

The equation declares ``field_specs: list[FieldSpec]``. That ordered list IS the
positional layout of the state tuple that flows through ``init_state`` / ``step``
(constants live on the equation instance, never in the state). The generic
:class:`~sweep.sem.propagator.SEMPropagator`, the vmap batching, and the
boundary-saving framework all derive everything they need (which fields take a
source, which are sampled by receivers, which are CPML memory variables) from
this spec — no hard-coded field names anywhere outside the equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FieldSpec:
    """One evolving field of an equation."""

    name: str
    supports_source: bool = False     # a point source injects into this field
    supports_receiver: bool = False   # receivers sample this field
    internal: bool = False            # history / auxiliary (not source/receiver)
    boundary_related: bool = False    # CPML memory variable (zeroed in BS reverse)
    seis_key: str = ""                # legacy dict key; see resolve_field_types
    snap_key: str = ""                # legacy snapshot dict key
    # Storage layout, used to DERIVE the field's zero tensor (see zeros_for):
    #   "global"    (n_global,)             continuous-Galerkin nodal field
    #   "element"   (E, ngll**dim)          element-local / DG-style
    #   "element2"  (E, ngll**dim, 2)       element-local with a 2-vector per node
    # Until now each equation hand-wrote init_state and the layouts survived only
    # as a comment above the return tuple -- a comment that can drift from the
    # tuple below it without anything noticing.
    layout: str = "global"


LAYOUTS = ("global", "element", "element2")


def zeros_for(spec: FieldSpec, mesh, *, device, dtype):
    """The zero tensor for one field, from its declared ``layout``."""
    if spec.layout == "global":
        return torch.zeros(mesh.n_global, dtype=dtype, device=device)
    dim = 3 if hasattr(mesh, "ney") else 2
    shape = (mesh.n_elements,) + (mesh.ngll,) * dim
    if spec.layout == "element":
        return torch.zeros(shape, dtype=dtype, device=device)
    if spec.layout == "element2":
        return torch.zeros(shape + (2,), dtype=dtype, device=device)
    raise ValueError(
        f"field {spec.name!r} has unknown layout {spec.layout!r}; "
        f"valid: {LAYOUTS}")


def init_state_from_specs(eq, device, dtype) -> tuple:
    """The whole state tuple, derived from ``eq.field_specs``.

    Equations that need nothing more than "one zero tensor per declared field"
    -- which is all of them today -- inherit this instead of repeating the
    allocation. The ORDER is ``field_specs`` order, which is already the
    contract every other part of the package relies on.
    """
    return tuple(zeros_for(s, eq.mesh, device=device, dtype=dtype)
                 for s in eq.field_specs)


class DerivedInitState:
    """Mixin: ``init_state`` derived from ``field_specs``.

    Every SEM equation's ``init_state`` was the same three lines -- allocate one
    zero tensor per field -- differing only in HOW MANY and in each field's
    shape. Both are already declared (``field_specs`` and its ``layout``), so
    the method is derivable and the 13 copies are pure ceremony. An equation
    whose state needs more than zeros can still override it.
    """

    def init_state(self, dt: float, device, dtype) -> tuple:
        return init_state_from_specs(self, device, dtype)


class DirichletNodes:
    """Mixin: zero a field on the Dirichlet node set.

    A pressure-release (free) surface in the acoustic potential/displacement
    forms is a node set the step zeroes every iteration. The five acoustic
    equations carried a byte-identical copy of this; the equation only has to
    declare ``dirichlet_idx``.
    """

    def _apply_dirichlet(self, *fields):
        idx = self.dirichlet_idx
        if idx.numel() == 0:
            return fields
        return tuple(f.index_fill(0, idx, 0.0) for f in fields)


# Mesh family an equation's ``build`` takes, and the builders for it. Derived
# from the ``mesh`` annotation on ``build`` rather than declared, so it cannot
# drift from the signature it describes.
_MESH_ANNOT = {"Quad4Mesh": "quad4", "Quad9Mesh": "quad9",
               "Hex8Mesh": "hex8", "Hex27Mesh": "hex27"}
MESH_NDIM = {"quad4": 2, "quad9": 2, "hex8": 3, "hex27": 3}


def mesh_kind(equation) -> str:
    """``"quad4"`` / ``"quad9"`` / ``"hex8"`` / ``"hex27"`` for an equation.

    PropSEM used to pick the mesh with a chain of ``is``/``in`` tests against
    hand-kept tuples, in four separate places, so "is this equation deformed?"
    had four answers that could disagree. It is one question, answered by the
    signature of ``build``.
    """
    import inspect
    ann = str(inspect.signature(equation.build).parameters["mesh"].annotation)
    ann = ann.replace("'", "").rsplit(".", 1)[-1]
    kind = _MESH_ANNOT.get(ann)
    if kind is None:
        raise ValueError(
            f"{equation.__name__}.build has mesh annotation {ann!r}; annotate "
            f"it with one of {sorted(_MESH_ANNOT)} so the mesh can be derived.")
    return kind


def model_names(equation) -> list:
    """Positional model arguments of ``build``, e.g. ``["vp", "vs", "rho"]``.

    Also derived: the equations do not agree on a fixed set (the acoustic
    potential form takes ``vp, rho``; VTI adds ``epsilon, delta``), so a
    two-way acoustic/elastic split in the propagator was always going to have
    exceptions bolted onto it.
    """
    import inspect
    p = list(inspect.signature(equation.build).parameters.values())
    return [q.name[:-2] if q.name.endswith("_e") else q.name
            for q in p[1:] if q.kind is not q.KEYWORD_ONLY]


def format_field_specs(specs) -> str:
    return "\n".join(f"  {s.name}" for s in specs)


def resolve_field_types(equation, kinds, role):
    """Validate a ``source_type`` / ``receiver_type`` list against an equation.

    Mirrors the FD ``PropBase._resolve_field_types``: a list of field names,
    checked against ``field_specs`` and its ``supports_source`` /
    ``supports_receiver`` flags, with the valid options listed in the error.
    ``field_specs`` is class-level on the SEM equations, so this can run at
    construction rather than at the first forward -- same timing as FD.

    Lives here, next to :class:`FieldSpec`, because BOTH the low-level
    :class:`~sweep.sem.propagator.SEMPropagator` and the FD-facing
    :class:`~sweep.sem.prop_base.PropSEM` need it, and the latter imports the
    former.

    The valid names come from the equation, and SEM ships elastic equations in
    BOTH formulations -- do not assume one from the other:

    ==========================================  ==================  ==========
    equation                                    formulation         names
    ==========================================  ==================  ==========
    ``ElasticSEM2D`` / ``2DDeformed`` / ``3D``  2nd-order           ``u_x``,
                                                displacement        ``u_z``
                                                (SPECFEM/Newmark)
    ``ElasticVSSEM2DDeformed`` / ``VSSEM3D``    1st-order           ``v_x``,
                                                velocity-stress     ``v_z``
    ==========================================  ==================  ==========

    Neither exposes a stress receiver. In the velocity-stress family the
    stresses DO exist as fields, element-local (``E, ngll, ngll``, DG-style,
    so material jumps can live on element faces), and the receiver gather
    reads an element-local field from the receiver's own element (Biot's flux
    and stresses are recorded that way); the elastics just do not flag theirs
    ``supports_receiver``. In the displacement family there is no
    stress state at all; it is formed inside the stiffness apply and never
    stored.

    So FD's ``vx``/``sxx`` names transfer only partially on the elastics:
    ``v_x`` exists in the velocity-stress family, ``sxx`` in neither.
    """
    attr = "supports_source" if role == "source" else "supports_receiver"
    specs = list(equation.field_specs)
    avail = [s for s in specs if getattr(s, attr, False)]
    # FD-aligned DERIVED receiver components (receivers only, never sources):
    # the equation declares {name: (base_field, op)}, e.g. the fluid-solid
    # form's {"p": ("chi", "pressure"), "vx": ("u_x", "velocity")}. Field data
    # arrives as pressure / particle velocity, so the solver does the
    # conversion (p = -chi_ddot, v = u_dot) instead of every caller.
    derived = getattr(equation, "derived_receiver_types", {}) \
        if role == "receiver" else {}
    if not kinds:
        # Receivers default to every component. Sources default to the LAST
        # source field, which for the elastic equations is the z displacement
        # / velocity -- i.e. the historical unit VERTICAL force. Defaulting to
        # "all" here would silently turn that into a diagonal force. An
        # equation may override (the fluid-solid form defaults to chi: a
        # scalar source in the water column).
        if role == "receiver":
            # Biot records its solid velocity by default, as the FD Biot does,
            # not all eight of its receiver fields
            declared = getattr(equation, "default_receiver_type", None)
            return list(declared) if declared else [s.name for s in avail]
        declared = getattr(equation, "default_source_type", None)
        return list(declared) if declared else [avail[-1].name]
    name = getattr(equation, "__name__", type(equation).__name__)
    derived_note = ("\nDerived receiver components: "
                    + ", ".join(sorted(derived))) if derived else ""
    out = []
    for want in kinds:
        if want in derived:
            base = derived[want][0]
            spec = next((s for s in specs if s.name == base), None)
            if spec is None or not getattr(spec, "supports_receiver", False):
                raise ValueError(
                    f"derived receiver '{want}' on {name} maps to '{base}', "
                    f"which is not a receiver-capable field -- the "
                    f"derived_receiver_types declaration is wrong.")
            if want in out:
                raise ValueError(f"duplicate {role}_type entry '{want}'")
            out.append(want)
            continue
        spec = next((s for s in specs if s.name == want), None)
        if spec is None:
            raise ValueError(
                f"Unknown {role}_type entry '{want}' for {name}. Available "
                f"{role}_type values:\n{format_field_specs(avail)}"
                f"{derived_note}")
        if not getattr(spec, attr, False):
            raise ValueError(
                f"Field '{want}' is not valid for {role}_type on {name}. "
                f"Available {role}_type values:\n{format_field_specs(avail)}"
                f"{derived_note}")
        if spec.name in out:
            raise ValueError(f"duplicate {role}_type entry '{want}'")
        out.append(spec.name)
    return out

def check_element_arrays(mesh, **arrays):
    """Refuse a model array that is not ``(E, ngll, ..., ngll)`` on this mesh.

    Twelve equations open ``build`` with this loop, in a 2-D and a 3-D variant
    whose wording had drifted apart. The mesh knows its own dimension
    (``ibool`` is ``(E,) + (ngll,) * ndim``), so the caller passes only the
    arrays, by keyword, which is what keeps each message pointing at the right
    argument.
    """
    ng, ndim = mesh.ngll, mesh.ibool.dim() - 1
    want = (mesh.n_elements,) + (ng,) * ndim
    axes = ", ".join(["ngll"] * ndim)
    for name, t in arrays.items():
        if tuple(t.shape) != want:
            raise ValueError(
                f"{name} shape {tuple(t.shape)} != (E, {axes}) = {want}")
