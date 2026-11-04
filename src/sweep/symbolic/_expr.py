"""Symbols of the PDE front end: models, coefficients, fields and the derivative atoms.

Models, coefficients and fields are sympy symbols, so coefficients are ordinary
sympy expressions (``vp**2 * (1 + 2*eps)``). Derivatives are unevaluated
function atoms (``D2(p, x)``, ``D1(vx, x)``, ``Lap(p)``, ``Dt(p, 2)``): sympy
keeps them intact through ``expand`` and ``diff``, and the lowering reads them
back off the expression. ``laplace`` stays one atom until the dimension is known
at ``build``.
"""
from __future__ import annotations

import itertools
import keyword

import sympy
from sympy.core.symbol import Str

AXES = ("z", "y", "x")

# Names the generated step uses for itself. Every generated helper starts with
# an underscore, which a user name may not.
_RESERVED = {"self", "dt", "h", "b", "kwargs", "torch", "wavefields", "models", "pd",
             "cpml_axis_update", "cpml_update", "FieldSpec", "ModelSpec"}

_declared = itertools.count()


def _check_name(name, kind):
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name {name!r} is not a Python identifier")
    if name.startswith("_") or name in _RESERVED:
        raise ValueError(f"{kind} name {name!r} is reserved for the generated code")


class _Declared(sympy.Symbol):
    """A symbol that is its declaration: sympy compares symbols by name and
    caches expressions on that, so a field or model declared again under the
    same name (a notebook cell run twice) would hand the old object, with its
    old attributes, back out of a cached ``vz.dz`` or ``vp**2``."""

    def __new__(cls, name, kind):
        _check_name(name, kind)
        obj = sympy.Symbol.__xnew__(cls, name)
        obj._decl = next(_declared)
        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self._decl,)


class Model(_Declared):
    """A model parameter (``vp``, ``rho``, ``epsilon`` ...), passed to the
    propagator in declaration order unless ``PDE(models=...)`` says otherwise."""

    def __new__(cls, name, **kwargs):
        return _Declared.__new__(cls, name, "model")

    def __init__(self, name, *, unit="", aliases=(), description=""):
        self.unit = unit
        self.aliases = tuple(aliases)
        self.description = description or f"Model parameter `{name}`."
        self._order = next(_declared)


class Coefficient(_Declared):
    """A coefficient derived from the models, such as the Lamé parameters.

    ``Coefficient("mu", rho * vs**2)`` is computed once per run, before the
    time loop, in declaration order (a coefficient may use earlier ones), and
    the equations use it by name. A coefficient written inline is computed
    again at every step instead.
    """

    def __new__(cls, name, expr, **kwargs):
        return _Declared.__new__(cls, name, "coefficient")

    def __init__(self, name, expr, *, description=""):
        expr = sympy.sympify(expr)
        if expr.atoms(_Op, Lap):
            raise ValueError(f"coefficient {name!r}: {expr} contains a derivative")
        unknown = [s for s in expr.free_symbols if not isinstance(s, (Model, Coefficient))]
        if unknown:
            raise ValueError(f"coefficient {name!r}: {', '.join(map(str, unknown))} is "
                             "neither a Model nor a Coefficient")
        if not expr.free_symbols:
            raise ValueError(f"coefficient {name!r}: {expr} uses no model; write the number itself")
        self.expr = expr
        self.description = description or f"Coefficient `{name}` = `{expr}`, computed once per run."
        self._order = next(_declared)


def _parse_stagger(name, stagger):
    """``{axis: +1 | -1}``: the field sits half a cell after (+1) or before (-1)
    the integer node along each axis named."""
    if stagger is None:
        return {}
    items = (stagger,) if isinstance(stagger, str) else tuple(stagger)
    out = {}
    for s in items:
        ax = s.lstrip("+-") if isinstance(s, str) else None
        if ax not in AXES or len(s) - len(ax) > 1:
            raise ValueError(f"field {name!r}: stagger entry {s!r}; expected 'x', 'y' or 'z' "
                             "(half a cell after the node) or '-x', '-y', '-z' (before it)")
        if ax in out:
            raise ValueError(f"field {name!r}: stagger names the {ax} axis twice")
        out[ax] = -1 if s.startswith("-") else 1
    return out


class Field(_Declared):
    """A wavefield advanced by the PDE.

    Args:
        name: the field name, as ``source_type`` / ``receiver_type`` see it.
        time_order: order of the time derivative of its equation: 2
            (``Eq(p.dt2, ...)``) or 1 (``Eq(vx.dt, ...)``, a staggered-grid
            system).
        stagger: first-order fields only. Where the field sits on the
            staggered grid: ``None`` (on the integer nodes), an axis
            ``"x"`` (half a cell after the node along x), ``"-x"`` (half a
            cell before it), or several, ``("z", "x")``.
        source: the field can be a source field.
        receiver: the field can be recorded.
        aliases: extra names for ``source_type`` / ``receiver_type``.
        description: shown by ``describe_field``.
    """

    def __new__(cls, name, **kwargs):
        return _Declared.__new__(cls, name, "field")

    def __init__(self, name, *, time_order=2, stagger=None, source=False, receiver=False,
                 aliases=(), description=""):
        if time_order not in (1, 2):
            raise NotImplementedError(
                f"field {name!r}: time_order={time_order!r}; fields are first order "
                "(time_order=1) or second order (time_order=2)")
        if stagger is not None and time_order != 1:
            raise ValueError(f"field {name!r}: stagger= applies to first-order fields "
                             "(time_order=1) only")
        self.time_order = time_order
        self.stagger = _parse_stagger(name, stagger)
        self.source = bool(source)
        self.receiver = bool(receiver)
        self.aliases = tuple(aliases)
        self.description = description or f"Wavefield `{name}`."
        self._order = next(_declared)

    def position(self, axis):
        """Half-cell offset along ``axis``: -1, 0 or +1."""
        return self.stagger.get(axis, 0)

    @property
    def dt2(self):
        return Dt(self, 2)

    @property
    def dt(self):
        return Dt(self, 1)

    @property
    def dz2(self):
        return D2(self, Str("z"))

    @property
    def dy2(self):
        return D2(self, Str("y"))

    @property
    def dx2(self):
        return D2(self, Str("x"))

    @property
    def dz(self):
        return D1(self, Str("z"))

    @property
    def dy(self):
        return D1(self, Str("y"))

    @property
    def dx(self):
        return D1(self, Str("x"))


class _Op(sympy.Function):
    """An unevaluated derivative of a field."""

    nargs = 2

    @property
    def field(self):
        return self.args[0]


class Dt(_Op):
    """``d^n f / dt^n``; args ``(field, n)``."""

    @property
    def order(self):
        return int(self.args[1])

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}.dt{'' if self.order == 1 else self.order}"

    def _latex(self, printer):
        f, n = printer._print(self.args[0]), int(self.args[1])
        if n == 1:
            return r"\frac{\partial %s}{\partial t}" % f
        return r"\frac{\partial^{%d} %s}{\partial t^{%d}}" % (n, f, n)


class D2(_Op):
    """``d^2 f / d axis^2``; args ``(field, axis)``."""

    @property
    def axis(self):
        return str(self.args[1])

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}.d{self.axis}2"

    def _latex(self, printer):
        return r"\frac{\partial^{2} %s}{\partial %s^{2}}" % (printer._print(self.args[0]), self.axis)


class D1(_Op):
    """``d f / d axis``; args ``(field, axis)``."""

    @property
    def axis(self):
        return str(self.args[1])

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}.d{self.axis}"

    def _latex(self, printer):
        return r"\frac{\partial %s}{\partial %s}" % (printer._print(self.args[0]), self.axis)


class Lap(sympy.Function):
    """The Laplacian of a field, expanded per axis once the dimension is known."""

    nargs = 1

    @property
    def field(self):
        return self.args[0]

    def _sympystr(self, printer):
        return f"laplace({printer._print(self.args[0])})"

    def _latex(self, printer):
        return r"\nabla^{2} %s" % printer._print(self.args[0])


def laplace(f):
    """``laplace(p)``: the sum of the second derivatives of ``p`` over every axis."""
    if not isinstance(f, Field):
        raise TypeError(f"laplace() takes a Field, got {type(f).__name__}")
    return Lap(f)


class Eq:
    """One equation ``lhs = rhs`` of the system.

    Not ``sympy.Eq``, which evaluates ``Eq(a, a)`` to ``True``.
    """

    def __init__(self, lhs, rhs):
        self.lhs = sympy.sympify(lhs)
        self.rhs = sympy.sympify(rhs)

    def __repr__(self):
        return f"Eq({self.lhs}, {self.rhs})"

    def _repr_latex_(self):
        return f"$\\displaystyle {sympy.latex(self.lhs)} = {sympy.latex(self.rhs)}$"
