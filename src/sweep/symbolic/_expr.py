"""Symbols of the PDE front end: models, fields and the derivative atoms.

Models and fields are sympy symbols, so coefficients are ordinary sympy
expressions (``vp**2 * (1 + 2*eps)``). Derivatives are unevaluated function
atoms (``D2(p, x)``, ``Lap(p)``, ``Dt(p, 2)``): sympy keeps them intact through
``expand`` and ``diff``, and the lowering reads them back off the expression.
``laplace`` stays one atom until the dimension is known at ``build``.
"""
from __future__ import annotations

import itertools
import keyword

import sympy
from sympy.core.symbol import Str

AXES = ("z", "y", "x")

# Names the generated step uses for itself. Every generated helper starts with
# an underscore, which a user name may not.
_RESERVED = {"self", "dt", "h", "b", "kwargs", "torch", "wavefields", "models",
             "cpml_axis_update", "FieldSpec", "ModelSpec"}

_declared = itertools.count()


def _check_name(name, kind):
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name {name!r} is not a Python identifier")
    if name.startswith("_") or name in _RESERVED:
        raise ValueError(f"{kind} name {name!r} is reserved for the generated code")


class Model(sympy.Symbol):
    """A model parameter (``vp``, ``rho``, ``epsilon`` ...), passed to the
    propagator in declaration order unless ``PDE(models=...)`` says otherwise."""

    def __new__(cls, name, **kwargs):
        _check_name(name, "model")
        return sympy.Symbol.__xnew__(cls, name)

    def __init__(self, name, *, unit="", aliases=(), description=""):
        self.unit = unit
        self.aliases = tuple(aliases)
        self.description = description or f"Model parameter `{name}`."
        self._order = next(_declared)


class Field(sympy.Symbol):
    """A wavefield advanced by the PDE.

    Args:
        name: the field name, as ``source_type`` / ``receiver_type`` see it.
        time_order: order of the time derivative of its equation (2 only).
        source: the field can be a source field.
        receiver: the field can be recorded.
        aliases: extra names for ``source_type`` / ``receiver_type``.
        description: shown by ``describe_field``.
    """

    def __new__(cls, name, **kwargs):
        _check_name(name, "field")
        return sympy.Symbol.__xnew__(cls, name)

    def __init__(self, name, *, time_order=2, source=False, receiver=False,
                 aliases=(), description=""):
        if time_order != 2:
            raise NotImplementedError(
                f"field {name!r}: time_order={time_order!r}; only second-order "
                "fields (time_order=2) are supported so far")
        self.time_order = time_order
        self.source = bool(source)
        self.receiver = bool(receiver)
        self.aliases = tuple(aliases)
        self.description = description or f"Wavefield `{name}`."
        self._order = next(_declared)

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
