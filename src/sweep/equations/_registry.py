"""Registry mapping public names to :class:`WaveEquation` subclasses.

``@register_equation`` co-locates an equation's public export name(s) with its
class, replacing the hand-maintained import wall + ``globals()`` scan in
``__init__.py``. Adding an equation is then "write the module, decorate the
class" -- no edits here and none in ``__init__.py``.

``method`` tags which *discretization method* an equation belongs to. sweep's
finite-difference stack is ``"fd"`` (the default); a sibling method such as SEM
registers with ``method="sem"``, so one registry can serve both while their
implementations stay separate (no shared base class).
"""
from __future__ import annotations

# public name (canonical + every alias) -> class
_REGISTRY: dict[str, type] = {}
# class -> its canonical (primary) public name
_CANONICAL: dict[type, str] = {}
# class -> discretization-method tag
_METHOD: dict[type, str] = {}


def register_equation(name=None, *, aliases=(), method="fd"):
    """Class decorator registering a :class:`WaveEquation` subclass for discovery.

    Args:
        name: canonical public export name; defaults to ``cls.__name__``.
        aliases: extra public names resolving to the same class (author-named
            entry points, 2-D/3-D "Default" routing, back-compat aliases).
        method: discretization-method tag. ``"fd"`` (default) is the
            finite-difference stack; ``"sem"`` is the spectral-element sibling.

    A name may only be claimed once. Re-registering the *same* logical class
    (same module + qualname -- what a notebook re-run or ``importlib.reload``
    produces) is allowed, because that is a redefinition rather than a clash.
    """
    def decorate(cls):
        canonical = name or cls.__name__
        _CANONICAL[cls] = canonical
        _METHOD[cls] = method
        for public in (canonical, *aliases):
            existing = _REGISTRY.get(public)
            if (
                existing is not None
                and existing is not cls
                and (existing.__module__, existing.__qualname__)
                != (cls.__module__, cls.__qualname__)
            ):
                raise ValueError(
                    f"equation name {public!r} already registered to "
                    f"{existing.__module__}.{existing.__qualname__}; "
                    f"cannot reassign to {cls.__module__}.{cls.__qualname__}"
                )
            _REGISTRY[public] = cls
        return cls

    return decorate


def get_equation(name):
    """Return the equation class registered under ``name`` (canonical or alias)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown equation {name!r}; available: {list_equations()}"
        ) from None


def equation_classes(method=None):
    """Mapping of every registered public name (canonical + aliases) -> class.

    Args:
        method: optional discretization-method filter (``"fd"`` / ``"sem"``);
            ``None`` (default) returns every registered name.
    """
    if method is None:
        return dict(_REGISTRY)
    return {n: cls for n, cls in _REGISTRY.items() if _METHOD.get(cls, "fd") == method}


def equation_method(name):
    """Return the discretization-method tag for ``name`` (``"fd"`` / ``"sem"`` / ...)."""
    return _METHOD.get(get_equation(name), "fd")


def list_equations(method=None):
    """Canonical equation names (no aliases), sorted.

    Args:
        method: if given, restrict to equations of that discretization method.
    """
    return sorted(
        {
            canonical
            for cls, canonical in _CANONICAL.items()
            if method is None or _METHOD.get(cls, "fd") == method
        }
    )
