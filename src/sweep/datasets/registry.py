"""Public registry of benchmark models and the ``load(...)`` dispatcher.

This is the unified entry point for both the *embedded* demo models (tiny
base85 blobs that ship inside the wheel) and the *downloadable* full-size
benchmarks (Marmousi, Overthrust, BP, Hess, ...). Every record carries its
own ``license`` / ``citation`` string, which is printed the first time a
model is materialised so that attribution/usage terms travel with the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Entry:
    """One registry record.

    Attributes
    ----------
    name, variant
        The ``(name, variant)`` key used to look the model up.
    loader
        Callable returning a dict with at least ``vp`` and ``dh``. It may
        download and cache data on first call.
    kind
        ``"embedded"`` for in-wheel demo blobs (no network), ``"download"``
        for benchmarks fetched on demand, ``"gated"`` for entries that
        require the user to accept external terms / place files manually.
    license
        Human-readable license / usage terms. Printed on first load.
    redistributable
        ``True`` only when the license permits re-hosting the bytes
        (e.g. CC-BY-4.0). ``False`` for "download-and-cite / AS-IS /
        non-commercial" data that must be fetched from the official source.
    """

    name: str
    variant: str
    loader: Callable[..., dict[str, Any]]
    description: str = ""
    citation: str = ""
    license: str = ""
    kind: str = "download"
    redistributable: bool = False
    default: bool = False  # canonical variant picked when `variant` is omitted
    tags: list[str] = field(default_factory=list)


_REGISTRY: dict[tuple[str, str], Entry] = {}
_ANNOUNCED: set[tuple[str, str]] = set()


def register(entry: Entry, *, replace: bool = False) -> None:
    """Register a benchmark loader.

    Parameters
    ----------
    replace
        If ``True``, silently overwrite an existing key instead of raising.
    """
    key = (entry.name, entry.variant)
    if key in _REGISTRY and not replace:
        raise ValueError(f"benchmark {entry.name}:{entry.variant} already registered")
    _REGISTRY[key] = entry


def available() -> list[tuple[str, str]]:
    """List ``(name, variant)`` pairs registered."""
    return sorted(_REGISTRY.keys())


def info(name: str, variant: str | None = None) -> Entry:
    """Look up a registry entry.

    If ``variant`` is omitted: use the sole variant when there is only one;
    otherwise use the one marked ``default=True``. If a name has several
    variants and none (or more than one) is marked default, raise and list
    the choices rather than guess.
    """
    if variant is None:
        matches = sorted(k for k in _REGISTRY if k[0] == name)
        if not matches:
            raise KeyError(
                f"unknown benchmark {name!r}; have {sorted({k[0] for k in _REGISTRY})}"
            )
        if len(matches) == 1:
            variant = matches[0][1]
        else:
            defaults = [k for k in matches if _REGISTRY[k].default]
            if len(defaults) == 1:
                variant = defaults[0][1]
            else:
                choices = [v for _, v in matches]
                why = "no default" if not defaults else "multiple defaults"
                raise KeyError(
                    f"{name!r} has multiple variants ({why}); "
                    f"pass variant=... one of {choices}"
                )
    try:
        return _REGISTRY[(name, variant)]
    except KeyError:
        raise KeyError(f"unknown variant {name}:{variant!r}")


def catalog(name: str | None = None) -> list[Entry]:
    """Pretty-print the dataset registry and return the matching entries.

    Pass ``name`` to filter to one model family (e.g. ``catalog("bp-2004")``).
    Columns: kind (embedded/download), re-host (may the bytes be re-hosted),
    verified (byte-verified against a real download), and the license.
    """
    entries = [info(n, v) for (n, v) in available() if name in (None, n)]
    if not entries:
        print(f"(no datasets match {name!r})")
        return entries
    keys = [f"{e.name}:{e.variant}" for e in entries]
    w = max(len(k) for k in keys)
    print(f"{'name:variant'.ljust(w)}  kind      re-host  verified  license")
    print(f"{'-' * w}  --------  -------  --------  -------")
    for e, key in zip(entries, keys):
        rehost = "yes" if e.redistributable else "no"
        verified = "no" if "unverified" in e.tags else "yes"
        print(f"{key.ljust(w)}  {e.kind:8s}  {rehost:7s}  {verified:8s}  {e.license[:46]}")
    return entries


def _announce(entry: Entry) -> None:
    """Print citation + license once per process for a given entry."""
    key = (entry.name, entry.variant)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    bits = [f"[sweep.datasets] {entry.name}:{entry.variant}"]
    if entry.citation:
        bits.append(f"  cite:    {entry.citation}")
    if entry.license:
        bits.append(f"  license: {entry.license}")
    if entry.kind == "download" and not entry.redistributable:
        bits.append(
            "  note:    fetched from the official source under the terms above; "
            "do not re-host the bytes."
        )
    if "unverified" in entry.tags:
        bits.append(
            "  CAUTION: loader implemented from documented specs but not yet "
            "byte-verified against a real download; confirm grid shape / units."
        )
    print("\n".join(bits))


def load(name: str, variant: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Load (and download if needed) a benchmark dataset.

    Returns a dict with at least ``vp`` and ``dh``. Most entries also
    provide ``dt``, ``nt``, ``geometry``, and ``vp_init``. The license and
    citation are printed once per process on first load.
    """
    entry = info(name, variant)
    _announce(entry)
    out = entry.loader(**kwargs)
    out.setdefault("name", entry.name)
    out.setdefault("variant", entry.variant)
    out.setdefault("citation", entry.citation)
    out.setdefault("license", entry.license)
    return out


__all__ = ["Entry", "register", "available", "info", "load", "catalog"]
