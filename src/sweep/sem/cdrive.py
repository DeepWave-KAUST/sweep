"""Shared plumbing of the ``impl='c'`` autograd drivers (``prop_c.py``, ``dd.py``).

Every driver does the same things around its C entry calls: copy the constant
tables into the pybind struct, set the scalar dims, allocate the per-step
state (some of it as the rows of ONE buffer so a single memset re-zeroes a
whole family), pick the memory mode -- boundary rings, save-all tapes or
nothing -- and carry the forward's saves over to the backward. This module
holds those steps so a driver keeps only what is specific to its equation:
which fields exist, and the order its C entries run in.

Two rules the helpers enforce that the hand-written drivers could not:

* a struct goes to a C entry only after :func:`check_filled` -- an unset
  tensor field is a null pointer on the other side, which crashes or reads
  garbage instead of raising;
* the forward's saves reach the backward by NAME (:func:`take` /
  :func:`give`), never as a positional tuple both sides must keep in step.
"""

from __future__ import annotations

import torch


class Alloc:
    """Zero-filled fp32 buffers for one part of a driver.

    ``n`` is the length of a per-node field (``B * n_global`` under a shot
    batch), ``nt`` the record length of a tape. ``empty`` is the zero-length
    tensor a driver hands to the fields an entry will not read.
    """

    def __init__(self, dev, n, nt=0):
        self.dev, self.n, self.nt = dev, int(n), int(nt)
        self.empty = torch.empty(0, dtype=torch.float32, device=dev)

    def zeros(self, n=None):
        return torch.zeros(self.n if n is None else int(n),
                           dtype=torch.float32, device=self.dev)

    def block(self, k, n=None):
        """``k`` fields as the rows of ONE ``(k, n)`` buffer: unpack it into
        the fields, keep it where an exchange sends the whole family.

        A C entry that re-zeroes the family every step does it with a single
        memset from the first row, so the row order is part of the contract
        with that entry: unpack in the order the entry expects.
        """
        return torch.zeros(k, self.n if n is None else int(n),
                           dtype=torch.float32, device=self.dev)

    def chunks(self, sizes, with_parent=False):
        """Like :meth:`block` for fields of different lengths.

        ``with_parent`` also returns the ONE buffer they are views of, which
        is what a heterogeneous block exchange indexes (``dd._DDExchange
        .block``) and what a C entry re-zeroes with a single memset.
        """
        buf = torch.zeros(sum(int(s) for s in sizes),
                          dtype=torch.float32, device=self.dev)
        parts = tuple(torch.split(buf, [int(s) for s in sizes]))
        return (parts, buf) if with_parent else parts

    def hist(self, n=None, zero=True):
        """One save-all tape: ``(nt, n)``.

        ``zero=False`` skips the fill, for a tape whose every entry the forward
        writes before the backward reads it. That is most of them, and the fill
        is not free: at nex=160, nt=200 the 2-D elastic save-all path spends
        **13% of its device time** (4.26 of 32.7 ms) zeroing tapes it
        immediately overwrites -- its six per-element-node histories are 3.07 GB
        there.

        Which tapes qualify was MEASURED, not reasoned: filling every tape with
        NaN and running the suite leaves 707 tests green and fails only the five
        that exercise the 2-D acoustic w_sum tape, whose ``combine`` legitimately
        reads a row the forward never wrote. So that one keeps its zeros and the
        rest do not need them.

        The default stays ``True`` on purpose: a tape added later should get
        zeros until someone measures it, not garbage because nobody thought
        about it.
        """
        t = torch.empty(self.nt, self.n if n is None else int(n),
                        dtype=torch.float32, device=self.dev)
        return t.zero_() if zero else t


def memory_mode(c, need_bwd):
    """What the forward keeps for the backward: ``'ring'`` (the C-PML band
    per step, asked for with ``memory='boundary'``), ``'tape'`` (the save-all
    history) or ``'none'`` (no backward will run)."""
    if not need_bwd:
        return "none"
    return "ring" if bool(c.get("bs")) else "tape"


# The nine inv_J streams of a deformed 3-D mesh, under the names every deformed
# kernel indexes them by and in the ``eq._I`` order the producer emits. The
# drivers' fill lists are built FROM this instead of repeating it: there were
# three copies, and a metric name list that drifts from its producer is the
# same class of failure as the columnar flag that reached only one of the two
# construction sites (see prop_c.metric_consts).
METRIC_NAMES = ("ixx", "ixy", "ixz", "iex", "iey", "iez", "igx", "igy", "igz")


# The elastic C-PML coefficient set, under the names the C struct gives them.
# Six of the twenty-eight arrive as (coef0, coef1, coef2) triples on the
# profile object and flatten to "<nm>_<j>"; EL_CPML_FLAT is that flattened form,
# which is what a driver's fill list needs. There were four copies of these
# names -- prop_c and dd for the producer side, elastic2d and fluid_solid for
# the fill side -- kept in step only by fill() raising on a drift.
EL_CPML_FIELDS = ("Sx0", "Sx1", "Sx2", "Sz0", "Sz1", "Sz2",
                  "La1", "La2", "La3", "La4")
EL_CPML_TRIPLES = ("zx1", "zx2", "xz1", "xz2", "l1", "l2")
EL_CPML_FLAT = EL_CPML_FIELDS + tuple(
    f"{nm}_{j}" for nm in EL_CPML_TRIPLES for j in range(3))
AC_CPML_FIELDS = ("a_x", "b_x", "dbx_dx", "a_z", "b_z", "dbz_dz")


def fill(s, c, keys):
    """Copy the constant tables ``keys`` of the consts dict onto the struct."""
    for k in keys:
        setattr(s, k, c[k])


def put(s, names, make):
    """``s.<name> = make()`` for every name; a tensor ``make`` is shared as is."""
    for nm in names:
        setattr(s, nm, make() if callable(make) else make)


def dims(s, c, ndim, **ints):
    """The scalar dims every entry reads, plus any driver-specific ints."""
    s.dt, s.nt, s.ngll = c["dt"], c["nt"], c["ngll"]
    s.n_elements, s.n_global, s.n_rec = c["n_elements"], c["n_global"], c["n_rec"]
    s.ndim = ndim
    for k, v in ints.items():
        setattr(s, k, int(v))


def check_filled(s):
    """Raise on any tensor field the driver left unset (see the module doc)."""
    unset = [k for k in dir(s) if not k.startswith("_") and getattr(s, k) is None]
    if unset:
        raise RuntimeError(f"{type(s).__name__}: {len(unset)} field(s) never "
                           f"set by the driver: {unset}")
    return s


def take(s, names):
    """The named fields of a struct, as a dict the backward can :func:`give`."""
    return {nm: getattr(s, nm) for nm in names}


def give(s, saved, rename=()):
    """Set the fields of ``saved`` on ``s``; ``rename`` maps a forward name
    onto the backward struct's name for it where the two differ."""
    r = dict(rename)
    for nm, t in saved.items():
        setattr(s, r.get(nm, nm), t)


def swap(s, pairs):
    """Exchange the two fields of every pair (the ``x`` / ``x_n`` carries)."""
    for a, b in pairs:
        x = getattr(s, a)
        setattr(s, a, getattr(s, b))
        setattr(s, b, x)


def roll(s, names):
    """``(a, b, c) <- (b, c, a)``: the time levels advance one step and the
    oldest buffer becomes the next scratch."""
    vals = [getattr(s, nm) for nm in names]
    for nm, v in zip(names, vals[1:] + vals[:1]):
        setattr(s, nm, v)


def grads_or_none(inputs, grads):
    """Autograd's return: a gradient where its input asked for one, else None."""
    return tuple(g if t.requires_grad else None for t, g in zip(inputs, grads))


def all_reduce_(t):
    """Sum ``t`` over the ranks when a process group is up; no-op otherwise."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t)
    except Exception:
        pass
    return t


def dd_grads(ctx, bs_, grads):
    """A strip backward's return tuple: the wavelet gradient, then ``None``
    for ``parts``/``exchange``/``bs``, then the material gradients.

    The wavelet's gradient is a global scalar reduction per step, so every
    strip holds a partial and they sum -- across the strips this rank owns,
    then across ranks. The material gradients are already deduplicated by the
    caller (elements are uniquely owned; per-node ones are masked by
    ``part.own``), so they pass through as they are.
    """
    gw = bs_[0].grad_wavelet
    for b in bs_[1:]:
        gw = gw + b.grad_wavelet
    return (all_reduce_(gw) if ctx.needs_input_grad[0] else None,
            None, None, None, *grads)
