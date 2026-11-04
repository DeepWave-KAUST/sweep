"""The step's time-invariant coefficients, made once per rollout.

A step multiplies its wavefields by coefficients that depend on the models and
``dt`` alone -- ``dt / rho``, ``vp**2 * dt**2``, ``dt * mu``.  Recomputed every
step, each one is a new field, and the product that reads it keeps that field
for the backward: an uncompiled 2-D Elastic rollout kept three of them per step
next to the five fields its gradient needs.

An uncompiled eager rollout (full tape, and each checkpointed chunk) hands its
step a fresh dict, ``invariants``; :func:`once` computes a coefficient the first
time a step asks for it and gives every later step the same tensor.  The
values are bitwise those computed in place, so records do not change; the
model's gradient accumulates through one node instead of one per step, which
moves it by rounding.  Wherever ``invariants`` is None -- compiled steps,
boundary saving's reverse sub-steps at ``-dt``, other callers -- the coefficient
is computed in place, as before.

Only an expression of the models, ``dt`` and constants may go through
:func:`once`; a key names one expression of one step.
"""


def once(invariants, key, compute):
    """``compute()``, made once per rollout when ``invariants`` is a dict."""
    if invariants is None:
        return compute()
    value = invariants.get(key)
    if value is None:
        value = invariants[key] = compute()
    return value
