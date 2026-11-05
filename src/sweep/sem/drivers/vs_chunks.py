"""Checkpointing in chunks for the velocity-stress CUDA drivers (2-D and 3-D,
elastic and Biot).

The save-all adjoint keeps, per step, the velocities and the stretched velocity
gradients -- four (2-D) or nine (3-D) element-local arrays, several times the state
itself -- so a long record outgrows the device. With ``memory=Ckpt(chunks=K)`` the
forward keeps only the state at the start of every K-step chunk; the backward
reruns each chunk from its checkpoint with the histories on (one more forward in
all) and runs the adjoint through it, the adjoint state carried from chunk to chunk.
Device memory: ``nt / K`` states plus one chunk of histories, against ``nt`` steps of
histories for the save-all. The entries take the chunk as their step range
``[t0, t1)``; records and the wavelet keep the global step index.
"""

from __future__ import annotations

# what a velocity-stress forward carries from one step to the next (the divergence
# and its split are rebuilt every step); the fields a struct lacks or leaves empty
# (no y in 2-D, no C-PML, no GSLS memories) are skipped
_STATE = ("vx", "vy", "vz", "sig", "mvel", "msig", "rmem",
          "p", "q", "r", "mp", "mv", "mq")          # the last six: Biot's


def state_fields(s):
    """The names of the state fields of the forward struct ``s``."""
    out = []
    for nm in _STATE:
        t = s.__dict__.get(nm)
        if t is not None and t.numel() > 0:
            out.append(nm)
    return tuple(out)


def chunk_starts(nt, K):
    return list(range(0, int(nt), int(K)))


def forward_chunks(entry, s, nt, K):
    """Run the forward ``entry`` on ``s`` over the steps in ``K``-step chunks and
    return the state at the start of each chunk: ``(names, [states])``."""
    names = state_fields(s)
    ckpts = []
    for t in chunk_starts(nt, K):
        ckpts.append(tuple(s.__dict__[nm].clone() for nm in names))
        s.t0, s.t1 = t, min(t + K, nt)
        entry(s)
    s.t0, s.t1 = 0, 0
    return names, ckpts


def backward_chunks(fwd_entry, bwd_entry, f, b, nt, K, names, ckpts):
    """The chunks in reverse: restore each checkpoint into the recompute struct
    ``f``, rerun its steps with the histories on (``b`` holds the same history
    tensors), then the adjoint entry over those steps."""
    for t, ck in zip(reversed(chunk_starts(nt, K)), reversed(ckpts)):
        for nm, v in zip(names, ck):
            f.__dict__[nm].copy_(v)
        f.t0, f.t1 = t, min(t + K, nt)
        fwd_entry(f)
        b.t0, b.t1 = f.t0, f.t1
        bwd_entry(b)


def chunk_of(c, need_bwd):
    """The chunk length the run takes: ``None`` for the save-all (or a run with no
    adjoint, which saves nothing), else ``c["ckpt"]``; a chunk as long as the record
    is the save-all."""
    K = c.get("ckpt")
    if not need_bwd or K is None or int(K) >= int(c["nt"]):
        return None
    if int(K) < 1:
        raise ValueError(f"Ckpt(chunks={K}): a chunk is at least one step")
    return int(K)
