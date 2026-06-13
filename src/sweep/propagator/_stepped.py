"""Helpers for driving the stepped C++ forward (``it_begin``/``it_end``).

The compiled forward rotates buffer roles host-side once per time step
(``AcousticWavefieldTensor::swap_pml()``), but ``bind()`` assigns roles
purely by list position. The rotation lives in C++ handles that die with
the call, while the *contents* of the Python tensors end up role-rotated —
so a continuation call must re-order the wavefield list to keep every role
pointing at the right physical tensor.

Per completed step, ``swap_pml()`` applies:

* a left 3-cycle of the ``u`` triple buffer (slots ``0:3``), and
* a swap of each CPML ``psi`` buffer with its double-buffer partner
  (race-free forward: read old psi, write psin, swap).

After ``k`` steps, with ``L`` the original list: ``u_prev`` is in
``L[k % 3]``, ``u_now`` in ``L[(k+1) % 3]``; for odd ``k`` every
``(psi, psin)`` slot pair is exchanged. ``zeta*`` slots never move in the
forward layout.

Forward wavefield layouts (see ``acoustic.h::bind``):

* acoustic2d, 9 slots: ``u_prev,u_now,u_next, psix,psiz, zetax,zetaz,
  psixn,psizn`` → psi pairs ``(3,7),(4,8)``
* acoustic3d, 12 slots: ``u_prev,u_now,u_next, psix,psiz, zetax,zetaz,
  psiy,zetay, psixn,psizn,psiyn`` → psi pairs ``(3,9),(4,10),(7,11)``

The elastic/DAS/VTI-1st families keep every field at a fixed slot (no
``swap``), i.e. ``u_blocks=()``, ``psi_pairs=()``.

Backward (``bw_it_begin``/``bw_it_end``, descending segments) rotates TWO
independent lists:

* the adjoint wavefield list — one ``swap_aux()`` per adjoint step
  (including the it==0 tail).  ``swap_aux`` rotates the ``u`` triple buffer
  (same direction as forward) AND swaps every psi *and zeta* pair with its
  double-buffer partner, so the adjoint pair set includes the zeta slots:

  - acoustic2d, 11 slots: ``u_prev,u_now,u_next, psix,psiz, zetax,zetaz,
    psixn,psizn, zetaxn,zetazn`` → pairs ``(3,7),(4,8),(5,9),(6,10)``
  - acoustic3d, 15 slots: ``u_prev,u_now,u_next, psix,psiz, zetax,zetaz,
    psiy,zetay, psixn,psizn,psiyn, zetaxn,zetazn,zetayn`` → pairs
    ``(3,9),(4,10),(7,11),(5,12),(6,13),(8,14)``

* the 3-tensor reconstruction list (boundary-saving only) — one ``swap()``
  per MAIN-LOOP iteration (NOT the it==0 tail): pure ``u`` 3-cycle, no
  pairs.

The two counters advance differently across a segment ``(b, e)`` processed
in descending order: ``k_adj += b - e`` (every it runs an adjoint step) but
``k_f += b - max(e, 1)`` (the it==0 tail runs no reconstruction step).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

ACOUSTIC2D_PSI_PAIRS: Tuple[Tuple[int, int], ...] = ((3, 7), (4, 8))
ACOUSTIC3D_PSI_PAIRS: Tuple[Tuple[int, int], ...] = ((3, 9), (4, 10), (7, 11))

# Adjoint (fused backward) wavefield pair sets: psi AND zeta double-buffers
# are both swapped by ``swap_aux()``; rotating only the psi pairs (forward
# habit) leaves the zeta roles flipped after odd-length segments.
ACOUSTIC2D_ADJ_PAIRS: Tuple[Tuple[int, int], ...] = (
    (3, 7), (4, 8), (5, 9), (6, 10),
)
ACOUSTIC3D_ADJ_PAIRS: Tuple[Tuple[int, int], ...] = (
    (3, 9), (4, 10), (7, 11), (5, 12), (6, 13), (8, 14),
)


def acoustic_psi_pairs(ndim: int) -> Tuple[Tuple[int, int], ...]:
    return ACOUSTIC2D_PSI_PAIRS if ndim == 2 else ACOUSTIC3D_PSI_PAIRS


def acoustic_adj_pairs(ndim: int) -> Tuple[Tuple[int, int], ...]:
    return ACOUSTIC2D_ADJ_PAIRS if ndim == 2 else ACOUSTIC3D_ADJ_PAIRS


def rotate_wavefield_roles(
    wavefields: Sequence[torch.Tensor],
    steps_done: int,
    u_blocks: Sequence[int] = (0,),
    psi_pairs: Sequence[Tuple[int, int]] = (),
    block_size: int = 3,
) -> List[torch.Tensor]:
    """Wavefield list to bind for a continuation call after ``steps_done``
    completed time steps.

    Always rotate the ORIGINAL list (segment arithmetic is cumulative),
    not the previous call's rotated one.
    """
    out = list(wavefields)
    r = steps_done % block_size
    if r:
        for s in u_blocks:
            out[s:s + block_size] = [
                wavefields[s + (r + j) % block_size] for j in range(block_size)
            ]
    if steps_done % 2:
        for a, b in psi_pairs:
            out[a], out[b] = out[b], out[a]
    return out


def u_now_slot(steps_done: int, block_start: int = 0, block_size: int = 3) -> int:
    """Index (in the ORIGINAL wavefield list) of the tensor holding ``u_now``
    after ``steps_done`` completed steps — the field a domain-decomposition
    driver must halo-exchange before the next step."""
    return block_start + (steps_done + 1) % block_size


def u_next_slot(steps_done: int, block_start: int = 0, block_size: int = 3) -> int:
    """Index (in the ORIGINAL wavefield list) of the tensor the CURRENT step
    is writing (``u_next`` before that step's swap) after ``steps_done``
    completed steps.  Between ``run_phase(.., 1)`` and ``run_phase(.., 2)``
    this is the tensor whose cut-strip values an overlap DD driver
    exchanges; after phase 2's swap the same tensor IS ``u_now``, so the
    received halo lands exactly where the next step's stencil reads it."""
    return block_start + (steps_done + 2) % block_size


def rotate_adjoint_roles(
    wavefields: Sequence[torch.Tensor],
    k_adj: int,
    pairs: Sequence[Tuple[int, int]],
) -> List[torch.Tensor]:
    """Adjoint wavefield list to bind for a backward continuation after
    ``k_adj`` completed adjoint steps (``swap_aux()`` calls, it==0 tail
    included).  ``pairs`` is the full psi+zeta pair set
    (:data:`ACOUSTIC2D_ADJ_PAIRS` / :data:`ACOUSTIC3D_ADJ_PAIRS`)."""
    return rotate_wavefield_roles(wavefields, k_adj, psi_pairs=pairs)


def rotate_recon_roles(
    wavefields: Sequence[torch.Tensor],
    k_f: int,
) -> List[torch.Tensor]:
    """Reconstruction (forward) wavefield list to bind after ``k_f``
    completed main-loop reconstruction steps — pure u 3-cycle, no pairs."""
    return rotate_wavefield_roles(wavefields, k_f)


class SteppedBindingRunner:
    """Drive one propagator's captured ``ForwardInput`` over ``[0, nt)`` in
    segments, handling the buffer-role rotation between calls.

    ``func``/``params`` come straight from the compiled propagator
    (``_backend_impl.forward_func`` and the populated input); ``wavefields``
    is the ORIGINAL Python-side list whose tensors persist across calls.

    ``u_blocks`` selects which 3-tensor blocks of the list the per-step host
    rotation cycles (default ``(0,)`` — the acoustic u triple buffer).  Pass
    ``u_blocks=()`` together with ``psi_pairs=()`` for equations whose fields
    all update in place at fixed slots (elastic/DAS/VTI-1st families): the
    SAME wavefield list is then bound for every segment.
    """

    def __init__(self, func, params, wavefields, psi_pairs=(), u_blocks=(0,)):
        self.func = func
        self.p = params
        self.L = list(wavefields)
        self.psi_pairs = tuple(psi_pairs)
        self.u_blocks = tuple(u_blocks)
        self.k = 0

    def run_to(self, it_end: int) -> None:
        p = self.p
        p.it_begin, p.it_end = self.k, int(it_end)
        p.wavefields = rotate_wavefield_roles(
            self.L, self.k, u_blocks=self.u_blocks, psi_pairs=self.psi_pairs
        )
        self.func(p)
        self.k = int(it_end)

    def run_phase(self, it_end: int, phase: int) -> None:
        """Phase-split single step (SPECFEM-style comm/compute overlap).

        ``phase == 1`` runs only the cut-adjacent boundary strips of step
        ``self.k`` (no swap — ``self.k`` does not advance); the driver then
        exchanges :attr:`u_next` on a comm stream while ``phase == 2``
        computes the interior + the legacy tail (source/record/swap/ckpt)
        and advances ``self.k``.  Requires ``it_end == self.k + 1`` and
        ``params.cut_face_mask`` set by the driver.  ``step_phase`` is reset
        to 0 after every call so legacy ``run_to`` paths stay clean.
        """
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase}")
        if int(it_end) != self.k + 1:
            raise ValueError(
                f"run_phase drives exactly one step: it_end={it_end} but "
                f"k={self.k}"
            )
        p = self.p
        p.it_begin, p.it_end = self.k, int(it_end)
        p.step_phase = int(phase)
        p.wavefields = rotate_wavefield_roles(
            self.L, self.k, u_blocks=self.u_blocks, psi_pairs=self.psi_pairs
        )
        try:
            self.func(p)
        finally:
            p.step_phase = 0
        if phase == 2:
            self.k = int(it_end)

    @property
    def u_next(self) -> torch.Tensor:
        """Tensor the CURRENT step is writing (``L[(k+2) % 3]``, i.e.
        ``u_next`` before this step's swap) — what an overlap DD driver
        exchanges after ``run_phase(.., 1)``.  Only meaningful with the
        default rotating u block (``u_blocks=(0,)``)."""
        assert self.u_blocks, "u_next requires a rotating u block"
        return self.L[u_next_slot(self.k, self.u_blocks[0])]

    @property
    def u_now(self) -> torch.Tensor:
        """Tensor holding u_now after the steps run so far (halo-exchange
        target for a DD driver).  Only meaningful with the default rotating
        u block (``u_blocks=(0,)``)."""
        assert self.u_blocks, "u_now requires a rotating u block"
        return self.L[u_now_slot(self.k, self.u_blocks[0])]

    @property
    def u_prev(self) -> torch.Tensor:
        """Tensor holding u_prev after the steps run so far. Needed (with
        u_now) when a DD driver exchanges every K>1 steps with a K*M-wide
        halo: a stale u_prev re-seeds the halo error through the leapfrog
        update even after u_now is refreshed.  Only meaningful with the
        default rotating u block (``u_blocks=(0,)``)."""
        assert self.u_blocks, "u_prev requires a rotating u block"
        return self.L[self.u_blocks[0] + self.k % 3]


class SteppedBackwardRunner:
    """Drive one propagator's captured ``BackwardInput`` over ``[0, nt)`` in
    DESCENDING segments, handling the dual buffer-role rotation between
    calls.

    ``func``/``params`` come straight from the compiled propagator
    (``_backend_impl.backward_bs_func`` / ``backward_func`` and the populated
    input); ``adjoint_wavefields`` (11/15 tensors acoustic, 15/36 elastic)
    and, for boundary-saving, ``recon_wavefields`` (3 tensors acoustic,
    7/12 elastic incl. the trailing ``fv*_prev`` carries) are the ORIGINAL
    Python-side lists whose tensors persist across calls.  The caller binds
    ``params.grads_out`` (and ``params.illum_out`` where the equation
    produces illuminations), zeroed once, before the first segment.

    Rotation: acoustic rotates the adjoint u triple + psi/zeta pairs and the
    recon u triple (the defaults).  The elastic family has NO buffer-role
    rotation — every field updates in place at a fixed slot — so construct
    with ``adj_pairs=()``, ``adj_u_blocks=()``, ``recon_u_blocks=()`` and the
    SAME lists are bound for every segment.

    Counters (see module docstring): ``k_adj`` advances by ``b - e`` per
    segment ``(b, e)``; ``k_f`` by ``b - max(e, 1)`` (the it==0 adjoint-only
    tail runs no reconstruction step).
    """

    def __init__(self, func, params, adjoint_wavefields,
                 recon_wavefields=None, adj_pairs=(),
                 adj_u_blocks=(0,), recon_u_blocks=(0,)):
        self.func = func
        self.p = params
        self.L_adj = list(adjoint_wavefields)
        self.L_recon = list(recon_wavefields) if recon_wavefields else None
        self.adj_pairs = tuple(adj_pairs)
        self.adj_u_blocks = tuple(adj_u_blocks)
        self.recon_u_blocks = tuple(recon_u_blocks)
        self.k_adj = 0
        self.k_f = 0

    def _bind_lists(self):
        p = self.p
        p.adjoint_wavefields = rotate_wavefield_roles(
            self.L_adj, self.k_adj,
            u_blocks=self.adj_u_blocks, psi_pairs=self.adj_pairs,
        )
        if self.L_recon is not None:
            p.forward_wavefields = rotate_wavefield_roles(
                self.L_recon, self.k_f, u_blocks=self.recon_u_blocks
            )

    def run_segment(self, bw_it_begin: int, bw_it_end: int):
        """Run the segment [bw_it_end, bw_it_begin); segments must be issued
        in descending order and partition [0, nt) exactly."""
        b, e = int(bw_it_begin), int(bw_it_end)
        p = self.p
        p.bw_it_begin, p.bw_it_end = b, e
        self._bind_lists()
        out = self.func(p)
        self.k_adj += b - e
        self.k_f += b - max(e, 1)
        return out

    def run_phase(self, bw_it_begin: int, bw_it_end: int, phase: int):
        """Phase-split single backward step (elastic backward_bs only).

        ``phase == 1`` runs the adjoint-source injection, the stress
        reconstruction (+restore), the gradient accumulation and the
        STRESS-adjoint; the DD driver then exchanges the adjoint-velocity
        and recon-stress halos.  ``phase == 2`` runs the VELOCITY-adjoint,
        the fv_prev capture and the velocity reconstruction (+restore); the
        driver then exchanges the adjoint-stress and recon-velocity halos.
        Counters advance only after phase 2.  ``step_phase`` is reset to 0
        after every call so plain ``run_segment`` paths stay clean.
        """
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase}")
        b, e = int(bw_it_begin), int(bw_it_end)
        if b != e + 1:
            raise ValueError(
                f"run_phase drives exactly one step: got segment [{e}, {b})"
            )
        p = self.p
        p.bw_it_begin, p.bw_it_end = b, e
        p.step_phase = int(phase)
        self._bind_lists()
        try:
            out = self.func(p)
        finally:
            p.step_phase = 0
        if phase == 2:
            self.k_adj += 1
            self.k_f += 1 if e >= 1 else 0
        return out

    @property
    def lambda_now(self) -> torch.Tensor:
        """Tensor holding the adjoint u_now after the segments run so far
        (halo-exchange target for a DD driver)."""
        return self.L_adj[u_now_slot(self.k_adj)]

    @property
    def recon_u_now(self) -> torch.Tensor:
        """Tensor holding the reconstructed forward u_now after the segments
        run so far (halo-exchange target for a DD driver)."""
        assert self.L_recon is not None
        return self.L_recon[u_now_slot(self.k_f)]
