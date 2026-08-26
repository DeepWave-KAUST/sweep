"""Declarative domain-decomposition time-loop schedules.

The DD driver used to carry five hand-written time loops -- two forward, three
adjoint -- and branch on equation family in about twenty places. A schedule is
not a family property, though: it is a property of what the compiled kernel's
``step_phase`` values MEAN, and of which fields must have crossed the cut before
which sub-step. Declared here, interpreted once in ``dd_propagator``.

Companion to :mod:`sweep.equations.slot_table`: that says WHICH TENSOR sits at
which bind index; this says WHEN it is shipped. Neither is derivable from the
other.

What this schema deliberately does NOT express
----------------------------------------------
Worth stating, because a schema that looks complete invites trust it has not
earned:

* **What a ``step_phase`` means.** The spec says "call with 1". It cannot say
  that acoustic's 1 is a *spatial* strip split (``acoustic2d/forward.cu``) while
  elastic's 1 is a *physics* half-step (``elastic2d/forward.cu``: ``do_v`` /
  ``do_s``). That asymmetry is exactly why acoustic's phased forward hard-
  requires ``cut_face_mask != 0`` and elastic's does not -- and hence why an
  elastic DD run is legal at ``world_size == 1`` while an acoustic phased one
  would abort. The spec is a sequencer, not a semantics.
* **Exchange sufficiency.** Nothing here proves that shipping
  ``{adjoint velocity, recon stress}`` after elastic phase 1 is *enough*. That
  proof lives in the kernels' stencil reads. A spec can be internally
  consistent and ship the wrong fields; the honest artifact against that is a
  multi-rank parity test, not a dataclass field.
* **The compiled preconditions**, and especially the *absences* among them:
  elastic's phased backward has no ``cut_face_mask`` check, which is what makes
  a single-rank elastic DD run legal. A sequencer has no way to say "this check
  is deliberately missing".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Which of the driver's tensor lists a reference resolves against.
Buf = Literal["fwd", "adj", "recon", "coupling", "coeffs"]
TimeRole = Literal["u_prev", "u_now", "u_next"]


@dataclass(frozen=True)
class FieldRef:
    """A set of tensors to halo-exchange, named without indices.

    Three mutually exclusive resolution modes:

    * ``at`` -- ONE tensor identified by its time-level role *after* the
      counters have advanced. Only the rotating families (acoustic / VRZ) have
      one; it is exactly what ``SteppedBindingRunner.u_now`` and
      ``SteppedBackwardRunner.lambda_now`` / ``recon_u_now`` return.
    * ``roles`` -- the slots of ``buf`` whose :class:`~sweep.equations.slot_table.Slot`
      role is in this set. Deliberately NOT ``range(nv)`` / ``range(nv, nphys)``:
      those encode an assumption that velocity slots are a contiguous prefix,
      true for elastic and false for DAS-Zhao, whose physical block is split
      around the PML block. For ``buf == "recon"`` the lookup is by NAME against
      the table's ``recon`` tuple, so the trailing ``fv*_prev`` carries drop out
      of every group on their own -- they have no slot, hence no role.
    * neither -- the whole buffer list (``coupling`` / ``coeffs``: fixed-size
      workspaces sized by ``cuda_layout.dd_coupling_nvar`` and
      ``dd_adjoint_coeff_nvar``).
    """

    buf: Buf
    at: TimeRole | None = None
    roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.at is not None and self.roles is not None:
            raise ValueError("FieldRef is time-role OR slot-role resolved, not both")


@dataclass(frozen=True)
class ExchangeGroup:
    """One halo shipment issued after a phase.

    ``batched`` collapses every tensor of every ref into ONE batched P2P per cut
    axis instead of one round per tensor. For a single-tensor group the two are
    op-for-op identical, so the flag is a declaration about the WIRE, not about
    the bits -- which is why changing it is a separate, separately-gated kind of
    change even though the values cannot move.

    ``when`` names a runtime predicate the driver supplies; ``why`` is the
    justification, kept as data so it lives next to the thing it justifies
    rather than as a comment block somewhere in the driver.

    ``overlappable`` says the driver MAY hoist this shipment onto a comm stream
    and let the following phase compute over it. Default False: an exchange is
    serial unless someone has proven otherwise for that specific equation. The
    flag grants *permission*, and the driver still owes the proof -- for
    acoustic that proof is "no source sits in a shipped strip", because a later
    phase of the same step atomically adds the source into the very buffer the
    comm stream is packing.
    """

    refs: tuple[FieldRef, ...]
    batched: bool = True
    when: str | None = None
    why: str = ""
    overlappable: bool = False


@dataclass(frozen=True)
class Phase:
    """One compiled-kernel call inside a time step.

    ``step_phase`` is bound to ``params.step_phase``; ``None`` means the
    unphased legacy call.

    ``advances`` marks the phase whose C++ body performs the host-side
    buffer-role swap, so the Python counters bump after it. This is the one fact
    the stepped runners used to encode by METHOD NAME rather than by argument:
    the elastic runner advances after phase 2, the VRZ one after phase **1**,
    because VRZ's phases 2 and 3 deliberately re-read the already-advanced
    lists -- which is precisely what makes them see the tensors the driver just
    exchanged.
    """

    step_phase: int | None
    advances: bool
    after: tuple[ExchangeGroup, ...] = ()
    label: str = ""


@dataclass(frozen=True)
class DDLoop:
    direction: Literal["fwd", "rev"]
    phases: tuple[Phase, ...]
    #: lowest ``it`` executed (reverse loops only). 1 = "step 0 contributes no
    #: gradient" (elastic / VRZ); 0 = acoustic, whose ``it == 0`` adjoint-only
    #: tail does contribute the wavelet gradient.
    floor: int = 0
    #: may boundary-tail truncation raise the floor? Whether an equation is
    #: ALLOWED to truncate stays where it already is (the propagator refuses
    #: ``tail_steps`` for anything but Acoustic/Acoustic3D); this flag only says
    #: the loop knows how to move its floor.
    tail_truncatable: bool = False
    #: skip the exchanges attached to the LAST phase on the floor iteration.
    #: Sound in general -- that halo is read by the next step's first phase and
    #: there is no next step -- but enabled only where today's code already does
    #: it, because turning it on elsewhere changes the NCCL round count.
    drop_trailing_exchange_on_floor: bool = False
    #: phases run once before the loop, at the final step's segment.
    prologue: tuple[Phase, ...] = ()


@dataclass(frozen=True)
class DDSpec:
    forward: DDLoop
    backward: DDLoop
    #: Optional second forward schedule the driver may use INSTEAD of
    #: ``forward`` when it can discharge the overlap's proof obligation.
    #:
    #: Modelled as a separate loop rather than as a flag on ``forward``,
    #: because the two are not a reordering of one schedule: the serial
    #: variant issues ONE unphased kernel call over the whole grid, the
    #: overlapped one issues TWO (cut strips, then interior). Different call
    #: sequences deserve different declarations -- and stating both means the
    #: ``u_now`` / ``u_next`` question answers itself, since each loop names
    #: the tensor as of its own resolution point instead of relying on the
    #: interpreter to shift the role by one advance.
    forward_overlapped: DDLoop | None = None
    name: str = ""


# --------------------------------------------------------------------------- #
# field references
# --------------------------------------------------------------------------- #
U_NOW = FieldRef("fwd", at="u_now")
U_NEXT = FieldRef("fwd", at="u_next")
LAMBDA = FieldRef("adj", at="u_now")
RECON_U = FieldRef("recon", at="u_now")

VEL_F = FieldRef("fwd", roles=("vel",))
STR_F = FieldRef("fwd", roles=("stress",))
VEL_A = FieldRef("adj", roles=("vel",))
STR_A = FieldRef("adj", roles=("stress",))
VEL_R = FieldRef("recon", roles=("vel",))
STR_R = FieldRef("recon", roles=("stress",))

COUPLING = FieldRef("coupling")
COEFFS = FieldRef("coeffs")


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #
#: Elastic (2-D and 3-D). Half-step protocol: the velocity kernel reads stress
#: through its stencil and the stress kernel reads velocity, so each half-step
#: ships what the OTHER half needs. Both phases launch over the full grid -- the
#: split is physics, not space -- which is why there is nothing here to overlap
#: and no cut-face precondition on the compiled side.
ELASTIC_DD = DDSpec(
    name="elastic",
    forward=DDLoop(
        direction="fwd",
        phases=(
            Phase(1, advances=False, label="velocity",
                  after=(ExchangeGroup(
                      (VEL_F,),
                      why="the stress kernel reads velocity through its stencil"),)),
            Phase(2, advances=True, label="stress",
                  after=(ExchangeGroup(
                      (STR_F,),
                      why="the next step's velocity kernel reads stress "
                          "through its stencil"),)),
        ),
    ),
    backward=DDLoop(
        direction="rev",
        floor=1,   # elastic boundary-saving reconstruction bottoms out at it==1
        phases=(
            Phase(3, advances=False, label="injections",
                  after=(ExchangeGroup(
                      (STR_A, VEL_R), when="inj_cross",
                      why="in phased mode the compiled backward injects step it's "
                          "terms at the tail of step it+1's phase 2, so the "
                          "post-phase-2 exchange ships them -- but the FIRST "
                          "reverse step has no preceding phase 2. Only needed "
                          "when an injection actually lands in a phase-2 field: "
                          "a body-force source writes recon velocity and a "
                          "stress receiver writes adjoint stress. The default "
                          "combination (stress source, velocity receivers) "
                          "touches neither, and then the strips are untouched "
                          "since the previous phase-2 exchange."),)),
            Phase(1, advances=False, label="stress-adjoint",
                  after=(ExchangeGroup((VEL_A, STR_R),),)),
            Phase(2, advances=True, label="velocity-adjoint",
                  after=(ExchangeGroup((STR_A, VEL_R),),)),
        ),
    ),
)


#: Acoustic (Acoustic, Acoustic3D). One unphased call per step; the next step's
#: stencil reads ``u_now`` across the cut.
ACOUSTIC_FWD = DDLoop(
    direction="fwd",
    phases=(Phase(None, advances=True, label="step",
                  after=(ExchangeGroup(
                      (U_NOW,), batched=False, overlappable=True,
                      why="the next step's stencil reads u_now across the cut. "
                          "Overlappable only because the driver can prove no "
                          "later writer lands in a shipped strip: the source "
                          "injection atomically adds into the very buffer the "
                          "comm stream is packing, so a source within M of a "
                          "cut line forbids it."),)),),
)

#: The comm/compute-overlap variant of the acoustic forward. Phase 1 computes
#: ONLY the cut-adjacent strips -- exactly what the halo ships -- so the
#: exchange can run on a comm stream while phase 2 computes the strict
#: complement. ``U_NEXT`` rather than ``U_NOW`` because this loop resolves the
#: tensor BEFORE the advance: ``u_next`` at k and ``u_now`` at k+1 are the same
#: slot by construction, and naming it per-loop beats shifting roles in the
#: interpreter.
#:
#: The driver still owes the proof that no later writer lands in a shipped
#: strip. Two writers exist: the source injection, which atomically adds into
#: the very buffer the comm stream is packing (hence the "no source within M of
#: a cut line" test), and -- with irregular topography -- phase 2's air-clear
#: pre-pass, whose range is widened by M. The latter is benign because phase 1
#: already zeroed those same cells and it re-writes the same zeros, and because
#: the widened range stops at the physical bounds and never reaches the halo pad
#: the receive-copy writes.
ACOUSTIC_FWD_OVERLAP = DDLoop(
    direction="fwd",
    phases=(
        Phase(1, advances=False, label="cut strips",
              after=(ExchangeGroup(
                  (U_NEXT,), batched=False, overlappable=True,
                  why="shipped async on a comm stream while phase 2 computes "
                      "the interior; joined before the next step's phase 1"),)),
        Phase(2, advances=True, label="interior + tail"),
    ),
)

ACOUSTIC_DD = DDSpec(
    name="acoustic",
    forward=ACOUSTIC_FWD,
    forward_overlapped=ACOUSTIC_FWD_OVERLAP,
    backward=DDLoop(
        direction="rev",
        floor=0,   # the it==0 adjoint-only tail still contributes grad_wavelet
        tail_truncatable=True,
        drop_trailing_exchange_on_floor=True,
        phases=(Phase(None, advances=True, label="reverse step",
                      after=(ExchangeGroup(
                          (LAMBDA, RECON_U), batched=False,
                          why="the next reverse step reads both across the cut"),)),),
    ),
)


#: Variable-density VRZ. Its forward is the acoustic one; its BACKWARD is the
#: only three-phase adjoint in the tree, because the gradient is a spatial
#: divergence of a coupling field rather than a pointwise product -- so the
#: coupling has to be built from the POST-exchange lambda/p, shipped, and only
#: then differentiated.
#:
#: Note ``advances`` sits on phase 1, not on the last phase. Phases 2 and 3
#: deliberately re-read the ALREADY-advanced lists, which is exactly what makes
#: them see the tensors the driver exchanged in between. Putting the advance at
#: the end would bind pre-advance lists and silently differentiate the wrong
#: step.
VRZ_DD = DDSpec(
    name="acoustic_vrz",
    forward=ACOUSTIC_FWD,
    forward_overlapped=ACOUSTIC_FWD_OVERLAP,
    backward=DDLoop(
        direction="rev",
        floor=1,
        prologue=(Phase(4, advances=False, label="adjoint coeff build",
                        after=(ExchangeGroup(
                            (COEFFS,),
                            why="the fused adjoint's transpose fast path reads "
                                "C0/Cx/Cy/Cz over [ix-M, ix+M], i.e. into the cut "
                                "halo. Model-only and constant within a backward, "
                                "so built and shipped once instead of per step."),)),),
        phases=(
            Phase(1, advances=True, label="advance adjoint + reconstruct",
                  after=(ExchangeGroup(
                      (LAMBDA, RECON_U), batched=False,
                      why="phase 2 builds the coupling from the POST-exchange "
                          "lambda and p"),)),
            Phase(2, advances=False, label="build coupling",
                  after=(ExchangeGroup(
                      (COUPLING,),
                      why="the gradient is div(c/e), so the divergence at a cut "
                          "seam needs the neighbour's coupling values"),)),
            Phase(3, advances=False, label="divergence -> gradient"),
        ),
    ),
)
