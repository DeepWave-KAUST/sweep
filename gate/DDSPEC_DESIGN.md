## 0. What actually varies across the five schedules

Strip the five verified schedules down and only six things differ:

| | acoustic fwd | elastic fwd | acoustic bwd | elastic bwd | VRZ bwd |
|---|---|---|---|---|---|
| phases/step | 1 (`None`) | 2 | 1 (`None`) | 3 | 3 (+1 prologue) |
| `step_phase` seq | – | 1,2 | – | 3,1,2 | 4 \| 1,2,3 |
| which phase advances counters | the only one | 2 (last) | the only one | 2 (last) | **1 (first)** |
| loop floor | 0 | 0 | 0, tail-movable | 1 | 1 |
| exchange after phase | u_now | vel / stress | λ, recon-u | ph2? / ph1 / ph2 | λ,p / c,e / – |
| batched | no | yes | no | yes | no / yes |

Everything else the driver branches on — `grads_out` prefix, `illum_out` length, `nv`/`nphys`, `_is_vrz`, `_FAMILIES` (`/home/wangs0j/sweep-local/dd-refactor/src/sweep/parallel/dd_propagator.py:71-76`) — is not schedule shape at all. §4 sends those elsewhere.

So the schema needs exactly: an ordered list of phases; per phase a `step_phase`, an "advances the counters" bit, and a list of exchange groups; per group a set of field references and a batching bit; per loop a direction, a floor, and two policy flags.

---

## 1. The dataclasses

New file `src/sweep/parallel/dd_spec.py`. Nothing here imports torch or the driver — it is data.

```python
"""Declarative DD time-loop schedules.

The DD driver used to encode five hand-written time loops (two forward, three
adjoint) and branch on equation family in ~20 places.  A schedule is not a
family property, though: it is a property of what the compiled kernel's
``step_phase`` values MEAN and of which fields must have crossed the cut before
which sub-step.  Declared here, interpreted once in dd_propagator.

Companion to ``equations/slot_table.py``: that says WHICH TENSOR is at which
bind index; this says WHEN it is shipped.  Neither is derivable from the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Buf = Literal["fwd", "adj", "recon", "coupling", "coeffs"]
TimeRole = Literal["u_prev", "u_now", "u_next"]


@dataclass(frozen=True)
class FieldRef:
    """A set of tensors to halo-exchange, named without indices.

    Three resolution modes, mutually exclusive:

    * ``at`` -- ONE tensor identified by its time-level role after the
      counters have advanced.  Only rotating families (acoustic/VRZ) have
      one; it is exactly ``SteppedBindingRunner.u_now``/``u_next`` and
      ``SteppedBackwardRunner.lambda_now``/``recon_u_now``
      (propagator/_stepped.py:216-247, :403-414).
    * ``roles`` -- the slots of ``buf`` whose ``SlotTable`` role is in this
      set.  NOT ``range(nv)``/``range(nv, nphys)``: those assume velocities
      are a contiguous prefix, which is true for elastic and false for
      DAS-Zhao (the caveat already written at dd_propagator.py:404-407).
      For ``buf == "recon"`` the lookup is by NAME against the table's
      ``recon`` tuple, so the trailing ``fv*_prev`` carries fall out of every
      group automatically -- they have no slot, hence no role.
    * neither -- the whole buffer list (``coupling``, ``coeffs``: fixed-size
      workspaces sized by ``cuda_layout.dd_coupling_nvar`` /
      ``dd_adjoint_coeff_nvar``).
    """

    buf: Buf
    at: TimeRole | None = None
    roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.at is not None and self.roles is not None:
            raise ValueError("FieldRef is time-role OR slot-role resolved")


@dataclass(frozen=True)
class ExchangeGroup:
    """One halo shipment issued after a phase.

    ``batched=True`` collapses every tensor of every ref into ONE
    ``dist.batch_isend_irecv`` per cut axis (fast_halo.py:105-136);
    ``batched=False`` issues one round per tensor, in ref order.  For a
    single-tensor group the two are op-for-op identical (compare
    FastHaloExchanger.__call__ fast_halo.py:68-76 with FastHaloGroup.__call__
    :127-136) -- the flag is a declaration about the WIRE, not about the bits.

    ``when`` names a runtime predicate (see PREDICATES) that must hold for the
    shipment to be issued; ``why`` is the reason, kept as data so it lives next
    to the thing it justifies instead of as a 12-line comment in the driver
    (dd_propagator.py:1173-1186).

    ``overlappable`` says the driver MAY hoist this shipment onto a comm stream
    and let the following phase compute over it -- see the argument in the
    design note.  Default False: an exchange is serial unless someone has
    proven otherwise for that specific equation.
    """

    refs: tuple[FieldRef, ...]
    batched: bool = True
    when: str | None = None
    why: str = ""
    overlappable: bool = False


@dataclass(frozen=True)
class Phase:
    """One compiled-kernel call inside a time step.

    ``step_phase`` is the value bound to ``params.step_phase``; ``None`` means
    the unphased legacy call (``run_to`` / ``run_segment`` today).

    ``advances`` marks the phase whose C++ body performs the host-side
    buffer-role swap, so the Python counters must be bumped after it.  This is
    the one fact the three ``_stepped.py`` runner methods currently encode by
    NAME rather than by argument: ``run_phase`` advances after phase 2
    (_stepped.py:218-219, :358-360) while ``run_vrz_phase`` advances after
    phase 1 (_stepped.py:398-400), because VRZ's phases 2 and 3 deliberately
    re-read the ALREADY-advanced lists -- which is exactly what makes them see
    the tensors the driver just exchanged.
    """

    step_phase: int | None
    advances: bool
    after: tuple[ExchangeGroup, ...] = ()
    label: str = ""


@dataclass(frozen=True)
class DDLoop:
    direction: Literal["fwd", "rev"]
    phases: tuple[Phase, ...]
    #: lowest ``it`` executed (rev only).  1 = "step 0 contributes no gradient"
    #: (elastic/VRZ); 0 = acoustic, whose it==0 adjoint-only tail does
    #: contribute grad_wavelet (acoustic2d/backward.cu:689-719).
    floor: int = 0
    #: may boundary-tail truncation raise the floor?  Legality per equation
    #: stays where it already is -- propagator/_c.py:1513-1516 refuses
    #: tail_steps for anything but Acoustic/Acoustic3D.  This flag only says
    #: the LOOP knows how to move.
    tail_truncatable: bool = False
    #: skip the exchanges attached to the LAST phase on the floor iteration.
    #: Sound in general (that halo is read by the next step's first phase, and
    #: there is no next step); enabled only where today's code already does it,
    #: because turning it on elsewhere changes the NCCL round count.
    drop_trailing_exchange_on_floor: bool = False
    #: phases run once before the loop, at the final step's segment.
    prologue: tuple[Phase, ...] = ()


@dataclass(frozen=True)
class DDSpec:
    forward: DDLoop
    backward: DDLoop
    name: str = ""
```

### The three literal specs

```python
# --- field references -------------------------------------------------------
U_NOW   = FieldRef("fwd",   at="u_now")
U_NEXT  = FieldRef("fwd",   at="u_next")
LAMBDA  = FieldRef("adj",   at="u_now")
RECON_U = FieldRef("recon", at="u_now")

VEL_F, STR_F = FieldRef("fwd", roles=("vel",)),   FieldRef("fwd", roles=("stress",))
VEL_A, STR_A = FieldRef("adj", roles=("vel",)),   FieldRef("adj", roles=("stress",))
VEL_R, STR_R = FieldRef("recon", roles=("vel",)), FieldRef("recon", roles=("stress",))

COUPLING = FieldRef("coupling")
COEFFS   = FieldRef("coeffs")


# --- acoustic (Acoustic, Acoustic3D) ---------------------------------------
ACOUSTIC_FWD = DDLoop(
    direction="fwd",
    phases=(Phase(None, advances=True, label="step",
                  after=(ExchangeGroup(
                      (U_NOW,), batched=False, overlappable=True,
                      why="next step's stencil reads u_now across the cut; "
                          "safe to overlap only if no later writer touches the "
                          "shipped strips -- see acoustic2d/forward.cu:276-283"),
                  )),),
)

ACOUSTIC_DD = DDSpec(
    name="acoustic",
    forward=ACOUSTIC_FWD,
    backward=DDLoop(
        direction="rev", floor=0,
        tail_truncatable=True,
        drop_trailing_exchange_on_floor=True,
        phases=(Phase(None, advances=True, label="reverse step",
                      after=(ExchangeGroup((LAMBDA, RECON_U), batched=False),)),),
    ),
)


# --- elastic (Elastic 2-D and 3-D) -----------------------------------------
_PH1 = (VEL_A, STR_R)     # phase 1 PRODUCES adjoint velocity + recon stress
_PH2 = (STR_A, VEL_R)     # phase 2 PRODUCES adjoint stress   + recon velocity

ELASTIC_DD = DDSpec(
    name="elastic",
    forward=DDLoop(
        direction="fwd",
        phases=(
            Phase(1, advances=False, label="velocity", after=(ExchangeGroup((VEL_F,)),)),
            Phase(2, advances=True,  label="stress",   after=(ExchangeGroup((STR_F,)),)),
        ),
    ),
    backward=DDLoop(
        direction="rev", floor=1,
        phases=(
            Phase(3, advances=False, label="injections", after=(ExchangeGroup(
                _PH2, when="inj_cross",
                why="phase 3 writes ph2 fields when a body-force source injects "
                    "recon VELOCITY or a stress receiver injects adjoint STRESS; "
                    "phase 1 reads both across the cut.  With the default "
                    "stress-source/velocity-receiver pair the strips are "
                    "unchanged since the previous ph2 shipment."),)),
            Phase(1, advances=False, label="stress-adjoint",   after=(ExchangeGroup(_PH1),)),
            Phase(2, advances=True,  label="velocity-adjoint", after=(ExchangeGroup(_PH2),)),
        ),
    ),
)


# --- VRZ (AcousticVRZ3D) ---------------------------------------------------
VRZ_DD = DDSpec(
    name="acoustic_vrz",
    forward=ACOUSTIC_FWD,          # literally the same object: VRZ's forward
                                   # IS the acoustic forward, which today is
                                   # only visible as "it falls into the
                                   # family == 'acoustic' branch".
    backward=DDLoop(
        direction="rev", floor=1,
        prologue=(Phase(4, advances=False, label="adjoint coeffs",
                        after=(ExchangeGroup((COEFFS,)),)),),
        phases=(
            Phase(1, advances=True,  label="advance + recon",
                  after=(ExchangeGroup((LAMBDA, RECON_U), batched=False),)),
            Phase(2, advances=False, label="build c/e",
                  after=(ExchangeGroup((COUPLING,)),)),
            Phase(3, advances=False, label="divergence -> grad"),
        ),
    ),
)
```

### The predicate registry

The only runtime-conditional exchange in all five schedules. Note it now derives the velocity field names from the slot table instead of the hardcoded `_vel = ("vx", "vy", "vz")` at `dd_propagator.py:1184` — a fourth hand-copy of the bind order that the slot-table work missed.

```python
def _inj_cross(ctx) -> bool:
    vel = ctx.field_names("vel")
    return (any(t in vel for t in ctx.source_type)
            or any(t not in vel for t in ctx.receiver_type))

PREDICATES = {"inj_cross": _inj_cross}
```

### Where the spec is attached

`CUDALayoutSpec.dd_schedule: DDSpec | None = None`, next to `dd_coupling_nvar` / `dd_adjoint_coeff_nvar` (`src/sweep/equations/cuda_layout.py:56-64`). "Which equations does DD support" then becomes "which equations declare a schedule", which is exactly the assertion `_DD_EQUATIONS` (`dd_propagator.py:98-105`) makes in a comment: *the CUDA forward AND backward honour the stepped range*. Declaring a schedule is the equation author making that claim; `_family_of`'s `NotImplementedError` (`:116-125`) survives verbatim, raised on `dd_schedule is None`.

---

## 2. The interpreter

Two pieces: a runner adapter that erases the fwd/bwd and phased/unphased asymmetry, and one loop.

### 2a. Collapsing `_stepped.py`'s three methods into one

`run_segment` (`_stepped.py:313-323`), `run_phase` (`:325-361`) and `run_vrz_phase` (`:362-401`) differ in exactly two ways: the phase value, and *which* phase advances. Their advance arithmetic is already the same rule — for a single step `b == e+1`, `k_adj += b-e` gives `+1` and `k_f += b - max(e,1)` gives `+1 if e>=1 else 0`, identical to `run_phase`'s `self.k_f += 1 if e >= 1 else 0` (`:360`). So:

```python
# SteppedBackwardRunner
def run(self, b, e, phase=None, advance=True):
    """Unified entry.  ``phase=None`` is the legacy unphased segment."""
    b, e = int(b), int(e)
    if phase is not None and b != e + 1:
        raise ValueError(f"phased backward drives exactly one step: [{e}, {b})")
    p = self.p
    p.bw_it_begin, p.bw_it_end = b, e
    if phase is not None:
        p.step_phase = int(phase)
    self._bind_lists()
    try:
        out = self.func(p)
    finally:
        p.step_phase = 0
    if advance:
        self.k_adj += b - e
        self.k_f += b - max(e, 1)
    return out

def at(self, buf, role):
    L, k = ((self.L_adj, self.k_adj) if buf == "adj" else (self.L_recon, self.k_f))
    return L[{"u_prev": lambda k: k % 3,
              "u_now": u_now_slot, "u_next": u_next_slot}[role](k)]
```

with the analogous 4-line `run(it_end, phase, advance)` / `at(...)` on `SteppedBindingRunner`. `run_segment` / `run_phase` / `run_vrz_phase` become one-line shims for the non-DD callers, or are deleted with them.

### 2b. The loop

```python
_BUFS = {"fwd": "L_fwd", "adj": "L_adj", "recon": "recon",
         "coupling": "coupling", "coeffs": "adj_coeffs"}


def _role_index(table, buf, roles):
    """Slot indices of ``buf`` whose SlotTable role is in ``roles``."""
    if buf == "recon":
        role_of = {s.name: s.role for s in table.slots}
        return tuple(i for i, n in enumerate(table.recon) if role_of.get(n) in roles)
    pool = table.slots if buf == "adj" else table._fwd()
    return tuple(i for i, s in enumerate(pool) if s.role in roles)


class ModelParallel:
    # ------------------------------------------------------------ resolution
    def _tensors(self, ref, runner):
        if ref.at is not None:
            return (runner.at(ref.buf, ref.at),)
        L = getattr(self, _BUFS[ref.buf])
        if ref.roles is None:
            return tuple(L)
        key = (ref.buf, ref.roles)
        idx = self._role_idx.get(key)
        if idx is None:
            idx = self._role_idx[key] = _role_index(self._table, *key)
        return tuple(L[i] for i in idx)

    def _ship(self, halo, runner, grp):
        ts = [t for ref in grp.refs for t in self._tensors(ref, runner)]
        if not ts:
            return                       # workspace this equation does not have
        if grp.batched:
            self._exchange_group(halo, ts)
        else:
            for t in ts:
                self._exchange(halo, t)

    # ------------------------------------------------------------- the loop
    def _loop_floor(self, L):
        if not L.tail_truncatable:
            return L.floor
        tail = int(getattr(self.bp, "boundary_tail_steps", 0) or 0)
        bs_it0 = max(0, self.nt - tail) if tail > 0 else 0
        return max(L.floor, bs_it0 + 1 if bs_it0 > 0 else 0)

    def _run_dd_loop(self, L, runner, halo, preds):
        """Interpret ONE DDLoop.  The only time loop left in this file."""
        fwd = L.direction == "fwd"
        seg = (lambda it: (it + 1,)) if fwd else (lambda it: (it + 1, it))

        for ph in L.prologue:                     # VRZ coeff build, at (nt, nt-1)
            runner.run(*seg(self.nt - 1), phase=ph.step_phase, advance=ph.advances)
            for grp in ph.after:
                self._ship(halo, runner, grp)

        floor = self._loop_floor(L)
        its = range(self.nt) if fwd else range(self.nt - 1, floor - 1, -1)
        last_phase = L.phases[-1]
        for it in its:
            on_floor = (not fwd) and it == floor
            for ph in L.phases:
                runner.run(*seg(it), phase=ph.step_phase, advance=ph.advances)
                if on_floor and ph is last_phase and L.drop_trailing_exchange_on_floor:
                    continue
                for grp in ph.after:
                    if grp.when is not None and not preds[grp.when]:
                        continue
                    self._ship(halo, runner, grp)
```

`forward()` (`dd_propagator.py:897-903`) collapses to:

```python
sg = self._prepare_call(wavelet, sources_global, receivers_global, models)
fhalo = self._halo("_fwd_halo")
spec = self._spec.forward
with torch.no_grad():
    runner = SteppedBindingRunner(self.f_func, self.fp, self.L_fwd,
                                  psi_pairs=self._table.pairs(adjoint=False),
                                  u_blocks=self._table.u_blocks)
    if self._overlap_eligible(spec, sg):
        self._run_dd_loop_overlapped(spec, runner, fhalo)
    else:
        self._run_dd_loop(spec, runner, fhalo, preds={})
```

and `_run_adjoint`'s three-way branch (`:1106-1194`, 89 lines) collapses to:

```python
bhalo = self._halo("_bwd_halo")
spec = self._spec.backward
preds = {n: f(self._pred_ctx()) for n, f in PREDICATES.items()}
with torch.no_grad():
    br = SteppedBackwardRunner(self.b_func, self.bp, self.L_adj, self.recon,
                               adj_pairs=self._table.pairs(adjoint=True),
                               adj_u_blocks=self._table.u_blocks,
                               recon_u_blocks=self._table.u_blocks)
    self._run_dd_loop(spec, br, bhalo, preds)
```

Note the second win, free: `adj_pairs` now comes from `SlotTable.pairs(adjoint=True)` (`slot_table.py:155-168`), which deletes the `adjoint_extra_nvar` discriminator dance at `dd_propagator.py:1138-1141` **and** its dead `else acoustic_psi_pairs(...)` fallback, **and** the 3-D-centric stale comment block at `:1130-1137` that the review flagged as wrong in three places. VRZ's table is `ACOUSTIC3D.slots[:12]` (`slot_table.py:223-224`), so `pairs(adjoint=True) == pairs(adjoint=False)` automatically — the exact fact the comment tries to explain in prose. Elastic's table yields `pairs() == ()` and `u_blocks == ()` (`slot_table.py:145-153`), matching the explicit `()`s passed today at `:1165-1166` and `:1070-1071`.

---

## 3. The overlap: stays driver-side, spec contributes one bit

**Claim: the overlap is an execution policy, not a schedule. It stays in the driver. The spec contributes exactly one thing — the *invariant* that makes it legal.**

Three arguments, in increasing order of force.

**(i) It preserves the schedule's only semantic content.** A DDSpec asserts a happens-before relation: *the fields of group G must have crossed the cut before the compute that reads them across the cut*. Variant A satisfies it (`exchange_start` at `dd_propagator.py:1056`, `exchange_finish` at `:1059`, `compute.wait_stream(comm)` at `:1060` — the join before the next step's phase 1). Variant B satisfies it. They are two *implementations of one spec*, which is precisely what the docstring already says ("the overlap is a pure reordering", `:1036-1040`). If the spec encoded both, it would no longer be a specification; it would be a program with a branch, and we would have moved the `if` rather than removed it.

**(ii) Its predicate is per-call and per-process, not per-equation.** `_overlap_ok` (`:371-373`) folds in `world`, `cut_mask & ~0x3`, and `SWEEP_DD_DISABLE_OVERLAP` read from `os.environ` (`:370`). `_src_away_from_cuts` (`:473-484`) is re-evaluated on **every** call because the source moves between shots. A module-level `DDSpec` constant cannot hold any of that without becoming `spec(world, cut_mask, env, sg)` — i.e. code with extra ceremony.

**(iii) The decisive one — the invariant IS expressible, and separating it is what makes the driver's runtime check meaningful.** The reason overlap is conditional is not "streams are hard". It is that acoustic's phase 2 *writes into the tensor phase 1's exchange is packing*: `add_source` sits below the `if (phase == 1) continue;` at `acoustic2d/forward.cu:259-260` and does `atomicAdd(&u[u_idx], ...)` into `view.u_next` (`common/common.cu:31`) — the same buffer whose strip `sbuf.copy_(sview)` is reading on the comm stream (`fast_halo.py:88-89`). So the group carries `overlappable=True` meaning *"a later phase of this step may write these strips; overlap is legal only where the driver can prove no writer lands in one"*, and `_src_away_from_cuts` is the driver's proof obligation. Spec = invariant; driver = discharge.

That split also explains, correctly, why *no other group* gets the bit. Elastic's phase split is a **physics** split, not a **space** split (`elastic2d/forward.cu:172-173`, `do_v`/`do_s`): both phases launch over the full grid, so phase 2 writes the strips the vel exchange is packing *and* there is no interior-only work to hide the transfer behind. There is nothing to overlap, for a structural reason the default `overlappable=False` records honestly rather than by omission.

Concretely, the overlap becomes a second interpreter that consumes the same spec — ~15 lines, and the *only* place in the driver that knows about streams:

```python
def _overlap_eligible(self, L, sg):
    g = L.phases[0].after[0] if L.phases[0].after else None
    return (g is not None and g.overlappable and len(L.phases) == 2
            and self._overlap_ok and self._src_away_from_cuts(sg))

def _run_dd_loop_overlapped(self, L, runner, halo):
    p1, p2 = L.phases
    grp = p1.after[0]
    if self._comm_stream is None:
        self._comm_stream, self._comm_evt = torch.cuda.Stream(), torch.cuda.Event()
    comm, evt = self._comm_stream, self._comm_evt
    compute = torch.cuda.current_stream()
    hs = halo["x"]                                  # _overlap_ok => axes == ("x",)
    for it in range(self.nt):
        runner.run(it + 1, phase=p1.step_phase, advance=p1.advances)
        evt.record()
        (t,) = self._tensors(grp.refs[0], runner)   # spec says WHICH tensor
        view = self._halo_view(t, "x")
        with torch.cuda.stream(comm):
            comm.wait_event(evt)
            hs.exchange_start(view)
        runner.run(it + 1, phase=p2.step_phase, advance=p2.advances)
        with torch.cuda.stream(comm):
            hs.exchange_finish(view)
        compute.wait_stream(comm)
```

One wrinkle worth naming: today's serial variant B ships `runner.u_now` after the advance while variant A ships `runner.u_next` before it — the same physical tensor `L_fwd[(it+2)%3]`, as the verification confirms. Under one spec the group says `at="u_now"`, and the overlap interpreter resolves it *before* `advances` has fired, so it must ask for `u_next`. Rather than special-case it, note that `at="u_now" resolved at k` and `at="u_next" resolved at k-1` are the same slot by construction (`_stepped.py:108-122`); the overlap interpreter resolves with `runner.at(buf, "u_next")` when it resolves pre-advance. This is real and easy to get wrong — it is on the migration list as its own gated step (§5, step 5) for that reason.

---

## 4. `grads_out`, illumination, tail bound

**None of the three belongs in `DDSpec`.** Two are output-binding facts about the compiled backward, which single-domain `_c.py` needs too; putting them in a DD-only schedule would create a second source of truth. One is a loop-shape fact and does belong.

**`grads_out` layout → `CUDALayoutSpec.grads_out_has_wavelet: bool`.**
The C-side contract is `grads_out.size() == models.size() + 1` with slot 0 = grad_wavelet (`acoustic2d/backward.cu:101-103`, re-checked at `:138-140`) versus `== models.size()` (`elastic2d/backward.cu:306-309`). That is a property of the equation's compiled adjoint, discovered nowhere else. Declaring it kills three branches:

```python
self._ngrad_prefix = 1 if layout.grads_out_has_wavelet else 0
# :738-742
self.gbufs = ([torch.zeros_like(self.bp.forward_source)] * self._ngrad_prefix
              + [torch.zeros_like(m) for m in self.bp.models])
# :1207
model_grads = self.gbufs[self._ngrad_prefix:]
# :815  (the per-shot grad-wavelet resize)
if self._ngrad_prefix and self.bp is not None:
```

**Illumination → `CUDALayoutSpec.illum_nvar: int`** (2 for acoustic, 0 for elastic). Same category: `illum_out.size() == 2` (`acoustic2d/backward.cu:104-106`) versus *must be empty* (`elastic2d/backward.cu:310-312`). Allocation at `:745-750` becomes one unconditional comprehension:

```python
self.illum = [torch.zeros_like(self.bp.models[0]) for _ in range(layout.illum_nvar)]
```

Both values are already implied by `SlotTable` + the existing hand-written counts, so `test_slot_table_consistency.py` gains two more derived-equals-declared assertions and they cannot drift.

**Tail stop bound → `DDLoop.tail_truncatable: bool` + `_loop_floor` (shown in §2b).**
This one *is* loop shape: it moves where the reverse loop stops. Three things stay exactly where they are and must not be absorbed:

- **Legality** stays in `propagator/_c.py:1513-1516`, which raises `NotImplementedError` for any equation but Acoustic/Acoustic3D. The flag says "this loop knows how to move its floor"; it does not re-implement the refusal. Conflating them would let a spec author "enable" tail on elastic and get the error from a different module.
- **Rank invariance** survives verbatim, and for a stronger reason than before: `_loop_floor` reads only `self.nt` and `bp.boundary_tail_steps`, both replicated identically on every rank (`dd_propagator.py:245-252`, `:271`, `:336`), and `tail_truncatable` is a static constant. There is now no code path by which a floor could become rank-dependent — which the old inline computation inside one `elif` branch did not make obvious.
- **`drop_trailing_exchange_on_floor` is a separate flag**, not a consequence of truncation. The `if it == stop: break` at `:1159-1160` fires even with `stop == 0`, and it is sound only for exchanges attached to the *last* phase (consumed by the next step, which does not exist). VRZ's exchanges are attached to phases 1 and 2 and are consumed *within the same step* by phases 2 and 3 — dropping them on the floor iteration would be **wrong**. The interpreter's `ph is last_phase` guard encodes exactly that distinction; the per-loop flag then keeps today's behaviour where the sound generalisation would still change the wire (elastic's confirmed-dead trailing stress exchange).

---

## 5. Migration, and what world=1 cannot see

### The gates

- **G1** — `. gate/env.sh && $PY gate/ddgate.py --compare gate/base_dd1.pt`, world=1 on the dev box. 10 configs (5 equations × fs/nofs, `gate/ddgate.py:80-88`), comparing tile record and every model gradient with `torch.equal` (`_cmp`).
- **G2** — world=2 on 2×V100 via ibex. `gate/dd_verify.sbatch` **does not exist in the tree today**; `ddgate.py` already supports it, so the file is a wrapper:

```bash
#!/bin/bash
#SBATCH -N1 -n1 -c8 --gres=gpu:v100:2 -t 0:40:00 --job-name dd_verify
. gate/env.sh
srun torchrun --standalone --nproc-per-node=2 gate/ddgate.py \
     --ranks 2 --compare gate/base_dd2.pt
```

Creating it, and recording `base_dd2.pt`, is step −1.

### What G1 structurally cannot catch

1. **The acoustic forward's phased path never executes at world=1.** `_overlap_ok` requires `world > 1` (`:371`), so world=1 takes variant B: `run_to` with `step_phase == 0`. Every change to acoustic phase-1/phase-2 sequencing, and the whole overlap reordering, is invisible.
2. **Every exchange is a no-op.** `_halo` returns `None` for `world == 1` (`:427-428`). Therefore: a wrong field in a group, wrong tensor *order* inside a batched group, a group attached to the wrong phase, a dropped `when`-guarded shipment, a wrong number of NCCL rounds — **all pass G1 with byte-identical output.** This is the big one: a spec refactor is 80% about exchange placement, and G1 is blind to 100% of it.
3. **`cut_face_mask == 0`.** Every cut-aware kernel branch (`phys_x0/x1`, `cut_x_lo()`, the strip launch ranges at `acoustic2d/forward.cu:244-254`) is untaken; the asymmetric cut-aware pad (`:346-348`) degenerates.
4. **Rank-collective consistency.** A floor or predicate that became rank-dependent cannot deadlock one rank. G2 at world=2 catches it as a hang.
5. **The 2×2 / y-cut mesh.** `ddgate.configs` only emits `py=2` at `ranks >= 4` (`gate/ddgate.py:85-86`), so even G2 covers only `1×2`. `_halo_view`'s perpendicular-extent rule (`:437-456`) is documented as biting *only* at 2-axis meshes. Any FieldRef/crop change wants a 4-rank run.
6. **The tail path has no gate config at all.** `ddgate` never sets `tail_steps`, so `_loop_floor` is exercised only by `test/test_dd_tail_two_tile.py` — which is world=2. Close this in step 0.

Summary: **G1 gates the Python plumbing; G2 gates the schedule.** Any step that touches exchange structure is *not* gated by G1, regardless of how green it comes back.

### The order

**Step 0 — prep, zero behaviour change.**
Add `SlotTable` helpers (`recon` role lookup, `field_names(role)`); add `grads_out_has_wavelet` / `illum_nvar` to `CUDALayoutSpec` with today's values; extend `test_slot_table_consistency.py` to assert derived == hand-written; add a `tail_steps` config to `ddgate.configs` and re-baseline both `base_dd1.pt` and `base_dd2.pt`. Nothing reads the new fields yet.
*Gate:* pytest + G1 (must be byte-identical) + G2 baseline capture.

**Step 1 — output-binding declarations (§4).**
Driver reads `grads_out_has_wavelet` / `illum_nvar` at `:738-750`, `:815`, `:1207`. **This is the one step G1 covers end to end**: gradients are compared, and the two layouts differ in length and offset, so a wrong prefix is a loud failure at world=1.
*Gate:* G1 sufficient. G2 for completeness.

**Step 2 — spec types + interpreter, routing the elastic forward and elastic backward only.**
Acoustic/VRZ stay on the old code. G1 *does* execute both elastic loops phased (elastic's forward has no `cut_face_mask != 0` requirement — `elastic2d/backward.cu:273-323` has none either, unlike `acoustic2d/forward.cu:88-89`), so a wrong `step_phase`, a wrong phase order, or a wrong `advances` shows up as a wrong gradient at world=1.
*Gate:* G1 for phase sequencing and counters; **G2 mandatory** for the four exchange groups (`_PH1`/`_PH2` ordering, `inj_cross`, batching) — G1 cannot see any of them.

**Step 3 — VRZ backward through the interpreter.**
Exercises `prologue` and the `advances`-on-first-phase case. G1 runs all four phases (VRZ backward is phased at world=1), so a wrong `advances` bit corrupts the gradient visibly — phases 2 and 3 would bind pre-advance lists.
*Gate:* G1 for the counters; **G2 mandatory** for the coupling and coeff shipments.

**Step 4 — acoustic backward through the interpreter.**
Unphased single phase, `tail_truncatable`, `drop_trailing_exchange_on_floor`. G1 covers `_loop_floor` arithmetic *only if step 0 added the tail config*; the `drop_trailing` flag is invisible at world=1 by construction.
*Gate:* G1 (with tail config) + **G2** + `test/test_dd_tail_two_tile.py`.

**Step 5 — acoustic forward, serial variant, plus the overlap wrapper.**
The riskiest step, because of the `u_now`/`u_next` pre-advance resolution described in §3. Three runs required:
  - G1 — covers the serial variant only, with no exchanges. Weak.
  - **G2 with `SWEEP_DD_DISABLE_OVERLAP=1`** — serial variant with real exchanges. The env kill-switch at `:370` exists precisely as the bit-exact reference (`:365-367`).
  - **G2 without it** — the overlap path. Additionally run `test/dd_src_on_cut_check.py`, which is the only thing that exercises the `_src_away_from_cuts` fallback.

**Step 6 — deletion.**
Remove `_FAMILIES` (`:71-76`), `_forward_loop_acoustic` (`:1035-1064`), `_forward_loop_elastic` (`:1066-1078`), the three-way branch (`:1106-1194`), `self._st`, `self._nv`, `self._nphys`, and the dead `half_step` key. Fold `_family_of` into "does the equation declare a `dd_schedule`". Two constraints: `test/test_dd_supported_equations.py:21` imports `_DD_EQUATIONS`, `_FAMILIES` and `_family_of` by name and must be updated in the same commit; and `ddgate._cmp` compares **error text**, so the `NotImplementedError` message (`:116-125`) must stay byte-identical.
*Gate:* G1 + G2 + pytest. Pure deletion — any diff is a bug.

**Step 7 — optional, deliberately NOT bit-exact-on-the-wire.**
Flip acoustic backward's `batched=False → True` (merging the λ and recon-u rounds into one; the `_exchange_group` docstring's own rationale at `:467-468` — "acoustic exchanges a single field" — is simply false of this loop), and enable `drop_trailing_exchange_on_floor` for elastic. Values stay bit-exact (`FastHaloGroup` is documented bit-identical, `fast_halo.py:115-119`), but the NCCL round count changes, and the λ/recon-u merge interacts with `FastHaloGroup`'s stated fixed-address assumption (`:114-115`) — those tensors *rotate*, so the `_group_cache` keyed on data_ptrs (`:167-168`) grows to ≤9 entries instead of 3+3 exchangers. Workable, but it is a different kind of change and gets its own step with a round-count assertion alongside G2.

---

## 6. What the schema genuinely cannot express

Being honest here matters more than the schema being complete.

1. **What a `step_phase` *means*.** The spec says "call with 1". It cannot say that acoustic's 1 is a **spatial** strip split (`acoustic2d/forward.cu:244-254`) while elastic's 1 is a **physics** half-step (`elastic2d/forward.cu:172-173`, `do_v`/`do_s`). That asymmetry is why acoustic's phased forward hard-requires `cut_face_mask != 0` (`acoustic2d/forward.cu:88-89`) and elastic's does not — hence why elastic runs the full phased protocol at world=1 and acoustic would `TORCH_CHECK`-abort. The spec is a sequencer, not a semantics.

2. **Read/write sets, hence exchange *sufficiency*.** Nothing in a `DDSpec` proves that shipping `{adjoint velocity, recon stress}` after elastic phase 1 is enough. That proof lives in the kernels (`elastic2d/kernels.cuh:139-142`, `:282-284`). A spec can be internally consistent and ship the wrong fields. Declaring per-phase read sets is possible in principle but they are *stencil-conditional*: elastic's `m_*` slots are own-cell only (`kernels.cuh:186-198`), acoustic's psix taps are gated by the `cx` band and clamp on a cut face (`kernels.cuh:243-251`). Encoding "reads across the cut" faithfully means re-deriving the kernel in Python. I would not attempt it — the honest artifact is a parity test (`test/dd_corner_check*.py`), not a dataclass field.

3. **The ~12 C++ `TORCH_CHECK` preconditions per equation, and especially the *anti*-preconditions.** A `requires=(...)` tuple would only feed a startup assertion that duplicates checks the kernel already makes. Worse, some load-bearing facts are absences: elastic's phased backward has *no* `cut_face_mask != 0` check, which is exactly what makes a world=1 elastic DD run legal. A sequencer has no vocabulary for "this check is deliberately missing".

4. **`first_segment` coupling.** `first_segment = (it_hi == nt)` (`acoustic2d/backward.cu:443`, `elastic2d/backward.cu:1253`, VRZ via `_sc_recompute` at `acoustic_vrz3d/backward.cu:348-349`) makes the loop's first call do extra work — seed from `u_last_two`, rim-zero, build scratch. *Which phase* that lands in is a C++ decision that has already moved once: elastic relocated the seed into phase 3 precisely because seeding in phase 1 would erase the first step's source un-injection (`elastic2d/backward.cu:1338-1349`). A spec that permutes phases would silently move the seed. The schema records the order; it cannot record that the order is load-bearing for a reason outside itself.

5. **`_capture`'s hidden full-`nt` unphased forward.** `_prepare_call` calls `_capture` on the first call (`:982`) and again on lazy adjoint promotion (`:1014`); `fwrap` invokes the real compiled forward (`:596-598`). That is a complete unphased `nt`-step run per rank, with `cut_face_mask` set and no exchange. Numerically inert (`L_fwd` and `record` are zeroed at `:1030-1032`) but a real execution and a real cost — and it is not a phase of any loop, so a time-loop schema has nowhere to put it.

6. **Per-step host prologue duplication.** Elastic's backward re-runs `mu`/`lambda` allocation, `neg_forward_source`, `boundary_saver.allocate` and `prefetch_initial_backward_chunk` three times per reverse step (`elastic2d/backward.cu:1332-1333`, `:1351`, `:1368-1377`, `:1398`). The spec makes the repetition *visible* (three phases where monolithic had one call) but cannot express "hoist this". Fixing it is a C++ change.

7. **Per-tile ownership bookkeeping.** `self.record.zero_()` when a tile owns no real receivers (`:884-885`); the zero-amplitude dummy source (`:951`, `:973-974`) that a source-less tile carries and which may sit inside a cut strip. Orthogonal to the time loop, and correctly so.

8. **`nt == 1` for VRZ.** The prologue's segment `(nt, nt-1) == (1, 0)` makes `bw_stepped()` false (`csrc/shared/wavetypes.h:286-287`), the stepped preconditions are skipped while the phased ones still run, and the loop body is empty → a silent zero gradient. `min_nt` on `DDLoop` would be forcing it; this is a driver-side assertion.

9. **Whether the overlap actually overlaps.** `dist.batch_isend_irecv` + `req.wait()` under `torch.cuda.stream(comm)` puts NCCL kernels on NCCL's own internal stream. Whether phase 2 genuinely runs concurrently with the P2P is a property of NCCL's channel behaviour, not of anything in this repo. `overlappable=True` declares *permission*, never *benefit* — and the schema should not pretend otherwise.