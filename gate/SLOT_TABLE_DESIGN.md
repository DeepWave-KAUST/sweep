Read the three consumers and the C++ bind sites, then prototyped the schema and ran the derivations against the repo's hard-coded values. Everything below is verified, not asserted — the prototype lives at `/tmp/claude-215201/-ibex-user-wangs0j-sweep-stack/1ee29558-b4a1-4d23-b589-713261f8ec7b/scratchpad/slot_proto.py` and reproduces every current constant exactly.

---

# 1. The schema

New module `/home/wangs0j/sweep-local/dd-refactor/src/sweep/equations/slot_table.py`, imported by `cuda_layout.py`. The table is per **equation**, not per C++ struct — `Acoustic` and `AcousticVRZ` share `AcousticWavefieldTensor::bind()` (`src/sweep/csrc/cuda/common/acoustic.h:324`) but have different maximal lists (11 vs 9), so the struct is the wrong owner.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Role = Literal[
    "u",         # rotating time-level buffer (2nd-order-in-time)
    "vel", "stress", "phys",      # non-rotating physical state
    "pml_psi", "pml_zeta", "pml_mem",   # CPML auxiliaries
    "dbuf",      # double-buffer shadow of another slot (swap_pml/swap_aux)
]
PHYS_ROLES = frozenset({"u", "vel", "stress", "phys"})
PML_ROLES  = frozenset({"pml_psi", "pml_zeta", "pml_mem", "dbuf"})

@dataclass(frozen=True)
class Slot:
    name: str                      # C++ member name with the "_t" stripped
    role: Role
    axis: str | None = None        # 'x'|'y'|'z' DIFFERENCING axis; None = no axis
    dbuf_of: str | None = None     # role=="dbuf": name of the slot it shadows
    adjoint_only: bool = False     # bound in adjoint_wavefields, never in wavefields
    reserved: bool = False         # bound + checkpointed but no kernel reads it

@dataclass(frozen=True)
class SlotTable:
    """Positional truth for ONE equation's CUDA wavefield list.

    ``slots`` is literally the ``tensors[i++]`` sequence of that equation's
    bind(), maximal variant (adjoint list).  The forward list is the
    subsequence with ``adjoint_only=False`` — which for every family in the
    library is also a PREFIX, because every bind() is one straight-line i++
    run with the adjoint extras last.
    """
    slots: tuple[Slot, ...]
    ring: tuple[tuple[int, ...], ...] = ()        # rotating blocks, e.g. ((0,1,2),)
    recon: tuple[str, ...] | None = None          # backward_bs forward_wavefields list
    checkpoint: tuple[str, ...] | None = None     # checkpoint_tensors() order (a PERMUTED SUBSET)
    field_ids: tuple[str, ...] | None = None      # source/receiver field-index order
    aux_storage: Literal["full", "slab"] = "full" # POLICY, not fact — see §4
    accepted_lengths: tuple[int, ...] | None = None  # C++ TORCH_CHECK sizes; default (n_forward, n_adjoint)
```

Hanging it off the existing spec, additively:

```python
# src/sweep/equations/cuda_layout.py  (all 14 existing fields untouched)
@dataclass(frozen=True)
class CUDALayoutSpec:
    ...
    slots: "SlotTable | None" = None   # NEW, optional, default None
```

Derivations as `SlotTable` properties (the whole point — everything else falls out):

```python
    def _fwd(self):  return [s for s in self.slots if not s.adjoint_only]
    def index(self, name): return [s.name for s in self.slots].index(name)

    @property
    def base_nvar(self):  return sum(1 for s in self._fwd() if s.role in PHYS_ROLES)
    @property
    def pml_nvar(self):   return sum(1 for s in self._fwd() if s.role in PML_ROLES)
    @property
    def adjoint_extra_nvar(self): return sum(1 for s in self.slots if s.adjoint_only)
    @property
    def n_forward(self):  return len(self._fwd())
    @property
    def n_adjoint(self):  return len(self.slots)

    def pairs(self, *, adjoint: bool):
        """(shadowed, shadow) index pairs, ordered by the SHADOW slot's index."""
        return tuple((self.index(s.dbuf_of), i)
                     for i, s in enumerate(self.slots)
                     if s.role == "dbuf" and (adjoint or not s.adjoint_only))

    @property
    def u_blocks(self):   return tuple(b[0] for b in self.ring)
    @property
    def block_size(self): return len(self.ring[0]) if self.ring else 3
    @property
    def vel_idx(self):    return tuple(i for i, s in enumerate(self.slots) if s.role == "vel")
    @property
    def phys_idx(self):   return tuple(i for i, s in enumerate(self._fwd()) if s.role in PHYS_ROLES)
    @property
    def nrecon(self):     return None if self.recon is None else len(self.recon)

    @property
    def pml_slot_axes(self):
        if self.aux_storage != "slab": return None          # <-- back-compat gate
        return tuple(s.axis for s in self._fwd() if s.role in PML_ROLES)
    @property
    def checkpoint_slot_axes(self):
        if self.aux_storage != "slab" or self.checkpoint is None: return None
        return tuple(self.slots[self.index(n)].axis for n in self.checkpoint)
    @property
    def checkpoint_nvar(self):
        return None if self.checkpoint is None else len(self.checkpoint)
```

Two hard cases, both expressed without special-casing:

**Acoustic adjoint-only zeta double-buffer** — the `zetaxn/zetazn` (2-D) and `zetaxn/zetazn/zetayn` (3-D) slots are just `Slot(..., role="dbuf", dbuf_of="zetax", adjoint_only=True)`. They fall out of `_fwd()` so `pml_nvar` stays 6/9, and they land in `adjoint_extra_nvar` = 2/3 and in `pairs(adjoint=True)` only:

```python
ACOUSTIC2D = SlotTable(
    slots=(Slot("u_prev","u"), Slot("u_now","u"), Slot("u_next","u"),
           Slot("psix","pml_psi","x"),  Slot("psiz","pml_psi","z"),
           Slot("zetax","pml_zeta","x"), Slot("zetaz","pml_zeta","z"),
           Slot("psixn","dbuf","x",dbuf_of="psix"),
           Slot("psizn","dbuf","z",dbuf_of="psiz"),
           Slot("zetaxn","dbuf","x",dbuf_of="zetax",adjoint_only=True),
           Slot("zetazn","dbuf","z",dbuf_of="zetaz",adjoint_only=True)),
    ring=((0,1,2),),
    recon=("u_prev","u_now","u_next"),
    checkpoint=("u_prev","u_now","psix","psiz","zetax","zetaz"),
    aux_storage="slab", accepted_lengths=(3,7,9,11))

ACOUSTIC_VRZ2D = replace(ACOUSTIC2D,                     # same bind(), different equation
    slots=tuple(s for s in ACOUSTIC2D.slots if not s.adjoint_only),
    aux_storage="full", accepted_lengths=(3,9))
```

**Elastic fixed-slot (no rotation)** — `ring=()`. Then `u_blocks == ()` and `pairs(...) == ()` for both directions, which is exactly what `SteppedBindingRunner`/`SteppedBackwardRunner` are handed today at `dd_propagator.py:1036` and `:1131`. No `half_step`, no family enum, no `if family == "elastic"` — the absence of `role=="u"` slots and the empty `ring` *are* the statement.

The 3-D mid-list insertions (`psiy@7`, `zetay@8`, `vy@1`, `m_vxy@10`) need no mechanism at all: the table is the bind sequence, so insertion is just where you type the `Slot`.

---

# 2. Derivation of the three hand-maintained copies, and whether they provably match

I built `ACOUSTIC2D/3D` and `ELASTIC2D/3D` tables in the prototype and printed every derived quantity. Results:

### (b) `_stepped.py` pair tuples — `src/sweep/propagator/_stepped.py:60,61,66-68,69-71`

```
ACOUSTIC2D_PSI_PAIRS = table.pairs(adjoint=False)
ACOUSTIC3D_PSI_PAIRS = table.pairs(adjoint=False)
ACOUSTIC2D_ADJ_PAIRS = table.pairs(adjoint=True)
ACOUSTIC3D_ADJ_PAIRS = table.pairs(adjoint=True)
u_blocks / adj_u_blocks / recon_u_blocks = table.u_blocks
```

| current | derived | match |
|---|---|---|
| `((3,7),(4,8))` | `((3,7),(4,8))` | ✅ |
| `((3,9),(4,10),(7,11))` | `((3,9),(4,10),(7,11))` | ✅ |
| `((3,7),(4,8),(5,9),(6,10))` | same | ✅ |
| `((3,9),(4,10),(7,11),(5,12),(6,13),(8,14))` | same | ✅ |

The 3-D adjoint tuple's *non-monotone* first components `(3,4,7,5,6,8)` are reproduced because the canonical order is "sorted by the **shadow** slot index" (9,10,11,12,13,14), which the enumeration gives for free. **All four match, provably, for every family** (elastic/DAS/VTI/TTI all yield `()`, matching the explicit `()`s at `dd_propagator.py:1036,1131`).

Bonus: the VRZ discriminator hack at `dd_propagator.py:1103-1106` (`adj_pairs if adjoint_extra_nvar else psi_pairs`) disappears — VRZ's table has no `adjoint_only` dbuf slots, so `pairs(adjoint=True) == pairs(adjoint=False)` automatically. That comment block (`:1095-1102`) documents a bug class the schema makes unrepresentable.

### (c) `cuda_layout.py` positional tuples

| spec field | site | current | derived | match |
|---|---|---|---|---|
| `base_nvar` | `acoustic.py:189` / `acoustic3d.py:156` | 3 / 3 | 3 / 3 | ✅ |
| `pml_nvar` | `acoustic.py:195` / `acoustic3d.py:159` | 6 / 9 | 6 / 9 | ✅ |
| `adjoint_extra_nvar` | `acoustic.py:198` / `acoustic3d.py:162` | 2 / 3 | 2 / 3 | ✅ |
| `checkpoint_nvar` | `acoustic.py:201` / `acoustic3d.py:165` | 6 / 8 | 6 / 8 | ✅ |
| `pml_slot_axes` | `acoustic.py:204` | `('x','z','x','z','x','z')` | identical | ✅ |
| `pml_slot_axes` | `acoustic3d.py:169` | `('x','z','x','z','y','y','x','z','y')` | identical | ✅ |
| `checkpoint_slot_axes` | `acoustic.py:205` | `(None,None,'x','z','x','z')` | identical | ✅ |
| `checkpoint_slot_axes` | `acoustic3d.py:170` | `(None,None,'x','y','z','x','y','z')` | identical | ✅ |
| `base/pml` | `elastic.py:359,360` / `elastic3d.py:575,576` | 5,10 / 9,27 | 5,10 / 9,27 | ✅ |
| `pml_slot_axes` | `elastic.py:368` / `elastic3d.py:584` | `('x','z')*5` / `('x','y','z')*9` | identical | ✅ |
| `checkpoint_slot_axes` | `elastic.py:369-370` / `elastic3d.py:585` | `(None,)*5+('x','z')*5` / `(None,)*9+('x','y','z')*9` | identical | ✅ |

**All match.** The families that would *not* match are exactly those whose current value is `None` while the table knows a real axis letter: `AcousticVRZ`/`AcousticVRZ3D` (`acoustic_vrz.py:237-246,356-365`), `ElasticVRR` (`elastic_vrr.py:477-478`), `DASZhao`/`DASZhao3D`, `DASMu`/`DASMu3D`, `AcousticVTI1st`/`3D`, `ElasticTTISG`. That is not a derivation error — it is the fact/policy split handled by `aux_storage` (§4).

### (a) `dd_propagator._FAMILIES` — `src/sweep/parallel/dd_propagator.py:70-75`

```python
nwf    = table.n_forward                  # forward wavefield list length
nphys  = table.base_nvar                  # physical prefix
nv     = len(table.vel_idx)               # velocity slots
nrecon = table.nrecon                     # len(table.recon), see §3
```

| entry | current | derived (2-D / 3-D) | match |
|---|---|---|---|
| acoustic `nwf` | `(9,12)` | 9 / 12 | ✅ |
| elastic `nwf` | `(15,36)` | 15 / 36 | ✅ |
| elastic `nphys` | `(5,9)` | 5 / 9 | ✅ |
| elastic `nv` | `(2,3)` | 2 / 3 | ✅ |
| acoustic `nrecon` | `(3,3)` | 3 / 3 | ✅ |
| elastic `nrecon` | `(7,12)` | 7 / 12 | ✅ |
| acoustic `nphys`/`nv` | `None` | 3 / 0 | **n/a** — guarded by `if self.family == "elastic"` at `dd_propagator.py:385-387`; derived values are correct and the guard can go |
| `half_step` | `False`/`True` | — | **dead field**: `grep -rn half_step src/sweep/` returns only the two definition sites `:72` and `:74`. Nothing reads it. Delete it rather than derive it. |

**Two mismatches to fix while collapsing, both pre-existing:**

1. `dd_propagator.py:679` sizes the *adjoint* fallback list with `self._nwf`:
   ```python
   self.L_adj = [torch.zeros_like(self.bp.models[0]) for _ in range(self._nwf)]
   ```
   For acoustic that is 9/12, but `acoustic2d/backward.cu:98` / `acoustic3d/backward.cu:122` require 11/15. It is latent only because `bp.adjoint_wavefields` is never actually empty (`_c.py:1238,1254` always allocates). The derived expression is `table.n_adjoint` (11/15), which is the correct one. `dd_propagator.py:627` (forward fallback) correctly wants `n_forward` — the schema makes the two visibly different quantities instead of one `_nwf`.

2. `dd_propagator.py:1038` (`stress = [self.L_fwd[f] for f in range(self._nv, self._nphys)]`) and `:1134-1137` assume velocities are a contiguous prefix and stresses follow. True for elastic, but it is an unstated invariant. Derive `table.vel_idx` / `table.stress_idx` (explicit index tuples) instead of the two counts and the assumption is gone — and it is the assumption that breaks on DAS-Zhao, whose physical block is *split* around the PML block (`das.h:83-99`: phys 0-5, PML 6-13, projections 14-16).

---

# 3. `nrecon` — where its truth actually lives, and how far it is derivable

`nrecon` is the length of `params.forward_wavefields` on the **stepped boundary-saving backward**. Its true definition is a C++ `TORCH_CHECK` in each equation's `backward.cu`, and there is currently **no Python-side declaration at all** — `_FAMILIES` is the only copy, and it is unvalidated.

| family | C++ site | value | what it is |
|---|---|---|---|
| acoustic 2-D | `acoustic2d/backward.cu:108` `TORCH_CHECK(p.forward_wavefields.size()==3)`, bound `use_pml=false` at `:478` | 3 | the no-PML **prefix** = `base_nvar` |
| acoustic 3-D | `acoustic3d/backward.cu:132`, bound at `:832` | 3 | same |
| elastic 2-D | `elastic2d/backward.cu:314` — message names it: `[vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]`; the split bind is `elastic2d/backward.cu:1316-1325` (first 5 → `ElasticWavefieldTensor::bind(..., use_pml=false)`, `[5]`/`[6]` → `fvx_prev`/`fvz_prev`) | 7 | `base_nvar` + **velocity carries** |
| elastic 3-D | `elastic3d/backward.cu:536`, split bind at `:843-853` (first 9, then `[9..11]`) | 12 | same |

So the closed form is

```
nrecon = base_nvar + n_carry
n_carry = ndim  for a staggered first-order scheme whose gradient kernel
                consumes v at time it+1  (elastic)
        = 0     for a second-order-in-time scheme (acoustic)
```

**Is it derivable from the slot table? Only with one extra declaration — and that declaration is legitimate, not a fudge.** The carry tensors (`fvx_prev`, …) are *not* bind slots: they do not appear in `ElasticWavefieldTensor::bind()` at all, they are separate `torch::Tensor` locals spliced onto the same Python list. Nothing in the bind order can tell you they exist. The knowledge "the gradient kernel needs the previous-step velocity" is a property of `backward_bs`, not of the layout.

Recommended form — declare the list by name, derive the count:

```python
ELASTIC2D = SlotTable(
    slots=(...15 slots...),
    ring=(),
    recon=("vx","vz","sxx","szz","sxz","fvx_prev","fvz_prev"),   # <- the source of truth
    ...)
# nrecon == len(recon) == 7
```

This is strictly better than `nrecon=(7,12)` because the *names* are what `elastic2d/backward.cu:1318-1319` spells out in its error message, so a Python-vs-C++ drift test can compare strings, not just a count. The alternative — a rule `recon = phys_prefix + [f"f{v}_prev" for v in vel_names] if recon_velocity_carry else phys_prefix` — is derivable from one bool, but it silently assumes the phys block is a prefix (false for DAS-Zhao) and invents names. Prefer the explicit tuple.

Also derivable once `recon` exists: the reconstruction list's own rotation, `recon_u_blocks` (`_stepped.py:276`) = the `ring` blocks whose members are all in `recon` → `(0,)` for acoustic, `()` for elastic — matching `dd_propagator.py:1131`.

---

# 4. Back-compat plan

**Invariant: `slots=None` must be indistinguishable from today, and `slots=<table>` must not change a single allocated byte.**

The one genuinely dangerous trap is `pml_slot_axes`. The `axis` letter on a `Slot` is a *fact* (which derivative that CPML memory variable accumulates — verified against kernels for DAS at `das2d/kernels.cuh:206,209,212,215`, for TTI at `elastic_tti_sg2d/kernels.cuh:301-306,408-413`, for VTI at `acoustic_vti_1st_2d/kernels.cuh:143-147,222-226`). Whether the runtime *slab-allocates* that slot is a *policy* — `_c.py:1195-1200` branches on `if not axes`. Ten equations currently declare `pml_slot_axes=None` and get full-grid aux. If `pml_slot_axes` were derived unconditionally from `axis`, adding a table to `DASZhao` would silently switch it to slab allocation and change results. Hence the `aux_storage: "full" | "slab"` gate: the derived property returns `None` unless the equation opts in, exactly reproducing today's behaviour for all ten.

Phased rollout:

**Phase 0 — additive, zero consumer change.**
Add `slot_table.py`, add `CUDALayoutSpec.slots = None`, declare tables for `Acoustic`, `Acoustic3D`, `AcousticVRZ`, `AcousticVRZ3D`, `Elastic`, `Elastic3D`. Add one test:

```python
# test/test_slot_table_consistency.py
@pytest.mark.parametrize("eq", ALL_EQUATIONS_WITH_SLOTS)
def test_derived_equals_declared(eq):
    spec, t = eq.cuda_layout, eq.cuda_layout.slots
    assert t.base_nvar          == spec.base_nvar
    assert t.pml_nvar           == spec.pml_nvar
    assert t.adjoint_extra_nvar == spec.adjoint_extra_nvar
    assert t.pml_slot_axes        == spec.pml_slot_axes
    assert t.checkpoint_slot_axes == spec.checkpoint_slot_axes
    assert (t.checkpoint_nvar or spec.resolved_checkpoint_nvar()) \
           == spec.resolved_checkpoint_nvar()

def test_stepped_pairs_match_hardcoded():
    assert ACOUSTIC2D.pairs(adjoint=False) == _stepped.ACOUSTIC2D_PSI_PAIRS
    assert ACOUSTIC3D.pairs(adjoint=True)  == _stepped.ACOUSTIC3D_ADJ_PAIRS
    ...

def test_dd_families_match_hardcoded():
    for fam, (t2, t3) in {"acoustic": (ACOUSTIC2D, ACOUSTIC3D),
                          "elastic":  (ELASTIC2D,  ELASTIC3D)}.items():
        f = dd_propagator._FAMILIES[fam]
        assert f["nwf"]    == (t2.n_forward, t3.n_forward)
        assert f["nrecon"] == (t2.nrecon,    t3.nrecon)
```
This is where the collapse pays off before any refactor risk is taken: from here on, drift is a red test.

**Phase 1 — consumers read the table with a legacy fallback.**

```python
# src/sweep/equations/slot_table.py
def slot_table_of(equation):
    return getattr(getattr(equation, "cuda_layout", None), "slots", None)

# src/sweep/propagator/_stepped.py  (keep the module constants as the fallback)
def psi_pairs_for(equation, ndim):
    t = slot_table_of(equation)
    return t.pairs(adjoint=False) if t is not None else acoustic_psi_pairs(ndim)

def adj_pairs_for(equation, ndim):
    t = slot_table_of(equation)
    if t is not None:
        return t.pairs(adjoint=True)
    extra = getattr(equation.cuda_layout, "adjoint_extra_nvar", 0)   # legacy VRZ hack
    return acoustic_adj_pairs(ndim) if extra else acoustic_psi_pairs(ndim)
```

```python
# src/sweep/parallel/dd_propagator.py — _FAMILIES becomes the fallback branch
t = slot_table_of(equation)
if t is not None:
    self._nwf, self._nadj = t.n_forward, t.n_adjoint
    self._nrecon = t.nrecon
    self._vel_idx, self._phys_idx = t.vel_idx, t.phys_idx
else:
    st = _FAMILIES[self.family]; i = 0 if self.ndim == 2 else 1
    self._nwf = self._nadj = st["nwf"][i]        # <- preserves today's :679 behaviour
    self._nrecon = st["nrecon"][i]
    ...
```
Note the fallback deliberately keeps `_nadj = _nwf` so the legacy path is byte-identical, including the `:679` wart; only the table path gets the corrected `n_adjoint`. `_family_of` (`dd_propagator.py:107-123`) and `_DD_EQUATIONS` (`:97-104`) stay as-is — they encode "has a stepped forward *and* backward", which is a C++ capability claim the slot table does not and should not make.

**Phase 2 — allocation.** Replace the block-structured `_c.py:1190-1200` with a per-slot loop, still gated:

```python
def _forward_wavefield_shapes(self):
    cl = self._cuda_layout(); t = getattr(cl, "slots", None)
    if t is not None and t.aux_storage == "slab":
        return [self._aux_slab_shape(s.axis, [self.B, 1]) if s.axis
                else [self.B, 1, *self.shape_cuda] for s in t._fwd()]
    ...legacy base+pml path unchanged...
```
This is the one place the table buys correctness rather than just deduplication: the current code assumes "all `base_nvar` slots first, then all `pml_nvar`", which is **false for DAS-Zhao** (`das.h:83-99`). Latent today only because `DASZhao.pml_slot_axes` is `None` (`das.py:1128-1134`); adding one would slab-allocate slots 9-16 instead of 6-13 and hand slabs to `das35/das54x/das54z`. The per-slot loop makes that bug unrepresentable.

**Phase 3 — delete.** Once all four DD equations declare a table, `_FAMILIES` and the four `ACOUSTIC*_PAIRS` tuples can be deleted (or kept as `_LEGACY_*` behind the fallback for out-of-tree equations). `pml_slot_axes` / `checkpoint_slot_axes` / `base_nvar` / `pml_nvar` / `adjoint_extra_nvar` / `checkpoint_nvar` stay as public fields forever, computed in `__post_init__` when `slots` is given and left alone when it isn't. Never make `slots` required: an equation without one must keep hitting exactly the same `NotImplementedError` it does today.

---

# 5. Where the table does NOT fit — honestly

**5.1 `AcousticLSRTM` / `AcousticLSRTM3D` — needs a block dimension the flat table lacks.** Their lists are 18 / 24 tensors that are **two back-to-back copies** of the 9- / 12-slot acoustic layout: `acoustic_lsrtm2d/forward.cu:70-71` binds `[0,9)` as `bg` and `[9,18)` as `sc`; `acoustic_lsrtm3d/forward.cu:71-72` binds `[0,12)` and `[12,24)`. A flat 18-tuple of `Slot`s does yield the right `base_nvar=6` / `pml_nvar=12` (`acoustic_lsrtm.py:166,169`), but it cannot express (i) that indices 0-8 and 9-17 are two independently-bound structs, (ii) that the reconstruction path binds `[0,7)` of *each block* (`acoustic_lsrtm2d/backward.cu:391,570`), or (iii) that `dbuf_of` must resolve within a block. This needs either `blocks: tuple[tuple[int,int],...]` or a `group: str` field on `Slot`. **I would not force LSRTM into v1** — declare `slots=None` for it and leave it on the legacy path.

Related: the table *would* make the confirmed defect at `acoustic_lsrtm3d/backward.cu:619` (`bind(slice(fw,0,9), 3, /*use_pml=*/false)`, which `acoustic.h:341-345` rejects since it requires exactly 3 when `use_pml=false`) a statically checkable inconsistency — `9 ∉ accepted_lengths_no_pml`. Its siblings at `:843` and `acoustic_lsrtm2d/backward.cu:391,570` all pass `true`.

**5.2 There are FOUR index namespaces per equation, and no single table unifies them.** The slot table describes the CUDA bind list. It cannot also be:
- the **eager field vector**, because for `Acoustic` `FIELD_SPECS` is 6 entries `(h1,h2,psix,psiz,zetax,zetaz)` (`acoustic.py:57`) and for `Acoustic3D` 8 — i.e. the *checkpoint* layout, not the 9/12-slot bind layout;
- the **source/receiver field-id space**, because for `Elastic3D` `FIELD_SPECS` is a 27-entry *reduced* CPML set (`elastic3d.py:355-383`) and `elastic_field_ptr` follows that order (`elastic.h:847-876`), so field id 19 = `m_szzz` while bind index 19 = `m_sxxy`. Agreement holds only for 0..18.

The `field_ids: tuple[str,...]` slot is there to *record* that divergence (a reordered subset of slot names), not to remove it. Any consumer must keep asking "which namespace is this integer in?" — the table makes the question answerable, not moot.

**5.3 `ElasticVRR` — index-identical twin, no mechanism to keep it in sync.** Same bind order as `Elastic` (`elastic_vr2d/forward.cu:49-56` reuses `ElasticWavefieldTensor`), but slots 0/1 carry momentum `px = ρvx` and the Python names are `px,pz,m_pxx,…` (`elastic_vrr.py:337-356`). Expressible as a second table, but the schema does not *enforce* that the two stay positionally identical. A `SlotTable.rename({"vx":"px", ...})` constructor is the mitigation; the risk (someone edits one table and not the other) remains real.

**5.4 Variant selection by list length is a C++ inference the table can only mirror, not own.** `acoustic.h:370-374` derives `double_buffer_aux`/`double_buffer_psi` **from `tensors.size()`** — the length *chooses the semantics*. I verified empirically that every accepted length is a prefix of the maximal table (2-D: 3,7,9,11; 3-D: 3,9,12,15 — note 3-D `7` is a prefix but is *not* accepted, `acoustic.h:341-345`), so `accepted_lengths` plus prefix semantics reproduces it. But that is an observed property of today's `bind()`, not something the schema guarantees. If someone adds a variant where a shorter list means *different roles* rather than *fewer roles*, the table silently lies. Guard it with a test that parses the `TORCH_CHECK` size literals out of `acoustic.h:335-345` / each `backward.cu` and compares to `accepted_lengths`.

**5.5 Things adjacent to the wavefield list that are deliberately out of scope** (each is its own positional order, each currently hand-maintained, none covered): `ElasticAdjointWorkspaceTensor::bind` 8/18 tensors (`elastic.h:686-693,696-713`) vs VRR's flat 14-slot workspace (`elastic_vr2d/backward.cu:356-369`); `ElasticCPMLTensor::allocate`'s Z-first profile order with the y-block again inserted mid-list (`elastic.h:32-58`); the 16-entry TTI stiffness list (`elastic_tti_sg2d/tensors.h:105-161`); the `u_last_two` snapshot sets. The schema extends to them the same way (`workspace: tuple[str,...]`, `pml_profiles: tuple[str,...]`), but I would land the wavefield list alone first.

**5.6 Two spec-vs-C++ mismatches the table cannot repair, only surface.** `DASZhao3D.last_two_storage_nvar=13` (`das.py:1258`) is dead metadata — `das3d/forward.cu:174` emits `torch::empty({0})` and `das3d/backward.cu` never references `u_last_two`. And `DASMu` 2-D slots 13/14 (`m_sxxz`, `m_szzx`) are bound and checkpointed but read by no kernel (`das.py:1296-1297` calls them "Reserved elastic auxiliary field") — that is what `Slot.reserved` is for, but their `axis` letter is name-derived and physically meaningless, so a slab allocator keyed on it would be sizing a buffer nothing reads.

---

**Files to create/touch:** new `/home/wangs0j/sweep-local/dd-refactor/src/sweep/equations/slot_table.py`; `+1 field` in `/home/wangs0j/sweep-local/dd-refactor/src/sweep/equations/cuda_layout.py`; table declarations in `acoustic.py:187`, `acoustic3d.py:154`, `acoustic_vrz.py:235,354`, `elastic.py:357`, `elastic3d.py:573`; fallback accessors in `/home/wangs0j/sweep-local/dd-refactor/src/sweep/propagator/_stepped.py:60-80`; fallback branch in `/home/wangs0j/sweep-local/dd-refactor/src/sweep/parallel/dd_propagator.py:70-75,383-387,679,1103-1109`; per-slot shapes in `/home/wangs0j/sweep-local/dd-refactor/src/sweep/propagator/_c.py:1190-1200`.