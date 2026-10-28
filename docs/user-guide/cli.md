# CLI

SWEEP ships a small command-line interface for inspecting the available
equations and their compiled-binding status, plus `sweep datasets` for the
benchmark velocity models. The CLI is intentionally narrow — it answers
introspection questions about the installed engine, not full FWI / LSRTM task
runs.

Implementation: `src/sweep/cli.py`.

## `sweep list equations`

Lists every equation class in two tables: the finite-difference ones exported
by `sweep.equations`, run with `PropTorch`, and the spectral-element ones of
`sweep.sem`, run with `PropSEM`. Each row gives the required model
parameters, whether the class has a compiled CUDA kernel, and whether the
CUDA core can run here right now; an SEM row also gives the mesh family.

```bash
sweep list equations
```

Example output on a machine with a CUDA GPU where the shipped core fits:

```text
Available equations:

Finite difference (sweep.equations, run with PropTorch):

  Equation               Models                                                              Torch Binding  Binding Ready
  ---------------------  ------------------------------------------------------------------  -------------  -------------
  Acoustic               ['vp']                                                              yes            yes
  Acoustic1st            ['vp', 'rho']                                                       no             no
  Acoustic3D             ['vp']                                                              yes            yes
  AcousticBorn           ['vp', 'mp']                                                        yes            yes
  AcousticBorn3D         ['vp', 'mp']                                                        yes            yes
  AcousticCurvilinear    ['vp']                                                              no             no
  AcousticTTI            ['vp', 'epsilon', 'delta', 'theta']                                 yes            yes
  AcousticTTIAlkhalifah  ['vv', 'v', 'eta']                                                  no             no
  AcousticTTIBorn        ['vp', 'epsilon', 'delta', 'theta', 'mp']                           yes            yes
  AcousticTTILiang       ['vp', 'epsilon', 'delta', 'theta']                                 yes            yes
  AcousticTariq          ['vv', 'v', 'eta']                                                  no             no
  AcousticVRR            ['vp', 'rx', 'rz']                                                  no             no
  AcousticVRZ            ['vp', 'z']                                                         yes            yes
  AcousticVRZ3D          ['vp', 'z']                                                         yes            yes
  AcousticVTI            ['vp', 'epsilon', 'delta']                                          yes            yes
  AcousticVTI1st         ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTI1st3D       ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIAlkhalifah  ['vv', 'v', 'eta']                                                  no             no
  AcousticVTIBorn        ['vp', 'epsilon', 'delta', 'mp']                                    yes            yes
  AcousticVTIDefault     ['vp', 'epsilon', 'delta']                                          yes            yes
  AcousticVTIDefault3D   ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIDuveneck    ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIDuveneck3D  ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTILiang       ['vp', 'epsilon', 'delta']                                          yes            yes
  Biot                   ['H', 'mu', 'C', 'M', 'rho', 'rho_f', 'm', 'b']                     no             no
  DASElastic             ['vp', 'vs', 'rho']                                                 yes            yes
  DASElastic3D           ['vp', 'vs', 'rho']                                                 yes            yes
  DASMu                  ['vp', 'vs', 'rho']                                                 yes            yes
  DASMu3D                ['vp', 'vs', 'rho']                                                 yes            yes
  DASZhao                ['vp', 'vs', 'rho']                                                 yes            yes
  DASZhao3D              ['vp', 'vs', 'rho']                                                 yes            yes
  Elastic                ['vp', 'vs', 'rho']                                                 yes            yes
  Elastic3D              ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticBorn            ['vp', 'vs', 'rho', 'dvp', 'dvs', 'drho']                           yes            yes
  ElasticCurvilinear     ['vp', 'vs', 'rho']                                                 no             no
  ElasticTTI             ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  no             no
  ElasticTTI2nd          ['vh', 'vs', 'rho', 'epsilon', 'eta', 'theta']                      yes            yes
  ElasticTTISG           ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  yes            yes
  ElasticTTISG3D         ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  yes            yes
  ElasticVRR             ['vp', 'vs', 'Rp_x', 'Rp_z', 'Rs_x', 'Rs_z']                        yes            yes
  ViscoAcoustic          ['vp', 'Q', 'omega']                                                yes            yes
  ViscoElastic           ['vp', 'vs', 'rho', 'Qp', 'Qs']                                     yes            yes

Spectral element (sweep.sem, run with PropSEM):

  Equation                    Mesh                   Models                                                              Torch Binding  Binding Ready
  --------------------------  ---------------------  ------------------------------------------------------------------  -------------  -------------
  AcousticElasticSEM2D        quad9 2-D, topography  ['vp', 'vs', 'rho']                                                 yes            yes
  AcousticPotentialSEM2D      quad9 2-D, topography  ['vp']                                                              yes            yes
  AcousticSEM2D               quad4 2-D              ['vp']                                                              no             no
  AcousticSEM2DDeformed       quad9 2-D, topography  ['vp']                                                              no             no
  AcousticSEM3D               hex8 3-D               ['vp']                                                              yes            yes
  AcousticSEM3DDeformed       hex27 3-D, topography  ['vp']                                                              yes            yes
  BiotAnisoPermSEM2DDeformed  quad9 2-D, topography  ['H', 'mu', 'C', 'M', 'rho', 'rho_f', 'm', 'b_xx', 'b_xz', 'b_zz']  no             no
  BiotVSSEM2DDeformed         quad9 2-D, topography  ['H', 'mu', 'C', 'M', 'rho', 'rho_f', 'm', 'b']                     no             no
  ElasticSEM2D                quad4 2-D              ['vp', 'vs', 'rho']                                                 no             no
  ElasticSEM2DDeformed        quad9 2-D, topography  ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticSEM3D                hex8 3-D               ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticSEM3DDeformed        hex27 3-D, topography  ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticVSSEM2DDeformed      quad9 2-D, topography  ['vp', 'vs', 'rho']                                                 no             no
  ElasticVSSEM3D              hex8 3-D               ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticVSSEM3DDeformed      hex27 3-D, topography  ['vp', 'vs', 'rho']                                                 no             no
  ElasticVTISEM2DDeformed     quad9 2-D, topography  ['vp', 'vs', 'rho', 'epsilon', 'delta']                             no             no
```

The unified facades `DAS` (formerly `DASModeler`) and `AcousticAniso` are
also exported from `sweep.equations` but do not appear in this listing —
they wrap / dispatch to the raw equation classes above and are not
`WaveEquation` subclasses themselves.

The two right-most columns distinguish:

- **Torch Binding** — whether the equation's source code declares compiled-extension
  support (`supports_torch_binding()` returns `True`).
- **Binding Ready** — whether `impl="c"` can run right now: PyTorch present, a
  CUDA GPU visible, and a CUDA core at hand — the wheel's prebuilt `lib/cu12/`
  or `lib/cu13/` core for your torch's CUDA major, `SWEEP_CORE`, a cached local
  build, or an nvcc of torch's CUDA major that can build one. Nothing is loaded
  or compiled to answer. A `yes` / `no` mismatch means no GPU is visible, or no
  core fits this process and none can be built (a torch built for a CUDA major
  with no shipped core, a card older than the shipped archs and PTX):
  `sweep.backend.torch.binding.diagnostics()["reason"]` says why, and
  `["shipped_core"]` which core, if any, fits
  (see [Building the CUDA core](../dev/building.md)).

On an SEM row, **Mesh** is the mesh family the class builds on (`quad4` and
`hex8` are flat, `quad9` and `hex27` curved) with `topography` when the class
takes one, and **Models** is the `models=` list `PropSEM` takes: on
`AcousticPotentialSEM2D` the density is the constant `density=`, not a model.
**Torch Binding** says that `PropSEM(impl='c')` has CUDA kernels for the
class, and **Binding Ready** is the same check as on the FD rows.

## `sweep show <Equation>`

Prints the wavefields, required model order, and compiled-binding status for a
single equation class:

```bash
sweep show Acoustic
```

```text
=== Acoustic ===
  Wavefields: ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']
  Needed models: ['vp']
  Torch binding support: yes
  Torch binding available: yes
```

```bash
sweep show ElasticTTISG
```

```text
=== ElasticTTISG ===
  Wavefields: ['vx', 'vy', 'vz', 'sxx', 'szz', 'syz', 'sxz', 'sxy', 'm_vxx', 'm_vxz', 'm_vyx', 'm_vyz', 'm_vzx', 'm_vzz', 'm_txxx', 'm_txzz', 'm_txyx', 'm_tyzz', 'm_txzx', 'm_tzzz']
  Needed models: ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']
  Torch binding support: yes
  Torch binding available: yes
```

An SEM class prints its mesh and C-PML, its fields, the fields a source drives,
the receiver types (the receiver fields and the components derived from them,
such as `p` and `vx`) and the `models=` that `PropSEM` takes:

```bash
sweep show AcousticPotentialSEM2D
```

```text
=== AcousticPotentialSEM2D ===
  Spectral element: sweep.sem, run with PropSEM
  Mesh: quad9, 2-D, topography: yes
  C-PML: yes
  Wavefields: ['chi', 'chi_prev', 'psi_x', 'psi_z', 'zeta_x', 'zeta_z']
  Source fields: ['chi']
  Receiver types: ['chi', 'p']
  Needed models: ['vp']
  Torch binding support: yes
  Torch binding available: yes
```

On the elastic SEM classes `source_type` also takes `PropTorch`'s names: `vx`,
`vz` (and `vy`) for a force and, in 2-D, `sxx`, `szz`, `sxz` for a moment
tensor (see [SEM · Source types](sem.md#source-types)). The Biot classes
name the `source_type` values `PropSEM` takes for them (a force, a moment, a
pore-pressure or flux injection), so `show` prints those instead of their
source fields:

```text
  Source types: ['vx', 'vz', 'qx', 'qz', 'v_x', 'v_z', 'q_x', 'q_z', 'fx', 'fz', 'p', 'sxx', 'szz', 'sxz']
```

A class without an absorbing layer says so, as `ElasticSEM2D` does:

```text
  C-PML: no -- a reflecting box, abcn=0 only
```

### Unknown names, and the exit code

`show` resolves the name through the **equation registries**, the
finite-difference one and then the SEM one, not the `sweep.equations`
namespace, so only real equations answer. An unknown name
exits **1** and offers the closest registered spellings:

```bash
sweep show Acoustic3d ; echo "exit=$?"
```

```text
No such wave equation: Acoustic3d
  Did you mean: Acoustic3D, Acoustic, AcousticVRZ3D?
  `sweep list equations` names all 43 finite-difference and 16 spectral-element equations.
exit=1
```

`DAS` and `AcousticAniso` are facades: they pick a raw equation class from
their constructor arguments, so they have no wavefields of their own. `show`
says so, and also exits 1 -- there is nothing to introspect.

A successful `show` exits 0, so `sweep show <Eq> >/dev/null` is a usable
"is this equation available here" probe in a script.

The `Wavefields` list is the **full** internal state — it includes CPML memory
variables and other auxiliary fields. The user-facing source / receiver field
choices are a subset; use the equation's `available_fields(role="source")` or
`available_fields(role="receiver")` Python helper for that. For an SEM class
`show` prints those two subsets itself.

## Python-side equivalents

The CLI is a thin wrapper around the introspection helpers in
`sweep.equations`. They are accessible directly from Python too:

```python
from sweep.equations import (
    equation_classes,
    supports_torch_binding,
    torch_binding_supported_equations,
    Acoustic,
)

print(sorted(equation_classes().keys()))
print(torch_binding_supported_equations())
print(supports_torch_binding("ElasticTTISG"))      # True

eq = Acoustic(spatial_order=8, backend="torch")
print([f.name for f in eq.available_fields()])
print([f.name for f in eq.available_fields(role="source")])
print(eq.describe_field("h1"))
print([m.name for m in eq.available_models()])
print(eq.describe_model("vp"))
```

The SEM registry has its own helpers in `sweep.sem.equations`, and
`propsem_models` gives the `models=` list `PropSEM` takes:

```python
import sweep.sem.equations as sem
from sweep.sem.prop_base import propsem_models

print(sorted(sem.equation_classes()))
cls = sem.equation_classes()["AcousticPotentialSEM2D"]
print(sem.summary(cls))       # mesh, ndim, topography, cpml, fields, sources, receivers, derived
print(propsem_models(cls))    # ['vp']
print(sem.describe())         # one line per class
```

## `sweep datasets`

Lists, describes, pre-downloads and locates the benchmark velocity models
(`list`, `info`, `download`, `where`); see [Datasets · CLI](datasets.md#cli).

```bash
sweep datasets list
```
