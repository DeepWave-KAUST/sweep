# CLI

SWEEP ships a small command-line interface for inspecting the available
equations and their compiled-binding status, plus `sweep datasets` for the
benchmark velocity models. The CLI is intentionally narrow — it answers
introspection questions about the installed engine, not full FWI / LSRTM task
runs.

Implementation: `src/sweep/cli.py`.

## `sweep list equations`

Lists every equation class exported by `sweep.equations` together with its
required model parameters, whether it declares a compiled CUDA binding, and
whether the CUDA core can run here right now.

```bash
sweep list equations
```

Example output on a machine with a CUDA GPU where the shipped core fits:

```text
Available equations:

  Equation               Models                                                              Torch Binding  Binding Ready
  ---------------------  ------------------------------------------------------------------  -------------  -------------
  Acoustic               ['vp']                                                              yes            yes
  Acoustic1st            ['vp', 'rho']                                                       no             no
  Acoustic3D             ['vp']                                                              yes            yes
  AcousticCurvilinear    ['vp']                                                              no             no
  AcousticLSRTM          ['vp', 'mp']                                                        yes            yes
  AcousticLSRTM3D        ['vp', 'mp']                                                        yes            yes
  AcousticTTI            ['vp', 'epsilon', 'delta', 'theta']                                 no             no
  AcousticTTIAlkhalifah  ['vv', 'v', 'eta']                                                  no             no
  AcousticTTILiang       ['vp', 'epsilon', 'delta', 'theta']                                 no             no
  AcousticTariq          ['vv', 'v', 'eta']                                                  no             no
  AcousticVRR            ['vp', 'rx', 'rz']                                                  no             no
  AcousticVRZ            ['vp', 'z']                                                         yes            yes
  AcousticVRZ3D          ['vp', 'z']                                                         yes            yes
  AcousticVTI            ['vp', 'epsilon', 'delta']                                          no             no
  AcousticVTI1st         ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTI1st3D       ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIAlkhalifah  ['vv', 'v', 'eta']                                                  no             no
  AcousticVTIDefault     ['vp', 'epsilon', 'delta']                                          no             no
  AcousticVTIDefault3D   ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIDuveneck    ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTIDuveneck3D  ['vp', 'epsilon', 'delta', 'rho']                                   yes            yes
  AcousticVTILiang       ['vp', 'epsilon', 'delta']                                          no             no
  DASElastic             ['vp', 'vs', 'rho']                                                 yes            yes
  DASElastic3D           ['vp', 'vs', 'rho']                                                 yes            yes
  DASMu                  ['vp', 'vs', 'rho']                                                 yes            yes
  DASMu3D                ['vp', 'vs', 'rho']                                                 yes            yes
  DASZhao                ['vp', 'vs', 'rho']                                                 yes            yes
  DASZhao3D              ['vp', 'vs', 'rho']                                                 yes            yes
  Elastic                ['vp', 'vs', 'rho']                                                 yes            yes
  Elastic3D              ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticAPM             ['vp', 'vs', 'rho']                                                 yes            yes
  ElasticCurvilinear     ['vp', 'vs', 'rho']                                                 no             no
  ElasticTTI             ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  no             no
  ElasticTTI2nd          ['vh', 'vs', 'rho', 'epsilon', 'eta', 'theta']                      yes            yes
  ElasticTTISG           ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  yes            yes
  ElasticTTISG3D         ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  yes            yes
  ElasticVRR             ['vp', 'vs', 'Rp_x', 'Rp_z', 'Rs_x', 'Rs_z']                        yes            yes
  ViscoAcoustic          ['vp', 'Q', 'omega']                                                yes            yes
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

### Unknown names, and the exit code

`show` resolves the name through the **equation registry**, not the
`sweep.equations` namespace, so only real equations answer. An unknown name
exits **1** and offers the closest registered spellings:

```bash
sweep show Acoustic3d ; echo "exit=$?"
```

```text
No such wave equation: Acoustic3d
  Did you mean: Acoustic3D, Acoustic, AcousticVRZ3D?
  `sweep list equations` names all 38 registered equations.
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
`available_fields(role="receiver")` Python helper for that.

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

## `sweep datasets`

Lists, describes, pre-downloads and locates the benchmark velocity models
(`list`, `info`, `download`, `where`); see [Datasets · CLI](datasets.md#cli).

```bash
sweep datasets list
```
