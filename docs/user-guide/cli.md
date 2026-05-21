# CLI

SWEEP ships a small command-line interface for inspecting the available
equations and their compiled-binding status. The CLI is intentionally narrow —
it answers introspection questions about the installed engine, not full FWI /
LSRTM task runs.

Implementation: `src/sweep/cli.py`.

## `sweep list equations`

Lists every equation class exported by `sweep.equations` together with its
required model parameters, whether the compiled PyTorch extension is supported,
and whether that extension is actually loadable in the current environment.

```bash
sweep list equations
```

Example output on a machine with the compiled `sweep._C` extension installed:

```text
Available equations:

  Equation               Models                                                              Torch Binding  Binding Ready
  ---------------------  ------------------------------------------------------------------  -------------  -------------
  Acoustic               ['vp']                                                              yes            yes
  Acoustic1st            ['vp', 'rho']                                                       no             no
  Acoustic3D             ['vp']                                                              yes            yes
  AcousticLSRTM          ['vp', 'mp']                                                        yes            yes
  AcousticLSRTM3D        ['vp', 'mp']                                                        yes            yes
  AcousticTTI            ['vp', 'epsilon', 'delta', 'theta']                                 no             no
  AcousticTTIAlkhalifah  ['vv', 'v', 'eta']                                                  no             no
  AcousticTTILiang       ['vp', 'epsilon', 'delta', 'theta']                                 no             no
  AcousticTariq          ['vv', 'v', 'eta']                                                  no             no
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
  ElasticTTI             ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  no             no
  ElasticTTISG           ['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']  yes            yes
```

The unified facades `DAS` (formerly `DASModeler`) and `AcousticAniso` are
also exported from `sweep.equations` but do not appear in this listing —
they wrap / dispatch to the raw equation classes above and are not
`WaveEquation` subclasses themselves.

The two right-most columns distinguish:

- **Torch Binding** — whether the equation's source code declares compiled-extension
  support (`supports_torch_binding()` returns `True`).
- **Binding Ready** — whether the compiled binding can actually be loaded right
  now (PyTorch present, CUDA visible, `sweep._C` importable). A `yes` /
  `no` mismatch usually means you have not built the CUDA extra (see
  [Installation](../getting-started/installation.md)).

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

The `Wavefields` list is the **full** internal state — it includes CPML memory
variables and other auxiliary fields. The user-facing source / receiver field
choices are a subset; use the equation's `available_fields(role="source")` or
`available_fields(role="receiver")` Python helper for that.

## Python-side equivalents

The CLI is a thin wrapper around the introspection helpers in
`sweep.equations`. They are accessible directly from Python too:

```python
from sweep.equations import (
    _equation_classes,
    supports_torch_binding,
    torch_binding_supported_equations,
    Acoustic,
)

print(sorted(_equation_classes().keys()))
print(torch_binding_supported_equations())
print(supports_torch_binding("ElasticTTISG"))      # True

eq = Acoustic(spatial_order=8, backend="torch")
print([f.name for f in eq.available_fields()])
print([f.name for f in eq.available_fields(role="source")])
print(eq.describe_field("h1"))
print([m.name for m in eq.available_models()])
print(eq.describe_model("vp"))
```
