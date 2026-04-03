# Equations

This page should summarize the available equation classes and their required model parameters.

## Suggested Structure

For each equation, document:

- Physical system
- Required `models`
- Exposed `wavefields`
- Supported dimensions
- Supported backends
- Whether torch binding acceleration exists

## Example Table

| Equation | Models | Notes | PyTorch Binding |
| --- | --- | --- | --- |
| `Acoustic/Acoustic3D` | `['vp']` | Second-order acoustic wave equation | ✅ |
| `Acoustic1st` | `['vp', 'rho']` | First-order acoustic wave equation | ❌ |
| `Elastic/Elastic3D` | `['vp', 'vs', 'rho']` | 2D Elastic wave propagation (Velocity-Stress) | ✅ |

## TODO

- Add one subsection per equation
- Link to implementation files where useful
- Clarify which equations are production-ready
