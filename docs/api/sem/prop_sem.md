# PropSEM

The spectral-element counterpart of [`PropTorch`](../propagators/prop_torch.md):
the same constructor options with the same meaning, `models=[...]` input,
`source_type` / `receiver_type` selection and a `(nshots, nt, nrec, ncomp)`
seismogram, with a class from [SEM equations](equations.md) as the equation.
The options, the source types and the receiver components are explained in the
[SEM propagator](../../user-guide/sem.md) guide.

::: sweep.sem.PropSEM

::: sweep.sem.prop_base.propsem_models

## Mesh and time step

`shape` must tile into elements, `(n - 1) % n_order == 0` on every axis, and
the time step must satisfy the CFL condition on the smallest GLL spacing.
`suggest_discretization` gives both from the domain, the velocities and the
source frequency.

::: sweep.sem.suggest_discretization

::: sweep.sem.SEMDiscretization
