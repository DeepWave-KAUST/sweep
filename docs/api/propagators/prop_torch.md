# PropTorch

The PyTorch propagator wrapper that drives any `WaveEquation` from
`sweep.equations` through a time loop. It builds a backend (the eager or the
compiled `impl='c'` propagator, both `PropBase` subclasses) and forwards the
shared keyword arguments to it — those are the ones the user mostly tunes.

::: sweep.propagator.torch.PropTorch

## Shared keywords (`PropBase`)

`PropTorch` is not itself a `PropBase` subclass: it forwards every shared
keyword argument (`shape`, `dh`, `dt`, `abcn`, `pml_type`, `free_surface`,
`use_ckpt`, checkpointing options, …) to a `PropBase`-derived backend, whose
constructor is documented below. `memory=`, `impl=`, `backend=` and the
`backend_options=` / `eager_options=` / `cuda_options=` blocks are
`PropTorch`'s own (see above).

::: sweep.propagator.base.PropBase
    options:
      members: ["__init__"]
      heading_level: 3
