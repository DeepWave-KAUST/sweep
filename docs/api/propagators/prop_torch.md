# PropTorch

The PyTorch propagator wrapper that drives any `WaveEquation` from
`sweep.equations` through a time loop. Accepts an array of keyword arguments
inherited from the base class — those are the ones the user mostly tunes.

::: sweep.propagator.torch.PropTorch

## Base class (full keyword reference)

`PropTorch` forwards every shared keyword argument to `PropBase`. The complete
list of solver knobs (`shape`, `dh`, `dt`, `abcn`, `pml_type`, `free_surface`,
`use_ckpt`, checkpointing options, boundary-saving options, …) is documented
on the base class.

::: sweep.propagator.base.PropBase
    options:
      members: ["__init__"]
      heading_level: 3
