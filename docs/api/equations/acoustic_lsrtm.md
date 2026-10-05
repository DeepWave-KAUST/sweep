# AcousticLSRTM

::: sweep.equations.AcousticLSRTM

## Term III on its own

Wu & Alkhalifah (2015, eq. 18-20) weight the image-point term III of the RWI
velocity gradient separately: `(II+IV) + beta*III`. Set
`SWEEP_LSRTM_SPLIT_III=1` and the `impl='c'` full or boundary-saving backward
of `AcousticLSRTM` / `AcousticLSRTM3D` puts II+IV in `vp.grad` and keeps III
aside, on the same grid:

```python
import os
from sweep.propagator import last_grad_split_iii

os.environ["SWEEP_LSRTM_SPLIT_III"] = "1"   # read by every backward
loss.backward()
grad_vp = vp.grad + beta * last_grad_split_iii()
```

Without the switch `vp.grad` holds II+III+IV. The checkpointing modes return
no vp gradient, split or not. `last_grad_split_iii()` returns `None` until a
split backward has run, and afterwards the III of the most recent one.
[Notebook 32](../../notebooks/32_rwi_acoustic_vs_lsrtm_gradient.ipynb) shows
the split on the paper's Figure 1 setup.

::: sweep.propagator.last_grad_split_iii
    options:
      heading_level: 3
