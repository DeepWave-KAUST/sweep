# SEM domain decomposition

`sweep.sem.dd` splits one model into x strips across GPUs on the `impl='c'`
backend, the SEM mirror of the FD [`ModelParallel`](../parallel.md): a built
`PropSEM` carries the configuration, global coordinates go in and
`gather_record` assembles the record. Usage and scope are in
[Domain decomposition](../../user-guide/sem.md#domain-decomposition-multi-gpu-phase-1).

```python
from sweep.sem.dd import SEMModelParallel, StripTopology

ddp = SEMModelParallel(prop, StripTopology(px=world, rank=rank))
rec = ddp(wavelet, src_node, rec_nodes, [vp])
full = ddp.gather_record(rec)
```

::: sweep.sem.dd.SEMModelParallel

::: sweep.sem.dd.StripTopology
