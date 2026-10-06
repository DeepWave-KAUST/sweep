# Examples

> :material-github: **All examples on GitHub** &mdash; [`examples/`](https://github.com/DeepWave-KAUST/sweep/tree/dev/examples) (clone, run, modify)

Runnable example scripts and notebooks live in the `examples/` directory of
the repository. The **notebooks** under `docs/notebooks/` are cell-by-cell
tutorials and the easiest entry point; they are grouped by topic below. Scripts
for workflows that don't fit a notebook (e.g. multi-process multi-GPU FWI) are
linked at the bottom of this page.

## Notebooks

!!! tip "New here? Start with Hello · SWEEP"

    The smallest end-to-end SWEEP story in one notebook: parameters → model →
    `Acoustic()` + `PropTorch()` → one shot gather → one `.backward()` for a
    vp gradient → 5-line Adam loop. No external data.
    [:material-notebook-outline: Open notebook](../notebooks/00_hello_fwi.ipynb)

<div class="grid cards" markdown>

-   [![FWI](../figures/gallery/01_fwi_acoustic_marmousi.png){ loading=lazy }](fwi/index.md)

    **FWI**

    ---

    Acoustic, variable-density and elastic Marmousi, multiscale and
    frequency-selection encoding, batched streamer windows, 3-D Overthrust.

    [:octicons-arrow-right-24: 7 notebooks](fwi/index.md)

-   [![NN · FWI](../figures/gallery/13_ifwi_siren.png){ loading=lazy }](nn-fwi/index.md)

    **NN · FWI**

    ---

    A neural network produces the velocity model; its weights are inverted
    through the propagator.

    [:octicons-arrow-right-24: 1 notebook](nn-fwi/index.md)

-   [![Wavefields](../figures/gallery/30_wavefield_visco_elastic.png){ loading=lazy }](wavefields/index.md)

    **Wavefields**

    ---

    Elastic sources, DAS, topography, per-edge free surfaces, attenuation,
    and solver settings.

    [:octicons-arrow-right-24: 8 notebooks](wavefields/index.md)

-   [![Anisotropic](../figures/gallery/27_wavefield_elastic_tti_3d.png){ loading=lazy }](anisotropic/index.md)

    **Anisotropic**

    ---

    Pseudo-acoustic VTI, rotated staggered-grid TTI in 2-D and 3-D, and the
    displacement formulation.

    [:octicons-arrow-right-24: 4 notebooks](anisotropic/index.md)

-   [![Imaging](../figures/gallery/08_rtm_acoustic_marmousi.png){ loading=lazy }](imaging/index.md)

    **Imaging**

    ---

    RTM, custom imaging conditions, the reflection-FWI gradient, and
    angle-domain gathers.

    [:octicons-arrow-right-24: 5 notebooks](imaging/index.md)

-   [![HPC](../figures/gallery/25_domain_decomposition.png){ loading=lazy }](hpc/index.md)

    **HPC**

    ---

    Memory strategies, boundary compression, multi-GPU data parallel, and
    domain decomposition.

    [:octicons-arrow-right-24: 5 notebooks](hpc/index.md)

-   [![Extending](../figures/gallery/18_extending_add_new_equation.png){ loading=lazy }](extending/index.md)

    **Extending**

    ---

    Add a wave equation of your own and run it like a built-in.

    [:octicons-arrow-right-24: 1 notebook](extending/index.md)

-   [![Sensitivity & resolution](../figures/gallery/21_radiation_elastic.png){ loading=lazy }](sensitivity/index.md)

    **Sensitivity & resolution**

    ---

    Radiation patterns of the scattering sources behind multiparameter trade-off.

    [:octicons-arrow-right-24: 2 notebooks](sensitivity/index.md)

</div>

## Scripts

For workflows that don't fit a single notebook — e.g. multi-process
multi-GPU FWI — see:

- [**Multi-GPU DDP** (`fwi_marmousi_dist.py`)](multi_gpu_dist.md) — Torch
  `torchrun` driver that scales one-shot-per-rank across multiple GPUs and
  syncs gradients with `torch.distributed`.

- [**Model-parallel FWI** (`dd_fwi_marmousi_2d.py`, `dd_fwi_marmousi_elastic_2d.py`,
  `dd_fwi_overthrust_update.py`)](../user-guide/parallel.md) — the other axis:
  one model split across GPUs instead of one shot per GPU. Acoustic and elastic
  2-D on Marmousi, plus a 3-D Overthrust model update. The two Marmousi
  scripts take `--check <tag>` to compare against an earlier run in `--outdir`
  (e.g. the same problem undivided, `--px 1`); the Overthrust script has no
  comparison mode.

Browse [`examples/`](https://github.com/DeepWave-KAUST/sweep/tree/dev/examples)
on GitHub for the full collection of runnable scripts.
