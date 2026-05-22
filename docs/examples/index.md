# Examples

> :material-github: **All examples on GitHub** &mdash; [`examples/`](https://github.com/DeepWave-KAUST/sweep/tree/dev/examples) (clone, run, modify)

Runnable example scripts and notebooks live in the `examples/` directory of
the repository. The **notebooks** under `examples/notebooks/` (cards below)
are cell-by-cell tutorials and the easiest entry point; for a flat overview
of all of them see the [home page gallery](../index.md). Scripts for
workflows that don't fit a notebook (e.g. multi-process multi-GPU FWI) are
linked at the bottom of this page.

## Notebooks (start here)

<div class="grid cards" markdown>

-   **Hello · SWEEP**

    ---

    Smallest end-to-end SWEEP story in one notebook: parameters → model →
    `Acoustic()` + `PropTorch()` → one shot gather → one `.backward()` for a
    vp gradient → 5-line Adam loop. No external data.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/00_hello_fwi.ipynb)

-   **FWI · Acoustic · Marmousi**

    ---

    Load Marmousi from `sweep.datasets`, build a 192×320 window, forward-model
    observed gathers, and invert the smooth start with Adam + MSE. Each phase
    is one cell.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/01_fwi_acoustic_marmousi.ipynb)

-   **FWI · Elastic · Marmousi**

    ---

    Same skeleton, but the equation is `Elastic` and the model is the
    `(vp, vs, rho)` triplet. `vs` and `rho` are derived from `vp` with Poisson
    + Gardner relations so the example still runs with zero downloads.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/02_fwi_elastic_marmousi.ipynb)

-   **FWI · multiscale**

    ---

    Three-band frequency progression (3 → 6 → 12 Hz) of acoustic FWI on
    Marmousi 25 m. Each band feeds its final model into the next; loss
    drops monotonically across the chain and avoids cycle skipping.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/03_fwi_multiscale.ipynb)

-   **DAS · Zhao vs Mu**

    ---

    Forward-model the same three-layer elastic medium with the two DAS
    formulations and compare the resulting strain-rate gathers side by side.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/04_das_zhao_vs_mu.ipynb)

-   **Wavefield · VTI + shear suppression**

    ---

    Run Duveneck (1st-order), Liang (2nd-order pseudo-acoustic), and
    Alkhalifah (η-acoustic) through the unified `AcousticAniso` factory
    on the canonical Duveneck Fig 2 setup; the trailing cell demonstrates
    the δ→ε disk taper that kills the pseudo-acoustic shear artefact.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/05_wavefield_vti.ipynb)

-   **Wavefield · Elastic**

    ---

    Wavefield snapshots from three different stress-source loadings on a
    uniform elastic medium — explosion, vertical dipole, and pure shear —
    to visualize P/S excitation and radiation patterns.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/06_wavefield_elastic.ipynb)

-   **Memory · strategies**

    ---

    Same forward + backward step run under five memory strategies (eager
    full vs. eager ckpt vs. c boundary-saving / chunk-ckpt / recursive-ckpt)
    with side-by-side peak-memory and wallclock charts.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/07_memory_strategies.ipynb)

-   **RTM · Acoustic · Marmousi**

    ---

    Reverse-time migration on full 12.5 m Marmousi with a 15 Hz Ricker —
    background subtraction, near-offset mask, illumination compensation,
    and `solver.rtm()`. Produces a clean reflectivity image in <3 s on
    30 shots.

    [:material-notebook-outline: Open notebook](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/08_rtm_acoustic_marmousi.ipynb)

</div>



## Scripts

For workflows that don't fit a single notebook — e.g. multi-process
multi-GPU FWI — see:

- [**Multi-GPU DDP** (`fwi_marmousi_dist.py`)](multi_gpu_dist.md) — Torch
  `torchrun` driver that scales one-shot-per-rank across multiple GPUs and
  syncs gradients with `torch.distributed`.

Browse [`examples/`](https://github.com/DeepWave-KAUST/sweep/tree/dev/examples)
on GitHub for the full collection of runnable scripts.
