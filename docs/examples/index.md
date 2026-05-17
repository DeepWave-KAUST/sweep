# Examples

Runnable example scripts live in the `examples/` directory of the repository.
Each card below points to a doc walk-through of the corresponding script.

## 2D Inversion

<div class="grid cards" markdown>

-   ![Marmousi 2D Acoustic FWI](../figures/examples/acoustic_fwi_torch_epoch_0100.png){ loading=lazy }

    **Acoustic FWI · Marmousi**

    ---

    Full-waveform inversion on the Marmousi 2D acoustic model. Torch path
    has eager and compiled C++ / CUDA implementations plus optional MPI
    shot parallelism; JAX path runs through `PropJax`.

    [:octicons-arrow-right-24: Torch](acoustic_fwi_torch.md) ·
    [:octicons-arrow-right-24: JAX](acoustic_fwi_jax.md)

-   ![Marmousi 2D Elastic FWI](../figures/examples/elastic_fwi_marmousi_torch_epoch_0100.png){ loading=lazy }

    **Elastic FWI · Marmousi**

    ---

    PyTorch elastic FWI on Marmousi, inverting Vp and Vs simultaneously.

    [:octicons-arrow-right-24: View details](elastic_fwi_torch_marmousi.md)

-   ![Overthrust 2D Elastic observed data](../figures/examples/elastic_fwi_overthrust_torch_observed_data.png){ loading=lazy }

    **Elastic FWI · Overthrust**

    ---

    Elastic FWI on a slice of the SEG / EAGE Overthrust model.

    [:octicons-arrow-right-24: View details](elastic_fwi_torch_overthrust.md)

-   ![2D Acoustic LSRTM Ricker wavelet](../figures/examples/acoustic_lsrtm_torch_ricker.png){ loading=lazy }

    **2D Acoustic LSRTM**

    ---

    Least-squares reverse-time migration on Marmousi with PyTorch.

    [:octicons-arrow-right-24: View details](acoustic_lsrtm_torch.md)

</div>

## 3D Inversion

<div class="grid cards" markdown>

-   ![SEG/EAGE Overthrust 3D cutaway](../assets/cards/overthrust_3d.png){ loading=lazy }

    **3D Acoustic FWI**

    ---

    Source-encoded 3D acoustic FWI on the SEG/EAGE Overthrust model. The
    cutaway reveals the layered thrust structure the inversion has to
    recover. Torch path adds the compiled C++ / CUDA implementation; JAX
    path runs through `PropJax` with optional `jax.pmap`.

    [:octicons-arrow-right-24: Torch](acoustic_fwi_3d_torch.md) ·
    [:octicons-arrow-right-24: JAX](acoustic_fwi_3d_jax.md)

-   ![3D Acoustic LSRTM init model](../figures/examples/acoustic_fwi_3d_torch_init_model.png){ loading=lazy }

    **3D Acoustic LSRTM**

    ---

    3D extension of the LSRTM example on the Overthrust model with PyTorch.

    [:octicons-arrow-right-24: View details](acoustic_lsrtm_3d_torch.md)

</div>

## Reducing memory

<div class="grid cards" markdown>

-   ![Memory tactics summary](../figures/examples/memory_method_compare_3d_summary.png){ loading=lazy }

    **Overview**

    ---

    Survey of source encoding, boundary saving, and checkpointing in the
    SWEEP propagator.

    [:octicons-arrow-right-24: View details](reducing_memory.md)

-   ![Encoded observed vs synthetic gathers at epoch 100](../figures/examples/acoustic_fwi_encoding_torch_data_epoch_0100.png){ loading=lazy }

    **Source-encoded FWI**

    ---

    Aggregate many shots into a few encoded super-shots and invert with
    PyTorch. The cover compares encoded observed vs synthetic gathers at
    epoch 100.

    [:octicons-arrow-right-24: View details](acoustic_fwi_encoding_torch.md)

-   ![Memory method comparison summary](../figures/examples/memory_method_compare_3d_summary.png){ loading=lazy }

    **Method compare**

    ---

    Benchmark `full` / `boundary` / `disk` / `ckpt` memory modes across
    2D and 3D cases.

    [:octicons-arrow-right-24: View details](memory_method_compare.md)

</div>

## Modeling

<div class="grid cards" markdown>

-   ![Anisotropic wavefield](../figures/examples/wavefields_anisotropic_tti.png){ loading=lazy }

    **Overview**

    ---

    Landing page for the forward-modeling and wavefield-validation script
    family.

    [:octicons-arrow-right-24: View details](wavefields.md)

-   ![Elastic CUDA FD orders](../figures/examples/wavefields_cuda_fd_orders_elastic_vz.png){ loading=lazy }

    **CUDA FD orders**

    ---

    Compare 2nd / 4th / 6th / 8th-order finite-difference stencils in the
    compiled CUDA path.

    [:octicons-arrow-right-24: View details](wavefields_cuda_fd_orders.md)

-   ![Elastic free-surface vz snapshots](../figures/examples/wavefields_free_surface_elastic_vz.png){ loading=lazy }

    **Free surface**

    ---

    Elastic and acoustic free-surface boundary validation runs — Rayleigh
    waves are clearly visible in the elastic vz comparison.

    [:octicons-arrow-right-24: View details](wavefields_free_surface.md)

-   ![Acoustic TTI snapshots](../figures/examples/wavefields_anisotropic_tti.png){ loading=lazy }

    **qP**

    ---

    Acoustic VTI / TTI quasi-P (qP) wavefield comparisons.

    [:octicons-arrow-right-24: View details](wavefields_anisotropic.md)

-   ![Elastic TTI rotation snapshots](../figures/examples/elastic_tti_rotation_vz_snapshots.png){ loading=lazy }

    **Elastic TTI**

    ---

    Rotated staggered-grid TTI (`ElasticTTISG`) snapshots across tilt angles
    and free-surface modes.

    [:octicons-arrow-right-24: View details](elastic_tti_wavefields.md)

-   ![DAS strain-rate gathers](../figures/examples/wavefields_das_figure4_eager.png){ loading=lazy }

    **DAS**

    ---

    Strain-rate gathers from the DAS modelling paper, reproduced with the
    SWEEP DAS family.

    [:octicons-arrow-right-24: View details](wavefields_das.md)

</div>

## Multi-GPU

<div class="grid cards" markdown>

-   ![Multi-GPU FWI loss](../figures/examples/acoustic_fwi_3d_torch_loss.png){ loading=lazy }

    **Overview**

    ---

    Notes on Torch Distributed and JAX `pmap` scaling for FWI workloads.

    [:octicons-arrow-right-24: View details](multi_gpu.md)

-   ![Torch DDP vs JAX pmap iter-time on 2/4 A100](../figures/examples/multi_gpu_pmap_vs_ddp.png){ loading=lazy }

    **Multi-GPU FWI · Marmousi**

    ---

    Distributed FWI on Marmousi compared across three data-parallel modes
    on 2 and 4 A100 GPUs: Torch DDP with the compiled C kernel, Torch DDP
    eager, and JAX `pmap` (XLA).

    [:octicons-arrow-right-24: Torch DDP](multi_gpu_torch.md) ·
    [:octicons-arrow-right-24: JAX pmap](multi_gpu_jax.md)

</div>

## Helpers

<div class="grid cards" markdown>

-   ![Marmousi true / smooth / linear](../figures/examples/marmousi_true_smooth_linear.png){ loading=lazy }

    **Model helper scripts**

    ---

    Download, slice, and visualise the Marmousi and Overthrust models.

    [:octicons-arrow-right-24: View details](model_assets.md)

</div>
