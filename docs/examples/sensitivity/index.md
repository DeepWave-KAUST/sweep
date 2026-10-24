# Sensitivity & resolution

Radiation patterns of the scattering virtual sources and finite-frequency
sensitivity kernels: the angular and spatial sensitivity behind multiparameter
trade-off.

<div class="grid cards" markdown>

-   ![Acoustic radiation pattern](../../figures/gallery/20_radiation_acoustic.png){ loading=lazy }

    **Sensitivity · Acoustic radiation**

    ---

    Scattered-wavefield snapshots of the partial-derivative virtual sources for the
    acoustic `(Vp, ρ)` / `(Vp, Iₚ)` parameterizations — the angular
    sensitivity behind
    multiparameter trade-off. Reproduces Operto et al. (2013) Fig. 2.

    [:material-notebook-outline: Open notebook](../../notebooks/20_radiation_acoustic.ipynb)

-   ![Elastic radiation pattern](../../figures/gallery/21_radiation_elastic.png){ loading=lazy }

    **Sensitivity · Elastic radiation**

    ---

    Elastic `(Vp, Vs, ρ)` P-P / P-S / S-S radiation patterns: the analytic
    Born kernel
    against sweep's Born-differenced numerics, the δln relative sizes (Vs/Vp
    ≈ 0.6), and
    scattered-wavefield snapshots. Cf. Operto Fig. 8(c,d) / Forgues & Lambaré.

    [:material-notebook-outline: Open notebook](../../notebooks/21_radiation_elastic.ipynb)

-   ![Poroelastic traveltime kernels](../../figures/gallery/38_kernels_poroelastic.png){ loading=lazy }

    **Sensitivity · Poroelastic kernels**

    ---

    Traveltime kernels of the fast P, slow P and S waves in a Biot rock, in the
    primary, density-normalised and wave-speed parameterizations — by
    `torch.autograd` through `Biot`, no adjoint code. Reproduces Morency et al.
    (2009).

    [:material-notebook-outline: Open notebook](../../notebooks/38_kernels_poroelastic.ipynb)

-   ![Poroelastic radiation pattern](../../figures/gallery/39_radiation_poroelastic.png){ loading=lazy }

    **Sensitivity · Poroelastic radiation**

    ---

    Fast-P → fast-P / S radiation patterns of nine rock properties in four
    parameterizations: the first-order Born amplitude of the Biot equations
    against `sweep`'s numerics, and why low-frequency scattering is elastic.
    After Yang et al. (2019).

    [:material-notebook-outline: Open notebook](../../notebooks/39_radiation_poroelastic.ipynb)

</div>
