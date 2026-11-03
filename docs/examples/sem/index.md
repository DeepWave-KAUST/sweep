# SEM

The spectral-element solver, `sweep.sem.PropSEM`: `PropTorch`'s call on a
Quad9 mesh that follows the free surface. Start with *Hello · SEM* for one
forward and one gradient, then the eight short *SEM basics* notebooks, each on
one part of the API, from the mesh to 3-D and to anisotropic and porous rocks. The examples after them compare
topography against the finite-difference solvers and run FWI on Marmousi, one
model and every shot's window in one batched call. The
[SEM user guide](../../user-guide/sem.md) has the rest of the API.

## Basics

<div class="grid cards" markdown>

-   ![Hello SEM](../../figures/gallery/43_sem_forward_backward.png){ loading=lazy }

    **Hello · SEM · forward and backward**

    ---

    `PropSEM` on a two-layer model: one shot gather, then one `.backward()`
    for the vp gradient on the grid, with the full tape and with
    `memory=BoundarySaving()`.

    [:material-notebook-outline: Open notebook](../../notebooks/43_sem_forward_backward.ipynb)

-   ![SEM mesh and time step](../../figures/gallery/50_sem_mesh_and_time_step.png){ loading=lazy }

    **Basics 1 · The mesh and the time step**

    ---

    The tiling rules for `shape` and `abcn`, `suggest_discretization` for `dh`
    and `dt`, `solver.mesh` with its elements and GLL nodes, a mesh that follows
    a topography, and what a too-large `dt` does.

    [:material-notebook-outline: Open notebook](../../notebooks/50_sem_mesh_and_time_step.ipynb)

-   ![SEM equations](../../figures/gallery/51_sem_equations_and_models.png){ loading=lazy }

    **Basics 2 · Choosing an equation**

    ---

    The equation table, what each equation takes as `models=` and which run on
    `impl='c'`, and four 2-D equations on one model, record against record.

    [:material-notebook-outline: Open notebook](../../notebooks/51_sem_equations_and_models.ipynb)

-   ![SEM sources and receivers](../../figures/gallery/52_sem_sources_and_receivers.png){ loading=lazy }

    **Basics 3 · Sources, receivers and records**

    ---

    Grid-index positions, force, explosion and shear sources with their
    radiation, derived receiver components, and several shots with their own
    wavelets in one call.

    [:material-notebook-outline: Open notebook](../../notebooks/52_sem_sources_and_receivers.ipynb)

-   ![SEM boundaries](../../figures/gallery/53_sem_boundaries_and_snapshots.png){ loading=lazy }

    **Basics 4 · Boundaries and snapshots**

    ---

    `abcn` from none to the default, `free_surface` by edge, an elastic surface
    over a hill, wavefield snapshots, and the one refused C-PML layout.

    [:material-notebook-outline: Open notebook](../../notebooks/53_sem_boundaries_and_snapshots.ipynb)

-   ![SEM gradients](../../figures/gallery/54_sem_gradients_and_memory.png){ loading=lazy }

    **Basics 5 · Gradients and memory**

    ---

    vp, vs and rho gradients from one multi-shot call, then `Full()`,
    `BoundarySaving()` and `Ckpt()` on both backends: time, peak memory and the
    gradient, and the smaller rim-mode boundary saving.

    [:material-notebook-outline: Open notebook](../../notebooks/54_sem_gradients_and_memory.ipynb)

-   ![SEM fluid-solid](../../figures/gallery/55_sem_fluid_solid.png){ loading=lazy }

    **Basics 6 · Water over an elastic sea floor**

    ---

    `AcousticElasticSEM2D`: the model on the GLL nodes with the sea floor on
    element edges, hydrophones and ocean-bottom nodes in one call.

    [:material-notebook-outline: Open notebook](../../notebooks/55_sem_fluid_solid.ipynb)

-   ![SEM 3-D](../../figures/gallery/56_sem_3d.png){ loading=lazy }

    **Basics 7 · A 3-D model**

    ---

    Acoustic and elastic 3-D runs on `impl='c'`, a 3-D topography and one 3-D
    gradient: the same call with one more axis.

    [:material-notebook-outline: Open notebook](../../notebooks/56_sem_3d.ipynb)

-   ![SEM VTI and Biot](../../figures/gallery/57_sem_vti_and_biot.png){ loading=lazy }

    **Basics 8 · Anisotropic and poroelastic rocks**

    ---

    VTI with the Thomsen parameters, and Biot's porous rock with its slow P
    wave, anisotropic permeability and `equation_options`.

    [:material-notebook-outline: Open notebook](../../notebooks/57_sem_vti_and_biot.ipynb)

</div>

## Examples

<div class="grid cards" markdown>

-   ![FD vs SEM topography](../../figures/gallery/41_fd_vs_sem_topography.png){ loading=lazy }

    **Wavefield · FD vs SEM on topography**

    ---

    The same Gaussian hill two ways: FD on the grid (the image method
    acoustic, the boundary-fitted curvilinear grid elastic) and SEM on a
    Quad9 mesh deformed to the surface, snapshots and gathers side by side.

    [:material-notebook-outline: Open notebook](../../notebooks/41_fd_vs_sem_topography.ipynb)

-   ![FWI Acoustic Marmousi SEM](../../figures/gallery/42_fwi_acoustic_marmousi_sem.png){ loading=lazy }

    **FWI · Acoustic · Marmousi · SEM**

    ---

    The Marmousi workflow on the spectral-element solver: the grid is
    interpolated onto the GLL nodes and the gradient comes back on the grid.
    `impl='eager'` or `'c'`.

    [:material-notebook-outline: Open notebook](../../notebooks/42_fwi_acoustic_marmousi_sem.ipynb)

-   ![FWI batched local windows SEM](../../figures/gallery/44_sem_batched_local_window_fwi.png){ loading=lazy }

    **FWI · Batched per-shot local windows · SEM**

    ---

    The SEM twin of the streamer example: every shot's window in one batched
    `PropSEM` call (`impl='c'`, `memory=BoundarySaving()`), frequency continuation
    from a smooth start.

    [:material-notebook-outline: Open notebook](../../notebooks/44_sem_batched_local_window_fwi.ipynb)

</div>
