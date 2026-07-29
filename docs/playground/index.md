# Playground

Interactive, browser-only demos of the physics behind SWEEP — draw a model,
fire a source, and watch waves propagate. Everything runs client-side in your
browser; there is no server and nothing to install.

<div class="grid cards" markdown>

-   :material-waves:{ .lg .middle } __Wave modeling__

    ---

    Paint a velocity model and fire a source. Swap the equation live: acoustic
    (4th order), visco-acoustic, or elastic P-SV.

    [:octicons-arrow-right-24: Open](modeling.md)

-   :material-pulse:{ .lg .middle } __Seismogram & free surface__

    ---

    Record a live shot gather at surface receivers. Toggle the free surface to
    watch surface multiples appear.

    [:octicons-arrow-right-24: Open](seismogram.md)

-   :material-chart-bell-curve:{ .lg .middle } __Numerical dispersion__

    ---

    Drive points-per-wavelength down and watch grid dispersion appear; compare
    2nd- vs 4th-order stencils.

    [:octicons-arrow-right-24: Open](dispersion.md)

-   :material-ellipse-outline:{ .lg .middle } __Anisotropic wavefront__

    ---

    Deform a qP wavefront with Thomsen ε, δ and a tilt angle — from a circle to
    elliptical and anelliptic (VTI / TTI).

    [:octicons-arrow-right-24: Open](anisotropy.md)

</div>

!!! note "How these work"

    Each demo is a small, self-contained finite-difference solver written in
    JavaScript — the same wave equations SWEEP solves, shrunk to a size that runs
    in real time in a browser tab. They build intuition; they are not the
    production solver.
