# Numerical dispersion

Push the source **Frequency** up (or the **Velocity** down) to lower the
points-per-wavelength (PPW), and watch numerical dispersion ripple behind the
wavefront and stretch the recorded trace into a long tail. The **PPW** badge
turns amber below ~8 and red below ~5.

Then switch between **2nd** and **4th** order at the same settings: the
higher-order stencil suppresses most of the dispersion, which is why production
solvers use wide stencils.

<iframe src="../assets/dispersion.html" width="100%" height="760"
        style="border:1px solid var(--md-default-fg-color--lightest); border-radius:10px"
        loading="lazy" title="Numerical dispersion and points-per-wavelength"></iframe>

The dashed line on the trace marks the exact arrival time (offset ÷ velocity);
a well-sampled pulse sits on it with little trailing energy.
