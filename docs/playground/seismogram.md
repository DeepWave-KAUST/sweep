# Seismogram & free surface

Fire a source and watch the wavefield (top) while a shot gather builds up at the
surface receivers (bottom). The gather shows the direct wave, the reflection
hyperbola from the interface, and — with the free surface on — the surface
multiple.

Toggle **Free surface** and re-fire: the extra multiple event appears or
vanishes. This is why marine field data must be modeled with a free surface —
the recorded data already contains ghosts and surface multiples.

<iframe src="../assets/seismogram.html" width="100%" height="950"
        style="border:1px solid var(--md-default-fg-color--lightest); border-radius:10px"
        loading="lazy" title="Seismogram and free surface"></iframe>

The free surface is a pressure-release condition (p = 0) at the top boundary;
the other three sides are absorbing.
