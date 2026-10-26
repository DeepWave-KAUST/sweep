# SEMPropagator

The time loop under [`PropSEM`](prop_sem.md). It drives any SEM equation from
its `field_specs` and takes a built equation, source and receiver stamps and a
wavelet; `PropSEM` assembles those from the FD-shaped call. Use it for what
`PropSEM` does not cover, such as an equation whose `build` takes keywords
`PropSEM` does not supply, or stamps placed by hand.

::: sweep.sem.SEMPropagator

## Point and moment stamps

A stamp holds the Lagrange weights of one point, and their derivatives, in the
element that contains it: a source is spread over that element's GLL nodes
with them, and a receiver reads the field through them. A moment-tensor source
differentiates the basis, which jumps across element edges, so a source on an
edge is stamped in every element that meets there (`EdgeStamp`).

::: sweep.sem.build_point_stamp

::: sweep.sem.PointStamp

::: sweep.sem.PointStamp3D

::: sweep.sem.build_moment_stamp

::: sweep.sem.EdgeStamp

## Boundary saving

The eager boundary-saved rollout; `PropSEM(memory=BoundarySaving())` is the
usual way to ask for it. See
[Boundary saving](../../user-guide/sem.md#boundary-saving).

::: sweep.sem.run_boundary_saved

## Anisotropy

::: sweep.sem.thomsen_to_vti_cij
