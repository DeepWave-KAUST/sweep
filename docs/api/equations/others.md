# Other equations

Exported equation classes without a page of their own; see the
[equation index](index.md) for the families. Compiled `impl='c'` kernels exist
for `ElasticVRR`, `AcousticVTI1st` and `AcousticVTI1st3D`; every other class
on this page runs on the eager backend only.

## Curvilinear-grid topography

`AcousticCurvilinear` and `ElasticCurvilinear` are **eager-only**: they have no
CUDA kernel, so `impl='auto'` runs them eager and an explicit `impl='c'` falls
back to eager with a `UserWarning`.

::: sweep.equations.AcousticCurvilinear

::: sweep.equations.ElasticCurvilinear

## Vector reflectivity (VRR)

::: sweep.equations.AcousticVRR

::: sweep.equations.ElasticVRR

## Anisotropic acoustic family

`AcousticAniso` is a factory: it returns an instance of one of the classes
below. The author-named aliases (`AcousticVTILiang`, `AcousticTTILiang`,
`AcousticVTIAlkhalifah`, `AcousticTTIAlkhalifah`, `AcousticVTIDuveneck`,
`AcousticVTIDuveneck3D`) and the `AcousticVTIDefault` / `AcousticVTIDefault3D`
routing names are these same classes.

::: sweep.equations.AcousticAniso

::: sweep.equations.AcousticVTI

::: sweep.equations.AcousticTTI

::: sweep.equations.AcousticTariq

::: sweep.equations.AcousticVTI1st

::: sweep.equations.AcousticVTI1st3D
