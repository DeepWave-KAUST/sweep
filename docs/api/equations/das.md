# DAS family

Distributed Acoustic Sensing equations — strain / strain-rate forward operators
for fiber-optic cable measurements. There are two formulations, each in 2D and
3D: Zhao (`DASZhao` / `DASZhao3D`, stress / normal-strain-rate) and Mu
(`DASMu` / `DASMu3D`, velocity-stress plus integrated strain). The `DAS`
facade runs either one (`method="zhao"` or `method="mu"`; 2D or 3D from
`shape`). `DASElastic` / `DASElastic3D` are aliases of `DASZhao` /
`DASZhao3D`, and `DASModeler` is the former name of `DAS`.

## Facade

::: sweep.equations.DAS

## 2-D

::: sweep.equations.DASZhao

::: sweep.equations.DASMu

## 3-D

::: sweep.equations.DASZhao3D

::: sweep.equations.DASMu3D
