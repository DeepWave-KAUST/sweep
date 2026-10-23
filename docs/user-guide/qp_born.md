# Pure-qP VTI / TTI: forward, adjoint and Born equations

`AcousticVTI` / `AcousticTTI` implement the decoupled pure-qP equation with the
"new acoustic approximation" of Liang, Cao, Sun and Yin (2023, *Geophysics*
88(1), WA335, [doi:10.1190/geo2022-0292.1](https://doi.org/10.1190/geo2022-0292.1)).
`AcousticVTILSRTM` / `AcousticTTILSRTM` are their Born (LSRTM) equations. This
page writes down the three operators the code implements: the forward step, its
adjoint (the FWI gradient) and the Born equation. A worked example, RTM and LSRTM
of a tilted anisotropic bed, is the notebook
[LSRTM · pure-qP VTI / TTI](../notebooks/34_lsrtm_qp_vti_tti.ipynb).

## Forward

Setting the qSV phase velocity to zero in **every** direction (rather than only
along the symmetry axis, as Alkhalifah's acoustic approximation does) gives a
dispersion relation without qSV branch. In the wavenumber domain (paper eqs
12-14)

$$
\frac{\partial^2 P}{\partial t^2}
 = V^2\left[(1+2\varepsilon) + S_k\right]\frac{\partial^2 P}{\partial x^2}
 + V^2\left(1 + S_k\right)\frac{\partial^2 P}{\partial z^2},
\qquad
S_k = \frac{-2(\varepsilon-\delta)\,k_x^2 k_z^2}
           {(1+2\varepsilon)k_x^4 + k_z^4 + 2(1+\delta)k_x^2 k_z^2}.
$$

$S_k$ depends only on the **direction** of the wavenumber, so this operator is
linear in $P$. In TTI media (appendix A, eq. A-7), with $c = \cos\theta$,
$s = \sin\theta$ and $S_k$ evaluated in the rotated frame
$k_x' = k_x c - k_z s$, $k_z' = k_x s + k_z c$:

$$
\frac{\partial^2 P}{\partial t^2}
 = V^2\left[(1+2\varepsilon)c^2 + s^2 + S_k\right] P_{xx}
 + V^2\left[(1+2\varepsilon)s^2 + c^2 + S_k\right] P_{zz}
 - 2\varepsilon V^2 \sin 2\theta\, P_{xz}.
$$

The space-domain scheme (eqs 15-23) replaces the wavenumber direction by the
gradient direction of the wavefield, $\mathbf{n} = \nabla P/|\nabla P|$:

$$
S_n = \frac{-2(\varepsilon-\delta) X^2 Z^2}{(1+2\varepsilon) X^4 + Z^4 + 2(1+\delta) X^2 Z^2 + 10^{-26}},
\qquad (X, Z) = (P_x,\,P_z) \;\;\text{(VTI)},\quad
(X, Z) = (P_x c - P_z s,\; P_x s + P_z c) \;\;\text{(TTI)},
$$

($S_n$ is homogeneous of degree 0, so $\nabla P$ can stand in for $\mathbf n$;
the $10^{-26}$ keeps it finite where $\nabla P = 0$). With the second-order CPML
of `Acoustic` written out per axis, one step of the code is

$$
\begin{aligned}
\mathrm{tmp}_x &= (1+b_x)\,L_x P + b_x'\,D_x P + D_x(a_x\psi_x), &
L^{\rm pml}_x &= (1+b_x)\,\mathrm{tmp}_x + a_x\zeta_x, &
p_x &= (1+b_x)\,D_x P + a_x\psi_x,\\
\psi_x' &= b_x\,D_x P + a_x\psi_x, & \zeta_x' &= b_x\,\mathrm{tmp}_x + a_x\zeta_x, &&
\end{aligned}
$$

(and the same along $z$), $m = D_z(p_x)$ for the tilted mixed derivative, and

$$
P^{n+1} = 2P^n - P^{n-1} + V^2\Delta t^2
\Big[(A + S_n) L^{\rm pml}_x + (C + S_n) L^{\rm pml}_z - E\, m\Big] + f^n,
$$

$A = 1+2\varepsilon,\ C = 1,\ E = 0$ (VTI) or
$A = (1+2\varepsilon)c^2 + s^2,\ C = (1+2\varepsilon)s^2 + c^2,\ E = 2\varepsilon\sin2\theta$
(TTI), with $S_n$ evaluated on the CPML-corrected $(p_x, p_z)$.

**Stability.** The step is explicit, and the paper's appendix B gives its
condition for VTI (eqs. B-8 to B-10):

$$
\Delta t\,V\sqrt{\frac{(1+2\varepsilon) + S_m}{\Delta x^2} + \frac{1+S_m}{\Delta z^2}}
\le \frac{\sqrt2}{\sqrt{\sum_{m=1}^{M} c_{2m}\,[1-(-1)^m]}},
$$

with $S_m$ the $S$ of the Nyquist corner and $c_{2m}$ the second-derivative
coefficients. For TTI the rotation mixes the two axes, so both take the
operator's largest coefficient, $1+2\varepsilon+\max(S,0)$ -- over the
directions $S$ peaks at $(\delta-\varepsilon)/(\sqrt{1+2\varepsilon}+1+\delta)$ --
a bound from the safe side. `PropTorch` evaluates the condition on every cell
before a forward and warns once when `dt` breaks it
(`sweep.equations.qP_vti.qp_stability_ratio`); the run is not stopped.

## Adjoint (FWI gradient) and `s_gradient`

Through $S_n(\nabla P)$ the discrete step is **nonlinear in $P$**. Its exact
derivative is the transpose of the step linearised about the forward state of
the same step. Write $g = V^2\Delta t^2\lambda^{n+1}$ and
$Q_{x,z} = (L^{\rm pml}_x + L^{\rm pml}_z)\,\partial S_n/\partial p_{x,z}$. The adjoint of
the bracket is

$$
\bar L^{\rm pml}_x = g(A+S),\quad \bar L^{\rm pml}_z = g(C+S),\quad
\bar p_x = g\,Q_x + D_z(gE),\quad \bar p_z = g\,Q_z,
$$

and it flows through the CPML exactly as the acoustic adjoint does:

$$
\begin{aligned}
\bar t_x &= (1+b_x)\bar L^{\rm pml}_x + b_x\bar\zeta_x', &
\bar u_x &= b_x'\,\bar t_x + (1+b_x)\bar p_x + b_x\bar\psi_x',\\
\lambda^n &= 2\lambda^{n+1} - \lambda^{n+2}
  + L_x^{\!\top}\!\big[(1+b_x)\bar t_x\big] + D_x^{\!\top}\bar u_x + (z), \\
\bar\psi_x &= a_x\big(\bar p_x + \bar\psi_x' + D_x^{\!\top}\bar t_x\big), &
\bar\zeta_x &= a_x\big(\bar L^{\rm pml}_x + \bar\zeta_x'\big),
\end{aligned}
$$

with $L^{\top} = L$ (symmetric taps) and $D^{\top} = -D$. The model gradients of
step $n$ are $\lambda^{n+1}$ times the derivative of the bracket, e.g. for VTI

$$
\frac{\partial J}{\partial V} \mathrel{+}= \lambda^{n+1}\, 2V\Delta t^2 R,\quad
\frac{\partial J}{\partial \varepsilon} \mathrel{+}= g\big[2L^{\rm pml}_x + S_\varepsilon\,\Sigma\big],\quad
\frac{\partial J}{\partial \delta} \mathrel{+}= g\, S_\delta\,\Sigma,
\qquad \Sigma = L^{\rm pml}_x + L^{\rm pml}_z,
$$

($R$ the bracket; TTI adds the $\theta$ derivatives of $A$, $C$, $E$ and of
$S_n$ through the rotation).

**That exact gradient is ill-conditioned.** $\partial S_n/\partial p \sim 1/|\nabla P|$
and $Q \sim |\Sigma|/|\nabla P|$ does not shrink with the amplitude ($S_n$ is
homogeneous of degree 0), so the adjoint carries O(1) coefficients set by the
direction of weak parts of the wavefield, which fp32 rounding decides. Measured
on the canonical gradient test (48 x 56, $\varepsilon > \delta$): eager fp32 and
eager fp64 differ by 2-3 %, two eager fp32 runs on one GPU do not agree bit for
bit, and a 20 m/s finite-difference check of the fp64 gradient is off by 21 %
(5 % at 5 m/s). With $\varepsilon = \delta$ ($S_n \equiv 0$) the same comparison
agrees to $10^{-7}$. This, not the CPML, is why eager boundary saving was
refused for these equations: any reconstruction round-off moves the gradient
by several percent.

`s_gradient` picks the adjoint (`AcousticVTI`, `AcousticTTI` and the Born pair;
the forward is identical in all three):

| `s_gradient` | adjoint propagation | model gradient | conditioning |
| --- | --- | --- | --- |
| `'adjoint'` (default) | the qP operator with $S_n$ of the **adjoint field's own** gradient: the adjoint of the wavenumber-domain eq. 14 is the same operator, so back-propagated residuals travel with their own anisotropic kinematics | $S_n$, $S_\varepsilon$, $S_\delta$, $S_\theta$ of the forward field | nonlinear in $\lambda$: fp32 floor $\sim$3e-3 (model gradients), $\sim$3e-2 (wavelet) |
| `'frozen'` | transpose of the step with $S_n$ held at the **forward** field (no $Q$) | same | linear in $\lambda$: rounding only, $\le$1e-4 |
| `'exact'` | the exact transpose above (with $Q$) | same | percent-level floor; `memory=Full()` only |

`'frozen'` carries the adjoint with the incident wave's direction, and where
the forward field has passed and is weak ($S_n \to 0$) elliptically. For the
Born pair it is worse: at $m_p = 0$ the scattered field is zero, so `'frozen'`
and `'exact'` back-propagate the residual elliptically ($\delta$ lost) in the
first LSRTM iteration. Hence the default.

### impl='c'

`qp_vti2d` / `qp_tti2d` (and `qp_vti_lsrtm2d` / `qp_tti_lsrtm2d`) run the same
discretisation. The adjoint step is two launches: the state-dependent
coefficients, the CPML adjoints of the own cell and the fused model gradient,
then the transposed stencils. Because the backward needs the forward state of
step $n+1$ at reverse step $n$ (in `'frozen'` / `'exact'` the adjoint
coefficients themselves, in every mode the imaging terms near the CPML):

* `memory=Full()` stores $(P, \psi_x, \psi_z, \zeta_x, \zeta_z)$ per step
  (5 grids; the Born pair: the scattered state and the background update, 6);
* `memory=BoundarySaving(...)` stores the same five fields on the whole CPML
  frame -- the absorbing pad plus $M+1$ cells of the physical box, all four
  faces, corners included (`cuda_layout.boundary_frame`) -- and rebuilds $P$
  in the deep interior by reverse stepping; fp32 / fp16 / bf16 on gpu, cpu or
  disk (int8 is refused, see below). In `'adjoint'` and `'frozen'` mode it
  reproduces the full-storage gradient to $\le$2e-4 (fp32); `'exact'` is refused.
* No checkpointing, no domain decomposition, no free surface.

Measured against eager on V100 (canonical 48 x 56 and 72 x 90 / 420 steps,
orders 4 and 8, $\varepsilon \gtrless \delta$, VTI and TTI): records 1e-6-2e-4
(the same as eager on CPU against eager on GPU); `'frozen'` gradients
1e-5-1e-3 in full and fp32 boundary saving (the strict element-wise criterion
of `backend_gradient_matrix.py` holds wherever eager-cpu against eager-gpu
holds); `'adjoint'` gradients within eager's own gpu/cpu spread. int8 boundary
storage is refused: $S_n$ depends on the direction of $\nabla P$ only, so the
quantised frame turns into direction errors of the rebuilt field, and the
gradients move by 8-15 % away from the source.

## Born equation (LSRTM)

Perturb the model, $V \to V + \delta V$ (likewise $\varepsilon, \delta, \theta$),
and the wavefield, $P = P_0 + \delta P$, in the **linear** wavenumber-domain
equation above. Writing it as $P_{tt} = V^2\mathcal L[\varepsilon,\delta,\theta]\,P$,
the first-order terms give

$$
\frac{\partial^2 \delta P}{\partial t^2} = V^2\mathcal L\,\delta P
 + 2V\delta V\,\mathcal L P_0
 + V^2\big(\delta_\varepsilon\mathcal L\,\delta\varepsilon + \delta_\delta\mathcal L\,\delta\delta
 + \delta_\theta\mathcal L\,\delta\theta\big) P_0 .
$$

For velocity LSRTM with the reflectivity $m_p = 2\,\delta V/V$ (as
`AcousticLSRTM`):

$$
\frac{\partial^2 P_0}{\partial t^2} = V^2\mathcal L P_0 + f,\qquad
\frac{\partial^2 \delta P}{\partial t^2} = V^2\mathcal L\,\delta P + m_p\,V^2\mathcal L P_0 ,
$$

the scattered field obeys **the same qP operator** as the background, driven by
$m_p$ times the background's $V^2\mathcal L P_0$ ($= \partial_t^2 P_0$ away from
the source). In VTI, $\delta_\varepsilon\mathcal L = 2\partial_{xx} + \partial_\varepsilon S_k(\partial_{xx}+\partial_{zz})$
and $\delta_\delta\mathcal L = \partial_\delta S_k(\partial_{xx}+\partial_{zz})$
give the extra scattering sources of a multi-parameter Born, not implemented here.

In the space domain each field evaluates the anisotropy operator on **its own
gradient**: $S_n(\nabla P_0)$ in the background step and in the scattering
source, $S_n(\nabla\delta P)$ in the scattered step:

$$
\delta P^{n+1} = 2\delta P^n - \delta P^{n-1} + V^2\Delta t^2\,\mathcal L_h[\delta P^n]
 + m_p\,V^2\Delta t^2\,\mathcal L_h[P_0^n],
$$

$\mathcal L_h[Q]$ being the bracket of the forward step with $S_n(\nabla Q)$ and
the CPML memories of $Q$. This is what `AcousticVTILSRTM` / `AcousticTTILSRTM`
step (`qP_lsrtm.py`; source on `h1`, receivers on `sh1`).

**Why not linearise the space-domain step instead.** Linearising
$\mathcal L_h[P_0 + \delta P]$ in $\delta P$ gives
$\mathcal L_h[P_0] + J(P_0)\,\delta P$ with

$$
J(P_0)\,\delta P = (A + S_n(\nabla P_0))L^{\rm pml}_x\delta P + (C + S_n(\nabla P_0))L^{\rm pml}_z\delta P
 - E\,D_z(\delta p_x)
 + \Sigma_0\Big(\frac{\partial S_n}{\partial p_x}\delta p_x + \frac{\partial S_n}{\partial p_z}\delta p_z\Big).
$$

It carries the scattered wave with the **incident** wave's direction
$S_n(\nabla P_0)$, plus a correction of size $|\Sigma_0|/|\nabla P_0| \sim k$ that
does not shrink with the incident amplitude ($S_n$ is homogeneous of degree 0).
Behind the incident wavefront $\nabla P_0 \to 0$, $S_n(\nabla P_0) \to 0$ and $J$
propagates $\delta P$ through an elliptical medium ($\delta$ disappears). The
scattered wave of the true (nonlinear) space-domain model has the opposite
behaviour, $S_n(\nabla(P_0 + \delta P)) \approx S_n(\nabla\delta P)$ wherever it
dominates. $J$ is still the exact derivative of the discrete forward map, so it
is what the FWI gradient of `AcousticVTI` / `AcousticTTI` transposes; it is just
not a scattering propagator.

The Born map above is homogeneous of degree one in $m_p$ (scaling $m_p$ scales
the data) but, through $S_n(\nabla\delta P)$, not exactly additive. With
$\lambda_s$ the adjoint of the scattered field (propagated as `s_gradient`
says; by default with its own $S_n$),

$$
\frac{\partial J}{\partial m_p} = \sum_n \lambda_s^{n+1}\, V^2\Delta t^2\,\mathcal L_h[P_0^n].
$$

`impl='c'` returns this reflectivity gradient only (the LSRTM case: background
models and wavelet without gradient); asking it for the background models' or
the wavelet's gradient raises, since those need the background adjoint --
`impl='eager'` returns them all.
