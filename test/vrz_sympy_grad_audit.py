#!/usr/bin/env python3
"""SymPy audit of acoustic_vrz2d ``calculate_grad_vrz2d`` gradient formula.

Build the DISCRETE 1-D VRZ forward RHS with explicit 2nd-order FD stencils,
form the per-step adjoint inner product  loss = sum_i lam_i * rhs_i , and let
SymPy differentiate w.r.t. vp_j and z_j.  That derivative IS the exact discrete
adjoint gradient (what eager autograd computes).  Then evaluate the kernel's
analytic formula and compare.

Forward (matches kernels.cuh):  rhs_i = kappa_i ( b_i*lap_p_i + db_i*dp_i )
  kappa = vp*z,  b = vp/z,  db = (d vp)/z + vp*d(1/z)   [product-rule form].
"""
import sympy as sp
import random

h = sp.symbols('h', positive=True)
N = 7
j = 3

vp = sp.symbols(f'vp0:{N}', positive=True)
z  = sp.symbols(f'z0:{N}',  positive=True)
p  = sp.symbols(f'p0:{N}')
lam = sp.symbols(f'lam0:{N}')

invz = [1/z[i] for i in range(N)]
kappa = [vp[i]*z[i] for i in range(N)]
b = [vp[i]/z[i] for i in range(N)]

def d1(f, i):
    return (f[i+1] - f[i-1])/(2*h)
def d2(f, i):
    return (f[i-1] - 2*f[i] + f[i+1])/(h**2)

rhs = {}
for i in range(1, N-1):
    dvp = d1(vp, i); dinvz = d1(invz, i)
    db = dvp*invz[i] + vp[i]*dinvz
    rhs[i] = kappa[i]*(b[i]*d2(p, i) + db*d1(p, i))

loss = sum(lam[i]*rhs[i] for i in rhs)

grad_vp_exact = sp.simplify(sp.diff(loss, vp[j]))
grad_z_exact  = sp.simplify(sp.diff(loss, z[j]))

q = [kappa[i]*lam[i] for i in range(N)]
div_b_grad_p = b[j]*d2(p, j) + (d1(vp, j)*invz[j] + vp[j]*d1(invz, j))*d1(p, j)
g_kappa = -lam[j]*div_b_grad_p
g_beta  =  d1(q, j)*d1(p, j)
grad_vp_code = sp.simplify(z[j]*g_kappa + invz[j]*g_beta)
grad_z_code  = sp.simplify(vp[j]*g_kappa - b[j]*invz[j]*g_beta)

# --- PROPOSED FIX: op-by-op reverse-mode exact discrete transpose ----------
# rhs_i = vp_i^2 lap_i + vp_i (dvp_i)(dp_i) + vp_i^2 z_i (dinvz_i)(dp_i)
# Buffers:  c_i = lam_i vp_i dp_i ;  e_i = lam_i vp_i^2 z_i dp_i
dp = [d1(p, i) if 0 < i < N-1 else sp.Integer(0) for i in range(N)]
c = [lam[i]*vp[i]*dp[i] for i in range(N)]
e = [lam[i]*vp[i]**2*z[i]*dp[i] for i in range(N)]
lap_j, dpj, dvpj, dinvzj = d2(p, j), d1(p, j), d1(vp, j), d1(invz, j)
grad_vp_fix = sp.simplify(
    2*vp[j]*lam[j]*lap_j + lam[j]*dvpj*dpj - d1(c, j) + 2*vp[j]*z[j]*lam[j]*dinvzj*dpj)
grad_z_fix = sp.simplify(
    lam[j]*vp[j]**2*dinvzj*dpj + d1(e, j)/z[j]**2)

print("=== PROPOSED FIX vs exact discrete adjoint ===")
for name, exact, fix in [("grad_vp", grad_vp_exact, grad_vp_fix),
                         ("grad_z",  grad_z_exact,  grad_z_fix)]:
    d = sp.simplify(exact - fix)
    print(f"  {name}:  exact - fix = {d}   ->  {'MATCH ✅' if d == 0 else 'MISMATCH ❌'}")

print("\n=== current kernel formula vs exact ===")
for name, exact, code in [("grad_vp", grad_vp_exact, grad_vp_code),
                          ("grad_z",  grad_z_exact,  grad_z_code)]:
    diff_plus  = sp.simplify(exact - code)
    diff_minus = sp.simplify(exact + code)
    subs = {**{vp[i]: 1.5+random.random() for i in range(N)},
            **{z[i]: 1.0+random.random() for i in range(N)},
            **{p[i]: random.random()-0.5 for i in range(N)},
            **{lam[i]: random.random()-0.5 for i in range(N)}, h: 10.0}
    ev_e = float(exact.subs(subs)); ev_c = float(code.subs(subs))
    print(f"\n{name}:")
    print(f"  numeric: exact={ev_e:.6e}  code={ev_c:.6e}  exact/code={ev_e/ev_c:.4f}")
    if diff_plus == 0 or diff_minus == 0:
        print("  MATCH (up to overall sign) -- formula is exact discrete adjoint")
    else:
        print("  MISMATCH -- kernel formula is NOT the exact discrete adjoint")
        print(f"  (exact + code) = {sp.nsimplify(diff_minus) if diff_minus != 0 else 0}")
