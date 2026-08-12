import sys, os; sys.path.insert(0, os.getcwd())
"""Correctness: every variant must reproduce master's (p, hits_boundary, alpha)."""
import numpy as np, jax, jax.numpy as jnp
from variants_alpha import *
from bench_qr_alpha import make_problem, outer_factorization
from variants_alpha import solve_triangular_regularized

def outer_any_shape(J, f):
    """Mirror desc/optimize/least_squares.py:301-313 for tall AND wide J_a."""
    from jax.scipy.linalg import qr as jqr, qr_multiply as jqm
    Qt_fa, R = jqm(J, f, mode="right")
    if J.shape[0] >= J.shape[1]:
        p_newton = solve_triangular_regularized(R, -Qt_fa)
    else:
        Q, Rt = jqr(J.T, mode="economic")
        p_newton = Q @ solve_triangular_regularized(Rt.T, -f, lower=True)
    return Qt_fa, R, p_newton


def report(m, n, cond_exp, seed=0):
    J, f = make_problem(m, n, seed=seed, cond_exp=cond_exp)
    Qt_fa, R, p_newton = outer_any_shape(J, f)
    G, Rtz = prep_cho(R, Qt_fa)
    pn_svd, Vt, s, c = prep_svd_of_R(R, Qt_fa)
    pn = float(jnp.linalg.norm(p_newton))
    print(f"\n m={m} n={n} cond~1e{cond_exp:g}  ||p_newton||={pn:.3e}  "
          f"||pn_svdR - pn_qr||/||pn_qr||={float(jnp.linalg.norm(pn_svd-p_newton)/jnp.linalg.norm(p_newton)):.2e}")
    print(f"  {'Delta/||pn||':>12} {'variant':10} {'alpha':>12} {'||p||/Delta':>12} "
          f"{'rel err vs A':>13} {'KKT resid':>11}")
    for frac in [0.5, 0.1, 0.01]:
        tr = frac*pn
        pA, hA, aA = step_qr_dense(p_newton, Qt_fa, R, tr, 0.0)
        pB, hB, aB = step_cho_shift(p_newton, G, Rtz, tr, 0.0)
        pC, hC, aC = step_svd_of_R(pn_svd, Vt, s, c, tr, 0.0)
        for nm, (p_, h_, a_) in [("A qr", (pA,hA,aA)), ("B cho",(pB,hB,aB)), ("C svdR",(pC,hC,aC))]:
            # KKT: (R'R + a I) p + R'z = 0, scaled
            kkt = float(jnp.linalg.norm(R.T@(R@p_) + a_*p_ + R.T@Qt_fa) /
                        max(float(jnp.linalg.norm(R.T@Qt_fa)), 1e-300))
            rel = float(jnp.linalg.norm(p_-pA)/max(float(jnp.linalg.norm(pA)),1e-300))
            print(f"  {frac:12.2f} {nm:10} {float(a_):12.5e} "
                  f"{float(jnp.linalg.norm(p_))/tr:12.6f} {rel:13.2e} {kkt:11.2e}")

for m,n,ce in [(750,500,6.0),(500,500,6.0),(400,500,6.0),(750,500,10.0),(750,500,2.0)]:
    report(m,n,ce)

# rank-deficient R
print("\n--- rank-deficient (10 exact zero singular values) ---")
rng = np.random.default_rng(3)
n=300; m=450
U,_ = np.linalg.qr(rng.standard_normal((m,n))); V,_ = np.linalg.qr(rng.standard_normal((n,n)))
s = np.logspace(0,-6,n); s[-10:]=0.0
J = jnp.asarray((U*s)@V.T); f = jnp.asarray(rng.standard_normal(m))
Qt_fa,R,p_newton = outer_any_shape(J,f)
G,Rtz = prep_cho(R,Qt_fa); pn_svd,Vt,sv,c = prep_svd_of_R(R,Qt_fa)
tr = 0.1*float(jnp.linalg.norm(p_newton))
for nm,(p_,h_,a_) in [("A qr",step_qr_dense(p_newton,Qt_fa,R,tr,0.0)),
                      ("B cho",step_cho_shift(p_newton,G,Rtz,tr,0.0)),
                      ("C svdR",step_svd_of_R(pn_svd,Vt,sv,c,tr,0.0))]:
    kkt=float(jnp.linalg.norm(R.T@(R@p_)+a_*p_+R.T@Qt_fa)/jnp.linalg.norm(R.T@Qt_fa))
    print(f"  {nm:8} alpha={float(a_):.5e}  ||p||/Delta={float(jnp.linalg.norm(p_))/tr:.6f}  KKT={kkt:.2e}")

# Givens structured factorization: does it reproduce the dense QR?
print("\n--- Givens retriangularization vs dense QR of [R; sqrt(a) I] ---")
J,f = make_problem(180,120,cond_exp=6.0)
Qt_fa,R,_ = outer_any_shape(J,f)
Rn, zn = np.asarray(R), np.asarray(Qt_fa)
for a in [1e-8,1e-3,1.0]:
    Rt, Qtz, fl = givens_retriangularize(Rn, zn, a, count_flops=True)
    G_ = Rn.T@Rn + a*np.eye(120)
    err = np.linalg.norm(Rt.T@Rt - G_)/np.linalg.norm(G_)
    A = np.vstack([Rn, np.sqrt(a)*np.eye(120)])
    zp = np.concatenate([zn, np.zeros(120)])
    Qd, Rd = np.linalg.qr(A)
    pd_ = np.linalg.solve(Rd, -(Qd.T@zp)); pg = np.linalg.solve(Rt, -Qtz)
    print(f"  a={a:8.1e}  ||Rt'Rt-(R'R+aI)||/|.|={err:.2e}  "
          f"||p_givens-p_dense||/||p||={np.linalg.norm(pg-pd_)/np.linalg.norm(pd_):.2e}  "
          f"flops={fl:.3e} (n^3/6={120**3/6:.3e}, dense 10n^3/3={10*120**3/3:.3e})")
