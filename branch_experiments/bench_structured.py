"""Variant E2 (blocked structured QR) vs A (dense) and C (SVD-of-R), CPU."""
import sys, os, json, time; sys.path.insert(0, os.getcwd())
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from bench_qr_alpha import make_problem
from bench_variants import outer_qr, timeit
from variants_alpha import prep_svd_of_R, step_svd_of_R, step_qr_dense
from structured_qr import structured_alpha_iteration
from jax.scipy.linalg import qr_multiply
from variants_alpha import solve_triangular_regularized

def dense_alpha_iter(R, zp, alpha):
    n = R.shape[1]
    A = jnp.vstack([R, jnp.sqrt(alpha)*jnp.eye(n)])
    Qtz, Rtil = qr_multiply(A, zp, mode="right")
    p = solve_triangular_regularized(Rtil, -Qtz)
    q = solve_triangular_regularized(Rtil.T, p, lower=True)
    return p,q,Rtil
dense_alpha_iter = jax.jit(dense_alpha_iter)

rows=[]
for n in [500,1000,2000,4000]:
    m = int(1.5*n)
    J,f = make_problem(m,n,cond_exp=6.0)
    z,R,pn = outer_qr(J,f)
    zp = jnp.concatenate([z, jnp.zeros(n)])
    t_dense,_ = timeit(dense_alpha_iter, R, zp, 1.0)
    best=None
    for blk in [32,64,128,256]:
        t,_ = timeit(structured_alpha_iteration, R, z, 1.0, blk)
        if best is None or t<best[1]: best=(blk,t)
        print(f"  n={n} blk={blk:4d} structured={t*1e3:9.2f}ms", flush=True)
    pnC,Vt,s,c = prep_svd_of_R(R,z)
    t_svd,_ = timeit(step_svd_of_R, pnC,Vt,s,c, 0.1*float(jnp.linalg.norm(pn)), 0.0)
    rows.append(dict(n=n,m=m,t_dense_iter=t_dense,t_struct_iter=best[1],best_block=best[0],
                     t_svd_call=t_svd, speedup_struct=t_dense/best[1]))
    print(f"n={n:5d}: dense={t_dense*1e3:9.2f}ms  structured(blk={best[0]})={best[1]*1e3:8.2f}ms "
          f"({t_dense/best[1]:5.2f}x)  svdR_whole_call={t_svd*1e3:7.2f}ms", flush=True)
json.dump(rows, open("structured_cpu.json","w"), indent=2)
