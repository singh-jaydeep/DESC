import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import qr_multiply
from bench_qr_alpha import make_problem
from verify_variants import outer_any_shape
from structured_qr import structured_retriangularize

print(f"{'n':>5} {'m':>5} {'blk':>4} {'alpha':>9} {'||Rt^T Rt-(R^TR+aI)||_rel':>25} {'||Qtz-ref||_rel':>17} {'||p-p_dense||_rel':>18}")
for (m,n) in [(150,100),(300,200),(200,200),(150,200)]:
    J,f = make_problem(m,n,cond_exp=6.0)
    z,R,_ = outer_any_shape(J,f)
    nn = R.shape[1]
    zp = jnp.concatenate([z, jnp.zeros(nn)])
    for blk in [16,64,256]:
        for a in [1e-10,1e-3,1.0,1e3]:
            Rt,Qtz = structured_retriangularize(R,z,a,block=blk)
            G = R.T@R + a*jnp.eye(nn)
            e1 = float(jnp.linalg.norm(Rt.T@Rt-G)/jnp.linalg.norm(G))
            A = jnp.vstack([R, jnp.sqrt(a)*jnp.eye(nn)])
            Qtz_ref, Rd = qr_multiply(A, zp, mode="right")
            # sign convention can differ per row; compare the solved step instead
            pd = jnp.linalg.solve(Rd, -Qtz_ref)
            ps = jnp.linalg.solve(Rt, -Qtz)
            e2 = float(jnp.linalg.norm(jnp.abs(Qtz)-jnp.abs(Qtz_ref))/jnp.linalg.norm(Qtz_ref))
            e3 = float(jnp.linalg.norm(ps-pd)/jnp.linalg.norm(pd))
            print(f"{n:5d} {m:5d} {blk:4d} {a:9.1e} {e1:25.2e} {e2:17.2e} {e3:18.2e}")
