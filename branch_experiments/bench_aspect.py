"""Does QR-then-SVD(R) beat SVD(J_a)?  Depends on m/n: the QR compresses m->n first."""
import sys, json, time; sys.path.insert(0,".")
import numpy as np, jax, jax.numpy as jnp
from bench_qr_alpha import make_problem
from bench_variants import timeit
from verify_variants import outer_any_shape as outer_qr
from variants_alpha import prep_svd_of_R, prep_svd_of_J, prep_cho

rows=[]
for n in [500, 1000, 2000]:
    for ratio in [0.5, 1.0, 1.5, 3.0, 6.0, 12.0]:
        m = int(ratio*n)
        if m < 50: continue
        J,f = make_problem(m,n,cond_exp=6.0)
        t_qr,(z,R,pn) = timeit(outer_qr,J,f)
        t_svdR,_ = timeit(prep_svd_of_R,R,z)
        t_svdJ,_ = timeit(prep_svd_of_J,J,f)
        t_cho,_  = timeit(prep_cho,R,z)
        rows.append(dict(n=n,m=m,ratio=ratio,t_qr=t_qr,t_svdR=t_svdR,
                         t_svdJ=t_svdJ,t_cho=t_cho,
                         prep_C=t_qr+t_svdR, prep_D=t_svdJ))
        print(f"n={n:5d} m/n={ratio:5.1f} m={m:6d} | QR(J)={t_qr*1e3:8.1f} +SVD(R)={t_svdR*1e3:8.1f} "
              f"= prepC {(t_qr+t_svdR)*1e3:8.1f} | SVD(J)={t_svdJ*1e3:8.1f} | C/D={(t_qr+t_svdR)/t_svdJ:5.2f}",flush=True)
json.dump(rows,open("aspect_sweep.json","w"),indent=2)
