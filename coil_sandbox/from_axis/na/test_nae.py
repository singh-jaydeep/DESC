import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, jax.numpy as jnp
from qsc import Qsc
from nae import NearAxis, helicity
for name in ["r1 section 5.1", "r1 section 5.2", "r1 section 5.3", "r2 section 5.5"]:
    q = Qsc.from_paper(name, nphi=61)
    q.calculate_grad_B_tensor()
    na = NearAxis(q.nfp, 61)
    rc = jnp.asarray(q.rc); zs = jnp.asarray(q.zs)
    g = na.geometry(rc, zs)
    h = helicity(g["n"], na.phi, q.nfp)
    out = na.solve(rc, zs, q.etabar, h, iota0=0.0)
    # pyQSC tensor cylindrical [i][j] -> cartesian at each phi
    T = np.array(q.grad_B_tensor_cylindrical)  # (3,3,nphi), i,j in (R,phi,z)
    ph = np.asarray(na.phi)
    G = np.zeros((61,3,3))
    for k in range(61):
        e = np.array([[np.cos(ph[k]), np.sin(ph[k]),0],[-np.sin(ph[k]),np.cos(ph[k]),0],[0,0,1]])  # rows: eR, ephi, ez
        G[k] = e.T @ T[:,:,k] @ e
    print(f"{name:16s} spsi={q.spsi} sG={q.sG} hel {q.helicity} vs {h}  iota {q.iota:.6f} vs {float(out['iota']):.6f}  "
          f"|dG|/|G| {np.abs(G-np.asarray(out['G'])).max()/np.abs(G).max():.2e} resid {float(out['resid']):.1e}")
