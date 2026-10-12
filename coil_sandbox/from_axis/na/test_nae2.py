"""Validate nae2 against pyQSC r2 (X2*, Y2*, Z2*, B20) and check grad grad B (tangential FD, symmetry, trace)."""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pylib")); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
from qsc import Qsc
from nae import NearAxis, helicity
from nae2 import second_order
from folded import Layout

def case(name, rc, zs, nfp, eta, B2c):
    nphi = 61
    q0 = Qsc(rc=np.asarray(rc), zs=np.asarray(zs), nfp=nfp, etabar=eta, B2c=B2c, order="r2", nphi=nphi)
    na = NearAxis(nfp, nphi)
    g = na.geometry(jnp.asarray(rc), jnp.asarray(zs)); h = helicity(g["n"], na.phi, nfp)
    q = na.solve(jnp.asarray(rc), jnp.asarray(zs), eta, h, iota0=q0.iota)
    r2 = second_order(q, B2c)
    err = {k: float(np.max(np.abs(np.asarray(r2[k]) - getattr(q0, k))) / max(1e-12, np.max(np.abs(getattr(q0, k)))))
           for k in ("X20", "X2s", "X2c", "Y20", "Y2s", "Y2c", "Z20", "Z2s", "Z2c", "B20")}
    GG = np.asarray(r2["GG"])
    sym = max(np.abs(GG - GG.transpose(0, 2, 1, 3)).max(), np.abs(GG - GG.transpose(0, 1, 3, 2)).max()) / np.abs(GG).max()
    tr = np.abs(np.einsum("qiik->qk", GG)).max() / np.abs(GG).max()
    # tangential check: t_i GG[i, j, k] = d(G_jk)/dl along the axis, G from the first order (validated).
    # Cartesian components are not periodic over a field period, so differentiate the (periodic) cylindrical ones:
    # G_cart = R G_cyl R^T, dG_cart/dphi = R (dG_cyl/dphi + W G_cyl - G_cyl W) R^T, W = R^T dR/dphi.
    G = np.asarray(q["G"]); dl = np.asarray(q["dl"]); D = np.asarray(na.D); ph = np.asarray(na.phi)
    c, s_ = np.cos(ph), np.sin(ph); z0 = np.zeros_like(ph); o = np.ones_like(ph)
    R = np.stack([np.stack([c, -s_, z0], -1), np.stack([s_, c, z0], -1), np.stack([z0, z0, o], -1)], 1)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]])
    Gc = np.einsum("qji,qjk,qkl->qil", R, G, R)
    dGc = np.einsum("pq,qjk->pjk", D, Gc) + np.einsum("ij,qjk->qik", W, Gc) - np.einsum("qij,jk->qik", Gc, W)
    dGdl = np.einsum("qij,qjk,qlk->qil", R, dGc, R) / dl[:, None, None]
    tGG = np.einsum("qi,qijk->qjk", np.asarray(q["t"]), GG)
    tan = np.abs(tGG - dGdl).max() / np.abs(dGdl).max()
    print(f"{name:22s} max rel err vs pyQSC: " + " ".join(f"{k} {v:.1e}" for k, v in err.items()))
    print(f"{'':22s} B20 residual {float(r2['B20_residual']):.3e} (pyQSC {q0.B20_residual:.3e})   GG symmetry {sym:.1e}  trace {tr:.1e}  tangential FD {tan:.1e}")

q = Qsc.from_paper("r2 section 5.1")
case("r2 section 5.1 (QA)", q.rc, q.zs, q.nfp, q.etabar, q.B2c)
q = Qsc.from_paper("r2 section 5.2")
case("r2 section 5.2 (QH)", q.rc, q.zs, q.nfp, q.etabar, q.B2c)
for run in ["runs2/elong_qa2s2_e5", "runs2/elong_qh5s2_e5"]:
    z = np.load(run + ".npz"); L = Layout(int(z["nfp"]), int(z["nc"]), int(z["K"]), int(z["Kax"]), str(z["mode"]))
    rc, zs, eta, C = L.split(jnp.asarray(z["p"]))
    case(run.split("/")[-1] + " B2c=0", np.asarray(rc), np.asarray(zs), L.nfp, float(eta), 0.0)
