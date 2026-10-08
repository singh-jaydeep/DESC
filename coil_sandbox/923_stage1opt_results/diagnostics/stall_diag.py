"""Why did the live-eps_1 run stall? Check the objective AT THE STALLED GEOMETRY.

T3 only validated the gradient at the baseline (freeze point). This repeats it at the
displaced boundary, and checks the health of the chain that feeds it.
"""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
sys.path.insert(0, "922_stage1opt_results")
import numpy as np, jax, jax.numpy as jnp
from desc.io import load
from desc.examples import get
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid
import contour_curve as CC

K, M, N, n, Mt, Nt = 8, 32, 64, 360, 16, 16

def constants(eq):
    th = 2*np.pi*np.arange(M)/M; ze = 2*np.pi*np.arange(N)/N
    g = LinearGrid(rho=np.array([1.0]), theta=th, zeta=ze, NFP=1, sym=False)
    nd = np.asarray(g.nodes)
    nt = np.rint(nd[:,1]/(2*np.pi)*M).astype(int) % M
    nz = np.rint(nd[:,2]/(2*np.pi)*N).astype(int) % N
    return dict(transforms=get_transforms(CC.DATA_KEYS, obj=eq, grid=g),
                profiles=get_profiles(CC.DATA_KEYS, obj=eq, grid=g),
                perm=jnp.asarray(np.lexsort((nz, nt))),
                Rm=jnp.asarray(eq.surface.R_basis.modes),
                Zm=jnp.asarray(eq.surface.Z_basis.modes),
                theta=jnp.asarray(2*np.pi*np.arange(n)/n),
                trunc=CC.truncation_index(M, N, Mt, Nt))

for tag, path in [("BASELINE", None),
                  ("STALLED ", "922_stage1opt_results/runs/live_eps1_L1/eq_lo.h5")]:
    if path and not os.path.exists(path):
        print(f"{tag}: {path} not written yet"); continue
    eq = get("precise_QA") if path is None else load(path)
    const = constants(eq); NFP = int(eq.NFP)
    (xs, ws), diag = CC._curves_from_params(eq.params_dict, const, M, N, NFP, K, 6, 0.0,
                                            diagnostics=True)
    e = np.asarray(CC._eps1_live_all(xs, ws))
    # eigenvalue separation -- is lambda_0 still simple?
    seps = []
    for k in range(K):
        x, w = np.asarray(xs[k]), np.asarray(ws[k])
        W = w.sum(); D = x - (w[:,None]*x).sum(0)/W
        lam = np.linalg.eigvalsh((D*w[:,None]).T @ D / W)
        seps.append(lam[0]/lam[1])
    print(f"\n=== {tag} ===")
    print(f"  eps_1 mean {e.mean():.6f}  range {e.min():.6f}-{e.max():.6f}")
    print(f"  lam0/lam1  min {min(seps):.5f}  max {max(seps):.5f}   (small = lam0 simple)")
    print(f"  monotonic={diag['monotonic']}  dPhi/dzeta {diag['dPhi_dzeta_min']:.3e}.."
          f"{diag['dPhi_dzeta_max']:.3e}")
    print(f"  worst newton residual {max(diag['newton_residual']):.2e}   I/G "
          f"{diag['I']/diag['G']:+.2e}")
    print(f"  Phi~ truncation kept {diag['mode_truncation_kept_fraction']*100:.2f}% of |Phi_mn|")
    # gradient vs central differences AT THIS GEOMETRY
    p0 = eq.params_dict
    def f_of_Rb(rb):
        pp = dict(p0); pp["Rb_lmn"] = rb
        x, w = CC._curves_from_params(pp, const, M, N, NFP, K, 6, 0.0)
        return jnp.mean(CC._eps1_live_all(x, w))
    rb0 = jnp.asarray(p0["Rb_lmn"])
    gA = np.asarray(jax.grad(f_of_Rb)(rb0))
    print(f"  |grad|_inf over Rb_lmn = {np.abs(gA).max():.4e}   (a zero gradient would "
          f"explain a zero GN step)")
    worst = 0.0
    for i in np.argsort(-np.abs(gA))[:4]:
        h = 1e-7
        gN = float((f_of_Rb(rb0.at[i].add(h)) - f_of_Rb(rb0.at[i].add(-h)))/(2*h))
        r = abs(gA[i]-gN)/max(abs(gN),1e-30); worst = max(worst, r)
        print(f"    dRb[{i:3d}] analytic {gA[i]:+.8e}  FD {gN:+.8e}  rel {r:.2e}")
    print(f"  -> worst gradient error {worst:.2e}  "
          f"{'OK' if worst < 1e-6 else '*** GRADIENT IS WRONG HERE ***'}")
