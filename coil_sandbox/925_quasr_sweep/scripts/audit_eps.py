"""Audit: is eps_1 / eps_2 well-defined and stable on a QUASR boundary?

Questions, in order:
  1. K_vc needs the interior field. What does eps_1 read on the UNSOLVED namelist equilibrium,
     and how far is that from the solved answer? (If they agree, we skip 371k solves.)
  2. Does the solved equilibrium reproduce QUASR's own iota? (Their device is a coil field;
     ours is a fixed-boundary vacuum solve of the fitted boundary.)
  3. Do the proxy-class guards pass -- Phi_mod monotone in zeta, contour Newton converged?
  4. How much do the reported eps_1 / eps_2 depend on the two arbitrary knobs, the contour
     PHASE and the contour count K?
"""
import sys, os, time, warnings
sys.path.insert(0, "sandbox")
sys.path.insert(0, "922_stage1opt_results")
import boot; boot.setup(default="gpu")
warnings.filterwarnings("ignore")
import numpy as np

import quasr as Q
from contour_curve import ContourClamshell, _curves_from_params
import clamshell as CL

CACHE = "/tmp/claude-1000/-home-singh-Documents-DESC2-coil-sandbox/91f880f0-34c8-4d34-ba49-8b36d79ca617/scratchpad/quasr_cache"
ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1630198
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def eps_report(eq, K, phase=0.0, n=240, stride=8, want_eps2=True):
    """eps_1 (live) per contour, eps_2 (DP-optimal 2-plane), and the class guards."""
    o = ContourClamshell(eq, quantity="eps1", K=K, M=64, N=128, n=n, phase=phase,
                         Mt=16, Nt=16, stride=stride)
    o.build(verbose=0)
    e1 = np.asarray(o.compute(eq.params_dict))
    d = o.diagnostics()
    out = dict(eps1=e1, monotonic=d["monotonic"], newton=d["worst_newton_residual"],
               IoverG=d["I"] / d["G"], kept=d["mode_truncation_kept_fraction"])
    if want_eps2:
        (xs, ws), _ = o._chain(eq.params_dict, o._constants, diagnostics=True)
        e2, brk = [], []
        for k in range(K):
            X, W = np.asarray(xs[k]), np.asarray(ws[k])
            v, _ = CL.eps_B(X, W, 2, stride=stride)
            e2.append(v)
            brk.append(CL.best_breaks(X, W, 2, stride=stride)[1])
        out["eps2"] = np.asarray(e2)
        out["breaks"] = brk
    return out


print(f"=== QUASR {ID} ===", flush=True)
eq, rec = Q.equilibrium(ID, CACHE)
K = 2 * rec["nc_per_hp"]
print(f"nfp {rec['nfp']}  {'QH' if rec['helicity'] else 'QA'}  nc/hp {rec['nc_per_hp']} -> K={K}"
      f"  A {rec['aspect_ratio']:.2f}  a {rec['minor_radius']:.4f}  qs 1e{rec['qs_error']:.2f}")
print(f"QUASR iota profile {rec['iota_profile']} at tflux {rec['tf_profile']}", flush=True)

# ---- 1. unsolved ----------------------------------------------------------
print("\n-- (1) UNSOLVED (boundary only, interior = initial guess) --", flush=True)
t0 = time.time()
try:
    u = eps_report(eq, K, want_eps2=False)
    print(f"  monotonic {u['monotonic']}  newton {u['newton']:.1e}  I/G {u['IoverG']:+.2e}")
    print(f"  eps_1 mean {u['eps1'].mean():.6f}  min {u['eps1'].min():.6f}  max {u['eps1'].max():.6f}")
except Exception as e:
    u = None
    print(f"  RAISED {type(e).__name__}: {str(e)[:300]}")
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ---- 2. solve -------------------------------------------------------------
print(f"\n-- (2) vacuum solve at L=M=N={RES} --", flush=True)
from desc.objectives import ForceBalance, ObjectiveFunction
t0 = time.time()
eq.change_resolution(L=RES, M=RES, N=RES, L_grid=2 * RES, M_grid=2 * RES, N_grid=2 * RES)
obj = ObjectiveFunction(ForceBalance(eq=eq), deriv_mode="batched", jac_chunk_size=100)
eq.solve(objective=obj, maxiter=100, verbose=0, ftol=1e-8, xtol=1e-10, gtol=1e-10)
ts = time.time() - t0
g = eq.compute(["iota", "rho", "|F|", "<|B|>_rms"],
               grid=__import__("desc").grid.LinearGrid(L=20, M=2 * RES, N=2 * RES, NFP=eq.NFP))
import desc.grid as DG
lg = DG.LinearGrid(rho=np.linspace(0.1, 1.0, 10), M=2 * RES, N=2 * RES, NFP=eq.NFP)
di = eq.compute(["iota", "rho"], grid=lg)
iot = np.asarray(lg.compress(di["iota"]))
print(f"  solved in {ts:.0f}s   iota(rho) {iot[0]:.4f} .. {iot[-1]:.4f}")
print(f"  QUASR mean_iota {rec['mean_iota']}  edge {rec['iota_profile'][-1]:.4f}"
      f"   -> edge rel err {abs(abs(iot[-1])/abs(rec['iota_profile'][-1]) - 1)*100:.2f}%")
from desc.objectives import ForceBalance as FB2
print(f"  |F| max {float(np.max(np.abs(g['|F|']))):.3e}", flush=True)

# ---- 3 + 4. eps on the solved equilibrium, and the two knobs --------------
print("\n-- (3,4) eps on the SOLVED equilibrium --", flush=True)
t0 = time.time()
s = eps_report(eq, K)
print(f"  monotonic {s['monotonic']}  newton {s['newton']:.1e}  I/G {s['IoverG']:+.2e}"
      f"  modes kept {s['kept']:.4f}")
print(f"  eps_1  mean {s['eps1'].mean():.6f}  per contour {np.array2string(s['eps1'], precision=5)}")
print(f"  eps_2  mean {s['eps2'].mean():.6f}  per contour {np.array2string(s['eps2'], precision=5)}")
print(f"  C = 4 eps_2/eps_1  mean {np.mean(4*s['eps2']/s['eps1']):.4f}")
if u is not None:
    print(f"  UNSOLVED vs SOLVED eps_1 mean: {u['eps1'].mean():.6f} vs {s['eps1'].mean():.6f}"
          f"  -> {abs(u['eps1'].mean()/s['eps1'].mean()-1)*100:.1f}% off")
print(f"  ({time.time()-t0:.0f}s for K={K} incl. the eps_2 DP)", flush=True)

print("\n  phase sweep (same K, contours cut at a different offset):", flush=True)
for ph in (0.0, 0.1, 0.25, 0.5):
    r = eps_report(eq, K, phase=ph)
    print(f"    phase {ph:4.2f}  eps_1 mean {r['eps1'].mean():.6f}  spread "
          f"{r['eps1'].min():.5f}..{r['eps1'].max():.5f}   eps_2 mean {r['eps2'].mean():.6f}")

print("\n  K sweep (more contours = finer sampling of the same foliation):", flush=True)
for kk in (K, 2 * K, 3 * K):
    r = eps_report(eq, kk, want_eps2=False)
    print(f"    K {kk:3d}  eps_1 mean {r['eps1'].mean():.6f}  spread "
          f"{r['eps1'].min():.5f}..{r['eps1'].max():.5f}")

print("\n  nodes-per-contour sweep (discretisation of each curve):", flush=True)
for nn in (120, 240, 480):
    r = eps_report(eq, K, n=nn, want_eps2=False)
    print(f"    n {nn:4d}  eps_1 mean {r['eps1'].mean():.8f}")

out = os.path.join(CACHE, f"eq_solved_{ID}.h5")
eq.save(out)
print(f"\nsaved solved equilibrium -> {out}", flush=True)
