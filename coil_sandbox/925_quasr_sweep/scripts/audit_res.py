"""Does eps_1 need a solved interior, and is it resolution-converged?"""
import sys, time, warnings
sys.path.insert(0, "sandbox"); sys.path.insert(0, "922_stage1opt_results")
import boot; boot.setup(default="gpu")
warnings.filterwarnings("ignore")
import numpy as np, quasr as Q
from contour_curve import ContourClamshell
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.grid import LinearGrid
CACHE = "/tmp/claude-1000/-home-singh-Documents-DESC2-coil-sandbox/91f880f0-34c8-4d34-ba49-8b36d79ca617/scratchpad/quasr_cache"
ID = 1630198

def eps1(eq, K, phase=0.0):
    o = ContourClamshell(eq, quantity="eps1", K=K, M=64, N=128, n=240, phase=phase, Mt=16, Nt=16)
    o.build(verbose=0)
    return np.asarray(o.compute(eq.params_dict)), o.diagnostics()

def fquality(eq):
    """Relative force error the way DESC normalizes it, plus the vacuum-specific |J| check."""
    g = LinearGrid(L=16, M=2*eq.M, N=2*eq.N, NFP=eq.NFP)
    d = eq.compute(["|F|", "|grad(|B|^2)|/2mu0", "<|B|>_rms", "|J|", "|B|", "sqrt(g)"], grid=g)
    w = np.abs(np.asarray(d["sqrt(g)"]))
    rel = float(np.sum(np.asarray(d["|F|"])*w) / np.sum(np.asarray(d["|grad(|B|^2)|/2mu0"])*w))
    JB = float(np.sum(np.asarray(d["|J|"])*w)/np.sum(w)) * 4e-7*np.pi / max(float(np.mean(d["|B|"])),1e-30)
    return rel, JB

eq, rec = Q.equilibrium(ID, CACHE)
K = 2*rec["nc_per_hp"]
print(f"ID {ID}  K={K}  QUASR iota edge {rec['iota_profile'][-1]:.4f}", flush=True)

print("\n-- UNSOLVED (interior = DESC's initial guess) --", flush=True)
try:
    e, d = eps1(eq, K)
    print(f"  monotonic {d['monotonic']}  newton {d['worst_newton_residual']:.1e}  I/G {d['I']/d['G']:+.2e}")
    print(f"  eps_1 mean {e.mean():.6f}  per contour {np.array2string(e, precision=5)}")
    unsolved = e.mean()
except Exception as ex:
    unsolved = None
    print(f"  RAISED {type(ex).__name__}: {str(ex)[:400]}")

print("\n-- SOLVED at increasing resolution --", flush=True)
for (L,M,N) in [(8,10,10),(10,10,10),(12,12,12)]:
    eq2, _ = Q.equilibrium(ID, CACHE)
    eq2.change_resolution(L=L, M=M, N=N, L_grid=2*L, M_grid=2*M, N_grid=2*N)
    t0=time.time()
    eq2.solve(objective=ObjectiveFunction(ForceBalance(eq=eq2), deriv_mode="batched", jac_chunk_size=100),
              maxiter=150, verbose=0, ftol=1e-10, xtol=1e-12, gtol=1e-12)
    ts=time.time()-t0
    lg = LinearGrid(rho=np.array([1.0]), M=2*M, N=2*N, NFP=eq2.NFP)
    iot = float(np.asarray(lg.compress(eq2.compute("iota", grid=lg)["iota"]))[0])
    rel, JB = fquality(eq2)
    e, d = eps1(eq2, K)
    print(f"  L{L} M{M} N{N}: {ts:5.0f}s  iota_edge {iot:.5f} (rel {abs(abs(iot/rec['iota_profile'][-1])-1)*100:.2f}%)"
          f"  |F|/|grad B^2/2mu0| {rel:.2e}  mu0<|J|>/<|B|> {JB:.2e}")
    print(f"           eps_1 mean {e.mean():.6f}  monotonic {d['monotonic']}  newton {d['worst_newton_residual']:.1e}", flush=True)
    if unsolved is not None:
        print(f"           unsolved would have read {unsolved:.6f}  ->  {abs(unsolved/e.mean()-1)*100:.1f}% off", flush=True)
