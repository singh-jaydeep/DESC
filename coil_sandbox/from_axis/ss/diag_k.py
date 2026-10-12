"""Where do the QS two-term rows and force error live? And does a tight re-solve change iota/QS?"""
import os, sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
from desc.io import load
from desc.grid import LinearGrid
import desc.objectives as O
from qs_scale_invariant import QuasisymmetryTwoTermScaleInvariant as QS2


def report(eq, tag):
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        g = LinearGrid(rho=np.array([r]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        o = QS2(eq, helicity=(1, 0), grid=g); o.build(verbose=0)
        f = np.asarray(o.compute_unscaled(*[eq.params_dict]))
        b = O.QuasisymmetryBoozer(eq, helicity=(1, 0), normalize=False, grid=LinearGrid(rho=np.array([r]), M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, sym=False)); b.build(verbose=0)
        bq = float(np.linalg.norm(b.compute_unscaled(eq.params_dict)))
        print(f"{tag} rho {r}: QS2 rms {np.sqrt(np.mean(f**2)):.3e} max {np.abs(f).max():.3e}  booz {bq:.3e}", flush=True)
    g = LinearGrid(rho=np.linspace(0.0, 1, 11), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    d = eq.compute(["|F|_normalized", "iota"], grid=g)
    F = g.compress(np.asarray(d["|F|_normalized"]) , surface_label="rho") if False else None
    rho = g.nodes[:, 0]
    for r in np.unique(rho):
        m = rho == r
        print(f"{tag}  rho {r:.1f} |F|n max {np.asarray(d['|F|_normalized'])[m].max():.2e} iota {np.asarray(d['iota'])[m][0]:.4f}", flush=True)


for f in sys.argv[1:]:
    eq = load(f)
    report(eq, os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f))
    if "--resolve" in os.environ.get("DIAG", ""):
        eq.solve(objective=O.ObjectiveFunction(O.ForceBalance(eq, jac_chunk_size=200)), verbose=0, maxiter=300,
                 ftol=1e-10, xtol=1e-12, gtol=1e-10)
        report(eq, "resolved")
