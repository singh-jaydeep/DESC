"""Write iota0_lo.npy for the ORIGINAL precise_QA into a run directory, so that a --from
restart ties its iota bound to the true baseline. Matches stage1_v1.py's construction exactly."""
import os, sys
os.chdir("/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import GenericObjective

eq = get("precise_QA")
gi = LinearGrid(rho=np.array([0.0, 0.25, 0.5, 0.75, 1.0]), M=eq.M, N=eq.N, NFP=eq.NFP)
o = GenericObjective("iota", thing=eq, grid=gi)
o.build(verbose=0)
iota0 = np.asarray(o.compute(*o.xs(eq)))
out = sys.argv[1]
np.save(out, iota0)
print(f"wrote {out}  dim_f={iota0.size}  iota {iota0.min():.6f}..{iota0.max():.6f}")
print(f"  values: {np.array2string(iota0, precision=6)}")
