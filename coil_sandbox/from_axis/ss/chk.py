import os, sys
sys.path.insert(0, "../../sandbox")
import boot; boot.setup(default="gpu")
import numpy as np
from desc.io import load
from desc.objectives import ForceBalance, ObjectiveFunction
for f in sys.argv[1:]:
    eq = load(f)
    o = ObjectiveFunction(ForceBalance(eq, jac_chunk_size=200)); o.build(verbose=0)
    e = np.asarray(o.compute_scaled_error(o.x(eq)))
    d = eq.compute(["R0/a", "a"])
    print(f, eq.L, eq.M, eq.N, "cost", 0.5 * float(np.sum(e**2)), "A", float(d["R0/a"]), "a", float(d["a"]))
