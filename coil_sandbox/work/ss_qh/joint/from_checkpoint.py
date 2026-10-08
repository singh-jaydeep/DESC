"""Rebuild a run_al.py result from an outer-iteration checkpoint (a run stopped before its own end).

    python work/ss_qh/joint/from_checkpoint.py RUN_DIR START.h5 K [--vacuum]     (sandbox root, CPU)

Unpacks RUN_DIR/x_outerKK.npy into the start coilset, writes RUN_DIR/result_outerKK.h5 and .json with the same
dense check (common.evaluate against the bounds) and linking/margins that run_al.py reports at the end.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import json
import numpy as np
import common as S
from desc.io import load

run, start, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
vac = "--vacuum" in sys.argv
eq, _ = S.load_equilibrium("precise_QH")
cs = load(start)
cs.params_dict = cs.unpack_params(np.load(f"{run}/x_outer{k:02d}.npy"))
P = S.paper_bounds(eq, S.n_unique(cs), None, "bounds/precise_qh_gil.json")
m = S.evaluate(cs, eq, S.surface_tree(eq), vacuum=vac)
active = [q for q, v in S.margins(m, P).items() if abs(v - 1) <= 0.01]
outer = json.load(open(f"{run}/outer.json"))[k - 1]
print(f"CHECKPOINT outer {k}: opt {outer['optimality']:.2e} violation {outer['constr_violation']:.2e}")
print(f"RESULT bn {m['bn']:.4e}  {S.report(m, P)}  linked {m['linked']}  -> {S.verdict(m, P)}  active {active}")
cs.save(f"{run}/result_outer{k:02d}.h5")
json.dump(dict(checkpoint=k, outer=outer, bounds=P, result=m, margins=S.margins(m, P), active=active),
          open(f"{run}/result_outer{k:02d}.json", "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"wrote {run}/result_outer{k:02d}.h5, .json")
