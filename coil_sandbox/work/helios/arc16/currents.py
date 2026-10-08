"""Per-coil currents and lengths at every chain stage, for every 16-coil arcB2 run.
The current SUM is constrained, so current can only be moved between coils -- this table
shows WHERE in the chain a coil gets starved (see HANDOFF next-step 0)."""
import sys, os
os.chdir("/home/singh/Documents/DESC2/coil_sandbox"); sys.path.insert(0, "sandbox")
os.environ["SANDBOX_DEVICE"] = "cpu"
import boot; boot.setup(default="cpu")
import glob
import numpy as np, common as S
from desc.io import load

STAGES = [("fit", "fit.h5"), ("repair", "repair/result.h5"),
          ("legA(held)", "legA/result.h5"), ("legB(free)", "legB/result.h5")]
print(f"{'run':<20} {'stage':<11} {'currents (MA)':<32} {'min/max':>8} {'lengths (m)'}")
for d in sorted(glob.glob("work/helios/arc16/cut*/")):
    lab = os.path.basename(d.rstrip("/"))[3:]
    for sname, sp in STAGES:
        p = os.path.join(d, sp)
        if not os.path.exists(p):
            continue
        cs = load(p)
        I = np.array([float(c.current) / 1e6 for c in S.iter_unique(cs)])
        L = [S.curve_metrics(c)["L"] for c in S.iter_unique(cs)]
        r = I.min() / I.max()
        flag = "  <-- STARVED" if r < 0.5 else ""
        print(f"{lab:<20} {sname:<11} " + " ".join(f"{v:7.2f}" for v in I)
              + f" {r:8.2f} " + " ".join(f"{v:6.2f}" for v in L) + flag)
    print()
