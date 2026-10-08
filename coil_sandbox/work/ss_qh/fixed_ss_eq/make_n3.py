"""Truncate the cold planar N7 start to N=3 (modes |n|>=4 dropped; all <= 6 mm vs r0 0.29 m)."""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np, common as S
from desc.io import load
cs = load("../sweep_qh/initial starts/start_planarN7.h5")
for c in S.iter_unique(cs):
    c.change_resolution(N=3)
cs.save("work/ss_qh/fixed_ss_eq/start_planarN3.h5")
cs = load("work/ss_qh/fixed_ss_eq/start_planarN3.h5")
print("N", [c.N for c in S.iter_unique(cs)], "r_n sizes", [np.asarray(c.r_n).size for c in S.iter_unique(cs)])
