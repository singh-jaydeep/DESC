"""Pre-check for run_al.py: QuadraticFlux field grid LinearGrid(N=50) vs N=100 (verified exact) for planar N7."""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.io import load
from desc.grid import LinearGrid

eq, _ = S.load_equilibrium("work/ss_qh/arcB2/ss_k4/eq_k4.h5")
x, n, _ = S._bn_grid_data(eq, 72, 96, True)
for p in ["../sweep_qh/initial starts/start_planarN7.h5", "work/ss_qh/fixed_ss_eq/easy_R3/result.h5"]:
    cs = load(p)
    B = {N: np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=LinearGrid(N=N))) for N in (50, 100)}
    d = np.linalg.norm(B[50] - B[100], axis=1) / np.linalg.norm(B[100], axis=1)
    bn = {N: np.mean(np.abs(np.sum(B[N] * n, 1))) / np.mean(np.linalg.norm(B[N], axis=1)) for N in B}
    print(f"{p.split('/')[-2]}/{p.split('/')[-1]}: max rel |dB| {d.max():.2e}; bn N50 {bn[50]:.10e} N100 {bn[100]:.10e}")
