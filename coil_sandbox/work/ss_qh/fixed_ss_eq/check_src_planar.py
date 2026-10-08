"""Pitfall 2: does LinearGrid(N=100) match N=400 as the Biot-Savart source for planar N7?"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S
from desc.io import load
from desc.grid import LinearGrid

eq, _ = S.load_equilibrium("work/ss_qh/arcB2/ss_k4/eq_k4.h5")
cs = load("../sweep_qh/initial starts/start_planarN7.h5")
x, n, _ = S._bn_grid_data(eq, 72, 96, True)
B = {N: np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=LinearGrid(N=N))) for N in (100, 400)}
d = np.linalg.norm(B[100] - B[400], axis=1) / np.linalg.norm(B[400], axis=1)
print(f"planar N7: max rel |dB| (N=100 vs 400) = {d.max():.2e}, mean {d.mean():.2e}")
for N in (100, 400):
    print(N, S.bn_stats(cs, eq, LinearGrid(N=N), vacuum=True))
