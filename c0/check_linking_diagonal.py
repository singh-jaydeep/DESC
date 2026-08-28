"""Check whether CoilSetLinkingNumber measures linking or self-linking (writhe).

`CoilSetLinkingNumber.compute` returns `jnp.abs(link).sum(axis=0)` over the FULL
matrix returned by `CoilSet._compute_linking_number`, diagonal included. The diagonal
is each coil's Gauss integral with itself -- its writhe -- which is nonzero for any
non-planar curve. The docstring says the value is the sum over "every OTHER coil".

Run:  python c0/check_linking_diagonal.py
"""

import numpy as np

from desc.coils import initialize_modular_coils
from desc.examples import get
from desc.grid import LinearGrid
from desc.io import load
from desc.objectives import CoilSetLinkingNumber

eq = get("precise_QA")
cases = {
    "circular start (planar coils)": initialize_modular_coils(
        eq, num_coils=4, r_over_a=3.0
    ).to_FourierXYZ(N=10),
    "reference (optimized, non-planar)": load(
        "c0/ref_fourier_nc4/solved_fourierxyz_xyz_nc4_N10_nc4_s0.h5"
    ),
}

for name, cs in cases.items():
    print(f"\n=== {name} ===")
    for N in (50, 200, 400):
        M = np.asarray(
            cs._compute_linking_number(params=cs.params_dict, grid=LinearGrid(N=N))
        )
        diag = np.diag(M)
        offd = M - np.diag(diag)
        obj = CoilSetLinkingNumber(cs, grid=LinearGrid(N=N))
        obj.build(verbose=0)
        f = np.abs(np.asarray(obj.compute(cs.params_dict)))
        print(
            f"  N={N:>3}  objective max = {f.max():.4e}   "
            f"|diag|max = {np.abs(diag).max():.4e}   "
            f"|offdiag|max = {np.abs(offd).max():.4e}"
        )

print("""
Read: |offdiag| ~ 1e-16 means the coils are genuinely UNLINKED -- the constraint is
inactive for its stated purpose. The objective value tracks |diag| instead, which is
writhe. It converges to a nonzero constant with resolution rather than to 0, so
`target=0` is unsatisfiable and is really a penalty on coil non-planarity.
The planar circular case has zero writhe, which is why it alone converges to 0.
""")
