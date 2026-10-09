"""QUASR 1328095 (nfp 4 QH, A 8, a 0.125) for the arcB2 hinge study: rescale Psi so <|B|> matches precise_QH.

    python work/ss_qh/joint/quasr1328095/prep_eq.py      (sandbox root, CPU)

The solved equilibrium in 925_quasr_sweep/viewers/ was loaded with PHIEDGE = 1 (|B| ~ 27 T), while QUASR's own coils
carry ~1 T. In vacuum the force balance is homogeneous in B, so scaling Psi by c scales B by c and leaves the
geometry, iota and the normalized force residual unchanged (exact). Matching precise_QH's <|B|> lets the precise_QH
bounds (absolute, a-scaled: a matches to 0.6%) and current sum carry over.
"""
import sys; sys.path.insert(0, "sandbox")
import boot; boot.setup(default="cpu")
import numpy as np
import common as S

src, dst = "925_quasr_sweep/viewers/eq_1328095.h5", "work/ss_qh/joint/quasr1328095/eq_q1328095.h5"
ref, _ = S.load_equilibrium("precise_QH")
eq, _ = S.load_equilibrium(src)
Bref = float(np.mean(np.asarray(ref.compute("|B|")["|B|"])))


def stats(e):
    f = e.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    return (float(np.mean(np.asarray(e.compute("|B|")["|B|"]))), float(np.mean(np.asarray(e.compute("iota")["iota"]))),
            float(f["<|F|>_vol"] / f["<|grad(|B|^2)|/2mu0>_vol"]))


b0, i0, f0 = stats(eq)
eq.Psi = eq.Psi * Bref / b0
b1, i1, f1 = stats(eq)
print(f"Psi -> {float(eq.Psi):.6f}; <|B|> {b0:.4f} -> {b1:.4f} (precise_QH {Bref:.4f}); iota {i0:.5f} -> {i1:.5f}; "
      f"normalized force {f0:.3e} -> {f1:.3e}; R0 {float(eq.compute('R0')['R0']):.4f} a {float(eq.compute('a')['a']):.4f}")
eq.save(dst)
print(f"wrote {dst}")
