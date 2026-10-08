"""Raise the modes per arc of an arc coilset EXACTLY (zero-padding), so only the freedom changes.

    python pad_arcs.py work/ss_qa/ss_v2_qs10_k3/coils_k3.h5 --M 5 --out work/ss_qa/coils_k3_M5.h5 --eq ...

`PiecewisePlanarArcCoil` has no change_resolution. Its shape is a (B, M) array of transverse
modes whose basis does not depend on M, so appending zero columns keeps the curve -- PROVIDED the
frozen `arc_ref` (and `rotmat`, `shift`) are copied: a new coil recomputes arc_ref from its own
hinges, and an optimized coil's arc_ref was frozen when it was built. MEASURED on the precise_QA
arcB2 coils: without the copy the padded curve moved by 27 mm; with it, 0.0 m.

Rebuilds the same container (a MixedCoilSet of one-coil symmetric CoilSets) and checks the curves
and the field on the plasma boundary against the input.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox"))
import boot  # noqa: E402

boot.setup(default="cpu")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

import common as S  # noqa: E402
from desc.coils import CoilSet, MixedCoilSet, PiecewisePlanarArcCoil  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402


def pad_coil(c, M):
    if M < c.M:
        raise SystemExit(f"--M {M} is below the current M={c.M}; this tool only raises it")
    sh = np.asarray(c.shape).reshape(c.B, c.M)
    new = PiecewisePlanarArcCoil(current=c.current, hinges=np.asarray(c.hinges), tilts=np.asarray(c.tilts),
                                 shape=np.hstack([sh, np.zeros((c.B, M - c.M))]), name=c.name)
    p = new.params_dict
    for k in ("arc_ref", "rotmat", "shift"):
        p[k] = c.params_dict[k]
    new.params_dict = p
    return new


def pad(cs, M):
    if isinstance(cs, MixedCoilSet):
        return MixedCoilSet(*[pad(sub, M) for sub in cs], name=cs.name)
    if isinstance(cs, CoilSet):
        return CoilSet(*[pad_coil(c, M) for c in cs], NFP=cs.NFP, sym=cs.sym, name=cs.name)
    return pad_coil(cs, M)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("coils")
    ap.add_argument("--M", type=int, required=True, help="new modes per arc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eq", default="precise_QA", help="equilibrium for the boundary-field check")
    a = ap.parse_args()
    cs = load(a.coils)
    new = pad(cs, a.M)
    g = LinearGrid(N=400)
    dx = max(np.max(np.abs(np.asarray(c0.compute("x", grid=g, basis="xyz")["x"])
                           - np.asarray(c1.compute("x", grid=g, basis="xyz")["x"])))
             for c0, c1 in zip(S.iter_unique(cs), S.iter_unique(new)))
    eq, _ = S.load_equilibrium(a.eq)
    d = eq.compute(["R", "phi", "Z"], grid=LinearGrid(rho=np.array([1.0]), M=16, N=16, NFP=eq.NFP))
    x = np.stack([d["R"], d["phi"], d["Z"]], 1)
    B0 = np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=S.source_grid(cs)[0]))
    B1 = np.asarray(new.compute_magnetic_field(x, basis="rpz", source_grid=S.source_grid(new)[0]))
    c1 = next(S.iter_unique(new))
    print(f"{S.n_unique(new)} unique coils, {S.expected_coils(new, eq)} total; B={c1.B}, M {next(S.iter_unique(cs)).M}"
          f" -> {c1.M}; free shape params per coil {c1.B * c1.M}\n"
          f"max curve change {dx:.3e} m; max boundary-field change {np.max(np.abs(B1 - B0)):.3e} T "
          f"(|B| ~ {np.mean(np.linalg.norm(B0, axis=1)):.3f} T)")
    if dx > 1e-12:
        raise SystemExit("padding changed the curves: NOT saved")
    new.save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
