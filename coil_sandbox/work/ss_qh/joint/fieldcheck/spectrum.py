"""(1) Where B.n's residual sits vs where |B|'s symmetry-breaking sits, by toroidal band, on the target boundary.
(2) A cheap QS proxy: Boozer fit of the actual coil |B| on the TARGET's surfaces, in the target's Boozer angles
(no tracing, no equilibrium solve). Compared against the traced-surface results to see whether it can screen coilsets.
Usage: spectrum.py NAME COILS EQ_TARGET [OUTDIR]   (vacuum; currents rescaled to the target's mean |B| on the boundary)"""
import json
import os
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
import common as S  # noqa: E402

name, coils_f, eqt_f = sys.argv[1:4]
out = sys.argv[4] if len(sys.argv) > 4 else "work/ss_qh/joint/fieldcheck/spectrum"
os.makedirs(out, exist_ok=True)
eq = S.load_equilibrium(eqt_f)
eq = eq[0] if isinstance(eq, tuple) else eq
cs = desc.io.load(coils_f)
src, _ = S.source_grid(cs, 24, 400)
NFP = eq.NFP
log = open(f"{out}/{name}.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def Bcoil(d):
    x = np.column_stack([d["R"], d["phi"], d["Z"]])
    return np.asarray(cs.compute_magnetic_field(x, basis="rpz", source_grid=src))


g = LinearGrid(rho=np.array([1.0]), M=24, N=24, NFP=NFP, sym=False)
d = eq.compute(["R", "phi", "Z", "|B|"], grid=g)
scale = float(np.mean(d["|B|"]) / np.mean(np.linalg.norm(Bcoil(d), axis=1)))
for c in S.iter_unique(cs):
    c.current = c.current * scale

M, N = 14, 14
mm, nn = np.meshgrid(np.arange(0, M + 1), np.arange(-N, N + 1), indexing="ij")
modes = np.column_stack([mm.ravel(), nn.ravel()])
modes = modes[~((modes[:, 0] == 0) & (modes[:, 1] < 0))]
symm = modes[:, 1] == modes[:, 0]  # QH: n = m (NFP units) in cos(m tB - n NFP zB); sign checked below
symm2 = modes[:, 1] == -modes[:, 0]
an = np.abs(modes[:, 1])


def fit(tB, zB, f, parity):
    arg = np.outer(tB, modes[:, 0]) - np.outer(NFP * zB, modes[:, 1])
    A = np.cos(arg) if parity == "cos" else np.sin(arg)
    return np.linalg.lstsq(A, f, rcond=None)[0]


def bands(c, mask):
    return {b: float(np.sqrt(np.sum(c[mask & (an >= lo) & (an <= hi)] ** 2)))
            for b, (lo, hi) in {"0-2": (0, 2), "3-5": (3, 5), "6-9": (6, 9), "10+": (10, 99)}.items()}


res = {"name": name, "coils": coils_f, "eq": eqt_f, "scale": scale}
say(f"=== {name}  coils {coils_f}  target {eqt_f}  current scale {scale:.5f}")
# ---- (1) boundary spectra ----
g = LinearGrid(rho=np.array([1.0]), M=48, N=48, NFP=NFP, sym=False)
d = eq.compute(["R", "phi", "Z", "|B|", "n_rho", "theta_B", "zeta_B"], grid=g, M_booz=16, N_booz=16)
B = Bcoil(d)
B0 = float(np.mean(d["|B|"]))
bn = np.sum(B * d["n_rho"], axis=1) / B0
cbn = fit(d["theta_B"], d["zeta_B"], bn, "sin")  # stellarator symmetric: B.n is odd
cB = fit(d["theta_B"], d["zeta_B"], (np.linalg.norm(B, axis=1) - d["|B|"]) / B0, "cos")
ce = fit(d["theta_B"], d["zeta_B"], d["|B|"] / B0, "cos")
keep = symm if np.sum(np.abs(ce[symm])) > np.sum(np.abs(ce[symm2])) else symm2
brk = ~keep
res["bn_rms"] = float(np.sqrt(np.mean(bn ** 2)))
res["bn_bands"] = bands(cbn, np.ones_like(brk))
res["dB_brk_bands"] = bands(cB, brk)
res["dB_sym_L2"] = float(np.sqrt(np.sum(cB[keep] ** 2)))
say(f"boundary: rms B.n/B0 {res['bn_rms']:.3e}  (<|B.n|>/B0 {np.mean(np.abs(bn)):.3e})")
say("  band |n|      B.n L2     d|B| breaking L2   ratio")
for b in res["bn_bands"]:
    x, y = res["bn_bands"][b], res["dB_brk_bands"][b]
    say(f"  {b:>6s}   {x:.3e}   {y:.3e}        {y / max(x, 1e-30):.2f}")
say(f"  d|B| along the symmetric modes (harmless for QS): {res['dB_sym_L2']:.3e}")
# ---- (2) proxy QS on target surfaces ----
res["proxy"] = {}
say("proxy QS (coil |B| on target surfaces, target Boozer angles): rho | breaking L2 coil  (target eq) | ripple 6-9 coil")
for rho in (0.25, 0.5, 0.75, 0.95, 1.0):
    g = LinearGrid(rho=np.array([rho]), M=40, N=40, NFP=NFP, sym=False)
    d = eq.compute(["R", "phi", "Z", "|B|", "theta_B", "zeta_B"], grid=g, M_booz=16, N_booz=16)
    out_ = []
    for f in (np.linalg.norm(Bcoil(d), axis=1), np.asarray(d["|B|"])):
        c = fit(d["theta_B"], d["zeta_B"], f, "cos")
        c = c / c[(modes[:, 0] == 0) & (modes[:, 1] == 0)][0]
        out_.append((float(np.sqrt(np.sum(c[brk] ** 2))), float(np.sqrt(np.sum(c[brk & (an >= 6) & (an <= 9)] ** 2)))))
    res["proxy"][rho] = dict(coil=out_[0][0], target=out_[1][0], ripple=out_[0][1])
    say(f"  {rho:.2f} | {out_[0][0]:.3e}  ({out_[1][0]:.3e}) | {out_[0][1]:.3e}")
json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
