"""Where does the coil field's QS error come from?
  (1) largest symmetry-breaking Boozer |B|_mn (QH helicity (1, NFP)) at rho 0.5 and 1: target vs traced equilibrium
  (2) largest boundary differences dR_mn, dZ_mn: traced surface vs the target's own rho0 surface (both fit M=N=10)
Usage: modes.py CASE EQ_TARGET EQ_TRACED [RHO0]"""
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import numpy as np  # noqa: E402
import desc.io  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
import common as S  # noqa: E402

F = "work/ss_qh/joint/fieldcheck"
case, eqt_f, eqs_f = sys.argv[1:4]
rho0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
eqt = S.load_equilibrium(eqt_f)[0]
eqs = desc.io.load(eqs_f)
NFP = eqt.NFP
out = open(f"{F}/{case}/modes.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")


def booz(e, rho, M=16, N=16):
    g = LinearGrid(rho=np.array([rho]), M=2 * e.M_grid, N=2 * e.N_grid, NFP=NFP, sym=False)
    d = e.compute(["|B|_mn_B", "<|B|>_vol"], grid=g, M_booz=M, N_booz=N)
    from desc.basis import DoubleFourierSeries

    basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="cos")
    modes = basis.modes[:, 1:]  # (m, n); n in units of NFP (toroidal mode number n * NFP)
    B = np.asarray(d["|B|_mn_B"]) / float(d["<|B|>_vol"])
    # QH helicity (1, NFP): |B| depends on theta_B - NFP zeta_B, i.e. the symmetric modes have n = +-m (n in NFP units)
    sym_ = modes[:, 1] == modes[:, 0]
    sym2 = modes[:, 1] == -modes[:, 0]
    keep = sym_ if np.sum(np.abs(B[sym_])) > np.sum(np.abs(B[sym2])) else sym2
    return modes, B, keep


for rho in (0.5, 1.0):
    say(f"--- Boozer |B|_mn / <|B|>, symmetry-breaking, rho {rho} (traced) vs rho {rho * rho0:g} (target)")
    rows = {}
    for tag, e, r in (("target", eqt, rho * rho0), ("traced", eqs, rho)):
        modes, B, keep = booz(e, r)
        br = np.where(keep, 0.0, B)
        rows[tag] = (modes, br, float(np.sqrt(np.sum(br ** 2))))
    say(f"  L2 of breaking modes: target {rows['target'][2]:.2e}  traced {rows['traced'][2]:.2e}")
    modes, bt = rows["traced"][0], rows["traced"][1]
    bg = rows["target"][1]
    order = np.argsort(-np.abs(bt))[:10]
    say("   (m, n)    traced     target")
    for i in order:
        say(f"   ({modes[i, 0]:2d},{modes[i, 1]:3d})  {bt[i]:+.2e}  {bg[i]:+.2e}")
    # grouped by |n| / NFP band
    nb = np.abs(modes[:, 1])
    say("  breaking L2 traced/target by |n| (per field period; 8 = 32 coils): " + "  ".join(
        f"{k}: {np.sqrt(np.sum(bt[nb == k] ** 2)):.1e}/{np.sqrt(np.sum(bg[nb == k] ** 2)):.1e}" for k in range(0, 17)))

rng = np.random.default_rng(0)
n = 9600
th, ze = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
d = eqt.compute(["R", "Z"], grid=Grid(np.column_stack([np.full(n, rho0), th, ze]), sort=False, jitable=True))
st = FourierRZToroidalSurface.from_values(np.column_stack([d["R"], ze, d["Z"]]), th, M=10, N=10, NFP=NFP, sym=True,
                                          check_orientation=False)
ss = eqs.surface.copy()
ss.change_resolution(M=10, N=10)  # the L12 equilibrium carries its fitted M=N=10 boundary padded to 12
assert np.array_equal(st.R_basis.modes, ss.R_basis.modes)
dR, dZ = ss.R_lmn - st.R_lmn, ss.Z_lmn - st.Z_lmn
say(f"--- boundary: traced surface minus target rho {rho0} surface (mm); total L2 R {np.linalg.norm(dR) * 1e3:.2f}, "
    f"Z {np.linalg.norm(dZ) * 1e3:.2f}")
for lab, dd, basis in (("R", dR, ss.R_basis), ("Z", dZ, ss.Z_basis)):
    o = np.argsort(-np.abs(dd))[:8]
    say(f"  {lab}: " + "  ".join(f"(m{basis.modes[i, 1]},n{basis.modes[i, 2]}) {dd[i] * 1e3:+.2f}" for i in o))
    nb = np.abs(basis.modes[:, 2])
    say(f"  {lab} L2 by |n|: " + "  ".join(f"{k}: {np.linalg.norm(dd[nb == k]) * 1e3:.2f}" for k in range(0, 11)))
    mb = np.abs(basis.modes[:, 1])
    say(f"  {lab} L2 by |m|: " + "  ".join(f"{k}: {np.linalg.norm(dd[mb == k]) * 1e3:.2f}" for k in range(0, 11)))
