"""res 8 vs res 12: physics of the final equilibrium (single_stage.Physics) and the refined coils' RESULT line.
Also evaluates each equilibrium at the OTHER resolution's grids is not needed: upres_L12.log already gives the pure
discretization shift on one boundary."""
import re
import sys

sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox")
sys.path.insert(0, "/home/singh/Documents/DESC2/coil_sandbox/sandbox")
import boot  # noqa: E402

boot.setup("gpu")
import desc.io  # noqa: E402
import single_stage as SS  # noqa: E402

J = "work/ss_qh/joint"
CASES = [("res 8  (cont)", f"{J}/cont/k4b/eq_k4.h5", f"{J}/cont/refine/run.log", f"{J}/cont/k4b/run.log"),
         ("res 12 (hires)", f"{J}/hires/k4b/eq_k4.h5", f"{J}/hires/refine/run.log", f"{J}/hires/k4b/run.log")]
for tag, eqf, rlog, klog in CASES:
    e = desc.io.load(eqf)
    p = SS.Physics(e, "1,4")()
    print(f"{tag}  L,M,N {e.L},{e.M},{e.N}  V {p['V']:.5f}  {SS.fmt_phys(p)}")
    txt = open(klog).read()
    for pat in [r"k=4 solver.*\n.*", r"k=4 coils .*"]:
        m = re.findall(pat, txt)
        print("   ", m[-1].replace("\n", " | ") if m else "-")
    m = re.findall(r"^RESULT.*", open(rlog).read(), re.M)
    print("    refine", m[-1] if m else "-")
