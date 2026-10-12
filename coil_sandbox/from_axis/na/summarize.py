"""Table of finished near-axis runs: python summarize.py [glob]  (default runs2/s1_*.json)."""
import glob, json, sys

pat = sys.argv[1] if len(sys.argv) > 1 else "runs2/s1_*.json"
rows = []
for f in sorted(glob.glob(pat)):
    d = json.load(open(f))
    fi, ar = d["final"], d["args"]
    feas = (fi["min_dpc"] >= 0.995 * ar["dpc"] and fi["min_dcc"] >= 0.995 * ar["dcc"]
            and fi["max_kappa"] <= 1.005 * ar["kmax"] and max(fi["Lcoil"]) <= 1.005 * ar["Lmax"]
            and min(fi["I_over_uniform"]) >= -1e-3 and max(fi["I_over_uniform"]) <= ar["imax"] + 1e-3)
    rows.append((f.split("/")[-1][:-5], ar["nfp"], ar["nc"], fi["hel"], ar["iota"], fi["iota"], fi["cost"],
                 fi["max_dB"], fi["max_dG_rel"], fi["max_elong"], fi["max_abs_Zaxis"], fi["axis_rc"][1],
                 fi["axis_zs"][1], fi["eta"], min(fi["I_over_uniform"]), max(fi["I_over_uniform"]), feas,
                 fi.get("optimality") or float("nan"), fi.get("nit")))
hdr = ("run", "nfp", "nc", "hel", "iota0", "iota", "cost", "dB", "dG", "elong", "|Z|ax", "R1", "Z1", "eta",
       "Imin", "Imax", "feas", "opt", "nit")
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join([r[0], str(r[1]), str(r[2]), f"{r[3]:+.0f}", f"{r[4]:.2f}", f"{r[5]:.3f}",
                             f"{r[6]:.2e}", f"{r[7]:.1%}", f"{r[8]:.1%}", f"{r[9]:.1f}", f"{r[10]:.3f}",
                             f"{r[11]:.3f}", f"{r[12]:.3f}", f"{r[13]:.2f}", f"{r[14]:.2f}", f"{r[15]:.2f}",
                             "yes" if r[16] else "NO", f"{r[17]:.1e}", str(r[18])]) + " |")
