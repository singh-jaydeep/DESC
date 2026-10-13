"""Table of the loose-hinge scan (loose.sh): field cost relative to the z = 0 state at the same iota target."""
import json, glob, os
base = {"30": "runs2/li_qa2_iota0.30", "42": "runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10", "50": "runs2/hi_qa2_iota0.50"}
A = 0.167
print(f"{'iota tgt':8s} {'H/a':>5s} {'tilt':>6s} | {'cost':>7s} {'/z0':>5s} {'iota':>6s} {'B20':>6s} {'dGG':>6s} {'elong':>5s} | max|z| max tilt(deg) nit status")
for I, b in base.items():
    if not os.path.exists(b + ".json"):
        continue
    f0 = json.load(open(b + ".json"))["final"]
    rows = [("0", "z=0", f0)]
    for f in sorted(glob.glob(f"runs2/lo_qa2_i{I}_h*_t*.json")):
        H, T = f.split("_h")[1].split("_t")[0], f.split("_t")[-1][:-5]
        rows.append((f"{float(H) / A:.1f}", T, json.load(open(f))["final"]))
    for H, T, d in rows:
        tl = max(abs(v) for v in d.get("hinge_tilt_deg", [0]))
        print(f"0.{I:6s} {H:>5s} {T:>6s} | {d['cost_field']:7.4f} {d['cost_field'] / f0['cost_field']:5.2f} {d['iota']:6.3f} "
              f"{d.get('B20_residual', float('nan')):6.3f} {d.get('max_dGG_rel', float('nan')):6.3f} {d['max_elong']:5.2f} | "
              f"{d.get('max_abs_hinge_z', 0):.4f} {tl:5.2f} {d.get('nit')} {d['status'][:28]}")
