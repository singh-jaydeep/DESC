"""Table of the bound-weakening scan (loose2.sh): field cost relative to the unweakened state at the same hinge setting."""
import json, glob, os
base = {"z0": "runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10", "lh": "runs2/lo_qa2_i42_h0.0334_t2"}
print(f"{'hinge':5s} {'knob':>12s} | {'cost':>7s} {'/base':>5s} {'iota':>6s} {'B20':>6s} {'dGG':>6s} {'elong':>5s} | {'dpc':>6s} {'dcc':>6s} {'kmax':>6s} {'Ltot':>6s} nit status")
for S, b in base.items():
    f0 = json.load(open(b + ".json"))["final"]
    rows = [("base", f0)]
    for f in sorted(glob.glob(f"runs2/l2_qa2_{S}_*.json")):
        rows.append((os.path.basename(f)[len(f"l2_qa2_{S}_"):-5], json.load(open(f))["final"]))
    cold = next((d for n, d in rows if n == "cold_nc4"), None)
    for n, d in rows:
        ref = cold if n.startswith("cold") and cold else f0
        print(f"{S:5s} {n:>12s} | {d['cost_field']:7.4f} {d['cost_field'] / ref['cost_field']:5.2f} {d['iota']:6.3f} "
              f"{d.get('B20_residual', float('nan')):6.3f} {d.get('max_dGG_rel', float('nan')):6.3f} {d['max_elong']:5.2f} | "
              f"{d['min_dpc']:6.4f} {d['min_dcc']:6.4f} {d['max_kappa']:6.2f} {d['L_total']:6.2f} {d.get('nit')} {d['status'][:28]}")
