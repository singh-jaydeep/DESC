"""Acceptance: the Modal rescore of the best coils must reproduce the laptop's numbers. Exit 1 if not."""
import json, sys
s = json.load(open(sys.argv[1]))["free_boundary"]
REF = dict(qs=[4.245e-3, 4.288e-3, 8.449e-3, 1.750e-2], iota_axis=0.3562, iota_edge=0.3133, bn_rms=4.01e-4)
TOL = dict(qs=[0.05, 0.05, 0.05, 0.10], iota=0.003, bn_rms_max=6e-4)   # free boundary is path dependent: not bitwise
ok = True
for i, (v, r, t) in enumerate(zip(s["qs_booz"], REF["qs"], TOL["qs"])):
    good = abs(v / r - 1) <= t; ok &= good
    print(f"QS rho {(0.25, 0.5, 0.75, 1.0)[i]}: {v:.3e} vs laptop {r:.3e} ({100 * (v / r - 1):+.1f}%, tol {100 * t:.0f}%) {'ok' if good else 'FAIL'}")
for k in ("iota_axis", "iota_edge"):
    good = abs(s[k] - REF[k]) <= TOL["iota"]; ok &= good
    print(f"{k}: {s[k]:.4f} vs {REF[k]:.4f} {'ok' if good else 'FAIL'}")
good = s["bn_rms"] <= TOL["bn_rms_max"]; ok &= good
print(f"B.n rms: {s['bn_rms']:.2e} (laptop {REF['bn_rms']:.2e}, must be <= {TOL['bn_rms_max']:.0e}) {'ok' if good else 'FAIL'}")
m = json.load(open(sys.argv[2]))
print("BENCHMARK step minutes:", [st.get("minutes") for st in m["attempts"][-1]["steps"]], "(laptop: shells 4+8 ~125 min)", "gpu", m["attempts"][-1]["gpu"])
print("ACCEPTANCE", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
