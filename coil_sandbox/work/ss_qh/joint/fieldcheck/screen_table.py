"""Table of the proxy-QS screen (spectrum/*.json), sorted by coil-field QS at rho 0.5."""
import glob
import json
import re

rows = []
for f in glob.glob("spectrum/*.json"):
    d = json.load(open(f))
    if d["name"] in ("level", "free"):
        continue
    t = open(f.replace(".json", ".txt")).read()
    bn = float(re.search(r"<\|B\.n\|>/B0 ([0-9.e+-]+)", t).group(1))
    p = {float(k): v for k, v in d["proxy"].items()}
    rows.append((p[0.5]["coil"], d["name"], bn, p))
rows.sort()
print("| coilset | <|B.n|>/B | QS coil 0.25 | **0.5** | 0.75 | 1.0 | QS target 0.5 | 1.0 | coil/target 0.5 | ripple 6-9 at 0.5 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for q, n, bn, p in rows:
    print(f"| {n} | {bn:.2e} | {p[0.25]['coil']:.2e} | **{q:.2e}** | {p[0.75]['coil']:.2e} | {p[1.0]['coil']:.2e} | "
          f"{p[0.5]['target']:.2e} | {p[1.0]['target']:.2e} | {q / p[0.5]['target']:.1f} | {p[0.5]['ripple']:.2e} |")
