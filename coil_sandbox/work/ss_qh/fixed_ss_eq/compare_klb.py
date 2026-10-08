"""A/B of the CoilCurvature lower bound on the planar restart: planarN7_trinf (bounds (0, k)) vs
planarN7_klb (bounds (-inf, k)), same start and settings. Per 25-iteration window: inner optimality,
constraint violation, slack-gradient sign flips per accepted step, and watched kappa rows near the lower bound."""
import json, re, sys
import numpy as np

HERE = "work/ss_qh/fixed_ss_eq"
RUNS = [("lb 0", f"{HERE}/planarN7_trinf"), ("lb -inf", f"{HERE}/planarN7_klb")]
W = 25
row = re.compile(r"^\s+(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")


def solver_rows(path):
    out = {}
    for line in open(f"{path}/run.log"):
        m = row.match(line)
        if m:  # it, cost, cost_red, step, optimality, viol, mu, ymax, gtolk
            out[int(m.group(1))] = dict(cost=float(m.group(2)), opt=float(m.group(5)), viol=float(m.group(6)),
                                        ymax=float(m.group(8)))
    return out


def diag(path):
    acc = {}
    for line in open(f"{path}/diag.jsonl"):
        r = json.loads(line)
        if r["kind"] != "accept":
            continue
        lb = sum(1 for k, v in (r.get("watched") or {}).items()
                 if "coil curvature" in k and np.isfinite(v.get("d_lb", np.inf)) and v["d_lb"] < 2e-3)
        acc[r["iteration"]] = dict(flips=r["n_sign_flips"], lb=lb)
    return acc


data = {lab: (solver_rows(p), diag(p)) for lab, p in RUNS}
last = min(max(s) for s, _ in data.values())
print(f"windows of {W} iterations, up to it {last}\n")
print(f"{'it':>9} | " + " | ".join(f"{lab:^44}" for lab, _ in RUNS))
print(f"{'':>9} | " + " | ".join(f"{'opt med/min':>14} {'viol':>8} {'flips med':>9} {'lb rows':>8}" for _ in RUNS))
for a in range(0, last + 1, W):
    cells = []
    for lab, _ in RUNS:
        s, d = data[lab]
        it = [i for i in range(a, a + W) if i in s and i >= 1]
        if not it:
            cells.append(" " * 44); continue
        o = np.array([s[i]["opt"] for i in it])
        fl = [d[i]["flips"] for i in it if i in d]
        lb = [d[i]["lb"] for i in it if i in d]
        cells.append(f"{np.median(o):.1e}/{o.min():.1e} {s[it[-1]]['viol']:.1e} {np.median(fl) if fl else 0:>9.0f} "
                     f"{np.mean(lb) if lb else 0:>8.1f}")
    print(f"{a:>4}-{a + W - 1:<4} | " + " | ".join(cells))
for lab, p in RUNS:
    print(lab, "dense checks:", [l.strip() for l in open(f"{p}/run.log") if "[it " in l and "] bn" in l][-3:])
