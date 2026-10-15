"""Print the run_na2.py command that continues a saved run with the same args plus overrides.
python cont.py runs2/s1_qa2_z0_s2 OUT [--key value ...]"""
import json, sys, shlex
src, out, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
a = json.load(open(src + ".json"))["args"]
skip = {"out", "init", "seed", "axis0", "eta0", "rcoil"}
over = {extra[i].lstrip("-").replace("-", "_"): extra[i + 1] for i in range(0, len(extra), 2)}
cmd = ["python", "run_na2.py", "--init", src, "--out", out]
for k, v in a.items():
    if k in skip or k in over:
        continue
    cmd += [f"--{k}", str(v)]
for k, v in over.items():
    cmd += [f"--{k}", v]
print(" ".join(shlex.quote(c) for c in cmd))
