"""Coil-count chain (QA nfp2, iota 0.42): a shortened version of the recipe behind the z = 0 best state, for nc coils per
half period. Stages: 1 first-order cold start (seed 2), 2 second order at weak weights, 3 second order at the final
weights (z = 0 result), 4 hinges within 0.2 a and tilt <= 2 degrees. Each stage takes the arguments of the matching
reference run. python ncchain.py NC   ->  runs2/nc{NC}_s1 .. _s4"""
import json, shlex, subprocess, sys, os
nc = sys.argv[1]
R = "runs2/"
stages = [(R + "s1_qa2_z0_s2", {"emax": "5.0", "wlen": "0.001", "wbend": "0.001", "maxiter": "600"}),
          (R + "o2_qa2s2_gg1e-2_w1e-3", {}),
          (R + "o2w1_qa2s2_gg1e-1_w1e-2_w1x10", {}),
          (R + "lo_qa2_i42_h0.0334_t2", {})]
prev = None
for i, (ref, over) in enumerate(stages, 1):
    out = f"{R}nc{nc}_s{i}"
    a = dict(json.load(open(ref + ".json"))["args"], nc=nc, out=out, **over)
    if prev:
        a["init"] = prev
        for k in ("seed", "axis0", "eta0", "rcoil"):
            a.pop(k, None)
    else:
        a.pop("init", None)
    prev = out
    if os.path.exists(out + ".json"):
        continue
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=5G", "-p", "MemorySwapMax=0", "python", "run_na2.py"]
    for k, v in a.items():
        if v is not None:
            cmd += [f"--{k}", str(v)]
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if subprocess.run(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT).returncode or not os.path.exists(out + ".json"):
        sys.exit(f"stage {i} failed: {out}.log")
