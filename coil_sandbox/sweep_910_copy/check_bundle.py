"""Verify this bundle is intact and that the DESC being imported is the vendored one.

    python check_bundle.py            # from the bundle root
    python check_bundle.py --deep     # also rebuild the starts and compare hashes

Run this FIRST on any new machine. It is cheap and it catches the failure that would
otherwise be found halfway through a 41-hour sweep: running against a different DESC.

WHY THIS MATTERS HERE. The sweep depends on DESC code that is **not in master** -- the
whole arc representation, the mean-square curvature constraint, unsigned `|curvature|`,
signed coil-coil distance, and a heavily modified augmented Lagrangian (see
`fixed_configs.md` sec 11). Against stock DESC the sweep does not merely give different
numbers: most of it does not import.

The vendored `vendor/desc/` is a snapshot of the WORKING TREE, not of any commit. Six
files under `desc/` had uncommitted changes when it was taken, so checking out the
recorded `head` does **not** reproduce it. `vendor/` is authoritative.
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "vendor", "MANIFEST.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deep", action="store_true",
                    help="also rebuild the cold starts and compare to starts.json")
    a = ap.parse_args()

    if not os.path.exists(MAN):
        raise SystemExit(f"no manifest at {MAN} -- this is not a complete bundle")
    man = json.load(open(MAN))
    prov = man["provenance"]
    print(f"bundle {man['bundle']}  created {man['created']}  eq {man['equilibrium']}")
    print(f"  branch {prov['branch']}  head {prov['head'][:12]}")
    print(f"  {len(man['desc_changed_vs_master'])} DESC files differ from master "
          f"({prov['master'][:12]})")
    n_dirty = len(prov["desc_uncommitted_at_vendor_time"])
    print("  " + (f"{n_dirty} were uncommitted when vendored -- vendor/ is "
                  "authoritative, NOT the commit\n" if n_dirty else
                  "vendor/desc is a clean checkout of that commit\n"))

    bad, missing = [], []

    # 1. every vendored + tool file present and unmodified
    for group, base in (("vendor_sha256", os.path.join(HERE, "vendor")),
                        ("tools_sha256", HERE)):
        for rel, want in man[group].items():
            p = os.path.join(base, rel)
            if not os.path.exists(p):
                missing.append(rel)
            elif sha(p) != want:
                bad.append(rel)
    n = len(man["vendor_sha256"]) + len(man["tools_sha256"])
    print(f"[{'FAIL' if bad or missing else ' ok '}] file integrity: {n} files")
    for r in missing[:10]:
        print(f"        MISSING  {r}")
    for r in bad[:10]:
        print(f"        MODIFIED {r}")

    # 2. the data files the sweep needs
    need = ["sweep_grid.json", "RUNS.md", "fixed_configs.md",
            os.path.join("initial starts", "starts.json")]
    gone = [f for f in need if not os.path.exists(os.path.join(HERE, f))]
    print(f"[{'FAIL' if gone else ' ok '}] bundle data: {len(need)-len(gone)}/{len(need)}"
          + (f"  MISSING {gone}" if gone else ""))
    starts = os.path.join(HERE, "initial starts")
    h5 = [f for f in os.listdir(starts) if f.endswith(".h5")] if os.path.isdir(starts) else []
    print(f"[{' ok ' if len(h5) == 7 else 'FAIL'}] cold starts: {len(h5)}/7 h5 files")

    # 3. which DESC actually imports
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, os.path.join(HERE, "vendor"))
    try:
        import desc
    except Exception as e:  # pragma: no cover
        print(f"[FAIL] import desc: {e}")
        raise SystemExit(1)
    where = os.path.dirname(os.path.abspath(desc.__file__))
    vend = os.path.join(HERE, "vendor", "desc")
    same = os.path.realpath(where) == os.path.realpath(vend)
    print(f"[{' ok ' if same else 'WARN'}] desc imported from {where}")
    if not same:
        print("        NOT the vendored copy. Either run from the bundle root, or set")
        print(f"        PYTHONPATH={vend!r} -- otherwise the sweep uses different code.")

    # 3b. what compute is actually available.
    # In a SUBPROCESS, because this process set JAX_PLATFORMS=cpu above to keep the
    # integrity checks cheap -- querying jax here would always report "no GPU".
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import jax,json;d=jax.devices();"
             "print(json.dumps([len(d),d[0].device_kind,"
             "sorted({x.platform for x in d})]))"],
            capture_output=True, text=True, timeout=180, env=env)
        n, kind, plats = json.loads(out.stdout.strip().splitlines()[-1])
        gpu = "gpu" in plats or "cuda" in plats
        print(f"[{' ok ' if gpu else 'WARN'}] jax devices: {n}x {kind} "
              f"({', '.join(plats)})")
        if not gpu:
            print("        No GPU visible. The generated commands default to "
                  "--device gpu; they will")
            print("        fall back to CPU at roughly half speed. Either fix the "
                  "install or regenerate")
            print("        with: python tools/sweep_configs.py --device cpu --json "
                  "sweep_grid.json")
    except Exception as e:
        print(f"[WARN] could not query jax devices: {type(e).__name__}: {e}")
        gpu = None

    # 4. the non-standard pieces the sweep cannot run without
    print("\n[ .. ] non-standard DESC features (fixed_configs.md sec 11):")
    feats, fail = [], False
    try:
        from desc.coils import PiecewisePlanarArcCoil  # noqa: F401
        feats.append((" ok ", "PiecewisePlanarArcCoil", "the arc family"))
    except Exception:
        feats.append(("FAIL", "PiecewisePlanarArcCoil", "the arc family")); fail = True
    try:
        from desc.objectives import CoilMeanSquaredCurvature  # noqa: F401
        feats.append((" ok ", "CoilMeanSquaredCurvature", "the kappa_MS constraint"))
    except Exception:
        feats.append(("FAIL", "CoilMeanSquaredCurvature", "the kappa_MS constraint")); fail = True
    import inspect
    from desc.objectives import CoilSetMinDistance, CoilCurvature
    sig = inspect.signature(CoilSetMinDistance.__init__).parameters
    for kw, why in (("distance_method", "segment distances"), ("signed", "topology guard"),
                    ("pair_mode", "per-coil reduction")):
        ok = kw in sig
        feats.append((" ok " if ok else "FAIL", f"CoilSetMinDistance({kw}=)", why))
        fail |= not ok
    doc = (CoilCurvature.__doc__ or "")
    ok = "magnitude" in doc or "unsigned" in doc
    feats.append((" ok " if ok else "FAIL", "CoilCurvature -> |curvature|",
                  "master targets SIGNED curvature"))
    fail |= not ok
    src = inspect.getsource(sys.modules["desc.optimize.aug_lagrangian_ls"])
    for kw in ("max_inner_iter", "track_outer", "y_update_gate"):
        ok = kw in src
        feats.append((" ok " if ok else "FAIL", f"lsq-auglag option {kw}", ""))
        fail |= not ok
    for st, name, why in feats:
        print(f"        [{st}] {name:38s} {why}")

    if a.deep:
        print("\n[ .. ] deep check: rebuilding the seven cold starts")
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import subprocess
        rc = subprocess.call([sys.executable, os.path.join(HERE, "tools",
                                                           "dump_initial_starts.py"),
                              os.path.join(HERE, "initial starts"), "--check"])
        print(f"[{' ok ' if rc == 0 else 'FAIL'}] starts reproduce their pinned hashes")
        fail |= rc != 0

    problems = bool(bad or missing or gone or len(h5) != 7 or fail)
    print("\n" + ("BUNDLE NOT USABLE AS-IS -- see above" if problems
                  else "BUNDLE OK -- ready to run"))
    if not problems:
        print("\nnext:  python tools/run_sweep.py sweep_grid.json --dry-run")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
