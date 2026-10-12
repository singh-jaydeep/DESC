"""Pull finished (or running) jobs from the Volume into ../ss/cloud_runs/<name>/ and verify them.
    python harvest.py NAME [NAME ...]        (needs the modal CLI and token)
Nothing is deleted from the Volume. A job counts as safely harvested only if this prints VERIFIED for it."""
import glob, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME = os.environ.get("FA_VOLUME", "from-axis-runs")
DEST = os.path.join(HERE, "..", "ss", "cloud_runs")
os.makedirs(DEST, exist_ok=True)
import h5py
for name in sys.argv[1:]:
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, name, DEST], check=True)
    d = os.path.join(DEST, name)
    bad = []
    for f in glob.glob(os.path.join(d, "**", "*.h5"), recursive=True):
        if os.path.basename(f).startswith("tmp_"):
            continue
        try:
            with h5py.File(f, "r") as h:
                len(h.keys())
        except Exception as e:  # noqa: BLE001
            bad.append((f, repr(e)))
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    done = os.path.exists(os.path.join(d, "DONE"))
    n = len(glob.glob(os.path.join(d, "**", "*"), recursive=True))
    print(f"{name}: {n} files, DONE={done}, attempts={len(meta.get('attempts', []))}, commit={meta.get('attempts', [{}])[-1].get('commit')}")
    for f in glob.glob(os.path.join(d, "**", "score.json"), recursive=True):
        s = json.load(open(f)); fb = s.get("free_boundary", {})
        print("   ", os.path.relpath(f, d), "QS", fb.get("qs_booz") or fb.get("qs_booz_rho.25,.5,.75,1"), "iota", fb.get("iota_axis"), fb.get("iota_edge"), "B.n", fb.get("bn_rms"))
    print("   ", "VERIFIED" if (done and not bad) else f"NOT VERIFIED (done={done}, unreadable h5: {bad})")
