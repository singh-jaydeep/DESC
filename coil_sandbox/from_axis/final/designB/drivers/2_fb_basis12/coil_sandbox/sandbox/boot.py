"""Call `boot.setup()` FIRST in every script, before anything imports `desc`.

It does two things that only work if nothing has imported `desc` yet:

1. Puts the DESC checkout first on sys.path, so it shadows any other DESC installed on the
   machine. By default that is the main repo, the parent of this sandbox (`DESC2/`, branch
   `js/coil-auglag`, which has everything the scripts need plus the slack-free AL). Set
   SANDBOX_DESC=vendor to use the old snapshot in `<sandbox>/vendor` instead (to reproduce
   runs made before 2026-10-07).
2. Selects the device. JAX reads its platform once, when it is first imported, and a plain
   `import desc` defaults to CPU. The device comes from `--device gpu|cpu` on the command line,
   else the SANDBOX_DEVICE environment variable, else the script's default.
"""

import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor")
MAIN = os.path.dirname(ROOT)


def desc_root():
    """The directory that must come first on sys.path: the main repo, or vendor/."""
    which = os.environ.get("SANDBOX_DESC", "main")
    if which not in ("main", "vendor"):
        raise SystemExit(f"SANDBOX_DESC must be main or vendor, got {which!r}")
    return VENDOR if which == "vendor" else MAIN


def _device_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--device="):
            return a.split("=", 1)[1]
    return None


def setup(default="gpu"):
    if "desc" in sys.modules:
        warnings.warn(f"`desc` was imported (from {getattr(sys.modules['desc'], '__file__', '?')}) before "
                      f"boot.setup(); the chosen DESC and the device choice may not take effect.")
    elif desc_root() not in sys.path[:1]:
        sys.path.insert(0, desc_root())
    dev = _device_from_argv() or os.environ.get("SANDBOX_DEVICE") or default
    if dev not in ("gpu", "cpu"):
        raise SystemExit(f"--device must be gpu or cpu, got {dev!r}")
    if dev == "gpu":
        if os.environ.get("JAX_PLATFORMS", "") == "cpu":
            os.environ.pop("JAX_PLATFORMS")  # set_device("gpu") does not clear it
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # do not grab the whole card
    from desc import set_device

    set_device(dev)
    import jax

    d = jax.devices()
    import desc

    print(f"DESC {os.path.dirname(os.path.realpath(desc.__file__))}", flush=True)
    print(f"DEVICE requested={dev} obtained={d[0].platform} ({len(d)}x {d[0].device_kind})", flush=True)
    if dev == "gpu" and d[0].platform != "gpu":
        print(f"DEVICE WARNING: gpu requested but JAX is on {d[0].platform!r}", flush=True)
    return dev
