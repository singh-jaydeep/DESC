"""Select the JAX device from the command line before anything imports DESC's backend.

    import device; device.setup(default="gpu")   # first, before numpy-heavy DESC imports

Reads and removes ``--device gpu|cpu`` from ``sys.argv`` (so scripts need not declare it)
and calls ``desc.set_device``. JAX fixes its platform at first use, so this must run
before any other DESC import.
"""

import sys


def setup(default="gpu"):
    """Set the device from ``--device`` (removed from argv), else ``default``."""
    device = default
    if "--device" in sys.argv:
        i = sys.argv.index("--device")
        device = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    from desc import set_device

    set_device(device)
    return device
