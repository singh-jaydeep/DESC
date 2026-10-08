"""Make the bundle's VENDORED DESC win over any other DESC on the machine.

Import this before anything imports `desc`. Every entry point in `tools/` does.

WHY. The sweep depends on DESC code that is not in master -- the arc representation, the
mean-square curvature constraint, unsigned `|curvature|`, signed coil-coil distance, and a
heavily modified augmented Lagrangian (`fixed_configs.md` sec 11). If a different DESC is
installed on the machine, `import desc` finds it and the sweep either fails to import or,
worse, runs against different code and produces numbers that look fine.

That is not hypothetical: on the machine this bundle was built on, DESC is installed from
the source repo, so an unaided `import desc` resolved to the repo copy rather than to
`vendor/desc`. The two happen to agree there. On any other machine they will not.

This prepends `<bundle>/vendor` to `sys.path`, so `vendor/desc` shadows an installed one.
It is a no-op if the vendored copy is absent, so the tools still work in-tree.
"""

import os
import sys
import warnings

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(BUNDLE, "vendor")


def _activate():
    if not os.path.isdir(os.path.join(VENDOR, "desc")):
        return None  # running in-tree, not from a bundle
    if "desc" in sys.modules:
        where = getattr(sys.modules["desc"], "__file__", "?")
        warnings.warn(
            f"`desc` was already imported from {where} before the bundle could "
            f"prepend {VENDOR}. The vendored DESC is NOT in use. Import "
            f"tools/_bundle.py first.",
            stacklevel=2,
        )
        return None
    if VENDOR not in sys.path[:1]:
        sys.path.insert(0, VENDOR)
    return VENDOR


ACTIVE = _activate()
