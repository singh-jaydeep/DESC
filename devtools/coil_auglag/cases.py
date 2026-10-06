"""Dispatch to a problem builder by the `case` key of the run config."""


def build(case="qa", **kw):
    """Build (eq, coilset, obj, cons) for the named case."""
    if case == "qh":
        from setup_qh import build as b
    else:
        from setup import build as b
    return b(**kw)
