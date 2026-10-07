"""Dispatch to a problem builder by the `case` key of the run config."""


def build(case="qa", **kw):
    """Build (eq, coilset, obj, cons) for the named case."""
    if case == "qh":
        from setup_qh import build as b
    elif kw.pop("al", False):  # run_al's node-mode defaults do not apply
        from setup import build_al as b

        kw.pop("distance", None)
        kw = {k: v for k, v in kw.items() if not (k == "pair_N" and v is None)}
    else:
        from setup import build as b
    return b(**kw)
