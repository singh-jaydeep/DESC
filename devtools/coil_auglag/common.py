"""Shared loaders for the crawl investigation scripts."""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
jax.config.update("jax_enable_x64", True)
from cases import build  # noqa: E402

from desc.objectives import ObjectiveFunction  # noqa: E402
from desc.objectives.getters import maybe_add_self_consistency  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402


def load_run(tag, extra_kw=None):
    """Rebuild a saved run's problem and its iterates in reduced coordinates."""
    kw = json.load(open(f"{tag}.json"))
    kw.update(extra_kw or {})
    eq, c0, obj, cons = build(**kw)
    cons = maybe_add_self_consistency(c0, cons)
    proj = LinearConstraintProjection(obj, ObjectiveFunction(cons))
    proj.build(verbose=0)
    X = np.load(f"{tag}_allx.npy")
    Y = np.array([np.asarray(proj.project(jnp.asarray(x))) for x in X])
    return obj, proj, Y


def groups(obj):
    """Return objective names and their row slices in the residual vector."""
    names, sl, i = [], [], 0
    for o in obj.objectives:
        names.append(o.name[:14])
        sl.append(slice(i, i + o.dim_f))
        i += o.dim_f
    return names, sl


def phase_tangent(p, modes):
    """d/dc of FourierXYZ coefficients under s -> s + c (cos for n>0, sin for n<0)."""
    t = {}
    for key in ("X_n", "Y_n", "Z_n"):
        a = np.asarray(p[key])
        out = np.zeros_like(a)
        for i, n in enumerate(modes):
            if n == 0:
                continue
            j = int(np.where(modes == -n)[0][0])
            out[j] += (-abs(n) if n > 0 else abs(n)) * a[i]
        t[key] = out
    return t


def load_phase(tag, yfile, extra_kw=None):
    """Rebuild `tag` at yfile with the parameter phase of each coil pinned.

    yfile holds reduced coordinates of `tag`, or a full state if it ends in _x.npy.
    """
    from desc.objectives import FixParameters

    kw = json.load(open(f"{tag}.json"))
    kw.update(extra_kw or {})
    _, _, proj0 = None, None, None
    if yfile.endswith("_x.npy"):  # full-state vector
        obj0 = build(**json.load(open(f"{tag}.json")))[2]
        obj0.build(verbose=0)
        x_full = np.load(yfile)
    else:
        obj0, proj0, _ = load_run(tag)
        x_full = np.asarray(proj0.recover(jnp.asarray(np.load(yfile))))
    pd = obj0.unpack_state(x_full, False)[0]
    eq, c0, obj, cons = build(**kw)
    obj.build(verbose=0)  # normalizations from initial coils
    c0.params_dict = [{k: np.asarray(v) for k, v in p.items()} for p in pd]
    modes = c0[0].X_basis.modes[:, 2]
    fix = []
    for p in pd:
        t = phase_tangent(p, modes)
        best = max(
            ((k, i) for k in t for i in np.where(np.abs(modes) == 1)[0]),
            key=lambda ki: abs(t[ki[0]][ki[1]]),
        )
        fix.append({best[0]: np.array([best[1]])})
    cons = cons + (FixParameters(c0, fix),)
    cons = maybe_add_self_consistency(c0, cons)
    for c in cons:
        c.build(verbose=0)
    proj = LinearConstraintProjection(obj, ObjectiveFunction(cons))
    proj.build(verbose=0)
    y = np.asarray(proj.project(jnp.asarray(c0.pack_params(c0.params_dict))))
    return obj, proj, y, fix
