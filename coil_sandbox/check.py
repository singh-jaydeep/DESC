"""Check the sandbox on a new machine: the intended DESC is the one imported (the main repo,
the parent of this sandbox, or vendor/ with SANDBOX_DESC=vendor), the pieces the scripts need
exist, and which device JAX sees.

    python check.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    env0 = dict(os.environ)
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, os.path.join(HERE, "sandbox"))
    import boot

    root = boot.desc_root()
    sys.path.insert(0, root)
    import inspect

    import desc

    where = os.path.realpath(os.path.dirname(desc.__file__))
    ok_path = where == os.path.realpath(os.path.join(root, "desc"))
    print(f"[{' ok ' if ok_path else 'FAIL'}] desc imported from {where}")

    import desc.objectives as O
    import desc.optimize._desc_wrappers  # noqa: F401
    import desc.optimize.aug_lagrangian_ls as AL
    from desc.coils import initialize_modular_coils  # noqa: F401

    feats = []
    try:
        from desc.coils import PiecewisePlanarArcCoil  # noqa: F401

        feats.append((True, "PiecewisePlanarArcCoil"))
    except Exception:
        feats.append((False, "PiecewisePlanarArcCoil"))
    for n in ("CoilMeanSquaredCurvature", "CoilSetDistancePenalty", "PlasmaCoilSetDistancePenalty"):
        feats.append((hasattr(O, n), n))
    if hasattr(O, "CoilSetDistancePenalty"):
        sig = inspect.signature(O.CoilSetDistancePenalty.__init__).parameters
        feats += [("max_active_pairs" in sig, "CoilSetDistancePenalty(max_active_pairs=)"),
                  ("signed" in sig, "CoilSetDistancePenalty(signed=)")]
    if root == boot.MAIN:  # pieces only the main repo has
        import desc.optimize as OPT

        for n in ("CoilSetDistanceRows", "PlasmaCoilSetDistanceRows", "PlasmaCoilDistanceField",
                  "CoilArclengthResidual"):
            feats.append((hasattr(O, n), n))
        for n in ("lsq_composite", "lsq_auglag_composite"):
            feats.append((hasattr(OPT, n), f"desc.optimize.{n}"))
    src = inspect.getsource(AL)
    feats += [(k in src, f"lsq-auglag: {k}") for k in ("second_order", "max_inner_iter", "y_update_gate")]
    bad = False
    for ok, n in feats:
        print(f"[{' ok ' if ok else 'FAIL'}] {n}")
        bad |= not ok

    env = {k: v for k, v in env0.items() if k != "JAX_PLATFORMS"}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        o = subprocess.run([sys.executable, "-c", "import jax;d=jax.devices();print(len(d), d[0].device_kind, d[0].platform)"],
                           capture_output=True, text=True, timeout=180, env=env)
        line = o.stdout.strip().splitlines()[-1]
        print(f"[{' ok ' if 'gpu' in line else 'WARN'}] jax devices: {line}"
              + ("" if "gpu" in line else "   (no GPU: pass --device cpu, or fix the jax install)"))
    except Exception as e:
        print(f"[WARN] could not query jax devices: {type(e).__name__}: {e}")
    print("\n" + ("SANDBOX NOT USABLE AS-IS" if bad or not ok_path else "SANDBOX OK"))
    raise SystemExit(1 if bad or not ok_path else 0)


if __name__ == "__main__":
    main()
