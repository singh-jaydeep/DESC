"""Pure measurement core -- deliberately free of any ``modal`` import.

This module is executed two ways: imported inside a Modal container, and run as
``python -m modal_bench._bench_core '<json cfg>'`` in a subprocess of that
container. The subprocess route is what gives each measurement a clean jit cache
and a clean peak-memory high-water mark while keeping every method at a given
``n`` on the SAME physical GPU.
"""

import json
import os
import sys


def init_gpu(mem_fraction="0.95", deterministic=False):
    """Select the GPU BEFORE anything imports jax, then verify it took.

    ``desc/backend.py`` runs ``set_device("cpu")`` at import time whenever no
    device has been chosen yet, and that sets ``JAX_PLATFORMS=cpu`` and
    ``CUDA_VISIBLE_DEVICES=""`` before it goes on to import jax. Since jax binds
    its backend lazily on first use, any process that reaches ``desc.backend``
    without choosing a device first ends up on CPU -- silently, with
    ``memory_stats()`` returning None and timings that look like plausible GPU
    numbers but are not.

    ``deterministic`` sets ``--xla_gpu_deterministic_ops=true``, which forces XLA
    to avoid nondeterministic reductions/atomics. Repeated identical solves were
    measured landing up to 41x apart in final cost, so this is the lever for
    testing whether that noise is GPU nondeterminism amplified by the
    ``actual_reduction > 0`` acceptance branch.

    ``mem_fraction`` is raised from XLA's 0.75 default because this benchmark is
    partly about WHERE EACH METHOD RUNS OUT OF MEMORY. At the default, an A100
    80GB reports a 59.4 GB limit, so "does not fit on an A100" would mean XLA's
    ceiling rather than the card's.
    """
    # Surface allocation failures as real OOM rather than letting XLA's default
    # preallocation mask them, and keep peak_bytes_in_use meaningful.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    if deterministic:
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_deterministic_ops=true").strip()

    import desc

    desc.set_device("gpu")

    import jax

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"expected gpu backend, got {jax.default_backend()!r}; "
            f"devices={jax.devices()}"
        )
    return jax


def build_R(n, cond, seed):
    """Upper triangular R with a prescribed diagonal spectrum.

    Conditioning is set by the diagonal (log-spaced over ``cond``); the strict
    upper part is small random noise. Shapes -- hence time and memory -- are
    exactly those of a real reduced R. Values are NOT those of a real R, so the
    Gram residual reported here is indicative only and must be confirmed against
    solve-captured R separately.
    """
    import jax
    import numpy as np

    from desc.backend import jnp

    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    G = jax.random.normal(k1, (n, n), dtype=jnp.float64) * (1e-3 / np.sqrt(n))
    d = jnp.logspace(0, -np.log10(cond), n, dtype=jnp.float64)
    R = jnp.triu(G, 1) + jnp.diag(d)
    z = jax.random.normal(k2, (n,), dtype=jnp.float64)
    return R, z


def gram_probe(R, Rtil, alpha, seed, n_probe=8):
    """Relative Gram residual via random probe vectors -- O(n^2), not O(n^3).

    Checks ``Rtil.T@Rtil == R.T@R + alpha*I`` along ``n_probe`` random unit
    directions. Cheap enough to run at every n, including sizes where forming the
    full Gram matrix would not fit.
    """
    import jax

    from desc.backend import jnp

    n = R.shape[1]
    V = jax.random.normal(
        jax.random.PRNGKey(seed + 99), (n, n_probe), dtype=jnp.float64
    )
    V = V / jnp.linalg.norm(V, axis=0, keepdims=True)
    lhs = Rtil.T @ (Rtil @ V)
    rhs = R.T @ (R @ V) + alpha * V
    return float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))


def measure_one(cfg):
    """One alpha factorization: time, peak device memory, Gram residual.

    The unit measured is exactly the body of the LM alpha loop -- the only thing
    that differs between ``tr_method="qr"`` (master) and the structured routes.
    """
    import time

    import numpy as np

    jax = init_gpu(
        cfg.get("mem_fraction", "0.95"), deterministic=cfg.get("deterministic", False)
    )

    from desc.backend import jnp, qr_multiply
    from desc.optimize.tr_subproblems import structured_retriangularize_fixed

    method, n, block = cfg["method"], cfg["n"], cfg.get("block")
    alpha, reps, cond = cfg["alpha"], cfg["reps"], cfg["cond"]
    dev = jax.local_devices()[0]
    out = dict(cfg, device=str(dev.device_kind), jax=jax.__version__)

    R, z = build_R(n, cond, cfg.get("seed", 0))
    jax.block_until_ready(R)
    out["R_GB"] = (dev.memory_stats() or {}).get("bytes_in_use", 0) / 1024**3

    if method == "qr":  # master's dense route, verbatim from the loop body
        zp = jnp.concatenate([z, jnp.zeros(n)])

        @jax.jit
        def fac(R, zp, alpha):
            A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
            Qtz, Rtil = qr_multiply(A, zp, mode="right")
            return Rtil, Qtz

        call = lambda: fac(R, zp, alpha)  # noqa: E731
    elif method == "qr-fixed":
        call = lambda: structured_retriangularize_fixed(
            R, z, alpha, block=block
        )  # noqa: E731
    else:
        raise ValueError(f"unknown method {method}")

    t0 = time.perf_counter()
    Rtil, Qtz = jax.block_until_ready(call())
    out["compile_s"] = time.perf_counter() - t0

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        ts.append(time.perf_counter() - t0)
    out["times_s"] = ts
    out["time_s"] = float(np.median(ts))
    out["time_spread"] = float((max(ts) - min(ts)) / np.median(ts))

    st = dev.memory_stats() or {}
    out["peak_GB"] = st.get("peak_bytes_in_use", 0) / 1024**3
    out["limit_GB"] = st.get("bytes_limit", 0) / 1024**3
    out["gram_rel"] = gram_probe(R, Rtil, alpha, cfg.get("seed", 0))
    out["dtype"] = str(Rtil.dtype)
    out["ok"] = True
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    try:
        res = measure_one(cfg)
    except Exception as e:  # OOM included: report it, do not crash the sweep
        import traceback

        res = dict(
            cfg,
            ok=False,
            error=f"{type(e).__name__}: {str(e)[:400]}",
            traceback=traceback.format_exc()[-800:],
        )
    print("RESULT " + json.dumps(res), flush=True)
