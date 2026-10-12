"""Plumbing test for app.py: the shipped DESC imports, the device is what was asked, {OUT} is writable."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sandbox"))
import boot; dev = boot.setup(default=os.environ.get("SANDBOX_DEVICE", "cpu"))
import numpy as np, jax, desc
out, tag = sys.argv[1], sys.argv[2]
from desc.backend import jnp  # what every DESC script uses: enables x64
x = jnp.linalg.svd(jnp.asarray(np.random.default_rng(0).standard_normal((300, 200))), compute_uv=False)
info = dict(tag=tag, device=str(dev), jax=jax.__version__, desc=desc.__file__, x64=bool(jax.config.jax_enable_x64),
            dtype=str(x.dtype), svd0=float(x[0]), time=time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(info, open(os.path.join(out, f"smoke_{tag}.json"), "w"), indent=1)
print("SMOKE", json.dumps(info), flush=True)
