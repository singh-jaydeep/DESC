"""Make a run tag at a given full state, so ctr.py can start from it.

usage: python mkstage.py TAG X_FILE KW_JSON
Writes TAG.json (the builder kwargs) and TAG_allx.npy ([x]). Start from the
circular coils with X_FILE = init (the builder's own coils).
"""

import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cases import build  # noqa: E402

tag, xfile, kw = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
_, cs, obj, _ = build(**kw)
obj.build(verbose=0)
x = np.asarray(obj.x(cs)) if xfile == "init" else np.load(xfile)
json.dump(kw, open(f"{tag}.json", "w"))
np.save(f"{tag}_allx.npy", x[None])
print(", ".join(f"{o.name}: {o.dim_f} rows" for o in obj.objectives))
