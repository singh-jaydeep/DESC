"""QUASR (quasr.flatironinstitute.org) -> DESC.

371,701 vacuum QA/QH devices, each with an optimized FourierXYZ coil set. The site's REST
scheme is undocumented; it was read off the navigator's JS bundle and every endpoint here is
verified. `id = str(ID).zfill(7)`, `p = id[:4]`:

    database.json.gz                        the whole index, 37 MB, pandas "split" orient
    nml/{p}/input.{id}                      VMEC namelist, boundary only (RBC/ZBS to M=N=10)
    simsopt_serials/{p}/serial{id}.json     simsopt graph: coils, currents, target surfaces
    records/{p}/{id}.json                   that device's index row alone

Two traps this module exists to absorb:

* The namelist has no MPOL/NTOR, so `InputReader` dies with "M_pol is not assigned". We
  prepend them (plus PHIEDGE/NCURR) before handing it to DESC. DESC's own reader is then
  EXACT -- a, A and V match the index row to ~1e-6 -- so do not hand-parse RBC/ZBS.
* The coil dof ORDER is not guessed: every simsopt DOFs object carries `names`
  ("xc(0)", "xs(1)", "xc(1)", ...), so we read the ordering off the file. The base curves
  are then resampled and refitted by DESC, which sidesteps simsopt's RotatedCurve sign
  conventions entirely -- `CoilSet.from_symmetry` rebuilds the copies, and
  `check_reconstruction` confirms that against five numbers in the index row.

Scale: QUASR normalises to R0 ~ 1 m, B0 ~ 1 T, so minor radius is 0.04 .. 0.36 m. Bounds
must therefore be a-scaled (`common.paper_bounds`), never absolute.
"""

import gzip
import json
import os
import re
import urllib.request

import numpy as np

BASE = "https://quasr.flatironinstitute.org/"
MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _pad(ID):
    s = str(int(ID)).zfill(7)
    return s, s[:4]


def url(ID, what):
    """URL for one device's file. what: nml | simsopt | record | poincare."""
    s, p = _pad(ID)
    return {
        "nml": f"{BASE}nml/{p}/input.{s}",
        "simsopt": f"{BASE}simsopt_serials/{p}/serial{s}.json",
        "record": f"{BASE}records/{p}/{s}.json",
        "poincare": f"{BASE}graphics/poincare_png/{p}/poincare{s}.png",
    }[what]


def fetch(ID, what, cache, force=False):
    """Download one file into `cache`, returning its local path. Cached by default."""
    s, _ = _pad(ID)
    name = {"nml": f"input.{s}", "simsopt": f"serial{s}.json",
            "record": f"record{s}.json", "poincare": f"poincare{s}.png"}[what]
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, name)
    if force or not os.path.exists(dest):
        with urllib.request.urlopen(url(ID, what), timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def fetch_index(cache, force=False):
    """Download database.json.gz (37 MB) once."""
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, "database.json.gz")
    if force or not os.path.exists(dest):
        with urllib.request.urlopen(BASE + "database.json.gz", timeout=600) as r, \
                open(dest, "wb") as f:
            f.write(r.read())
    return dest


def index(cache):
    """The whole index as a dict of numpy arrays (object arrays for the list columns).

    Columns: qs_error (log10), coil_length_per_hp, total_coil_length,
    total_coil_length_threshold, mean_iota, max_kappa, max_msc, min_coil2coil_dist,
    nc_per_hp, nfp, aspect_ratio, ID, minor_radius, Nfourier_coil, Nsurfaces, volume,
    min_coil2surface_dist, mean_elongation, max_elongation, message, iota_profile,
    tf_profile, surface_types, helicity.
    """
    with gzip.open(fetch_index(cache)) as f:
        d = json.load(f)
    cols = d["columns"]
    data = d["data"]
    out = {}
    for j, c in enumerate(cols):
        v = [r[j] for r in data]
        try:
            out[c] = np.asarray(v, dtype=float) if not isinstance(v[0], (str, list)) \
                else np.asarray(v, dtype=object)
        except (TypeError, ValueError):
            out[c] = np.asarray(v, dtype=object)
    out["ID"] = out["ID"].astype(int)
    for k in ("nfp", "nc_per_hp", "helicity", "Nfourier_coil", "Nsurfaces"):
        out[k] = out[k].astype(int)
    return out


def record(ID, cache):
    """One device's index row, as a dict."""
    with open(fetch(ID, "record", cache)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# the boundary
# ---------------------------------------------------------------------------
def patched_namelist(ID, cache, mpol=11, ntor=10, B0=1.0):
    """QUASR's namelist with the keys DESC's InputReader requires. Returns the new path.

    PHIEDGE is set to B0 * pi * a^2 using the index row's minor radius, matching QUASR's
    B0 ~ 1 T normalisation. It only scales |B|; it does NOT affect the boundary, iota, or
    eps_1 (the current-potential contours depend on the RATIO Phi_theta/Phi_zeta, and the
    contour levels scale with G, so an overall field scale cancels exactly).
    """
    src = fetch(ID, "nml", cache)
    a = float(record(ID, cache)["minor_radius"])
    psi = B0 * np.pi * a**2
    txt = open(src).read()
    head = f"&INDATA\nMPOL = {mpol}\nNTOR = {ntor}\nPHIEDGE = {psi!r}\nNCURR = 1\n"
    dest = os.path.join(cache, f"input.q{_pad(ID)[0]}")
    with open(dest, "w") as f:
        f.write(txt.replace("&INDATA\n", head, 1))
    return dest


def equilibrium(ID, cache, L=None, M=None, N=None, B0=1.0):
    """An UNSOLVED DESC Equilibrium carrying QUASR's boundary, plus the index row.

    Unsolved is enough for anything that only needs the boundary surface: `bn_stats(...,
    vacuum=True)`, the dense check, coil projection. It is NOT enough for eps_1, which needs
    `K_vc` and therefore the interior field -- solve it first (see `solve`).
    """
    from common import load_equilibrium

    eq, kind = load_equilibrium(patched_namelist(ID, cache, B0=B0))
    rec = record(ID, cache)
    if any(v is not None for v in (L, M, N)):
        eq.change_resolution(L=L or eq.L, M=M or eq.M, N=N or eq.N,
                             L_grid=None, M_grid=None, N_grid=None)
    return eq, rec


def check_boundary(ID, cache, eq=None, rec=None):
    """Does the parsed boundary reproduce the index row? a, A and V, relative."""
    if eq is None:
        eq, rec = equilibrium(ID, cache)
    d = eq.surface.compute(["a", "R0/a", "V"])
    got = dict(a=float(d["a"]), A=float(d["R0/a"]), V=float(d["V"]))
    want = dict(a=rec["minor_radius"], A=rec["aspect_ratio"], V=rec["volume"])
    return {k: dict(got=got[k], want=want[k], rel=abs(got[k] / want[k] - 1)) for k in got}


# ---------------------------------------------------------------------------
# the coils
# ---------------------------------------------------------------------------
def _resolve(obj, objs):
    """Follow simsopt's {"$type": "ref"} indirection."""
    while isinstance(obj, dict) and obj.get("$type") == "ref":
        obj = objs[obj["value"]]
    return obj


def _array(obj):
    return np.asarray(obj["data"], dtype=float) if isinstance(obj, dict) else np.asarray(obj)


def _curve_xyz(curve, objs, npts=400):
    """Sample a simsopt CurveXYZFourier on npts uniform parameter points.

    The dof ordering is read from the DOFs object's `names` ("xc(0)", "xs(1)", ...), so
    nothing here depends on simsopt's internal layout. simsopt's parameter is t in [0, 1)
    and x(t) = sum_m [ xc(m) cos(2 pi m t) + xs(m) sin(2 pi m t) ].
    """
    dofs = _resolve(curve["dofs"], objs)
    x = _array(dofs["x"])
    names = dofs["names"]
    names = names["data"] if isinstance(names, dict) else names
    t = np.arange(npts) / npts
    out = np.zeros((npts, 3))
    comp = {"x": 0, "y": 1, "z": 2}
    pat = re.compile(r"^([xyz])([cs])\((\d+)\)$")
    for val, nm in zip(x, names):
        m = pat.match(nm)
        if m is None:
            raise ValueError(f"unexpected simsopt dof name {nm!r}")
        c, kind, order = m.group(1), m.group(2), int(m.group(3))
        basis = np.cos(2 * np.pi * order * t) if kind == "c" else np.sin(2 * np.pi * order * t)
        out[:, comp[c]] += val * basis
    return out


def _current(cur, objs):
    """Total current of a (possibly nested ScaledCurrent) simsopt current, in amperes."""
    cur = _resolve(cur, objs)
    scale = 1.0
    while cur.get("@class") == "ScaledCurrent":
        scale *= float(cur["scale"])
        cur = _resolve(cur["current_to_scale"], objs)
    return scale * float(cur["current"])


def base_coils(ID, cache, npts=400):
    """QUASR's UNIQUE (half-period) coils: a list of (points, current).

    Only the coils whose curve is a bare CurveXYZFourier are returned -- the RotatedCurve
    copies are discarded and rebuilt by `CoilSet.from_symmetry`, which avoids depending on
    simsopt's rotation/flip sign conventions. `check_reconstruction` verifies that choice.
    """
    with open(fetch(ID, "simsopt", cache)) as f:
        objs = json.load(f)["simsopt_objs"]
    out = []
    for v in objs.values():
        if not isinstance(v, dict) or v.get("@class") != "Coil":
            continue
        curve = _resolve(v["curve"], objs)
        if curve.get("@class") != "CurveXYZFourier":
            continue
        out.append((_curve_xyz(curve, objs, npts), _current(v["current"], objs)))
    return out


def coilset(ID, cache, N=16, npts=400, sym=True, full=True):
    """QUASR's coil set as a DESC CoilSet of FourierXYZCoil.

    N=16 matches the database's own Nfourier_coil, and npts >> 2N+1 makes DESC's fit an
    interpolation of the same series rather than a smoothing of it.

    full=True replicates to the whole torus (2 * nfp * nc_per_hp leaf coils), which is what
    `common.evaluate` and `bn_stats` need. full=False returns only the nc_per_hp unique
    coils, which is the shape `make_start.py` produces and `run_one.py` expects -- note
    `common.iter_unique` counts LEAVES, so a full set reports every coil as independent and
    `expected_coils` over-counts it by nfp * 2.
    """
    from desc.coils import CoilSet, FourierXYZCoil

    rec = record(ID, cache)
    base = base_coils(ID, cache, npts=npts)
    coils = [FourierXYZCoil.from_values(I, pts, N=N, basis="xyz") for pts, I in base]
    if not full:
        return CoilSet(*coils)
    return CoilSet.from_symmetry(coils, NFP=rec["nfp"], sym=sym)


def check_reconstruction(ID, cache, cs=None, eq=None, rec=None):
    """Does the rebuilt coil set reproduce the index row? Returns (table, metrics).

    Six independent numbers: coil count, length per half period, max curvature, max
    mean-squared curvature, and the minimum coil-coil and coil-surface distances. A wrong
    stellarator-symmetry convention would break the count and both distances at once, so
    this is the test that licenses discarding simsopt's RotatedCurve objects. `metrics` also
    carries `bn`, which QUASR minimised and which no geometric check would notice.

    Note `common.iter_unique` yields LEAVES, and a `from_symmetry` set has all of them as
    leaves, so `evaluate`'s L_total is the whole torus: compare it to `total_coil_length`.
    """
    import common as S

    if cs is None:
        cs = coilset(ID, cache)
    if eq is None:
        eq, rec = equilibrium(ID, cache)
    rec = rec or record(ID, cache)
    m = S.evaluate(cs, eq, vacuum=True)
    got = dict(n_coils=m["n_coils"], total_coil_length=m["L_total"], max_kappa=m["kappa"],
               max_msc=m["kms"], min_coil2coil_dist=m["d_cc"], min_coil2surface_dist=m["d_pc"])
    want = dict(n_coils=2 * rec["nfp"] * rec["nc_per_hp"],
                total_coil_length=rec["total_coil_length"], max_kappa=rec["max_kappa"],
                max_msc=rec["max_msc"], min_coil2coil_dist=rec["min_coil2coil_dist"],
                min_coil2surface_dist=rec["min_coil2surface_dist"])
    table = {k: dict(got=got[k], want=want[k],
                     rel=abs(got[k] / want[k] - 1) if want[k] else float(got[k]))
             for k in got}
    return table, m
