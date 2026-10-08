"""Interactive 3D view (plotly, offline HTML) of arc coilsets on helios_repro.

    python work/helios/plot_coils_3d.py OUT.html "label=path.h5" ["label=path.h5" ...]

The FIRST coilset is the main one: the plasma boundary is coloured by its B.n (coils + plasma,
tesla), its three unique coils are drawn thick (one colour per coil, arc 0 dark / arc 1 light,
hinges as black dots) and its symmetry copies thin. Further coilsets are extra traces, hidden
until clicked in the legend.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, "sandbox")
import boot  # noqa: E402

boot.setup(default="cpu")
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

import common as S  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402
from desc.integrals.singularities import compute_B_plasma  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives._coils import _independent_coil_indices  # noqa: E402

PAL = [("#1f5fa8", "#7fb0e8"), ("#c2521d", "#f2a47a"), ("#16825b", "#7fd3b0")]
NARC = 200


def arc_positions(cs):
    """(ncoil, 2*NARC+1, 3): every physical coil, s from 0 to 2 pi inclusive (hinges at 0, pi, 2 pi)."""
    s = np.linspace(0, 2 * np.pi, 2 * NARC + 1)
    g = Grid(np.stack([0 * s, 0 * s, s], 1), sort=False, jitable=False)
    return np.asarray(cs._compute_position(grid=g, basis="xyz"))


def smooth_positions(cs):
    return np.asarray(cs._compute_position(grid=LinearGrid(N=300), basis="xyz"))


def main():
    out = sys.argv[1]
    items = [a.rsplit("=", 1) for a in sys.argv[2:]]
    eq, _ = S.load_equilibrium("sample_equilibria/helios_repro.h5")
    main_label, main_path = items[0]
    cs = load(main_path)
    m = S.evaluate(cs, eq)

    # plasma boundary over one field period (exact virtual-casing grid), then rotated copies
    nt, nz = 96, 96
    g = LinearGrid(rho=np.array([1.0]), theta=nt, zeta=nz, NFP=eq.NFP, sym=False)
    d = eq.compute(["R", "phi", "Z", "n_rho"], grid=g)
    xr = np.stack([d["R"], d["phi"], d["Z"]], 1)
    Bp = np.asarray(compute_B_plasma(eq, g, S.vc_source_grid(eq), chunk_size=64))
    B = np.asarray(cs.compute_magnetic_field(xr, basis="rpz", source_grid=S.source_grid(cs)[0])) + Bp
    Bn = np.sum(B * np.asarray(d["n_rho"]), 1).reshape(nz, nt)
    R = np.asarray(d["R"]).reshape(nz, nt)
    P = np.asarray(d["phi"]).reshape(nz, nt)
    Z = np.asarray(d["Z"]).reshape(nz, nt)
    # close the surface in theta, and join consecutive periods in zeta by repeating the first row
    close = lambda A: np.concatenate([A, A[:, :1]], axis=1)
    R, P, Z, Bn = map(close, (R, P, Z, Bn))
    lim = float(np.ceil(np.abs(Bn).max() * 10) / 10)
    traces = []
    for k in range(eq.NFP):
        dphi = 2 * np.pi * k / eq.NFP
        Pk = np.concatenate([P, P[:1] + 2 * np.pi / eq.NFP], axis=0) + dphi
        Rk = np.concatenate([R, R[:1]], axis=0)
        Zk = np.concatenate([Z, Z[:1]], axis=0)
        Bk = np.concatenate([Bn, Bn[:1]], axis=0)
        traces.append(go.Surface(x=Rk * np.cos(Pk), y=Rk * np.sin(Pk), z=Zk, surfacecolor=Bk, cmin=-lim, cmax=lim,
                                 colorscale="RdBu_r", showscale=(k == 0), opacity=0.95, name="plasma B·n",
                                 colorbar=dict(title="B·n (T)", len=0.6, x=0.02), hovertemplate="B·n %{surfacecolor:.3f} T"
                                 "<extra></extra>", lighting=dict(ambient=0.8, diffuse=0.4, specular=0.05)))

    # main coilset
    pos = arc_positions(cs)
    uniq = [int(i) for i in _independent_coil_indices(cs)]
    rows_per_unique = pos.shape[0] // len(uniq)
    for r in range(pos.shape[0]):
        u = min(r // rows_per_unique, len(uniq) - 1)
        base = r in uniq
        for a in range(2):
            seg = pos[r, a * NARC:(a + 1) * NARC + 1]
            traces.append(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                line=dict(color=PAL[u % 3][a], width=11 if base else 4),
                opacity=1.0 if base else 0.55,
                name=f"{main_label}: coil {u} arc {a}" if base else f"{main_label}: copies",
                legendgroup=f"main{u}{a}" if base else "copies", showlegend=base or (r == 1 and a == 0),
                hovertemplate=f"coil {u}{'' if base else ' (copy)'} arc {a}<extra></extra>"))
    H = np.array([pos[r, [0, NARC]] for r in uniq]).reshape(-1, 3)
    traces.append(go.Scatter3d(x=H[:, 0], y=H[:, 1], z=H[:, 2], mode="markers", marker=dict(size=6, color="black"),
                               name=f"{main_label}: hinges", hovertemplate="hinge<extra></extra>"))

    # comparison coilsets (hidden until clicked)
    greys = ["#555555", "#999999", "#b07cc6"]
    for k, (lab, p) in enumerate(items[1:]):
        other = load(p)
        pp = arc_positions(other) if S.is_arc(other) else smooth_positions(other)
        for r in range(pp.shape[0]):
            q = pp[r] if S.is_arc(other) else np.vstack([pp[r], pp[r][:1]])
            traces.append(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="lines",
                                       line=dict(color=greys[k % 3], width=5), name=lab, legendgroup=f"o{k}",
                                       showlegend=(r == 0), visible="legendonly", hoverinfo="skip"))

    title = (f"{main_label} on helios_repro: ⟨|B·n|⟩/⟨|B|⟩ {m['bn']:.3e}, max |B·n| {m['bn_max_T']:.2f} T, "
             f"{m['n_coils']} coils, {S.verdict(m, S.paper_bounds(eq, 3, None, 'bounds/helios_kruger2026.json'))}"
             "<br><sub>thick = the 3 unique coils (arc 0 dark, arc 1 light, hinges black); thin = symmetry copies; "
             "click legend entries to show the comparison coilsets</sub>")
    fig = go.Figure(traces)
    fig.update_layout(title=dict(text=title, x=0.01), template="plotly_white", height=850,
                      scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=0.9, y=-1.3, z=0.9))),
                      legend=dict(x=0.82, y=0.95, bgcolor="rgba(255,255,255,0.7)"), margin=dict(l=0, r=0, t=70, b=0))
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print("wrote", out)
    try:
        png = os.path.splitext(out)[0] + ".png"
        fig.write_image(png, width=1400, height=900, scale=1)
        print("wrote", png)
    except Exception as e:  # static export needs a working kaleido/chrome
        print(f"(no static PNG: {type(e).__name__}: {str(e)[:120]})")


if __name__ == "__main__":
    main()
