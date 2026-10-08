"""Collect benchmark runs and place them against the published Hurwitz front.

    python -m bench.compare --runs runs/E2 runs/E4 --out compare_E2_E4

Writes `<out>.csv` (one row per run) and `<out>.png` (our points over their
front). Pure numpy/pandas/matplotlib -- no DESC, no JAX, no GPU.

The reference front is `hurwitz_pareto_final.csv`, extracted from the plotly
figure embedded in the authors' own `analysis.ipynb` (Zenodo 13913510): 218
optima with exact field error, max force, and constraint values. These are the
authors' numbers, not digitized pixels.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
# imports below must follow matplotlib.use()
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(HERE, "hurwitz_pareto_final.csv")

# Fitted on their published front: max force vs minimum coil-surface distance,
# R^2 = 0.926, Spearman +0.977. Lets us place a coilset on their FORCE axis
# without a self-force model, since d_cs is the mediating variable they identify.
FORCE_FIT = {"slope": 64823.0, "intercept": -4488.0, "r2": 0.926}


def load_runs(dirs):
    """Load every run JSON found under the given directories."""
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            if path.endswith(".FAILED.json"):
                continue
            with open(path) as fh:
                r = json.load(fh)
            m, s, c = r["metrics"], r["solver"], r["config"]
            rows.append(
                {
                    "tag": r["tag"],
                    "exp": os.path.basename(os.path.normpath(d)),
                    "method": c.get("method"),
                    "nc": c.get("nc"),
                    "rep": c.get("rep"),
                    "order": c.get("order"),
                    "r_over_a": c.get("r_over_a"),
                    "current_mode": c.get("current_mode"),
                    "inner_reduction": r.get("options", {}).get("inner_reduction"),
                    "ctol": c.get("ctol"),
                    "bound_scale": json.dumps(c.get("bound_scale", {})),
                    "normalized_BdotN": m["normalized_BdotN"],
                    "normalized_BdotN_areawt": m["normalized_BdotN_areawt"],
                    "max_BdotN_over_B": m["max_BdotN_over_B"],
                    "max_length": m["max_length_paperR0"],
                    "max_kappa": m["max_max_kappa_paperR0"],
                    "kappa_MS": m["max_kappa_MS_paperR0"],
                    "d_cc": m["coil_coil_distance_paperR0"],
                    "d_cs": m["coil_surface_distance_paperR0"],
                    "arclength_var": m["max_arclength_variance"],
                    "n_linked_pairs": m["n_linked_pairs"],
                    "linking_frac_dev": m.get("linking_max_frac_dev"),
                    "current_drift": r.get("current_drift"),
                    "pass": r["paper_filter"]["pass"],
                    "failed_checks": ",".join(r["paper_filter"]["failed"]),
                    "success": s["success"],
                    "message": s["message"],
                    "cost": s["cost"],
                    "constr_violation": s["constr_violation"],
                    "nit": s["nit"],
                    "n_outer": s["n_outer"],
                    "wall_s": r["wall_s"],
                    # their force axis, inferred from d_cs (NOT a force calculation)
                    "force_proxy": FORCE_FIT["slope"]
                    * m["coil_surface_distance_paperR0"]
                    + FORCE_FIT["intercept"],
                }
            )
    return pd.DataFrame(rows)


def plot(df, ref, out_png):
    """Render the comparison figure."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    # -- panel 1: field error vs coil-surface distance (both measured) --
    a = ax[0]
    a.scatter(
        ref["normalized_BdotN"],
        ref["dcs"],
        s=14,
        c="0.7",
        label=f"Hurwitz front (N={len(ref)})",
        zorder=1,
    )
    if len(df):
        ok = df["pass"]
        a.scatter(
            df.loc[ok, "normalized_BdotN"],
            df.loc[ok, "d_cs"],
            s=52,
            marker="o",
            color="tab:blue",
            edgecolor="k",
            linewidth=0.6,
            label="ours (passes filter)",
            zorder=3,
        )
        a.scatter(
            df.loc[~ok, "normalized_BdotN"],
            df.loc[~ok, "d_cs"],
            s=52,
            marker="x",
            color="tab:red",
            label="ours (fails filter)",
            zorder=3,
        )
    a.set_xscale("log")
    a.set_xlabel(r"$\langle |B\cdot n|\rangle / \langle B\rangle$  (paper convention)")
    a.set_ylabel("min coil-surface distance [m] @ $R_0$=1")
    a.set_title("Measured quantities")
    a.legend(fontsize=8, frameon=False)

    # -- panel 2: their force axis, ours inferred from d_cs --
    a = ax[1]
    a.scatter(
        ref["normalized_BdotN"],
        ref["max_max_force"],
        s=14,
        c="0.7",
        label="Hurwitz front (measured force)",
        zorder=1,
    )
    if len(df):
        ok = df["pass"]
        a.scatter(
            df.loc[ok, "normalized_BdotN"],
            df.loc[ok, "force_proxy"],
            s=52,
            marker="o",
            color="tab:blue",
            edgecolor="k",
            linewidth=0.6,
            label="ours (force INFERRED from $d_{cs}$)",
            zorder=3,
        )
        a.scatter(
            df.loc[~ok, "normalized_BdotN"],
            df.loc[~ok, "force_proxy"],
            s=52,
            marker="x",
            color="tab:red",
            label="ours (fails filter)",
            zorder=3,
        )
    a.set_xscale("log")
    a.set_xlabel(r"$\langle |B\cdot n|\rangle / \langle B\rangle$")
    a.set_ylabel("max point-wise force [N/m]")
    a.set_title(
        f"Force axis — ours is a $d_{{cs}}$ proxy "
        f"($R^2$={FORCE_FIT['r2']:.2f}), not a force model"
    )
    a.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    return fig


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--out", default="compare")
    args = p.parse_args()

    df = load_runs(args.runs)
    ref = pd.read_csv(args.ref)
    if not len(df):
        print("no runs found in", args.runs)
        return
    df.to_csv(f"{args.out}.csv", index=False)
    plot(df, ref, f"{args.out}.png")

    cols = [
        "tag",
        "normalized_BdotN",
        "d_cs",
        "d_cc",
        "max_length",
        "max_kappa",
        "kappa_MS",
        "n_linked_pairs",
        "n_outer",
        "pass",
        "failed_checks",
    ]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    q = ref["normalized_BdotN"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    print("\nreference front normalized_BdotN quantiles:")
    print("  min %.3e | q25 %.3e | med %.3e | q75 %.3e | max %.3e" % tuple(q.values))
    if df["pass"].any():
        best = df.loc[df.loc[df["pass"], "normalized_BdotN"].idxmin()]
        pct = float((ref["normalized_BdotN"] < best["normalized_BdotN"]).mean())
        print(
            f"\nbest filter-passing run: {best['tag']}  "
            f"BdotN={best['normalized_BdotN']:.3e}"
        )
        print(
            f"  beats {100*(1-pct):.1f}% of their front "
            f"({int(round((1-pct)*len(ref)))} of {len(ref)} optima)"
        )
    else:
        print(
            "\nno run passed the paper filter; "
            "failed checks: "
            + ", ".join(
                sorted({c for s in df["failed_checks"] for c in s.split(",") if c})
            )
        )
    print(f"\nwrote {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
