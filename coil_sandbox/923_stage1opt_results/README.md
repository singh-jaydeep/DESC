# 923_stage1opt_results — 2026-09-23 session artifacts

**New to this work? Start with `HANDOFF.md`** — orientation, workflow, and the traps.
Then `SESSION.md` for the 2026-09-23 results. Code lives in `../922_stage1opt_results/`; the full chronology,
including every wrong turn, is in `../922_stage1opt_results/NOTES.md` sections 9m-9p.

    HANDOFF.md     START HERE if you are new: the question, the files, the workflow, the traps
    SESSION.md     what we did, the two results, and the caveats (READ SECTION 6)
    equilibria/    11 stage-1 results (.h5)
    coilsets/      8 stage-2 coilsets + their circles starts (.h5 / .json)
    viewers/       5 offline HTML viewers - open directly in a browser, no server
    logs/          every stage-1 / stage-2 run log and the chain drivers' output
    diagnostics/   measurement scripts; every table in SESSION.md regenerates from these

Headline: `--target-mode worst` + an `L_grad(B)` floor gives eps_1 -17.8% with planar coils
-9.23% (0.519 %/point vs proportional's 0.450), min L_gradB -1.81% (vs -25.81%) and pdrot
median -11.85% (vs +12.02%).

The best equilibrium is `equilibria/worst_leg3.h5`; its coils are `coilsets/worst3_{planar,xyz}.h5`
and its viewer is `viewers/view_worst3.html`.

DO NOT USE `equilibria/aspin_leg2_iotaleak.h5` - iota exceeded its bound by 0.013 through a
`--from` reference leak that is now fixed. `aspin_leg2_iotatied.h5` is the clean equivalent.

The viewers have a "Surface field" selector (L_gradB absolute, or the diverging ratio vs
precise_QA) and a "streamlines" toggle (current-potential contours shaded by their own eps_1).
