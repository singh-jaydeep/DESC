# Operating procedure: running the sweep

For the agent executing the 42 runs. Read `README.md` first for what the bundle is, and
`fixed_configs.md` for what the numbers mean. This document is only about **how to work
through it**.

The user checks in every 1–2 hours. **Proceed uninterrupted between check-ins.** Stopping
to ask about something this document already covers wastes hours of machine time.

---

## 0. Before the first run

```bash
python check_bundle.py                          # integrity, vendored DESC, and the GPU
python tools/run_sweep.py sweep_grid.json --dry-run
```

**The commands default to `--device gpu`.** `check_bundle.py` reports what JAX can
actually see; if it says `WARN ... No GPU visible`, fix the install or regenerate for CPU
(`python tools/sweep_configs.py --device cpu --json sweep_grid.json`) — do not just let
it fall back, because half the sweep would then be on the other device and the chains
would not be comparable.

Before starting, cap the device allocator, or JAX takes most of the card up front:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Verify the GPU is really being used, on the first run

Every run log states which device it *obtained*, not just which was requested:

```
DEVICE requested=gpu obtained=gpu (1x NVIDIA GeForce RTX 5070 Laptop GPU)
```

On the first run, confirm all three:

1. `grep DEVICE runs/<tag>.log` shows `obtained=gpu`, and **no** `DEVICE WARNING` line.
   `set_device("gpu")` does not fail loudly on a box with no GPU — a run can execute on
   CPU at roughly half speed with nothing in the record to say so. That line is the
   evidence.
2. `nvidia-smi` during the solve shows the python process holding memory and non-zero
   utilisation.
3. The wall time is in the right range. `tools/monitor.py` prints the device on every
   check and flags a requested/obtained mismatch.

If the first run came out on CPU, **stop and redo it** — do not continue a chain whose
anchor used a different device from its rungs.

**Run `arcB2` first, as a pilot.** It is the cheapest chain (~3.4 h) and it is a
*complete* one: a cold anchor, two warm rungs, the second cold anchor, and a `--dcc-signed`
continuation. If anything about the plumbing is wrong you find out in three hours rather
than on hour thirty.

```bash
python tools/run_sweep.py sweep_grid.json --rep arcB2
```

Then stop, report, and let the user look before committing the remaining ~37 h.

## 1. The loop

`run_sweep.py` already does the per-run mechanics: solve → adjudicate → next, strictly
serial, one process at a time, refusing to start a warm run whose donor produced no final
`h5`. **Use it rather than hand-running configs** — it enforces two invariants that cost
this campaign real runs (§6).

Your job during a run is monitoring and the stall decision. Every ~5 minutes:

```bash
python tools/monitor.py runs/<tag>.log
```

It reads only the log, so it cannot disturb the solve. That matters: two runs died to a
`verify` running concurrently with a solve. **Never run anything heavy while a solve is
running** — no adjudication, no viewer, no second config.

## 2. When to stop a run early

`--maxiter` is 800. A run may stop earlier **only** if it is genuinely stalled.

**Do not judge a stall before iteration 500.** Measured: both ×1.25 and both ×1.5 runs hit
a 500 ceiling *still descending*, and the ×1.5 certification run was still improving ~1%
per 50 iterations at 800. Flat progress before 500 usually means the penalty phase, not a
stall. `monitor.py` refuses to call a stall before `--min-iter` (default 500) for this
reason.

The criterion it applies, and the one to use:

> **stalled** = at iteration ≥ 500, `bn` improved < 0.5% over the last 3 checkpoints
> (~150 iterations).

**A stall may be arrival, not failure.** Measured at one endpoint: four constraints inside
0.5% of their bounds simultaneously, `|Pg|/|g| = 0.993` — a perfectly good descent
direction — but `t_max = 9.17e-06`, so a near-active row blocks the step almost
immediately. That reads exactly like a stall and is the answer. You cannot tell the two
apart from the log; you tell them apart from the margins, i.e. by adjudicating.

### How to stop — this part matters

**Do not kill the process.** A run stopped by hand writes checkpoints but **no final `h5`
and no record**. Seven runs were lost that way on 2026-09-09; they had to be reconstructed
offline and the status board carried a void result for a week. In a chain it is worse:
the next rung cannot start at all.

Instead:

```bash
# let it run to maxiter if it is close, otherwise:
python tools/promote_checkpoint.py runs/<tag>          # scans every checkpoint
python tools/adjudicate.py runs/<tag> --length-max <L> # then adjudicate normally
```

`promote_checkpoint.py` adjudicates **every** checkpoint and promotes the last *cleanly
feasible* one to `<tag>.h5`, repairing the chain. It refuses to trust the last iterate,
and that refusal is the point: measured on one run, `bn` kept falling to it=350 and looked
like the best point of the run, but `kappa` had drifted to 1.0048 from it=200 and the run
was trading that drift for `d_cc`. **The apparent improvement past it=175 *was* the
constraint drift.**

Anything promoted is stopped rather than converged: its `bn` is a **lower bound**. Record
that (§4).

## 3. After each run

`run_sweep.py` adjudicates automatically. If you ran a config by hand, do it yourself —
serially, never alongside a solve:

```bash
python tools/adjudicate.py runs/<tag> --length-max <L>   # add --dcc-converge at x1.4, x1.6
```

Then read the verdict and act on it:

| result | what it means | what to do |
|---|---|---|
| `FEASIBLE` | every paper bound met, unlinked | record it, next run |
| **`linked` > 0** | **VOID** — the flux is meaningless | **stop this chain**, §5 |
| fails by < ~0.2% on a smooth rep | enforcement-grid effect, not a real violation | record the number, continue |
| fails `length` at ~1.0000, or `d_pc` at 0.999 | saturation, not violation | record, continue |
| fails an arc bound by > 1% | the backoff did not hold | record, continue, flag for the user |
| `d_cc` drift > 0.2% from `--dcc-converge` | not grid-converged | **do not quote `d_cc`**, flag it |

**Never trust the solver's own verdict.** With `ctol=1e-4` a stalled run reports
`precision` (SUCCESS) for violations in 1e-6..1e-4. Only `adjudicate.py` decides.

## 4. What to record, and where

**Do not hand-edit `sweep_grid.json` or `RUNS.md`.** Both are *generated* —
`sweep_grid.json` is the input config the runner reads, and `RUNS.md` comes from
`sweep_configs.py --runlist`. Editing either makes the config stop describing what
actually runs, and your edits are lost the next time they are regenerated.

Results already persist to `runs/<tag>/adjudication.json`. Regenerate the table:

```bash
python tools/results.py --md RESULTS.md
```

Add anything a table cannot hold to **`LOG.md`**, one entry per run, in this shape:

```markdown
## sw_arcB5_L1p2   (finished 14:07, 52 min, 800 iters)
bn 1.02e-04, max 8.1e-04. FEASIBLE, 0 linked.
margins: length 0.999 · kappa 0.981 · kMS 0.758 · d_cc 1.004 · d_pc 1.121
notes: —
```

Entries that need a note, and the exact wording to use:

* **stopped early** — "STOPPED at it=NNN on a stall: bn improved X% over the last 3
  checkpoints. Promoted checkpoint it=MMM (last cleanly feasible). bn is a LOWER BOUND."
* **failed a bound** — name it and give the number: "FAILS kappa_MS by 3.2% (5.829 vs
  5.648)". Never just "failed".
* **linked** — "VOID: N linked pairs {list}. Chain stopped; downstream runs not attempted."

Then say which run you are starting next, and start it.

## 4a. Crashes and resuming

Resume is deliberately simple: **finished configs are skipped, the crashed one is re-run
from its start.** No augmented-Lagrangian state is saved or restored, and nothing tries to
continue a solve from the middle. The most you lose is the elapsed time of the one run
that was in flight.

```bash
python tools/run_sweep.py sweep_grid.json --status    # where am I?
python tools/run_sweep.py sweep_grid.json --resume    # carry on
```

`--status` classifies every run, and each class resumes differently:

| state | meaning | what `--resume` does |
|---|---|---|
| `done` | final `h5` **and** a readable adjudication | skipped |
| `SOLVED, NOT ADJUDICATED` | the solve finished, the crash hit during adjudication | **adjudicates only, no re-solve** (seconds, not an hour) |
| `PARTIAL` | checkpoints only — crashed or killed mid-solve | re-runs that config from the start |
| `to do` | not started | runs it |

Three properties worth knowing, all verified by crashing a run on purpose:

* **A chain repairs itself.** The crashed config is re-run, and the rung that warm-starts
  from it then finds its donor and proceeds. Runs after it in the chain were never
  started, so nothing downstream is stale.
* **The log is appended, never truncated.** Each attempt is delimited by
  `===== attempt started <timestamp> =====`, so the record of *why* the run died survives
  the resume. That is the only forensic evidence you get.
* **A truncated `adjudication.json` does not count as done.** It is parsed, not just
  stat'ed, so a crash part-way through the write causes a re-adjudication rather than
  silently dropping a sweep point.

**When re-running is not the cheapest option.** If a run died *late* — past ~500
iterations — its last checkpoint may already be a bankable point, and re-running costs the
whole hour again. In that case:

```bash
python tools/promote_checkpoint.py runs/<tag>     # bank the last feasible checkpoint
python tools/adjudicate.py runs/<tag> --length-max <L>
```

Use this by exception, not by default. A promoted point is stopped rather than converged,
its `bn` is a **lower bound**, and it must be recorded as such (§4). For a cold anchor in
particular, prefer re-running: the cold-vs-warm distinction is a measured 9–21% effect,
and the ×1.4 cold control exists to measure it.

**If the box itself went down**, re-run `python check_bundle.py` before resuming — a hard
power loss can truncate an `h5` mid-write, and the integrity check is cheaper than
discovering it as a warm-start donor.

## 5. When a run fails, does the chain continue?

| failure | continue the chain? |
|---|---|
| **linked pairs > 0** | **No.** A linked donor seeds a linked geometry into every rung below it. Stop that representation and move to the next one. |
| fails a bound by a small margin (< ~1%) | Yes — record the number and continue. The next rung re-solves against the correct bounds. |
| fails a bound badly (> ~5%), or `d_cc` far out | No. Warm-starting from a badly infeasible donor is the exact configuration that produced the linked coilsets: a converged donor is bound-saturated, so the receiving run's penalty phase begins with no slack. Stop the chain, flag it. |
| stalled but feasible | Yes — promote and continue. |
| solve crashed (OOM, traceback) | Retry once. If it repeats, stop that chain and flag it. |

Chains are independent across representations, so stopping one costs you that
representation, not the sweep. Move to the next `--rep` and keep going.

## 6. Things that will bite, in one place

* **One process at a time.** Not a guideline — this box OOMs otherwise, and WSL2 has
  **no OOM killer**, so it takes the machine down rather than the process. `run_sweep.py`
  wraps each run in a `systemd-run` cgroup at `MemoryMax=12G`; keep it.
* **On GPU, the cgroup does not bound VRAM.** `MemoryMax=12G` caps host RAM only. Set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION`) or JAX will
  grab most of the card up front. The `--dcc-signed` warm rungs are the ones that will
  exhaust VRAM first, not the cold anchors — the linking number inside an AD trace is
  O(N²) in grid nodes, and at 801 nodes that broadcast alone is 3.9 GB. VRAM exhaustion
  usually raises `RESOURCE_EXHAUSTED` and kills the process rather than the box, and
  `monitor.py` flags it; a **host** RAM exhaustion takes the machine down.
* **Every run log states the device it obtained.** Check it on the first run of each
  representation, not just the first run overall.
* **Do not mix devices within a chain.** GPU and CPU results are not bit-identical. Run
  the whole sweep on one device; if you switch, restart the affected chain from its cold
  anchor.
* **`check_bundle.py --deep` may report differing start hashes on GPU.** The pins were
  taken on CPU. Small floating-point differences are expected and are *not* corruption —
  but confirm the seven starts still read FEASIBLE, and note it.
* **Never run adjudication, the viewer, or a second config while a solve runs.**
* **`linked pairs` is a hard rejection.** Two points in this campaign were banked as
  improvements before anyone read that field, and both improvements *were* the linking
  (8.6% and 18.4%).

## 7. Roughly how long

≈41 h sequential on CPU; less on GPU, though the measured speedup on this workload is
only ~2×. Per representation: arcB2 ~3.4 h, arcB3 ~4.0 h, arcB5 ~5.1 h, arcB7 ~6.2 h,
planarN7 ~7.0 h, xyzN4 ~6.7 h, xyzN6 ~8.5 h. Those are CPU estimates at `maxiter=800`,
and only arcB3, arcB5 and xyzN4 are measured — the rest are extrapolated by DOF. Report
the real per-run time in `LOG.md`; the first chain calibrates the rest.

## 8. When to actually stop and ask

Everything above is yours to decide. Stop and wait for the user only if:

* two chains fail the same way — that is a systematic problem, not a run;
* a start fails `check_bundle.py` (the geometry is not what was pinned);
* a run OOMs the machine rather than the process;
* you are about to do something this document does not cover.

Otherwise: record it, say what you are doing next, and do it.
