"""Compile-and-time profiling for DESC optimization loops.

The point of this module is to answer, for a real ``Optimizer.optimize`` run,
the question "where is the time going, and how much of it is XLA compiling the
same thing again?"

Timing alone cannot answer that. A call that compiles once and then executes
slowly and a call that recompiles on every invocation look identical on a
stopwatch. So instead of timing, this module listens to JAX's own compilation
events via ``jax.monitoring``, which fire once per actual compilation and never
on a cache hit:

    /jax/core/compile/jaxpr_trace_duration           trace  (python -> jaxpr)
    /jax/core/compile/jaxpr_to_mlir_module_duration  lower  (jaxpr -> StableHLO)
    /jax/core/compile/backend_compile_duration       compile (StableHLO -> code)

Those are the three expensive stages of the four in the JAX execution model;
the fourth, execute, is what you actually want to be paying for. Each event
carries a ``fun_name``, so a recompiling ``lax.scan`` shows up by name.

The events fire synchronously on the calling thread, so attribution is just a
stack: whatever span is innermost when an event arrives is the code that caused
the compile. Spans are opened by ``span()`` context managers placed at a handful
of seams in the optimizer, and by ``wrap()`` around the objective callables that
the solvers are handed.

Usage is via ``Optimizer.optimize(..., profile=...)``; see ``Profiler`` for the
accepted forms. Nothing in this module does anything at all unless a profiler is
active, so the seams cost one attribute lookup when it is off.

Notes
-----
Wall times recorded here are *dispatch* wall times. JAX dispatches
asynchronously, so a span may exit before the work it queued has run. Compile
times are unaffected by this (compilation is synchronous), which is why they,
not wall time, are the headline numbers. Pass ``block=True`` to force a
``block_until_ready`` on values returned through ``wrap()`` if you want wall
times that mean something -- at the cost of perturbing what you are measuring.
"""

import json
import os
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

# event name -> the stage bucket we file it under
_STAGE_OF_EVENT = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower",
    "/jax/core/compile/backend_compile_duration": "compile",
}
_STAGES = ("trace", "lower", "compile")

# The active profiler, or None. Module level because the seams must be able to
# find it without any object carrying a reference: anything stored on an
# ObjectiveFunction instance would land in its pytree children (see
# desc/io/optimizable_io.py) and change the jit cache key, which would create
# the very recompiles we are trying to measure.
_ACTIVE = None

# jax.monitoring has no way to unregister a listener, only a global
# clear_event_listeners() that would stomp on anyone else's. So we register
# exactly one listener, once, forever, and let it no-op when _ACTIVE is None.
_LISTENER_REGISTERED = False


def _listener(event, duration, **kwargs):
    stage = _STAGE_OF_EVENT.get(event)
    if stage is None or _ACTIVE is None:
        return
    _ACTIVE._record_compile(stage, duration, kwargs.get("fun_name", "?"))


def _ensure_listener():
    global _LISTENER_REGISTERED
    if _LISTENER_REGISTERED:
        return
    from jax import monitoring

    monitoring.register_event_duration_secs_listener(_listener)
    _LISTENER_REGISTERED = True


class _Frame:
    """One open span: what it is, and what compiled while it was on the stack."""

    __slots__ = ("label", "path", "t0", "index", "self_", "incl", "funs", "collapsing")

    def __init__(self, label, path, index, collapsing):
        self.label = label
        self.path = path
        self.index = index
        self.collapsing = collapsing
        self.t0 = time.perf_counter()
        # "self_" is what compiled with this frame innermost; "incl" also counts
        # everything its children compiled.
        self.self_ = dict.fromkeys(_STAGES, 0.0)
        self.incl = dict.fromkeys(_STAGES, 0.0)
        self.self_["n"] = 0
        self.incl["n"] = 0
        # fun_name -> [how many times it compiled, seconds spent compiling it]
        self.funs = defaultdict(lambda: [0, 0.0])


class _NullSpan:
    """Context manager that does nothing, returned when profiling is off."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL_SPAN = _NullSpan()


class Profiler:
    """Records per-span wall time and JAX compilation activity.

    Parameters
    ----------
    path : str or None
        File to stream JSONL records to, one record per closed span. If None,
        records are kept in memory only and can be read from ``self.records``.
    level : int
        Whether to descend into nested *equilibrium solves*. Not a depth limit;
        see ``max_depth`` for that. 1 (default) collapses the ``eq.solve`` calls
        that re-enter ``Optimizer.optimize`` -- the one in
        ``ProximalProjection._update_equilibrium`` and the one in its ``build``
        -- into a single opaque span each. Compiles inside are still counted
        against that span, they are just not broken down further. 2 descends
        into them, which is much more verbose and re-enters every seam.
    max_depth : int or None
        Deepest span nesting to record. None (default) records every level.
        Spans deeper than this are not recorded, and the compiles they would
        have been charged for roll up to the deepest ancestor that is -- so
        totals stay correct at whatever depth you look. ``max_depth=1`` gives
        just ``build`` and ``solve``; 2 adds ``f``, ``jac``, ``grad``, ``hess``;
        3 adds ``tangents``, ``update_eq``, ``obj_jvp``.
    block : bool
        Whether to ``block_until_ready`` on values returned through ``wrap()``,
        making wall times real rather than dispatch-only. Perturbs the thing
        being measured; off by default.
    verbose : int
        1 (default) prints a summary table on ``stop()``. 0 is silent.

    """

    def __init__(self, path=None, level=1, block=False, verbose=1, max_depth=None):
        self.path = path
        self.level = level
        self.max_depth = max_depth
        self.block = block
        self.verbose = verbose
        self.records = []
        self._stack = []
        self._collapsed = 0  # >0 means we are inside an opaque span
        self._counts = defaultdict(int)  # label -> how many times entered
        self._step = None  # current outer iteration, set by set_step()
        self._step_depth = None  # span depth of the solver that owns _step
        self._fh = None
        self._t0 = None
        self._inert = False

    # ------------------------------------------------------------------ setup

    def start(self, header=None):
        """Activate this profiler and begin recording.

        If a profiler is already running this one goes inert rather than raising.
        That happens when ``profile`` reaches a nested solve -- e.g. via the
        ``solve_options`` that ``ProximalProjection`` passes to ``eq.solve`` --
        and the outer profiler is already recording those compiles anyway.
        """
        global _ACTIVE
        if _ACTIVE is not None:
            warnings.warn(
                "A Profiler is already active, so this one will not record. Its "
                "spans are already being captured by the outer profiler.",
                UserWarning,
            )
            self._inert = True
            return self
        _ensure_listener()
        self._t0 = time.perf_counter()
        if self.path is not None:
            # resolve now, so the path reported at the end is unambiguous -- a
            # relative path in a notebook lands in the kernel's cwd, which is
            # often not where the user is looking
            self.path = os.path.abspath(os.path.expanduser(str(self.path)))
            self._fh = open(self.path, "w")
        _ACTIVE = self
        self._emit({"kind": "header", **self._header(), **(header or {})})
        if self.verbose and self.path is not None:
            # announce up front, not just in the summary: on a long run you want
            # to know where records are landing before it finishes, and a
            # relative path resolves against the kernel's cwd, not the notebook's
            print(f"Profiling to {self.path}")
        return self

    def stop(self):
        """Deactivate, flush, and optionally print the summary."""
        global _ACTIVE
        if _ACTIVE is not self:
            # never started, or went inert because another profiler was running;
            # still close any handle so we do not leak it
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            return self
        _ACTIVE = None
        total = time.perf_counter() - self._t0
        self._emit({"kind": "total", "wall": total})
        if self.verbose:
            self.summary(total)
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @staticmethod
    def _header():
        import jax

        cache_dir = jax.config.jax_compilation_cache_dir
        return {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
            # a persistent cache would recognise identical computations by
            # content hash and mask exactly the problem this profiler looks for,
            # so it is worth recording whether one is on
            "compilation_cache_dir": cache_dir,
        }

    # ------------------------------------------------------------- recording

    def _record_compile(self, stage, duration, fun_name):
        if not self._stack:
            return
        innermost = self._stack[-1]
        innermost.self_[stage] += duration
        for frame in self._stack:
            frame.incl[stage] += duration
        if stage == "compile":
            innermost.self_["n"] += 1
            innermost.funs[fun_name][0] += 1
            innermost.funs[fun_name][1] += duration
            for frame in self._stack:
                frame.incl["n"] += 1

    def set_step(self, step):
        """Tag subsequent records with an outer solver iteration number.

        Every DESC solver reports its iterations, including the ``eq.solve``
        nested inside ``ProximalProjection`` and the one inside its build. Only
        the outermost solver's iteration count is meaningful as a label for the
        run, so this keeps whichever caller sits at the shallowest span depth and
        ignores the nested ones.
        """
        depth = len(self._stack)
        if self._step_depth is None or depth <= self._step_depth:
            self._step_depth = depth
            self._step = step

    @contextmanager
    def span(self, label, collapse=False):
        """Open a span. Compiles occurring inside are attributed to it.

        Parameters
        ----------
        label : str
            Name for this span, e.g. ``"jac_scaled_error"``.
        collapse : bool
            Mark this span opaque at ``level < 2``: spans opened inside it are
            suppressed, and everything they would have recorded rolls up here.

        """
        too_deep = self.max_depth is not None and len(self._stack) >= self.max_depth
        if self._collapsed or too_deep:
            # not recorded; compiles inside are charged to the innermost frame
            # still on the stack, so nothing is lost, only broken out less finely
            yield None
            return
        path = tuple(f.label for f in self._stack) + (label,)
        key = "/".join(path)
        self._counts[key] += 1
        frame = _Frame(label, path, self._counts[key], collapse)
        self._stack.append(frame)
        if collapse and self.level < 2:
            self._collapsed += 1
        try:
            yield frame
        finally:
            if collapse and self.level < 2:
                self._collapsed -= 1
            self._stack.pop()
            self._close(frame)

    def _close(self, frame):
        wall = time.perf_counter() - frame.t0
        record = {
            "kind": "span",
            "path": "/".join(frame.path),
            "label": frame.label,
            "depth": len(frame.path),
            "index": frame.index,
            "step": self._step,
            "wall": wall,
            "t_start": frame.t0 - self._t0,
        }
        for stage in _STAGES:
            record[f"incl_{stage}"] = frame.incl[stage]
            record[f"self_{stage}"] = frame.self_[stage]
        record["incl_ncompiles"] = frame.incl["n"]
        record["self_ncompiles"] = frame.self_["n"]
        record["funs"] = dict(frame.funs)
        self.records.append(record)
        self._emit(record)

    def _emit(self, record):
        if self._fh is not None:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()  # so a crash or a kill still leaves usable data

    # ---------------------------------------------------------------- output

    def spans(self):
        """list: the span records, in the order they closed."""
        return [r for r in self.records if r.get("kind") == "span"]

    def summary(self, total=None):
        """Print an aggregated table, one row per distinct span path."""
        spans = self.spans()
        if not spans:
            print("\nProfile: no spans recorded.")
            return
        agg = {}
        for r in spans:
            a = agg.setdefault(
                r["path"],
                {"n": 0, "wall": 0.0, "compile": 0.0, "trace": 0.0, "ncomp": 0},
            )
            a["n"] += 1
            a["wall"] += r["wall"]
            # self_, not incl_, so that a parent and its children do not
            # double-count the same compile in one table
            a["compile"] += r["self_compile"]
            a["trace"] += r["self_trace"] + r["self_lower"]
            a["ncomp"] += r["self_ncompiles"]
        rows = sorted(agg.items(), key=lambda kv: -kv[1]["compile"])

        width = max(max(len(p) for p in agg), 20)
        print("\n" + "=" * (width + 46))
        print("Optimization profile" + (f"  (total {total:.2f} s)" if total else ""))
        print("=" * (width + 46))
        print(
            f"{'span':<{width}} {'calls':>6} {'wall/s':>9} "
            f"{'compile/s':>10} {'trace/s':>8} {'#cmp':>5}"
        )
        print("-" * (width + 46))
        for path, a in rows:
            print(
                f"{path:<{width}} {a['n']:>6d} {a['wall']:>9.2f} "
                f"{a['compile']:>10.2f} {a['trace']:>8.2f} {a['ncomp']:>5d}"
            )
        print("-" * (width + 46))
        comp = sum(a["compile"] for a in agg.values())
        print(f"{'total compile':<{width}} {'':>6} {'':>9} {comp:>10.2f}")

        repeats = self._repeat_offenders()
        if repeats:
            print(
                "\nRecompiled after the first call (a cached computation would "
                "compile once):"
            )
            for path, fun, calls, secs in repeats:
                print(
                    f"  {path} -> {fun}: compiled on {calls} separate calls, "
                    f"{secs:.2f} s total"
                )
        if self.path:
            print(f"\nRecords written to {self.path}")

    def _repeat_offenders(self, min_calls=2):
        """Spans that compiled the same fun_name on more than one call.

        This is the signal the whole module exists for. A computation whose
        cache key is stable compiles on its first call and never again, so any
        (span, fun_name) pair that compiles on two or more separate calls is
        missing a cache.
        """
        per_fun = defaultdict(lambda: {"calls": 0, "secs": 0.0})
        for r in self.spans():
            for fun, (_, secs) in r["funs"].items():
                key = (r["path"], fun)
                per_fun[key]["calls"] += 1
                per_fun[key]["secs"] += secs
        out = [
            (path, fun, v["calls"], v["secs"])
            for (path, fun), v in per_fun.items()
            if v["calls"] >= min_calls
        ]
        return sorted(out, key=lambda t: -t[3])


# --------------------------------------------------------------------- seams
# Everything below is what the optimizer itself calls. All of it is a cheap
# no-op when no profiler is active.


def get_profiler():
    """Profiler or None: the currently active profiler."""
    return _ACTIVE


def span(label, collapse=False):
    """Open a profiling span if profiling is on, else do nothing.

    Parameters
    ----------
    label : str
        Name for this span.
    collapse : bool
        Whether to treat this span as opaque at ``level < 2``, suppressing
        spans nested inside it.

    """
    if _ACTIVE is None:
        return _NULL_SPAN
    return _ACTIVE.span(label, collapse=collapse)


def set_step(step):
    """Tag subsequent records with an outer solver iteration number."""
    if _ACTIVE is not None:
        _ACTIVE.set_step(step)


def wrap(fn, label):
    """Wrap a callable so each call becomes a span. Returns fn if profiling off.

    Used at the seam in ``_desc_wrappers`` where bound objective methods become
    the plain ``fun``/``jac``/``grad``/``hess`` callables the solvers take. This
    keeps the instrumentation entirely outside the objective, which matters:
    attaching anything to an ObjectiveFunction instance would change its pytree
    structure and so its jit cache key.

    Parameters
    ----------
    fn : callable
        The function to wrap, typically a bound method of an ObjectiveFunction.
    label : str
        Name for the spans this produces.

    """
    if _ACTIVE is None:
        return fn

    def profiled(*args, **kwargs):
        prof = _ACTIVE
        if prof is None:  # profiling stopped after the wrap
            return fn(*args, **kwargs)
        with prof.span(label):
            out = fn(*args, **kwargs)
            if prof.block and hasattr(out, "block_until_ready"):
                out.block_until_ready()
            return out

    profiled.__name__ = getattr(fn, "__name__", label)
    profiled.__doc__ = getattr(fn, "__doc__", None)
    return profiled


def make_profiler(profile):
    """Build a Profiler from the ``profile`` argument to ``Optimizer.optimize``.

    Parameters
    ----------
    profile : None, bool, str, path-like or dict
        None or False for no profiling. True profiles to stdout only. A string
        or path is taken as the output file. A dict is passed as keyword
        arguments to ``Profiler``.

    Returns
    -------
    profiler : Profiler or None

    """
    if profile is None or profile is False:
        return None
    if profile is True:
        return Profiler()
    if isinstance(profile, dict):
        return Profiler(**profile)
    return Profiler(path=str(profile))
