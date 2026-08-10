"""Tests for the optimization profiler in desc.optimize._profiling."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desc.optimize._profiling import Profiler, make_profiler, span, wrap


def _make_body():
    """Fresh function object each call, cheap to run but not free to compile."""
    W = jnp.asarray(np.random.default_rng(0).normal(size=(8, 8)))

    def body(v):
        x = v
        for _ in range(4):
            x = jnp.tanh(W @ x) + 0.5 * x
        return x

    return body


def _scan_with(body, xs):
    """Eager lax.scan whose body callable is built fresh on every call."""
    _, out = jax.lax.scan(lambda c, x: (c, body(x)), (), xs)
    return out


@pytest.mark.unit
def test_profiler_distinguishes_cached_from_recompiling():
    """A stable scan body compiles once; a fresh one compiles every call.

    This is the whole point of the profiler: timing cannot tell these apart,
    because the recompiling version also produces correct results and its cost
    looks like slow execution.
    """
    xs = jnp.asarray(np.random.default_rng(1).normal(size=(12, 8)))

    stable_body = _make_body()

    def stable_scan_body(carry, x):
        return carry, stable_body(x)

    def cached(xs):
        _, out = jax.lax.scan(stable_scan_body, (), xs)
        return out

    fresh_body = _make_body()

    prof = Profiler(verbose=0)
    with prof:
        for _ in range(3):
            with span("cached"):
                cached(xs).block_until_ready()
        for _ in range(3):
            with span("recompiling"):
                _scan_with(fresh_body, xs).block_until_ready()

    by_label = {}
    for record in prof.spans():
        by_label.setdefault(record["label"], []).append(record)

    cached_spans = by_label["cached"]
    recompiling_spans = by_label["recompiling"]
    assert len(cached_spans) == 3
    assert len(recompiling_spans) == 3

    # a stable cache key means nothing compiles after the first call
    assert sum(s["self_ncompiles"] for s in cached_spans[1:]) == 0
    # a fresh body means every call compiles
    assert all(s["self_ncompiles"] > 0 for s in recompiling_spans)

    offenders = prof._repeat_offenders()
    assert [path for path, _, _, _ in offenders] == ["recompiling"]
    assert offenders[0][1] == "jit(scan)"


@pytest.mark.unit
def test_profiler_attributes_compiles_to_innermost_span():
    """Compiles land on the innermost open span, and roll up into its parents."""
    xs = jnp.asarray(np.random.default_rng(2).normal(size=(12, 8)))
    body = _make_body()

    prof = Profiler(verbose=0)
    with prof:
        with span("outer"):
            with span("inner"):
                _scan_with(body, xs).block_until_ready()

    spans = {s["label"]: s for s in prof.spans()}
    assert spans["inner"]["self_ncompiles"] > 0
    # the parent did no compiling of its own, but sees the child's inclusively
    assert spans["outer"]["self_ncompiles"] == 0
    assert spans["outer"]["incl_ncompiles"] == spans["inner"]["incl_ncompiles"]
    assert spans["outer"]["incl_compile"] >= spans["inner"]["self_compile"]


@pytest.mark.unit
def test_profiler_collapse_suppresses_nested_spans():
    """collapse=True hides nested spans but keeps their compiles attributed."""
    xs = jnp.asarray(np.random.default_rng(3).normal(size=(12, 8)))
    body = _make_body()

    prof = Profiler(level=1, verbose=0)
    with prof:
        with span("opaque", collapse=True):
            with span("hidden"):
                _scan_with(body, xs).block_until_ready()

    labels = [s["label"] for s in prof.spans()]
    assert labels == ["opaque"]
    assert prof.spans()[0]["self_ncompiles"] > 0

    # level 2 descends instead
    prof2 = Profiler(level=2, verbose=0)
    with prof2:
        with span("opaque", collapse=True):
            with span("hidden"):
                _scan_with(_make_body(), xs).block_until_ready()
    assert sorted(s["label"] for s in prof2.spans()) == ["hidden", "opaque"]


@pytest.mark.unit
def test_profiler_step_ignores_nested_solvers():
    """Only the shallowest solver's iteration counter labels the records."""
    prof = Profiler(verbose=0)
    with prof:
        with span("solve"):
            prof.set_step(7)  # outer solver, depth 1
            with span("jac"):
                with span("eq_solve"):
                    prof.set_step(99)  # nested solver, deeper -- must be ignored
                with span("after"):
                    pass
    steps = {s["label"]: s["step"] for s in prof.spans()}
    assert steps["after"] == 7
    assert steps["eq_solve"] == 7


@pytest.mark.unit
def test_profiler_is_inert_when_off():
    """The seams cost nothing and change nothing when no profiler is active."""
    assert span("anything") is span("anything else")  # the shared null span

    def f(x):
        return x + 1

    assert wrap(f, "label") is f  # not even wrapped

    with span("no profiler running"):
        pass  # must not raise


@pytest.mark.unit
def test_make_profiler_forms():
    """The profile= argument accepts the documented shapes."""
    assert make_profiler(None) is None
    assert make_profiler(False) is None
    assert isinstance(make_profiler(True), Profiler)
    assert make_profiler("out.jsonl").path == "out.jsonl"
    prof = make_profiler({"level": 2, "block": True, "verbose": 0})
    assert prof.level == 2 and prof.block is True


@pytest.mark.unit
def test_profiler_writes_jsonl(tmp_path):
    """Records stream to disk as JSON lines, with a header and a total."""
    path = tmp_path / "profile.jsonl"
    prof = Profiler(path=str(path), verbose=0)
    with prof:
        with span("work"):
            pass

    records = [json.loads(ln) for ln in path.read_text().splitlines()]
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "header"
    assert kinds[-1] == "total"
    assert "span" in kinds
    header = records[0]
    for key in ["jax_version", "backend", "x64", "compilation_cache_dir"]:
        assert key in header


@pytest.mark.unit
def test_profiler_wrap_records_a_span_per_call():
    """wrap() turns a callable into one span per invocation."""
    prof = Profiler(verbose=0)
    with prof:
        wrapped = wrap(lambda x: x * 2, "f")
        for _ in range(3):
            wrapped(jnp.ones(3))

    spans = [s for s in prof.spans() if s["label"] == "f"]
    assert [s["index"] for s in spans] == [1, 2, 3]


@pytest.mark.unit
def test_profiler_stops_on_exception():
    """A failure inside a span must not leave the profiler globally active."""
    from desc.optimize import _profiling

    prof = Profiler(verbose=0)
    with pytest.raises(RuntimeError, match="boom"):
        with prof:
            with span("work"):
                raise RuntimeError("boom")
    assert _profiling.get_profiler() is None
    # the span still closed and was recorded
    assert [s["label"] for s in prof.spans()] == ["work"]


@pytest.mark.unit
def test_profiler_max_depth_limits_recorded_spans():
    """max_depth caps nesting; suppressed compiles roll up, totals are preserved."""
    xs = jnp.asarray(np.random.default_rng(4).normal(size=(12, 8)))

    def nested(body):
        with span("a"):
            with span("b"):
                with span("c"):
                    _scan_with(body, xs).block_until_ready()

    full = Profiler(verbose=0)
    with full:
        nested(_make_body())

    capped = Profiler(verbose=0, max_depth=2)
    with capped:
        nested(_make_body())

    assert sorted(s["label"] for s in full.spans()) == ["a", "b", "c"]
    assert sorted(s["label"] for s in capped.spans()) == ["a", "b"]

    # the compile that "c" would have owned is charged to "b", the deepest
    # recorded ancestor, so the top-level inclusive total is unchanged
    by_label = {s["label"]: s for s in capped.spans()}
    assert by_label["b"]["self_ncompiles"] > 0
    full_top = {s["label"]: s for s in full.spans()}["a"]
    assert by_label["a"]["incl_ncompiles"] == full_top["incl_ncompiles"]


@pytest.mark.unit
def test_profiler_max_depth_none_records_everything():
    """The default records every level."""
    prof = Profiler(verbose=0)
    with prof:
        with span("a"):
            with span("b"):
                with span("c"):
                    pass
    assert len(prof.spans()) == 3
