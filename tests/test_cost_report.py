"""Tests for cost/latency accounting, reports, and the registry + gates."""
import pytest

from ageval import cost as cost_mod
from ageval.gates import Thresholds, evaluate_gates
from ageval.models import (
    CostSummary,
    RunAggregate,
    Step,
    Trace,
)
from ageval.registry import EvalRegistry
from ageval.report import render_json, render_markdown
from ageval.runner import EvalRunner, get_agent
from ageval.suites import get_suite


def _trace(in_tokens=1000, out_tokens=500, model="gen-small", steps=2):
    return Trace(
        trace_id="tr-1", task_id="t-1", agent_id="a", agent_version="1",
        steps=[Step(kind="thought", text="x", input_tokens=in_tokens,
                    output_tokens=out_tokens, duration_ms=100.0)]
        + [Step(kind="final", text="done")] * (steps - 1),
        model=model,
    )


def test_cost_uses_model_pricing():
    summary = cost_mod.cost_for_trace(_trace())
    # gen-small: 1000 * 0.50 + 500 * 1.50 per 1M tokens = 0.00125
    assert summary.cost_usd == pytest.approx(0.00125)
    assert summary.total_tokens == 1500
    assert summary.latency_ms == 100.0


def test_cost_unknown_model_falls_back_to_default_price():
    summary = cost_mod.cost_for_trace(_trace(model="mystery-model"))
    assert summary.cost_usd == pytest.approx(0.00125)


def test_latency_percentiles():
    assert cost_mod.latency_percentiles([]) == (0.0, 0.0)
    p50, p95 = cost_mod.latency_percentiles([100.0, 200.0, 300.0])
    assert p50 == 200.0
    assert p95 == 300.0


def test_aggregate_cost_sums():
    total = cost_mod.aggregate_cost([
        CostSummary(100, 50, 150, 0.001, 100.0, 1, 2),
        CostSummary(200, 100, 300, 0.002, 200.0, 2, 3),
    ])
    assert total.input_tokens == 300
    assert total.cost_usd == pytest.approx(0.003)
    assert total.tool_calls == 3


def _aggregate(**overrides):
    base = {
        "task_success": 1.0, "tool_accuracy": 1.0,
        "trajectory_efficiency": 1.0, "groundedness": 1.0, "refusal": 1.0,
        "cost_usd": 0.001, "latency_ms_p50": 500.0, "latency_ms_p95": 800.0,
        "tasks_run": 4, "tasks_passed": 4,
    }
    base.update(overrides)
    return RunAggregate(**base)


def test_gates_fail_closed_without_baseline():
    report = evaluate_gates("r1", _aggregate(), baseline=None)
    assert report.verdict == "FAIL"
    assert all("no baseline" in r.reason for r in report.results)


def test_gates_pass_with_matching_baseline():
    report = evaluate_gates("r1", _aggregate(), baseline=_aggregate())
    assert report.verdict == "PASS"
    assert all(r.passed for r in report.results)


def test_gates_fail_on_regression_beyond_delta():
    report = evaluate_gates("r1", _aggregate(task_success=0.90),
                            baseline=_aggregate(task_success=1.0))
    assert report.verdict == "FAIL"
    failed = [r for r in report.results if not r.passed]
    assert any(r.metric == "task_success" for r in failed)


def test_gates_fail_on_cost_cap_breach():
    report = evaluate_gates("r1", _aggregate(cost_usd=1.0),
                            baseline=_aggregate())
    assert report.verdict == "FAIL"
    assert any(r.metric == "cost_usd" and not r.passed for r in report.results)


def test_gates_small_regression_within_delta_passes():
    report = evaluate_gates("r1", _aggregate(task_success=0.97),
                            baseline=_aggregate(task_success=1.0))
    assert report.verdict == "PASS"


def test_registry_save_list_get_and_duplicate(tmp_path):
    registry = EvalRegistry(tmp_path)
    runner = EvalRunner(registry)
    run = runner.run_suite(get_suite("travel-booking"), get_agent("scripted-good"))
    # First run: no baseline -> gates FAIL, but it is still recorded.
    assert run.gates.verdict == "FAIL"
    fetched = registry.get_run(run.run_id)
    assert fetched.suite_id == "travel-booking"
    assert len(registry.list_runs()) == 1
    with pytest.raises(ValueError, match="already exists"):
        registry.save_run(run)
    with pytest.raises(KeyError):
        registry.get_run("no-such-run")


def test_registry_baseline_requires_explicit_promotion(tmp_path):
    registry = EvalRegistry(tmp_path)
    runner = EvalRunner(registry)
    # Run 1: FAIL (no baseline) and NOT a baseline candidate until promoted.
    run1 = runner.run_suite(get_suite("support-triage"), get_agent("scripted-good"))
    assert registry.get_baseline("support-triage", "1.0.0",
                                 "scripted-good") is None
    # Promote it explicitly; the next run gates against it and passes.
    registry.promote(run1.run_id)
    baseline = registry.get_baseline("support-triage", "1.0.0", "scripted-good")
    assert baseline is not None
    assert baseline.task_success == pytest.approx(1.0)
    run2 = runner.run_suite(get_suite("support-triage"), get_agent("scripted-good"))
    assert run2.gates.verdict == "PASS"
    with pytest.raises(KeyError):
        registry.promote("no-such-run")


def test_report_markdown_and_json(tmp_path):
    registry = EvalRegistry(tmp_path)
    runner = EvalRunner(registry)
    run1 = runner.run_suite(get_suite("support-triage"), get_agent("scripted-good"))
    registry.promote(run1.run_id)
    run = runner.run_suite(get_suite("support-triage"), get_agent("scripted-good"))
    md = render_markdown(run)
    assert "Gate Verdict: PASS" in md
    assert "support-001" in md
    assert "Aggregate Scores" in md
    data = json_loads(render_json(run))
    assert data["run_id"] == run.run_id
    assert data["aggregate"]["task_success"] == 1.0


def json_loads(text):
    import json

    return json.loads(text)
