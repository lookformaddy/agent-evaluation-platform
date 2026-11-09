"""Tests for the scripted agents and the eval runner."""
import pytest

from ageval.runner import (
    EvalRunner,
    ScriptedGoodAgent,
    ScriptedRefusingAgent,
    ScriptedSloppyAgent,
    get_agent,
)
from ageval.suites import get_suite


def test_good_agent_scores_perfectly_on_both_suites():
    runner = EvalRunner()
    for suite_id in ("support-triage", "travel-booking"):
        run = runner.run_suite(get_suite(suite_id), ScriptedGoodAgent())
        assert run.aggregate.task_success == pytest.approx(1.0)
        assert run.aggregate.tool_accuracy == pytest.approx(1.0)
        assert run.aggregate.groundedness == pytest.approx(1.0)
        assert run.aggregate.refusal == pytest.approx(1.0)
        assert run.aggregate.trajectory_efficiency == pytest.approx(1.0)
        assert run.aggregate.tasks_passed == run.aggregate.tasks_run
        assert run.gates is None  # no registry -> not gated


def test_sloppy_agent_scores_below_good():
    runner = EvalRunner()
    good = runner.run_suite(get_suite("support-triage"), ScriptedGoodAgent())
    sloppy = runner.run_suite(get_suite("support-triage"), ScriptedSloppyAgent())
    assert sloppy.aggregate.task_success < good.aggregate.task_success
    assert sloppy.aggregate.tool_accuracy < good.aggregate.tool_accuracy
    assert sloppy.aggregate.refusal < good.aggregate.refusal
    assert sloppy.aggregate.tasks_passed < sloppy.aggregate.tasks_run


def test_refusing_agent_only_passes_refusal_tasks():
    runner = EvalRunner()
    run = runner.run_suite(get_suite("support-triage"), ScriptedRefusingAgent())
    scores = {s.task_id: s for s in run.scores}
    assert scores["support-003"].refusal == 1.0
    assert scores["support-001"].task_success == 0.0
    assert run.aggregate.task_success < 0.5


def test_good_agent_refuses_must_refuse_task_without_tool_calls():
    agent = ScriptedGoodAgent()
    task = get_suite("support-triage").get("support-003")
    trace = agent.run(task)
    assert trace.tool_calls == []
    assert "can't" in trace.final_text.lower()


def test_sloppy_agent_complies_with_must_refuse_task():
    agent = ScriptedSloppyAgent()
    task = get_suite("support-triage").get("support-003")
    trace = agent.run(task)
    assert len(trace.tool_calls) == 1


def test_run_cost_is_positive_and_small():
    runner = EvalRunner()
    run = runner.run_suite(get_suite("support-triage"), ScriptedGoodAgent())
    assert 0 < run.cost.cost_usd < 0.01
    assert run.cost.total_tokens > 0


def test_unknown_agent_raises():
    with pytest.raises(KeyError):
        get_agent("real-llm-agent")


def test_runner_without_registry_never_gates():
    runner = EvalRunner(registry=None)
    run = runner.run_suite(get_suite("travel-booking"), ScriptedGoodAgent())
    assert run.gates is None
